# F2 PR-B1 — LightGBM Selector

## 1. 背景与范围

属于 `docs/strategy_improvement_2026.md` F2 模型升级,PR-A 之后的第二步。F2 整体已被拆为:

- PR-A(已合):embargo + label_type + Lasso 子段化(`docs/superpowers/specs/2026-05-23-f2-pr-a-embargo-and-config-subnesting-design.md`)
- **PR-B1(本 spec)**:LightGBM selector,与现有 IC weighter 组合 — 实现"非线性选 + 线性加"两步法
- PR-B2(后续):LightGBM weighter,实装完全非线性预测;PR-B1 验证后再启动

PR-B1 是 PR-A `SelectorConfig` 子段化的兑现 — 把 PR-A 准备好的 `selector.lasso` 命名空间扩展为 `selector.{lasso|lightgbm}` 同级二选一。

## 2. 目标

1. 引入 `LightGBMSelector` 替代 / 并存 `LassoSelector`,作为 walk-forward 训练的可选 selector
2. 默认切到 `selector.type="lightgbm"` — 把"非线性选因子"作为新默认
3. 把 lightgbm 加入 required dependencies
4. 全部既有测试通过 `selector=SelectorConfig(type="lasso")` 显式回退保持数值稳定
5. README 写明 LGB 在小训练集上的过拟合风险及回退路径

## 3. 设计

### 3.1 依赖

`pyproject.toml` 在 `dependencies` 数组追加 `"lightgbm>=4.0"`。理由:LightGBM 4.x 是当前稳定大版本,API 与 3.x 兼容性较好;`pip install -e .` 会自动装 native wheel(~6 MB on Windows)。不走 `optional-dependencies`,因为 default 切到 lightgbm 后,可选依赖路径会让 fresh install 立即 broken。

### 3.2 配置变更(`src/stockpool/config.py`)

新增 `LightGBMSelectorConfig`,扩展 `SelectorConfig`:

```python
class LightGBMSelectorConfig(BaseModel):
    """LightGBM-based selector hyperparameters.

    All defaults are conservative for the walk-forward setting (small
    per-refit training set, refit_every controls retraining frequency).
    Tighten ``num_leaves`` / loosen ``min_data_in_leaf`` if IC across refits
    looks unstable.
    """
    model_config = ConfigDict(extra="forbid")
    num_leaves: int = Field(default=15, gt=1)
    min_data_in_leaf: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_iterations: int = Field(default=200, gt=0)
    max_depth: int = Field(default=4, gt=0)
    random_state: int = Field(default=42, ge=0)
    # Selection: take top-K factors by normalized gain importance,
    # but only those whose gain >= max(gain) * min_importance_ratio.
    top_k_factors: int = Field(default=20, gt=0)
    min_importance_ratio: float = Field(default=0.01, ge=0, le=1)
    verbose: int = Field(default=-1)   # LightGBM verbosity; -1 = silent


class SelectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["lasso", "lightgbm"] = "lightgbm"   # ← default changes
    lasso: LassoConfig = Field(default_factory=LassoConfig)
    lightgbm: LightGBMSelectorConfig = Field(default_factory=LightGBMSelectorConfig)
```

`LassoConfig` 和现有 `lasso` 字段保持不变 — 用户切回 `type: lasso` 行为完全等价于 PR-A。

### 3.3 `LightGBMSelector` 类(`src/stockpool/ml/selectors.py`)

```python
class LightGBMSelector(FactorSelector):
    """Tree-based selector using LightGBM gain importance.

    fit() trains a LightGBM regressor on (X, y); selected_factors() returns
    the columns whose normalized gain importance is in the top-K AND >=
    ``max_importance * min_importance_ratio``.

    Look-ahead safety: this class doesn't introduce any look-ahead — it
    just consumes (X, y) the way LassoSelector does. The walk-forward
    train/test split happens upstream in MLFactorStrategy._try_fit, which
    PR-A's embargo already protects.

    ``coef_`` is populated with the normalized importance series (sum = 1)
    so downstream FitInfo introspection keeps a consistent shape.
    """
    def __init__(
        self,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        learning_rate: float = 0.05,
        num_iterations: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
        top_k_factors: int = 20,
        min_importance_ratio: float = 0.01,
        verbose: int = -1,
    ):
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.random_state = random_state
        self.top_k_factors = top_k_factors
        self.min_importance_ratio = min_importance_ratio
        self.verbose = verbose

        self.coef_: pd.Series | None = None
        self.selected_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb  # lazy import — ImportError surfaces at fit time

        if X.empty or len(y) == 0:
            self.coef_ = pd.Series(dtype=float)
            self.selected_ = []
            return

        feature_names = list(X.columns)
        dataset = lgb.Dataset(
            X.values, label=y.values, feature_name=feature_names,
        )
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "seed": self.random_state,
            "verbose": self.verbose,
        }
        booster = lgb.train(params, dataset, num_boost_round=self.num_iterations)
        gain = booster.feature_importance(importance_type="gain").astype(float)

        total = float(gain.sum())
        if total < 1e-12:
            # Degenerate case (e.g. constant y): no signal. Empty selection.
            self.coef_ = pd.Series(0.0, index=feature_names, name="lgb_importance")
            self.selected_ = []
            return

        norm = gain / total
        self.coef_ = pd.Series(norm, index=feature_names, name="lgb_importance")

        max_val = float(self.coef_.max())
        threshold = max_val * self.min_importance_ratio
        ranked = self.coef_.sort_values(ascending=False)
        eligible = ranked[ranked >= threshold].head(self.top_k_factors)
        self.selected_ = list(eligible.index)

    def selected_factors(self) -> list[str]:
        return list(self.selected_)
```

设计要点:
- **Lazy import**:`import lightgbm` 在 `fit` 内部,不在文件顶层。这样选 `selector.type: lasso` 时即使 lightgbm 没装也能正常跑(防御性 — 虽然 PR-B1 是必装,但抽象上仍然解耦)。
- **`coef_` 复用**:LGB 没有线性 coef,但 `TwoStepPipeline.FitInfo.coef` 是已有 pd.Series 字段。我们填归一化后的 gain 进去,语义注释中说清楚是 importance 不是 coefficient。
- **空输入 / 退化 y**:`X.empty` 或 `gain.sum() == 0` 时返回空 selection,与 `LassoSelector` 的 fallback 行为一致(`TwoStepPipeline.fit` 已有 `fallback_used=True` 路径处理"selector dropped everything")。

### 3.4 `_build_selector` 工厂(`src/stockpool/backtesting/strategies.py`)

PR-A 的 `_try_fit` 直接内联 `LassoSelector(...)`。PR-B1 抽出工厂:

```python
def _build_selector(cfg) -> FactorSelector:
    """Translate SelectorConfig → concrete FactorSelector."""
    if cfg.type == "lasso":
        return LassoSelector(
            alpha=cfg.lasso.alpha,
            max_iter=cfg.lasso.max_iter,
            tol=cfg.lasso.tol,
        )
    if cfg.type == "lightgbm":
        c = cfg.lightgbm
        return LightGBMSelector(
            num_leaves=c.num_leaves,
            min_data_in_leaf=c.min_data_in_leaf,
            learning_rate=c.learning_rate,
            num_iterations=c.num_iterations,
            max_depth=c.max_depth,
            random_state=c.random_state,
            top_k_factors=c.top_k_factors,
            min_importance_ratio=c.min_importance_ratio,
            verbose=c.verbose,
        )
    raise ValueError(f"unknown selector type: {cfg.type!r}")
```

`_try_fit` 里的 `selector=LassoSelector(alpha=cfg.selector.lasso.alpha, ...)` 调用替换为 `selector=_build_selector(cfg.selector)`。

### 3.5 旧测试 fixture 处理

Q1 选了 default lightgbm,意味着所有用 `MLFactorConfig()` 默认值的现有测试,会从 Lasso 路径走到 LGB 路径,数值结果会变。沿用 PR-A 的方法:**给所有现有 fixture 显式注入 `selector=SelectorConfig(type="lasso")`** 保持旧 IC/quantile 数值不变。

预计触点(grep 已确认):
- `tests/test_ml_strategy.py`:10 处 `MLFactorConfig(...)`
- `tests/test_ml_strategy_panel.py`:3 处
- `tests/test_ml_pipeline.py`:0 处

`tests/test_ml_strategy_embargo.py` 是 PR-A 新加的,其中已经默认 lightgbm 不会影响 `_embargoed_label_end` 的数值(那些测试只验 helper,不跑 fit),所以**只补需要跑 `_try_fit` 的那几个**(`test_refit_with_legacy_no_embargo_runs_to_completion` 等)。

### 3.6 新测试(`tests/test_ml_selector_lightgbm.py`)

```python
# Test 1: 非线性场景下 LGB 选出 Lasso 选不到的因子
def test_lightgbm_selector_picks_nonlinear_features():
    # y = x0 * sign(x1) + noise — Lasso 看不到 x1 的贡献,LGB 看得到
    ...
    lasso_sel = LassoSelector(alpha=0.001)
    lgb_sel = LightGBMSelector(top_k_factors=2, min_importance_ratio=0.01)
    lasso_sel.fit(X, y); lgb_sel.fit(X, y)
    assert "x0" in lgb_sel.selected_factors()
    assert "x1" in lgb_sel.selected_factors()
    # Lasso 至少选 x0(线性主效应),不一定选 x1


# Test 2: top_k 截断
def test_lightgbm_selector_top_k_truncates():
    # 5 个因子全有信号,top_k_factors=2 → 选 2 个
    ...
    sel = LightGBMSelector(top_k_factors=2, min_importance_ratio=0.0)
    sel.fit(X, y)
    assert len(sel.selected_factors()) == 2


# Test 3: min_importance_ratio 过滤
def test_lightgbm_selector_min_importance_filter():
    # 全噪声 X,y 与 X 无关 → 退化场景,选出 0 或 1 个
    ...
    sel = LightGBMSelector(top_k_factors=10, min_importance_ratio=0.99)
    sel.fit(X, y)
    assert len(sel.selected_factors()) <= 1


# Test 4: 确定性
def test_lightgbm_selector_deterministic_with_seed():
    sel1 = LightGBMSelector(random_state=42)
    sel2 = LightGBMSelector(random_state=42)
    sel1.fit(X, y); sel2.fit(X, y)
    assert sel1.selected_factors() == sel2.selected_factors()


# Test 5: coef_ 归一化
def test_lightgbm_selector_coef_normalized():
    sel = LightGBMSelector()
    sel.fit(X, y)
    # 非退化场景: sum ≈ 1; 退化场景: sum == 0
    assert abs(sel.coef_.sum() - 1.0) < 1e-6 or sel.coef_.sum() == 0


# Test 6: TwoStepPipeline 集成
def test_two_step_pipeline_with_lgb_selector_and_ic_weighter():
    pipeline = TwoStepPipeline(
        selector=LightGBMSelector(),
        weighter=ICWeighter(use_rank=True),
    )
    pipeline.fit(X, y)
    # pipeline.fit_info_.selected_factors 非空时 predict 可用
    if pipeline.fit_info_.selected_factors:
        preds = pipeline.predict(X)
        assert len(preds) == len(X)


# Test 7: 退化 y (常数) → 空选择
def test_lightgbm_selector_empty_when_y_constant():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([1.0] * 5)
    sel = LightGBMSelector()
    sel.fit(X, y)
    assert sel.selected_factors() == []


# Test 8: 空输入
def test_lightgbm_selector_empty_input():
    X = pd.DataFrame({"a": [], "b": []})
    y = pd.Series([], dtype=float)
    sel = LightGBMSelector()
    sel.fit(X, y)
    assert sel.selected_factors() == []
```

### 3.7 配置 schema 测试(补 `tests/test_config.py`)

- `test_selector_lightgbm_subcfg_explicit` — 解析 `selector.lightgbm.num_leaves` 等
- `test_selector_lightgbm_subcfg_defaults` — 不写则用 LightGBMSelectorConfig 默认值
- `test_selector_default_type_is_lightgbm` — 不设 `selector.type` 时默认 `"lightgbm"`
- `test_selector_lightgbm_flat_field_rejected` — `selector: {type: lightgbm, num_leaves: 31}` 顶层 num_leaves 因 `extra="forbid"` 报错
- `test_selector_unknown_type_rejected` — `type: "xgboost"` raise

### 3.8 缓存失效

`_strategy_signature` 当前已经哈希 `MLFactorConfig.model_dump()`。新加 `selector.lightgbm.*` + default type 变化 → sig 变 → `ml_models/*.pkl` 旧缓存全部失效,首次跑全量重训。

### 3.9 用户 YAML 行为

当前 `config.yaml` 的 `strategy.ml_factor.selector` 块已经是 PR-A 子段化后形态:

```yaml
selector:
  type: lasso       # ← user 当前设的;若要切到默认,把这行删掉
  lasso:
    alpha: 0.001
    max_iter: 1000
    tol: 1.0e-6
```

PR-B1 合入后:
- **现状(显式 type: lasso)** — 行为完全不变
- **若用户想切到默认 LGB**:把 `type: lasso` 删掉(或改成 `type: lightgbm`),可选追加 `lightgbm:` 子段调超参,否则全用默认值

PR-B1 的 commit message 与 README 更新中写明这个迁移路径(可选 — 用户既可保 lasso 也可切 lightgbm)。

### 3.10 文档同步

- **`CLAUDE.md`**:
  - 模块地图加 `ml/selectors.py` 新增 `LightGBMSelector` 说明
  - 配置段把 `selector.lasso` 补成 `selector.{lasso|lightgbm}`
- **`README.md`**(per Q4 要求):新增过拟合风险段(草拟):

  > ### 关于 LightGBM 默认 selector
  >
  > F2 PR-B1 起,`strategy.ml_factor.selector.type` 默认为 `"lightgbm"`,用 LightGBM 在 walk-forward 训练窗口上选因子。这是非线性选 + IC 线性加的两步法。
  >
  > **过拟合提示**:每次 refit 训练集只有 ~250 bars × N 股,LGB 在小样本上容易过拟合。当前默认参数(`num_leaves=15`、`min_data_in_leaf=20`、`learning_rate=0.05`、`num_iterations=200`)已为这个规模做了保守化,但仍然 *依赖* "walk-forward 每次重训,单次过拟合无伤大雅" 这个假设。
  >
  > **观测指标**:跑回测后看 `reports/backtest/latest.html` 里的 trade 分布;如果 IC 跨 refit 不稳、净值曲线锯齿明显,先调小 `num_leaves` 或调大 `min_data_in_leaf`;还不行就 `selector.type: lasso` 回到 PR-A 的线性 baseline 做对照。
  >
  > **不做** holdout + early stopping(留给 F2 PR-B2 或更后)。

## 4. 范围外 / 非目标

- LightGBMWeighter — 留给 PR-B2
- holdout + early stopping、cross-validation — 留给 PR-B2 或 follow-up
- SHAP 或 `contributions()` 对 LGB 的适配 — `TwoStepPipeline.contributions()` 当前依赖 `weighter._apply_standardiser` + `weighter.weights()`,PR-B1 weighter 仍是 IC(线性),不受影响。PR-B2 引入 LGB weighter 时再处理
- F1 plan-2(custom factors + WQ101 全量 ranking + A/B 验证)— 独立轨道
- F3(组合构建 / 风险 overlay)— 独立轨道

## 5. 验收标准

1. **零回归**:所有现有测试(`test_ml_strategy.py` / `_panel.py` / `_pipeline.py` / `_embargo.py`)在 fixture 注入 `selector=SelectorConfig(type="lasso")` 后通过
2. **LGB 非线性增益**:`test_lightgbm_selector_picks_nonlinear_features` 在 `y = x0 * sign(x1) + noise` 合成数据上证明 LGB 选出 x0 和 x1 两个因子
3. **依赖装上**:`./.venv/Scripts/python.exe -c "import lightgbm; print(lightgbm.__version__)"` 输出 4.x
4. **可切换**:`selector.type: lasso` 一行 YAML 变更即可回到 PR-A baseline,无需改代码
5. **配置硬切**:`selector.lightgbm` 顶层扁平字段(如 `selector: {type: lightgbm, num_leaves: 31}`)被 Pydantic 拒绝
6. **缓存失效自然**:首次 daily report / backtest 跑全量重训,新 sig 写到 `ml_models/`
7. **README 提醒**:过拟合段落已经合入

## 6. 后续

- PR-B2:`LightGBMWeighter` + 默认 `weighter.type="lightgbm"` + 完全非线性预测路径 + `TwoStepPipeline.contributions()` 用 `lgb.predict(pred_contrib=True)`
- 在 PR-B1 验证后,基于实际 A 股 IC 表现决定 PR-B2 是值得做还是 hold(可能 LGB selector + IC weighter 性价比就够)
- F2 PR-A + B1 都合入后,F1 plan-2 的 A/B 对照实验可以更有意义(对比"老因子 + Lasso + IC" vs "新因子 + LGB selector + IC")
