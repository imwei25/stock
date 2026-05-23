# F2 PR-B2 — LightGBM Weighter + WeighterConfig 子段化

## 1. 背景与范围

属于 `docs/strategy_improvement_2026.md` F2 模型升级,完结篇:

- PR-A(已合):embargo + label_type + Lasso 子段化
- PR-B1(已合):LightGBMSelector + 默认 `selector.type="lightgbm"`
- **PR-B2(本 spec)**:LightGBMWeighter + 默认 `weighter.type="lightgbm"` + `WeighterConfig` 子段化 + `contributions()` 下沉到 weighter ABC

PR-B2 一次完成 4 件相关的事:
1. 新增 `LightGBMWeighter` 类(完全非线性预测路径)
2. `WeighterConfig` 从扁平字段重构为 `ic`/`ir`/`equal`/`lightgbm` 子段(对称于 PR-A 的 `SelectorConfig`)
3. 默认 `weighter.type` 切到 `"lightgbm"`
4. `contributions(X)` 从 `TwoStepPipeline` 下沉到 `FactorWeighter` ABC 抽象方法,各 weighter 自实现

## 2. 目标

1. 把"先进的、完全非线性"两步法作为默认:LGB selector + LGB weighter
2. 保留所有现有 weighter 实现(IC/IR/Equal)作为可选 type,YAML 一行切回
3. 通过 `contributions()` 多态把 LGB 的 SHAP 语义嵌进现有诊断接口,无需在 Pipeline 内做 isinstance 分支
4. 完成 `WeighterConfig` 的对称子段化,跟 `SelectorConfig` 结构一致,IR-only 字段不再漂在顶层
5. lightgbm 依赖已在 PR-B1 装好,PR-B2 不再引入新依赖

## 3. 设计

### 3.1 配置变更(`src/stockpool/config.py`)

把现有扁平 `WeighterConfig` 重构为 4 个子段 + 顶层 `type` 字段:

```python
class ICWeighterConfig(BaseModel):
    """IC weighter 超参(原 WeighterConfig 顶层扁平字段中的 IC 相关项)。"""
    model_config = ConfigDict(extra="forbid")
    use_rank: bool = True            # True = Spearman rank IC, False = Pearson
    min_abs_ic: float = Field(default=0.0, ge=0.0)


class IRWeighterConfig(BaseModel):
    """IR weighter 超参(扁平字段中的 IR 专属项 + 共享的 use_rank)。"""
    model_config = ConfigDict(extra="forbid")
    n_chunks: int = Field(default=6, gt=0)
    use_rank: bool = True
    min_abs_ir: float = Field(default=0.0, ge=0.0)


class EqualWeighterConfig(BaseModel):
    """Equal weighter 无超参,占位以保持对称结构。"""
    model_config = ConfigDict(extra="forbid")


class LightGBMWeighterConfig(BaseModel):
    """LightGBM weighter 超参。默认值与 LightGBMSelectorConfig 一致 —
    walk-forward 保守。weighter 关心 prediction quality 而非 ranking,
    需要更多迭代时用户调 num_iterations 即可。"""
    model_config = ConfigDict(extra="forbid")
    num_leaves: int = Field(default=15, gt=1)
    min_data_in_leaf: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_iterations: int = Field(default=200, gt=0)
    max_depth: int = Field(default=4, gt=0)
    random_state: int = Field(default=42, ge=0)
    verbose: int = Field(default=-1)


class WeighterConfig(BaseModel):
    """Step-2 (factor weighting) settings.

    PR-B2 把原扁平字段拆成 ic/ir/equal/lightgbm 子段(对称 SelectorConfig),
    并默认 ``type='lightgbm'``。
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["ic", "ir", "equal", "lightgbm"] = "lightgbm"
    ic: ICWeighterConfig = Field(default_factory=ICWeighterConfig)
    ir: IRWeighterConfig = Field(default_factory=IRWeighterConfig)
    equal: EqualWeighterConfig = Field(default_factory=EqualWeighterConfig)
    lightgbm: LightGBMWeighterConfig = Field(default_factory=LightGBMWeighterConfig)
```

**硬切**:`extra="forbid"` 拒绝顶层扁平字段(`weighter: {type: ic, use_rank: true}` 直接报错)。用户必须把 `use_rank` 等挪到 `ic:` 子段下。

### 3.2 `FactorWeighter` ABC 增加 `contributions()` 抽象方法

`src/stockpool/ml/weighters.py` 当前 ABC:

```python
class FactorWeighter(ABC):
    @abstractmethod
    def fit(self, X, y) -> None: ...
    @abstractmethod
    def weights(self) -> pd.Series: ...
    @abstractmethod
    def predict(self, X) -> pd.Series: ...
```

加 `contributions()`:

```python
class FactorWeighter(ABC):
    @abstractmethod
    def fit(self, X, y) -> None: ...
    @abstractmethod
    def weights(self) -> pd.Series: ...
    @abstractmethod
    def predict(self, X) -> pd.Series: ...

    @abstractmethod
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-bar per-factor contribution to ``predict(X)``.

        Linear weighters return ``standardised(X) * weights``.
        Non-linear weighters return their model-specific decomposition
        (e.g. LightGBM uses SHAP). Row sums equal ``predict(X)`` by
        construction for linear weighters; for LightGBM, row sums equal
        ``predict(X) - mean_base_value`` (the SHAP convention).
        """
```

3 个线性 weighter(IC/IR/Equal)各实现 `contributions(X)`,搬迁现有 `TwoStepPipeline.contributions()` 的内联逻辑:

```python
class _LinearWeighterContributionsMixin:
    """Shared contributions() for linear-combination weighters
    (IC/IR/Equal)。Linear weighters compute ``z(X) @ weights`` for
    prediction; contributions are the per-factor components ``z(X) * weights``。"""
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._weights is None or self._weights.empty:
            return pd.DataFrame(index=X.index)
        Xs = self._apply_standardiser(X)
        w = self._weights.to_numpy()
        return pd.DataFrame(Xs * w, index=X.index, columns=self._feature_names)
```

`ICWeighter` / `IRWeighter` / `EqualWeighter` 类增加 `_LinearWeighterContributionsMixin` 作为 mixin。它们的 `_apply_standardiser` 和 `_weights` 已经存在。

`TwoStepPipeline.contributions(X)` 退化为:

```python
def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
    if self.fit_info_ is None:
        raise RuntimeError("Pipeline not fitted yet")
    selected = self.fit_info_.selected_factors
    if not selected:
        return pd.DataFrame(index=X.index)
    missing = [c for c in selected if c not in X.columns]
    if missing:
        raise KeyError(f"contributions() missing columns: {missing}")
    return self.weighter.contributions(X[selected])
```

`TwoStepPipeline.contributions()` 现在只做"投选 + delegate"两件事,所有 weighter-specific 数学下沉到 weighter 自己。

### 3.3 `LightGBMWeighter` 类(`src/stockpool/ml/weighters.py`)

```python
class LightGBMWeighter(FactorWeighter):
    """Tree-based weighter using LightGBM.

    ``fit(X, y)`` 训练 LGB,然后在训练集上一次性算 mean|SHAP|
    作为 ``weights_`` 缓存(per Q1+Q5 design decisions)。
    ``predict(X)`` 直接调 ``booster.predict(X.values)``。
    ``contributions(X)`` 走 ``booster.predict(X.values, pred_contrib=True)``
    返回每行每因子的 SHAP 值(末列 base value 丢弃)。

    Unlike linear weighters, this class does NOT inherit
    `_StandardisingMixin` — LightGBM is scale-invariant by construction.
    Look-ahead safety: ABC contract, plus the standard "predict only
    sees future data through `X`, never `y`".
    """

    def __init__(
        self,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        learning_rate: float = 0.05,
        num_iterations: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
        verbose: int = -1,
    ):
        # validation: see Plan; mirrors LightGBMSelector init checks
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose

        self._booster = None
        self._feature_names: list[str] | None = None
        self._weights: pd.Series | None = None   # cached mean|SHAP|

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb  # lazy import

        if X.empty or len(y) == 0:
            self._feature_names = list(X.columns)
            self._weights = pd.Series(dtype=float)
            self._booster = None
            return

        self._feature_names = list(X.columns)
        dataset = lgb.Dataset(
            X.values, label=y.values, feature_name=self._feature_names,
        )
        params = {
            "objective": "regression", "metric": "rmse",
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "seed": self.random_state,
            "verbose": self.verbose,
        }
        self._booster = lgb.train(
            params, dataset, num_boost_round=self.num_iterations,
        )

        # Cache mean|SHAP| as weights (Q1+Q5):
        # pred_contrib returns (n, n_features + 1) — last col is base value.
        contribs = self._booster.predict(X.values, pred_contrib=True)
        feature_contribs = contribs[:, :-1]  # drop base col
        mean_abs = np.abs(feature_contribs).mean(axis=0)
        self._weights = pd.Series(mean_abs, index=self._feature_names, name="lgb_mean_abs_shap")

    def weights(self) -> pd.Series:
        if self._weights is None:
            raise RuntimeError("Weighter not fitted yet")
        return self._weights.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._booster is None:
            return pd.Series(0.0, index=X.index)
        # Reorder columns to fit-time order; missing cols → error.
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"predict() missing columns: {missing}")
        Xn = X[self._feature_names].values
        preds = self._booster.predict(Xn)
        return pd.Series(preds, index=X.index, name="score")

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._booster is None:
            return pd.DataFrame(index=X.index)
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"contributions() missing columns: {missing}")
        Xn = X[self._feature_names].values
        contribs = self._booster.predict(Xn, pred_contrib=True)
        # Drop base value col (last); return per-feature SHAP.
        return pd.DataFrame(
            contribs[:, :-1], index=X.index, columns=self._feature_names,
        )
```

### 3.4 `_build_weighter` 工厂(`src/stockpool/backtesting/strategies.py`)

当前实现读扁平字段:

```python
def _build_weighter(cfg) -> FactorWeighter:
    if cfg.type == "ic":
        return ICWeighter(use_rank=cfg.use_rank, min_abs_ic=cfg.min_abs_ic)
    if cfg.type == "ir":
        return IRWeighter(n_chunks=cfg.n_chunks, use_rank=cfg.use_rank, min_abs_ir=cfg.min_abs_ir)
    if cfg.type == "equal":
        return EqualWeighter()
    raise ValueError(...)
```

PR-B2 重写为读子段:

```python
def _build_weighter(cfg) -> FactorWeighter:
    """Translate WeighterConfig → concrete FactorWeighter (PR-B2 subnested)."""
    if cfg.type == "ic":
        return ICWeighter(use_rank=cfg.ic.use_rank, min_abs_ic=cfg.ic.min_abs_ic)
    if cfg.type == "ir":
        return IRWeighter(
            n_chunks=cfg.ir.n_chunks,
            use_rank=cfg.ir.use_rank,
            min_abs_ir=cfg.ir.min_abs_ir,
        )
    if cfg.type == "equal":
        return EqualWeighter()
    if cfg.type == "lightgbm":
        c = cfg.lightgbm
        return LightGBMWeighter(
            num_leaves=c.num_leaves,
            min_data_in_leaf=c.min_data_in_leaf,
            learning_rate=c.learning_rate,
            num_iterations=c.num_iterations,
            max_depth=c.max_depth,
            random_state=c.random_state,
            verbose=c.verbose,
        )
    raise ValueError(f"unknown weighter type: {cfg.type!r}")
```

Import 顶部加 `LightGBMWeighter`。

### 3.5 旧测试 fixture 处理

Q2 选了 `weighter.type` 默认切到 `"lightgbm"`,意味着所有用 `MLFactorConfig()` 默认值的 fit-exercising 测试,需要显式注入 `weighter=WeighterConfig(type="ic")` 保留旧 IC 数值。同时由于 PR-B1 已经把 selector 切到 lightgbm,既有 fixture **已经有 `selector=SelectorConfig(type="lasso")`**(PR-B1 Task 5 添加的)。PR-B2 在这些 fixture 上再补一个 `weighter=WeighterConfig(type="ic")`。

预计触点:
- `tests/test_ml_strategy.py`:10 处(已被 PR-B1 patched)
- `tests/test_ml_strategy_panel.py`:3 处
- `tests/test_ml_strategy_embargo.py`:3 处(走 `_try_fit` 的那 3 个)

### 3.6 新测试(`tests/test_ml_weighter_lightgbm.py`)

```python
# 1. fit+predict round-trip
def test_lightgbm_weighter_fit_predict_round_trip():
    X, y = linear_signal_xy(n=500)
    w = LightGBMWeighter(random_state=1)
    w.fit(X, y)
    preds = w.predict(X)
    assert len(preds) == len(X)
    # 训练集上预测应正相关 y
    assert preds.corr(y, method="spearman") > 0.3


# 2. weights() 返回 mean|SHAP|,长度 = n_features,sum > 0
def test_lightgbm_weighter_weights_are_mean_abs_shap():
    X, y = linear_signal_xy(n=500)
    w = LightGBMWeighter(random_state=2)
    w.fit(X, y)
    ws = w.weights()
    assert len(ws) == len(X.columns)
    assert (ws >= 0).all()  # |SHAP| 非负
    assert ws.sum() > 0


# 3. contributions(X) 形状 + row sum 等于 predict(X) - base value
def test_lightgbm_weighter_contributions_sum_to_predict_minus_base():
    X, y = linear_signal_xy(n=300)
    w = LightGBMWeighter(random_state=3)
    w.fit(X, y)
    preds = w.predict(X)
    contribs = w.contributions(X)
    # row sum + base_value ≈ predict
    # 通过 contribs.sum(axis=1) 与 (preds - mean_base) 比较
    row_sums = contribs.sum(axis=1)
    # 它们应正相关接近 1:1(精确值受 LGB 内部 base 影响)
    corr = float(row_sums.corr(preds))
    assert corr > 0.95


# 4. 确定性
def test_lightgbm_weighter_deterministic_with_seed():
    X, y = linear_signal_xy(n=400)
    w1 = LightGBMWeighter(random_state=42)
    w2 = LightGBMWeighter(random_state=42)
    w1.fit(X, y); w2.fit(X, y)
    np.testing.assert_array_almost_equal(
        w1.predict(X).values, w2.predict(X).values,
    )


# 5. 空输入
def test_lightgbm_weighter_empty_input():
    X = pd.DataFrame({"a": [], "b": []}, dtype=float)
    y = pd.Series([], dtype=float)
    w = LightGBMWeighter()
    w.fit(X, y)
    assert w.weights().empty
    preds = w.predict(X)
    assert len(preds) == 0


# 6. predict 时缺列 → KeyError
def test_lightgbm_weighter_predict_missing_columns_raises():
    X, y = linear_signal_xy(n=200)
    w = LightGBMWeighter(random_state=6)
    w.fit(X, y)
    X_missing = X.drop(columns=[X.columns[0]])
    with pytest.raises(KeyError):
        w.predict(X_missing)


# 7. weights() 在 fit 之前调 → RuntimeError
def test_lightgbm_weighter_weights_before_fit_raises():
    w = LightGBMWeighter()
    with pytest.raises(RuntimeError):
        w.weights()


# 8. TwoStepPipeline 集成:LGB selector + LGB weighter
def test_two_step_pipeline_lgb_selector_lgb_weighter():
    X, y = linear_signal_xy(n=500, n_signal=3, n_noise=2)
    pipeline = TwoStepPipeline(
        selector=LightGBMSelector(top_k_factors=3),
        weighter=LightGBMWeighter(),
    )
    info = pipeline.fit(X, y)
    if info.selected_factors:
        preds = pipeline.predict(X)
        assert len(preds) == len(X)
        contribs = pipeline.contributions(X)
        assert contribs.shape == (len(X), len(info.selected_factors))
```

### 3.7 配置测试(补 `tests/test_config.py`)

- `test_weighter_default_type_is_lightgbm`
- `test_weighter_ic_subcfg_explicit` — `weighter: {type: ic, ic: {use_rank: false, min_abs_ic: 0.05}}` 解析
- `test_weighter_ic_subcfg_defaults`
- `test_weighter_ir_subcfg_explicit + defaults`(确保 `n_chunks`、`use_rank`、`min_abs_ir` 都到 `ir` 子段下)
- `test_weighter_equal_subcfg_defaults`
- `test_weighter_lightgbm_subcfg_explicit + defaults`(7 个 LGB 字段)
- `test_weighter_flat_use_rank_rejected` — 旧扁平 `weighter: {type: ic, use_rank: true}` 触发 ValidationError
- `test_weighter_unknown_type_rejected` — `type="catboost"` raise

### 3.8 用户 YAML 迁移

当前 `config.yaml` 的 `weighter` 块(PR-A 之后状态):

```yaml
weighter:
  type: ic                # ic | ir | equal
  use_rank: true          # rank IC (Spearman) 更稳健
  min_abs_ic: 0.0
  n_chunks: 6             # 仅 type=ir 时用
  min_abs_ir: 0.0
```

PR-B2 合入后必须改为:

```yaml
weighter:
  type: ic            # 显式保留 ic;若想切 LGB 删掉这行或改 lightgbm
  ic:
    use_rank: true
    min_abs_ic: 0.0
  ir:                 # 可选 — 不写就用默认值
    n_chunks: 6
    use_rank: true
    min_abs_ir: 0.0
  # equal: {}         # 默认空块也可写
  # lightgbm:         # 用 LGB 时可调超参,否则用 LightGBMWeighterConfig 默认
  #   num_leaves: 15
```

提交 commit message + README 写明这条迁移。

### 3.9 现有 `TwoStepPipeline.contributions()` 测试

`tests/test_ml_pipeline.py` 中有现存的 contributions 测试。下沉后接口不变(`pipeline.contributions(X)` 仍返回 DataFrame),数值也不变(linear weighter 的 `contributions(X) = z(X) * weights`,搬到 weighter 内部数学等价)。这些测试应该自动通过。

如果有测试直接 mock `_apply_standardiser` 或 `_weights` 之类的 private,需要看是否要调整 — 应该没有,但确认一遍。

### 3.10 缓存失效

`_strategy_signature` hash MLFactorConfig.model_dump()。`WeighterConfig` 结构变 + default 切 → sig 变 → `ml_models/*.pkl` 旧缓存全部失效。符合预期。

### 3.11 文档同步

- **`CLAUDE.md`**:
  - 模块地图 `ml/` 子包描述更新为 "Lasso 或 LightGBM selector / IC&IR&Equal&LightGBM weighter / TwoStepPipeline"
  - 配置段 `weighter` 描述更新为子段化形式 + LGB 默认
- **`README.md`**:扩展 PR-B1 的 "关于 LightGBM 默认 selector" 段,加 weighter:

  > F2 PR-B2 起,`weighter.type` 默认也是 `"lightgbm"`,完成完全非线性两步法。weighter 与 selector 各训练一次 LGB,refit 训练时间约为 PR-B1 的 1.8-2.2 倍。
  >
  > **A/B 对照**:回到 PR-B1 的"LGB selector + IC 加权"对照 baseline,设 `weighter.type: ic`(YAML 里加一行,默认 `ic.use_rank: true` 即可)。

### 3.12 LGB selector 与 weighter 训练独立

当 selector.type=lightgbm 且 weighter.type=lightgbm 时,**两个 LGB 独立训练**:
- Selector 在 full X 上训练,按 gain importance 排序选 top-K
- Weighter 在仅含 selected_factors 列的 X 子集上训练
- 训练成本 ~2x 单 LGB,但避免选了又用、模型不一致

如果未来发现这是瓶颈,可在 F2 后续 PR 加 `share_lgb_model` 配置选项;不在 PR-B2 范围内。

## 4. 范围外 / 非目标

- holdout + early stopping、cross-validation — 留作 follow-up
- SHAP 替代库(shap、treeshap)— LGB 自带 `pred_contrib=True` 已够
- LGB selector 与 weighter 模型复用 — 见 §3.12,留作未来优化
- F1 plan-2(custom factors + WQ101 全量 ranking + A/B 验证) — 独立轨道
- F3(组合构建 / 风险 overlay) — 独立轨道

## 5. 验收标准

1. **零回归**:所有现有测试在 fixture 注入 `selector=lasso, weighter=ic` 后通过
2. **LGB weighter fit-predict 通**:`test_lightgbm_weighter_fit_predict_round_trip` 训练集 IC > 0.3
3. **SHAP 接近 predict**:`test_lightgbm_weighter_contributions_sum_to_predict_minus_base` row sums 与 `predict(X)` 相关 > 0.95
4. **WeighterConfig 硬切**:旧扁平 `weighter: {type: ic, use_rank: true}` 立即报 ValidationError
5. **可切换**:`weighter.type: ic` 一行 YAML 回到 PR-B1 的"LGB select + IC 加权"baseline
6. **`contributions()` 多态生效**:`pipeline.contributions(X)` 在 LGB weighter 下返回 SHAP shape,在 IC weighter 下返回标准化 X × weights,无 isinstance 分支
7. **缓存失效自然**:首次跑 daily report 后 `ml_models/*.pkl` 旧 sig 文件不再被使用
8. **README 与 CLAUDE.md 同步**

## 6. 后续

PR-B2 合入后 F2 完结。下一步:
- F3 — 组合构建(vol-targeted sizing + max DD 熔断 + sector cap)
- F1 plan-2 — 自定义因子 + WQ101 全量 ranking + A/B 对照(此时有完整的"老 vs 新"两条端到端路径)
- 实跑验证 — 用户在自己 terminal 跑 `python -m stockpool backtest`,对比 (LGB+LGB) vs (LGB+IC) vs (Lasso+IC) baseline
