# F2 PR-A — Embargo + label_type + Lasso 子段化

## 1. 背景与范围

属于 `docs/strategy_improvement_2026.md` 中 F2 模型升级的第一个 PR。F2 整体分两步:

- **PR-A(本 spec)**:embargo 修 walk-forward 标签泄露,label_type 抽象出来给后续实验留接口,SelectorConfig 子段化为 PR-B 引入 lightgbm 让路。**所有默认行为变化只来自 embargo bug fix 一项**;label_type 默认 `return` 保持现行,Lasso 行为本身完全不变。
- **PR-B(后续)**:新增 LightGBMSelector / LightGBMWeighter,默认切换。

PR-A 不引入新依赖,不改 Strategy ABC,不动 backtest engine。

## 2. 目标

1. 修复 walk-forward 训练集尾部 label 与测试集首样本的信息重叠(`horizon=3` 时尤其明显)。
2. 把 `selector.lasso.*` 子段化,为 PR-B 的 `selector.lightgbm.*` 准备好命名空间。
3. 增加 `label_type` 字段,为后续 vol-adjusted / cross-sectional rank 标签留接口(实现可空,PR-B 或更后再补)。

## 3. 设计

### 3.1 配置变更(`src/stockpool/config.py`)

新增 `LassoConfig` 嵌套模型,`SelectorConfig` 重构:

```python
class LassoConfig(BaseModel):
    alpha: float = Field(default=0.001, ge=0.0)
    max_iter: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-6, gt=0.0)


class SelectorConfig(BaseModel):
    """Factor selector configuration.

    PR-A only supports type='lasso'; PR-B will add 'lightgbm'.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["lasso"] = "lasso"
    lasso: LassoConfig = Field(default_factory=LassoConfig)
```

**硬切**(Q3 已定):`SelectorConfig` 加 `extra="forbid"`,旧扁平字段(`selector: {type: lasso, alpha: 0.001, ...}`)立即触发 ValidationError。用户必须把 `alpha/max_iter/tol` 挪到 `selector.lasso` 下。这样升级时错误显式,不会因静默忽略导致 alpha 默认值被悄悄用上。

`MLFactorConfig` 新增两个字段:

```python
class MLFactorConfig(BaseModel):
    # ... 现有字段 ...

    # walk-forward embargo: 训练集与测试集之间额外留 N 天空档,
    # 防止 `horizon`-day forward return 与训练集尾部样本信息重叠。
    # None = 自动 = horizon(默认就修 bug);0 = 显式关闭(向后兼容旧行为)。
    embargo_days: int | None = None

    # 训练标签类型。PR-A 只实装 "return"(当前行为);
    # "vol_adjusted" / "cross_sec_rank" 留接口,PR-B 或更后实现。
    label_type: Literal["return", "vol_adjusted", "cross_sec_rank"] = "return"
```

`embargo_days` 改默认 = None(auto = horizon)是 PR-A 唯一的行为变化。

### 3.2 标签计算扩展(`src/stockpool/ml/dataset.py`)

`forward_return_panel` 增加 `label_type` 参数:

```python
def forward_return_panel(
    close: pd.DataFrame,
    horizon: int,
    label_type: Literal["return", "vol_adjusted", "cross_sec_rank"] = "return",
    vol_window: int = 20,
) -> pd.DataFrame:
    """前向收益面板,可配置标签类型。

    Args:
        close: T × N 收盘价宽表。
        horizon: 前瞻天数 h。
        label_type:
            "return"            — close[t+h]/close[t] - 1(当前行为)。
            "vol_adjusted"      — return / rolling_std(past returns, vol_window)。
            "cross_sec_rank"    — 每日横截面 rank 后线性映射到 [-0.5, +0.5]。
        vol_window: 仅 label_type='vol_adjusted' 时使用,过去 N 日 std。

    Returns:
        T × N 标签宽表,行末 h 天为 NaN(无 forward 数据)。
    """
```

实现:
- `"return"`:`close.pct_change(horizon).shift(-horizon)`(当前实现,1 行调用)
- `"vol_adjusted"`:先算 daily return,rolling_std 用过去 `vol_window` 天(避免泄露),然后 `forward_return / vol_proxy`
- `"cross_sec_rank"`:对 forward_return 每行 `rank(pct=True) - 0.5`,nan 保留 nan

PR-A 只**实装** `"return"` 路径;另外两个 raise `NotImplementedError`,留接口给后续 PR。这样 spec 的"接口已定义"承诺成立,但实现工作量受限。

### 3.3 Embargo 在 walk-forward 训练中的位置(`src/stockpool/backtesting/strategies.py`)

`MLFactorStrategy._refit` 当前用 bar index 切训练集:

```python
# 现有代码 (strategies.py:513)
label_end = current_bar - cfg.horizon
```

PR-A 改为:

```python
effective_embargo = (
    cfg.embargo_days if cfg.embargo_days is not None else cfg.horizon
)
label_end = current_bar - cfg.horizon - effective_embargo
```

`per_stock` 路径中 `train_start = max(0, label_end - cfg.train_window)`,所以 embargo 自动让训练窗口整体后退一段;`pooled` 路径中 host 走同样的 `current_bar - horizon - effective_embargo` 切片,pool 内其他股的 `_build_truncated_pool` 用 `current_date` 截断,需要同步——把 `current_date` 也减去 `effective_embargo` 个交易日(用 `daily_df["date"].iloc[label_end - 1]` 取真实交易日,避免周末偏移)。

加 Pydantic 校验:`embargo_days >= 0`(None 单独允许)。

**`LassoSelector` 实例化路径更新**:`strategies.py:549-554` 当前从 `cfg.selector.alpha` 读,改为 `cfg.selector.lasso.alpha`(以及 `max_iter`、`tol`)。这是子段化的直接后果,纯路径重命名。

新增辅助 helper(strategies.py 内部,不导出)统一切两个路径,避免分散逻辑:

```python
def _embargoed_label_end(self, current_bar: int) -> int:
    cfg = self.cfg
    eff = cfg.embargo_days if cfg.embargo_days is not None else cfg.horizon
    return current_bar - cfg.horizon - eff
```

`_refit` 和 `_build_truncated_pool` 都调它。

### 3.4 缓存失效

`_strategy_signature` 当前对 `MLFactorConfig.model_dump()` 做 sha256。新增 `embargo_days` / `label_type` / 子段化后的 `selector.lasso.*` 会改变 dump,sig 自动变化,`ml_models/*.pkl` 旧缓存全部失效。**符合预期**:首次跑会全量重训。

不需要手写迁移脚本——首次预测被 cache miss → 重训 → 写新缓存即可。

### 3.5 测试

新建 `tests/test_ml_dataset_labels.py`:

- `test_forward_return_panel_return_type` — `label_type="return"` 与当前 `pct_change(h).shift(-h)` 等价
- `test_forward_return_panel_vol_adjusted_not_implemented` — `label_type="vol_adjusted"` 当前 raise `NotImplementedError`(承诺接口存在)
- `test_forward_return_panel_cross_sec_rank_not_implemented` — 同上
- `test_forward_return_panel_unknown_label_type_raises` — 不在 Literal 内的字符串 raise

新建 `tests/test_ml_strategy_embargo.py`:

- `test_embargo_default_equals_horizon` — `embargo_days=None` 时,`_refit` 截到 `current_bar - horizon - horizon`(总 2*horizon)
- `test_embargo_explicit_zero_preserves_legacy` — `embargo_days=0` 时,行为等价于旧版(截到 `current_bar - horizon`)
- `test_embargo_explicit_positive` — `embargo_days=5` 时,截到 `current_bar - horizon - 5`
- `test_embargo_eliminates_label_leak` — 构造一个 horizon-day forward return 强自相关的合成数据集(典型 spurious case),验证不加 embargo 时 train_IC 显著高于 test_IC,加 embargo 后差距缩小

补 `tests/test_config.py`:

- `test_lasso_subcfg_explicit` — 新写法 `selector: {type: lasso, lasso: {alpha: 0.01}}` 正常解析
- `test_lasso_subcfg_default` — `selector: {type: lasso}` 用 LassoConfig 默认值
- `test_lasso_flat_fields_rejected` — 旧写法 `selector: {type: lasso, alpha: 0.01}` 触发 ValidationError(因 `extra="forbid"`)
- `test_embargo_days_default_is_none` — 不写 `embargo_days` 时 = `None`
- `test_embargo_days_zero_valid` — 显式 `0` 接受
- `test_embargo_days_negative_rejected` — `-1` raise
- `test_label_type_default_is_return` — 不写 = `"return"`
- `test_label_type_unknown_rejected` — 写其他字符串 raise

补现有 `tests/test_ml_pipeline.py` / `tests/test_ml_strategy.py`:
- 受影响的 `MLFactorConfig(...)` 构造调用,在 PR-A 前默认 `embargo_days=0` 等价于旧行为。**全部已有测试需用 `embargo_days=0` 显式注入**,否则会因 auto embargo 行为差异而失败。

### 3.6 用户 YAML 迁移

PR-A 合入后,用户 `config.yaml` 当前结构:

```yaml
strategy:
  ml_factor:
    selector:
      type: lasso
      alpha: 0.001       # 旧扁平字段
      max_iter: 1000
      tol: 1.0e-6
```

必须改为:

```yaml
strategy:
  ml_factor:
    selector:
      type: lasso
      lasso:             # 新子段
        alpha: 0.001
        max_iter: 1000
        tol: 1.0e-6
    # 可选新增字段,不写就用 default:
    # embargo_days: null        # null = auto = horizon
    # label_type: return        # 默认值
```

PR-A 的 commit message 与 CLAUDE.md 更新中明确写出这个迁移点;不提供自动迁移脚本(项目规模小,手工改一次即可)。

## 4. 范围外 / 非目标

- 不实装 `"vol_adjusted"` / `"cross_sec_rank"` 标签计算(留 NotImplementedError 占位)。
- 不引入 lightgbm 依赖或新 selector/weighter type。
- 不改回测引擎、Strategy ABC、CLI。
- 不写自动 YAML 迁移工具。
- 不动 F1 plan-1 已交付的 `factors_analysis` 相关代码。

## 5. 验收标准

1. **零回归**:所有现有测试通过(测试方需要在 fixture 里显式传 `embargo_days=0` 保旧行为)。
2. **embargo 真生效**:`test_embargo_eliminates_label_leak` 在合成数据集上证明 train IC 与 test IC 差距收敛。
3. **配置硬切干净**:旧 YAML 立即报 ValidationError;新 YAML 子段写法正常解析。
4. **接口承诺**:`forward_return_panel(label_type="vol_adjusted")` 抛 NotImplementedError(而非静默忽略)。
5. **缓存失效自然**:首次跑 daily report / backtest 走全量重训,新 sig 写到 `ml_models/`。
6. **docs 同步**:`CLAUDE.md` 模块地图/配置段更新;`README.md` 配置示例更新。

## 6. 后续

PR-A 合入后,F2 路线接 PR-B(LightGBM selector + weighter + 默认切换 + lightgbm 依赖),然后进 F3。F1 plan-2(custom factors + A/B 验证)可与 PR-B 并行或在它之后。
