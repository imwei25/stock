# F3 PR-C — Vol-target sizing

## 1. 背景与范围

属于 `docs/strategy_improvement_2026.md` §5 (F3 组合构建与风险控制) 三个独立子 PR 的第一个:

- **PR-C(本 spec)** — Sizing 子段化 + vol-target 仓位 实现
- **PR-D**(后续)— Risk overlay(max DD 熔断 + sector cap)
- **PR-E**(后续)— MLFactor score 平滑(EMA)

PR-C 不依赖任何待做项,可独立启动。**不引入新依赖**(numpy 现有),**不改 Strategy ABC**,**不改 signal frame schema**,**不改 ml 子包**。改动集中在 `BacktestConfig` schema、`backtesting/framework.py` 引擎构造、新模块 `backtesting/sizing.py`,加上 5 处顶层接线和文档同步。

### 1.1 解决的问题

当前 `BacktestConfig.position_size: float = 0.1` 是死值:无论茅台(年化 vol ~25%)还是 ST 妖股(年化 vol ~60%),每次开仓都用 10% 总资产。后者一次黑天鹅就给账户砸出大坑,前者风险敞口偏小。vol-target sizing 按"波动大的票仓位小、波动小的票仓位大"调整,使每只票贡献的风险大致相等,**主效应是降低组合 max DD**,而非提升 Sharpe(见 §6 验证哲学)。

### 1.2 非目标(明确写出避免 scope creep)

- 不做 PR-D 的 risk overlay(熔断 / sector cap)—— sizing 钩子设计好后,PR-D 复用同一处接入即可
- 不做 PR-E 的 score smoothing
- 不动 `cfg.backtest.engine`(`single` / `multi_lot`),vol-target 只对 `multi_lot` 生效 ——`single` 引擎不支持多仓位概念,sizing 没有语义
- 不引入新的 vol 估计法(EWMA / Parkinson 等),只有简单滚动 std
- 不解决 portfolio-level sector 集中度(留给 PR-D + 上层聚合层)

## 2. 设计

### 2.1 配置 schema(`src/stockpool/config.py`)

新增三个 Pydantic 子模型 + `BacktestConfig.sizing` 字段 + `position_size` 字段保留为 deprecated alias。

```python
class FixedSizingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: float = Field(default=0.1, gt=0.0, le=1.0)


class VolTargetSizingConfig(BaseModel):
    """Vol-target sizing parameters.

    Formula (β, relative-to-baseline):
        size = fixed.size * (reference_vol_annual / recent_vol_annual)
        size = clip(size, min_size, max_size)

    `fixed.size` doubles as the baseline anchor: at recent_vol = reference_vol,
    the lot equals fixed.size. Vol estimator: simple rolling std over `vol_window`
    bars of daily simple returns, annualised with sqrt(252).
    """
    model_config = ConfigDict(extra="forbid")
    reference_vol_annual: float = Field(default=0.30, gt=0.0)
    vol_window: int = Field(default=20, gt=1)
    min_size: float = Field(default=0.03, gt=0.0, le=1.0)
    max_size: float = Field(default=0.20, gt=0.0, le=1.0)
    fallback_to: Literal["fixed", "skip"] = "fixed"

    @model_validator(mode="after")
    def _check_min_le_max(self):
        if self.min_size > self.max_size:
            raise ValueError(
                f"min_size ({self.min_size}) must be <= max_size ({self.max_size})"
            )
        return self


class SizingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["fixed", "vol_target"] = "vol_target"  # ← 默认 vol_target
    fixed: FixedSizingConfig = Field(default_factory=FixedSizingConfig)
    vol_target: VolTargetSizingConfig = Field(default_factory=VolTargetSizingConfig)


class BacktestConfig(BaseModel):
    # ... existing fields unchanged ...
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    position_size: float | None = Field(default=None, gt=0.0, le=1.0)
    # ↑ Deprecated. None = 用 sizing.fixed.size。非 None 时:
    # (1) 与 sizing 显式冲突 → ValidationError
    # (2) 单独出现 → 迁移到 sizing.fixed.size + DeprecationWarning

    @model_validator(mode="after")
    def _migrate_position_size(self):
        if self.position_size is None:
            return self
        # 用户同时显式写了 sizing.fixed.size 或 sizing.type → 冲突
        sizing_explicit = (
            self.sizing.type != "vol_target"        # type 被显式改过
            or self.sizing.fixed.size != 0.1        # fixed.size 被显式改过
        )
        if sizing_explicit:
            raise ValueError(
                "Cannot set both backtest.position_size (deprecated) and "
                "backtest.sizing. Migrate position_size into sizing.fixed.size."
            )
        warnings.warn(
            "backtest.position_size is deprecated; use "
            "backtest.sizing.fixed.size (and sizing.type=fixed) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # 迁移并保持 backwards-compat:type 切回 fixed
        self.sizing = SizingConfig(
            type="fixed",
            fixed=FixedSizingConfig(size=self.position_size),
        )
        self.position_size = None  # 迁移后清空,避免下次校验再触发
        return self
```

**冲突检测的近似性说明**:`_migrate_position_size` 通过"`sizing.type != "vol_target"` or `sizing.fixed.size != 0.1`"判断 sizing 是否被显式写过 —— 这是基于默认值差异的启发式,不是字段级 set/unset 跟踪。极端情况:用户写 `sizing: {type: vol_target, fixed: {size: 0.1}}`(显式等于默认)+ `position_size: 0.1`,会被判定为"未显式" → 静默迁移 + warning。这种边界 case 行为不算 bug —— 用户得到的是预期的 fixed 0.1 行为,只是没察觉到 sizing.type 被覆写回 fixed。**这是合理的损失**,换来 99% case 的简洁实现。如未来要求严格,可改用 `model_fields_set` 检查显式字段。

### 2.2 引擎层(新模块 `backtesting/sizing.py`)

完全独立于 Pydantic config —— 用 Protocol 而非依赖 `SizingConfig`:

```python
"""Lot sizing strategies for MultiLotBacktestEngine.

A LotSizer is a callable that, given the engine's current bar index and the
recent OHLC arrays, returns the fraction of starting capital to commit on the
next entry. Engine remains config-agnostic: callers build a LotSizer from
config (via build_lot_sizer in the strategy_factory wiring layer) and inject it.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class LotSizer(Protocol):
    def __call__(
        self, bar_idx: int, opens: np.ndarray, closes: np.ndarray
    ) -> float:
        """Return the lot size (fraction of initial equity) for a buy at bar_idx.

        Convention: bar_idx is the EXECUTION bar (where the fill happens at
        opens[bar_idx]). The sizer may only inspect closes[:bar_idx] — it must
        not peek at the execution bar's close to preserve look-ahead safety
        (matching the engine's T+1 contract).
        """


class FixedLotSizer:
    """Returns a constant size regardless of state."""

    def __init__(self, size: float):
        if not (0 < size <= 1.0):
            raise ValueError(f"size must be in (0, 1], got {size}")
        self.size = size

    def __call__(self, bar_idx: int, opens: np.ndarray, closes: np.ndarray) -> float:
        return self.size


class VolTargetLotSizer:
    """Scales size inversely to recent realised volatility (β formula).

    size = baseline_size * (reference_vol_annual / recent_vol_annual)
    size = clip(size, min_size, max_size)

    Recent vol estimated as simple rolling std of daily simple returns on
    `closes[bar_idx - vol_window : bar_idx]`, annualised with sqrt(252).

    Fallback (cold-start, NaN, or zero vol):
      - "fixed": return baseline_size (treat as if at reference_vol)
      - "skip":  return 0.0 (engine should skip the buy)
    """

    _ANNUALISATION = 252.0

    def __init__(
        self,
        baseline_size: float,            # = sizing.fixed.size
        reference_vol_annual: float,
        vol_window: int,
        min_size: float,
        max_size: float,
        fallback: str = "fixed",         # "fixed" | "skip"
    ):
        if fallback not in ("fixed", "skip"):
            raise ValueError(f"fallback must be 'fixed' or 'skip', got {fallback!r}")
        self.baseline_size = baseline_size
        self.reference_vol_annual = reference_vol_annual
        self.vol_window = vol_window
        self.min_size = min_size
        self.max_size = max_size
        self.fallback = fallback

    def _fallback_size(self) -> float:
        if self.fallback == "skip":
            return 0.0
        return float(np.clip(self.baseline_size, self.min_size, self.max_size))

    def __call__(self, bar_idx: int, opens: np.ndarray, closes: np.ndarray) -> float:
        # 需要 vol_window+1 个 close 才能算 vol_window 个 returns
        if bar_idx < self.vol_window + 1:
            return self._fallback_size()
        window_closes = closes[bar_idx - self.vol_window - 1 : bar_idx]
        if np.any(~np.isfinite(window_closes)) or np.any(window_closes <= 0):
            return self._fallback_size()
        rets = np.diff(window_closes) / window_closes[:-1]
        if rets.size == 0:
            return self._fallback_size()
        recent_vol_daily = float(np.std(rets, ddof=1))
        if not np.isfinite(recent_vol_daily) or recent_vol_daily <= 0:
            return self._fallback_size()
        recent_vol_annual = recent_vol_daily * np.sqrt(self._ANNUALISATION)
        raw = self.baseline_size * (self.reference_vol_annual / recent_vol_annual)
        return float(np.clip(raw, self.min_size, self.max_size))
```

**对 vol 窗口的设计取舍**:vol_window=20 默认值意味着回测最初 ~21 个交易日强制走 fallback。在 500-bar 标准 A/B 上,等于丢掉了前 ~4% 的样本期 —— 默认 `fallback=fixed` 时这段时间引擎以 baseline_size 行为运行,与 fixed sizing arm 完全一致,所以**冷启动期对 A/B 归因是中性的**(两 arm 在前 21 bar 行为相同,差异从第 22 bar 才开始累积)。

**look-ahead 安全**:取窗口 `closes[bar_idx - vol_window - 1 : bar_idx]` —— 注意切片**不含** `closes[bar_idx]`,即执行 bar 自己的 close。引擎调用时,sizer 看到的 closes 数组是 `signals["close"].values`(full length),但只能用到决策 bar `t-1` 及之前的数据。决策时点是 bar `t-1` 收盘后,执行在 bar `t` 开盘;算 vol 用 `closes[t-1-vol_window : t-1+1]`,即 `closes[t-vol_window-1 : t]`(Python 切片)= 长度 `vol_window+1` 的窗口,产生 `vol_window` 个 returns。

### 2.3 引擎层(`backtesting/framework.py`)

`MultiLotBacktestEngine.__init__` 签名扩展:

```python
def __init__(
    self,
    strategy: Strategy,
    position_size: float | None = None,    # ← 旧参数,deprecated
    lot_sizer: LotSizer | None = None,     # ← 新参数
    costs: TradeCosts = TradeCosts(),
    risk_free_rate: float = 0.02,
    max_concurrent_lots: int | None = None,
):
    if lot_sizer is not None and position_size is not None:
        raise ValueError(
            "Pass either `lot_sizer` or `position_size`, not both. "
            "`position_size` is deprecated; prefer `lot_sizer=FixedLotSizer(size)`."
        )
    if lot_sizer is None:
        # 老调用方未传 lot_sizer:回退到 fixed 行为
        size = position_size if position_size is not None else 0.1
        lot_sizer = FixedLotSizer(size)
    # 框架层不发 DeprecationWarning:Pydantic 已在 YAML 读入路径发过;
    # 直接用 Python 调引擎(测试代码)传 position_size= 不打扰。
    self.strategy = strategy
    self.lot_sizer = lot_sizer
    self.costs = costs
    self.risk_free_rate = risk_free_rate
    self.max_concurrent_lots = max_concurrent_lots
    # 注意:不再存 self.position_size(老测试 monkey-patch 这个字段的话会被发现)
```

**注**:`position_size` 字段在引擎层故意**不发** DeprecationWarning ——Pydantic 那一层已经处理了 YAML 读入路径,框架层若再发会双重提示。直接用 Python 调引擎的测试代码本身不涉及 config schema,继续支持 `position_size=` 不打扰。

`_simulate_multi_lot` 在 `__init__` 之后已通过 `engine.lot_sizer` 传入。函数签名变化:

```python
def _simulate_multi_lot(
    signals: pd.DataFrame,
    *,
    strategy: Strategy,
    lot_sizer: LotSizer,                  # ← 替换 position_size, max_concurrent_lots 不变
    max_concurrent_lots: int | None,
    max_holding_days: int,
    costs: TradeCosts,
    risk_free_rate: float,
) -> BacktestResult:
```

`MultiLotBacktestEngine.run_on_signals` 改为 `lot_sizer=self.lot_sizer` 透传。

第 4 步(原 framework.py:511-528)改造:

```python
# 4. Maybe open a new lot — fills at open[t]; first-day exposure is
#    open[t] → close[t].
bctx = BarContext(
    bar_idx=t - 1, date=prev_date,
    close=prev_close, signal=prev_signal,
)
capacity_ok = (
    max_concurrent_lots is None
    or len(open_lots) < max_concurrent_lots
)
if strategy.should_enter(bctx) and capacity_ok:
    size = lot_sizer(t, opens, closes)  # opens/closes 是 numpy array
    if size > 0 and cash >= size:
        cash -= size
        committed = size * (1 - costs.buy_cost)
        open_lots.append(_OpenLot(
            entry_idx=t,
            entry_price=open_t,
            committed_cash=committed,
            current_value=committed * (close_t / open_t),
            days_held=0,
            lot_size=size,                 # ← 新增字段,见 §2.4
        ))
```

**`size=0`(skip fallback) 不报 warning** —— 这是有意行为,频率会被 `metrics["trade_count"]` 间接反映。归因 `size=0` 占比的工作留给 A/B 报告自己 derive。

### 2.4 `Trade` / `_OpenLot` dataclass 新增 `lot_size`

```python
@dataclass(frozen=True)
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    ret: float
    days_held: int
    lot_size: float = 0.1          # ← 新增,默认 0.1 保持老 fixture 兼容

@dataclass
class _OpenLot:
    entry_idx: int
    entry_price: float
    committed_cash: float
    current_value: float
    days_held: int = 0
    lot_size: float = 0.1          # ← 新增
```

`_OpenLot → Trade` 转换时把 `lot_size` 透传(framework.py:493-500 附近)。

**为何不把它放进 metrics 字段**:`compute_metrics` 是聚合统计,per-trade lot_size 是 raw 字段,放在 `Trade` 对象上让 A/B 报告或自定义分析脚本能直接 access,无需重新跑回测。

### 2.5 顶层接线

新增 helper(放在 `backtesting/sizing.py` 或 `backtest_runner.py`,推荐前者保持 sizing 模块自洽):

```python
def build_lot_sizer(cfg: "SizingConfig") -> LotSizer:
    """Build a LotSizer from BacktestConfig.sizing.

    Pure factory; no I/O, no side effects. Lives in backtesting.sizing to keep
    the dependency direction strategy_factory → sizing → config (config types
    are accepted as a TYPE_CHECKING reference; runtime accepts duck-typed objs).
    """
    if cfg.type == "fixed":
        return FixedLotSizer(cfg.fixed.size)
    if cfg.type == "vol_target":
        vt = cfg.vol_target
        return VolTargetLotSizer(
            baseline_size=cfg.fixed.size,
            reference_vol_annual=vt.reference_vol_annual,
            vol_window=vt.vol_window,
            min_size=vt.min_size,
            max_size=vt.max_size,
            fallback=vt.fallback_to,
        )
    raise ValueError(f"Unknown sizing.type: {cfg.type!r}")
```

**对 doc §5 字段名的偏离说明**:doc §5 原草图里同时列了 `target_vol_annual` 和 `reference_vol_annual` 两个字段,反映当时还在 (α)/(β) 公式间犹豫。本 spec 确定走 β 公式,只需 `reference_vol_annual` 一个锚,**故删除 `target_vol_annual`**。如未来想加 (α) 公式作可选模式(`sizing.vol_target.formula: "beta" | "alpha"`),再开 sub-PR,届时重新引入 `target_vol_annual` 字段。

**改动以下文件,统一 wire 上 `build_lot_sizer`**:

| 文件 | 改动 |
|------|------|
| `cli.py` `cmd_backtest` | 引擎构造从 `position_size=cfg.backtest.position_size` 改为 `lot_sizer=build_lot_sizer(cfg.backtest.sizing)`;`engine_label` 字符串展示 sizing type(`multi_lot · vol_target` / `multi_lot · fixed 10%`) |
| `backtest_runner.py` | `backtest_stocks` / `prepare_pool` 路径同 cli |
| `backtest_composite.py` | `simulate_equity_curve` 适配层同上 |
| `strategy_factory.py` | `simulate` 函数同上(ML 路径) |
| `ab/config.py` | `ArmBacktestOverride` 加 `sizing: SizingConfig \| None = None`(并同步把现有的 `position_size: float \| None = None` 标 deprecated 但保留,与顶层 `BacktestConfig` 迁移策略对齐);`ab/runner.py` 不动。详见 §2.6 |

### 2.6 A/B 工具集成

`ab.yaml` 现有示例已支持 arm 级 `backtest:` 覆盖。`ab/config.py` 的 `ArmBacktestOverride` 是显式字段白名单(`extra="forbid"`),所以**必须显式加字段** —— 不是自动继承:

- 加 `sizing: SizingConfig | None = None`(None = 继承 base.backtest.sizing)
- 现有 `position_size: float | None = None` 保留并标 deprecated(与顶层 `BacktestConfig` 迁移策略对齐)。`build_effective_cfg` 在合并后调用 `AppConfig` 验证时,顶层 `_migrate_position_size` 自动处理迁移和冲突 —— 不需要在 `ArmBacktestOverride` 重复迁移逻辑

验证下面这种写法可解析:

```yaml
arms:
  fixed_baseline:
    backtest:
      sizing: {type: fixed, fixed: {size: 0.10}}
      equity_curve_holding_days: [10]
  vol_target_default:
    backtest:
      sizing:
        type: vol_target
      equity_curve_holding_days: [10]
```

**`build_effective_cfg` 的 merge 语义**:base.backtest 的 `sizing` 子段被 arm 显式提供时,**整段替换**(因为 `SizingConfig` 内部 type-discriminated,部分覆盖语义复杂)。这是 ArmOverride 既有约定的延伸,与 base.strategy 的"整段替换"对齐。**需要在 `test_ab.py` 加 case 显式断言这点,且在 `docs/superpowers/specs/2026-05-24-ab-testing-design.md` 引用本 spec 的对应段落作为 cross-reference**。

## 3. 测试

| 文件 | 新增 / 修改 |
|------|------------|
| `tests/test_sizing.py` 新增 | `FixedLotSizer` 返常量;`VolTargetLotSizer` 合成高/低 vol 数据(stub closes 数组),验证 size 反比 vol;clip 边界(min/max);cold-start fallback(bar_idx < vol_window+1);NaN/0 vol fallback;`fallback=skip` 返 0;`build_lot_sizer` 工厂分派 |
| `tests/test_multi_lot_engine.py` 补 | 引擎接受 `lot_sizer=` 时按动态 size 开仓;`lot_sizer=None` `position_size=None` 默认 0.1;同时传 lot_sizer 和 position_size 报错;`size > cash` 时跳过 buy(对应 `size_target_lot_sizer` 极端值或 skip fallback);`Trade.lot_size` 字段正确透传 |
| `tests/test_config.py` 补 | `SizingConfig` 三层校验;`position_size` 单独 → 自动迁移 + DeprecationWarning;`position_size` + 显式 `sizing.fixed.size=0.2` → ValidationError;`position_size` + 显式 `sizing.type=vol_target` → ValidationError;`vol_target.min_size > max_size` → ValidationError |
| `tests/test_backtest_composite.py` 现有 | 现有用例的数学契约不变(`sizing.type=fixed, fixed.size=0.10` 行为 = 老 `position_size=0.10`)。但因 default 改为 vol_target,fixture 必须显式 pin `sizing.type=fixed` 才能保持原有结果。修改:在 `make_default_config` 或类似 fixture 里把 `sizing.type` 设为 `fixed`(per §3 下方推荐 (b)) |
| `tests/test_cli_backtest.py` 现有 | smoke 烟雾;engine label 显示 sizing type 的回归 |
| `tests/test_ab.py` 补 | `ArmOverride.backtest.sizing` 合并 = 整段替换;两个 arm 用不同 sizing 配置时 effective_cfg 隔离 |
| `tests/test_cli_ab.py` 现有 | 不动 |

**测试关注点 — 老回归 vs 新行为**:

由于本 PR 改了 `sizing.type` 的默认值(`fixed → vol_target`),所有不显式 pin sizing 的旧测试都会改 behavior。**两条处理路径,需要明确选一条**:

- **(a) 全部旧测试改用 `position_size=0.1`(走迁移路径)**:同时验证迁移逻辑,但 DeprecationWarning 会污染 pytest 输出(可用 `filterwarnings`);
- **(b) 全部旧测试改用 `sizing={"type": "fixed", "fixed": {"size": 0.1}}`(走新路径但显式 type=fixed)**:更新工作量大但更干净;

**推荐 (b)**。理由:DeprecationWarning 应该真的引导用户改 YAML,而不是被 `filterwarnings` 静默掉;迁移逻辑由专门的 `test_config.py` 用例 cover 即可。

## 4. 文档同步

按项目"CLAUDE.md + README.md 双更新"原则:

- `CLAUDE.md` 配置段:`backtest` 字段表新增 `sizing` 子段说明;`position_size` 行加 deprecation marker;一并更新模块地图里 `backtesting/sizing.py` 行
- `CLAUDE.md` 测试段:加 `test_sizing.py` 一行
- `README.md` 快速命令 / 配置示例:`config.yaml` 默认 sizing 段加 1 行注释("默认 vol_target,可改 fixed 回到 0.1 死值");`backtest` 命令本身不变
- `docs/strategy_improvement_2026.md` §6 P1 表:PR-C 行从 🚧 移到 ✅,verdict 由实际 A/B 结果填入(由后续实现 PR 完成,不在本 spec 范围)
- 新 spec 文件交叉链接:本文件链接 doc §6 / strategy_improvement_2026.md;反向 doc §6 PR-C 行加 spec 链接

## 5. 实现顺序(供后续 plan 参考)

建议拆 4 个 checkpoint,每个独立可测可 commit:

1. **新模块 `backtesting/sizing.py`** + `test_sizing.py` —— 完全 standalone,无须接引擎。这一步先 LGTM 锁住 sizing 数学
2. **引擎签名 + `_simulate_multi_lot` 接 `lot_sizer`** + `Trade.lot_size` + `test_multi_lot_engine.py` 补测 —— 引擎层零回归是关键 gate
3. **`config.py` schema + migration + `test_config.py`** + 5 处顶层接线(cli / backtest_runner / backtest_composite / strategy_factory / ab/config)+ 旧测试 fixture 改成显式 `sizing.type=fixed`(per §3 (b))
4. **文档同步 + A/B 跑一遍 + 把 verdict 写回 strategy_improvement_2026.md §6**

A/B 结果归档到 `docs/ab_runs/<date>-pr-c-sizing.html`(per 项目惯例)。

## 6. A/B 验证

### 6.1 配置

新建 `ab_sizing.yaml`(spec 落地时一并提交):

```yaml
base_config: config.yaml

arms:
  fixed_baseline:
    strategy:
      name: ml_factor
      ml_factor:
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      sizing: {type: fixed, fixed: {size: 0.10}}
      equity_curve_holding_days: [10]

  vol_target:
    strategy:
      name: ml_factor
      ml_factor:
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      sizing:
        type: vol_target
        fixed: {size: 0.10}
        vol_target:
          reference_vol_annual: 0.30
          vol_window: 20
          min_size: 0.03
          max_size: 0.20
          fallback_to: fixed
      equity_curve_holding_days: [10]
```

两 arm 的 strategy 段完全一致,**差异只来自 sizing**。这样跑出来 Δ 归因干净。

### 6.2 Gate(必过)

- **`Δmax_dd / fixed.max_dd ≤ -0.20`** —— max DD 收窄 ≥ 20%
- **`Δsharpe ≥ -0.05`** —— Sharpe 不显著退步(允许小幅波动,因为单位风险下的均值收益不是 vol-target 的主目标)

### 6.3 成功目标(期望达成)

- max DD 收窄 ≥ 30% **且** 年化收益不退 > 2pp **且** Sharpe ≥ baseline

### 6.4 验证哲学说明(重要)

**vol-target 在 long-only A 股个股上,主效应是降低 max DD,不是提升 Sharpe**。这与 F2 系列 (PR-A/B1/B2) 的 Sharpe-driven 验证哲学不同 —— F2 改的是 alpha 信号本身,Sharpe 是主指标;PR-C 改的是仓位大小,组合每只票贡献的风险变得均衡,Sharpe 在 long-only 个股回测里改善有限,主要收益体现在 DD 控制上。

**写进 spec 是为了防止**:后续 Claude 看到 A/B 结果"Sharpe 持平 / 微跌"就误判为 regression,触发不必要的回滚。Sharpe 持平 + DD 收窄 ≥ 20% 就是成功。

### 6.5 零回归契约

`sizing.type=fixed, sizing.fixed.size=0.10` 必须与上一版默认行为完全一致:

- `test_multi_lot_engine.py` / `test_backtest_composite.py` 所有现有用例通过(配合 §3 (b) 的 fixture 更新)
- `python -m stockpool backtest --config config.yaml` 在改回 `sizing.type=fixed` 时净值曲线与本 PR 前 100% 复现(可手动 diff 一次 `reports/backtest/latest.html` 的 metrics)

## 7. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|-----|------|------|
| vol-target 在 16 股 × 500 bar 样本上 DD 改善 < 20%(不过 gate)| 中 | 不达预期但不 break 现有功能 | 仍保留 `sizing.type=fixed` 作 default 备选;A/B 报告归档,doc §6 verdict 写 ⚠️ tied;不强行切默认 |
| 用户 YAML 写法太多种,Pydantic 报错歧义 | 低 | 用户体验差 | §2.1 写明三种典型场景的预期行为;`test_config.py` 各 cover 一例 |
| `Trade.lot_size` 字段对老 fixture 不兼容 | 低 | 老测试报错 | default=0.1 + frozen=True;手搓 `Trade(...)` 调用的老 fixture 不需要改 |
| ML 模型缓存(`ml_models/*.pkl`)受 sizing 改动影响 | 无 | 无 | sizing 不影响 `_strategy_signature`;ml 缓存仍然有效 |
| `sizing.type=vol_target` 默认改变让 silent default change 难以察觉 | 中 | 用户升级版本后行为变 | (a) DeprecationWarning 显式提示;(b) CHANGELOG / README 显式标"breaking default change";(c) doc §6 路线图 P1 完成后立刻记录 |

## 8. 兼容性矩阵

| 用户输入 | 行为 |
|---------|------|
| 全新 YAML,不写 sizing / position_size | 默认 vol_target,所有默认值生效 |
| 全新 YAML,显式 `sizing.type=fixed` | fixed,size 默认 0.1 |
| 全新 YAML,显式 `sizing.type=vol_target` | vol_target,所有 vol_target.* 默认值生效 |
| 旧 YAML,只写 `position_size: 0.15` | DeprecationWarning + 迁移到 `sizing.type=fixed, sizing.fixed.size=0.15` |
| 旧 YAML,只写 `position_size: 0.10`(等于默认) | DeprecationWarning + 迁移到 `sizing.type=fixed, sizing.fixed.size=0.10` |
| 同时写 `position_size: 0.1` + `sizing.type=vol_target` | ValidationError(冲突) |
| 同时写 `position_size: 0.1` + `sizing.fixed.size=0.2` | ValidationError(冲突) |
| 同时写 `position_size: 0.1` + `sizing.fixed.size=0.1`(都等于默认) | 触发"近似性"判定的 false negative:静默迁移成 fixed.size=0.1(用户得到的是预期行为,但 sizing.type 静默回退到 fixed)。见 §2.1 末段说明 |

## 9. Cross-references

- 路线图:`docs/strategy_improvement_2026.md` §5 F3 + §6 P1
- A/B 工具:`docs/superpowers/specs/2026-05-24-ab-testing-design.md`
- 引擎契约:`docs/backtesting_framework.md` + `CLAUDE.md` "引擎约定" 段
- 前置 PR(无依赖,但行为基础来自):F2 PR-A/B1/B2 的子段化 schema 风格
