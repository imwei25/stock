# 回测框架(`stockpool.backtesting`)

策略无关的回测引擎。把"信号生成"和"执行规则"拆开,引擎不关心信号怎么来的,
策略只描述决策,不接触模拟器细节。新策略 = 写一个 `Strategy` 子类。

## 目录

1. [总览](#总览)
2. [核心 API](#核心-api)
   - [`Strategy`](#strategy)
   - [`BacktestEngine`](#backtestengine)
   - [`MultiLotBacktestEngine`](#multilotbacktestengine)
   - [`BacktestResult`](#backtestresult)
   - [`TradeCosts` / `Trade`](#tradecosts--trade)
   - [`BarContext` / `PositionContext`](#barcontext--positioncontext)
   - [`buy_and_hold_baseline`](#buy_and_hold_baseline)
   - [`compute_metrics`](#compute_metrics)
3. [引擎约定(必读)](#引擎约定必读)
4. [两个引擎怎么选](#两个引擎怎么选)
5. [内置策略](#内置策略)
6. [写一个新策略](#写一个新策略)
7. [与旧 API 的关系](#与旧-api-的关系)
8. [未支持的能力](#未支持的能力)

---

## 总览

```python
from stockpool.backtesting import (
    BacktestEngine, MultiLotBacktestEngine, TradeCosts,
    buy_and_hold_baseline, CompositeVerdictStrategy,
)

strategy = CompositeVerdictStrategy(weights, scoring, verdicts_cfg, indicators_cfg)

# 单仓位:同时只持一只票,信号反转换仓
single = BacktestEngine(
    strategy,
    costs=TradeCosts(buy_cost=0.0008, sell_cost=0.0013),
)
results = single.sweep_holding_days(daily_df, [5, 10, 20])

# 多仓位:每次 buy 都开一个独立的 lot,各自 N 天平仓。
# lot 大小由 LotSizer 决定 —— 项目 config 默认是 sizing.type=vol_target;
# 不传 lot_sizer 时引擎退回 FixedLotSizer(0.1)。
from stockpool.backtesting.sizing import FixedLotSizer, VolTargetLotSizer

multi = MultiLotBacktestEngine(
    strategy,
    lot_sizer=FixedLotSizer(0.1),     # 或 VolTargetLotSizer(...);position_size= 已废弃
    costs=TradeCosts(buy_cost=0.0008, sell_cost=0.0013),
)
multi_results = multi.run(daily_df, max_holding_days=10)
multi_results.trades              # 每个 lot 一条独立记录
multi_results.curve["position"]   # 当前开着的 lot 数

baseline = buy_and_hold_baseline(daily_df)
```

数据流:

```
daily_df ─► Strategy.generate_signals ─► signals (date, close, signal, ...)
                                              │
                                              ▼
                                     BacktestEngine._simulate
                                              │
                                              ▼
                            BacktestResult (curve, trades, metrics)
```

---

## 核心 API

所有符号都从 `stockpool.backtesting` 顶层包导出。

### `Strategy`

抽象基类。子类必须实现四个成员:

| 成员 | 类型 | 说明 |
|---|---|---|
| `name` | `property -> str` | 报告/日志中的策略标识 |
| `generate_signals(daily_df) -> DataFrame` | 方法 | 走样式信号生成。**必须满足 look-ahead 安全**:第 `i` 行只能依赖 `daily_df.iloc[:i+1]` |
| `should_enter(ctx: BarContext) -> bool` | 方法 | 空仓时每根 bar 调用一次 |
| `should_exit(ctx: PositionContext) -> bool` | 方法 | 持仓时每根 bar 调用一次。引擎另外强制 `days_held >= max_holding_days` 时退出,所以这里返回 `False` 仍受 `N` 限制 |
| `should_reset_timer(ctx: PositionContext) -> bool` | 方法(可选,默认 `False`) | 持仓时返回 `True` 把 `days_held` 重置为 `0` 并跳过本根 bar 的退出判定。用于"strong_buy 刷新 N 天计时"这类语义。**若与 `should_exit` 同时为真,重置胜出**(仓位续命,不平仓) |

`generate_signals` 返回的 DataFrame **必须**有列 `date`, `close`, `signal`。
其他列会保留在 `BacktestResult.signals` 中供报告使用。

### `BacktestEngine`

```python
BacktestEngine(strategy, costs=TradeCosts(), risk_free_rate=0.02)
```

**方法**:

| 方法 | 用途 |
|---|---|
| `run(daily_df, max_holding_days) -> BacktestResult` | 生成信号 + 模拟,一次跑完 |
| `run_on_signals(signals, max_holding_days) -> BacktestResult` | 传入已生成的信号帧;复用信号扫多个 N 时用 |
| `sweep_holding_days(daily_df, [N1, N2, ...]) -> dict[int, BacktestResult]` | 生成一次信号,跑多组 N |

### `MultiLotBacktestEngine`

```python
MultiLotBacktestEngine(
    strategy,
    lot_sizer=None,                # LotSizer 回调,决定每笔 buy 的 lot 大小
                                   #   (FixedLotSizer(0.1) / VolTargetLotSizer(...));
                                   #   项目 config 默认 sizing.type=vol_target
    position_size=None,            # 已废弃 = lot_sizer=FixedLotSizer(position_size);
                                   #   与 lot_sizer 互斥,两者都传 raise ValueError;
                                   #   两者都不传 → FixedLotSizer(0.1)
    costs=TradeCosts(),
    risk_free_rate=0.02,
    max_concurrent_lots=None,      # None = uncapped (cash自然封顶)
)
```

每个 enter 信号开一个**独立的 lot**(大小由 `LotSizer` 决定);每个 lot 有自己的
`days_held`,各自计 N、各自记账。`BacktestResult.trades` 里每个已平仓的 lot 是一条
记录。`curve["position"]` 变成"当前开着的 lot 数量"。

**资金模型**(`size` = sizer 对该笔 buy 返回的大小):

- 总资本起始 1.0;每个 lot 占用 `size` 的起始资本(如 0.1 = 10%)。
- 一笔 buy:`cash -= size`;新 lot 的有效投资金额 = `size * (1 - buy_cost)`。
- 若 sizer 返回 0(skip-fallback)或 buy 时 `cash < size`,该信号被跳过(不做部分成交)。
- 一笔 sell 或 lot 自己的 N 到期 → 关掉这个 lot:`cash += lot.current_value * (1 - sell_cost)`。
- 总净值 = `cash + Σ open_lots[i].current_value`。

**与单仓位的差异**:

| 维度 | `BacktestEngine` | `MultiLotBacktestEngine` |
|---|---|---|
| 同时持仓数 | 0 或 1 | 0 到 `max_concurrent_lots`(默认由现金封顶) |
| 重复 buy 信号 | 持仓时忽略 | 每次开新 lot(只要现金够) |
| 单笔 trade 收益 | 全仓收益 | 该 lot 独立收益 |
| `curve["position"]` | 0 或 1 | 开着的 lot 数量 |
| `Trade.ret` 分母 | 买入前全仓权益 `equity[t-1]` | 该 lot 的下单金额 `size`(扣 buy_cost 前) |

方法签名与 `BacktestEngine` 一致(`run` / `run_on_signals` / `sweep_holding_days`)。

### `BacktestResult`

| 字段 | 类型 | 说明 |
|---|---|---|
| `signals` | `DataFrame` | 策略产出的信号帧(保留全部额外列) |
| `curve` | `DataFrame` | 列:`date`, `equity`, `position`(净值序列,从 1.0 起) |
| `trades` | `list[Trade]` | 已平仓的交易 |
| `metrics` | `dict` | **全程口径**(从第 0 根 bar 起),见 [`compute_metrics`](#compute_metrics) |
| `max_holding_days` | `int` | 本次运行的 N |
| `strategy_name` | `str` | 来自 `strategy.name` |
| `metrics_active` | `dict \| None` | **活跃段口径**:净值曲线从第一笔 trade 的 `entry_idx` 起切片后重算的同一组指标。用于剔除 ml_factor 等策略的冷启动平头(训练样本不足时全程 neutral、equity 恒为 1),避免几何年化与夏普被前段稀释。无已平仓交易时为 `None`。`metrics` 语义不变,向后兼容 |

### `TradeCosts` / `Trade`

```python
@dataclass(frozen=True)
class TradeCosts:
    buy_cost: float = 0.0    # 进场扣减比例(佣金 + 滑点)
    sell_cost: float = 0.0   # 出场扣减比例(佣金 + 印花税 + 滑点)
```

值是**比例**而非百分比:`0.001` 即 0.1%。

```python
@dataclass(frozen=True)
class Trade:
    entry_idx: int        # 进场执行 bar 下标(= 信号 bar + 1)
    exit_idx: int         # 出场执行 bar 下标(= 信号 bar + 1)
    entry_price: float    # = open[entry_idx]
    exit_price: float     # = open[exit_idx]
    ret: float            # 净收益率(已扣除两端成本)
    days_held: int        # 持有 bar 数
    lot_size: float       # 多 lot 引擎:该 lot 的下单金额;单仓引擎恒为默认 0.1
```

**`ret` 口径**:分母是**买入前**投入的资金 —— 单仓引擎为买入前全仓权益
`equity[t-1]`,多 lot 引擎为该 lot 的下单金额 `size`(扣 `buy_cost` 之前)。
因此 `ret` 同时净掉 `buy_cost` 和 `sell_cost`:价格零波动的一次往返
`ret ≈ -(buy_cost + sell_cost)`。(P2-8 修复前分母误用扣过 buy_cost 的
entry_equity,导致每笔 ret 虚高约一个 buy_cost。)

### `BarContext` / `PositionContext`

引擎传给策略决策函数的只读快照。

```python
@dataclass(frozen=True)
class BarContext:           # → should_enter
    bar_idx: int
    date: pd.Timestamp
    close: float
    signal: Any

@dataclass(frozen=True)
class PositionContext:      # → should_exit
    bar_idx: int
    date: pd.Timestamp
    close: float
    signal: Any
    entry_idx: int          # 进场执行 bar 下标
    entry_price: float      # 进场成交价 = open[entry_idx]
    days_held: int          # 已持有 bar 数(含今天)
    max_holding_days: int   # 引擎配置的 N(可读但不可改)
```

`PositionContext` 暴露 `entry_price` 和 `close` ⇒ 策略可在 `should_exit` 里实现
止损 / 止盈 / 追踪止损,无需引擎特殊支持。

### `buy_and_hold_baseline`

```python
buy_and_hold_baseline(daily_df, risk_free_rate=0.02, label="buy_and_hold")
    -> BacktestResult
```

从第 0 根 bar 全仓持有到底,**不扣手续费**(基准默认不上摩擦,这样策略对比是
保守的)。`metrics["trade_count"] = 1`,`win_rate` / `avg_trade_return_pct` 为 `None`,
`metrics_active` 为 `None`(B&H 没有冷启动段)。

**口径(P3-16 统一)**:曲线与指标都锚定 `open[0]`(无 `open` 列时回退
`close[0]`):`equity[t] = close[t] / open[0]`(故 `equity[0]` 含第 0 日日内收益,
一般 ≠ 1.0);`total_return = close[-1] / open[0] - 1`,**包含** day-0 的
open→close 一段(指标内部在曲线前补一个 1.0 锚点再算,夏普/回撤/年化同样
看到 day-0 的波动)。

### `compute_metrics`

```python
compute_metrics(equity_series, trades, risk_free_rate=0.02,
                active_from_idx=None) -> dict
```

纯函数,不依赖策略或引擎状态。`active_from_idx` 不为 None 时先对净值序列做
`eq[active_from_idx:]` 切片再计算(活跃段口径,引擎用第一笔 trade 的
`entry_idx` 填)。返回:

| key | 含义 | 边界口径(P3-15) |
|---|---|---|
| `total_return` | `eq[-1] / eq[0] - 1` | 始终计算 |
| `annualized_return` | 几何年化(按 252 交易日) | 有效天数 < 60 时为 `None`(短窗口年化爆炸,如 10 天 +5% → +242%) |
| `max_drawdown` | 最大回撤(正值) | 始终计算 |
| `sharpe` | 日收益年化夏普,`risk_free_rate` 为年化无风险利率 | 有效天数 < 20 时为 `None` |
| `trade_count` | `len(trades)` | 始终计算 |
| `win_rate` | `ret > 0` 的交易占比 | 无已平仓交易时为 `None`(而非 0.0) |
| `avg_trade_return_pct` | 平均单笔净收益百分比 | 无已平仓交易时为 `None` |

`trades` 接受任何带 `.ret` 属性的对象 **或** `{"ret": ...}` 形式的 dict。
下游展示层(`backtest_report.py` / CLI stdout)对 `None` 显示 `—`。

---

## 引擎约定(必读)

| 约定 | 说明 |
|---|---|
| **T+1 + 次日开盘成交** | 决策只读 `signal[t-1]`,**成交价用 `open[t]`**(A 股集合竞价价)。除非次日开盘直接涨停否则视为可成交。signal 帧缺 `open` 时回退到 `close.shift(1)`,即旧的 close-to-close 行为。 |
| **多头单仓** | 不开空,不加仓。持仓时收到 enter 信号默认会被忽略,**除非**策略的 `should_reset_timer` 对该信号返回 True(见下"计时刷新"行)。 |
| **进场当日敞口** | `entry_equity = equity[t-1] * (1 - buy_cost)`,然后 `equity[t] = entry_equity * close[t]/open[t]`(open 到 close 的日内涨跌)。后续持仓日仍按 close-to-close 累计。 |
| **出场当日敞口** | `equity[t] = equity[t-1] * (open[t]/close[t-1]) * (1 - sell_cost)`,出场后当日剩余时间空仓。 |
| **Trade 下标** | `entry_idx` / `exit_idx` 指向**执行 bar `t`**,= 信号 bar + 1。`entry_price` / `exit_price` 即 `open[t]`。 |
| **持仓上限** | `days_held >= max_holding_days` 强制出场,优先于 `should_exit`。 |
| **末根 bar** | 末根 bar 的信号没有下一根可执行,等同于无操作。 |
| **未平仓** | 若最后仍持仓,该笔不计入 `trades`(`trade_count` 只统计已平仓)。 |
| **计时刷新** | `should_reset_timer` 返回 True 时,`days_held` 重置为 0,**优先于** `time_exit` 与 `should_exit`(reset 胜出)。`VerdictExecution` / `CompositeVerdictStrategy` 默认在 `strong_buy` 时刷新;传 `refresh_verdicts=()` 关闭。 |

**Look-ahead 安全契约**:`generate_signals` 必须保证第 `i` 行的所有计算只依赖
`daily_df.iloc[:i+1]`。引擎再延迟一根 bar 才使用 `signal[t-1]`,二者组合即可保证
回测无未来数据泄露。

---

## 两个引擎怎么选

| 你的语义 | 用哪个 |
|---|---|
| 一次只能持一只票,信号反转才换仓 | `BacktestEngine` |
| 每次看多都加新单,每单独立平仓 | `MultiLotBacktestEngine` |
| 想把固定金额平摊到多笔交易上 | `MultiLotBacktestEngine` |
| 不在意单笔统计,只看整体净值 | `BacktestEngine` 简单 |

**例子**:综合评级连续报出 `buy`、`strong_buy`、`buy`,标的横盘——

- `BacktestEngine`(默认 `refresh_verdicts=("strong_buy",)`):bar 1 进场后,
  bar 2 的 `strong_buy` **刷新计时器**(`days_held` 回到 0,N 重新起算);bar 3 的
  `buy` 被忽略。仓位最终因 N 满或 `sell` 而平仓。
- `BacktestEngine` + `refresh_verdicts=()`:旧语义,所有后续 buy/strong_buy 都被
  忽略,仓位一直持有到 N 满或 `sell`。
- `MultiLotBacktestEngine`:bar 1、bar 2、bar 3 各开一个 lot,各自从 0 计时,各自记
  入场价;同时 bar 2 的 strong_buy 还会**刷新已开 lot 的计时器**。N 到期或 sell
  触发各自平仓,`trades` 里独立记录。

## 内置策略

`stockpool.backtesting.strategies`:

| 类 | 用途 |
|---|---|
| `CompositeVerdictStrategy` | 项目原综合评级策略(`detect_signals` + 日周共振 + `verdict_of`) |
| `VerdictExecution` | 仅执行规则,不重新生成信号;搭配 `run_on_signals` 使用预生成信号帧 |
| `SMACrossStrategy` | 经典 SMA 金叉/死叉,作为框架可扩展性的示范 |

`CompositeVerdictStrategy` 和 `VerdictExecution` 共用三组参数:

| 参数 | 默认值 | 含义 |
|---|---|---|
| `buy_verdicts` | `("buy", "strong_buy")` | 触发 `should_enter` 的信号集合 |
| `sell_verdicts` | `("sell", "strong_sell")` | 触发 `should_exit` 的信号集合 |
| `refresh_verdicts` | `("strong_buy",)` | 持仓时触发 `should_reset_timer` 的信号集合;传 `()` 关闭刷新 |

---

## 写一个新策略

最小例子:RSI 超卖买入、超买卖出。

```python
import pandas as pd
from stockpool.backtesting import (
    BacktestEngine, BarContext, PositionContext, Strategy, TradeCosts,
)


class RSIReversionStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"rsi_reversion_{self.period}_{int(self.oversold)}_{int(self.overbought)}"

    def generate_signals(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, pd.NA)
        df["rsi"] = 100 - 100 / (1 + rs)
        df = df.dropna(subset=["rsi"]).reset_index(drop=True)

        def label(rsi):
            if rsi < self.oversold:    return "buy"
            if rsi > self.overbought:  return "sell"
            return "hold"
        df["signal"] = df["rsi"].map(label)
        return df[["date", "close", "signal", "rsi"]]

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal == "buy"

    def should_exit(self, ctx: PositionContext) -> bool:
        return ctx.signal == "sell"

    # Optional — omit to never refresh the N-day timer:
    # def should_reset_timer(self, ctx: PositionContext) -> bool:
    #     return ctx.signal == "buy"   # 比如让连续超卖刷新计时


engine = BacktestEngine(
    RSIReversionStrategy(period=14),
    costs=TradeCosts(buy_cost=0.0008, sell_cost=0.0013),
)
result = engine.run(daily_df, max_holding_days=10)
```

### 写一个带止损的策略

`PositionContext` 暴露了 `entry_price` 和 `close`,止损 / 止盈直接在 `should_exit`
里写,不用改引擎:

```python
def should_exit(self, ctx: PositionContext) -> bool:
    drawdown = (ctx.close - ctx.entry_price) / ctx.entry_price
    if drawdown <= -self.stop_loss_pct:
        return True
    if drawdown >= self.take_profit_pct:
        return True
    return ctx.signal == "sell"
```

---

## 与旧 API 的关系

`stockpool.backtest_composite` 是兼容层,内部全部委托给新框架:

| 旧 API | 新等价物 |
|---|---|
| `walk_forward_verdicts(daily, w, s, v, ind)` | `CompositeVerdictStrategy(...).generate_signals(daily)`(列名 `signal` → `verdict`) |
| `simulate_equity_curve(wf, [N], buy_cost, sell_cost, ...)` | `BacktestEngine(VerdictExecution(), TradeCosts(...)).run_on_signals(wf_renamed, N)` |
| `EquityResult` | 多个 `BacktestResult` 按 N 聚合而成的 dataclass |
| `verdict_bucket_stats(wf, forward_days)` | 留在 `backtest_composite`,与回测引擎无关的前瞻收益统计 |

新代码请直接用 `stockpool.backtesting`。

### 行为变更注意

`simulate_equity_curve` 底层用的是 `VerdictExecution()`(默认参数),而
`VerdictExecution` 现在默认 `refresh_verdicts=("strong_buy",)`。
意味着旧 API 在历史出现 `strong_buy` 信号的标的上,**净值曲线会与改动前不同**
(因为持仓被 `strong_buy` 续命)。

要复现改动前的语义,显式传空集合即可:

```python
from stockpool.backtesting import BacktestEngine, VerdictExecution, TradeCosts

engine = BacktestEngine(
    VerdictExecution(refresh_verdicts=()),   # 关掉刷新
    costs=TradeCosts(buy_cost=0.0008, sell_cost=0.0013),
)
```

(`backtest_composite.simulate_equity_curve` 目前没有对外暴露这个参数;
如果生产链路依赖旧行为,改 `backtest_composite.py:simulate_equity_curve`
那一行 `VerdictExecution()` 为 `VerdictExecution(refresh_verdicts=())` 即可。)

---

## 未支持的能力

明确不支持的部分(避免使用时踩坑):

- **做空** — 引擎是 long-only。需要做空请等扩展或自行实现。
- **仓位管理** — `BacktestEngine` 满仓进出;`MultiLotBacktestEngine` 固定额度分仓。
  无 Kelly / 比例追加 / 部分成交。
- **多标的组合** — 一次只回测一只标的的曲线。
- **滑点模型** — 只有定额成本比例,无成交量相关滑点 / 部分成交 / 跳空缺口处理。
- **资金成本** — 杠杆 / 融券利息不建模。
- **盘中 / 多周期** — 只支持日线级,周/月需先用 `resample_to_weekly` 之类的工具
  转好再传入。

如果需要其中某项,先看 `Strategy` 能否通过 `should_exit` 的 `PositionContext`
组合出来(止损/止盈/动态出场都可以);需要改引擎才能支持的能力(如做空、
多标的)按需扩展。
