# 回测框架(`stockpool.backtesting`)

策略无关的回测引擎。把"信号生成"和"执行规则"拆开,引擎不关心信号怎么来的,
策略只描述决策,不接触模拟器细节。新策略 = 写一个 `Strategy` 子类。

## 目录

1. [总览](#总览)
2. [核心 API](#核心-api)
   - [`Strategy`](#strategy)
   - [`BacktestEngine`](#backtestengine)
   - [`BacktestResult`](#backtestresult)
   - [`TradeCosts` / `Trade`](#tradecosts--trade)
   - [`BarContext` / `PositionContext`](#barcontext--positioncontext)
   - [`buy_and_hold_baseline`](#buy_and_hold_baseline)
   - [`compute_metrics`](#compute_metrics)
3. [引擎约定(必读)](#引擎约定必读)
4. [内置策略](#内置策略)
5. [写一个新策略](#写一个新策略)
6. [与旧 API 的关系](#与旧-api-的关系)
7. [未支持的能力](#未支持的能力)

---

## 总览

```python
from stockpool.backtesting import (
    BacktestEngine, TradeCosts, buy_and_hold_baseline,
    CompositeVerdictStrategy,
)

strategy = CompositeVerdictStrategy(weights, scoring, verdicts_cfg, indicators_cfg)
engine = BacktestEngine(
    strategy,
    costs=TradeCosts(buy_cost=0.0008, sell_cost=0.0013),
    risk_free_rate=0.02,
)

# 一次性扫多个 N
results = engine.sweep_holding_days(daily_df, [5, 10, 20])
baseline = buy_and_hold_baseline(daily_df)

results[5].metrics["sharpe"]        # 夏普
results[5].curve                    # date, equity, position
results[5].trades                   # list[Trade] — 已平仓交易明细
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

### `BacktestResult`

| 字段 | 类型 | 说明 |
|---|---|---|
| `signals` | `DataFrame` | 策略产出的信号帧(保留全部额外列) |
| `curve` | `DataFrame` | 列:`date`, `equity`, `position`(净值序列,从 1.0 起) |
| `trades` | `list[Trade]` | 已平仓的交易 |
| `metrics` | `dict` | 见 [`compute_metrics`](#compute_metrics) |
| `max_holding_days` | `int` | 本次运行的 N |
| `strategy_name` | `str` | 来自 `strategy.name` |

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
    entry_idx: int        # 进场 bar 在 signals 中的下标
    exit_idx: int         # 出场决策日(信号日,非成交日)
    entry_price: float
    exit_price: float
    ret: float            # 净收益率(已扣除两端成本)
    days_held: int        # 持有 bar 数
```

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
    entry_idx: int          # 进场 bar 下标
    entry_price: float      # 进场收盘价
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
保守的)。`metrics["trade_count"] = 1`,`win_rate` / `avg_trade_return_pct` 为 `None`。

### `compute_metrics`

```python
compute_metrics(equity_series, trades, risk_free_rate=0.02) -> dict
```

纯函数,不依赖策略或引擎状态。返回:

| key | 含义 |
|---|---|
| `total_return` | `eq[-1] / eq[0] - 1` |
| `annualized_return` | 几何年化(按 252 交易日) |
| `max_drawdown` | 最大回撤(正值) |
| `sharpe` | 日收益年化夏普,`risk_free_rate` 为年化无风险利率 |
| `trade_count` | `len(trades)` |
| `win_rate` | `ret > 0` 的交易占比 |
| `avg_trade_return_pct` | 平均单笔净收益百分比 |

`trades` 接受任何带 `.ret` 属性的对象 **或** `{"ret": ...}` 形式的 dict。

---

## 引擎约定(必读)

| 约定 | 说明 |
|---|---|
| **T+1 决策** | 决策只读 `signal[t-1]`,成交价用 `close[t]`。`close[t]` 永远不会被 bar `t` 决策时看到。 |
| **多头单仓** | 不开空,不加仓;持仓时收到 buy 信号被忽略。 |
| **进场成本** | `entry_equity = equity[t-1] * (1 - buy_cost)`,然后承接当日收益。 |
| **出场成本** | `exit_equity = equity[t-1] * (1 - sell_cost)`,出场日不再承接价格变动。 |
| **持仓上限** | `days_held >= max_holding_days` 强制出场,优先于 `should_exit`。 |
| **末根 bar** | 末根 bar 的信号没有下一根可执行,等同于无操作。 |
| **未平仓** | 若最后仍持仓,该笔不计入 `trades`(`trade_count` 只统计已平仓)。 |

**Look-ahead 安全契约**:`generate_signals` 必须保证第 `i` 行的所有计算只依赖
`daily_df.iloc[:i+1]`。引擎再延迟一根 bar 才使用 `signal[t-1]`,二者组合即可保证
回测无未来数据泄露。

---

## 内置策略

`stockpool.backtesting.strategies`:

| 类 | 用途 |
|---|---|
| `CompositeVerdictStrategy` | 项目原综合评级策略(`detect_signals` + 日周共振 + `verdict_of`) |
| `VerdictExecution` | 仅执行规则,不重新生成信号;搭配 `run_on_signals` 使用预生成信号帧 |
| `SMACrossStrategy` | 经典 SMA 金叉/死叉,作为框架可扩展性的示范 |

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

---

## 未支持的能力

明确不支持的部分(避免使用时踩坑):

- **做空** — 引擎是 long-only。需要做空请等扩展或自行实现。
- **仓位管理** — 单仓位、满仓进出。无 Kelly / 固定金额 / 比例分仓。
- **多标的组合** — 一次只回测一只标的的曲线。
- **滑点模型** — 只有定额成本比例,无成交量相关滑点 / 部分成交 / 跳空缺口处理。
- **资金成本** — 杠杆 / 融券利息不建模。
- **盘中 / 多周期** — 只支持日线级,周/月需先用 `resample_to_weekly` 之类的工具
  转好再传入。

如果需要其中某项,先看 `Strategy` 能否通过 `should_exit` 的 `PositionContext`
组合出来(止损/止盈/动态出场都可以);需要改引擎才能支持的能力(如做空、
多标的)按需扩展。
