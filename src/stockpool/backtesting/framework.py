"""Strategy-agnostic backtest engine.

A ``Strategy`` produces a per-bar signal frame plus entry/exit rules; a
``BacktestEngine`` consumes the strategy and a daily OHLCV history and returns
a ``BacktestResult``. The engine is long-only, single-position, T+1-compliant,
and supports configurable round-trip transaction costs.

See ``docs/backtesting_framework.md`` for the full API guide.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from stockpool.backtesting.metrics import compute_metrics


@dataclass(frozen=True)
class TradeCosts:
    """Round-trip transaction costs as fractions of position value.

    Both fields are applied multiplicatively to equity:
        entry_equity = equity_before_buy * (1 - buy_cost)
        exit_equity  = equity_before_sell * (1 - sell_cost)

    Use values, not percentages: ``buy_cost=0.001`` = 0.1%.
    """
    buy_cost: float = 0.0
    sell_cost: float = 0.0


@dataclass(frozen=True)
class Trade:
    """One closed long position."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    ret: float            # net of buy_cost and sell_cost
    days_held: int


@dataclass(frozen=True)
class BarContext:
    """Read-only view passed to ``should_enter`` (flat position)."""
    bar_idx: int
    date: pd.Timestamp
    close: float
    signal: Any


@dataclass(frozen=True)
class PositionContext:
    """Read-only view passed to ``should_exit`` (long position open)."""
    bar_idx: int
    date: pd.Timestamp
    close: float
    signal: Any
    entry_idx: int
    entry_price: float
    days_held: int          # bars held so far (includes today)
    max_holding_days: int   # the engine's configured upper bound (``N``)


class Strategy(ABC):
    """A strategy = signal generator + execution rules.

    Subclasses must implement four members:

      * ``name`` — short identifier used in reports and logs.
      * ``generate_signals(daily_df) -> DataFrame`` — walk-forward, look-ahead-safe.
        Required columns on the returned frame: ``date``, ``close``, ``signal``.
        Any extra columns are preserved on the engine's output.
      * ``should_enter(ctx) -> bool`` — called once per bar while flat.
      * ``should_exit(ctx) -> bool`` — called once per bar while long. The engine
        separately enforces ``days_held >= max_holding_days``, so returning
        ``False`` here is always bounded by ``N``.

    Look-ahead safety contract: signal row ``i`` may only depend on data
    available at bar ``i`` (i.e. ``daily_df.iloc[:i+1]``). The engine separately
    delays use of signal[t-1] until bar ``t``, so a correct walk-forward
    generator paired with the engine is fully T+1-compliant.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signals(self, daily_df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def should_enter(self, ctx: BarContext) -> bool: ...

    @abstractmethod
    def should_exit(self, ctx: PositionContext) -> bool: ...


@dataclass
class BacktestResult:
    """One run of the engine on a (strategy, history, max_holding_days) triple."""
    signals: pd.DataFrame
    curve: pd.DataFrame              # columns: date, equity, position
    trades: list[Trade]
    metrics: dict
    max_holding_days: int
    strategy_name: str


class BacktestEngine:
    """Strategy-agnostic, long-only, single-position equity simulator.

    Conventions:
      * Decisions are made on signal at end-of-bar ``t-1`` and realised at
        ``close[t]`` (T+1 compliant; ``close[t]`` is never consulted at ``t``).
      * No pyramiding: a fresh enter signal while already long is ignored.
      * Entry cost is deducted from equity *before* the day's price return.
      * Exit cost is deducted on exit day; no new price exposure that day.
      * Buy-and-hold baseline (``buy_and_hold_baseline``) applies no costs —
        it is the un-friction-ed reference.
    """

    def __init__(
        self,
        strategy: Strategy,
        costs: TradeCosts = TradeCosts(),
        risk_free_rate: float = 0.02,
    ):
        self.strategy = strategy
        self.costs = costs
        self.risk_free_rate = risk_free_rate

    def run(self, daily_df: pd.DataFrame, max_holding_days: int) -> BacktestResult:
        """Generate signals and simulate in one call."""
        signals = self.strategy.generate_signals(daily_df)
        return self.run_on_signals(signals, max_holding_days)

    def run_on_signals(
        self, signals: pd.DataFrame, max_holding_days: int,
    ) -> BacktestResult:
        """Simulate against a pre-generated signal frame.

        Useful when the same signals feed multiple ``max_holding_days`` (see
        ``sweep_holding_days``) or when signals were produced by an external
        process (e.g. cached from a previous run).
        """
        return _simulate(
            signals,
            strategy=self.strategy,
            max_holding_days=max_holding_days,
            costs=self.costs,
            risk_free_rate=self.risk_free_rate,
        )

    def sweep_holding_days(
        self,
        daily_df: pd.DataFrame,
        holding_days_list: Sequence[int],
    ) -> dict[int, BacktestResult]:
        """Run once per ``N`` in ``holding_days_list``, sharing the signal frame."""
        signals = self.strategy.generate_signals(daily_df)
        return {N: self.run_on_signals(signals, N) for N in holding_days_list}


def _simulate(
    signals: pd.DataFrame,
    *,
    strategy: Strategy,
    max_holding_days: int,
    costs: TradeCosts,
    risk_free_rate: float,
) -> BacktestResult:
    n = len(signals)
    if n == 0:
        empty_curve = pd.DataFrame({"date": [], "equity": [], "position": []})
        return BacktestResult(
            signals=signals,
            curve=empty_curve,
            trades=[],
            metrics=compute_metrics(pd.Series([], dtype=float), [], risk_free_rate),
            max_holding_days=max_holding_days,
            strategy_name=strategy.name,
        )

    dates = signals["date"].values
    closes = signals["close"].values
    sig_values = signals["signal"].values

    position = [0] * n
    equity = [1.0] * n
    trades: list[Trade] = []

    entry_idx: int | None = None
    entry_price: float | None = None
    entry_equity: float | None = None
    days_held = 0

    for t in range(1, n):
        prev_signal = sig_values[t - 1]
        prev_close = float(closes[t - 1])
        prev_date = pd.Timestamp(dates[t - 1])
        daily_ret = closes[t] / closes[t - 1] - 1

        if position[t - 1] == 0:
            ctx = BarContext(
                bar_idx=t - 1, date=prev_date,
                close=prev_close, signal=prev_signal,
            )
            if strategy.should_enter(ctx):
                position[t] = 1
                entry_idx = t - 1
                entry_price = prev_close
                days_held = 0
                entry_equity = equity[t - 1] * (1 - costs.buy_cost)
                equity[t] = entry_equity * (1 + daily_ret)
            else:
                equity[t] = equity[t - 1]
        else:
            held_now = days_held + 1
            assert entry_idx is not None and entry_price is not None and entry_equity is not None
            pctx = PositionContext(
                bar_idx=t - 1, date=prev_date,
                close=prev_close, signal=prev_signal,
                entry_idx=entry_idx, entry_price=entry_price,
                days_held=held_now, max_holding_days=max_holding_days,
            )
            time_exit = held_now >= max_holding_days
            if time_exit or strategy.should_exit(pctx):
                position[t] = 0
                exit_equity = equity[t - 1] * (1 - costs.sell_cost)
                equity[t] = exit_equity
                trades.append(Trade(
                    entry_idx=entry_idx,
                    exit_idx=t - 1,
                    entry_price=entry_price,
                    exit_price=prev_close,
                    ret=float(exit_equity / entry_equity - 1),
                    days_held=held_now,
                ))
                entry_idx = None
                entry_price = None
                entry_equity = None
                days_held = 0
            else:
                position[t] = 1
                days_held = held_now
                equity[t] = equity[t - 1] * (1 + daily_ret)

    curve = pd.DataFrame({
        "date": dates,
        "equity": equity,
        "position": position,
    })
    metrics = compute_metrics(curve["equity"], trades, risk_free_rate=risk_free_rate)
    return BacktestResult(
        signals=signals,
        curve=curve,
        trades=trades,
        metrics=metrics,
        max_holding_days=max_holding_days,
        strategy_name=strategy.name,
    )


def buy_and_hold_baseline(
    daily_df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    label: str = "buy_and_hold",
) -> BacktestResult:
    """Long-from-bar-0 reference baseline.

    No costs, no entry/exit logic. The returned ``BacktestResult`` has the same
    shape as a strategy run so reports can iterate uniformly.

    Notes:
      * ``metrics["trade_count"]`` is forced to ``1`` (single round-trip).
      * ``metrics["win_rate"]`` and ``avg_trade_return_pct`` are ``None`` —
        win/loss is undefined for a never-closed buy-and-hold position.
    """
    if len(daily_df) == 0:
        empty_curve = pd.DataFrame({"date": [], "equity": [], "position": []})
        empty_signals = pd.DataFrame({"date": [], "close": [], "signal": []})
        return BacktestResult(
            signals=empty_signals, curve=empty_curve, trades=[],
            metrics=compute_metrics(pd.Series([], dtype=float), [], risk_free_rate),
            max_holding_days=0, strategy_name=label,
        )

    closes = daily_df["close"].values
    eq = closes / closes[0]
    curve = pd.DataFrame({
        "date": daily_df["date"].values,
        "equity": eq,
        "position": [1] * len(daily_df),
    })
    metrics = compute_metrics(curve["equity"], trades=[], risk_free_rate=risk_free_rate)
    metrics["trade_count"] = 1
    metrics["win_rate"] = None
    metrics["avg_trade_return_pct"] = None

    signals = pd.DataFrame({
        "date": daily_df["date"].values,
        "close": closes,
        "signal": [label] * len(daily_df),
    })
    return BacktestResult(
        signals=signals, curve=curve, trades=[],
        metrics=metrics, max_holding_days=0, strategy_name=label,
    )
