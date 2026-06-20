"""Strategy-agnostic backtest engine.

A ``Strategy`` produces a per-bar signal frame plus entry/exit rules; a
``BacktestEngine`` consumes the strategy and a daily OHLCV history and returns
a ``BacktestResult``. The engine is long-only, single-position, T+1-compliant,
and supports configurable round-trip transaction costs.

See ``docs/backtesting_framework.md`` for the full API guide.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from stockpool.backtesting.sizing import LotSizer

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
    lot_size: float = 0.1


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

    Optional override (non-abstract, defaults to ``False``):

      * ``should_reset_timer(ctx) -> bool`` — when long, return ``True`` to
        refresh ``days_held`` to ``0`` for this bar instead of evaluating
        exits. Useful for "strong re-buy resets the N-day timer" semantics.
        If both ``should_reset_timer`` and ``should_exit`` would fire on the
        same bar, **reset wins** (the position is renewed, not closed).

    Look-ahead safety contract: signal row ``i`` may only depend on data
    available at bar ``i`` (i.e. ``daily_df.iloc[:i+1]``). The engine separately
    delays use of signal[t-1] until bar ``t`` (filling at ``open[t]``), so a
    correct walk-forward generator paired with the engine is fully T+1-compliant.

    Output schema: ``generate_signals`` must return a DataFrame with at least
    ``date``, ``open``, ``close``, ``signal``. The ``open`` column is read by
    the engine as the next-bar fill price; pass it through from the source
    OHLCV (or omit it to fall back to ``open[t] = close[t-1]``).
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

    def should_reset_timer(self, ctx: PositionContext) -> bool:
        """Optional hook: refresh ``days_held`` to 0 for the current bar.

        Default: never reset. Override to extend a hold when a re-entry
        signal appears (e.g. ``strong_buy`` while already long).
        """
        return False

    def predict_latest(self, daily_df: pd.DataFrame) -> dict:
        """Return the signal (+ extras) for the most recent bar only.

        Used by the daily-report path, which needs today's verdict without
        the cost of a full walk-forward. Default: run ``generate_signals``
        and take the last row. Subclasses may override for efficiency or to
        add caching (e.g. monthly model refit for ML strategies).

        Returns a dict containing at least ``'signal'``; may include
        ``'final_score'``, ``'score'``, etc. Returns ``{'signal': 'neutral'}``
        when no signal can be produced.
        """
        sig = self.generate_signals(daily_df)
        if len(sig) == 0:
            return {"signal": "neutral"}
        return dict(sig.iloc[-1])


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
      * Decisions are made on the signal at end-of-bar ``t-1`` and realised at
        ``open[t]`` — the next trading day's open price (T+1 compliant). On
        Chinese A-shares the open is the call-auction price, which models a
        realistic next-bar fill; the only fills you'd miss in practice are
        signals followed by a limit-up open.
      * Entry-bar exposure: equity rides ``open[t] → close[t]`` after buy_cost
        is deducted. Subsequent in-position bars use ``close[t-1] → close[t]``.
      * Exit-bar exposure: equity rides ``close[t-1] → open[t]``, sell_cost is
        applied, then the position is flat for the rest of the day.
      * ``Trade.entry_idx`` / ``exit_idx`` point at the *execution* bar (``t``),
        not the decision bar (``t-1``). ``Trade.entry_price`` / ``exit_price``
        are the ``open[t]`` values used as fills.
      * The engine reads ``open`` from the signal frame. If the frame omits
        the column, the engine falls back to ``open[t] = close[t-1]`` — which
        reproduces the legacy close-to-close arithmetic.
      * No pyramiding: a fresh enter signal while already long is ignored.
      * Buy-and-hold baseline (``buy_and_hold_baseline``) applies no costs and
        anchors on ``open[0]`` — it is the un-friction-ed reference.
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


def _opens_with_fallback(signals: pd.DataFrame) -> pd.Series:
    """Return the ``open`` column, or synthesize one as ``close.shift(1)``.

    The engine fills at next-day open. When a caller's signal frame lacks an
    ``open`` column (typical for hand-built test fixtures), we fall back to
    "no overnight gap": ``open[t] = close[t-1]``, with ``open[0] = close[0]``.
    Under this fallback the new open-based math reproduces the legacy
    close-to-close behaviour exactly.
    """
    closes = signals["close"]
    if "open" in signals.columns:
        opens = signals["open"].astype(float)
    else:
        opens = closes.shift(1)
        if len(opens) > 0:
            opens.iloc[0] = closes.iloc[0]
    return opens.reset_index(drop=True).values


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
    opens = _opens_with_fallback(signals)
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
        open_t = float(opens[t])
        close_t = float(closes[t])
        hold_ret = close_t / prev_close - 1  # close-to-close, for in-position bars

        if position[t - 1] == 0:
            ctx = BarContext(
                bar_idx=t - 1, date=prev_date,
                close=prev_close, signal=prev_signal,
            )
            if strategy.should_enter(ctx):
                # Fill at open[t]; exposure runs open[t] → close[t] this bar.
                position[t] = 1
                entry_idx = t
                entry_price = open_t
                days_held = 0
                entry_equity = equity[t - 1] * (1 - costs.buy_cost)
                equity[t] = entry_equity * (close_t / open_t)
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
            if strategy.should_reset_timer(pctx):
                # Refresh the N-day clock; skip exit checks this bar.
                position[t] = 1
                days_held = 0
                equity[t] = equity[t - 1] * (1 + hold_ret)
                continue
            time_exit = held_now >= max_holding_days
            if time_exit or strategy.should_exit(pctx):
                # Realize at open[t]: ride close[t-1] → open[t], pay sell_cost,
                # then flat for the rest of the day.
                position[t] = 0
                equity_at_open = equity[t - 1] * (open_t / prev_close)
                exit_equity = equity_at_open * (1 - costs.sell_cost)
                equity[t] = exit_equity
                trades.append(Trade(
                    entry_idx=entry_idx,
                    exit_idx=t,
                    entry_price=entry_price,
                    exit_price=open_t,
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
                equity[t] = equity[t - 1] * (1 + hold_ret)

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


@dataclass
class _OpenLot:
    """Internal: one open lot in the multi-lot engine."""
    entry_idx: int
    entry_price: float
    committed_cash: float    # cash actually invested, AFTER buy_cost
    current_value: float     # mark-to-market value of this lot
    days_held: int = 0
    lot_size: float = 0.1


class MultiLotBacktestEngine:
    """Multi-lot engine: each enter signal opens an independent lot.

    Differences from ``BacktestEngine``:

      * Multiple positions can be open concurrently. Each enter signal
        commits a lot whose size is determined by the injected ``LotSizer``
        (default: ``FixedLotSizer(0.1)``). Pass ``lot_sizer=`` for dynamic
        sizing (e.g. vol-target) or the deprecated ``position_size=`` for
        backwards compatibility — both, however, raises ``ValueError``.
      * Each lot has its own ``days_held`` timer. A lot exits when its own
        timer hits ``max_holding_days`` OR ``strategy.should_exit`` returns
        True for it.
      * Trade returns are per-lot (each closed lot ⇒ one ``Trade``).
        ``Trade.lot_size`` records the size of that specific lot, enabling
        per-trade attribution in downstream A/B reports.
      * ``curve["position"]`` becomes the *count of open lots* at each bar.

    Capital model:

      * Total equity starts at 1.0; lot sizes are fractions of that starting
        capital (e.g. ``0.1`` = lot is 10% of original equity).
      * "Cash" is the un-invested portion. Each buy deducts the sizer-returned
        size from cash and creates a lot worth ``size * (1 - buy_cost)``.
      * If the sizer returns 0 (skip-fallback) OR ``cash < size`` when a buy
        signal arrives, that buy is skipped (no partial fill).
      * Buy-and-hold and ``compute_metrics`` semantics are unchanged.

    All other conventions match ``BacktestEngine`` (T+1, long-only, costs).
    """

    def __init__(
        self,
        strategy: Strategy,
        position_size: float | None = None,
        lot_sizer: "LotSizer | None" = None,
        costs: TradeCosts = TradeCosts(),
        risk_free_rate: float = 0.02,
        max_concurrent_lots: int | None = None,
    ):
        if lot_sizer is not None and position_size is not None:
            raise ValueError(
                "Pass either `lot_sizer` or `position_size`, not both. "
                "`position_size` is deprecated; prefer "
                "`lot_sizer=FixedLotSizer(size)`."
            )
        if lot_sizer is None:
            # Bare engine call (legacy) — wrap fixed size.
            size = position_size if position_size is not None else 0.1
            from stockpool.backtesting.sizing import FixedLotSizer
            lot_sizer = FixedLotSizer(size)
        self.strategy = strategy
        self.lot_sizer = lot_sizer
        self.costs = costs
        self.risk_free_rate = risk_free_rate
        self.max_concurrent_lots = max_concurrent_lots

    def run(self, daily_df: pd.DataFrame, max_holding_days: int) -> BacktestResult:
        signals = self.strategy.generate_signals(daily_df)
        return self.run_on_signals(signals, max_holding_days)

    def run_on_signals(
        self, signals: pd.DataFrame, max_holding_days: int,
    ) -> BacktestResult:
        return _simulate_multi_lot(
            signals,
            strategy=self.strategy,
            lot_sizer=self.lot_sizer,
            max_concurrent_lots=self.max_concurrent_lots,
            max_holding_days=max_holding_days,
            costs=self.costs,
            risk_free_rate=self.risk_free_rate,
        )

    def sweep_holding_days(
        self,
        daily_df: pd.DataFrame,
        holding_days_list: Sequence[int],
    ) -> dict[int, BacktestResult]:
        signals = self.strategy.generate_signals(daily_df)
        return {N: self.run_on_signals(signals, N) for N in holding_days_list}


def _simulate_multi_lot(
    signals: pd.DataFrame,
    *,
    strategy: Strategy,
    lot_sizer: "LotSizer",
    max_concurrent_lots: int | None,
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
    opens = _opens_with_fallback(signals)
    sig_values = signals["signal"].values

    cash = 1.0
    open_lots: list[_OpenLot] = []
    trades: list[Trade] = []
    equity = [1.0] * n
    position = [0] * n

    for t in range(1, n):
        prev_signal = sig_values[t - 1]
        prev_close = float(closes[t - 1])
        prev_date = pd.Timestamp(dates[t - 1])
        open_t = float(opens[t])
        close_t = float(closes[t])
        hold_ret = close_t / prev_close - 1  # close-to-close for surviving lots

        # 1. Age all open lots.
        for lot in open_lots:
            lot.days_held += 1

        # 2. Per-lot exit / reset decisions. current_value reflects close[t-1];
        #    exits realize at open[t], which is close[t-1] * (open_t/prev_close).
        still_open: list[_OpenLot] = []
        for lot in open_lots:
            pctx = PositionContext(
                bar_idx=t - 1, date=prev_date,
                close=prev_close, signal=prev_signal,
                entry_idx=lot.entry_idx, entry_price=lot.entry_price,
                days_held=lot.days_held, max_holding_days=max_holding_days,
            )
            if strategy.should_reset_timer(pctx):
                lot.days_held = 0
                still_open.append(lot)
                continue
            time_exit = lot.days_held >= max_holding_days
            if time_exit or strategy.should_exit(pctx):
                value_at_open = lot.current_value * (open_t / prev_close)
                exit_value = value_at_open * (1 - costs.sell_cost)
                cash += exit_value
                trades.append(Trade(
                    entry_idx=lot.entry_idx,
                    exit_idx=t,
                    entry_price=lot.entry_price,
                    exit_price=open_t,
                    ret=float(exit_value / lot.committed_cash - 1),
                    days_held=lot.days_held,
                    lot_size=lot.lot_size,
                ))
            else:
                still_open.append(lot)
        open_lots = still_open

        # 3. Mark surviving lots with today's close-to-close return.
        for lot in open_lots:
            lot.current_value *= (1 + hold_ret)

        # 4. Maybe open a new lot — fills at open[t]; first-day exposure is
        #    open[t] → close[t]. Lot size now comes from the sizer (which sees
        #    closes up to bar t-1, preserving look-ahead safety).
        bctx = BarContext(
            bar_idx=t - 1, date=prev_date,
            close=prev_close, signal=prev_signal,
        )
        capacity_ok = (
            max_concurrent_lots is None
            or len(open_lots) < max_concurrent_lots
        )
        if strategy.should_enter(bctx) and capacity_ok:
            size = lot_sizer(t, opens, closes)
            if size > 0 and cash >= size:
                cash -= size
                committed = size * (1 - costs.buy_cost)
                open_lots.append(_OpenLot(
                    entry_idx=t,
                    entry_price=open_t,
                    committed_cash=committed,
                    current_value=committed * (close_t / open_t),
                    days_held=0,
                    lot_size=size,
                ))

        equity[t] = cash + sum(lot.current_value for lot in open_lots)
        position[t] = len(open_lots)

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
    # Anchor on open[0] to match the engine's next-open fill convention.
    # Bar 0 itself is the entry day: equity[0] = close[0] / open[0].
    if "open" in daily_df.columns and pd.notna(daily_df["open"].iloc[0]):
        base = float(daily_df["open"].iloc[0])
    else:
        base = float(closes[0])
    eq = closes / base
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
