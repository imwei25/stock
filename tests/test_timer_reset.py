"""Tests for the should_reset_timer hook (strong-buy refresh semantics)."""
from __future__ import annotations

import pandas as pd
import pytest

from stockpool.backtesting import (
    BacktestEngine,
    BarContext,
    MultiLotBacktestEngine,
    PositionContext,
    Strategy,
    TradeCosts,
    VerdictExecution,
)


def _signals(signal_list: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=len(signal_list), freq="B"),
        "close": closes,
        "signal": signal_list,
    })


# -------- single-position engine ----------

def test_strong_buy_resets_timer_extends_hold_past_N():
    """N=3; buy at bar 0, strong_buy at bar 3 (where time_exit would fire).
    Without reset: exit at bar 4 with held=3.
    With reset: refresh at bar 4, then exit on bar 7 with held=3.
    """
    sigs = _signals(
        ["buy", "hold", "hold", "strong_buy", "hold", "hold", "hold", "hold"],
        [100] * 8,
    )
    engine = BacktestEngine(VerdictExecution())
    r = engine.run_on_signals(sigs, max_holding_days=3)
    assert r.metrics["trade_count"] == 1
    # Signal day = bar 6 → executes at open[7]; exit_idx records the
    # execution bar, so it's 7 (was 6 under the old close-fill semantics).
    assert r.trades[0].exit_idx == 7


def test_plain_buy_while_long_still_ignored_by_default():
    """Default refresh_verdicts=('strong_buy',) — plain buy must NOT refresh."""
    sigs = _signals(
        ["buy", "hold", "buy", "hold", "hold"],
        [100, 100, 100, 100, 100],
    )
    engine = BacktestEngine(VerdictExecution())
    r = engine.run_on_signals(sigs, max_holding_days=10)
    # Entry bar 1, exit on bar 4 (held=3 stops at N=10? No, N=10 so no exit yet).
    # With N=10, position should stay open through the end → no closed trade.
    assert r.metrics["trade_count"] == 0


def test_refresh_verdicts_configurable_to_include_plain_buy():
    """Pass refresh_verdicts=('buy','strong_buy') to refresh on either."""
    sigs = _signals(
        ["buy", "hold", "buy", "hold", "hold", "hold"],
        [100] * 6,
    )
    engine = BacktestEngine(VerdictExecution(refresh_verdicts=("buy", "strong_buy")))
    r = engine.run_on_signals(sigs, max_holding_days=3)
    # Without refresh: buy on bar 1 → exit on bar 4 (held=3).
    # With refresh on plain buy: bar 3 prev=buy → refresh; exit on bar 6 (held=3).
    # But we only have 6 bars. So no closed trade by end-of-data.
    assert r.metrics["trade_count"] == 0


def test_refresh_verdicts_empty_disables_reset():
    """refresh_verdicts=() opts out — same as the original 'ignore' behavior."""
    sigs = _signals(
        ["buy", "hold", "hold", "strong_buy", "hold", "hold", "hold"],
        [100] * 7,
    )
    engine = BacktestEngine(VerdictExecution(refresh_verdicts=()))
    r = engine.run_on_signals(sigs, max_holding_days=3)
    # No reset → signal at bar 3 (held=3) → executes at open[4]; exit_idx=4.
    assert r.metrics["trade_count"] == 1
    assert r.trades[0].exit_idx == 4


def test_reset_wins_over_time_exit_at_N():
    """When held_now == N and signal is in refresh_verdicts, reset trumps time_exit."""
    sigs = _signals(
        ["buy", "hold", "hold", "strong_buy", "hold", "hold", "hold"],
        [100] * 7,
    )
    engine = BacktestEngine(VerdictExecution())  # default refreshes on strong_buy
    r = engine.run_on_signals(sigs, max_holding_days=3)
    # Entry bar 1 (signal at 0). At bar 4 prev=strong_buy, held_now would be 3
    # (time_exit). Reset wins → days_held back to 0, hold continues.
    # Forced exit at bar 7 (held=3 after refresh). Only 7 bars → exit on bar 7?
    # bar 4: refresh, days_held=0
    # bar 5: held_now=1
    # bar 6: held_now=2
    # bar 7: out of data — final bar, no t+1. Position stays open by end-of-data.
    assert r.metrics["trade_count"] == 0
    # Position is still long at the last bar.
    assert r.curve["position"].iloc[-1] == 1


def test_reset_wins_over_should_exit():
    """Even if a signal is in both sell_verdicts and refresh_verdicts, reset wins."""
    # Construct an odd strategy that returns True for both should_exit and should_reset_timer.
    class _ConflictStrategy(Strategy):
        @property
        def name(self): return "conflict"
        def generate_signals(self, daily_df):
            n = len(daily_df)
            return pd.DataFrame({
                "date": daily_df["date"].values,
                "close": daily_df["close"].values,
                "signal": ["buy"] + ["mixed"] * (n - 1),
            })
        def should_enter(self, ctx): return ctx.signal == "buy"
        def should_exit(self, ctx): return ctx.signal == "mixed"
        def should_reset_timer(self, ctx): return ctx.signal == "mixed"

    sigs = _signals(["buy", "mixed", "mixed", "mixed"], [100] * 4)
    r = BacktestEngine(_ConflictStrategy()).run_on_signals(sigs, max_holding_days=10)
    # Reset wins → position stays open every bar; never exits.
    assert r.metrics["trade_count"] == 0
    assert r.curve["position"].iloc[-1] == 1


# -------- multi-lot engine ----------

def test_multi_lot_strong_buy_resets_existing_lots():
    """Open 1 lot at bar 1; strong_buy at bar 3 refreshes that lot (and opens a new one)."""
    sigs = _signals(
        ["buy", "hold", "strong_buy", "hold", "hold", "hold", "hold", "hold"],
        [100] * 8,
    )
    engine = MultiLotBacktestEngine(VerdictExecution(), position_size=0.1)
    r = engine.run_on_signals(sigs, max_holding_days=3)
    # Bar 1: lot A opens.
    # Bar 3: prev=strong_buy → existing lot A's timer refreshes; new lot B also opens
    #        (strong_buy is also in buy_verdicts).
    # Both lots then count from 0 at bar 3 (well, B starts at 0, A reset to 0).
    # Neither exits within 8 bars (need 3 more bars after refresh for A, exits at bar 6).
    # Wait: A refreshed at bar 3 → bar 4: held=1, bar 5: held=2, bar 6: held=3 → exit.
    # B opens at bar 3 → bar 4: held=1, bar 5: held=2, bar 6: held=3 → exit.
    # Both exit at bar 6.
    assert r.metrics["trade_count"] == 2
    # Both lots' time_exit signals land at bar 5 → execute at open[6].
    exit_indices = sorted(t.exit_idx for t in r.trades)
    assert exit_indices == [6, 6]


def test_multi_lot_reset_not_triggered_by_plain_buy():
    """Plain buy after a strong_buy entry → opens a new lot, but does NOT refresh existing lots."""
    sigs = _signals(
        ["strong_buy", "hold", "buy", "hold", "hold", "hold", "hold"],
        [100] * 7,
    )
    engine = MultiLotBacktestEngine(VerdictExecution(), position_size=0.1)
    r = engine.run_on_signals(sigs, max_holding_days=3)
    # Lot A opens bar 1 (entry close[0]). Timer ticks each bar; not refreshed by plain buy.
    # Lot B opens bar 3 (signal=buy at bar 2).
    # Lot A exits at bar 4 (held=3). Lot B exits at bar 6 (held=3).
    # Execution bars are decision_bar + 1.
    exit_indices = sorted(t.exit_idx for t in r.trades)
    assert exit_indices == [4, 6]


# -------- default strategy ABC behavior ----------

def test_should_reset_timer_default_returns_false():
    """A custom Strategy that doesn't override should_reset_timer must default to False."""
    class _Minimal(Strategy):
        @property
        def name(self): return "minimal"
        def generate_signals(self, daily_df):
            n = len(daily_df)
            return pd.DataFrame({
                "date": daily_df["date"].values,
                "close": daily_df["close"].values,
                "signal": ["buy"] + ["hold"] * (n - 1),
            })
        def should_enter(self, ctx): return ctx.signal == "buy"
        def should_exit(self, ctx): return False

    # Even though there's a "strong_buy"-like situation, the minimal strategy
    # doesn't override should_reset_timer → engine takes default (no reset).
    sigs = _signals(["buy", "hold", "hold", "hold"], [100] * 4)
    r = BacktestEngine(_Minimal()).run_on_signals(sigs, max_holding_days=2)
    # Entered bar 1, time_exit signal at bar 2 (held=2) → execute open[3].
    assert r.metrics["trade_count"] == 1
    assert r.trades[0].exit_idx == 3
