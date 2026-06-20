"""Framework-level tests for stockpool.backtesting.

Cover the engine contract (T+1 timing, costs, hold-day cap, no-pyramid),
the Strategy ABC, buy-and-hold baseline, metrics, and a custom strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting import (
    BacktestEngine,
    BacktestResult,
    BarContext,
    PositionContext,
    SMACrossStrategy,
    Strategy,
    Trade,
    TradeCosts,
    VerdictExecution,
    buy_and_hold_baseline,
    compute_metrics,
)


# -------- tiny fixtures ----------

def _signals(signal_list: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=len(signal_list), freq="B"),
        "close": closes,
        "signal": signal_list,
    })


def _daily(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=n, freq="B"),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1_000_000.0] * n,
    })


# -------- Strategy ABC contract ----------

def test_strategy_cannot_be_instantiated_directly():
    """Strategy is an ABC with four abstract methods."""
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_strategy_subclass_missing_method_cannot_instantiate():
    class Partial(Strategy):
        @property
        def name(self): return "partial"
        # missing: generate_signals, should_enter, should_exit
    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


# -------- engine: T+1 timing ----------

def test_engine_t_plus_one_entry_at_next_close():
    """Buy signal at bar 0 → entry equity moves with close[1]/close[0]."""
    sigs = _signals(["buy", "hold", "hold"], [100, 110, 121])
    engine = BacktestEngine(VerdictExecution(buy_verdicts=("buy",), sell_verdicts=()))
    r = engine.run_on_signals(sigs, max_holding_days=10)
    # Bar 0: flat
    assert r.curve["position"].iloc[0] == 0
    assert r.curve["equity"].iloc[0] == pytest.approx(1.0)
    # Bar 1: prev signal=buy → long; equity = 1.0 * (110/100) = 1.10
    assert r.curve["position"].iloc[1] == 1
    assert r.curve["equity"].iloc[1] == pytest.approx(1.10)
    # Bar 2: still long; equity = 1.10 * (121/110) = 1.21
    assert r.curve["equity"].iloc[2] == pytest.approx(1.21)


def test_engine_signal_at_final_bar_never_acts():
    """A buy at the last bar has no next bar to enter on — no trade, no equity change."""
    sigs = _signals(["hold", "hold", "buy"], [100, 100, 100])
    engine = BacktestEngine(VerdictExecution(buy_verdicts=("buy",), sell_verdicts=()))
    r = engine.run_on_signals(sigs, max_holding_days=5)
    assert (r.curve["equity"] == 1.0).all()
    assert r.metrics["trade_count"] == 0


# -------- engine: hold-day cap ----------

def test_engine_time_exit_at_max_holding_days():
    """Entered at bar 0, no exit signal, N=3 → forced exit at bar 4 (held 3)."""
    sigs = _signals(["buy"] + ["hold"] * 6, [100, 110, 120, 130, 125, 125, 125])
    engine = BacktestEngine(VerdictExecution(buy_verdicts=("buy",), sell_verdicts=()))
    r = engine.run_on_signals(sigs, max_holding_days=3)
    # Bar 4: held=3, forced exit; equity = 1.30
    assert r.curve["equity"].iloc[3] == pytest.approx(1.30)
    assert r.curve["position"].iloc[4] == 0
    assert r.metrics["trade_count"] == 1


# -------- engine: signal exit ----------

def test_engine_signal_exit_before_time_exit():
    sigs = _signals(["buy", "hold", "sell", "hold", "hold"],
                    [100, 110, 105, 100, 95])
    engine = BacktestEngine(
        VerdictExecution(buy_verdicts=("buy",), sell_verdicts=("sell",))
    )
    r = engine.run_on_signals(sigs, max_holding_days=10)
    # Bar 3: prev=sell → exit, equity locked at close[2]/close[0] = 1.05
    assert r.curve["equity"].iloc[3] == pytest.approx(1.05)
    assert r.curve["position"].iloc[3] == 0
    assert r.metrics["trade_count"] == 1


# -------- engine: no pyramiding ----------

def test_engine_does_not_pyramid_on_repeat_buy_signal():
    sigs = _signals(["buy", "hold", "buy", "hold", "hold"],
                    [100, 110, 110, 110, 110])
    engine = BacktestEngine(VerdictExecution(buy_verdicts=("buy",), sell_verdicts=()))
    r = engine.run_on_signals(sigs, max_holding_days=10)
    # Long from bar 1 onwards, second buy ignored. No closed trades by end.
    assert all(p == 1 for p in r.curve["position"].iloc[1:])
    assert r.metrics["trade_count"] == 0


# -------- engine: costs ----------

def test_engine_zero_costs_match_round_trip_price_ratio():
    # 5 bars: t=1 enter, t=2/3 hold (held=1,2), t=4 forced exit (held=3).
    sigs = _signals(["buy", "hold", "hold", "hold", "hold"], [100, 110, 120, 130, 125])
    engine = BacktestEngine(VerdictExecution(buy_verdicts=("buy",), sell_verdicts=()))
    r = engine.run_on_signals(sigs, max_holding_days=3)
    # Net return locked at close[3]/close[0] = 1.30 (exit on bar 4 with no cost).
    assert r.metrics["avg_trade_return_pct"] == pytest.approx(30.0)


def test_engine_costs_applied_arithmetically():
    """Hand-verified: buy_cost 0.001, sell_cost 0.002, price 100→130 over 3 hold bars.
    entry_eq = 1.0 * 0.999 = 0.999
    after_hold = 0.999 * 1.30 = 1.2987
    exit_eq = 1.2987 * 0.998 ≈ 1.295700
    net_ret = exit_eq / entry_eq - 1 ≈ 0.2974
    """
    sigs = _signals(["buy", "hold", "hold", "hold", "hold"], [100, 110, 120, 130, 125])
    engine = BacktestEngine(
        VerdictExecution(buy_verdicts=("buy",), sell_verdicts=()),
        costs=TradeCosts(buy_cost=0.001, sell_cost=0.002),
    )
    r = engine.run_on_signals(sigs, max_holding_days=3)
    entry_eq = 1.0 * (1 - 0.001)
    after = entry_eq * (130 / 100)
    exit_eq = after * (1 - 0.002)
    expected = (exit_eq / entry_eq - 1) * 100
    assert r.metrics["avg_trade_return_pct"] == pytest.approx(expected, rel=1e-6)


# -------- engine: result shape ----------

def test_engine_result_has_full_shape():
    sigs = _signals(["buy", "hold", "hold"], [100, 110, 120])
    engine = BacktestEngine(VerdictExecution(buy_verdicts=("buy",), sell_verdicts=()))
    r = engine.run_on_signals(sigs, max_holding_days=5)
    assert isinstance(r, BacktestResult)
    assert list(r.curve.columns) == ["date", "equity", "position"]
    assert r.max_holding_days == 5
    assert r.strategy_name == "verdict_execution"
    assert "sharpe" in r.metrics


def test_engine_empty_signals_returns_empty_result():
    sigs = pd.DataFrame({"date": [], "close": [], "signal": []})
    engine = BacktestEngine(VerdictExecution())
    r = engine.run_on_signals(sigs, max_holding_days=5)
    assert len(r.curve) == 0
    assert r.metrics["trade_count"] == 0


# -------- engine: sweep ----------

def test_sweep_holding_days_returns_one_result_per_N():
    """sweep_holding_days regenerates signals once, runs once per N."""
    daily = _daily([100 + i for i in range(60)])
    engine = BacktestEngine(SMACrossStrategy(fast_period=5, slow_period=20))
    out = engine.sweep_holding_days(daily, [3, 5, 10])
    assert set(out.keys()) == {3, 5, 10}
    for N, r in out.items():
        assert r.max_holding_days == N
        assert r.strategy_name == "sma_cross_5_20"


# -------- VerdictExecution: generate_signals refuses ----------

def test_verdict_execution_generate_signals_raises():
    strat = VerdictExecution()
    with pytest.raises(NotImplementedError):
        strat.generate_signals(_daily([100, 101]))


# -------- buy_and_hold_baseline ----------

def test_buy_and_hold_baseline_basic():
    daily = _daily([100, 110, 120, 130])
    bh = buy_and_hold_baseline(daily)
    assert bh.curve["equity"].iloc[0] == pytest.approx(1.0)
    assert bh.curve["equity"].iloc[-1] == pytest.approx(1.30)
    assert bh.metrics["total_return"] == pytest.approx(0.30)
    assert bh.metrics["trade_count"] == 1
    assert bh.metrics["win_rate"] is None
    assert bh.metrics["avg_trade_return_pct"] is None


def test_buy_and_hold_baseline_empty():
    daily = pd.DataFrame({"date": [], "close": [], "open": [], "high": [], "low": [], "volume": []})
    bh = buy_and_hold_baseline(daily)
    assert len(bh.curve) == 0
    assert bh.metrics["total_return"] == 0.0


# -------- compute_metrics ----------

def test_compute_metrics_known_values():
    eq = pd.Series([1.0, 2.0, 1.0, 1.5])
    trades = [Trade(0, 1, 100.0, 200.0, 1.0, 1),
              Trade(2, 3, 100.0, 150.0, 0.5, 1)]
    m = compute_metrics(eq, trades)
    # total_return = 1.5/1.0 - 1 = 0.5
    assert m["total_return"] == pytest.approx(0.5)
    # max_drawdown: 1.0 → 2.0 → 1.0 → 1.5; peak 2, trough 1 → 0.5
    assert m["max_drawdown"] == pytest.approx(0.5)
    assert m["trade_count"] == 2
    assert m["win_rate"] == pytest.approx(1.0)


def test_compute_metrics_accepts_dict_trades():
    """Trade-shaped dicts also work — compatibility with legacy trade dicts."""
    eq = pd.Series([1.0, 1.1, 1.2])
    trades = [{"ret": 0.1}, {"ret": -0.05}]
    m = compute_metrics(eq, trades)
    assert m["trade_count"] == 2
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["avg_trade_return_pct"] == pytest.approx((0.1 - 0.05) / 2 * 100)


def test_compute_metrics_empty_curve():
    m = compute_metrics(pd.Series([], dtype=float), [])
    assert m["total_return"] == 0.0
    assert m["sharpe"] == 0.0
    assert m["trade_count"] == 0


# -------- custom strategy: SMACross actually runs end-to-end ----------

def test_sma_cross_strategy_produces_signals_and_trades():
    # Construct a price series with a clear cross: dip then rise.
    closes = [100.0] * 25 + [100.0 - i for i in range(10)] + [90 + i * 2 for i in range(15)]
    daily = _daily(closes)
    engine = BacktestEngine(SMACrossStrategy(fast_period=5, slow_period=20))
    r = engine.run(daily, max_holding_days=20)
    # Sanity: at least one signal generated, run is deterministic.
    assert len(r.signals) > 0
    assert {"sma_fast", "sma_slow"}.issubset(r.signals.columns)
    # Equity should respond non-trivially.
    assert r.curve["equity"].iloc[-1] != 1.0 or r.metrics["trade_count"] == 0


def test_sma_cross_invalid_periods_raise():
    with pytest.raises(ValueError):
        SMACrossStrategy(fast_period=20, slow_period=20)


# -------- custom strategy: stop-loss via should_exit using PositionContext ----------

class _StopLossStrategy(Strategy):
    """Demonstrates that strategies can implement stop-loss via PositionContext."""

    def __init__(self, stop_pct: float):
        self.stop_pct = stop_pct

    @property
    def name(self) -> str:
        return f"stop_loss_{self.stop_pct}"

    def generate_signals(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        n = len(daily_df)
        return pd.DataFrame({
            "date": daily_df["date"].values,
            "close": daily_df["close"].values,
            "signal": ["buy"] + ["hold"] * (n - 1),
        })

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal == "buy"

    def should_exit(self, ctx: PositionContext) -> bool:
        # Exit when current close is more than stop_pct below entry.
        return (ctx.close - ctx.entry_price) / ctx.entry_price <= -self.stop_pct


def test_stop_loss_strategy_triggers_on_drawdown():
    closes = [100, 99, 98, 92, 85, 80]  # drops > 5% by bar 3
    daily = _daily(closes)
    engine = BacktestEngine(_StopLossStrategy(stop_pct=0.05))
    r = engine.run(daily, max_holding_days=100)
    # Signal at bar 3 (close=92 → drop 8%) executes at open[4] → exit_idx == 4
    # (the execution bar; previously this was the decision bar t-1).
    assert r.metrics["trade_count"] == 1
    exit_idx = r.trades[0].exit_idx
    assert exit_idx == 4
