"""Tests for MultiLotBacktestEngine — fixed-size, independent lots."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting import (
    BacktestResult,
    MultiLotBacktestEngine,
    TradeCosts,
    VerdictExecution,
)


def _signals(signal_list: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=len(signal_list), freq="B"),
        "close": closes,
        "signal": signal_list,
    })


def _engine(position_size: float = 0.1,
            buy_cost: float = 0.0, sell_cost: float = 0.0,
            max_concurrent_lots: int | None = None) -> MultiLotBacktestEngine:
    return MultiLotBacktestEngine(
        VerdictExecution(buy_verdicts=("buy", "strong_buy"),
                         sell_verdicts=("sell", "strong_sell")),
        position_size=position_size,
        costs=TradeCosts(buy_cost=buy_cost, sell_cost=sell_cost),
        max_concurrent_lots=max_concurrent_lots,
    )


# -------- validation ----------

def test_position_size_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        _engine(position_size=0.0)
    with pytest.raises(ValueError):
        _engine(position_size=1.5)


# -------- no-signal flat ----------

def test_no_buy_signal_flat_equity():
    sigs = _signals(["hold"] * 5, [100, 110, 120, 130, 140])
    r = _engine().run_on_signals(sigs, max_holding_days=5)
    assert (r.curve["equity"] == 1.0).all()
    assert (r.curve["position"] == 0).all()
    assert r.metrics["trade_count"] == 0


# -------- each buy opens a fresh lot ----------

def test_each_buy_opens_new_lot():
    """3 buys → 3 concurrent lots; curve['position'] counts open lots."""
    sigs = _signals(
        ["buy", "buy", "buy", "hold", "hold", "hold", "hold"],
        [100, 100, 100, 100, 100, 100, 100],
    )
    r = _engine(position_size=0.1).run_on_signals(sigs, max_holding_days=10)
    # After all 3 buys (bars 1, 2, 3 act on signals at 0, 1, 2):
    assert r.curve["position"].iloc[3] == 3
    # No price movement → equity stays at 1.0 (no fees in this test).
    assert r.curve["equity"].iloc[-1] == pytest.approx(1.0)


def test_strong_buy_after_buy_opens_separate_lot():
    """The case the user asked about: buy then strong_buy → 2 lots."""
    sigs = _signals(
        ["buy", "strong_buy", "hold", "hold", "hold", "hold"],
        [100, 100, 100, 100, 100, 100],
    )
    r = _engine(position_size=0.1).run_on_signals(sigs, max_holding_days=20)
    # After both buys executed:
    assert r.curve["position"].iloc[2] == 2


# -------- per-lot timer ----------

def test_per_lot_timeout_independent():
    """Two lots opened on different bars time out on different bars (N=3 each)."""
    # Day 0,1: buy signals → lots open on bar 1, bar 2.
    # Day 2+: hold. With N=3:
    #   lot A (opened bar 1, entry close[0]=100): held=1@t=2, =2@t=3, =3@t=4 → exits at t=4
    #   lot B (opened bar 2, entry close[1]=100): held=1@t=3, =2@t=4, =3@t=5 → exits at t=5
    sigs = _signals(["buy", "buy", "hold", "hold", "hold", "hold", "hold"],
                    [100, 100, 100, 100, 100, 100, 100])
    r = _engine(position_size=0.1).run_on_signals(sigs, max_holding_days=3)
    assert r.metrics["trade_count"] == 2
    # exit_idx records the execution bar. Lot A's signal lands at bar 3 →
    # exit at open[4]; lot B's signal at bar 4 → exit at open[5].
    exits = sorted(t.exit_idx for t in r.trades)
    assert exits == [4, 5]


# -------- sell signal closes everything still open ----------

def test_sell_signal_closes_all_open_lots():
    """3 buys, then a sell signal → all 3 lots close on the same bar."""
    sigs = _signals(
        ["buy", "buy", "buy", "sell", "hold", "hold"],
        [100, 100, 100, 100, 100, 100],
    )
    r = _engine(position_size=0.1).run_on_signals(sigs, max_holding_days=20)
    # 3 lots opened on bars 1,2,3 (signals at 0,1,2). Sell signal at bar 3 →
    # all exit at open[4]; exit_idx now records the execution bar (4).
    assert r.metrics["trade_count"] == 3
    exit_indices = set(t.exit_idx for t in r.trades)
    assert exit_indices == {4}
    # After all exits, position count back to 0.
    assert r.curve["position"].iloc[-1] == 0


# -------- cash constraint ----------

def test_skip_buy_when_cash_insufficient():
    """position_size=0.4, four buys in a row → only first three open
    (cash after 2 buys = 0.2, < 0.4, but actually after 2 → 0.2, then can't take more)."""
    # position_size = 0.4. Cash starts at 1.0.
    # Buy 1: cash → 0.6, lot 1 opens.
    # Buy 2: cash → 0.2, lot 2 opens.
    # Buy 3: cash 0.2 < 0.4 → skipped.
    sigs = _signals(["buy", "buy", "buy", "hold", "hold"], [100, 100, 100, 100, 100])
    r = _engine(position_size=0.4).run_on_signals(sigs, max_holding_days=20)
    # Only 2 lots open after the 3 buy signals.
    assert r.curve["position"].iloc[3] == 2


# -------- max_concurrent_lots cap ----------

def test_max_concurrent_lots_cap_overrides_cash():
    """Even with plenty of cash, max_concurrent_lots=2 stops a 3rd buy."""
    sigs = _signals(["buy", "buy", "buy", "hold", "hold"], [100, 100, 100, 100, 100])
    r = _engine(position_size=0.1, max_concurrent_lots=2).run_on_signals(
        sigs, max_holding_days=20
    )
    # Max 2 lots open at any time.
    assert r.curve["position"].max() == 2


# -------- per-lot returns are independent ----------

def test_per_lot_returns_independent():
    """Two lots opened at different prices → different individual returns."""
    # Lot A opens at close[0]=100, exits at close[3]=130 → +30% gross.
    # Lot B opens at close[1]=110, exits at close[4]=132 → +20% gross.
    # Both have N=3.
    sigs = _signals(
        ["buy", "buy", "hold", "hold", "hold", "hold", "hold"],
        [100, 110, 120, 130, 132, 132, 132],
    )
    r = _engine(position_size=0.1).run_on_signals(sigs, max_holding_days=3)
    assert r.metrics["trade_count"] == 2
    rets = sorted(t.ret for t in r.trades)
    # B's net return: lot enters at close[1]=110, hits N=3 at bar 5 (held: 1@t=3, 2@t=4, 3@t=5).
    # exit_price = close[4] = 132. ret ≈ 132/110 - 1 = 0.20.
    # A's net return: lot enters at close[0]=100, hits N=3 at bar 4. exit_price = close[3] = 130.
    # ret ≈ 130/100 - 1 = 0.30.
    assert rets[0] == pytest.approx(0.20, rel=1e-6)
    assert rets[1] == pytest.approx(0.30, rel=1e-6)


# -------- equity continuity / total accounting ----------

def test_equity_equals_cash_plus_marked_lots():
    """Total equity at each bar must equal cash + sum(open lots).
    Indirect check: when all lots flat-priced, equity stays at 1.0 minus
    accumulated fees only.
    """
    sigs = _signals(["buy"] * 5 + ["hold"] * 10, [100] * 15)
    r = _engine(position_size=0.1, buy_cost=0.001, sell_cost=0.002).run_on_signals(
        sigs, max_holding_days=10
    )
    # No price movement; after k buys we've paid k * 0.1 * 0.001 in fees.
    # After 5 buys: total equity = 1.0 - 5 * 0.0001 = 0.9995.
    assert r.curve["equity"].iloc[5] == pytest.approx(0.9995, rel=1e-6)


# -------- exit cost applied per lot ----------

def test_sell_cost_applied_per_lot():
    """One lot, +30% gross, sell_cost 0.002 ⇒ net ret = 0.30*1 - 0.002*(1.30) ish.
    Hand check:
      committed = 0.1 * (1 - 0.001) = 0.0999
      final_value = 0.0999 * 1.30 * (1 - 0.002) = 0.1295701...
      ret = final_value / 0.0999 - 1 = 0.2974...
    """
    sigs = _signals(["buy"] + ["hold"] * 4, [100, 110, 120, 130, 130])
    r = _engine(position_size=0.1, buy_cost=0.001, sell_cost=0.002).run_on_signals(
        sigs, max_holding_days=3
    )
    assert r.metrics["trade_count"] == 1
    expected_ret = (0.1 * (1 - 0.001) * 1.30 * (1 - 0.002)) / (0.1 * (1 - 0.001)) - 1
    assert r.trades[0].ret == pytest.approx(expected_ret, rel=1e-6)


# -------- result shape ----------

def test_result_shape_matches_single_position():
    """Same BacktestResult contract — strategies/reports can iterate uniformly."""
    sigs = _signals(["buy", "buy", "hold", "hold"], [100, 100, 100, 100])
    r = _engine().run_on_signals(sigs, max_holding_days=10)
    assert isinstance(r, BacktestResult)
    assert list(r.curve.columns) == ["date", "equity", "position"]
    assert "sharpe" in r.metrics


# -------- empty signals ----------

def test_empty_signals_returns_empty_result():
    sigs = pd.DataFrame({"date": [], "close": [], "signal": []})
    r = _engine().run_on_signals(sigs, max_holding_days=5)
    assert len(r.curve) == 0
    assert r.metrics["trade_count"] == 0


# ============================================================================
# lot_sizer injection (PR-C)
# ============================================================================

from stockpool.backtesting.sizing import FixedLotSizer, VolTargetLotSizer


def test_engine_accepts_lot_sizer_kwarg():
    """Constructing with lot_sizer= works and overrides default sizing."""
    engine = MultiLotBacktestEngine(
        VerdictExecution(),
        lot_sizer=FixedLotSizer(0.25),
    )
    assert engine.lot_sizer.size == 0.25


def test_engine_default_when_neither_provided():
    """No lot_sizer, no position_size → default FixedLotSizer(0.1)."""
    engine = MultiLotBacktestEngine(VerdictExecution())
    assert isinstance(engine.lot_sizer, FixedLotSizer)
    assert engine.lot_sizer.size == 0.1


def test_engine_rejects_both_position_size_and_lot_sizer():
    with pytest.raises(ValueError, match="Pass either"):
        MultiLotBacktestEngine(
            VerdictExecution(),
            position_size=0.1,
            lot_sizer=FixedLotSizer(0.2),
        )


def test_engine_position_size_keyword_still_works():
    """Backwards compatibility: position_size= alone wraps in FixedLotSizer."""
    engine = MultiLotBacktestEngine(VerdictExecution(), position_size=0.15)
    assert isinstance(engine.lot_sizer, FixedLotSizer)
    assert engine.lot_sizer.size == 0.15


def test_trade_lot_size_recorded():
    """Trade.lot_size is populated from the active sizer at entry."""
    sigs = _signals(
        ["buy", "hold", "hold", "hold", "hold"],
        [100, 100, 100, 100, 100],
    )
    engine = MultiLotBacktestEngine(
        VerdictExecution(), lot_sizer=FixedLotSizer(0.07),
    )
    r = engine.run_on_signals(sigs, max_holding_days=3)
    assert len(r.trades) == 1
    assert r.trades[0].lot_size == pytest.approx(0.07)


def test_vol_target_dynamic_sizing_records_per_trade_size():
    """With vol_target, each trade's lot_size reflects vol at that bar."""
    # 30 bars of mild noise + 1 buy signal late enough for vol calc to kick in.
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0, 0.01, size=30)
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * (1 + r))
    sigs = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=31, freq="B"),
        "close": closes,
        "signal": ["hold"] * 25 + ["buy"] + ["hold"] * 5,
    })
    sizer = VolTargetLotSizer(
        baseline_size=0.1, reference_vol_annual=0.30,
        vol_window=20, min_size=0.03, max_size=0.20, fallback="fixed",
    )
    engine = MultiLotBacktestEngine(VerdictExecution(), lot_sizer=sizer)
    r = engine.run_on_signals(sigs, max_holding_days=3)
    assert len(r.trades) == 1
    # daily vol ~1%, annual ~16%; raw size = 0.1 * 0.30 / 0.16 ≈ 0.19 → near max
    assert 0.05 < r.trades[0].lot_size <= 0.20


def test_skip_fallback_zero_size_skips_buy():
    """When sizer returns 0 (skip fallback during cold-start), no lot opens."""
    sigs = _signals(
        ["buy", "hold", "hold"],
        [100, 100, 100],
    )
    sizer = VolTargetLotSizer(
        baseline_size=0.1, reference_vol_annual=0.30,
        vol_window=20, min_size=0.03, max_size=0.20, fallback="skip",
    )
    engine = MultiLotBacktestEngine(VerdictExecution(), lot_sizer=sizer)
    r = engine.run_on_signals(sigs, max_holding_days=3)
    assert r.metrics["trade_count"] == 0
    assert (r.curve["position"] == 0).all()


def test_size_exceeds_cash_skips_buy():
    """Sizer returns size > available cash → buy skipped (no partial fill)."""
    sigs = _signals(
        ["buy", "buy", "buy", "hold"],
        [100, 100, 100, 100],
    )
    # Each lot = 0.5 → first two consume all cash, third must skip.
    engine = MultiLotBacktestEngine(
        VerdictExecution(), lot_sizer=FixedLotSizer(0.5),
    )
    r = engine.run_on_signals(sigs, max_holding_days=10)
    # Only 2 lots opened (cash=1.0 → 0.5 → 0.0 < 0.5)
    assert r.curve["position"].iloc[3] == 2
