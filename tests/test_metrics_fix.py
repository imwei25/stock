"""Tests for trade-stat / metrics 口径 fixes (P2-8, P3-15, P3-16, P2-12).

Covers:
  * Trade.ret net of BOTH buy_cost and sell_cost (denominator = pre-buy cash),
    single-position and multi-lot engines.
  * compute_metrics edge guards: annualized_return None below 60 bars,
    sharpe None below 20 bars, win_rate/avg_trade_return_pct None with no trades.
  * buy_and_hold_baseline total_return anchored at open[0] (includes day-0
    intraday return), consistent with its curve.
  * Active-span metrics: compute_metrics(active_from_idx=...) and
    BacktestResult.metrics_active filled from the first trade's entry_idx.
  * backtest_report rendering tolerates None metrics and shows the
    active-span row when present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting import (
    BacktestEngine,
    MultiLotBacktestEngine,
    TradeCosts,
    VerdictExecution,
    buy_and_hold_baseline,
    compute_metrics,
)


def _signals(signal_list: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=len(signal_list), freq="B"),
        "close": closes,
        "signal": signal_list,
    })


def _engine(**costs) -> BacktestEngine:
    return BacktestEngine(
        VerdictExecution(buy_verdicts=("buy",), sell_verdicts=("sell",)),
        costs=TradeCosts(**costs),
    )


# ---------------------------------------------------------------- P2-8
# Trade.ret must be net of buy_cost AND sell_cost: denominator is the
# cash before the buy, not the post-buy-cost entry equity.

def test_single_engine_trade_ret_includes_buy_cost():
    """buy_cost 0.001 / sell_cost 0.002, 100 -> 130 over N=3.

    pre_buy_equity = 1.0
    exit_equity    = 1.0 * (1-0.001) * 1.30 * (1-0.002)
    ret            = exit_equity / 1.0 - 1   (NOT / (1*(1-0.001)))
    """
    sigs = _signals(["buy", "hold", "hold", "hold", "hold"],
                    [100, 110, 120, 130, 125])
    r = _engine(buy_cost=0.001, sell_cost=0.002).run_on_signals(
        sigs, max_holding_days=3)
    assert r.metrics["trade_count"] == 1
    exit_equity = 1.0 * (1 - 0.001) * 1.30 * (1 - 0.002)
    assert r.trades[0].ret == pytest.approx(exit_equity / 1.0 - 1, rel=1e-9)


def test_single_engine_equity_curve_unchanged_by_ret_fix():
    """The fix only touches Trade.ret — equity accounting must not move."""
    sigs = _signals(["buy", "hold", "hold", "hold", "hold"],
                    [100, 110, 120, 130, 125])
    r = _engine(buy_cost=0.001, sell_cost=0.002).run_on_signals(
        sigs, max_holding_days=3)
    exit_equity = 1.0 * (1 - 0.001) * 1.30 * (1 - 0.002)
    assert r.curve["equity"].iloc[-1] == pytest.approx(exit_equity, rel=1e-9)


def test_single_engine_trade_ret_zero_costs_is_price_ratio():
    sigs = _signals(["buy", "hold", "hold", "hold", "hold"],
                    [100, 110, 120, 130, 125])
    r = _engine().run_on_signals(sigs, max_holding_days=3)
    assert r.trades[0].ret == pytest.approx(0.30, rel=1e-9)


def test_multi_lot_trade_ret_includes_buy_cost():
    """Multi-lot: denominator is the lot's pre-cost order size, not committed."""
    sigs = _signals(["buy"] + ["hold"] * 4, [100, 110, 120, 130, 130])
    eng = MultiLotBacktestEngine(
        VerdictExecution(buy_verdicts=("buy",), sell_verdicts=("sell",)),
        position_size=0.1,
        costs=TradeCosts(buy_cost=0.001, sell_cost=0.002),
    )
    r = eng.run_on_signals(sigs, max_holding_days=3)
    assert r.metrics["trade_count"] == 1
    exit_value = 0.1 * (1 - 0.001) * 1.30 * (1 - 0.002)
    assert r.trades[0].ret == pytest.approx(exit_value / 0.1 - 1, rel=1e-9)


# ---------------------------------------------------------------- P3-15
# Metric edge guards.

def _rising_eq(n: int, total: float = 0.05) -> pd.Series:
    return pd.Series(np.linspace(1.0, 1.0 + total, n))


def test_annualized_return_none_below_60_days():
    m = compute_metrics(_rising_eq(10), [])
    assert m["annualized_return"] is None
    assert m["total_return"] == pytest.approx(0.05)


def test_annualized_return_computed_at_60_days():
    m = compute_metrics(_rising_eq(60), [])
    assert m["annualized_return"] is not None
    assert m["annualized_return"] > 0


def test_sharpe_none_below_20_days():
    m = compute_metrics(_rising_eq(19), [])
    assert m["sharpe"] is None


def test_sharpe_computed_at_20_days():
    eq = pd.Series(1.0 + 0.01 * np.arange(20) + 0.001 * np.sin(np.arange(20)))
    m = compute_metrics(eq, [])
    assert m["sharpe"] is not None


def test_no_trades_win_rate_and_avg_trade_none():
    m = compute_metrics(_rising_eq(60), [])
    assert m["trade_count"] == 0
    assert m["win_rate"] is None
    assert m["avg_trade_return_pct"] is None


def test_empty_curve_metrics_use_none_for_undefined():
    m = compute_metrics(pd.Series([], dtype=float), [])
    assert m["total_return"] == 0.0
    assert m["annualized_return"] is None
    assert m["sharpe"] is None
    assert m["win_rate"] is None
    assert m["avg_trade_return_pct"] is None


# ---------------------------------------------------------------- P3-16
# Buy & hold anchored at open[0] for BOTH curve and total_return.

def test_buy_and_hold_total_return_anchored_at_open0():
    daily = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=3, freq="B"),
        "open": [100.0, 111.0, 121.0],
        "close": [110.0, 120.0, 130.0],
    })
    bh = buy_and_hold_baseline(daily)
    # Curve anchored at open[0]: equity[0] = 110/100 = 1.10.
    assert bh.curve["equity"].iloc[0] == pytest.approx(1.10)
    assert bh.curve["equity"].iloc[-1] == pytest.approx(1.30)
    # total_return includes the day-0 intraday leg: 130/100 - 1, not 130/110 - 1.
    assert bh.metrics["total_return"] == pytest.approx(0.30, rel=1e-9)


def test_buy_and_hold_without_open_column_falls_back_to_close0():
    daily = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=4, freq="B"),
        "close": [100.0, 110.0, 120.0, 130.0],
    })
    bh = buy_and_hold_baseline(daily)
    assert bh.curve["equity"].iloc[0] == pytest.approx(1.0)
    assert bh.metrics["total_return"] == pytest.approx(0.30, rel=1e-9)


# ---------------------------------------------------------------- P2-12
# Active-span metrics.

def test_compute_metrics_active_from_idx_slices_equity():
    eq = np.concatenate([np.full(30, 1.0), np.linspace(2.0, 3.0, 70)])
    full = compute_metrics(pd.Series(eq), [])
    active = compute_metrics(pd.Series(eq), [], active_from_idx=30)
    assert full["total_return"] == pytest.approx(3.0 / 1.0 - 1)
    assert active["total_return"] == pytest.approx(3.0 / 2.0 - 1)


def test_compute_metrics_active_from_idx_none_is_full_span():
    eq = _rising_eq(80)
    assert compute_metrics(eq, [], active_from_idx=None) == compute_metrics(eq, [])


def test_engine_fills_metrics_active_from_first_trade_entry():
    """Cold-start head (60 flat 'hold' bars) must not dilute metrics_active."""
    n_head = 60
    closes = [100.0] * n_head + [100.0 + 0.5 * i for i in range(40)]
    sigs = _signals(["hold"] * (n_head - 1) + ["buy"] + ["hold"] * 40, closes)
    r = _engine().run_on_signals(sigs, max_holding_days=5)
    assert r.metrics["trade_count"] >= 1
    assert r.metrics_active is not None
    first_entry = min(t.entry_idx for t in r.trades)
    expected = compute_metrics(
        r.curve["equity"], r.trades,
        risk_free_rate=0.02, active_from_idx=first_entry,
    )
    assert r.metrics_active == expected


def test_engine_metrics_active_none_without_trades():
    sigs = _signals(["hold"] * 10, [100.0] * 10)
    r = _engine().run_on_signals(sigs, max_holding_days=5)
    assert r.metrics["trade_count"] == 0
    assert r.metrics_active is None


def test_multi_lot_engine_fills_metrics_active():
    sigs = _signals(["buy"] + ["hold"] * 6, [100, 110, 120, 130, 125, 125, 125])
    eng = MultiLotBacktestEngine(
        VerdictExecution(buy_verdicts=("buy",), sell_verdicts=("sell",)),
        position_size=0.1,
    )
    r = eng.run_on_signals(sigs, max_holding_days=3)
    assert r.metrics["trade_count"] == 1
    assert r.metrics_active is not None
    first_entry = min(t.entry_idx for t in r.trades)
    expected = compute_metrics(
        r.curve["equity"], r.trades,
        risk_free_rate=0.02, active_from_idx=first_entry,
    )
    assert r.metrics_active == expected


def test_buy_and_hold_metrics_active_is_none():
    daily = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=4, freq="B"),
        "close": [100.0, 110.0, 120.0, 130.0],
    })
    assert buy_and_hold_baseline(daily).metrics_active is None


# ---------------------------------------------------------------- report

def _equity_result(metrics_extra=None, metrics_active=None):
    from stockpool.backtest_composite import EquityResult
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    curve = pd.DataFrame({
        "date": dates,
        "equity": [1.0, 1.01, 1.02, 1.03, 1.04],
        "position": [1] * 5,
    })
    base = {
        "total_return": 0.04, "annualized_return": 0.05, "max_drawdown": 0.01,
        "sharpe": 1.0, "trade_count": 2, "win_rate": 0.5,
        "avg_trade_return_pct": 1.0,
    }
    if metrics_extra:
        base.update(metrics_extra)
    return EquityResult(
        curves={5: curve},
        metrics={5: base},
        buy_and_hold=None,
        buy_and_hold_metrics=None,
        metrics_active=metrics_active,
    )


def test_report_renders_none_metrics_as_dash(tmp_path):
    from stockpool.backtest_report import render_backtest_report
    res = _equity_result(metrics_extra={
        "annualized_return": None, "sharpe": None,
        "win_rate": None, "avg_trade_return_pct": None, "trade_count": 0,
    })
    out = render_backtest_report([("000001", "测试", res)],
                                 run_date="2026-06-11", output_dir=tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "—" in html  # None metrics render as em-dash, no TypeError


def test_report_shows_active_span_row(tmp_path):
    from stockpool.backtest_report import render_backtest_report
    active = {
        "total_return": 0.04, "annualized_return": 0.20, "max_drawdown": 0.01,
        "sharpe": 1.5, "trade_count": 2, "win_rate": 0.5,
        "avg_trade_return_pct": 1.0,
    }
    res = _equity_result(metrics_active={5: active})
    out = render_backtest_report([("000001", "测试", res)],
                                 run_date="2026-06-11", output_dir=tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "活跃段" in html


def test_report_without_active_metrics_backward_compatible(tmp_path):
    from stockpool.backtest_report import render_backtest_report
    res = _equity_result()  # metrics_active defaults to None
    out = render_backtest_report([("000001", "测试", res)],
                                 run_date="2026-06-11", output_dir=tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "N=5" in html
    assert "活跃段" not in html
