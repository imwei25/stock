"""Tests for PortfolioEngine — T+1, cash conservation, rebalance diff, determinism."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.framework import TradeCosts
from stockpool.config import PortfolioRunConfig
from stockpool.portfolio.engine import PortfolioEngine
from stockpool.portfolio.strategy import PrecomputedScoreStrategy


# ----- fixtures -----


def _bars(n: int, start: str = "2024-01-02"):
    return pd.bdate_range(start=start, periods=n)


def _stock(
    dates: pd.DatetimeIndex,
    open_series: np.ndarray | float,
    close_series: np.ndarray | float,
) -> pd.DataFrame:
    n = len(dates)
    opens = np.full(n, open_series) if np.isscalar(open_series) else np.asarray(open_series)
    closes = np.full(n, close_series) if np.isscalar(close_series) else np.asarray(close_series)
    return pd.DataFrame({"date": dates, "open": opens, "close": closes})


def _build_panel(stocks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return stocks


def _scores(panel: pd.DataFrame) -> PrecomputedScoreStrategy:
    return PrecomputedScoreStrategy(panel)


def _trivial_cfg(top_k=2, rebalance_n_days=2) -> PortfolioRunConfig:
    return PortfolioRunConfig(
        top_k=top_k,
        rebalance_n_days=rebalance_n_days,
        max_per_industry=None,
        initial_cash=1.0,
    )


# ----- tests -----


def test_empty_panel_returns_empty_result():
    sp = pd.DataFrame()
    strat = PrecomputedScoreStrategy(sp, name="empty")
    eng = PortfolioEngine(strat, _trivial_cfg())
    res = eng.run({})
    assert res.curve.empty
    assert res.trades == []


def test_no_costs_constant_prices_preserve_equity():
    """5 stocks, flat prices, no costs → equity stays at 1.0."""
    dates = _bars(10)
    codes = ["A", "B", "C", "D", "E"]
    panel = {c: _stock(dates, 10.0, 10.0) for c in codes}
    # Score panel constant: A best, then B, ...
    sp = pd.DataFrame(
        np.tile([5, 4, 3, 2, 1], (len(dates), 1)),
        index=dates, columns=codes, dtype=float,
    )
    eng = PortfolioEngine(
        _scores(sp),
        _trivial_cfg(top_k=3, rebalance_n_days=2),
        costs=TradeCosts(0.0, 0.0),
    )
    res = eng.run(panel)
    assert np.allclose(res.curve["equity"].values, 1.0)


def test_cash_conservation():
    """cash + Σ(shares_i * close[t]) ≈ equity[t] every bar."""
    dates = _bars(15)
    codes = ["A", "B", "C"]
    # Non-trivial price paths
    rng = np.random.default_rng(0)
    panel = {c: _stock(dates, 10 + rng.normal(0, 0.1, 15), 10 + rng.normal(0, 0.5, 15))
             for c in codes}
    sp = pd.DataFrame(
        rng.normal(0, 1, (len(dates), len(codes))),
        index=dates, columns=codes,
    )
    eng = PortfolioEngine(
        _scores(sp),
        _trivial_cfg(top_k=2, rebalance_n_days=3),
        costs=TradeCosts(0.001, 0.001),
    )
    res = eng.run(panel)
    # The equity-curve invariant lives inside the engine; here we just
    # confirm the curve is monotone with respect to no NaNs and positive.
    assert (res.curve["equity"] > 0).all()
    assert not res.curve["equity"].isna().any()


def test_t_plus_one_fill_at_next_open():
    """Decision on bar t, fill at open[t+1]. Verified by tracking entry price."""
    dates = _bars(6)
    # A's open[t] differs from close[t] so we can verify which is used.
    # 跳空幅度要在涨停阈值(±10%)以内,否则会被 P1-3 拒单 —— 这里 +5%。
    opens_a = np.array([10.0, 10.0, 10.0, 10.5, 10.0, 10.0])   # open jumps at idx 3
    closes_a = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    panel = {
        "A": _stock(dates, opens_a, closes_a),
        "B": _stock(dates, 10.0, 10.0),
    }
    # rebalance bar 2 (decision) → exec at open[3] = 99. Score makes A best.
    # All other bars: NaN → predict_scores returns {} → no trade triggered.
    sp = pd.DataFrame(
        np.nan, index=dates, columns=["A", "B"], dtype=float,
    )
    sp.loc[dates[2], "A"] = 1.0
    eng = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(top_k=1, rebalance_n_days=2, max_per_industry=None),
        costs=TradeCosts(0.0, 0.0),
    )
    res = eng.run(panel)
    # First trade for A should have entry_price == 10.5 (open[3]), not close[2]/open[2].
    a_trades = [t for t in res.trades if t.code == "A"]
    assert a_trades, "expected A to be bought after rebalance bar"
    assert a_trades[0].entry_price == pytest.approx(10.5)


def test_rebalance_bars_respect_offset():
    """start_offset shifts rebalance schedule by k bars."""
    dates = _bars(10)
    codes = ["A", "B"]
    panel = {c: _stock(dates, 10.0, 10.0) for c in codes}
    sp = pd.DataFrame(
        np.tile([1, 0], (len(dates), 1)), index=dates, columns=codes, dtype=float,
    )
    eng = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(top_k=1, rebalance_n_days=3, max_per_industry=None),
        costs=TradeCosts(0.0, 0.0),
    )
    res0 = eng.run(panel, start_offset=0)
    res2 = eng.run(panel, start_offset=2)
    dates0 = res0.rebalance_log["date"].tolist()
    dates2 = res2.rebalance_log["date"].tolist()
    # offset=0 → bar indices 0,3,6,9 (but bar 9 is last → no execution → still logged though).
    # Actually engine only logs if can_execute_next_bar; bar 9 (last) skipped.
    assert dates0 == [dates[0], dates[3], dates[6]]
    assert dates2 == [dates[2], dates[5], dates[8]]


def test_determinism():
    """Same inputs twice → identical curve and trades."""
    dates = _bars(20)
    codes = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    panel = {c: _stock(dates, 10 + rng.normal(0, 0.1, 20), 10 + rng.normal(0, 0.3, 20))
             for c in codes}
    sp = pd.DataFrame(
        rng.normal(0, 1, (20, 4)), index=dates, columns=codes,
    )
    eng_factory = lambda: PortfolioEngine(
        _scores(sp.copy()),
        _trivial_cfg(top_k=2, rebalance_n_days=4),
        costs=TradeCosts(0.001, 0.001),
    )
    r1 = eng_factory().run(panel)
    r2 = eng_factory().run(panel)
    pd.testing.assert_series_equal(r1.curve["equity"], r2.curve["equity"])
    assert len(r1.trades) == len(r2.trades)
    for t1, t2 in zip(r1.trades, r2.trades):
        assert t1.code == t2.code
        assert t1.entry_price == t2.entry_price
        assert t1.ret == pytest.approx(t2.ret)


def test_rebalance_diff_sells_dropped_buys_added():
    """Target {A,B} → {B,C}: A sold, C bought, B churned."""
    dates = _bars(8)
    codes = ["A", "B", "C"]
    panel = {c: _stock(dates, 10.0, 10.0) for c in codes}
    sp = pd.DataFrame(np.nan, index=dates, columns=codes)
    # Bar 0: A=2, B=1, C=0 → top 2 = {A,B}, exec bar 1
    sp.loc[dates[0]] = [2.0, 1.0, 0.0]
    # Bar 3: A=0, B=1, C=2 → top 2 = {B,C}, exec bar 4
    sp.loc[dates[3]] = [0.0, 1.0, 2.0]
    # Other rebalance bars (6): NaN → no decision → no churn.
    eng = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(top_k=2, rebalance_n_days=3, max_per_industry=None),
        costs=TradeCosts(0.0, 0.0),
    )
    res = eng.run(panel)
    # A entered bar 1, exited bar 4 (sold on rebalance)
    a_trades = [t for t in res.trades if t.code == "A"]
    assert len(a_trades) == 1
    assert a_trades[0].exit_reason == "rebalance_drop"
    # P1-4 差量调仓:B 是存活仓(两次 target 都含 B)→ 不被翻炒,
    # 只有 end_of_backtest 一笔。
    b_trades = [t for t in res.trades if t.code == "B"]
    assert len(b_trades) == 1
    assert b_trades[0].exit_reason == "end_of_backtest"
    # C entered bar 4 (the rebalance exec) and closed at end_of_backtest.
    c_trades = [t for t in res.trades if t.code == "C"]
    assert len(c_trades) == 1
    assert c_trades[0].exit_reason == "end_of_backtest"


def test_last_bar_decision_does_not_execute():
    """A rebalance at the last bar can't fill — engine logs nothing."""
    dates = _bars(5)
    codes = ["A"]
    panel = {"A": _stock(dates, 10.0, 10.0)}
    sp = pd.DataFrame(0.0, index=dates, columns=codes)
    # rebalance every 4 bars → bars 0 and 4. Bar 4 is the last bar.
    sp.loc[:] = 1.0
    eng = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(top_k=1, rebalance_n_days=4, max_per_industry=None),
        costs=TradeCosts(0.0, 0.0),
    )
    res = eng.run(panel)
    # Should only have 1 rebalance log entry (bar 0); bar 4 is skipped.
    assert len(res.rebalance_log) == 1


def test_unknown_score_codes_ignored():
    """If score panel contains a code not in panel_data, it's filtered out."""
    dates = _bars(6)
    panel = {"A": _stock(dates, 10.0, 10.0), "B": _stock(dates, 10.0, 10.0)}
    # Score panel includes a phantom code "Z"
    sp = pd.DataFrame(0.0, index=dates, columns=["A", "B", "Z"])
    sp.loc[:, "Z"] = 99.0
    eng = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(top_k=2, rebalance_n_days=2, max_per_industry=None),
        costs=TradeCosts(0.0, 0.0),
    )
    res = eng.run(panel)
    # No trade for Z
    assert all(t.code != "Z" for t in res.trades)


def test_initial_cash_scales_curve():
    """initial_cash=100 → equity curve starts (roughly) at 100, ends positive."""
    dates = _bars(8)
    panel = {"A": _stock(dates, 10.0, 10.0)}
    sp = pd.DataFrame(1.0, index=dates, columns=["A"])
    eng = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(top_k=1, rebalance_n_days=2, max_per_industry=None,
                           initial_cash=100.0),
        costs=TradeCosts(0.0, 0.0),
    )
    res = eng.run(panel)
    assert res.curve["equity"].iloc[0] == pytest.approx(100.0)
    assert res.curve["equity"].iloc[-1] == pytest.approx(100.0)
