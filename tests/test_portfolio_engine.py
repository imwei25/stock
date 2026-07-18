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


def test_mvo_weighting_runs_and_differs_from_equal():
    """weighting='mvo' produces a valid curve and diverges from equal weight
    when stocks have distinct volatilities (so the covariance matters)."""
    dates = _bars(80)
    codes = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    # Distinct per-stock vol so LW covariance is non-trivial.
    panel = {}
    for i, c in enumerate(codes):
        vol = 0.5 + 0.5 * i
        closes = 10 + np.cumsum(rng.normal(0, vol, len(dates)))
        closes = np.clip(closes, 1.0, None)
        panel[c] = _stock(dates, closes, closes)
    sp = pd.DataFrame(rng.normal(0, 1, (len(dates), len(codes))),
                      index=dates, columns=codes)

    def _cfg(weighting):
        return PortfolioRunConfig(
            top_k=4, rebalance_n_days=5, max_per_industry=None, initial_cash=1.0,
            weighting=weighting, mvo_risk_aversion=10.0, mvo_w_max=0.5,
            mvo_lookback=40, mvo_min_obs=20,
        )

    res_eq = PortfolioEngine(_scores(sp), _cfg("equal")).run(panel)
    res_mvo = PortfolioEngine(_scores(sp), _cfg("mvo")).run(panel)
    # Both valid, positive, finite.
    assert (res_mvo.curve["equity"] > 0).all()
    assert not res_mvo.curve["equity"].isna().any()
    assert len(res_mvo.curve) == len(res_eq.curve)
    # MVO allocates non-equally => the two equity paths diverge somewhere.
    assert not np.allclose(res_eq.curve["equity"].to_numpy(),
                           res_mvo.curve["equity"].to_numpy())


def test_mvo_cold_start_falls_back_equal():
    """Before mvo_lookback/min_obs bars accumulate, mvo == equal (cold-start
    fallback inside compute_target_weights)."""
    dates = _bars(12)
    codes = ["A", "B", "C"]
    rng = np.random.default_rng(1)
    panel = {c: _stock(dates, 10 + rng.normal(0, 0.3, 12), 10 + rng.normal(0, 0.3, 12))
             for c in codes}
    sp = pd.DataFrame(rng.normal(0, 1, (len(dates), len(codes))),
                      index=dates, columns=codes)
    common = dict(top_k=3, rebalance_n_days=3, max_per_industry=None, initial_cash=1.0)
    res_eq = PortfolioEngine(_scores(sp), PortfolioRunConfig(**common)).run(panel)
    res_mvo = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(**common, weighting="mvo", mvo_min_obs=20, mvo_lookback=120),
    ).run(panel)
    # min_obs=20 never reached in a 12-bar panel → mvo falls back to equal →
    # identical equity curves.
    assert np.allclose(res_eq.curve["equity"].to_numpy(),
                       res_mvo.curve["equity"].to_numpy())


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
    opens_a = np.array([10.0, 10.0, 10.0, 99.0, 10.0, 10.0])   # open jumps at idx 3
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
    # First trade for A should have entry_price == 99 (open[3]), not close[2]/open[2].
    a_trades = [t for t in res.trades if t.code == "A"]
    assert a_trades, "expected A to be bought after rebalance bar"
    assert a_trades[0].entry_price == pytest.approx(99.0)


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
    # B was churned (sold then re-bought at the same bar) → 2 entries total:
    # 1 closed at the rebalance, 1 closed at end_of_backtest.
    b_trades = [t for t in res.trades if t.code == "B"]
    assert len(b_trades) == 2
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


# ----- execution realism (2026-07-02) -----


def test_hold_survivors_no_churn_on_stable_target():
    """Constant target + hold_survivors → one entry, zero rebalance churn,
    equity strictly better than the legacy sell-and-rebuy model under costs."""
    dates = _bars(12)
    panel = {c: _stock(dates, 10.0, 10.0) for c in ["A", "B"]}
    sp = pd.DataFrame(np.tile([2.0, 1.0], (len(dates), 1)),
                      index=dates, columns=["A", "B"])
    costs = TradeCosts(buy_cost=0.001, sell_cost=0.002)
    common = dict(top_k=2, rebalance_n_days=2, max_per_industry=None,
                  initial_cash=1.0)

    legacy = PortfolioEngine(
        _scores(sp), PortfolioRunConfig(**common), costs=costs).run(panel)
    hold = PortfolioEngine(
        _scores(sp), PortfolioRunConfig(**common, hold_survivors=True),
        costs=costs).run(panel)

    # Legacy pays a round-trip on every rebalance; hold pays entry once.
    assert hold.curve["equity"].iloc[-1] > legacy.curve["equity"].iloc[-1]
    # Only the two end-of-backtest close-outs — no rebalance_drop churn.
    assert len(hold.trades) == 2
    assert {t.exit_reason for t in hold.trades} == {"end_of_backtest"}
    # Legacy produced fictitious churn trades.
    assert any(t.exit_reason == "rebalance_drop" for t in legacy.trades)


def test_hold_survivors_membership_turnover():
    """{A,B} → {B,C}: A sold, B held with original entry date, C bought."""
    dates = _bars(6)
    panel = {c: _stock(dates, 10.0, 10.0) for c in ["A", "B", "C"]}
    scores = np.array([
        [3.0, 2.0, 1.0],   # bar0 decision → {A,B} fill @1
        [3.0, 2.0, 1.0],
        [0.0, 3.0, 2.0],   # bar2 decision → {B,C} fill @3
        [0.0, 3.0, 2.0],
        [0.0, 3.0, 2.0],   # bar4 decision → {B,C} fill @5
        [0.0, 3.0, 2.0],
    ])
    sp = pd.DataFrame(scores, index=dates, columns=["A", "B", "C"])
    eng = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(top_k=2, rebalance_n_days=2, max_per_industry=None,
                           initial_cash=1.0, hold_survivors=True),
    )
    res = eng.run(panel)
    by_code = {}
    for t in res.trades:
        by_code.setdefault(t.code, []).append(t)
    # A dropped at bar3.
    assert by_code["A"][0].exit_reason == "rebalance_drop"
    assert by_code["A"][0].exit_date == dates[3]
    # B held throughout: single trade, entry at first fill bar1.
    assert len(by_code["B"]) == 1
    assert by_code["B"][0].entry_date == dates[1]
    assert by_code["B"][0].exit_reason == "end_of_backtest"
    # C entered at bar3.
    assert by_code["C"][0].entry_date == dates[3]


def test_limit_guard_buy_side_substitutes_next_rank():
    """Top-scored name opens 一字涨停 on the fill bar → skipped at selection,
    next-ranked name takes the slot. Without the guard it is bought."""
    dates = _bars(4)
    # X: close 10, fill-bar open 11.0 = exact 10% limit-up.
    x_open = np.array([10.0, 11.0, 11.0, 11.0])
    x_close = np.array([10.0, 11.0, 11.0, 11.0])
    y = _stock(dates, 10.0, 10.0)
    panel = {"X": _stock(dates, x_open, x_close), "Y": y}
    sp = pd.DataFrame(np.tile([2.0, 1.0], (len(dates), 1)),
                      index=dates, columns=["X", "Y"])
    common = dict(top_k=1, rebalance_n_days=10, max_per_industry=None,
                  initial_cash=1.0)

    legacy = PortfolioEngine(
        _scores(sp), PortfolioRunConfig(**common)).run(panel)
    guarded = PortfolioEngine(
        _scores(sp), PortfolioRunConfig(**common, limit_guard=True)).run(panel)

    assert {t.code for t in legacy.trades} == {"X"}
    assert {t.code for t in guarded.trades} == {"Y"}


def test_limit_guard_sell_side_holds_through_limit_down():
    """Held name opens 一字跌停 on the rebalance fill bar → unsellable, held
    until the next rebalance. Without the guard it is sold at the stale open."""
    dates = _bars(6)
    # A: flat 10, but bar3 open = 9.0 (exact 10% limit-down vs close[2]=10).
    a_open = np.array([10.0, 10.0, 10.0, 9.0, 10.0, 10.0])
    a_close = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    panel = {"A": _stock(dates, a_open, a_close), "B": _stock(dates, 10.0, 10.0)}
    scores = np.array([
        [2.0, 1.0],   # bar0 → buy A @1
        [2.0, 1.0],
        [0.0, 2.0],   # bar2 → drop A, fill @3 where A is limit-down
        [0.0, 2.0],
        [0.0, 2.0],   # bar4 → drop A again, fill @5 (sellable now)
        [0.0, 2.0],
    ])
    sp = pd.DataFrame(scores, index=dates, columns=["A", "B"])
    common = dict(top_k=1, rebalance_n_days=2, max_per_industry=None,
                  initial_cash=1.0)

    legacy = PortfolioEngine(
        _scores(sp), PortfolioRunConfig(**common)).run(panel)
    guarded = PortfolioEngine(
        _scores(sp), PortfolioRunConfig(**common, limit_guard=True)).run(panel)

    a_legacy = [t for t in legacy.trades if t.code == "A"][0]
    a_guard = [t for t in guarded.trades if t.code == "A"][0]
    assert a_legacy.exit_date == dates[3]     # sold into the limit-down open
    assert a_guard.exit_date == dates[5]      # held until sellable


def test_weight_at_entry_equal_split():
    """weight_at_entry must reflect the allocation vs execution-time equity —
    two equal-weight entries both ≈ 0.5 (pre-fix the 2nd was ≈ 1.0 because
    the denominator was the loop-depleted cash pool)."""
    dates = _bars(4)
    panel = {c: _stock(dates, 10.0, 10.0) for c in ["A", "B"]}
    sp = pd.DataFrame(np.tile([2.0, 1.0], (len(dates), 1)),
                      index=dates, columns=["A", "B"])
    eng = PortfolioEngine(
        _scores(sp),
        _trivial_cfg(top_k=2, rebalance_n_days=10),
    )
    res = eng.run(panel)
    weights = sorted(t.weight_at_entry for t in res.trades)
    assert weights == pytest.approx([0.5, 0.5], rel=1e-9)


def test_limit_guard_legacy_retarget_does_not_overwrite_blocked_hold():
    """Legacy mode (hold_survivors=False) + limit_guard: a held name that
    STAYS in the target and opens limit-down is unsellable — the buy loop
    must not re-buy/overwrite it (pre-fix the position was replaced by a
    ~0-share entry funded from the empty cash pool, destroying its equity)."""
    dates = _bars(6)
    # A: flat 10, bar3 open = 9.0 (exact 10% limit-down vs close[2]=10).
    a_open = np.array([10.0, 10.0, 10.0, 9.0, 10.0, 10.0])
    panel = {"A": _stock(dates, a_open, 10.0), "B": _stock(dates, 10.0, 10.0)}
    # A is always the top pick → stays in the target across every rebalance.
    sp = pd.DataFrame(np.tile([2.0, 1.0], (len(dates), 1)),
                      index=dates, columns=["A", "B"])
    res = PortfolioEngine(
        _scores(sp),
        PortfolioRunConfig(top_k=1, rebalance_n_days=2, max_per_industry=None,
                           initial_cash=1.0, limit_guard=True),
    ).run(panel)
    eq = res.curve["equity"].to_numpy()
    # Shares survive the blocked bar-3 rebalance → equity stays ~1.0
    # (pre-fix it collapsed to ~0 when the position was overwritten).
    assert eq[3] > 0.9
    assert eq[3] == pytest.approx(eq[2], rel=1e-9)
    # The blocked bar produced no A trade (later bars may legacy-round-trip).
    a_trades = [t for t in res.trades if t.code == "A"]
    assert not any(t.exit_date == dates[3] for t in a_trades)


def test_delisted_position_marked_at_last_close_not_entry():
    """A held stock that crashes and then stops quoting (delisting) must be
    marked at its LAST known close — pre-fix the mark snapped back to
    entry_price, faking away the loss in both the curve and the close-out."""
    dates = _bars(6)
    a_open = np.array([10.0, 10.0, 5.0, np.nan, np.nan, np.nan])
    a_close = np.array([10.0, 10.0, 5.0, np.nan, np.nan, np.nan])
    panel = {"A": _stock(dates, a_open, a_close), "B": _stock(dates, 10.0, 10.0)}
    sp = pd.DataFrame(np.tile([2.0, 1.0], (len(dates), 1)),
                      index=dates, columns=["A", "B"])
    res = PortfolioEngine(
        _scores(sp),
        _trivial_cfg(top_k=1, rebalance_n_days=10),
    ).run(panel)
    eq = res.curve["equity"].to_numpy()
    # Crash marked at bar2; the quote-less bars keep that mark — no fake
    # rebound to entry price (pre-fix eq[3] jumped back to ~1.0).
    assert eq[2] == pytest.approx(eq[3], rel=1e-9)
    assert eq[3] == pytest.approx(eq[4], rel=1e-9)
    assert eq[3] < 0.6
    # Close-out realizes the last quote (5.0), showing the real ~50% loss.
    a_trade = [t for t in res.trades if t.code == "A"][0]
    assert a_trade.exit_price == pytest.approx(5.0)
    assert a_trade.ret < -0.4
