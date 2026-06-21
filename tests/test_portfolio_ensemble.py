"""Tests for portfolio.ensemble.StaggeredRunner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.framework import TradeCosts
from stockpool.config import PortfolioRunConfig
from stockpool.portfolio.engine import PortfolioEngine
from stockpool.portfolio.ensemble import EnsembleResult, StaggeredRunner
from stockpool.portfolio.strategy import PrecomputedScoreStrategy


def _setup(n_bars=12, codes=("A", "B", "C", "D"), rebalance_n_days=4, top_k=2):
    dates = pd.bdate_range("2024-01-02", periods=n_bars)
    panel = {
        c: pd.DataFrame({"date": dates, "open": [10.0] * n_bars, "close": [10.0] * n_bars})
        for c in codes
    }
    sp = pd.DataFrame(
        np.tile(np.arange(len(codes), 0, -1, dtype=float), (n_bars, 1)),
        index=dates, columns=list(codes),
    )
    strat = PrecomputedScoreStrategy(sp)

    def factory():
        return PortfolioEngine(
            strategy=strat,
            portfolio_cfg=PortfolioRunConfig(
                top_k=top_k, rebalance_n_days=rebalance_n_days,
                max_per_industry=None,
            ),
            costs=TradeCosts(0.0, 0.0),
        )
    return panel, factory


def test_n_equals_one_matches_single_run():
    panel, factory = _setup()
    single = factory().run(panel, start_offset=0)
    ens = StaggeredRunner(factory).run(panel, n_offsets=1)
    assert ens.n_offsets == 1
    pd.testing.assert_series_equal(
        ens.ensemble_curve["equity"].reset_index(drop=True),
        single.curve["equity"].astype(float).reset_index(drop=True),
        check_names=False,
    )


def test_offsets_non_overlapping_when_n_equals_period():
    """rebalance_n_days=4, n_offsets=4 → rebalance bar sets pairwise disjoint."""
    panel, factory = _setup(n_bars=20, rebalance_n_days=4)
    ens = StaggeredRunner(factory).run(panel, n_offsets=4)
    sets = []
    for r in ens.individual_results:
        dates_set = set(pd.to_datetime(r.rebalance_log["date"]).tolist())
        sets.append(dates_set)
    # pairwise disjoint
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j]), f"offsets {i} and {j} overlap"


def test_ensemble_is_mean_of_individuals():
    panel, factory = _setup(n_bars=15, rebalance_n_days=3)
    ens = StaggeredRunner(factory).run(panel, n_offsets=3)
    # Manually compute equal-weighted mean of the three equity curves
    # aligned to the ensemble's date axis.
    target = pd.DataFrame(
        {f"k{i}": pd.Series(
            r.curve["equity"].values,
            index=pd.to_datetime(r.curve["date"]),
        ) for i, r in enumerate(ens.individual_results)}
    )
    expected = target.mean(axis=1).values
    np.testing.assert_allclose(
        ens.ensemble_curve["equity"].values, expected, atol=1e-12,
    )


def test_envelope_columns_and_ordering():
    panel, factory = _setup(n_bars=10, rebalance_n_days=3)
    ens = StaggeredRunner(factory).run(panel, n_offsets=3)
    env = ens.envelope
    assert list(env.columns) == ["date", "min", "p25", "median", "p75", "max"]
    # Quantile ordering invariant per bar.
    assert (env["min"] <= env["p25"]).all()
    assert (env["p25"] <= env["median"]).all()
    assert (env["median"] <= env["p75"]).all()
    assert (env["p75"] <= env["max"]).all()


def test_aggregated_metrics_structure():
    panel, factory = _setup()
    ens = StaggeredRunner(factory).run(panel, n_offsets=2)
    agg = ens.aggregated_metrics
    assert "ensemble" in agg
    assert "per_offset" in agg
    for k in ("total_return", "annualized_return", "sharpe", "max_drawdown"):
        assert k in agg["per_offset"]
        slot = agg["per_offset"][k]
        assert set(slot.keys()) == {"median", "min", "max"}


def test_invalid_n_offsets_raises():
    panel, factory = _setup()
    with pytest.raises(ValueError):
        StaggeredRunner(factory).run(panel, n_offsets=0)


def test_parallel_matches_serial():
    """parallel=True vs serial 跑 staggered 结果应在 rtol=1e-12 内一致。

    PR-T1.3: StaggeredRunner gains a `components` kwarg + `parallel: bool`
    on `.run()`. When components is provided and parallel=True, runs offsets
    via ProcessPoolExecutor; results must match the serial path within
    rtol=1e-12 (sub-ULP FP drift across spawn boundaries is expected and
    well below cost/slippage noise in any portfolio simulation).
    """
    # Synthetic 8 codes × 40 bars panel.
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=40)
    codes = [f"S{i:03d}" for i in range(8)]
    panel = {}
    for code in codes:
        prices = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, len(dates)))
        panel[code] = pd.DataFrame({
            "date": dates,
            "open": prices * (1 + rng.normal(0, 0.001, len(dates))),
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": rng.integers(int(1e5), int(5e6), len(dates)),
        })

    # Deterministic score panel.
    rng_scores = np.random.default_rng(0)
    scores = pd.DataFrame(
        rng_scores.standard_normal((len(dates), len(codes))),
        index=dates, columns=codes,
    )
    strategy = PrecomputedScoreStrategy(scores, name="test")
    portfolio_cfg = PortfolioRunConfig(top_k=3, rebalance_n_days=5)
    costs = TradeCosts(buy_cost=0.001, sell_cost=0.001)

    components = (strategy, portfolio_cfg, costs, 0.02, None, {})

    def _factory():
        return PortfolioEngine(
            strategy=strategy, portfolio_cfg=portfolio_cfg, costs=costs,
            risk_free_rate=0.02, eligibility=None, sector_map={},
        )

    # Serial run.
    runner_serial = StaggeredRunner(_factory, risk_free_rate=0.02)
    ens_serial = runner_serial.run(panel, n_offsets=3)

    # Parallel run — needs `components` and `parallel=True`.
    runner_par = StaggeredRunner(
        _factory, components=components, risk_free_rate=0.02,
    )
    ens_par = runner_par.run(panel, n_offsets=3, parallel=True)

    # Equity curve within rtol=1e-12 (sub-ULP FP drift across spawn boundaries
    # is expected with ProcessPoolExecutor; well below cost/slippage noise).
    np.testing.assert_allclose(
        ens_serial.ensemble_curve["equity"].values,
        ens_par.ensemble_curve["equity"].values,
        rtol=1e-12, atol=0,
    )
    # Envelope within rtol=1e-12.
    for col in ("min", "p25", "median", "p75", "max"):
        np.testing.assert_allclose(
            ens_serial.envelope[col].values,
            ens_par.envelope[col].values,
            rtol=1e-12, atol=0,
        )
    # Ensemble metrics within rtol=1e-12.
    s_m = ens_serial.aggregated_metrics["ensemble"]
    p_m = ens_par.aggregated_metrics["ensemble"]
    assert set(s_m) == set(p_m)
    for k in s_m:
        sv, pv = s_m[k], p_m[k]
        if isinstance(sv, float) and np.isnan(sv):
            assert isinstance(pv, float) and np.isnan(pv), f"{k}: NaN mismatch"
        else:
            assert np.isclose(sv, pv, rtol=1e-12, atol=0), f"{k}: {sv} vs {pv}"
