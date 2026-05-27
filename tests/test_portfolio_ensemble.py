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
