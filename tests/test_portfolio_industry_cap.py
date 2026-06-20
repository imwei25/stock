"""Tests for industry cap in PortfolioEngine._select_top_k."""
from __future__ import annotations

import pandas as pd
import pytest

from stockpool.portfolio.engine import _select_top_k


def _all_open(codes, px=10.0):
    return pd.Series({c: px for c in codes})


def test_greedy_cap_correct():
    """scores=[1.0,0.9,0.8,0.7,0.6], inds=[A,A,A,B,B], k=3, cap=2 → top order [A0,A1,B0]."""
    scores = {"a0": 1.0, "a1": 0.9, "a2": 0.8, "b0": 0.7, "b1": 0.6}
    sector_map = {"a0": "A", "a1": "A", "a2": "A", "b0": "B", "b1": "B"}
    out = _select_top_k(
        scores, k=3, opens_next=_all_open(scores),
        sector_map=sector_map, max_per_industry=2,
    )
    assert out == {"a0", "a1", "b0"}


def test_no_cap_when_max_none():
    scores = {"a0": 1.0, "a1": 0.9, "a2": 0.8}
    sector_map = {"a0": "A", "a1": "A", "a2": "A"}
    out = _select_top_k(
        scores, k=3, opens_next=_all_open(scores),
        sector_map=sector_map, max_per_industry=None,
    )
    assert out == {"a0", "a1", "a2"}


def test_no_cap_when_sector_map_empty_or_none():
    scores = {"a0": 1.0, "a1": 0.9, "a2": 0.8}
    out_none = _select_top_k(
        scores, k=3, opens_next=_all_open(scores),
        sector_map=None, max_per_industry=2,
    )
    out_empty = _select_top_k(
        scores, k=3, opens_next=_all_open(scores),
        sector_map={}, max_per_industry=2,
    )
    assert out_none == {"a0", "a1", "a2"}
    assert out_empty == {"a0", "a1", "a2"}


def test_all_unknown_skips_cap():
    """If every candidate has no sector, cap is skipped (else everyone is 'Unknown')."""
    scores = {"a0": 1.0, "a1": 0.9, "a2": 0.8, "a3": 0.7}
    sector_map = {}   # no codes mapped
    out = _select_top_k(
        scores, k=3, opens_next=_all_open(scores),
        sector_map=sector_map, max_per_industry=2,
    )
    assert out == {"a0", "a1", "a2"}


def test_partial_unknown_counts_in_unknown_bucket():
    """Some codes have sectors, others don't → unmapped go to 'Unknown' bucket
    that counts normally against the cap."""
    scores = {"a0": 1.0, "u0": 0.9, "u1": 0.8, "u2": 0.7, "b0": 0.6}
    # a0 has sector A; u0/u1/u2 are unmapped; b0 has sector B.
    sector_map = {"a0": "A", "b0": "B"}
    out = _select_top_k(
        scores, k=5, opens_next=_all_open(scores),
        sector_map=sector_map, max_per_industry=2,
    )
    # Walk: a0 (A=1) take; u0 (Unknown=1) take; u1 (Unknown=2) take;
    # u2 (Unknown=3, cap hit) skip; b0 (B=1) take.
    assert out == {"a0", "u0", "u1", "b0"}


def test_cap_with_missing_open_price_skipped():
    """Codes without an open[t+1] are skipped (don't count against cap)."""
    scores = {"a0": 1.0, "a1": 0.9, "a2": 0.8, "b0": 0.7}
    sector_map = {"a0": "A", "a1": "A", "a2": "A", "b0": "B"}
    opens = pd.Series({"a0": 10.0, "a1": float("nan"), "a2": 10.0, "b0": 10.0})
    out = _select_top_k(
        scores, k=3, opens_next=opens,
        sector_map=sector_map, max_per_industry=2,
    )
    # a0 take (A=1), a1 skip (no open), a2 take (A=2), b0 take (B=1).
    assert out == {"a0", "a2", "b0"}


def test_cap_engine_integration():
    """End-to-end: PortfolioEngine respects max_per_industry from cfg."""
    import numpy as np
    from stockpool.backtesting.framework import TradeCosts
    from stockpool.config import PortfolioRunConfig
    from stockpool.portfolio.engine import PortfolioEngine
    from stockpool.portfolio.strategy import PrecomputedScoreStrategy

    dates = pd.bdate_range("2024-01-02", periods=10)
    codes = ["a0", "a1", "a2", "b0", "b1"]
    panel = {c: pd.DataFrame({"date": dates, "open": [10.0] * 10, "close": [10.0] * 10})
             for c in codes}
    sp = pd.DataFrame(np.nan, index=dates, columns=codes)
    sp.loc[dates[0]] = [5.0, 4.0, 3.0, 2.0, 1.0]
    strat = PrecomputedScoreStrategy(sp)
    eng = PortfolioEngine(
        strategy=strat,
        portfolio_cfg=PortfolioRunConfig(top_k=3, rebalance_n_days=4, max_per_industry=2),
        costs=TradeCosts(0.0, 0.0),
        sector_map={"a0": "A", "a1": "A", "a2": "A", "b0": "B", "b1": "B"},
    )
    res = eng.run(panel)
    # Bar 0 rebalance → exec bar 1. Cap=2 per industry, k=3 → {a0,a1,b0}.
    target_codes = res.rebalance_log["target_codes"].iloc[0]
    assert set(target_codes) == {"a0", "a1", "b0"}
