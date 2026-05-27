"""Tests for portfolio.strategy: PrecomputedScoreStrategy semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.portfolio.strategy import (
    PortfolioStrategy,
    PrecomputedScoreStrategy,
)


def _panel_data(codes, dates):
    return {
        c: pd.DataFrame({"date": dates, "close": [10.0] * len(dates)})
        for c in codes
    }


def _score_panel(dates, codes, values):
    return pd.DataFrame(values, index=pd.to_datetime(dates), columns=codes)


def test_precomputed_returns_scores_for_known_date():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    codes = ["A", "B", "C"]
    sp = _score_panel(dates, codes, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    panel_data = _panel_data(codes, dates)

    s = PrecomputedScoreStrategy(sp, name="test")
    out = s.predict_scores(pd.Timestamp("2024-01-03"), panel_data)
    assert out == {"A": 0.4, "B": 0.5, "C": 0.6}
    assert s.name == "test"


def test_precomputed_returns_empty_for_unknown_date():
    dates = ["2024-01-02"]
    codes = ["A"]
    sp = _score_panel(dates, codes, [[0.1]])
    s = PrecomputedScoreStrategy(sp)
    assert s.predict_scores(pd.Timestamp("2030-01-01"), _panel_data(codes, dates)) == {}


def test_precomputed_drops_nan():
    dates = ["2024-01-02"]
    codes = ["A", "B"]
    sp = _score_panel(dates, codes, [[np.nan, 0.5]])
    out = PrecomputedScoreStrategy(sp).predict_scores(
        pd.Timestamp("2024-01-02"), _panel_data(codes, dates),
    )
    assert out == {"B": 0.5}


def test_precomputed_filters_to_panel_data_codes():
    dates = ["2024-01-02"]
    codes = ["A", "B", "C"]
    sp = _score_panel(dates, codes, [[0.1, 0.2, 0.3]])
    # panel_data only has A and C
    pd_subset = _panel_data(["A", "C"], dates)
    out = PrecomputedScoreStrategy(sp).predict_scores(
        pd.Timestamp("2024-01-02"), pd_subset,
    )
    assert out == {"A": 0.1, "C": 0.3}


def test_precomputed_sorts_unsorted_panel():
    # Unsorted on purpose
    sp = pd.DataFrame(
        [[0.4], [0.1]],
        index=pd.to_datetime(["2024-01-04", "2024-01-02"]),
        columns=["A"],
    )
    s = PrecomputedScoreStrategy(sp)
    assert s.predict_scores(pd.Timestamp("2024-01-02"), _panel_data(["A"], ["2024-01-02"])) == {"A": 0.1}
    assert s.predict_scores(pd.Timestamp("2024-01-04"), _panel_data(["A"], ["2024-01-04"])) == {"A": 0.4}


def test_portfolio_strategy_is_abstract():
    with pytest.raises(TypeError):
        PortfolioStrategy()  # type: ignore[abstract]
