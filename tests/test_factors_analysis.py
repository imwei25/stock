"""Tests for stockpool.factors_analysis core."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpool.factors_analysis import (
    FactorAnalysisResult,
    analyze_factors,
    classify_regimes,
    compute_daily_ic,
    pick_top_factors,
)


def _make_result_fixture() -> FactorAnalysisResult:
    factor_names = ["f1", "f2", "f3"]
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    return FactorAnalysisResult(
        factor_names=factor_names,
        daily_ic={
            "f1": pd.Series([0.1] * 20, index=dates),
            "f2": pd.Series([-0.05] * 20, index=dates),
            "f3": pd.Series([0.0] * 20, index=dates),
        },
        mean_ic=pd.Series({"f1": 0.1, "f2": -0.05, "f3": 0.0}),
        ic_ir=pd.Series({"f1": 2.0, "f2": -1.0, "f3": 0.0}),
        abs_ic_mean=pd.Series({"f1": 0.1, "f2": 0.05, "f3": 0.0}),
        half_life=pd.Series({"f1": 10.0, "f2": 5.0, "f3": float("nan")}),
        ic_correlation=pd.DataFrame(
            [[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=factor_names, columns=factor_names,
        ),
        regime_ic={
            "bull": pd.Series({"f1": 0.15, "f2": -0.05, "f3": 0.0}),
            "bear": pd.Series({"f1": 0.05, "f2": -0.10, "f3": 0.0}),
            "sideways": pd.Series({"f1": 0.10, "f2": 0.0, "f3": 0.0}),
        },
        horizon=3,
        ic_window=20,
        n_stocks=10,
        n_days=20,
        start_date=dates[0],
        end_date=dates[-1],
    )


def test_factor_analysis_result_to_dict_roundtrip(tmp_path):
    res = _make_result_fixture()
    out_path = tmp_path / "result.json"
    res.to_json(out_path)
    loaded = FactorAnalysisResult.from_json(out_path)
    assert loaded.factor_names == res.factor_names
    assert loaded.horizon == 3
    assert loaded.n_stocks == 10
    pd.testing.assert_series_equal(loaded.mean_ic, res.mean_ic)
    pd.testing.assert_frame_equal(loaded.ic_correlation, res.ic_correlation)
    assert set(loaded.regime_ic.keys()) == {"bull", "bear", "sideways"}
    pd.testing.assert_series_equal(loaded.regime_ic["bull"], res.regime_ic["bull"])
