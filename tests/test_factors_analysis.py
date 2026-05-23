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
    # NaN survives roundtrip (was the C1 bug case)
    assert pd.isna(loaded.half_life["f3"])
    assert loaded.half_life["f1"] == 10.0
    # date roundtrip
    assert loaded.start_date == res.start_date
    assert loaded.end_date == res.end_date
    # other scalar fields
    assert loaded.ic_window == res.ic_window
    assert loaded.n_days == res.n_days
    # daily_ic survives
    pd.testing.assert_series_equal(loaded.daily_ic["f1"], res.daily_ic["f1"], check_freq=False)
    pd.testing.assert_series_equal(loaded.ic_ir, res.ic_ir)
    pd.testing.assert_series_equal(loaded.abs_ic_mean, res.abs_ic_mean)


def test_factor_analysis_result_json_is_rfc_compliant(tmp_path):
    """to_json must emit valid RFC 8259 JSON (no bare NaN / Infinity tokens)."""
    res = _make_result_fixture()
    out_path = tmp_path / "result.json"
    res.to_json(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "NaN" not in text, "NaN must serialize to null per RFC 8259"
    assert "Infinity" not in text
    # And the file should parse with the strict json module
    import json
    parsed = json.loads(text)  # would fail with strict parser if NaN present;
                               # CPython's json is lenient but we still want None
    assert parsed["half_life"]["f3"] is None
