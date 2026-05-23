"""Smoke test for stockpool.factors_analysis_report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpool.factors_analysis import FactorAnalysisResult
from stockpool.factors_analysis_report import render_factor_analysis_report


def _build_result_for_report():
    factor_names = ["alpha_001", "momentum_20", "rsi_centered_14"]
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    rng = np.random.default_rng(0)
    return FactorAnalysisResult(
        factor_names=factor_names,
        daily_ic={
            n: pd.Series(rng.normal(0.05, 0.1, 50), index=dates)
            for n in factor_names
        },
        mean_ic=pd.Series({"alpha_001": 0.08, "momentum_20": 0.05, "rsi_centered_14": -0.02}),
        ic_ir=pd.Series({"alpha_001": 0.6, "momentum_20": 0.3, "rsi_centered_14": -0.1}),
        abs_ic_mean=pd.Series({"alpha_001": 0.08, "momentum_20": 0.06, "rsi_centered_14": 0.04}),
        half_life=pd.Series({"alpha_001": 12.0, "momentum_20": 8.0, "rsi_centered_14": 3.0}),
        ic_correlation=pd.DataFrame(
            [[1.0, 0.3, -0.1], [0.3, 1.0, 0.2], [-0.1, 0.2, 1.0]],
            index=factor_names, columns=factor_names,
        ),
        regime_ic={
            "bull": pd.Series({"alpha_001": 0.10, "momentum_20": 0.08, "rsi_centered_14": -0.01}),
            "bear": pd.Series({"alpha_001": 0.05, "momentum_20": 0.02, "rsi_centered_14": -0.05}),
            "sideways": pd.Series({"alpha_001": 0.07, "momentum_20": 0.05, "rsi_centered_14": -0.02}),
        },
        horizon=3, ic_window=60, n_stocks=20, n_days=50,
        start_date=dates[0], end_date=dates[-1],
    )


def test_render_html_writes_file(tmp_path):
    result = _build_result_for_report()
    out = tmp_path / "report.html"
    render_factor_analysis_report(result, out_path=out, picked=["alpha_001", "momentum_20"])
    assert out.exists()
    assert out.stat().st_size > 2048
    html = out.read_text(encoding="utf-8")
    # Each factor name should appear at least once
    for n in result.factor_names:
        assert n in html, f"missing {n} in HTML"
    # Picked section should be visible
    assert "alpha_001" in html and "momentum_20" in html
    # Regime headers should be present
    for regime in ("bull", "bear", "sideways"):
        assert regime in html


def test_render_html_handles_empty_regimes(tmp_path):
    result = _build_result_for_report()
    result.regime_ic.clear()
    out = tmp_path / "report.html"
    render_factor_analysis_report(result, out_path=out, picked=[])
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # No regime section when empty — but the document still renders
    assert "alpha_001" in html
