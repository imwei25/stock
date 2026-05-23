"""LightGBMSelector unit + integration tests (F2 PR-B1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LassoSelector, LightGBMSelector
from stockpool.ml.weighters import ICWeighter


def _nonlinear_xy(n: int = 500, seed: int = 0):
    """y = x0 * sign(x1) + small noise; x2/x3/x4 are pure noise.

    x0 is a linear main effect (Lasso can find it);
    x1 modulates x0 via sign → non-linear interaction (LGB can find it).
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    x4 = rng.normal(0, 1, n)
    y = x0 * np.sign(x1) + 0.1 * rng.normal(0, 1, n)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4})
    return X, pd.Series(y)


def _linear_signal_xy(n: int = 400, n_signal: int = 5, n_noise: int = 0, seed: int = 0):
    """y = sum(x_i for i in range(n_signal)) + small noise."""
    rng = np.random.default_rng(seed)
    cols = {}
    y = np.zeros(n)
    for i in range(n_signal):
        col = rng.normal(0, 1, n)
        cols[f"sig{i}"] = col
        y += col
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.normal(0, 1, n)
    y += 0.1 * rng.normal(0, 1, n)
    return pd.DataFrame(cols), pd.Series(y)


def test_lightgbm_selector_picks_nonlinear_features():
    """LGB finds the x0 main effect AND the x1 sign-modulator."""
    X, y = _nonlinear_xy(n=800, seed=1)
    sel = LightGBMSelector(top_k_factors=2, min_importance_ratio=0.01, random_state=1)
    sel.fit(X, y)
    picked = sel.selected_factors()
    assert "x0" in picked, f"expected x0 in selection, got {picked}"
    assert "x1" in picked, f"expected x1 in selection, got {picked}"


def test_lightgbm_selector_top_k_truncates():
    """top_k_factors=2 → exactly 2 selected when all 5 factors have signal."""
    X, y = _linear_signal_xy(n=500, n_signal=5, seed=2)
    sel = LightGBMSelector(top_k_factors=2, min_importance_ratio=0.0, random_state=2)
    sel.fit(X, y)
    assert len(sel.selected_factors()) == 2


def test_lightgbm_selector_min_importance_filter():
    """Tight ratio (0.99) keeps only the single strongest factor."""
    X, y = _linear_signal_xy(n=500, n_signal=5, seed=3)
    sel = LightGBMSelector(top_k_factors=10, min_importance_ratio=0.99, random_state=3)
    sel.fit(X, y)
    assert len(sel.selected_factors()) <= 1


def test_lightgbm_selector_deterministic_with_seed():
    """Same data + same random_state → identical selection."""
    X, y = _nonlinear_xy(n=500, seed=4)
    sel1 = LightGBMSelector(random_state=42)
    sel2 = LightGBMSelector(random_state=42)
    sel1.fit(X, y)
    sel2.fit(X, y)
    assert sel1.selected_factors() == sel2.selected_factors()


def test_lightgbm_selector_coef_normalized():
    """coef_ sums to ~1.0 in non-degenerate case (importance normalized)."""
    X, y = _linear_signal_xy(n=500, n_signal=4, seed=5)
    sel = LightGBMSelector(random_state=5)
    sel.fit(X, y)
    assert sel.coef_ is not None
    total = float(sel.coef_.sum())
    assert abs(total - 1.0) < 1e-6, f"expected sum ≈ 1.0, got {total}"


def test_lightgbm_selector_empty_when_y_constant():
    """Constant y → 0 gain on every split → empty selection."""
    X = pd.DataFrame({
        "a": np.linspace(0, 1, 50),
        "b": np.linspace(1, 0, 50),
        "c": np.random.default_rng(0).normal(0, 1, 50),
    })
    y = pd.Series([1.0] * 50)
    sel = LightGBMSelector(random_state=6)
    sel.fit(X, y)
    assert sel.selected_factors() == []


def test_lightgbm_selector_empty_input():
    """Empty X → empty selection (no crash)."""
    X = pd.DataFrame({"a": [], "b": []}, dtype=float)
    y = pd.Series([], dtype=float)
    sel = LightGBMSelector(random_state=7)
    sel.fit(X, y)
    assert sel.selected_factors() == []


def test_two_step_pipeline_with_lgb_selector_and_ic_weighter():
    """Integration: TwoStepPipeline(LGB selector + IC weighter) fit→predict round-trip."""
    X, y = _linear_signal_xy(n=500, n_signal=3, n_noise=2, seed=8)
    pipeline = TwoStepPipeline(
        selector=LightGBMSelector(top_k_factors=3, random_state=8),
        weighter=ICWeighter(use_rank=True),
    )
    info = pipeline.fit(X, y)
    assert len(info.selected_factors) <= 3
    if info.selected_factors:
        preds = pipeline.predict(X)
        assert len(preds) == len(X)
        corr = float(preds.corr(y, method="spearman"))
        assert corr > 0.1, f"expected positive Spearman corr, got {corr}"
