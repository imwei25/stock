"""LightGBMWeighter unit + integration tests (F2 PR-B2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LightGBMSelector
from stockpool.ml.weighters import LightGBMWeighter


def _linear_signal_xy(n: int = 500, n_signal: int = 3, n_noise: int = 2, seed: int = 0):
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


def test_lightgbm_weighter_fit_predict_round_trip():
    """Trained LGB predicts y positively correlated on the training set."""
    X, y = _linear_signal_xy(n=500, seed=1)
    w = LightGBMWeighter(random_state=1)
    w.fit(X, y)
    preds = w.predict(X)
    assert len(preds) == len(X)
    corr = float(preds.corr(y, method="spearman"))
    assert corr > 0.3, f"expected Spearman corr > 0.3, got {corr}"


def test_lightgbm_weighter_weights_are_mean_abs_shap():
    """weights() returns mean|SHAP| — non-negative, sum > 0 in normal fit."""
    X, y = _linear_signal_xy(n=500, seed=2)
    w = LightGBMWeighter(random_state=2)
    w.fit(X, y)
    ws = w.weights()
    assert len(ws) == len(X.columns)
    assert (ws >= 0).all(), f"|SHAP| should be non-negative, got {ws}"
    assert ws.sum() > 0


def test_lightgbm_weighter_contributions_shape_and_columns():
    """contributions(X) returns DataFrame with row=X.index, col=fit-time features."""
    X, y = _linear_signal_xy(n=300, seed=3)
    w = LightGBMWeighter(random_state=3)
    w.fit(X, y)
    contribs = w.contributions(X)
    assert contribs.shape == X.shape
    assert list(contribs.columns) == list(X.columns)
    pd.testing.assert_index_equal(contribs.index, X.index)


def test_lightgbm_weighter_contributions_row_sums_track_predict():
    """SHAP convention: row sums + base_value ≈ predict. Strong correlation."""
    X, y = _linear_signal_xy(n=300, seed=4)
    w = LightGBMWeighter(random_state=4)
    w.fit(X, y)
    preds = w.predict(X)
    contribs = w.contributions(X)
    row_sums = contribs.sum(axis=1)
    corr = float(row_sums.corr(preds))
    assert corr > 0.95, f"expected row_sums ↔ predict corr > 0.95, got {corr}"


def test_lightgbm_weighter_deterministic_with_seed():
    X, y = _linear_signal_xy(n=400, seed=5)
    w1 = LightGBMWeighter(random_state=42)
    w2 = LightGBMWeighter(random_state=42)
    w1.fit(X, y); w2.fit(X, y)
    np.testing.assert_array_almost_equal(
        w1.predict(X).values, w2.predict(X).values,
    )


def test_lightgbm_weighter_empty_input():
    X = pd.DataFrame({"a": [], "b": []}, dtype=float)
    y = pd.Series([], dtype=float)
    w = LightGBMWeighter(random_state=6)
    w.fit(X, y)
    assert w.weights().empty
    preds = w.predict(X)
    assert len(preds) == 0


def test_lightgbm_weighter_predict_missing_columns_raises():
    X, y = _linear_signal_xy(n=200, seed=7)
    w = LightGBMWeighter(random_state=7)
    w.fit(X, y)
    X_missing = X.drop(columns=[X.columns[0]])
    with pytest.raises(KeyError):
        w.predict(X_missing)


def test_two_step_pipeline_lgb_selector_lgb_weighter():
    """Integration: LGB selector + LGB weighter end-to-end."""
    X, y = _linear_signal_xy(n=500, n_signal=3, n_noise=2, seed=8)
    pipeline = TwoStepPipeline(
        selector=LightGBMSelector(top_k_factors=3, random_state=8),
        weighter=LightGBMWeighter(random_state=8),
    )
    info = pipeline.fit(X, y)
    if info.selected_factors:
        preds = pipeline.predict(X)
        assert len(preds) == len(X)
        contribs = pipeline.contributions(X)
        assert contribs.shape == (len(X), len(info.selected_factors))
        assert list(contribs.columns) == info.selected_factors
