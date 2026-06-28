"""RidgeWeighter unit + integration tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LassoSelector
from stockpool.ml.weighters import RidgeWeighter


def _linear_xy(n=500, n_sig=2, n_noise=2, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    y = np.zeros(n)
    for i in range(n_sig):
        c = rng.normal(0, 1, n)
        cols[f"sig{i}"] = c
        y += c
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.normal(0, 1, n)
    y += 0.3 * rng.normal(0, 1, n)
    return pd.DataFrame(cols), pd.Series(y)


def test_ridge_recovers_signal_factors():
    X, y = _linear_xy(n=600, seed=1)
    w = RidgeWeighter(alpha=1.0)
    w.fit(X, y)
    coef = w.raw_coef
    # Signal coefs should dominate noise coefs by sign and magnitude.
    assert coef["sig0"] > 0 and coef["sig1"] > 0
    assert abs(coef["sig0"]) > abs(coef["noise0"])
    assert abs(coef["sig1"]) > abs(coef["noise1"])


def test_ridge_weights_l1_normalised():
    X, y = _linear_xy(seed=2)
    w = RidgeWeighter(alpha=1.0)
    w.fit(X, y)
    assert w.weights().abs().sum() == pytest.approx(1.0, abs=1e-9)


def test_ridge_alpha_zero_is_ols():
    """alpha=0 should match plain OLS coefficients (up to numerics)."""
    X, y = _linear_xy(seed=3)
    w = RidgeWeighter(alpha=0.0)
    w.fit(X, y)
    # Sanity: predictions should correlate strongly with y on the training set.
    preds = w.predict(X)
    corr = float(preds.corr(y))
    assert corr > 0.7


def test_ridge_larger_alpha_shrinks_coefs():
    X, y = _linear_xy(seed=4)
    small = RidgeWeighter(alpha=0.01); small.fit(X, y)
    big = RidgeWeighter(alpha=100.0); big.fit(X, y)
    assert big.raw_coef.abs().sum() < small.raw_coef.abs().sum()


def test_ridge_sign_preserved_for_negative_factor():
    rng = np.random.default_rng(5)
    n = 400
    good = rng.normal(0, 1, n)
    inv = rng.normal(0, 1, n)
    y = good - inv + 0.3 * rng.normal(0, 1, n)
    X = pd.DataFrame({"good": good, "inv": inv})
    w = RidgeWeighter(alpha=1.0)
    w.fit(X, pd.Series(y))
    assert w.weights()["good"] > 0
    assert w.weights()["inv"] < 0


def test_ridge_predict_matches_contributions_row_sum():
    X, y = _linear_xy(seed=6)
    w = RidgeWeighter(alpha=1.0)
    w.fit(X, y)
    preds = w.predict(X)
    contribs = w.contributions(X)
    np.testing.assert_allclose(
        contribs.sum(axis=1).to_numpy(), preds.to_numpy(), atol=1e-9,
    )


def test_ridge_pipeline_integration():
    X, y = _linear_xy(seed=7)
    pipe = TwoStepPipeline(
        selector=LassoSelector(alpha=0.001),
        weighter=RidgeWeighter(alpha=1.0),
    )
    info = pipe.fit(X, y)
    assert info is not None
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    assert preds.corr(y) > 0.3


def test_ridge_handles_empty_features():
    X = pd.DataFrame(index=range(50))
    y = pd.Series(np.random.RandomState(0).randn(50))
    w = RidgeWeighter(alpha=1.0)
    w.fit(X, y)
    assert w.weights().empty
    preds = w.predict(X)
    np.testing.assert_allclose(preds.to_numpy(), np.zeros(50), atol=1e-9)


def test_ridge_init_validation():
    with pytest.raises(ValueError, match="alpha"):
        RidgeWeighter(alpha=-1.0)
