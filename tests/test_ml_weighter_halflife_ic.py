"""HalfLifeICWeighter unit + integration tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LassoSelector
from stockpool.ml.weighters import HalfLifeICWeighter


def _pooled_xy_with_regime_shift(
    n_stocks: int = 40,
    n_dates: int = 80,
    seed: int = 0,
):
    """Two factors:
      * ``early_good`` — positive IC in the first half of the window only.
      * ``late_good``  — positive IC in the second half only.
    A short halflife (recent-weighted) should favour ``late_good``;
    an equal-weight IC should treat them as ~tied.
    """
    rng = np.random.default_rng(seed)
    stocks = [f"S{i:03d}" for i in range(n_stocks)]
    dates = pd.date_range("2024-01-02", periods=n_dates, freq="B")
    idx = pd.MultiIndex.from_product([stocks, dates], names=["stock", "date"])
    n = len(idx)

    early = rng.normal(0, 1, n)
    late = rng.normal(0, 1, n)
    eps = 0.4 * rng.normal(0, 1, n)

    date_per_row = idx.get_level_values("date").to_numpy()
    early_active = date_per_row < dates[n_dates // 2]
    late_active = ~early_active
    y = (
        np.where(early_active, early, 0.0)
        + np.where(late_active, late, 0.0)
        + eps
    )

    X = pd.DataFrame({"early_good": early, "late_good": late}, index=idx)
    y_s = pd.Series(y, index=idx, name="y")
    return X, y_s


def _pooled_xy_simple(n_stocks=30, n_dates=60, seed=0):
    rng = np.random.default_rng(seed)
    stocks = [f"S{i:03d}" for i in range(n_stocks)]
    dates = pd.date_range("2024-01-02", periods=n_dates, freq="B")
    idx = pd.MultiIndex.from_product([stocks, dates], names=["stock", "date"])
    n = len(idx)
    good = rng.normal(0, 1, n)
    bad = rng.normal(0, 1, n)
    inv = rng.normal(0, 1, n)
    y = good - inv + 0.4 * rng.normal(0, 1, n)
    X = pd.DataFrame({"good": good, "bad": bad, "inv": inv}, index=idx)
    return X, pd.Series(y, index=idx)


def test_short_halflife_favours_recent_signal():
    """Short halflife (recent-dominant) → late_good outweighs early_good;
    long halflife (≈ no decay) → both ~tied."""
    X, y = _pooled_xy_with_regime_shift(seed=1)
    short = HalfLifeICWeighter(halflife=5.0)
    long = HalfLifeICWeighter(halflife=1e6)
    short.fit(X, y)
    long.fit(X, y)
    short_ratio = abs(short.ic["late_good"]) / max(abs(short.ic["early_good"]), 1e-9)
    long_ratio = abs(long.ic["late_good"]) / max(abs(long.ic["early_good"]), 1e-9)
    # Short halflife exaggerates the late-regime signal.
    assert short_ratio > long_ratio
    # And the late_good gets a strictly larger weight under short halflife.
    assert abs(short.weights()["late_good"]) > abs(short.weights()["early_good"])


def test_sign_inversion_for_negative_ic():
    X, y = _pooled_xy_simple(seed=2)
    w = HalfLifeICWeighter(halflife=30)
    w.fit(X, y)
    assert w.ic["good"] > 0
    assert w.ic["inv"] < 0
    ws = w.weights()
    assert ws["good"] > 0
    assert ws["inv"] < 0


def test_l1_normalised():
    X, y = _pooled_xy_simple(seed=3)
    w = HalfLifeICWeighter(halflife=30)
    w.fit(X, y)
    assert w.weights().abs().sum() == pytest.approx(1.0, abs=1e-9)


def test_min_abs_ic_filter():
    X, y = _pooled_xy_simple(seed=4)
    base = HalfLifeICWeighter(halflife=30, min_abs_ic=0.0)
    base.fit(X, y)
    cut = (abs(base.ic["bad"]) + abs(base.ic["good"])) / 2
    filt = HalfLifeICWeighter(halflife=30, min_abs_ic=cut)
    filt.fit(X, y)
    assert filt.weights()["bad"] == 0.0


def test_zero_signal_falls_back_to_equal():
    X, y = _pooled_xy_simple(seed=5)
    w = HalfLifeICWeighter(halflife=30, min_abs_ic=1e9)
    w.fit(X, y)
    np.testing.assert_allclose(w.weights().to_numpy(), [1 / 3] * 3, atol=1e-9)


def test_contributions_row_sum_equals_predict():
    X, y = _pooled_xy_simple(seed=6)
    w = HalfLifeICWeighter(halflife=30)
    w.fit(X, y)
    preds = w.predict(X)
    contribs = w.contributions(X)
    np.testing.assert_allclose(
        contribs.sum(axis=1).to_numpy(), preds.to_numpy(), atol=1e-9,
    )


def test_requires_multiindex():
    X = pd.DataFrame(np.random.RandomState(0).randn(100, 3), columns=list("abc"))
    y = pd.Series(np.random.RandomState(0).randn(100))
    w = HalfLifeICWeighter(halflife=30)
    with pytest.raises(ValueError, match="MultiIndex"):
        w.fit(X, y)


def test_thin_cross_sections_skipped():
    X, y = _pooled_xy_simple(n_stocks=5, n_dates=40, seed=7)
    w = HalfLifeICWeighter(halflife=30, min_stocks_per_day=10)
    w.fit(X, y)
    np.testing.assert_allclose(w.weights().to_numpy(), [1 / 3] * 3, atol=1e-9)


def test_pipeline_integration():
    X, y = _pooled_xy_simple(seed=8)
    pipe = TwoStepPipeline(
        selector=LassoSelector(alpha=0.001),
        weighter=HalfLifeICWeighter(halflife=30),
    )
    info = pipe.fit(X, y)
    assert info is not None
    preds = pipe.predict(X)
    corr = float(preds.corr(y, method="spearman"))
    assert corr > 0.1


def test_init_validation():
    with pytest.raises(ValueError, match="halflife"):
        HalfLifeICWeighter(halflife=0)
    with pytest.raises(ValueError, match="halflife"):
        HalfLifeICWeighter(halflife=-1)
    with pytest.raises(ValueError, match="min_stocks_per_day"):
        HalfLifeICWeighter(min_stocks_per_day=3)
    with pytest.raises(ValueError, match="min_abs_ic"):
        HalfLifeICWeighter(min_abs_ic=-0.1)
