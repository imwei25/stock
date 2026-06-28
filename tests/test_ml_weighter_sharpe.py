"""SharpeWeighter unit + integration tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LassoSelector
from stockpool.ml.weighters import SharpeWeighter


def _pooled_xy(
    n_stocks: int = 30,
    n_dates: int = 60,
    signal_strength: float = 1.0,
    noise: float = 0.5,
    seed: int = 0,
):
    """Build (X, y) with MultiIndex[(stock, date)] for pooled cross-sectional fit.

    y per (stock, date) = signal_strength * good_factor + noise * eps;
    bad_factor is pure noise; inv_factor has the opposite sign.
    """
    rng = np.random.default_rng(seed)
    stocks = [f"S{i:03d}" for i in range(n_stocks)]
    dates = pd.date_range("2024-01-02", periods=n_dates, freq="B")
    idx = pd.MultiIndex.from_product([stocks, dates], names=["stock", "date"])
    n = len(idx)

    good = rng.normal(0, 1, n)
    bad = rng.normal(0, 1, n)
    inv = rng.normal(0, 1, n)
    eps = rng.normal(0, 1, n)
    y_vals = signal_strength * good - signal_strength * inv + noise * eps

    X = pd.DataFrame(
        {"good": good, "bad": bad, "inv": inv},
        index=idx,
    )
    y = pd.Series(y_vals, index=idx, name="y")
    return X, y


def test_sharpe_weighter_predictive_factor_gets_largest_weight():
    X, y = _pooled_xy(n_stocks=40, n_dates=80, signal_strength=1.0, noise=0.3, seed=1)
    w = SharpeWeighter()
    w.fit(X, y)
    ws = w.weights()
    assert set(ws.index) == {"good", "bad", "inv"}
    # The two informative factors should both dominate the noise factor.
    assert abs(ws["good"]) > abs(ws["bad"])
    assert abs(ws["inv"]) > abs(ws["bad"])


def test_sharpe_weighter_sign_inversion_for_negative_sharpe():
    """A factor that drives returns *down* (negative LS) gets a negative weight."""
    X, y = _pooled_xy(n_stocks=40, n_dates=80, signal_strength=1.0, noise=0.3, seed=2)
    w = SharpeWeighter()
    w.fit(X, y)
    assert w.sharpe["good"] > 0
    assert w.sharpe["inv"] < 0
    assert w.weights()["good"] > 0
    assert w.weights()["inv"] < 0


def test_sharpe_weighter_l1_normalised():
    X, y = _pooled_xy(seed=3)
    w = SharpeWeighter()
    w.fit(X, y)
    ws = w.weights()
    assert ws.abs().sum() == pytest.approx(1.0, abs=1e-9)


def test_sharpe_weighter_threshold_filters_weak_factor():
    """min_abs_sharpe cuts factors whose |Sharpe| falls below threshold."""
    X, y = _pooled_xy(seed=4)
    base = SharpeWeighter(min_abs_sharpe=0.0)
    base.fit(X, y)
    sharpes = base.sharpe
    # Threshold above the noise factor's |Sharpe|, below the signal factor's.
    cut = (abs(sharpes["bad"]) + abs(sharpes["good"])) / 2
    filt = SharpeWeighter(min_abs_sharpe=cut)
    filt.fit(X, y)
    assert filt.weights()["bad"] == 0.0


def test_sharpe_weighter_zero_signal_falls_back_to_equal_weight():
    """When every |Sharpe| < min_abs_sharpe, fall back to equal weight."""
    X, y = _pooled_xy(seed=5)
    w = SharpeWeighter(min_abs_sharpe=1e9)
    w.fit(X, y)
    ws = w.weights()
    assert len(ws) == 3
    np.testing.assert_allclose(ws.to_numpy(), [1 / 3] * 3, atol=1e-9)


def test_sharpe_weighter_predict_is_weighted_sum():
    X, y = _pooled_xy(seed=6)
    w = SharpeWeighter()
    w.fit(X, y)
    preds = w.predict(X)
    assert preds.shape == (len(X),)
    # Contributions row sums equal predict (linear weighter contract).
    contribs = w.contributions(X)
    np.testing.assert_allclose(
        contribs.sum(axis=1).to_numpy(), preds.to_numpy(), atol=1e-9,
    )


def test_sharpe_weighter_requires_date_multiindex():
    """Per-stock mode (flat index) is not supported and must raise clearly."""
    X = pd.DataFrame(
        np.random.RandomState(0).randn(100, 3),
        columns=["a", "b", "c"],
    )
    y = pd.Series(np.random.RandomState(0).randn(100))
    w = SharpeWeighter()
    with pytest.raises(ValueError, match="MultiIndex"):
        w.fit(X, y)


def test_sharpe_weighter_handles_thin_cross_sections():
    """If too few stocks per day, those dates contribute no LS observation."""
    # 5 stocks per day, but min_stocks_per_day=10 → all dates skipped → fallback.
    X, y = _pooled_xy(n_stocks=5, n_dates=40, seed=7)
    w = SharpeWeighter(min_stocks_per_day=10)
    w.fit(X, y)
    # Every per-factor Sharpe is 0 → equal-weight fallback.
    np.testing.assert_allclose(
        w.weights().to_numpy(), [1 / 3] * 3, atol=1e-9,
    )


def test_sharpe_weighter_pipeline_integration():
    """TwoStepPipeline accepts SharpeWeighter and produces non-trivial predictions."""
    X, y = _pooled_xy(n_stocks=40, n_dates=80, seed=8)
    pipe = TwoStepPipeline(
        selector=LassoSelector(alpha=0.001),
        weighter=SharpeWeighter(),
    )
    info = pipe.fit(X, y)
    assert info is not None
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    # Predictions should be informative on the training set.
    corr = float(preds.corr(y, method="spearman"))
    assert corr > 0.1, f"expected positive Spearman corr, got {corr}"


def test_sharpe_weighter_init_rejects_bad_args():
    with pytest.raises(ValueError, match="quantile"):
        SharpeWeighter(quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        SharpeWeighter(quantile=0.5)
    with pytest.raises(ValueError, match="min_stocks_per_day"):
        SharpeWeighter(min_stocks_per_day=3)
    with pytest.raises(ValueError, match="min_valid_days"):
        SharpeWeighter(min_valid_days=1)
    with pytest.raises(ValueError, match="min_abs_sharpe"):
        SharpeWeighter(min_abs_sharpe=-1.0)
