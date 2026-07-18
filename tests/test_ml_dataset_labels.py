"""Tests for forward_return_panel/forward_return label_type interface (F2 PR-A)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.dataset import forward_return, forward_return_panel


def _close_panel(n_days: int = 20, n_stocks: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    codes = [f"s{i:02d}" for i in range(n_stocks)]
    return pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, (n_days, n_stocks)), axis=0),
        index=dates, columns=codes,
    )


def _close_series(n_days: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_days))
    return pd.DataFrame({"date": dates, "close": close})


def test_forward_return_panel_label_type_return_default():
    close = _close_panel()
    out_default = forward_return_panel(close, horizon=3)
    out_explicit = forward_return_panel(close, horizon=3, label_type="return")
    pd.testing.assert_frame_equal(out_default, out_explicit)
    expected = close.shift(-3) / close - 1.0
    pd.testing.assert_frame_equal(out_default, expected)


def test_forward_return_panel_label_type_vol_adjusted():
    """vol_adjusted = raw return ÷ trailing-20d realized vol as of bar t."""
    close = _close_panel(n_days=40)
    raw = forward_return_panel(close, horizon=3)
    adj = forward_return_panel(close, horizon=3, label_type="vol_adjusted")
    vol = close.pct_change().rolling(20, min_periods=10).std()
    expected = raw / vol.where(vol > 0)
    pd.testing.assert_frame_equal(adj, expected)
    # Warmup rows (< min_periods of vol history) have NaN labels.
    assert adj.iloc[:9].isna().all().all()


def test_forward_return_panel_label_type_cross_sec_rank():
    """cross_sec_rank = per-day pct rank of the raw return, centered.

    Rank ordering must match the raw return's ordering per day; values live
    in [-0.5, +0.5] and the per-day mean is ~0 when all cells are valid.
    """
    close = _close_panel(n_days=20, n_stocks=5)
    raw = forward_return_panel(close, horizon=3)
    ranked = forward_return_panel(close, horizon=3, label_type="cross_sec_rank")
    valid = raw.dropna(how="all")
    for t in valid.index[:10]:
        r, k = raw.loc[t], ranked.loc[t]
        assert (k >= -0.5).all() and (k <= 0.5).all()
        # order preserved
        assert (r.sort_values().index == k.sort_values().index).all()
    # full row → centered mean ≈ ((1..n)/n)/n mean − 0.5 = (n+1)/(2n) − 0.5
    n = close.shape[1]
    expected_mean = (n + 1) / (2 * n) - 0.5
    assert ranked.dropna(how="any").mean(axis=1).round(9).eq(
        round(expected_mean, 9)).all()
    # NaN cells stay NaN and don't shift others' ranks
    close_nan = close.copy()
    close_nan.iloc[:, 0] = np.nan
    ranked_nan = forward_return_panel(close_nan, horizon=3,
                                      label_type="cross_sec_rank")
    assert ranked_nan.iloc[:, 0].isna().all()


def test_forward_return_panel_cross_sec_rank_open_basis_composes():
    """cross_sec_rank applies on top of the open-basis return."""
    close = _close_panel(n_days=20, n_stocks=4)
    open_ = close * 0.998
    raw_open = forward_return_panel(close, horizon=3, open_=open_)
    ranked = forward_return_panel(close, horizon=3,
                                  label_type="cross_sec_rank", open_=open_)
    expected = raw_open.rank(axis=1, pct=True) - 0.5
    pd.testing.assert_frame_equal(ranked, expected)


def test_forward_return_panel_label_type_unknown_rejected():
    close = _close_panel()
    with pytest.raises(ValueError, match="label_type"):
        forward_return_panel(close, horizon=3, label_type="nonsense")


def test_forward_return_panel_horizon_must_be_positive():
    close = _close_panel()
    with pytest.raises(ValueError):
        forward_return_panel(close, horizon=0)


def test_forward_return_single_stock_label_type_return_default():
    df = _close_series()
    out_default = forward_return(df, horizon=3)
    out_explicit = forward_return(df, horizon=3, label_type="return")
    pd.testing.assert_series_equal(out_default, out_explicit)


def test_forward_return_single_stock_label_type_not_implemented_paths():
    """Panel-only transforms raise on the per-stock path (no cross-section)."""
    df = _close_series()
    with pytest.raises(NotImplementedError, match="vol_adjusted"):
        forward_return(df, horizon=3, label_type="vol_adjusted")
    with pytest.raises(NotImplementedError, match="cross_sec_rank"):
        forward_return(df, horizon=3, label_type="cross_sec_rank")


def test_ml_strategy_rejects_non_return_label_in_per_stock_mode():
    """cfg.label_type != 'return' + panel_mode='per_stock' raises at ctor
    (pre-fix it was silently ignored — labels stayed raw returns)."""
    from stockpool.backtesting.strategies import MLFactorStrategy
    from stockpool.config import MLFactorConfig

    cfg = MLFactorConfig(factors=["mom_20"], panel_mode="per_stock",
                         label_type="cross_sec_rank")
    with pytest.raises(ValueError, match="requires panel_mode='pooled'"):
        MLFactorStrategy(cfg)
