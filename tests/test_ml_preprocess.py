"""Unit tests for cross-sectional factor preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_panel(n_days=5, n_stocks=30, seed=0):
    """Synthetic factor panel: T × N DataFrame, dates index, codes columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    values = rng.standard_normal((n_days, n_stocks))
    return pd.DataFrame(values, index=dates, columns=codes)


def test_winsorize_clips_to_quantile():
    """Values outside [lo quantile, hi quantile] are clipped per day."""
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel(n_days=3, n_stocks=100, seed=1)
    out = winsorize_panel(df, 0.05, 0.95)
    for d in df.index:
        row = df.loc[d]
        lo_q = row.quantile(0.05)
        hi_q = row.quantile(0.95)
        assert out.loc[d].min() >= lo_q - 1e-9
        assert out.loc[d].max() <= hi_q + 1e-9


def test_winsorize_all_nan_row_passthrough():
    """A day with all-NaN cross-section is returned unchanged."""
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel(n_days=3, n_stocks=10, seed=2)
    df.iloc[1] = np.nan
    out = winsorize_panel(df, 0.01, 0.99)
    assert out.iloc[1].isna().all()
    assert out.shape == df.shape


def test_winsorize_invalid_bounds_raises():
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel()
    with pytest.raises(ValueError):
        winsorize_panel(df, 0.99, 0.01)
    with pytest.raises(ValueError):
        winsorize_panel(df, 0.5, 0.5)


def test_winsorize_preserves_index_columns():
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel()
    out = winsorize_panel(df, 0.01, 0.99)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape
