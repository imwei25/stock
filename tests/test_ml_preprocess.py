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


def test_cs_zscore_mean_zero_std_one():
    """After per-day cs zscore, each row has μ ≈ 0 and σ ≈ 1."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=3, n_stocks=50, seed=3)
    out = cs_zscore_panel(df)
    for d in df.index:
        row = out.loc[d].dropna()
        assert abs(row.mean()) < 1e-9
        assert abs(row.std(ddof=0) - 1.0) < 1e-9


def test_cs_zscore_constant_row_returns_zero():
    """A day where every stock has identical value → returns zeros (σ < 1e-12)."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=3, n_stocks=10, seed=4)
    df.iloc[1] = 7.5  # constant row
    out = cs_zscore_panel(df)
    assert (out.iloc[1] == 0.0).all()


def test_cs_zscore_handles_nan():
    """Partial NaN row: zscore computed on non-NaN values, NaN positions stay NaN."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=2, n_stocks=10, seed=5)
    df.iloc[0, :3] = np.nan
    out = cs_zscore_panel(df)
    assert out.iloc[0, :3].isna().all()
    valid = out.iloc[0, 3:]
    assert abs(valid.mean()) < 1e-9
    assert abs(valid.std(ddof=0) - 1.0) < 1e-9


def test_cs_zscore_preserves_index_columns():
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel()
    out = cs_zscore_panel(df)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape


def test_industry_neutralize_within_group_mean_zero():
    """Per-day, within each industry group, mean of values is 0."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=3, n_stocks=12, seed=6)
    # 3 industries × 4 stocks each.
    sector_map = {f"S{i:03d}": f"ind_{i // 4}" for i in range(12)}
    out = industry_neutralize_panel(df, sector_map)
    for d in df.index:
        row = out.loc[d]
        for ind in {"ind_0", "ind_1", "ind_2"}:
            members = [c for c, s in sector_map.items() if s == ind]
            assert abs(row[members].mean()) < 1e-9


def test_industry_neutralize_unknown_code_bucket():
    """Codes not in sector_map go to '_unknown_' bucket and are demeaned together."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=6, seed=7)
    sector_map = {"S000": "A", "S001": "A"}  # only 2 of 6 mapped
    out = industry_neutralize_panel(df, sector_map)
    unknown_cols = [f"S{i:03d}" for i in range(2, 6)]
    for d in df.index:
        assert abs(out.loc[d, unknown_cols].mean()) < 1e-9


def test_industry_neutralize_empty_sector_map_raises():
    """Empty sector_map raises — caller (apply_pipeline) wraps in try/skip."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel()
    with pytest.raises(ValueError):
        industry_neutralize_panel(df, {})


def test_industry_neutralize_preserves_index_columns():
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=8)
    sector_map = {f"S{i:03d}": f"ind_{i % 2}" for i in range(8)}
    out = industry_neutralize_panel(df, sector_map)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape
