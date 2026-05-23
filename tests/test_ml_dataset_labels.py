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


def test_forward_return_panel_label_type_vol_adjusted_not_implemented():
    close = _close_panel()
    with pytest.raises(NotImplementedError, match="vol_adjusted"):
        forward_return_panel(close, horizon=3, label_type="vol_adjusted")


def test_forward_return_panel_label_type_cross_sec_rank_not_implemented():
    close = _close_panel()
    with pytest.raises(NotImplementedError, match="cross_sec_rank"):
        forward_return_panel(close, horizon=3, label_type="cross_sec_rank")


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
    df = _close_series()
    with pytest.raises(NotImplementedError, match="vol_adjusted"):
        forward_return(df, horizon=3, label_type="vol_adjusted")
    with pytest.raises(NotImplementedError, match="cross_sec_rank"):
        forward_return(df, horizon=3, label_type="cross_sec_rank")
