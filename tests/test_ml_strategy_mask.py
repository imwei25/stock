"""Tests for tradability mask integration in ml/dataset pipeline."""
import numpy as np
import pandas as pd
import pytest


def _make_panel(close_dict):
    codes = list(close_dict.keys())
    n = len(next(iter(close_dict.values())))
    idx = pd.date_range("2024-01-01", periods=n)
    close = pd.DataFrame(close_dict, index=idx)
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": pd.DataFrame({c: [1000.0] * n for c in codes}, index=idx),
    }


def test_compute_factor_panel_no_mask_unchanged():
    from stockpool.ml.dataset import compute_factor_panel
    panel = _make_panel({"600000": list(np.linspace(10, 11, 30))})
    out_a = compute_factor_panel(panel, ["momentum_5"])
    out_b = compute_factor_panel(panel, ["momentum_5"], mask=None)
    pd.testing.assert_frame_equal(out_a["momentum_5"], out_b["momentum_5"])


def test_compute_factor_panel_with_mask_changes_values():
    from stockpool.ml.dataset import compute_factor_panel
    panel = _make_panel({"600000": list(np.linspace(10, 11, 30))})
    mask = pd.DataFrame(True, index=panel["close"].index, columns=panel["close"].columns)
    mask.iloc[5, 0] = False
    out = compute_factor_panel(panel, ["momentum_5"], mask=mask)
    assert np.isnan(out["momentum_5"].iloc[5, 0])
