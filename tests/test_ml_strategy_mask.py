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


def test_forward_return_panel_no_mask_unchanged():
    from stockpool.ml.dataset import forward_return_panel
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]})
    y_a = forward_return_panel(close, horizon=2)
    y_b = forward_return_panel(close, horizon=2, mask=None)
    pd.testing.assert_frame_equal(y_a, y_b)


def test_forward_return_panel_bidirectional_mask():
    from stockpool.ml.dataset import forward_return_panel
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]})
    mask = pd.DataFrame({"A": [True, True, False, True, True]})
    y = forward_return_panel(close, horizon=2, mask=mask)
    # t=0: mask[0]=T ∧ mask[2]=F → NaN
    # t=1: mask[1]=T ∧ mask[3]=T → (13-11)/11
    # t=2: mask[2]=F → NaN
    assert np.isnan(y["A"].iloc[0])
    assert y["A"].iloc[1] == pytest.approx(2.0 / 11.0)
    assert np.isnan(y["A"].iloc[2])
