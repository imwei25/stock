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


def test_build_factor_panel_no_mask_config_unchanged():
    from stockpool.strategy_factory import build_factor_panel
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    out_a = build_factor_panel(["momentum_5"], pool_data)
    out_b = build_factor_panel(["momentum_5"], pool_data, mask_config=None)
    pd.testing.assert_frame_equal(out_a["momentum_5"], out_b["momentum_5"])


def test_build_factor_panel_mask_disabled_equivalent_to_no_config():
    from stockpool.strategy_factory import build_factor_panel
    from stockpool.config import MaskConfig
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    out_a = build_factor_panel(["momentum_5"], pool_data, mask_config=MaskConfig(enabled=False))
    out_b = build_factor_panel(["momentum_5"], pool_data, mask_config=None)
    pd.testing.assert_frame_equal(out_a["momentum_5"], out_b["momentum_5"])


def test_build_factor_panel_mask_enabled_changes_output():
    from stockpool.strategy_factory import build_factor_panel
    from stockpool.config import MaskConfig
    n = 30
    closes = np.linspace(10, 11, n).copy()
    closes[10] = closes[9] * 1.099
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": [1000.0] * n,
    })
    pool_data = {"600000": df}
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    out = build_factor_panel(["momentum_5"], pool_data, mask_config=cfg)
    assert np.isnan(out["momentum_5"].iloc[10, 0])


def test_build_panel_no_mask_unchanged():
    from stockpool.ml.dataset import build_panel
    n = 30
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10, 11, n),
        "volume": [1000.0] * n,
    })
    stocks_data = {"600000": df}
    X_a, y_a = build_panel(stocks_data, ["momentum_5"], horizon=2)
    X_b, y_b = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=None)
    pd.testing.assert_frame_equal(X_a, X_b)
    pd.testing.assert_series_equal(y_a, y_b)


def test_build_panel_mask_drops_samples():
    from stockpool.ml.dataset import build_panel
    from stockpool.config import MaskConfig
    n = 30
    closes = np.linspace(10, 11, n).copy()
    closes[15] = closes[14] * 1.099
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": [1000.0] * n,
    })
    stocks_data = {"600000": df}
    cfg_no = MaskConfig(enabled=False)
    cfg_yes = MaskConfig(enabled=True, min_listing_days=0)
    _, y_no = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=cfg_no)
    _, y_yes = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=cfg_yes)
    assert len(y_yes) < len(y_no)
