"""Tests for stockpool.panel mask functions (tradability mask for factor input)."""
import numpy as np
import pandas as pd
import pytest


def test_limit_threshold_main_board():
    from stockpool.panel import _limit_threshold
    assert _limit_threshold("600000") == 0.098
    assert _limit_threshold("601398") == 0.098
    assert _limit_threshold("603986") == 0.098
    assert _limit_threshold("605589") == 0.098
    assert _limit_threshold("000001") == 0.098
    assert _limit_threshold("002001") == 0.098
    assert _limit_threshold("003001") == 0.098


def test_limit_threshold_chinext_star():
    from stockpool.panel import _limit_threshold
    assert _limit_threshold("300001") == 0.198
    assert _limit_threshold("301001") == 0.198
    assert _limit_threshold("688001") == 0.198


def test_limit_threshold_bse():
    from stockpool.panel import _limit_threshold
    assert _limit_threshold("830001") == 0.298
    assert _limit_threshold("870001") == 0.298


def test_listing_mask_mature_stock_all_true():
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=300)
    close = pd.DataFrame({"600000": np.arange(300, dtype=float)}, index=idx)
    mask = _listing_mask(close, min_days=252)
    assert mask["600000"].all()


def test_listing_mask_new_listing_blocks_first_n_days():
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=400)
    close = pd.DataFrame({
        "300001": [np.nan] * 50 + list(range(350)),
    }, index=idx)
    mask = _listing_mask(close, min_days=252)
    assert not mask["300001"].iloc[50:50+252].any()
    assert mask["300001"].iloc[50+252:].all()


def test_listing_mask_all_nan_stock_all_false():
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=100)
    close = pd.DataFrame({"600000": [np.nan] * 100}, index=idx)
    mask = _listing_mask(close, min_days=252)
    assert not mask["600000"].any()


def _make_panel(close_dict, volume_dict=None):
    codes = list(close_dict.keys())
    idx = pd.date_range("2024-01-01", periods=len(next(iter(close_dict.values()))))
    close = pd.DataFrame(close_dict, index=idx)
    if volume_dict is None:
        volume = pd.DataFrame({c: [1000.0] * len(idx) for c in codes}, index=idx)
    else:
        volume = pd.DataFrame(volume_dict, index=idx)
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": volume,
    }


def test_compute_mask_main_board_limit_up():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {
        "600000": [10.0, 10.99, 11.0, 11.01],
        "300001": [10.0, 10.99, 11.0, 11.01],
    }
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.loc[panel["close"].index[1], "600000"] == False
    assert mask.loc[panel["close"].index[1], "300001"] == True


def test_compute_mask_suspension_volume_zero():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {"600000": [10.0, 10.05, 10.1, 10.15]}
    volume_dict = {"600000": [1000.0, 0.0, 1000.0, 1000.0]}
    panel = _make_panel(close_dict, volume_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.loc[panel["close"].index[1], "600000"] == False


def test_compute_mask_three_conditions_intersect():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {"600000": [10.0, 10.05, 10.10, 10.15]}
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.iloc[0, 0] == False
    assert mask.iloc[1:, 0].all()


def test_compute_mask_shape_matches_close():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {f"600{i:03d}": [10.0 + i * 0.01] * 50 for i in range(5)}
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.shape == panel["close"].shape
    assert mask.index.equals(panel["close"].index)
    assert mask.columns.equals(panel["close"].columns)
