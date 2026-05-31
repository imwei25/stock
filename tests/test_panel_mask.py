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
