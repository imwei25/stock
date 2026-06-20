"""Smoke tests for single-stock volatility / range factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.single_stock_vol as _sv  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(17)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    high = close + rng.uniform(0.5, 2.0, size=close.shape)
    low = close - rng.uniform(0.5, 2.0, size=close.shape)
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close, "high": high, "low": low,
            "open": close.shift(1).fillna(close.iloc[0]), "volume": volume}


def test_atr_14_positive(panel):
    f = make_factor("atr_14")
    out = f.compute(panel)
    valid = out.iloc[15:]
    assert (valid > 0).all().all()


def test_cci_20_runs(panel):
    f = make_factor("cci_20")
    out = f.compute(panel)
    valid = out.iloc[25:]
    assert np.isfinite(valid.to_numpy()).any()


def test_amp_5_positive(panel):
    """振幅 = (high-low) / close 的 N 日均值,应 > 0。"""
    f = make_factor("amp_5")
    out = f.compute(panel)
    valid = out.iloc[10:]
    assert (valid > 0).all().all()


def test_park_vol_20_positive(panel):
    """Parkinson vol 必非负。"""
    f = make_factor("park_vol_20")
    out = f.compute(panel)
    valid = out.iloc[25:]
    assert (valid >= 0).all().all()


def test_no_look_ahead(panel):
    f = make_factor("atr_14")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("atr", "cci", "amp", "park_vol", "gk_vol"):
        spec = get_spec(name)
        assert spec is not None
