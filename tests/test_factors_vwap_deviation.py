"""Smoke tests for VWAP deviation family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.vwap_deviation as _vd  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(13)
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


def test_vwap_dev_5_returns_finite(panel):
    f = make_factor("vwap_dev_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape
    assert np.isfinite(out.iloc[10:].to_numpy()).any()


def test_vwap_dev_d_zero_centered(panel):
    """理论上 (close - vwap)/vwap 在长时间上应该接近 0(无系统性偏离)。"""
    f = make_factor("vwap_dev_20")
    out = f.compute(panel)
    # 不严格要求 0,但绝对均值应远小于 1
    assert out.iloc[30:].abs().mean().mean() < 0.5


def test_vwap_weighted_mom_10_runs(panel):
    f = make_factor("vwap_weighted_mom_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_vwap_above_ratio_in_unit(panel):
    """vwap_above_ratio_d 应 ∈ [0, 1]。"""
    f = make_factor("vwap_above_ratio_10")
    out = f.compute(panel)
    valid = out.iloc[15:].to_numpy()
    valid = valid[~np.isnan(valid)]
    assert ((valid >= 0) & (valid <= 1)).all()


def test_no_look_ahead(panel):
    f = make_factor("vwap_dev_10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("vwap_dev", "vwap_weighted_mom", "vwap_above_ratio"):
        spec = get_spec(name)
        assert spec is not None
