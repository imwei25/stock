"""Smoke tests for direct rolling statistic factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 触发注册
import stockpool.factors.original_stats as _orig  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B", "C"]
    rng = np.random.default_rng(42)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 3)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    high = close + rng.uniform(0.1, 2.0, size=close.shape)
    low = close - rng.uniform(0.1, 2.0, size=close.shape)
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close, "high": high, "low": low,
            "open": close.shift(1).fillna(close.iloc[0]), "volume": volume}


def test_close_std_20_registered(panel):
    f = make_factor("close_std_20")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape
    # 前 11 行(< 0.6 * 20 = 12 min_periods 实际是 .rolling(20) 用 std 默认)NaN
    assert out.iloc[:11].isna().all().all()
    assert out.iloc[30:].notna().any().any()


def test_close_skew_20_returns_finite(panel):
    f = make_factor("close_skew_20")
    out = f.compute(panel)
    valid = out.iloc[30:]
    assert np.isfinite(valid.to_numpy()).any()


def test_volume_std_60_normalized(panel):
    """volume_std 是变异系数 (std/mean), 应该 > 0。"""
    f = make_factor("volume_std_60")
    out = f.compute(panel)
    last_row = out.iloc[-1].dropna()
    assert (last_row > 0).all()


def test_range_std_d_matches_formula(panel):
    """range_std_20 应该等于 (high-low).rolling(20).std() / close 数值。"""
    f = make_factor("range_std_20")
    out = f.compute(panel)
    expected = (panel["high"] - panel["low"]).rolling(20).std() / panel["close"]
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-9)


def test_no_look_ahead_truncation_close_std_20(panel):
    """截断后 panel 算因子,前 N 行应与全 panel 算的前 N 行一致。"""
    f = make_factor("close_std_20")
    full = f.compute(panel)
    truncated = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(truncated)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    """注册表应该有这些 base name。"""
    for name in ("close_std", "close_skew", "close_kurt",
                 "volume_skew", "volume_kurt",
                 "range_std", "volume_std"):
        spec = get_spec(name)
        assert spec is not None
