"""Smoke tests for long-horizon factors (long-term reversal / 52w-high)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.long_horizon as _lh  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(9)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((300, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    return {"close": close, "high": close + 1, "low": close - 1,
            "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=codes)}


def test_long_term_reversal_value():
    """[t−N, t−21] 区间收益 = close[t−21]/close[t−N] − 1,精确可验。"""
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    close = pd.DataFrame(
        {"A": np.linspace(100.0, 400.0, 300)}, index=dates,
    )
    panel = {"close": close, "high": close, "low": close,
             "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=["A"])}
    out = make_factor("long_term_reversal_240").compute(panel)
    t = 260
    expected = close["A"].iloc[t - 21] / close["A"].iloc[t - 240] - 1.0
    assert out["A"].iloc[t] == pytest.approx(expected)


def test_high_proximity_le_one_at_high():
    """close ≤ 近 N 日最高价 → 比值 ≤ 1;创新高日 ≈ 1。"""
    out = make_factor("high_proximity_240").compute(_panel_dict())
    valid = out.iloc[241:]
    # high = close + 1,所以严格 < 1
    assert (valid <= 1.0 + 1e-9).all().all()


def _panel_dict():
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    rng = np.random.default_rng(2)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((300, 2)).cumsum(axis=0),
        index=dates, columns=["A", "B"],
    )
    return {"close": close, "high": close + 1, "low": close - 1,
            "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=["A", "B"])}


def test_no_look_ahead(panel):
    for name in ("long_term_reversal_240", "high_proximity_240"):
        f = make_factor(name)
        full = f.compute(panel)
        trunc = {k: v.iloc[:280] for k, v in panel.items()}
        short = f.compute(trunc)
        pd.testing.assert_frame_equal(full.iloc[:280], short, check_exact=False, rtol=1e-9)


def test_specs_registered():
    for name in ("long_term_reversal", "high_proximity"):
        assert get_spec(name) is not None
