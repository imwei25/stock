"""Smoke tests for the Amihud illiquidity factor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.liquidity as _liq  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(3)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((60, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close, "high": close + 1, "low": close - 1,
            "open": close, "volume": volume}


def test_amihud_nonnegative(panel):
    out = make_factor("amihud_20").compute(panel)
    valid = out.iloc[25:]
    assert (valid >= 0).all().all()
    assert np.isfinite(valid.to_numpy()).all()


def test_amihud_illiquid_higher(panel):
    """同收益下成交额更低的票,Amihud 更大。"""
    p = {k: v.copy() for k, v in panel.items()}
    p["volume"]["B"] = p["volume"]["B"] / 100.0  # B 成交额骤降 → 更不流动
    out = make_factor("amihud_20").compute(p)
    assert out["B"].iloc[30] > out["A"].iloc[30]


def test_zero_amount_not_inf(panel):
    """停牌 / 0 成交额日不产生 inf。"""
    p = {k: v.copy() for k, v in panel.items()}
    p["volume"].iloc[10, 0] = 0.0
    out = make_factor("amihud_20").compute(p)
    assert not np.isinf(out.to_numpy()[~np.isnan(out.to_numpy())]).any()


def test_no_look_ahead(panel):
    f = make_factor("amihud_20")
    full = f.compute(panel)
    trunc = {k: v.iloc[:40] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(full.iloc[:40], short, check_exact=False, rtol=1e-9)


def test_spec_registered():
    assert get_spec("amihud") is not None
