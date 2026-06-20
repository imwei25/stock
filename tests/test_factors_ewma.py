"""Smoke tests for EWMA-smoothed factors (halflife parameterized)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.ewma as _ewma  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_ewma_momentum_hl10_registered(panel):
    f = make_factor("ewma_momentum_hl10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape
    assert out.iloc[20:].notna().all().all()  # 20 期后必有值


def test_ewma_vol_hl10_positive(panel):
    f = make_factor("ewma_vol_hl10")
    out = f.compute(panel)
    valid = out.iloc[30:]
    assert (valid >= 0).all().all()


def test_ewma_close_dev_matches_formula(panel):
    f = make_factor("ewma_close_dev_hl10")
    out = f.compute(panel)
    c = panel["close"]
    ema = c.ewm(halflife=10).mean()
    std = c.ewm(halflife=10).std()
    expected = (c - ema) / std
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-9)


def test_ewma_no_look_ahead(panel):
    f = make_factor("ewma_momentum_hl10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("ewma_momentum", "ewma_vol", "ewma_turnover_z",
                 "ewma_close_dev", "ewma_volume_ratio"):
        spec = get_spec(name)
        assert spec is not None
