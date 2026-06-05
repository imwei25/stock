"""Smoke tests for second-order difference (acceleration) factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.acceleration as _acc  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(11)
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


def test_mom_accel_5_matches_formula(panel):
    f = make_factor("mom_accel_5")
    out = f.compute(panel)
    mom = panel["close"].pct_change(5, fill_method=None)
    expected = mom - mom.shift(5)
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-9)


def test_vol_accel_5_runs(panel):
    f = make_factor("vol_accel_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_turnover_accel_5_runs(panel):
    f = make_factor("turnover_accel_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_no_look_ahead(panel):
    f = make_factor("mom_accel_5")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("mom_accel", "vol_accel", "turnover_accel"):
        spec = get_spec(name)
        assert spec is not None
