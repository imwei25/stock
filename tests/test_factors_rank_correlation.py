"""Smoke tests for rank-correlation family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.rank_correlation as _rc  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B", "C", "D"]
    rng = np.random.default_rng(31)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 4)).cumsum(axis=0),
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


def test_corr_pv_20_in_unit(panel):
    """秩相关 ∈ [-1, 1]。"""
    f = make_factor("corr_pv_20")
    out = f.compute(panel)
    valid = out.iloc[25:].to_numpy()
    valid = valid[~np.isnan(valid)]
    assert (valid >= -1.0 - 1e-9).all()
    assert (valid <= 1.0 + 1e-9).all()


def test_corr_high_low_20_runs(panel):
    f = make_factor("corr_high_low_20")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_corr_mom_vol_10_runs(panel):
    f = make_factor("corr_mom_vol_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_no_look_ahead(panel):
    f = make_factor("corr_pv_20")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("corr_pv", "corr_high_low", "corr_close_vwap",
                 "corr_mom_vol"):
        spec = get_spec(name)
        assert spec is not None
