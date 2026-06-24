"""Smoke tests for return-distribution moment factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.return_moments as _rm  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(5)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((60, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    return {"close": close, "high": close + 1, "low": close - 1,
            "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=codes)}


def test_downside_vol_nonnegative(panel):
    out = make_factor("downside_vol_20").compute(panel)
    valid = out.iloc[25:]
    assert (valid >= 0).all().all()


def test_downside_vol_zero_when_no_drops():
    """全程上涨 → 无负收益 → 下行波动为 0。"""
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    close = pd.DataFrame(
        {"A": np.arange(100.0, 140.0), "B": np.arange(200.0, 240.0)}, index=dates,
    )
    panel = {"close": close, "high": close + 1, "low": close - 1,
             "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=close.columns)}
    out = make_factor("downside_vol_20").compute(panel)
    assert np.allclose(out.iloc[25:].to_numpy(), 0.0, atol=1e-12)


def test_skew_kurt_finite(panel):
    sk = make_factor("ret_skew_20").compute(panel).iloc[25:]
    ku = make_factor("ret_kurt_20").compute(panel).iloc[25:]
    assert np.isfinite(sk.to_numpy()).all()
    assert np.isfinite(ku.to_numpy()).all()


def test_no_look_ahead(panel):
    for name in ("ret_skew_20", "ret_kurt_20", "downside_vol_20"):
        f = make_factor(name)
        full = f.compute(panel)
        trunc = {k: v.iloc[:40] for k, v in panel.items()}
        short = f.compute(trunc)
        pd.testing.assert_frame_equal(full.iloc[:40], short, check_exact=False, rtol=1e-9)


def test_specs_registered():
    for name in ("ret_skew", "ret_kurt", "downside_vol"):
        assert get_spec(name) is not None
