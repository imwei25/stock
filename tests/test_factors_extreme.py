"""Smoke tests for the lottery / extreme-return factors (MAX / MIN)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.extreme as _ex  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((40, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    return {"close": close, "high": close + 1, "low": close - 1,
            "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=codes)}


def test_max_ge_min(panel):
    mx = make_factor("max_ret_20").compute(panel)
    mn = make_factor("min_ret_20").compute(panel)
    valid = mx.iloc[25:]
    assert (valid >= mn.iloc[25:]).all().all()


def test_max_captures_spike(panel):
    """注入一个大单日涨幅,MAX 应抓到它。"""
    p = {k: v.copy() for k, v in panel.items()}
    p["close"].iloc[25, 0] = p["close"].iloc[24, 0] * 1.095  # +9.5%
    out = make_factor("max_ret_20").compute(p)
    # 第 25 行窗口(rows 6..25,min_periods 已满)应抓到这根 +9.5% 的尖峰
    assert out["A"].iloc[25] == pytest.approx(0.095, abs=1e-3)


def test_no_look_ahead(panel):
    f = make_factor("max_ret_20")
    full = f.compute(panel)
    trunc = {k: v.iloc[:30] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(full.iloc[:30], short, check_exact=False, rtol=1e-9)


def test_specs_registered():
    for name in ("max_ret", "min_ret"):
        assert get_spec(name) is not None
