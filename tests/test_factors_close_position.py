"""Smoke tests for close-in-range position family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.close_position as _cp  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(99)
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


def test_close_pos_5_in_unit(panel):
    """close_pos_d ∈ [0, 1] (close 永远在 [low, high] 内)。"""
    f = make_factor("close_pos_5")
    out = f.compute(panel)
    valid = out.iloc[10:].to_numpy()
    valid = valid[~np.isnan(valid)]
    assert (valid >= -1e-9).all()  # 允许浮点误差
    assert (valid <= 1.0 + 1e-9).all()


def test_close_pos_zero_range_returns_nan():
    """high==low (涨停封板) 时 close_pos 应 NaN。"""
    f = make_factor("close_pos_5")
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    # B 列全 high==low==close
    close = pd.DataFrame({"A": np.arange(20) + 100.0,
                          "B": np.full(20, 100.0)},
                         index=dates)
    panel = {
        "close": close,
        "high": close.copy(), "low": close.copy(),
        "open": close.shift(1).fillna(close.iloc[0]),
        "volume": pd.DataFrame(1.0, index=dates, columns=close.columns),
    }
    panel["high"]["A"] = close["A"] + 1
    panel["low"]["A"] = close["A"] - 1
    # A 有 range, B 没有
    out = f.compute(panel)
    assert out["B"].iloc[10:].isna().all()  # 全 NaN
    assert out["A"].iloc[10:].notna().any()


def test_close_pos_cum_centered(panel):
    """close_pos_cum 是 (pos - 0.5) rolling sum,可正可负。"""
    f = make_factor("close_pos_cum_10")
    out = f.compute(panel)
    assert out.iloc[15:].notna().all().all()


def test_no_look_ahead(panel):
    f = make_factor("close_pos_10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("close_pos", "close_pos_cum", "close_pos_ema"):
        spec = get_spec(name)
        assert spec is not None
