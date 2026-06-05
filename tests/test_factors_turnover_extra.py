"""Smoke tests for short-window turnover / amount factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.turnover_extra as _te  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(5)
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


def test_turnover_z_5_runs(panel):
    f = make_factor("turnover_z_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_turnover_z_handles_volume_zero():
    """停牌日 volume=0 必须 NaN,不能 -inf 污染。"""
    f = make_factor("turnover_z_5")
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    volume = pd.DataFrame(1e6, index=dates, columns=["A"])
    volume.iloc[15] = 0.0  # 停牌日
    panel = {
        "close": pd.DataFrame(100.0, index=dates, columns=["A"]),
        "high": pd.DataFrame(101.0, index=dates, columns=["A"]),
        "low": pd.DataFrame(99.0, index=dates, columns=["A"]),
        "open": pd.DataFrame(100.0, index=dates, columns=["A"]),
        "volume": volume,
    }
    out = f.compute(panel)
    # 停牌日及附近不应有 -inf
    assert not np.isinf(out.to_numpy()).any()


def test_amount_z_10_runs(panel):
    f = make_factor("amount_z_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_volume_ratio_short_window(panel):
    f = make_factor("volume_ratio_short_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_no_look_ahead(panel):
    f = make_factor("turnover_z_10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("turnover_z", "amount_z", "volume_ratio_short"):
        spec = get_spec(name)
        assert spec is not None
