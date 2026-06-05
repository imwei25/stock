"""Smoke tests for cross-sectional market breadth factors.

注意 (spec §6.1.2): 涨停股算作上涨股、>MA20 股,**与 mask config 无关**。
本测试断言这一不变性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.cross_sec_breadth as _csb  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(41)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 5)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(1e6, index=dates, columns=codes)
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_breadth_above_ma_20_broadcast(panel):
    f = make_factor("breadth_above_ma_20")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape
    # 每一行的 5 列值必须相同(标量广播)
    valid = out.iloc[25:]
    for _, row in valid.iterrows():
        assert row.std() < 1e-9


def test_breadth_above_ma_in_unit(panel):
    """宽度 ∈ [0, 1]。"""
    f = make_factor("breadth_above_ma_20")
    out = f.compute(panel)
    valid = out.iloc[25:].to_numpy()
    valid = valid[~np.isnan(valid)]
    assert ((valid >= 0) & (valid <= 1)).all()


def test_breadth_advance_counts_up_stocks(panel):
    """涨股比例 = (上涨股数 / 总股数)。"""
    f = make_factor("breadth_advance")
    out = f.compute(panel)
    # 构造已知场景:第 50 行所有 5 只票都涨 → breadth=1.0
    expected = (panel["close"].pct_change(fill_method=None) > 0).mean(axis=1)
    actual = out.iloc[:, 0]
    pd.testing.assert_series_equal(
        actual, expected, check_names=False, check_exact=False, rtol=1e-9
    )


def test_breadth_limit_up_includes_limit_up_stocks():
    """spec §6.1.2: 涨停股必须算进涨停股占比分子。"""
    f = make_factor("breadth_limit_up")
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    close = pd.DataFrame({
        "A": [100, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0],
        "B": [100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9],
    }, index=dates)
    panel = {
        "close": close, "high": close + 1, "low": close - 1,
        "open": close.shift(1).fillna(close.iloc[0]),
        "volume": pd.DataFrame(1e6, index=dates, columns=close.columns),
    }
    out = f.compute(panel)
    # 第 1 天 A 涨 10%,触涨停,占比 = 1/2 = 0.5
    assert out.iloc[1, 0] == pytest.approx(0.5)


def test_no_look_ahead(panel):
    f = make_factor("breadth_above_ma_20")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("breadth_above_ma", "breadth_advance",
                 "breadth_limit_up", "breadth_dispersion"):
        spec = get_spec(name)
        assert spec is not None
