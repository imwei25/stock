"""stockpool.panel:Panel 数据结构 + build_panel_from_cache。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.panel import (
    OHLCV_FIELDS,
    assert_panel_valid,
    build_panel_from_cache,
    panel_shape,
)


def _write_stock_cache(tmp_path, code: str, T: int, base: float = 100.0):
    rng = np.random.default_rng(hash(code) & 0xFFFFFFFF)
    dates = pd.date_range("2024-01-02", periods=T, freq="B")
    close = base + np.cumsum(rng.standard_normal(T) * 0.5)
    df = pd.DataFrame({
        "date": dates,
        "open": close * 0.998,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.uniform(1e5, 1e6, T),
    })
    p = tmp_path / f"{code}_daily.parquet"
    df.to_parquet(p, index=False)
    return df


def test_build_panel_from_cache_aligns_codes(tmp_path):
    _write_stock_cache(tmp_path, "A", 40)
    _write_stock_cache(tmp_path, "B", 40, base=50.0)
    panel = build_panel_from_cache(["A", "B"], history_days=40, cache_dir=tmp_path)
    assert set(panel.keys()) == set(OHLCV_FIELDS)
    assert panel_shape(panel) == (40, 2)
    assert list(panel["close"].columns) == ["A", "B"]
    assert_panel_valid(panel)


def test_history_days_truncates(tmp_path):
    _write_stock_cache(tmp_path, "A", 100)
    panel = build_panel_from_cache(["A"], history_days=30, cache_dir=tmp_path)
    assert panel_shape(panel) == (30, 1)


def test_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_panel_from_cache(["NOPE"], history_days=10, cache_dir=tmp_path)


def test_misaligned_dates_get_unioned_and_filled(tmp_path):
    # A: 2024-01-02..05; B: 2024-01-04..09 → union has 6 days; A 没有 5..09
    df_a = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=4, freq="B"),
        "open": [1.0]*4, "high": [1.0]*4, "low": [1.0]*4,
        "close": [1.0, 2.0, 3.0, 4.0], "volume": [1.0]*4,
    })
    df_b = pd.DataFrame({
        "date": pd.date_range("2024-01-04", periods=6, freq="B"),
        "open": [2.0]*6, "high": [2.0]*6, "low": [2.0]*6,
        "close": [5.0]*6, "volume": [2.0]*6,
    })
    df_a.to_parquet(tmp_path / "A_daily.parquet", index=False)
    df_b.to_parquet(tmp_path / "B_daily.parquet", index=False)
    panel = build_panel_from_cache(["A", "B"], history_days=100, cache_dir=tmp_path)
    # A 在 B 的后段应该是 NaN
    last_two_a = panel["close"]["A"].tail(2)
    assert last_two_a.isna().all()
