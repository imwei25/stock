"""Smoke tests for the Size (log_mcap) factor."""
from __future__ import annotations

import numpy as np
import pandas as pd

import stockpool.factors.size as _size  # noqa: F401
from stockpool.factors import make_factor, get_spec
from stockpool.factors.context import set_mcap_panel


def _panel():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    codes = ["000001", "600000"]
    close = pd.DataFrame(100.0, index=dates, columns=codes)
    return {"close": close, "high": close + 1, "low": close - 1,
            "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=codes)}


def test_log_mcap_returns_injected_panel():
    panel = _panel()
    mcap = pd.DataFrame(
        [[20.0, 22.0]] * len(panel["close"]),
        index=panel["close"].index, columns=panel["close"].columns,
    )
    set_mcap_panel(mcap)
    try:
        out = make_factor("log_mcap").compute(panel)
        pd.testing.assert_frame_equal(out, mcap)
    finally:
        set_mcap_panel(None)


def test_log_mcap_none_degrades_to_nan():
    """mcap 未注入 → 全 NaN(不 fail loud),避免带崩 analyze 全因子计算。"""
    set_mcap_panel(None)
    out = make_factor("log_mcap").compute(_panel())
    assert out.isna().all().all()


def test_log_mcap_reindexes_to_panel_grid():
    """mcap 覆盖范围与 panel 不同 → 按 close 网格对齐,缺失补 NaN。"""
    panel = _panel()
    # mcap 只覆盖一只票 + 少几天
    mcap = pd.DataFrame(
        21.0, index=panel["close"].index[:10], columns=["000001"],
    )
    set_mcap_panel(mcap)
    try:
        out = make_factor("log_mcap").compute(panel)
        assert list(out.columns) == list(panel["close"].columns)
        assert out["600000"].isna().all()       # 不在 mcap → NaN
        assert out["000001"].iloc[:10].notna().all()
        assert out["000001"].iloc[10:].isna().all()
    finally:
        set_mcap_panel(None)


def test_spec_registered():
    assert get_spec("log_mcap") is not None
