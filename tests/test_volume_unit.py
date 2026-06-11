"""P1-6: volume 单位跨源统一为"股"的回归测试。

背景:mootdx bars / akshare stock_zh_a_hist 的成交量原始单位是"手"(1 手 =
100 股),baostock 是"股"。修复后个股路径在数据层统一放大为"股",
消费端(recommend_pool / portfolio eligibility)按 ``volume * close`` 计算
成交额,不再硬编码 ×100;缓存 marker 升级 schema 版本(v2)使旧缓存自动失效。

指数/板块路径 volume 跨源单位不保证一致(只做比值类用途),保持原样。
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockpool.config import PortfolioEligibilityConfig
from stockpool.data_sources import mootdx_backend
from stockpool.fetcher import (
    _fetch_from_akshare,
    _fetch_index_from_akshare,
    _marker_value,
    check_source_change,
    update_source_marker,
)
from stockpool.portfolio.eligibility import EligibilityFilter
from stockpool.recommend_pool import _apply_funnel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _raw_bars(volumes: list[float], start: str = "2026-01-05") -> pd.DataFrame:
    """mootdx bars 原始结构(volume 单位 = 手)。"""
    n = len(volumes)
    dates = pd.bdate_range(start, periods=n)
    closes = np.linspace(10.0, 10.0 + 0.1 * (n - 1), n)
    return pd.DataFrame({
        "date": dates,
        "open": closes - 0.1,
        "high": closes + 0.2,
        "low": closes - 0.2,
        "close": closes,
        "volume": np.asarray(volumes, dtype=float),
    })


def _make_akshare_df(periods: int, volume_lots: float = 1_000_000) -> pd.DataFrame:
    """akshare stock_zh_a_hist 原始结构(成交量单位 = 手)。"""
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": np.linspace(10, 11, periods),
        "收盘": np.linspace(10.1, 11.1, periods),
        "最高": np.linspace(10.3, 11.3, periods),
        "最低": np.linspace(9.9, 10.9, periods),
        "成交量": np.full(periods, volume_lots),
        "成交额": np.full(periods, 1e7),
    })


def _ohlcv_daily(n: int, close: float, volume_shares: float) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=n, freq="B")
    c = np.full(n, close)
    return pd.DataFrame({
        "date": dates,
        "open": c, "high": c + 0.1, "low": c - 0.1, "close": c,
        "volume": np.full(n, volume_shares),
    })


# ---------------------------------------------------------------------------
# mootdx 个股路径:volume 手 → 股
# ---------------------------------------------------------------------------

def test_mootdx_normalize_scales_volume_to_shares():
    out = mootdx_backend._normalize(_raw_bars([1000.0, 2000.0, 3000.0]))
    assert np.allclose(out["volume"], [100_000.0, 200_000.0, 300_000.0]), (
        "mootdx volume 原始单位是手,应 ×100 统一为股"
    )


def test_mootdx_normalize_placeholder_check_before_scaling():
    """未开盘占位行判断(volume < 1)必须在放大前以"手"为单位执行。

    末根 bar volume=0.5 手且 OHLC 同价 → 是占位行,应被丢弃;
    若实现错误地先 ×100 再判断(50 股 ≥ 1),该行会被错误保留。
    """
    raw = _raw_bars([1000.0, 2000.0, 0.5])
    raw.loc[2, ["open", "high", "low", "close"]] = 9.99  # OHLC 同价
    out = mootdx_backend._normalize(raw)
    assert len(out) == 2, "占位行(0.5 手 + OHLC 同价)应被丢弃"
    assert np.allclose(out["volume"], [100_000.0, 200_000.0])


def test_mootdx_fetch_stock_volume_in_shares():
    raw = _raw_bars([1000.0] * 4)
    empty_events = pd.DataFrame(columns=mootdx_backend._XDXR_EVENT_COLS)
    with patch.object(mootdx_backend, "_call_with_retry", return_value=raw), \
         patch.object(mootdx_backend, "_fetch_xdxr", return_value=empty_events):
        out = mootdx_backend.fetch_stock("000001")
    assert np.allclose(out["volume"], 100_000.0)


def test_mootdx_index_and_sector_volume_unscaled():
    """指数/板块路径 volume 跨源单位不保证一致(仅做比值用途),保持原样。"""
    raw = _raw_bars([1000.0, 2000.0, 3000.0])
    with patch.object(mootdx_backend, "_call_with_retry", return_value=raw.copy()):
        idx = mootdx_backend.fetch_index("sh000001")
        sec = mootdx_backend.fetch_sector("880305")
    assert np.allclose(idx["volume"], [1000.0, 2000.0, 3000.0])
    assert np.allclose(sec["volume"], [1000.0, 2000.0, 3000.0])


# ---------------------------------------------------------------------------
# akshare 个股路径:volume 手 → 股;指数路径保持原样
# ---------------------------------------------------------------------------

def test_akshare_stock_volume_scaled_to_shares():
    fake = _make_akshare_df(5, volume_lots=1_000_000)
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        out = _fetch_from_akshare("605589")
    assert np.allclose(out["volume"], 100_000_000.0), (
        "akshare 成交量原始单位是手,应 ×100 统一为股"
    )


def test_akshare_index_volume_unscaled():
    dates = pd.date_range("2026-01-02", periods=3, freq="B")
    fake = pd.DataFrame({
        "date": dates, "open": [10.0] * 3, "high": [10.2] * 3,
        "low": [9.8] * 3, "close": [10.1] * 3, "volume": [1234.0] * 3,
    })
    with patch("stockpool.fetcher.ak.stock_zh_index_daily", return_value=fake):
        out = _fetch_index_from_akshare("sh000001")
    assert np.allclose(out["volume"], 1234.0)


# ---------------------------------------------------------------------------
# 消费端:amount = volume * close(volume 已是股,不再 ×100)
# ---------------------------------------------------------------------------

def test_recommend_pool_funnel_amount_uses_shares():
    """阈值 5e7:close=10,volume=5_000_100 股 → 5.0001e7 过;4_999_900 → 拒。"""
    universe = {
        "PASS": _ohlcv_daily(30, 10.0, 5_000_100),
        "FAIL": _ohlcv_daily(30, 10.0, 4_999_900),
    }
    out = _apply_funnel(universe, {}, min_avg_amount_20d=5e7)
    assert set(out) == {"PASS"}


def test_eligibility_amount_uses_shares():
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=5e7, exclude_st=False, min_history_bars=1,
    )
    panel = {
        "PASS": _ohlcv_daily(30, 10.0, 5_000_100),
        "FAIL": _ohlcv_daily(30, 10.0, 4_999_900),
    }
    out = EligibilityFilter(cfg).eligible(pd.Timestamp("2026-12-31"), panel)
    assert out == {"PASS"}


# ---------------------------------------------------------------------------
# 缓存 schema 升级:marker 含 v2,旧口径缓存自动失效
# ---------------------------------------------------------------------------

def test_marker_value_includes_schema_version():
    assert _marker_value("mootdx") == "mootdx:hfq:v2"
    assert _marker_value("baostock") == "baostock:hfq:v2"


def test_legacy_v1_marker_triggers_refresh(tmp_path):
    """旧 marker('akshare:hfq',无 schema 版本)= volume 仍是手 → 必须全量重拉。"""
    (tmp_path / ".data_source").write_text("akshare:hfq", encoding="utf-8")
    assert check_source_change(tmp_path, "akshare") is True


def test_v2_marker_roundtrip_no_refresh(tmp_path):
    update_source_marker(tmp_path, "mootdx")
    assert check_source_change(tmp_path, "mootdx") is False
    assert (tmp_path / ".data_source").read_text(
        encoding="utf-8").strip() == "mootdx:hfq:v2"
