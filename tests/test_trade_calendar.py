"""P3-1 交易日历 + P3-2 代理处理线程安全。"""
import os
from unittest.mock import patch

import pandas as pd

import stockpool.fetcher as fetcher


def _cal_df(dates):
    return pd.DataFrame({"trade_date": pd.to_datetime(dates)})


def test_last_business_day_uses_trade_calendar():
    """2026-10-01(周四,国庆)不是交易日 → 应回退到日历里最近的交易日。"""
    cal = _cal_df(["2026-09-29", "2026-09-30", "2026-10-09"])
    fetcher._reset_trade_calendar_memo()
    with patch("stockpool.fetcher.ak.tool_trade_date_hist_sina", return_value=cal):
        out = fetcher._last_business_day(pd.Timestamp("2026-10-01"))
    assert out == pd.Timestamp("2026-09-30"), (
        f"节假日应回退到上一交易日,实际 {out}"
    )


def test_holiday_cache_not_stale():
    """缓存停在节前最后交易日,长假期间不应判 stale(避免全市场空拉)。"""
    cal = _cal_df(["2026-09-29", "2026-09-30", "2026-10-09"])
    cached = pd.DataFrame({"date": pd.to_datetime(["2026-09-29", "2026-09-30"])})
    fetcher._reset_trade_calendar_memo()
    with patch("stockpool.fetcher.ak.tool_trade_date_hist_sina", return_value=cal), \
         patch("stockpool.fetcher._today", return_value=pd.Timestamp("2026-10-05")):
        assert fetcher._is_stale(cached) is False


def test_calendar_failure_falls_back_to_bday():
    fetcher._reset_trade_calendar_memo()
    with patch("stockpool.fetcher.ak.tool_trade_date_hist_sina",
               side_effect=ConnectionError("down")):
        # 2026-06-13 是周六 → BDay 回退到周五 2026-06-12
        out = fetcher._last_business_day(pd.Timestamp("2026-06-13"))
    assert out == pd.Timestamp("2026-06-12")
    fetcher._reset_trade_calendar_memo()


def test_no_proxy_env_set_on_import():
    """P3-2:用 NO_PROXY=* 环境变量替代对 requests.get 的全局 monkeypatch
    (后者多线程嵌套 patch/restore 会把 patched 版本永久留在 requests 上)。"""
    assert os.environ.get("NO_PROXY") == "*"
    assert os.environ.get("no_proxy") == "*"
