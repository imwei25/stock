"""P1-10: mootdx 单次 800 根上限的分页拉取测试(全部 mock,离线运行)。

mootdx ``client.bars(symbol, frequency, start, offset)`` 的 ``start`` 是
"从最近一根往回数的偏移位置"。修复后 fetch_stock 按 start=0/800/1600/...
分页拼接,直到 ① 覆盖目标起始日期 ② 达到 min_bars ③ 空页/短页(数据到头)
④ 硬上限。fetcher 全量拉取时透传 ``min_bars=history_days + 60``。
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockpool import fetcher
from stockpool.data_sources import mootdx_backend


# ---------------------------------------------------------------------------
# 模拟 TDX bars 服务器
# ---------------------------------------------------------------------------

def _history(n: int) -> pd.DataFrame:
    """长度为 n 的完整日线历史(升序,volume 单位 = 手)。"""
    dates = pd.bdate_range(end="2026-06-10", periods=n)
    closes = np.linspace(10.0, 10.0 + 0.01 * (n - 1), n)
    return pd.DataFrame({
        "date": dates,
        "open": closes - 0.1,
        "high": closes + 0.2,
        "low": closes - 0.2,
        "close": closes,
        "volume": np.full(n, 1000.0),
    })


class _BarsServer:
    """模拟 ``_call_with_retry("bars", ...)``:按 start/offset 切片返回。

    ``empty_mode`` 控制翻页越过最早历史时的行为:
      * "empty" — 返回空 DataFrame;
      * "raise" — 模拟真实 _call_with_retry 重试耗尽后抛 RuntimeError。
    """

    def __init__(self, hist: pd.DataFrame, empty_mode: str = "empty"):
        self.hist = hist
        self.calls: list[dict] = []
        self.empty_mode = empty_mode

    def __call__(self, method_name: str, **kwargs):
        assert method_name == "bars"
        self.calls.append(dict(kwargs))
        s = int(kwargs.get("start", 0))
        n = int(kwargs["offset"])
        total = len(self.hist)
        if s >= total:
            if self.empty_mode == "raise":
                raise RuntimeError("empty result")
            return self.hist.iloc[0:0].copy()
        lo = max(total - s - n, 0)
        return self.hist.iloc[lo:total - s].copy()


def _fetch(hist: pd.DataFrame, server: _BarsServer | None = None, **kwargs):
    server = server or _BarsServer(hist)
    empty_events = pd.DataFrame(columns=mootdx_backend._XDXR_EVENT_COLS)
    with patch.object(mootdx_backend, "_call_with_retry", side_effect=server), \
         patch.object(mootdx_backend, "_fetch_xdxr", return_value=empty_events):
        out = mootdx_backend.fetch_stock("000001", **kwargs)
    return out, server


# ---------------------------------------------------------------------------
# 分页拼接
# ---------------------------------------------------------------------------

def test_multi_page_concat_order_and_dedup():
    """min_bars=1500 → 两页 (start=0, start=800) 拼接,升序无重复。"""
    hist = _history(2000)
    out, server = _fetch(hist, min_bars=1500)

    starts = [c.get("start", 0) for c in server.calls]
    assert starts == [0, 800], f"应按 start=0,800 分页,实际 {starts}"
    assert len(out) == 1600
    assert out["date"].is_monotonic_increasing
    assert out["date"].duplicated().sum() == 0
    # 拼接结果 = 完整历史的最后 1600 根
    expected = hist["date"].tail(1600).reset_index(drop=True)
    pd.testing.assert_series_equal(out["date"], expected, check_names=False)
    # volume 只放大一次(1000 手 → 100000 股),分页不应重复缩放
    assert np.allclose(out["volume"], 100_000.0)


def test_min_bars_satisfied_stops_paging():
    """历史足够长时,凑够 min_bars 即停,不继续翻页。"""
    hist = _history(4000)
    out, server = _fetch(hist, min_bars=900)
    assert len(server.calls) == 2, "900 根只需 2 页,不应继续翻页"
    assert len(out) == 1600


def test_short_page_terminates_and_warns(caplog):
    """服务器只有 1000 根:第二页返回 200 根(短页)→ 数据到头,warning。"""
    hist = _history(1000)
    with caplog.at_level(logging.WARNING, logger="stockpool.data_sources.mootdx_backend"):
        out, server = _fetch(hist, min_bars=3000)
    assert len(server.calls) == 2
    assert len(out) == 1000
    msgs = [r.getMessage() for r in caplog.records]
    assert any("000001" in m and "1000" in m and "3000" in m for m in msgs), (
        f"拉到头仍不满 min_bars 应 warning(代码/实拿/请求),实际日志: {msgs}"
    )


@pytest.mark.parametrize("empty_mode", ["empty", "raise"])
def test_empty_page_terminates_and_warns(empty_mode, caplog):
    """服务器恰好 800 根:第二页为空(或重试耗尽抛错)→ 终止 + warning。"""
    hist = _history(800)
    server = _BarsServer(hist, empty_mode=empty_mode)
    with caplog.at_level(logging.WARNING, logger="stockpool.data_sources.mootdx_backend"):
        out, server = _fetch(hist, server=server, min_bars=2000)
    assert len(out) == 800
    assert len(server.calls) == 2
    msgs = [r.getMessage() for r in caplog.records]
    assert any("000001" in m and "800" in m and "2000" in m for m in msgs)


def test_start_date_paginates_until_covered():
    """start 在 1200 根之前 → 第一页(≤800)不够,翻第二页后覆盖到目标日期。"""
    hist = _history(2000)
    target = hist["date"].iloc[-1200]
    out, server = _fetch(hist, start=target.strftime("%Y%m%d"))
    assert len(server.calls) >= 2, "目标日期超出首页范围时必须翻页"
    assert len(out) == 1200
    assert out["date"].iloc[0] == target


def test_no_start_no_min_bars_single_page():
    """兼容旧行为:无 start 也无 min_bars → 只拉最近一页(≤800 根)。"""
    hist = _history(2000)
    out, server = _fetch(hist)
    assert len(server.calls) == 1
    assert len(out) == 800


def test_hard_cap_limits_total_bars():
    """min_bars 远超硬上限 → 在 _MAX_TOTAL_BARS 处停止,防失控。"""
    cap = mootdx_backend._MAX_TOTAL_BARS
    hist = _history(cap + 2000)
    out, server = _fetch(hist, min_bars=cap + 1000)
    assert len(out) == cap
    assert len(server.calls) == cap // 800


def test_first_page_failure_raises():
    """首页就拉不到(真网络错误)必须向上抛,不能静默返回空。"""
    def boom(method_name, **kwargs):
        raise RuntimeError("network down")

    empty_events = pd.DataFrame(columns=mootdx_backend._XDXR_EVENT_COLS)
    with patch.object(mootdx_backend, "_call_with_retry", side_effect=boom), \
         patch.object(mootdx_backend, "_fetch_xdxr", return_value=empty_events):
        with pytest.raises(RuntimeError):
            mootdx_backend.fetch_stock("000001", min_bars=100)


# ---------------------------------------------------------------------------
# fetcher 透传 min_bars
# ---------------------------------------------------------------------------

def test_fetch_daily_full_pull_passes_min_bars(tmp_path):
    """无缓存全量拉取时,fetch_daily 应传 min_bars=history_days+60(warmup)。"""
    df = _history(200)
    df["volume"] = df["volume"] * 100.0  # fetch_stock 返回的已是股
    with patch("stockpool.data_sources.mootdx_backend.fetch_stock",
               return_value=df) as mocked:
        fetcher.fetch_daily("000001", history_days=1000, cache_dir=tmp_path,
                            source="mootdx")
    assert mocked.called
    assert mocked.call_args.kwargs.get("min_bars") == 1060, (
        f"应透传 min_bars=history_days+60,实际 {mocked.call_args.kwargs}"
    )
    assert mocked.call_args.kwargs.get("start") is None


def test_dispatch_akshare_baostock_ignore_min_bars():
    """akshare/baostock 分支不接收 min_bars(它们本就全量返回)。"""
    df = _history(10)
    with patch("stockpool.fetcher._fetch_from_akshare", return_value=df) as ak_mock:
        fetcher._dispatch_stock("akshare", "000001", None, min_bars=500)
    assert "min_bars" not in ak_mock.call_args.kwargs

    with patch("stockpool.data_sources.baostock_backend.fetch_stock",
               return_value=df) as bs_mock:
        fetcher._dispatch_stock("baostock", "000001", None, min_bars=500)
    assert "min_bars" not in bs_mock.call_args.kwargs
