"""P0 数据层修复的回归测试:复权统一 (hfq) + 盘中半根 bar + 增量接缝校验。

覆盖:
- mootdx 段内锚定 hfq (_apply_hfq) 的数学正确性
- baostock adjustflag=1 / akshare adjust=hfq
- 盘中 (15:05 前) 当日 bar 不入缓存
- 增量拉取与缓存重叠一根 bar,接缝不一致触发全量刷新
- mootdx 增量段锚定到缓存价格尺度
- 复权模式写入 source marker,旧格式 marker 触发迁移性全量刷新
"""
import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockpool.fetcher import fetch_daily, update_source_marker


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_akshare_df(start: str, periods: int, close_start: float = 10.1) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    closes = np.linspace(close_start, close_start + 1.0, periods)
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": closes - 0.1,
        "收盘": closes,
        "最高": closes + 0.2,
        "最低": closes - 0.2,
        "成交量": np.full(periods, 1_000_000),
        "成交额": np.full(periods, 1e7),
    })


def _ohlcv(dates, closes, volumes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes - 0.1,
        "high": closes + 0.2,
        "low": closes - 0.2,
        "close": closes,
        "volume": np.asarray(volumes, dtype=float),
    })


# ---------------------------------------------------------------------------
# mootdx 段内锚定 hfq
# ---------------------------------------------------------------------------

class TestApplyHfq:
    def _bars(self, closes, start="2026-01-05"):
        dates = pd.bdate_range(start, periods=len(closes))
        return _ohlcv(dates, closes, [1000.0] * len(closes))

    def test_no_events_returns_prices_unchanged(self):
        from stockpool.data_sources.mootdx_backend import _apply_hfq
        bars = self._bars([10.0, 10.1, 10.2])
        events = pd.DataFrame(columns=["date", "fenhong", "peigu", "peigujia", "songzhuangu"])
        out = _apply_hfq(bars, events)
        assert np.allclose(out["close"], [10.0, 10.1, 10.2])

    def test_dividend_and_split_makes_series_continuous(self):
        """10 送 10 + 每 10 股派 1 元:除权日理论价 = (20*10-1)/(10+10) = 9.95。
        若除权日实际 close 恰为理论价(经济上没涨没跌),hfq 调整后该日 close
        应回到 20.0,跨除权日收益率为 0。"""
        from stockpool.data_sources.mootdx_backend import _apply_hfq
        dates = pd.bdate_range("2026-01-05", periods=4)
        bars = _ohlcv(dates, [19.9, 20.0, 9.95, 10.0], [1000] * 4)
        events = pd.DataFrame({
            "date": [dates[2]],
            "fenhong": [1.0],
            "peigu": [0.0],
            "peigujia": [0.0],
            "songzhuangu": [10.0],
        })
        out = _apply_hfq(bars, events)
        # 段首因子 = 1:除权前价格不变
        assert out["close"].iloc[0] == pytest.approx(19.9)
        assert out["close"].iloc[1] == pytest.approx(20.0)
        # 除权日及之后乘以因子 20/9.95
        assert out["close"].iloc[2] == pytest.approx(20.0)
        assert out["close"].iloc[3] == pytest.approx(10.0 * 20.0 / 9.95)
        # OHLC 同步缩放,volume 不动
        assert out["open"].iloc[2] == pytest.approx((9.95 - 0.1) * 20.0 / 9.95)
        assert np.allclose(out["volume"], 1000)

    def test_pure_cash_dividend(self):
        """每 10 股派 0.5 元,prev_close=10 → 理论除权价 9.95,因子 10/9.95。"""
        from stockpool.data_sources.mootdx_backend import _apply_hfq
        dates = pd.bdate_range("2026-01-05", periods=3)
        bars = _ohlcv(dates, [10.0, 9.95, 10.05], [1000] * 3)
        events = pd.DataFrame({
            "date": [dates[1]],
            "fenhong": [0.5], "peigu": [0.0], "peigujia": [0.0], "songzhuangu": [0.0],
        })
        out = _apply_hfq(bars, events)
        assert out["close"].iloc[1] == pytest.approx(9.95 * 10.0 / 9.95)
        assert out["close"].iloc[2] == pytest.approx(10.05 * 10.0 / 9.95)

    def test_event_before_window_is_absorbed_as_constant(self):
        """窗口开始前的事件只贡献常数因子,段内锚定语义下直接忽略。"""
        from stockpool.data_sources.mootdx_backend import _apply_hfq
        bars = self._bars([10.0, 10.1, 10.2])
        events = pd.DataFrame({
            "date": [pd.Timestamp("2025-06-01")],
            "fenhong": [5.0], "peigu": [0.0], "peigujia": [0.0], "songzhuangu": [10.0],
        })
        out = _apply_hfq(bars, events)
        assert np.allclose(out["close"], [10.0, 10.1, 10.2])

    def test_event_on_first_bar_is_skipped(self):
        """事件落在段首 bar(无段内 prev_close)→ 无法算因子,按常数吸收。"""
        from stockpool.data_sources.mootdx_backend import _apply_hfq
        bars = self._bars([10.0, 10.1, 10.2])
        events = pd.DataFrame({
            "date": [bars["date"].iloc[0]],
            "fenhong": [1.0], "peigu": [0.0], "peigujia": [0.0], "songzhuangu": [0.0],
        })
        out = _apply_hfq(bars, events)
        assert np.allclose(out["close"], [10.0, 10.1, 10.2])

    def test_rights_issue(self):
        """每 10 股配 3 股、配股价 8 元,prev_close=10:
        理论除权价 = (10*10 + 3*8)/(10+3) = 124/13。"""
        from stockpool.data_sources.mootdx_backend import _apply_hfq
        dates = pd.bdate_range("2026-01-05", periods=2)
        ex_theory = (10.0 * 10 + 3 * 8.0) / 13
        bars = _ohlcv(dates, [10.0, ex_theory], [1000] * 2)
        events = pd.DataFrame({
            "date": [dates[1]],
            "fenhong": [0.0], "peigu": [3.0], "peigujia": [8.0], "songzhuangu": [0.0],
        })
        out = _apply_hfq(bars, events)
        assert out["close"].iloc[1] == pytest.approx(10.0)


def test_mootdx_fetch_stock_applies_hfq():
    """fetch_stock 集成:拉 bars + 拉 xdxr + 段内 hfq。"""
    from stockpool.data_sources import mootdx_backend
    dates = pd.bdate_range("2026-01-05", periods=4)
    raw = _ohlcv(dates, [19.9, 20.0, 9.95, 10.0], [1000] * 4)
    events = pd.DataFrame({
        "date": [dates[2]],
        "fenhong": [1.0], "peigu": [0.0], "peigujia": [0.0], "songzhuangu": [10.0],
    })
    with patch.object(mootdx_backend, "_call_with_retry", return_value=raw), \
         patch.object(mootdx_backend, "_fetch_xdxr", return_value=events):
        out = mootdx_backend.fetch_stock("000001")
    assert out["close"].iloc[2] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# baostock / akshare 复权参数
# ---------------------------------------------------------------------------

class _FakeRS:
    error_code = "0"
    error_msg = ""

    def __init__(self, fields: list[str], rows: list[list[str]]):
        self.fields = fields
        self._rows = list(rows)

    def next(self):
        return bool(self._rows)

    def get_row_data(self):
        return self._rows.pop(0)


def test_baostock_query_uses_hfq(monkeypatch):
    from stockpool.data_sources import baostock_backend
    captured: dict = {}

    def fake_query(code, fields, **kwargs):
        captured.update(kwargs)
        cols = fields.split(",")
        rows = [["2026-01-05", "10.0", "10.2", "9.8", "10.1", "100000"],
                ["2026-01-06", "10.1", "10.3", "9.9", "10.2", "110000"]]
        if "tradestatus" in cols:
            rows = [r + ["1"] for r in rows]
        return _FakeRS(cols, rows)

    fake_bs = types.SimpleNamespace(
        login=lambda: types.SimpleNamespace(error_code="0", error_msg=""),
        query_history_k_data_plus=fake_query,
    )
    monkeypatch.setitem(sys.modules, "baostock", fake_bs)
    monkeypatch.setattr(baostock_backend, "_logged_in", False)

    df = baostock_backend.fetch_stock("605589", start="2026-01-01")
    assert captured.get("adjustflag") == "1", (
        f"baostock 应使用后复权 adjustflag=1,实际 {captured.get('adjustflag')!r}"
    )
    assert len(df) == 2


def test_akshare_fetch_uses_hfq(tmp_path):
    fake = _make_akshare_df("2026-01-02", 30)
    captured: dict = {}

    def fake_hist(**kwargs):
        captured.update(kwargs)
        return fake

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=fake_hist):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")
    assert captured.get("adjust") == "hfq", (
        f"akshare 应使用后复权 adjust='hfq',实际 {captured.get('adjust')!r}"
    )


# ---------------------------------------------------------------------------
# 盘中半根 bar 不入缓存
# ---------------------------------------------------------------------------

def test_intraday_partial_bar_not_cached(tmp_path):
    """10:30 盘中运行:当日未完成 bar 必须被丢弃,不得写入缓存。"""
    fake = _make_akshare_df("2026-01-02", 30)
    last_day = pd.to_datetime(fake["日期"].iloc[-1])

    intraday = last_day + pd.Timedelta(hours=10, minutes=30)
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake), \
         patch("stockpool.fetcher._now", return_value=intraday), \
         patch("stockpool.fetcher._today", return_value=last_day):
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    assert df["date"].max() < last_day, "盘中运行不应返回当日半根 bar"
    cached = pd.read_parquet(tmp_path / "605589_daily.parquet")
    assert pd.Timestamp(cached["date"].max()) < last_day, "当日半根 bar 不得写入缓存"


def test_after_close_bar_is_cached(tmp_path):
    """15:30 收盘后运行:当日 bar 已完整,正常入缓存。"""
    fake = _make_akshare_df("2026-01-02", 30)
    last_day = pd.to_datetime(fake["日期"].iloc[-1])

    after_close = last_day + pd.Timedelta(hours=15, minutes=30)
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake), \
         patch("stockpool.fetcher._now", return_value=after_close), \
         patch("stockpool.fetcher._today", return_value=last_day):
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    assert df["date"].max() == last_day


# ---------------------------------------------------------------------------
# 增量重叠拉取 + 接缝校验
# ---------------------------------------------------------------------------

def _seed_akshare_cache(tmp_path, periods=60) -> pd.Timestamp:
    fake = _make_akshare_df("2026-01-02", periods)
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")
    return pd.to_datetime(fake["日期"].iloc[-1])


def test_incremental_fetch_overlaps_last_cached_day(tmp_path):
    """增量拉取必须从缓存最后一天(含)开始,而非 last+1。"""
    last = _seed_akshare_cache(tmp_path)
    captured: dict = {}
    cached_df = pd.read_parquet(tmp_path / "605589_daily.parquet")
    last_close = float(cached_df["close"].iloc[-1])

    new_days = pd.bdate_range(last, periods=4)  # 含重叠日
    incr = pd.DataFrame({
        "日期": new_days.strftime("%Y-%m-%d"),
        "开盘": [last_close] * 4,
        "收盘": [last_close, last_close + 0.1, last_close + 0.2, last_close + 0.3],
        "最高": [last_close + 0.3] * 4,
        "最低": [last_close - 0.3] * 4,
        "成交量": [1_000_000] * 4,
        "成交额": [1e7] * 4,
    })

    def fake_hist(**kwargs):
        captured.update(kwargs)
        return incr

    stale_today = new_days[-1]
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=fake_hist), \
         patch("stockpool.fetcher._today", return_value=stale_today), \
         patch("stockpool.fetcher._now", return_value=stale_today + pd.Timedelta(hours=18)):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    assert captured.get("start_date") == last.strftime("%Y%m%d"), (
        f"增量拉取应从缓存最后一天(含)开始,实际 start_date={captured.get('start_date')!r}"
    )
    combined = pd.read_parquet(tmp_path / "605589_daily.parquet")
    assert pd.Timestamp(combined["date"].max()) == new_days[-1]
    assert combined["date"].duplicated().sum() == 0


def test_seam_close_mismatch_triggers_full_refresh(tmp_path):
    """重叠 bar 的 close 与缓存对不上(如复权基准漂移)→ 丢弃缓存全量重拉。"""
    last = _seed_akshare_cache(tmp_path)
    cached_df = pd.read_parquet(tmp_path / "605589_daily.parquet")
    last_close = float(cached_df["close"].iloc[-1])

    new_days = pd.bdate_range(last, periods=2)
    incr = pd.DataFrame({
        "日期": new_days.strftime("%Y-%m-%d"),
        "开盘": [last_close * 1.05] * 2,
        "收盘": [last_close * 1.05, last_close * 1.06],  # 重叠日 close 偏 5%
        "最高": [last_close * 1.06] * 2,
        "最低": [last_close * 1.04] * 2,
        "成交量": [1_000_000] * 2,
        "成交额": [1e7] * 2,
    })
    full = _make_akshare_df("2026-01-02", 61, close_start=20.0)
    full.loc[len(full) - 1, "日期"] = new_days[-1].strftime("%Y-%m-%d")

    calls: list[dict] = []

    def fake_hist(**kwargs):
        calls.append(dict(kwargs))
        return incr if len(calls) == 1 else full

    stale_today = new_days[-1]
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=fake_hist), \
         patch("stockpool.fetcher._today", return_value=stale_today), \
         patch("stockpool.fetcher._now", return_value=stale_today + pd.Timedelta(hours=18)):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    assert len(calls) == 2, "接缝不一致应触发第二次(全量)拉取"
    assert calls[1].get("start_date") == "19900101", "第二次拉取应为全量"
    combined = pd.read_parquet(tmp_path / "605589_daily.parquet")
    # 缓存应被全量结果替换(close ≈ 20.x 而非 10.x)
    assert float(combined["close"].iloc[0]) > 15.0


def test_seam_volume_mismatch_triggers_full_refresh(tmp_path):
    """重叠 bar 成交量对不上(缓存里是盘中半根 bar)→ 全量重拉自愈。"""
    last = _seed_akshare_cache(tmp_path)
    cache_file = tmp_path / "605589_daily.parquet"
    cached_df = pd.read_parquet(cache_file)
    # 人为把最后一根 bar 改成"半根"(量只有一半)
    cached_df.loc[len(cached_df) - 1, "volume"] = 500_000
    cached_df.to_parquet(cache_file, index=False)
    last_close = float(cached_df["close"].iloc[-1])

    new_days = pd.bdate_range(last, periods=2)
    incr = pd.DataFrame({
        "日期": new_days.strftime("%Y-%m-%d"),
        "开盘": [last_close] * 2,
        "收盘": [last_close, last_close + 0.1],
        "最高": [last_close + 0.2] * 2,
        "最低": [last_close - 0.2] * 2,
        "成交量": [1_000_000] * 2,  # 完整 bar 的真实量
        "成交额": [1e7] * 2,
    })
    full = _make_akshare_df("2026-01-02", 61)

    calls: list[dict] = []

    def fake_hist(**kwargs):
        calls.append(dict(kwargs))
        return incr if len(calls) == 1 else full

    stale_today = new_days[-1]
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=fake_hist), \
         patch("stockpool.fetcher._today", return_value=stale_today), \
         patch("stockpool.fetcher._now", return_value=stale_today + pd.Timedelta(hours=18)):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    assert len(calls) == 2, "重叠 bar 成交量不一致应触发全量重拉"


def test_seam_match_appends_without_full_refresh(tmp_path):
    """重叠 bar 一致 → 正常增量合并,不触发第二次拉取。"""
    last = _seed_akshare_cache(tmp_path)
    cached_df = pd.read_parquet(tmp_path / "605589_daily.parquet")
    last_row = cached_df.iloc[-1]

    new_days = pd.bdate_range(last, periods=3)
    incr = pd.DataFrame({
        "日期": new_days.strftime("%Y-%m-%d"),
        "开盘": [float(last_row["open"])] * 3,
        "收盘": [float(last_row["close"]), float(last_row["close"]) + 0.1,
                float(last_row["close"]) + 0.2],
        "最高": [float(last_row["high"])] * 3,
        "最低": [float(last_row["low"])] * 3,
        # akshare 原始单位是"手",缓存里已是"股"(×100)——除回去保持重叠一致
        "成交量": [float(last_row["volume"]) / 100.0] * 3,
        "成交额": [1e7] * 3,
    })

    calls: list[dict] = []

    def fake_hist(**kwargs):
        calls.append(dict(kwargs))
        return incr

    stale_today = new_days[-1]
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=fake_hist), \
         patch("stockpool.fetcher._today", return_value=stale_today), \
         patch("stockpool.fetcher._now", return_value=stale_today + pd.Timedelta(hours=18)):
        # history_days 不得超过缓存长度,否则按设计触发全量回填而非增量
        df = fetch_daily("605589", history_days=60, cache_dir=tmp_path, source="akshare")

    assert len(calls) == 1, "接缝一致时不应触发全量重拉"
    assert len(df) == 60
    assert pd.Timestamp(df["date"].max()) == new_days[-1]


# ---------------------------------------------------------------------------
# mootdx 增量段锚定
# ---------------------------------------------------------------------------

def test_mootdx_incremental_anchors_to_cached_scale(tmp_path):
    """缓存是 hfq 价(尺度 2×raw),mootdx 增量段返回段内 hfq(从 raw 起算)。
    fetcher 必须用重叠 bar 把增量段锚定到缓存尺度:raw 21.0 → 42.0。"""
    dates = pd.bdate_range("2026-01-05", periods=10)
    cached = _ohlcv(dates, np.linspace(38.0, 40.0, 10), [1000.0] * 10)
    cache_file = tmp_path / "000001_daily.parquet"
    cached.to_parquet(cache_file, index=False)
    update_source_marker(tmp_path, "mootdx")

    last = dates[-1]
    next_day = last + pd.offsets.BDay(1)
    # 段内 hfq 从 raw 起算:重叠日 close=20.0(= 缓存 40.0 的一半尺度)
    fresh = _ohlcv([last, next_day], [20.0, 21.0], [1000.0, 1100.0])

    with patch("stockpool.data_sources.mootdx_backend.fetch_stock",
               return_value=fresh) as mocked, \
         patch("stockpool.fetcher._today", return_value=next_day.normalize()), \
         patch("stockpool.fetcher._now",
               return_value=next_day.normalize() + pd.Timedelta(hours=18)):
        # history_days ≤ 缓存长度(10)才走增量锚定路径
        df = fetch_daily("000001", history_days=10, cache_dir=tmp_path, source="mootdx")

    assert mocked.call_args.kwargs.get("start") == last.strftime("%Y%m%d")
    assert df["close"].iloc[-1] == pytest.approx(42.0), (
        f"增量段应锚定到缓存尺度 (21.0×2=42.0),实际 {df['close'].iloc[-1]}"
    )
    assert df["close"].iloc[-2] == pytest.approx(40.0)
    assert df["volume"].iloc[-1] == pytest.approx(1100.0), "volume 不应被缩放"


# ---------------------------------------------------------------------------
# marker:复权模式纳入,旧格式触发迁移
# ---------------------------------------------------------------------------

def test_marker_includes_adjust_mode(tmp_path):
    update_source_marker(tmp_path, "mootdx")
    content = (tmp_path / ".data_source").read_text(encoding="utf-8").strip()
    # v2 = volume 单位统一为"股"(P1-6);格式 <source>:<adjust>:<schema>
    assert content == "mootdx:hfq:v2", f"marker 应含复权模式+schema 版本,实际 {content!r}"


def test_legacy_marker_triggers_full_refresh(tmp_path):
    """旧格式 marker('mootdx',无复权标记)= 缓存是不复权旧数据 → 必须全量重拉。"""
    dates = pd.bdate_range("2026-01-05", periods=10)
    _ohlcv(dates, np.linspace(10.0, 11.0, 10), [1000.0] * 10).to_parquet(
        tmp_path / "000001_daily.parquet", index=False)
    (tmp_path / ".data_source").write_text("mootdx", encoding="utf-8")

    fresh = _ohlcv(dates, np.linspace(20.0, 22.0, 10), [1000.0] * 10)
    with patch("stockpool.data_sources.mootdx_backend.fetch_stock",
               return_value=fresh) as mocked, \
         patch("stockpool.fetcher._today", return_value=dates[-1]), \
         patch("stockpool.fetcher._now",
               return_value=dates[-1] + pd.Timedelta(hours=18)):
        df = fetch_daily("000001", history_days=10, cache_dir=tmp_path, source="mootdx")

    assert mocked.called, "旧格式 marker 必须触发重拉"
    assert mocked.call_args.kwargs.get("start") is None, "迁移重拉必须是全量"
    assert df["close"].iloc[0] == pytest.approx(20.0), "缓存应被新复权数据替换"
