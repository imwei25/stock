from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockpool.fetcher import (
    check_source_change,
    fetch_daily,
    resample_to_weekly,
    update_source_marker,
    validate_ohlcv,
)


def _make_akshare_df(start: str, periods: int) -> pd.DataFrame:
    """AKShare stock_zh_a_hist column structure (Chinese column names)."""
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": np.linspace(10, 11, periods),
        "收盘": np.linspace(10.1, 11.1, periods),
        "最高": np.linspace(10.3, 11.3, periods),
        "最低": np.linspace(9.9, 10.9, periods),
        "成交量": np.full(periods, 1_000_000),
        "成交额": np.full(periods, 1e7),
        "振幅": np.full(periods, 1.0),
        "涨跌幅": np.full(periods, 0.1),
        "涨跌额": np.full(periods, 0.01),
        "换手率": np.full(periods, 0.5),
    })


def test_fetch_creates_cache(tmp_path):
    fake = _make_akshare_df("2026-01-02", 30)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake) as mocked:
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    assert mocked.called
    assert len(df) == 30
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert (tmp_path / "605589_daily.parquet").exists()


def test_second_call_uses_cache_no_request(tmp_path):
    """Cache covers request window and is fresh → no second akshare call."""
    fake = _make_akshare_df("2026-01-02", 60)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    # Pretend "today" is the same business day as the last cached bar so the
    # staleness check passes (last cached == most recent business day).
    last_cached = pd.Timestamp("2026-01-02") + pd.offsets.BDay(59)
    fresh_today = last_cached
    with patch("stockpool.fetcher._today", return_value=fresh_today), \
         patch("stockpool.fetcher.ak.stock_zh_a_hist") as mocked:
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")
        assert not mocked.called

    assert len(df) == 30


def test_force_refresh_bypasses_cache(tmp_path):
    fake = _make_akshare_df("2026-01-02", 30)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake) as mocked:
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, force_refresh=True, source="akshare")
        assert mocked.called


def test_akshare_retry_then_succeed(tmp_path):
    fake = _make_akshare_df("2026-01-02", 30)
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("rate limit")
        return fake

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=flaky), \
         patch("stockpool.fetcher.time.sleep"):
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    assert calls["n"] == 3
    assert len(df) == 30


def test_akshare_all_retries_fail_raises(tmp_path):
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=ConnectionError("down")), \
         patch("stockpool.fetcher.time.sleep"):
        with pytest.raises(ConnectionError):
            fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")


def test_resample_to_weekly():
    daily = pd.DataFrame({
        "date": pd.date_range("2026-01-05", periods=10, freq="B"),
        "open":   [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9],
        "high":   [10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4],
        "low":    [9.5,  9.6,  9.7,  9.8,  9.9,  10.0, 10.1, 10.2, 10.3, 10.4],
        "close":  [10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1],
        "volume": [1_000_000] * 10,
    })

    weekly = resample_to_weekly(daily)

    assert len(weekly) == 2
    assert weekly.iloc[0]["open"] == pytest.approx(10.0)
    assert weekly.iloc[0]["high"] == pytest.approx(10.9)
    assert weekly.iloc[0]["low"] == pytest.approx(9.5)
    assert weekly.iloc[0]["close"] == pytest.approx(10.6)
    assert weekly.iloc[0]["volume"] == 5_000_000


def test_stale_cache_triggers_refetch(tmp_path):
    """Cache last bar is older than the most recent business day → incremental fetch is called."""
    fake = _make_akshare_df("2026-01-02", 60)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    # Advance today far past the cache's last date to make it stale.
    stale_today = pd.Timestamp("2026-01-02") + pd.offsets.BDay(59) + pd.Timedelta(days=30)
    with patch("stockpool.fetcher._today", return_value=stale_today), \
         patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake) as mocked:
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")
        assert mocked.called


# --- validate_ohlcv ---

def _make_ohlcv(closes: list[float], volumes: list[int],
                start: str = "2026-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates, "close": closes, "volume": volumes})


def test_validate_clean_data():
    df = _make_ohlcv([10.0 + i * 0.01 for i in range(10)], [1_000_000] * 10)
    assert validate_ohlcv(df) == []


def test_validate_zero_volume_flagged():
    df = _make_ohlcv([10.0, 10.1, 10.2, 10.3, 10.4],
                     [1_000_000, 0, 0, 1_000_000, 1_000_000])
    issues = validate_ohlcv(df)
    assert any("停牌" in w for w in issues)
    assert any("2" in w for w in issues)  # 2 suspended bars


def test_validate_large_move_flagged():
    # One bar jumps 50% (e.g., after long suspension)
    df = _make_ohlcv([10.0, 10.1, 15.2, 15.3], [1_000_000] * 4)
    issues = validate_ohlcv(df)
    assert any("涨跌幅" in w for w in issues)


def test_validate_calendar_gap_flagged():
    # Insert a 15-day gap between two dates
    dates = list(pd.bdate_range("2026-01-02", periods=3))
    dates.append(dates[-1] + pd.Timedelta(days=15))
    df = pd.DataFrame({
        "date": dates,
        "close": [10.0, 10.1, 10.2, 10.3],
        "volume": [1_000_000] * 4,
    })
    issues = validate_ohlcv(df)
    assert any("间隔" in w for w in issues)


def test_validate_chinese_holiday_gap_not_flagged():
    """Spring Festival / National Day routinely create 8-11 day gaps. Those
    are normal A-share calendar behavior, not data quality issues."""
    # 2026 春节: trading stops 2026-02-13 (Fri), resumes 2026-02-24 (Tue) — 11 days.
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2026-02-11", "2026-02-12", "2026-02-13",
            "2026-02-24", "2026-02-25", "2026-02-26",
        ]),
        "close": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
        "volume": [1_000_000] * 6,
    })
    issues = validate_ohlcv(df)
    assert not any("间隔" in w for w in issues), (
        f"11-day Spring Festival gap should NOT trigger; got {issues!r}"
    )


# --- source-change marker ---

def test_check_source_change_no_marker(tmp_path):
    """First-time use: no marker file → not a 'change', returns False."""
    assert check_source_change(tmp_path, "baostock") is False


def test_check_source_change_match(tmp_path):
    update_source_marker(tmp_path, "baostock")
    assert check_source_change(tmp_path, "baostock") is False


def test_check_source_change_mismatch(tmp_path):
    update_source_marker(tmp_path, "mootdx")
    assert check_source_change(tmp_path, "baostock") is True


def test_update_source_marker_writes_file(tmp_path):
    update_source_marker(tmp_path, "mootdx")
    marker = tmp_path / ".data_source"
    assert marker.exists()
    assert marker.read_text(encoding="utf-8").strip() == "mootdx"


def test_update_source_marker_overwrites(tmp_path):
    update_source_marker(tmp_path, "mootdx")
    update_source_marker(tmp_path, "baostock")
    assert (tmp_path / ".data_source").read_text(encoding="utf-8").strip() == "baostock"


def test_fetch_daily_auto_refresh_on_source_change(tmp_path):
    """If the cache was last filled by a different source, fetch_daily must
    bypass the cache even when the caller didn't pass force_refresh."""
    fake = _make_akshare_df("2026-01-02", 60)

    # Seed cache via akshare (also sets marker = akshare).
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="akshare")

    # Pin today so the cache is *not* stale on its own — only the source
    # change should trigger the refetch.
    fresh_today = pd.Timestamp("2026-01-02") + pd.offsets.BDay(59)

    # Now ask for the same data via baostock. The baostock backend must be
    # called even without force_refresh, because the cache was written by a
    # different source.
    baostock_df = _make_akshare_df("2026-01-02", 60).rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
    })[["date", "open", "high", "low", "close", "volume"]]
    baostock_df["date"] = pd.to_datetime(baostock_df["date"])

    with patch("stockpool.fetcher._today", return_value=fresh_today), \
         patch("stockpool.data_sources.baostock_backend.fetch_stock",
               return_value=baostock_df) as mocked_bs, \
         patch("stockpool.fetcher.ak.stock_zh_a_hist") as mocked_ak:
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, source="baostock")

    assert mocked_bs.called, "baostock backend should be invoked when source changes"
    assert not mocked_ak.called, "akshare should NOT be called when source=baostock"


# ---------------------------------------------------------------------------
# warmup_days tests (Task W2)
# ---------------------------------------------------------------------------

def test_fetch_daily_returns_history_plus_warmup(tmp_path, monkeypatch):
    """fetch_daily returns history_days + warmup_days bars when both > 0."""
    from stockpool import fetcher

    dates = pd.date_range("2022-01-01", periods=1000, freq="B")
    fake = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 1000,
        "high": [101.0] * 1000,
        "low": [99.0] * 1000,
        "close": [100.5] * 1000,
        "volume": [1_000_000] * 1000,
    })
    monkeypatch.setattr(fetcher, "_dispatch_stock", lambda src, c, start=None: fake)
    monkeypatch.setattr(fetcher, "_is_stale", lambda *a, **k: False)

    out = fetcher.fetch_daily(
        "000001", history_days=500, cache_dir=str(tmp_path),
        force_refresh=True, warmup_days=200,
    )
    assert len(out) == 700, f"expected 700 (500+200) rows, got {len(out)}"


def test_fetch_daily_default_warmup_zero(tmp_path, monkeypatch):
    """fetch_daily without warmup_days returns history_days bars (backward compat)."""
    from stockpool import fetcher

    dates = pd.date_range("2022-01-01", periods=1000, freq="B")
    fake = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 1000,
        "high": [101.0] * 1000,
        "low": [99.0] * 1000,
        "close": [100.5] * 1000,
        "volume": [1_000_000] * 1000,
    })
    monkeypatch.setattr(fetcher, "_dispatch_stock", lambda src, c, start=None: fake)
    monkeypatch.setattr(fetcher, "_is_stale", lambda *a, **k: False)

    out = fetcher.fetch_daily(
        "000001", history_days=500, cache_dir=str(tmp_path),
        force_refresh=True,
    )
    assert len(out) == 500


def test_fetch_index_daily_with_warmup(tmp_path, monkeypatch):
    """fetch_index_daily returns history_days + warmup_days bars."""
    from stockpool import fetcher

    dates = pd.date_range("2022-01-01", periods=1000, freq="B")
    fake = pd.DataFrame({
        "date": dates, "open": [3000.0] * 1000, "high": [3010.0] * 1000,
        "low": [2990.0] * 1000, "close": [3005.0] * 1000, "volume": [1e9] * 1000,
    })
    monkeypatch.setattr(fetcher, "_dispatch_index", lambda src, s: fake)
    monkeypatch.setattr(fetcher, "_is_stale", lambda *a, **k: False)

    out = fetcher.fetch_index_daily(
        "sh000001", history_days=300, cache_dir=str(tmp_path),
        force_refresh=True, warmup_days=100,
    )
    assert len(out) == 400


def test_fetch_sector_daily_with_warmup(tmp_path, monkeypatch):
    """fetch_sector_daily returns history_days + warmup_days bars."""
    from stockpool import fetcher

    dates = pd.date_range("2022-01-01", periods=1000, freq="B")
    fake = pd.DataFrame({
        "date": dates, "open": [1000.0] * 1000, "high": [1010.0] * 1000,
        "low": [990.0] * 1000, "close": [1005.0] * 1000, "volume": [1e8] * 1000,
    })
    monkeypatch.setattr(fetcher, "_dispatch_sector", lambda src, s, start=None: fake)
    monkeypatch.setattr(fetcher, "_is_stale", lambda *a, **k: False)

    out = fetcher.fetch_sector_daily(
        "化工", history_days=200, cache_dir=str(tmp_path),
        force_refresh=True, warmup_days=50,
    )
    assert len(out) == 250


# ---------------------------------------------------------------------------
# warmup_days tests (Task W3: universe loaders)
# ---------------------------------------------------------------------------

def test_fetch_universe_threads_warmup_days(tmp_path, monkeypatch):
    """fetch_universe forwards warmup_days to per-stock fetch_daily."""
    from stockpool import fetcher

    captured = []
    real_fetch_daily = fetcher.fetch_daily

    def spy(code, history_days, cache_dir, **kw):
        captured.append((code, history_days, kw.get("warmup_days", 0)))
        # Return synthetic data
        dates = pd.date_range("2023-01-01", periods=200, freq="B")
        return pd.DataFrame({
            "date": dates, "open": [1.0]*200, "high": [1.0]*200,
            "low": [1.0]*200, "close": [1.0]*200, "volume": [1]*200,
        })

    monkeypatch.setattr(fetcher, "fetch_daily", spy)
    fetcher.fetch_universe(
        ["000001", "000002"], history_days=100,
        cache_dir=str(tmp_path), warmup_days=50,
        max_workers=1,
    )
    # Each per-stock call should carry warmup_days=50
    assert all(w == 50 for _c, _h, w in captured), f"warmup_days not threaded: {captured}"
    assert {c for c, _h, _w in captured} == {"000001", "000002"}


def test_load_universe_cache_respects_warmup_days(tmp_path):
    """load_universe_cache tails to history_days + warmup_days when both given."""
    from stockpool.fetcher import load_universe_cache

    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    df = pd.DataFrame({
        "date": dates, "open": [1.0]*500, "high": [1.0]*500,
        "low": [1.0]*500, "close": [1.0]*500, "volume": [1]*500,
    })
    (tmp_path / "AAAAAA_daily.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path / "AAAAAA_daily.parquet")

    # Without warmup: 100 bars
    out = load_universe_cache(str(tmp_path), history_days=100)
    assert len(out["AAAAAA"]) == 100

    # With warmup 50: 150 bars
    out = load_universe_cache(str(tmp_path), history_days=100, warmup_days=50)
    assert len(out["AAAAAA"]) == 150


def test_load_universe_cache_default_warmup_zero(tmp_path):
    """load_universe_cache default (no warmup) keeps existing behavior."""
    from stockpool.fetcher import load_universe_cache

    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    df = pd.DataFrame({
        "date": dates, "open": [1.0]*200, "high": [1.0]*200,
        "low": [1.0]*200, "close": [1.0]*200, "volume": [1]*200,
    })
    df.to_parquet(tmp_path / "BBBBBB_daily.parquet")

    out = load_universe_cache(str(tmp_path), history_days=50)
    assert len(out["BBBBBB"]) == 50  # warmup default 0 → just history_days
