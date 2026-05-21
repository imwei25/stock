from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockpool.fetcher import fetch_daily, resample_to_weekly, validate_ohlcv


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
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    assert mocked.called
    assert len(df) == 30
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert (tmp_path / "605589_daily.parquet").exists()


def test_second_call_uses_cache_no_request(tmp_path):
    """Cache covers request window and is fresh → no second akshare call."""
    fake = _make_akshare_df("2026-01-02", 60)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    # Pretend "today" is the same business day as the last cached bar so the
    # staleness check passes (last cached == most recent business day).
    last_cached = pd.Timestamp("2026-01-02") + pd.offsets.BDay(59)
    fresh_today = last_cached
    with patch("stockpool.fetcher._today", return_value=fresh_today), \
         patch("stockpool.fetcher.ak.stock_zh_a_hist") as mocked:
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path)
        assert not mocked.called

    assert len(df) == 30


def test_force_refresh_bypasses_cache(tmp_path):
    fake = _make_akshare_df("2026-01-02", 30)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake) as mocked:
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, force_refresh=True)
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
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    assert calls["n"] == 3
    assert len(df) == 30


def test_akshare_all_retries_fail_raises(tmp_path):
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=ConnectionError("down")), \
         patch("stockpool.fetcher.time.sleep"):
        with pytest.raises(ConnectionError):
            fetch_daily("605589", history_days=30, cache_dir=tmp_path)


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
        fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    # Advance today far past the cache's last date to make it stale.
    stale_today = pd.Timestamp("2026-01-02") + pd.offsets.BDay(59) + pd.Timedelta(days=30)
    with patch("stockpool.fetcher._today", return_value=stale_today), \
         patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake) as mocked:
        fetch_daily("605589", history_days=30, cache_dir=tmp_path)
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
