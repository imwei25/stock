from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockpool.fetcher import fetch_daily, resample_to_weekly


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
    """Cache covers request window → no second akshare call."""
    fake = _make_akshare_df("2026-01-02", 60)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist") as mocked:
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
