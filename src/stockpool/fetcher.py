"""AKShare 数据获取 + Parquet 本地缓存."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import akshare as ak
import pandas as pd

log = logging.getLogger(__name__)

_AKSHARE_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
}

_RETRY_DELAYS = [2, 4, 8]


def _cache_path(cache_dir: str | Path, code: str) -> Path:
    return Path(cache_dir) / f"{code}_daily.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=_AKSHARE_COLUMN_MAP).copy()
    keep = ["date", "open", "high", "low", "close", "volume"]
    out = out[keep]
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return out


def _fetch_from_akshare(code: str, start: str | None = None) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start or "19900101",
                end_date="20991231",
                adjust="qfq",
            )
            return _normalize(raw)
        except Exception as e:
            last_err = e
            log.warning("AKShare attempt %d/%d for %s failed: %s",
                        attempt, len(_RETRY_DELAYS), code, e)
            if attempt < len(_RETRY_DELAYS):
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def fetch_daily(
    code: str,
    history_days: int,
    cache_dir: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return latest `history_days` daily K bars (English column names).

    Uses local Parquet cache, only triggers incremental fetch when needed.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, code)

    cached: pd.DataFrame | None = None
    if cache_file.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache_file)
        except Exception as e:
            log.warning("Cache %s corrupt (%s), refetching", cache_file, e)
            cache_file.unlink(missing_ok=True)
            cached = None

    need_fetch = (
        force_refresh
        or cached is None
        or len(cached) < history_days
    )

    if need_fetch:
        start = None
        if cached is not None and not force_refresh:
            last = cached["date"].max()
            start = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
        fresh = _fetch_from_akshare(code, start=start)
        if cached is not None and not force_refresh:
            combined = pd.concat([cached, fresh]).drop_duplicates("date").sort_values("date")
        else:
            combined = fresh
        combined = combined.reset_index(drop=True)
        combined.to_parquet(cache_file, index=False)
        cached = combined

    return cached.tail(history_days).reset_index(drop=True)


def resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily K → Weekly K (W-FRI: each week ends on Friday)."""
    df = daily.copy()
    df = df.set_index(pd.DatetimeIndex(df["date"]))
    weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    weekly = weekly.reset_index().rename(columns={"index": "date"})
    return weekly
