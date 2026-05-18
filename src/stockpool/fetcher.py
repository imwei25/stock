"""AKShare 数据获取 + Parquet 本地缓存."""
from __future__ import annotations

import contextlib
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
_STALE_CALENDAR_DAYS = 5  # trigger incremental fetch if cache is this many days old


@contextlib.contextmanager
def _no_proxy():
    """强制 AKShare 的 requests.get 直连，不走任何代理，退出后还原。"""
    import requests as _req
    _orig = _req.get

    def _direct_get(url, **kwargs):
        # proxies={} 会被 setdefault 覆盖；显式设 None 才能真正禁用
        kwargs["proxies"] = {"http": None, "https": None}
        return _orig(url, **kwargs)

    _req.get = _direct_get
    try:
        yield
    finally:
        _req.get = _orig


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _is_stale(cached: pd.DataFrame) -> bool:
    last = pd.Timestamp(cached["date"].max())
    return (_today() - last).days > _STALE_CALENDAR_DAYS


def _cache_path(cache_dir: str | Path, code: str) -> Path:
    return Path(cache_dir) / f"{code}_daily.parquet"


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return data-quality warnings for a normalized OHLCV DataFrame.

    Checks: suspended days (volume=0), large single-day moves (>20%),
    calendar gaps >7 days indicating missing data or long suspension.
    """
    issues: list[str] = []

    zero_vol = int((df["volume"] == 0).sum())
    if zero_vol:
        issues.append(f"检测到 {zero_vol} 根停牌K线(成交量为0)")

    pct = df["close"].pct_change().abs()
    big_moves = int((pct > 0.20).sum())
    if big_moves:
        issues.append(f"{big_moves} 个交易日涨跌幅 >20%(含停牌复牌后异常)")

    if len(df) >= 2:
        diffs = df["date"].sort_values().diff().dropna()
        max_gap = int(diffs.dt.days.max())
        if max_gap > 7:
            issues.append(f"最大日期间隔 {max_gap} 天(疑似长期停牌或数据缺失)")

    return issues


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
            with _no_proxy():
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
        or (cached is not None and not force_refresh and _is_stale(cached))
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


def _fetch_index_from_akshare(symbol: str) -> pd.DataFrame:
    """Fetch full history for a market index (e.g. 'sh000001').

    stock_zh_index_daily already returns English column names:
    date, open, close, high, low, volume.
    """
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            with _no_proxy():
                raw = ak.stock_zh_index_daily(symbol=symbol)
            raw = raw.copy()
            raw["date"] = pd.to_datetime(raw["date"])
            if "volume" not in raw.columns:
                raw["volume"] = 1.0
            out = raw[["date", "open", "high", "low", "close", "volume"]]
            return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        except Exception as e:
            last_err = e
            log.warning("Index fetch attempt %d/%d for %s failed: %s",
                        attempt, len(_RETRY_DELAYS), symbol, e)
            if attempt < len(_RETRY_DELAYS):
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def fetch_index_daily(
    symbol: str,
    history_days: int,
    cache_dir: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return latest `history_days` daily bars for a market index.

    stock_zh_index_daily fetches all history at once (no start_date param),
    so we always replace the cache on a stale hit rather than appending.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"idx_{symbol}.parquet"

    cached: pd.DataFrame | None = None
    if cache_file.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache_file)
        except Exception as e:
            log.warning("Index cache %s corrupt (%s), refetching", cache_file, e)
            cache_file.unlink(missing_ok=True)

    need_fetch = (
        force_refresh
        or cached is None
        or len(cached) < history_days
        or (cached is not None and not force_refresh and _is_stale(cached))
    )

    if need_fetch:
        fresh = _fetch_index_from_akshare(symbol)
        fresh.to_parquet(cache_file, index=False)
        cached = fresh

    assert cached is not None
    return cached.tail(history_days).reset_index(drop=True)


def _fetch_sector_from_akshare(sector_name: str, start: str | None = None) -> pd.DataFrame:
    """Fetch industry board daily history (东方财富).

    stock_board_industry_hist_em returns Chinese column names,
    so we reuse _normalize() for consistent output.
    """
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            with _no_proxy():
                raw = ak.stock_board_industry_hist_em(
                    symbol=sector_name,
                    period="日k",
                    start_date=start or "19900101",
                    end_date="20991231",
                )
            return _normalize(raw)
        except Exception as e:
            last_err = e
            log.warning("Sector fetch attempt %d/%d for '%s' failed: %s",
                        attempt, len(_RETRY_DELAYS), sector_name, e)
            if attempt < len(_RETRY_DELAYS):
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def fetch_sector_daily(
    sector_name: str,
    history_days: int,
    cache_dir: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return latest `history_days` daily bars for an industry sector board."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    safe = sector_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    cache_file = Path(cache_dir) / f"sector_{safe}.parquet"

    cached: pd.DataFrame | None = None
    if cache_file.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache_file)
        except Exception as e:
            log.warning("Sector cache %s corrupt (%s), refetching", cache_file, e)
            cache_file.unlink(missing_ok=True)

    need_fetch = (
        force_refresh
        or cached is None
        or len(cached) < history_days
        or (cached is not None and not force_refresh and _is_stale(cached))
    )

    if need_fetch:
        start = None
        if cached is not None and not force_refresh:
            last = cached["date"].max()
            start = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
        fresh = _fetch_sector_from_akshare(sector_name, start=start)
        if cached is not None and not force_refresh:
            combined = pd.concat([cached, fresh]).drop_duplicates("date").sort_values("date")
        else:
            combined = fresh
        combined = combined.reset_index(drop=True)
        combined.to_parquet(cache_file, index=False)
        cached = combined

    assert cached is not None
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
