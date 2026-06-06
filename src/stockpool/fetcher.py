"""数据获取 + Parquet 本地缓存。

后端可选: mootdx(默认, 通达信 TCP) / baostock / akshare。
板块(行业)K 线仅 akshare 直接提供,因此 fetch_sector_daily 始终走 akshare。
"""
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Literal

import akshare as ak
import pandas as pd

Source = Literal["mootdx", "baostock", "akshare"]

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


def _last_business_day(today: pd.Timestamp) -> pd.Timestamp:
    """今天本身若是工作日则返回今天,否则回退到上一个工作日(周末近似,不查节假日)。"""
    if today.weekday() < 5:
        return today
    return (today - pd.offsets.BDay(1)).normalize()


def _is_stale(cached: pd.DataFrame) -> bool:
    last = pd.Timestamp(cached["date"].max()).normalize()
    return last < _last_business_day(_today())


def _cache_path(cache_dir: str | Path, code: str) -> Path:
    return Path(cache_dir) / f"{code}_daily.parquet"


def _source_marker_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / ".data_source"


def check_source_change(cache_dir: str | Path, source: Source) -> bool:
    """Return True if cached data was last written by a different source.

    Returns False when no marker exists (first-time use — nothing to invalidate)
    or when the marker matches ``source``. Mootdx/baostock/akshare disagree on
    volume units and adjustment rules; mixing them in one parquet silently
    corrupts liquidity / return calculations downstream, so any mismatch
    means the existing cache for this directory must be discarded.
    """
    marker = _source_marker_path(cache_dir)
    if not marker.exists():
        return False
    try:
        prev = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return prev != source


def update_source_marker(cache_dir: str | Path, source: Source) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _source_marker_path(cache_dir).write_text(source, encoding="utf-8")


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return data-quality warnings for a normalized OHLCV DataFrame.

    Checks: suspended days (volume=0), large single-day moves (>20%),
    calendar gaps >7 days indicating missing data or long suspension.
    """
    issues: list[str] = []

    zero_vol = int((df["volume"] == 0).sum())
    if zero_vol:
        issues.append(f"检测到 {zero_vol} 根停牌K线(成交量为0)")

    pct = df["close"].pct_change(fill_method=None).abs()
    big_moves = int((pct > 0.20).sum())
    if big_moves:
        issues.append(f"{big_moves} 个交易日涨跌幅 >20%(含停牌复牌后异常)")

    if len(df) >= 2:
        diffs = df["date"].sort_values().diff().dropna()
        max_gap = int(diffs.dt.days.max())
        # A 股春节最长可断 11 天(如 2024-02-08 → 2024-02-19),国庆+中秋
        # 合并最长 11 天(2023);阈值放到 >14 才报,只剩真正的长期停牌
        # 或数据缺口能触发。
        if max_gap > 14:
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


def _dispatch_stock(source: Source, code: str, start: str | None) -> pd.DataFrame:
    if source == "akshare":
        return _fetch_from_akshare(code, start=start)
    if source == "mootdx":
        from stockpool.data_sources import mootdx_backend
        return mootdx_backend.fetch_stock(code, start=start)
    if source == "baostock":
        from stockpool.data_sources import baostock_backend
        return baostock_backend.fetch_stock(code, start=start)
    raise ValueError(f"unknown data source: {source}")


def _dispatch_index(source: Source, symbol: str) -> pd.DataFrame:
    if source == "akshare":
        return _fetch_index_from_akshare(symbol)
    if source == "mootdx":
        from stockpool.data_sources import mootdx_backend
        return mootdx_backend.fetch_index(symbol)
    if source == "baostock":
        from stockpool.data_sources import baostock_backend
        return baostock_backend.fetch_index(symbol)
    raise ValueError(f"unknown data source: {source}")


def fetch_daily(
    code: str,
    history_days: int,
    cache_dir: str | Path,
    force_refresh: bool = False,
    source: Source = "akshare",
    warmup_days: int = 0,
) -> pd.DataFrame:
    """Return latest ``history_days + warmup_days`` daily K bars (English column names).

    The ``warmup_days`` prefix lets long rolling factors warm up before the
    official backtest window (callers trim those bars before equity-curve
    iteration). Default 0 keeps backward compatibility.

    Uses local Parquet cache, only triggers incremental fetch when needed.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if check_source_change(cache_dir, source):
        log.warning(
            "Data source changed (cache=%s → cfg=%s); discarding %s and refetching.",
            _source_marker_path(cache_dir).read_text(encoding="utf-8").strip(),
            source, code,
        )
        force_refresh = True
    update_source_marker(cache_dir, source)
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
        or len(cached) < (history_days + warmup_days)
        or (cached is not None and not force_refresh and _is_stale(cached))
    )

    if need_fetch:
        start = None
        if cached is not None and not force_refresh:
            last = cached["date"].max()
            start = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
        fresh = _dispatch_stock(source, code, start=start)
        if cached is not None and not force_refresh:
            combined = pd.concat([cached, fresh]).drop_duplicates("date").sort_values("date")
        else:
            combined = fresh
        combined = combined.reset_index(drop=True)
        combined.to_parquet(cache_file, index=False)
        cached = combined

    return cached.tail(history_days + warmup_days).reset_index(drop=True)


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
    source: Source = "akshare",
    warmup_days: int = 0,
) -> pd.DataFrame:
    """Return latest ``history_days + warmup_days`` daily bars for a market index.

    The ``warmup_days`` prefix lets long rolling factors warm up before the
    official backtest window. Default 0 keeps backward compatibility.

    stock_zh_index_daily fetches all history at once (no start_date param),
    so we always replace the cache on a stale hit rather than appending.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if check_source_change(cache_dir, source):
        force_refresh = True
    update_source_marker(cache_dir, source)
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
        or len(cached) < (history_days + warmup_days)
        or (cached is not None and not force_refresh and _is_stale(cached))
    )

    if need_fetch:
        fresh = _dispatch_index(source, symbol)
        # 增量合并(mootdx/baostock 都能拿到全量或长尾,直接覆盖也安全)
        if cached is not None and not force_refresh:
            fresh = pd.concat([cached, fresh]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        fresh.to_parquet(cache_file, index=False)
        cached = fresh

    assert cached is not None
    return cached.tail(history_days + warmup_days).reset_index(drop=True)


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


def _dispatch_sector(source: Source, sector_name: str, start: str | None) -> pd.DataFrame:
    # akshare 走旧路径(stock_board_industry_hist_em);mootdx/baostock 都走 mootdx
    # 的通达信行业指数 (88xxxx),后者通过 client.index 拉取,非常稳定。
    if source == "akshare":
        return _fetch_sector_from_akshare(sector_name, start=start)
    from stockpool.data_sources import mootdx_backend
    out = mootdx_backend.fetch_sector(sector_name)
    if start is not None:
        out = out[out["date"] >= pd.to_datetime(start)].reset_index(drop=True)
    return out


def fetch_sector_daily(
    sector_name: str,
    history_days: int,
    cache_dir: str | Path,
    force_refresh: bool = False,
    source: Source = "akshare",
    warmup_days: int = 0,
) -> pd.DataFrame:
    """Return latest ``history_days + warmup_days`` daily bars for an industry sector board.

    The ``warmup_days`` prefix lets long rolling factors warm up before the
    official backtest window. Default 0 keeps backward compatibility.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if check_source_change(cache_dir, source):
        force_refresh = True
    update_source_marker(cache_dir, source)
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
        or len(cached) < (history_days + warmup_days)
        or (cached is not None and not force_refresh and _is_stale(cached))
    )

    if need_fetch:
        start = None
        if cached is not None and not force_refresh:
            last = cached["date"].max()
            start = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
        fresh = _dispatch_sector(source, sector_name, start=start)
        if cached is not None and not force_refresh:
            combined = pd.concat([cached, fresh]).drop_duplicates("date").sort_values("date")
        else:
            combined = fresh
        combined = combined.reset_index(drop=True)
        combined.to_parquet(cache_file, index=False)
        cached = combined

    assert cached is not None
    return cached.tail(history_days + warmup_days).reset_index(drop=True)


def list_universe(source: Source = "mootdx") -> pd.DataFrame:
    """List all A-share stocks for the training universe.

    Currently only mootdx is supported (TCP, ~50ms per call, no extra deps).
    Returns DataFrame with columns: code, name, market.
    """
    if source != "mootdx":
        raise NotImplementedError(
            f"list_universe currently only supports mootdx (got {source!r})"
        )
    from stockpool.data_sources import mootdx_backend
    return mootdx_backend.list_a_shares()


def fetch_universe(
    codes: list[str],
    history_days: int,
    cache_dir: str | Path,
    source: Source = "mootdx",
    force_refresh: bool = False,
    max_workers: int = 8,
    progress_every: int = 200,
) -> dict[str, pd.DataFrame]:
    """Bulk-fetch daily bars for many stocks in parallel, with per-stock caching.

    Each stock reuses the same parquet cache as `fetch_daily`, so subsequent
    calls only do incremental updates. Errors on individual stocks are logged
    and skipped — the returned dict contains only successful pulls.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Resolve source-change once *before* spawning workers so all threads see
    # the same force_refresh decision (avoids a race where the first worker
    # updates the marker and later workers think the cache is still valid).
    if check_source_change(cache_dir, source):
        log.warning(
            "Data source changed (cache=%s → cfg=%s); forcing full refresh of universe.",
            _source_marker_path(cache_dir).read_text(encoding="utf-8").strip(),
            source,
        )
        force_refresh = True
    update_source_marker(cache_dir, source)
    out: dict[str, pd.DataFrame] = {}
    failures: list[tuple[str, str]] = []

    def _one(code: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            df = fetch_daily(
                code, history_days, cache_dir,
                force_refresh=force_refresh, source=source,
            )
            return code, df, None
        except Exception as e:  # noqa: BLE001
            return code, None, str(e)

    total = len(codes)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, c): c for c in codes}
        for fut in as_completed(futures):
            code, df, err = fut.result()
            done += 1
            if err is not None or df is None:
                failures.append((code, err or "empty"))
            else:
                out[code] = df
            if done % progress_every == 0 or done == total:
                log.info("fetch_universe progress: %d/%d (ok=%d fail=%d)",
                         done, total, len(out), len(failures))

    if failures:
        log.warning("fetch_universe: %d failures (first 5): %s",
                    len(failures), failures[:5])
    return out


def load_universe_cache(
    cache_dir: str | Path,
    history_days: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every cached ``<code>_daily.parquet`` under ``cache_dir`` into memory.

    Skips files that fail to read. Used by ML strategies when
    ``training_universe='all'`` so the training pool comes from the previously-
    fetched full A-share cache, decoupled from the application stock pool.
    """
    cache = Path(cache_dir)
    if not cache.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    for path in cache.glob("*_daily.parquet"):
        code = path.stem.replace("_daily", "")
        try:
            df = pd.read_parquet(path)
            if history_days is not None and len(df) > history_days:
                df = df.tail(history_days).reset_index(drop=True)
            out[code] = df
        except Exception as e:
            log.warning("Universe cache: failed to read %s (%s)", path, e)
    return out


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
