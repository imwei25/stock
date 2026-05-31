"""baostock 5 张季度财务表的 PIT 缓存层。

参考 ``stockpool.ipo_dates`` 的 baostock login + parquet cache + mtime
staleness 模式。每张表缓存到 ``<cache_dir>/fundamentals_<table>.parquet``。

PIT 设计:long-form DataFrame 保留 ``pubDate`` 字段,factor 计算时按
``pubDate`` 而非 ``statDate`` 前向填充到日频(防 ~1 个月未来泄露)。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_VALID_TABLES = ("profit", "growth", "balance", "cash_flow", "dupont")
_TABLE_TO_BS_FN = {
    "profit": "query_profit_data",
    "growth": "query_growth_data",
    "balance": "query_balance_data",
    "cash_flow": "query_cash_flow_data",
    "dupont": "query_dupont_data",
}


def load_or_build_fundamentals(
    table: str,
    *,
    codes: list[str] | None = None,
    cache_dir: str | Path | None = None,
    max_age_days: int = 30,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """返回 long-form DataFrame: code / pubDate / statDate / <fields...>.

    Args:
        table: 五选一: profit / growth / balance / cash_flow / dupont
        codes: None → 拉全市场;否则只拉指定 6 位 code 列表
        cache_dir: 缓存目录;None → 不缓存(纯获取)
        max_age_days: 缓存有效期
        force_refresh: True 时无条件重拉

    Returns:
        long-form DataFrame, 每股每季一行。pubDate 是 datetime64。
        失败 + 无缓存时返回 empty DataFrame。
    """
    if table not in _VALID_TABLES:
        raise ValueError(
            f"unknown table={table!r}; valid: {_VALID_TABLES}"
        )

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"fundamentals_{table}.parquet"

        if not force_refresh and cache_path.exists():
            age = (time.time() - cache_path.stat().st_mtime) / 86400.0
            if age <= max_age_days:
                try:
                    return _read_cache(cache_path)
                except Exception as e:
                    log.warning("fundamentals cache corrupt (%s), rebuilding", e)
            else:
                log.info("fundamentals(%s) cache stale (%.1f d > %d d)",
                         table, age, max_age_days)

    try:
        df = _fetch_table(table, codes)
    except Exception as e:
        log.error("fundamentals(%s) fetch failed: %s", table, e)
        if cache_path is not None and cache_path.exists():
            log.info("fundamentals(%s): using stale cache", table)
            try:
                return _read_cache(cache_path)
            except Exception:
                pass
        return pd.DataFrame()

    if df.empty:
        return df

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log.info("fundamentals(%s) cache written: %s (%d rows)",
                 table, cache_path, len(df))
    return df


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # pubDate 必须是 datetime64;若被 parquet 推断为 object,显式 cast
    if not pd.api.types.is_datetime64_any_dtype(df["pubDate"]):
        df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
    return df


def _fetch_table(table: str, codes: list[str] | None) -> pd.DataFrame:
    """串行调 baostock 拉某张季度表。codes=None → 走 list_universe 全市场。

    每股每季一次 query (5500 × 16 = 88000 calls, 串行约 6-10 min/table)。
    Per-stock 失败 log warning 跳过,不抛出。
    """
    import baostock as bs

    if codes is None:
        # 走 fetcher 的 list_universe (returns DataFrame with `code` column)
        from stockpool.fetcher import list_universe
        universe_df = list_universe()
        codes = universe_df["code"].astype(str).str.zfill(6).tolist()

    fn_name = _TABLE_TO_BS_FN[table]
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        fn = getattr(bs, fn_name)
        all_rows: list[dict] = []
        all_fields: list[str] | None = None
        # 拉最近 16 季(过去 4 年)
        today = pd.Timestamp.today()
        quarters = _recent_quarters(today, n=16)
        for code in codes:
            bs_code = _to_bs_code(code)
            for year, q in quarters:
                try:
                    rs = fn(code=bs_code, year=year, quarter=q)
                    if rs.error_code != "0":
                        continue
                    if all_fields is None and rs.fields:
                        all_fields = list(rs.fields)
                    while rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        # 标准化 code 为 6 位
                        row["code"] = code
                        all_rows.append(row)
                except Exception as e:
                    log.warning("baostock %s %s %dQ%d failed: %s",
                                fn_name, bs_code, year, q, e)
    finally:
        bs.logout()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # pubDate / statDate 转 datetime
    for col in ("pubDate", "statDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["pubDate"]).reset_index(drop=True)
    log.info("fundamentals(%s): %d rows fetched across %d codes",
             table, len(df), df["code"].nunique())
    return df


def _to_bs_code(code: str) -> str:
    """6 位 code → baostock 格式 (sh./sz./bj.)."""
    if code.startswith(("60", "68")):
        return f"sh.{code}"
    if code.startswith(("00", "30")):
        return f"sz.{code}"
    if code.startswith(("8", "43")):
        return f"bj.{code}"
    return f"sh.{code}"  # 兜底


def _recent_quarters(today: pd.Timestamp, n: int = 16) -> list[tuple[int, int]]:
    """返回最近 n 个 (year, quarter) tuple,降序。"""
    quarters = []
    y, q = today.year, ((today.month - 1) // 3) + 1
    for _ in range(n):
        quarters.append((y, q))
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return quarters
