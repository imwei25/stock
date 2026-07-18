"""code → IPO 上市日期映射 (panel.compute_tradability_mask 用).

`_listing_mask` 当 `ipo_dates=None` 时退化到 first_valid_index 启发式 ——
但这个启发式会把"缓存历史短"的成熟股(panel union 比该股缓存起点早)
错认成"新上市股",对它前 252 个 panel row 错误标记 mask=False。

真实解法:用 baostock `query_stock_basic` 拉**实际 IPO 日期**,缓存到
``data/ipo_dates.parquet``。30 天有效期,过期自动重拉。

baostock 一次返回全部 A 股 (~5500 行,~3-5 秒),code 自带 ``sh./sz./bj.``
前缀需剥离。

2026-07-18(L1 收尾):新增 akshare fallback(``stock_info_sh_name_code`` /
``stock_info_sz_name_code`` / ``stock_info_bj_name_code`` 三张交易所名录,
含上市日期)。``auto``(默认)= baostock → akshare 链,与 ``industry_map``
同范式 —— baostock 账号被黑名单时 ipo_dates 仍可建,``ab-pool build`` 的
IPO 硬过滤不再被跳过。
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_CACHE_FILENAME = "ipo_dates.parquet"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_or_build_ipo_dates(
    cache_dir: str | Path,
    max_age_days: int = 30,
    force_refresh: bool = False,
    source: str = "auto",
) -> dict[str, pd.Timestamp]:
    """Return ``{code: ipo_timestamp}`` for all A-share stocks.

    ``source``: ``"auto"`` (default, baostock → akshare fallback chain),
    ``"baostock"``, or ``"akshare"``.

    Cache: ``<cache_dir>/ipo_dates.parquet``, mtime-based staleness.
    Returns ``{}`` if every source fails and no cache available; callers
    fall back to ``_listing_mask`` heuristic (with log warning).
    """
    cache_path = Path(cache_dir) / _CACHE_FILENAME

    if not force_refresh and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400.0
        if age_days <= max_age_days:
            try:
                df = pd.read_parquet(cache_path)
                return _df_to_dict(df)
            except Exception as e:
                log.warning("IPO date cache corrupt (%s), rebuilding", e)
        else:
            log.info("IPO date cache stale (%.1f d > %d d), rebuilding",
                     age_days, max_age_days)

    df = _fetch(source)
    if df is None:
        log.error("IPO date fetch failed on every source (%s)", source)
        if cache_path.exists():
            log.info("Using stale IPO date cache as fallback")
            try:
                return _df_to_dict(pd.read_parquet(cache_path))
            except Exception:
                pass
        return {}

    if df.empty:
        return {}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("IPO date map written: %s (%d codes)", cache_path, len(df))
    return _df_to_dict(df)


def _fetch(source: str) -> pd.DataFrame | None:
    """Dispatch by source; ``auto`` = baostock → akshare chain.

    Returns ``None`` when every attempted source raises or yields empty.
    """
    chain = {
        "auto": (("baostock", _fetch_from_baostock),
                 ("akshare", _fetch_from_akshare)),
        "baostock": (("baostock", _fetch_from_baostock),),
        "akshare": (("akshare", _fetch_from_akshare),),
    }.get(source)
    if chain is None:
        raise ValueError(f"unknown ipo_dates source: {source!r}")
    for name, fn in chain:
        try:
            df = fn()
        except Exception as e:
            log.warning("IPO date fetch via %s failed: %s", name, e)
            continue
        if df is not None and not df.empty:
            return df
        log.warning("IPO date fetch via %s returned no rows", name)
    return None


def _df_to_dict(df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Convert (code, ipo_date) DataFrame → ``{code: Timestamp}``."""
    return {
        str(r.code).zfill(6): pd.Timestamp(r.ipo_date)
        for r in df.itertuples(index=False)
        if pd.notna(r.ipo_date)
    }


def _fetch_from_baostock() -> pd.DataFrame:
    """``bs.query_stock_basic`` — 一次返回全 A 股 (code, code_name, ipoDate,
    outDate, type, status)。

    Filters:
      - type=1 (stock,丢指数/其他)
      - ipoDate 形如 ``YYYY-MM-DD``(丢空字符串)

    Returns DataFrame(code, ipo_date) with stripped prefix.
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            raise RuntimeError(f"baostock query_stock_basic failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()

    # 只要股票 (type=1),保留可解析的 ipoDate
    df = df[df["type"] == "1"].copy()
    df = df[df["ipoDate"].apply(lambda s: bool(_DATE_RE.match(str(s))))]
    out = pd.DataFrame({
        "code": df["code"].str.split(".").str[-1].str.zfill(6),
        "ipo_date": pd.to_datetime(df["ipoDate"]),
    }).reset_index(drop=True)
    log.info("IPO date map (baostock): %d codes, earliest=%s, latest=%s",
             len(out), out["ipo_date"].min().date(), out["ipo_date"].max().date())
    return out


# akshare exchange listings: (fetcher, code-column candidates, date-column
# candidates). Column names differ per exchange and shift across akshare
# versions, hence the candidate lists.
_AK_SOURCES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("stock_info_sh_name_code", ("证券代码", "COMPANY_CODE"), ("上市日期", "LISTING_DATE")),
    ("stock_info_sz_name_code", ("A股代码",), ("A股上市日期",)),
    ("stock_info_bj_name_code", ("证券代码",), ("上市日期",)),
)


def _fetch_from_akshare() -> pd.DataFrame:
    """三张交易所名录(沪/深/北)拼 IPO 日期;任一交易所失败只丢那一张。

    Returns DataFrame(code, ipo_date);全部失败时 raise(由 ``_fetch``
    的链式 fallback 捕获)。
    """
    import akshare as ak

    frames: list[pd.DataFrame] = []
    for fn_name, code_cols, date_cols in _AK_SOURCES:
        try:
            raw = getattr(ak, fn_name)()
        except Exception as e:
            log.warning("akshare %s failed: %s", fn_name, e)
            continue
        code_col = next((c for c in code_cols if c in raw.columns), None)
        date_col = next((c for c in date_cols if c in raw.columns), None)
        if code_col is None or date_col is None:
            log.warning("akshare %s: expected columns missing (has %s)",
                        fn_name, list(raw.columns)[:8])
            continue
        part = pd.DataFrame({
            "code": raw[code_col].astype(str).str.zfill(6),
            "ipo_date": pd.to_datetime(raw[date_col], errors="coerce"),
        })
        frames.append(part.dropna(subset=["ipo_date"]))
    if not frames:
        raise RuntimeError("akshare IPO date fetch: all exchange listings failed")
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["code"], keep="first")
        .reset_index(drop=True)
    )
    log.info("IPO date map (akshare): %d codes, earliest=%s, latest=%s",
             len(out), out["ipo_date"].min().date(), out["ipo_date"].max().date())
    return out
