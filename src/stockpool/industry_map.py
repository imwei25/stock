"""code → 行业映射 (Pool B 用).

A 股的 ``universe.parquet`` 只有 ``code/name/market``,没有行业字段。mootdx
有行业指数 (88xxxx) 但反向(stock → industry)不可用:TDX 服务器对
``block_hy.dat`` 返回 0 字节,触发 tdxpy 的空 bytearray bug。

可用的数据源:

* **baostock** (默认推荐): ``bs.query_stock_industry()`` 一次性返回所有
  A 股 + 证监会行业分类 (5500+ 行, 84 个行业, ~5-10 秒)。无 token、无代理
  依赖,稳定性最好。code 自带 ``sh./sz.`` 前缀需剥离。
* **akshare**: ``ak.stock_board_industry_*_em`` 走东财 HTTP,逐板块拉,
  ~1-2 分钟。受代理 / 网络抖动影响大。
* **auto**: 先 baostock,失败再 akshare,都失败返回 ``{}``。

汇总成 ``DataFrame(code, industry)``,写到 ``data/stock_industry_map.parquet``。
默认 30 天有效期,过期自动重拉。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import pandas as pd

log = logging.getLogger(__name__)

_CACHE_FILENAME = "stock_industry_map.parquet"
_UNKNOWN = "未知"

IndustrySource = Literal["auto", "baostock", "akshare"]


def load_or_build_industry_map(
    cache_dir: str | Path,
    max_age_days: int = 30,
    force_refresh: bool = False,
    source: IndustrySource = "auto",
) -> dict[str, str]:
    """Return ``{code: industry_name}`` for all A-share stocks.

    Cache: ``<cache_dir>/stock_industry_map.parquet``, mtime-based staleness.

    ``source`` selects the fetcher:

    * ``"auto"`` (default) — try baostock first, then akshare.
    * ``"baostock"`` — only baostock.
    * ``"akshare"`` — only akshare.

    Returns ``{}`` if all selected sources fail; callers should bucket those
    stocks as ``"未知"``.
    """
    cache_path = Path(cache_dir) / _CACHE_FILENAME

    if not force_refresh and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400.0
        if age_days <= max_age_days:
            try:
                df = pd.read_parquet(cache_path)
                return _df_to_dict(df)
            except Exception as e:
                log.warning("Industry map cache corrupt (%s), rebuilding", e)
        else:
            log.info("Industry map cache stale (%.1f d > %d d), rebuilding",
                     age_days, max_age_days)

    df = _dispatch_fetch(source)
    if df is None or df.empty:
        return {}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("Industry map written: %s (%d codes)", cache_path, len(df))
    return _df_to_dict(df)


def industry_of(code: str, mapping: dict[str, str]) -> str:
    """Return industry for ``code`` (6-digit), or ``"未知"`` if unmapped."""
    return mapping.get(code, _UNKNOWN)


def _df_to_dict(df: pd.DataFrame) -> dict[str, str]:
    """Convert (code, industry) DataFrame to a normalized dict."""
    return {str(r.code).zfill(6): str(r.industry)
            for r in df.itertuples(index=False)
            if str(r.industry).strip()}


def _dispatch_fetch(source: IndustrySource) -> pd.DataFrame | None:
    """Run the selected fetcher chain. Returns None if everything fails."""
    if source == "baostock":
        return _try("baostock", _fetch_from_baostock)
    if source == "akshare":
        return _try("akshare", _fetch_from_akshare)
    # auto: baostock first (faster + more stable on most networks), then akshare
    df = _try("baostock", _fetch_from_baostock)
    if df is not None and not df.empty:
        return df
    log.info("Industry map: baostock yielded nothing, falling back to akshare")
    return _try("akshare", _fetch_from_akshare)


def _try(label: str, fn) -> pd.DataFrame | None:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        log.error("Industry map: %s source failed: %s", label, e)
        return None


def _fetch_from_baostock() -> pd.DataFrame:
    """``bs.query_stock_industry`` — 一次性返回全 A 股证监会行业分类.

    code 形如 ``sh.600000``,剥离 ``sh./sz.`` 前缀。空 industry 字段
    (退市/次新股) 被丢弃 — 让调用方回退到 "未知" 桶。
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            raise RuntimeError(f"baostock query failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()

    out = pd.DataFrame({
        "code": df["code"].str.split(".").str[-1].str.zfill(6),
        "industry": df["industry"].astype(str),
    })
    out = out[out["industry"].str.strip() != ""].reset_index(drop=True)
    log.info("Industry map (baostock): %d codes, %d industries",
             len(out), out["industry"].nunique())
    return out


def _fetch_from_akshare() -> pd.DataFrame:
    """Pull every east-money industry board + its constituents.

    Stocks in multiple boards keep first-seen (board iteration order =
    east-money default, market-cap desc).
    """
    import akshare as ak

    boards = ak.stock_board_industry_name_em()
    name_col = _pick_column(boards, ["板块名称", "name", "板块"])
    board_names = boards[name_col].astype(str).tolist()
    log.info("Industry map (akshare): pulling %d boards ...", len(board_names))

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    failed: list[tuple[str, str]] = []

    for i, board in enumerate(board_names, 1):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=board)
        except Exception as e:  # noqa: BLE001
            failed.append((board, str(e)))
            continue
        code_col = _pick_column(cons, ["代码", "code", "股票代码"])
        for raw in cons[code_col].astype(str):
            code = raw.zfill(6)
            if code in seen:
                continue
            seen.add(code)
            rows.append((code, board))
        if i % 10 == 0 or i == len(board_names):
            log.info("Industry map progress: %d/%d boards, %d codes mapped",
                     i, len(board_names), len(rows))

    if failed:
        log.warning("Industry map: %d boards failed (first 3): %s",
                    len(failed), failed[:3])

    return pd.DataFrame(rows, columns=["code", "industry"])


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name from ``candidates`` that exists in ``df``."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"none of {candidates!r} found in DataFrame columns {list(df.columns)!r}"
    )
