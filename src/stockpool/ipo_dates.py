"""code → IPO 上市日期映射 (panel.compute_tradability_mask 用).

`_listing_mask` 当 `ipo_dates=None` 时退化到 first_valid_index 启发式 ——
但这个启发式会把"缓存历史短"的成熟股(panel union 比该股缓存起点早)
错认成"新上市股",对它前 252 个 panel row 错误标记 mask=False。

真实解法:用 baostock `query_stock_basic` 拉**实际 IPO 日期**,缓存到
``data/ipo_dates.parquet``。30 天有效期,过期自动重拉。

baostock 一次返回全部 A 股 (~5500 行,~3-5 秒),code 自带 ``sh./sz./bj.``
前缀需剥离。
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_CACHE_FILENAME = "ipo_dates.parquet"
_BASICS_FILENAME = "stock_basics.parquet"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_or_build_stock_basics(
    cache_dir: str | Path,
    max_age_days: int = 30,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """全 A 股基础名单(PIT 轻量版,P0-4):code/name/ipo_date/out_date/status/is_st。

    与 ``load_or_build_ipo_dates`` 同源(baostock ``query_stock_basic``),但
    保留 **干净中文名**(mootdx 名是双重编码乱码,P3-4)、**outDate/status**
    (已退市票也在表里 —— 这是构建历史时点名单的基础)和 **is_st** 标记
    (name 含 ST/*ST/PT)。缓存 ``<cache_dir>/stock_basics.parquet``,30 天。

    注意:is_st 是**当前快照**,按它回溯历史仍有轻微前视(改名时点未知);
    训练池已不再整段剔除 ST(见 fetch-universe),该标记用于
    ① 涨跌停 mask 的 ±5% 阈值 ② 应用层(Pool B/推荐)的当下剔除 —— 两者
    都是"以当下决策"的场景,无前视问题。
    """
    cache_path = Path(cache_dir) / _BASICS_FILENAME

    if not force_refresh and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400.0
        if age_days <= max_age_days:
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                log.warning("stock_basics cache corrupt (%s), rebuilding", e)

    try:
        df = _fetch_basics_from_baostock()
    except Exception as e:
        log.error("stock_basics fetch failed: %s", e)
        if cache_path.exists():
            log.info("Using stale stock_basics cache as fallback")
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass
        return pd.DataFrame(
            columns=["code", "name", "ipo_date", "out_date", "status", "is_st"]
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("stock_basics written: %s (%d codes, %d ST, %d delisted)",
             cache_path, len(df), int(df["is_st"].sum()),
             int(df["out_date"].notna().sum()))
    return df


def load_stock_basics_cached_only(cache_dir: str | Path) -> pd.DataFrame:
    """只读缓存版:parquet 不存在/损坏时返回空表,**绝不发起网络请求**。

    Pool B / 策略 mask 等高频路径用这个 —— 名单的构建责任在
    ``fetch-universe``(它调用 ``load_or_build_stock_basics``)。
    """
    cache_path = Path(cache_dir) / _BASICS_FILENAME
    if not cache_path.exists():
        return pd.DataFrame(
            columns=["code", "name", "ipo_date", "out_date", "status", "is_st"]
        )
    try:
        return pd.read_parquet(cache_path)
    except Exception as e:  # noqa: BLE001
        log.warning("stock_basics cache unreadable (%s)", e)
        return pd.DataFrame(
            columns=["code", "name", "ipo_date", "out_date", "status", "is_st"]
        )


def load_st_codes(cache_dir: str | Path) -> set[str]:
    """当前名称含 ST/PT 的代码集合(干净 baostock 名,P3-4)。

    只读缓存,不碰网络;缓存缺失返回空集(调用方按板块阈值/名称匹配兜底)。
    """
    df = load_stock_basics_cached_only(cache_dir)
    if df.empty:
        return set()
    return set(df.loc[df["is_st"], "code"].astype(str))


def _fetch_basics_from_baostock() -> pd.DataFrame:
    """query_stock_basic 全量 → 标准化 DataFrame(仅 type=1 股票)。"""
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

    df = df[df["type"] == "1"].copy()
    name = df["code_name"].astype(str)
    out = pd.DataFrame({
        "code": df["code"].str.split(".").str[-1].str.zfill(6),
        "name": name,
        "ipo_date": pd.to_datetime(df["ipoDate"], errors="coerce"),
        "out_date": pd.to_datetime(
            df["outDate"].replace("", pd.NA), errors="coerce"),
        "status": df["status"].astype(str),  # "1"=上市 "0"=退市
        # PT 是早年的退市预警标记,与 ST 同语义对待
        "is_st": name.str.upper().str.contains("ST|PT", regex=True, na=False),
    }).reset_index(drop=True)
    return out


def load_or_build_ipo_dates(
    cache_dir: str | Path,
    max_age_days: int = 30,
    force_refresh: bool = False,
) -> dict[str, pd.Timestamp]:
    """Return ``{code: ipo_timestamp}`` for all A-share stocks via baostock.

    Cache: ``<cache_dir>/ipo_dates.parquet``, mtime-based staleness.
    Returns ``{}`` if baostock fails and no cache available; callers fall
    back to ``_listing_mask`` heuristic (with log warning).
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

    try:
        df = _fetch_from_baostock()
    except Exception as e:
        log.error("IPO date fetch failed: %s", e)
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
