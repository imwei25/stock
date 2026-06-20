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


# 增量落盘频率:每抓 N 个 code 把已抓数据 dump 到 partial parquet。
# 200 codes ≈ 每 ~30s-1min 落一次,中断恢复粒度足够细,IO 开销也很小。
_CHECKPOINT_INTERVAL_CODES = 200

# 连续 N 个 code 全部失败(每 code 16 季全是 error_code != "0" 或 empty)
# 就判定 baostock session 已死,abort 整个 fetch。否则之前那种"baostock 早就
# 把我们踢了但代码 silently continue 跑完 4000 codes 拿到一堆空气然后落盘
# 当成功"的事会再发生。
_DEAD_SESSION_CONSECUTIVE_EMPTY_CODES = 20


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

    断点续抓:fetch 中途崩溃/被杀,已抓 codes 写在
    ``<cache_dir>/fundamentals_<table>.partial.parquet``;下次启动从该 partial
    出发,只补未抓的 codes。完整抓完后 partial 自动清理。
    """
    if table not in _VALID_TABLES:
        raise ValueError(
            f"unknown table={table!r}; valid: {_VALID_TABLES}"
        )

    cache_path: Path | None = None
    partial_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"fundamentals_{table}.parquet"
        partial_path = Path(cache_dir) / f"fundamentals_{table}.partial.parquet"

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
        df = _fetch_table(table, codes, partial_path=partial_path)
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
        # partial 被 final 替代,清掉避免之后误用
        if partial_path is not None and partial_path.exists():
            try:
                partial_path.unlink()
            except Exception as e:
                log.warning("could not remove partial cache %s: %s", partial_path, e)
    return df


def _write_partial(rows: list[dict], path: Path) -> None:
    """Snapshot 已抓 rows 到 partial parquet,供下次断点续抓恢复用。

    任意异常都吞掉(只 warning),不能让 checkpoint 失败拖死整个 fetch。
    """
    if not rows:
        return
    try:
        df = pd.DataFrame(rows)
        for col in ("pubDate", "statDate"):
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.dropna(subset=["pubDate"]).reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception as e:
        log.warning("partial checkpoint write failed (%s); continuing fetch", e)


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # pubDate 必须是 datetime64;若被 parquet 推断为 object,显式 cast
    if not pd.api.types.is_datetime64_any_dtype(df["pubDate"]):
        df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
    return df


def _fetch_table(
    table: str,
    codes: list[str] | None,
    partial_path: Path | None = None,
) -> pd.DataFrame:
    """串行调 baostock 拉某张季度表。codes=None → 走 list_universe 全市场。

    每股每季一次 query (5500 × 16 = 88000 calls, 串行约 6-10 min/table)。
    Per-stock 失败 log warning 跳过,不抛出。

    ``partial_path`` 非 None 时:
      1. 启动前若该 partial parquet 存在,载入已抓 rows + 记录已完成 codes,
         本次只抓剩余 codes(断点续抓)。
      2. 每 _CHECKPOINT_INTERVAL_CODES 抓完后把当前 all_rows snapshot 到 partial。
    """
    import baostock as bs

    if codes is None:
        # 走 fetcher 的 list_universe (returns DataFrame with `code` column)
        from stockpool.fetcher import list_universe
        universe_df = list_universe()
        codes = universe_df["code"].astype(str).str.zfill(6).tolist()

    # 断点续抓:加载 partial 已抓部分
    completed_codes: set[str] = set()
    all_rows: list[dict] = []
    if partial_path is not None and partial_path.exists():
        try:
            partial_df = pd.read_parquet(partial_path)
            completed_codes = set(
                partial_df["code"].astype(str).str.zfill(6).unique()
            )
            all_rows = partial_df.to_dict(orient="records")
            log.info(
                "fundamentals(%s): 断点续抓,partial 含 %d codes",
                table, len(completed_codes),
            )
        except Exception as e:
            log.warning(
                "fundamentals(%s): partial cache unreadable (%s); 从头开始",
                table, e,
            )
            completed_codes = set()
            all_rows = []

    todo_codes = [c for c in codes if c not in completed_codes]
    log.info(
        "fundamentals(%s): 本次需抓 %d codes(skip %d 已在 partial 中)",
        table, len(todo_codes), len(completed_codes),
    )
    if not todo_codes and all_rows:
        # 已抓完整,但没走到 final 落盘那一步;直接把 partial 当结果返回
        log.info("fundamentals(%s): partial 已含全部 codes,直接返回", table)
        df = pd.DataFrame(all_rows)
        for col in ("pubDate", "statDate"):
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df.dropna(subset=["pubDate"]).reset_index(drop=True)

    fn_name = _TABLE_TO_BS_FN[table]
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        fn = getattr(bs, fn_name)
        all_fields: list[str] | None = None
        # 拉最近 16 季(过去 4 年)
        today = pd.Timestamp.today()
        quarters = _recent_quarters(today, n=16)
        total = len(todo_codes) + len(completed_codes)
        try:
            from tqdm import tqdm
            code_iter = tqdm(
                todo_codes,
                desc=f"baostock {table}",
                unit="code",
                mininterval=2.0,
                initial=len(completed_codes),
                total=total,
            )
        except ImportError:
            code_iter = todo_codes
        consecutive_empty_codes = 0
        for i, code in enumerate(code_iter, start=1):
            bs_code = _to_bs_code(code)
            rows_for_code = 0
            for year, q in quarters:
                try:
                    rs = fn(code=bs_code, year=year, quarter=q)
                    if rs.error_code != "0":
                        # 之前是 silent continue;现在记 warning 但不抛(单季度
                        # 失败可能就是该公司还没披露,不算异常)
                        continue
                    if all_fields is None and rs.fields:
                        all_fields = list(rs.fields)
                    while rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        # 标准化 code 为 6 位
                        row["code"] = code
                        all_rows.append(row)
                        rows_for_code += 1
                except Exception as e:
                    log.warning("baostock %s %s %dQ%d failed: %s",
                                fn_name, bs_code, year, q, e)
            # Dead-session detector:某 code 16 季全部空 → 计数;连续 N 个
            # code 全空就判定 baostock 把我们踢了,abort 整个 fetch
            # 让 caller 报错。
            if rows_for_code == 0:
                consecutive_empty_codes += 1
                if consecutive_empty_codes >= _DEAD_SESSION_CONSECUTIVE_EMPTY_CODES:
                    log.error(
                        "fundamentals(%s): baostock session looks dead — "
                        "last %d codes returned zero rows; aborting fetch. "
                        "Partial cache preserved at %s for next retry.",
                        table, consecutive_empty_codes, partial_path,
                    )
                    if partial_path is not None:
                        _write_partial(all_rows, partial_path)
                    raise RuntimeError(
                        f"baostock {table} session dead after "
                        f"{consecutive_empty_codes} consecutive empty codes"
                    )
            else:
                consecutive_empty_codes = 0
            # 每 N 个 code 落一次 partial,中断恢复粒度足够细
            if partial_path is not None and i % _CHECKPOINT_INTERVAL_CODES == 0:
                _write_partial(all_rows, partial_path)
    finally:
        bs.logout()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # pubDate / statDate 转 datetime
    for col in ("pubDate", "statDate"):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
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
