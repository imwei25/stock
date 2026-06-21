"""AB candidate pool — stratified ~100-stock pool for AB tests.

Build: industry-stratified top-2-mcap + top-2-liquidity selection from
universe.parquet. 流通市值 (free-float market cap) is computed locally
as ``liqaShare × latest_close`` using the baostock profit table
(``data/fundamentals_profit.parquet``) — no akshare network call.
Persisted to ``data/ab_pool.parquet``; static unless rebuilt by hand.

See docs/superpowers/specs/2026-06-06-ab-candidate-pool-design.md.
"""
from __future__ import annotations

import logging
from datetime import date as _date
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from stockpool.config import AppConfig

log = logging.getLogger("stockpool")


class AbPoolConfig(BaseModel):
    """Build parameters for `python -m stockpool ab-pool build`.

    Defaults reproduce the spec's recipe exactly (28 SW-1 industries × 2 mcap
    + 2 liq ≈ 100 stocks). Section is fully optional in config.yaml.
    """
    model_config = ConfigDict(extra="forbid")

    cache_path: str = "data/ab_pool.parquet"
    industry_source: Literal["auto", "baostock", "akshare"] = "auto"
    min_listing_days: int = 252
    min_avg_amount_20d: float = 5.0e7
    per_industry_top_mcap: int = 2
    per_industry_top_liq: int = 2
    exclude_st: bool = True
    include_unknown_industry: bool = True


def _apply_hard_filters(
    df: pd.DataFrame,
    cfg: AbPoolConfig,
    today: _date | None = None,
) -> pd.DataFrame:
    """Apply pre-stratification hard filters.

    Drops in order:
      1. NaN circ_mv (stock missing from akshare snapshot)
      2. ST / *ST / 退 names (if cfg.exclude_st)
      3. IPO date within min_listing_days
      4. avg_amount_20d below min_avg_amount_20d

    Expects columns: code, name, industry, circ_mv, avg_amount_20d, ipo_date.
    ``today`` is injectable for deterministic tests.
    """
    if today is None:
        today = _date.today()
    out = df.copy()
    out = out[out["circ_mv"].notna()]
    if cfg.exclude_st:
        name_str = out["name"].astype(str)
        is_st = (
            name_str.str.upper().str.contains("ST", na=False)
            | name_str.str.contains("退", na=False)
        )
        out = out[~is_st]
    cutoff = pd.Timestamp(today) - pd.Timedelta(days=cfg.min_listing_days)
    ipo_ts = pd.to_datetime(out["ipo_date"], errors="coerce")
    out = out[ipo_ts <= cutoff]
    out = out[out["avg_amount_20d"] >= cfg.min_avg_amount_20d]
    return out.reset_index(drop=True)


def _stratified_select(df: pd.DataFrame, cfg: AbPoolConfig) -> pd.DataFrame:
    """Per-industry top-N by 流通市值 ∪ top-N by 20日均额, row-merged on overlap.

    Overlap semantics: a stock that appears in both top lists yields a SINGLE
    output row with source_tag="mcap+liq" (no row duplication). Buckets
    smaller than 2N contribute what they have.

    Skips "未知" bucket entirely when cfg.include_unknown_industry=False.
    """
    rows: list[dict] = []
    for industry, bucket in df.groupby("industry", sort=False):
        if industry == "未知" and not cfg.include_unknown_industry:
            continue
        top_mcap = set(
            bucket.nlargest(cfg.per_industry_top_mcap, "circ_mv")["code"]
        )
        top_liq = set(
            bucket.nlargest(cfg.per_industry_top_liq, "avg_amount_20d")["code"]
        )
        selected = top_mcap | top_liq
        if not selected:
            log.warning("ab_pool: industry %r yielded 0 selections", industry)
            continue
        for r in bucket[bucket["code"].isin(selected)].itertuples(index=False):
            in_mcap = r.code in top_mcap
            in_liq = r.code in top_liq
            tag = "mcap+liq" if (in_mcap and in_liq) else (
                "mcap" if in_mcap else "liq"
            )
            rows.append({
                "code": r.code,
                "name": r.name,
                "industry": industry,
                "circ_mv": r.circ_mv,
                "avg_amount_20d": r.avg_amount_20d,
                "source_tag": tag,
            })
    if not rows:
        return pd.DataFrame(
            columns=["code", "name", "industry", "circ_mv",
                     "avg_amount_20d", "source_tag"]
        )
    return pd.DataFrame(rows)


def _fetch_circ_mv_snapshot(cache_dir: str | _Path = "data") -> pd.DataFrame:
    """Compute 流通市值 (free-float market cap) entirely from local data.

    Source: baostock profit table (``data/fundamentals_profit.parquet``) for
    ``liqaShare`` (流通 A 股股数, point-in-time at each quarter's pubDate)
    × the latest ``close`` from each stock's per-stock daily parquet.

    Replaces a prior akshare ``stock_zh_a_spot_em`` snapshot. Same output
    schema (``code``, ``circ_mv``) but offline and proxy-independent;
    the trade-off is that ``liqaShare`` is updated quarterly rather than
    intraday, which is more than enough resolution for ab-pool's
    industry-stratified top-2 mcap selection.

    Stocks missing from the profit table OR with no daily cache get
    ``circ_mv = NaN`` (caller filters via ``_apply_hard_filters``).
    """
    cache_dir = _Path(cache_dir)
    prof_path = cache_dir / "fundamentals_profit.parquet"
    if not prof_path.exists():
        raise FileNotFoundError(
            f"{prof_path} not found. Run baostock fundamentals fetch first "
            f"(e.g. `python -m stockpool run --refresh-fundamentals`)."
        )
    prof = pd.read_parquet(prof_path)
    prof["code"] = prof["code"].astype(str).str.zfill(6)
    # Most-recent liqaShare per code (max pubDate per group).
    prof = prof.sort_values(["code", "pubDate"]).drop_duplicates(
        "code", keep="last",
    )
    prof = prof[["code", "liqaShare"]].copy()

    rows: list[dict] = []
    for code in prof["code"]:
        path = cache_dir / f"{code}_daily.parquet"
        if not path.exists():
            rows.append({"code": code, "close": float("nan")})
            continue
        try:
            df = pd.read_parquet(path, columns=["close"])
            close = float(df["close"].iloc[-1]) if len(df) else float("nan")
        except Exception as e:
            log.warning("ab_pool: latest close read failed for %s (%s)", code, e)
            close = float("nan")
        rows.append({"code": code, "close": close})
    closes_df = pd.DataFrame(rows)

    merged = prof.merge(closes_df, on="code", how="left")
    # baostock returns empty string for missing/uninitialised liqaShare
    # in some rows — coerce to NaN instead of raising in astype.
    merged["liqaShare"] = pd.to_numeric(merged["liqaShare"], errors="coerce")
    merged["circ_mv"] = merged["liqaShare"] * merged["close"]
    return merged[["code", "circ_mv"]]


def _compute_avg_amount_20d(
    codes: list[str],
    cache_dir: str | _Path,
) -> pd.DataFrame:
    """For each code, compute mean(volume * close * 100) over last 20 bars.

    Reads from <cache_dir>/<code>_daily.parquet. Missing files yield NaN.
    Returns columns: code, avg_amount_20d.

    Note: mootdx volume unit is 手 (= 100 股), so multiply by 100 to get 元.
    Matches recommend_pool._apply_funnel:172-174.
    """
    cache_dir = _Path(cache_dir)
    rows: list[dict] = []
    for code in codes:
        path = cache_dir / f"{code}_daily.parquet"
        if not path.exists():
            rows.append({"code": code, "avg_amount_20d": float("nan")})
            continue
        try:
            daily = pd.read_parquet(path)
            tail = daily.tail(20)
            avg = float((tail["volume"] * tail["close"] * 100).mean())
        except Exception as e:
            log.warning("ab_pool: avg_amount calc failed for %s (%s)", code, e)
            avg = float("nan")
        rows.append({"code": code, "avg_amount_20d": avg})
    return pd.DataFrame(rows)


def _load_industry_map(cache_dir: _Path, source: str) -> dict[str, str]:
    """Thin wrapper over industry_map.load_or_build_industry_map for mockability."""
    from stockpool.industry_map import load_or_build_industry_map
    return load_or_build_industry_map(cache_dir=cache_dir, source=source)


def _load_ipo_dates(cache_dir: _Path) -> dict[str, pd.Timestamp]:
    """Thin wrapper over ipo_dates.load_or_build_ipo_dates for mockability."""
    from stockpool.ipo_dates import load_or_build_ipo_dates
    return load_or_build_ipo_dates(cache_dir=cache_dir)


def build_ab_pool(cfg: "AppConfig", refresh: bool = False) -> _Path:
    """Build the AB candidate pool and persist to cfg.ab_pool.cache_path.

    Raises:
      FileNotFoundError — universe.parquet missing
      FileExistsError — cache_path exists without refresh=True
      RuntimeError — akshare snapshot empty / all industry buckets empty

    Returns: the cache_path Path on success.
    """
    out_path = _Path(cfg.ab_pool.cache_path)
    if out_path.exists() and not refresh:
        raise FileExistsError(
            f"{out_path} already exists. Pass --refresh to rebuild."
        )

    cache_dir = _Path(cfg.data.cache_dir)
    universe_path = cache_dir / "universe.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(
            f"{universe_path} not found. Run `python -m stockpool fetch-universe` first."
        )
    universe = pd.read_parquet(universe_path)
    universe["code"] = universe["code"].astype(str).str.zfill(6)

    snapshot = _fetch_circ_mv_snapshot(cache_dir)
    industry = _load_industry_map(cache_dir, cfg.ab_pool.industry_source)
    ipo_dates = _load_ipo_dates(cache_dir)
    liq = _compute_avg_amount_20d(list(universe["code"]), cache_dir)

    # Assemble candidate table — left-join universe ← snapshot ← industry ← ipo ← liq
    # Snapshot is now baostock-based and carries only ``code, circ_mv``;
    # names come from universe.parquet (mootdx) — stale ST-rename within
    # one update cycle is acceptable for ab-pool's coarse stratification.
    candidates = universe[["code", "name"]].merge(
        snapshot[["code", "circ_mv"]], on="code", how="left",
    )
    candidates["industry"] = candidates["code"].map(industry).fillna("未知")
    candidates["ipo_date"] = candidates["code"].map(
        lambda c: ipo_dates.get(c, pd.Timestamp("1900-01-01"))
    )
    candidates = candidates.merge(liq, on="code", how="left")

    filtered = _apply_hard_filters(candidates, cfg.ab_pool)
    selected = _stratified_select(filtered, cfg.ab_pool)

    if selected.empty:
        raise RuntimeError(
            "ab_pool: all industry buckets empty after filters — "
            "check liquidity floor / ST filter / IPO cutoff"
        )

    selected = selected.copy()
    selected["build_date"] = _date.today()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out_path, index=False)
    log.info("ab_pool: built %d codes across %d industries → %s",
             len(selected), selected["industry"].nunique(), out_path)
    return out_path


def load_ab_pool(cache_path: str | _Path) -> pd.DataFrame:
    """Read the persisted AB pool parquet. Raises FileNotFoundError if absent."""
    cache_path = _Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Run `python -m stockpool ab-pool build` first."
        )
    return pd.read_parquet(cache_path)
