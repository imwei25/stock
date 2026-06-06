"""AB candidate pool — stratified ~100-stock pool for AB tests.

Build: industry-stratified top-2-mcap + top-2-liquidity selection from
universe.parquet, with akshare 流通市值 snapshot. Persisted to
data/ab_pool.parquet; static unless rebuilt by hand.

See docs/superpowers/specs/2026-06-06-ab-candidate-pool-design.md.
"""
from __future__ import annotations

import logging
from datetime import date as _date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

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
