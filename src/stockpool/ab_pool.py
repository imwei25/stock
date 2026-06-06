"""AB candidate pool — stratified ~100-stock pool for AB tests.

Build: industry-stratified top-2-mcap + top-2-liquidity selection from
universe.parquet, with akshare 流通市值 snapshot. Persisted to
data/ab_pool.parquet; static unless rebuilt by hand.

See docs/superpowers/specs/2026-06-06-ab-candidate-pool-design.md.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict


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
