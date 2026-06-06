"""AB candidate pool — stratified ~100-stock pool for AB tests.

Build: industry-stratified top-2-mcap + top-2-liquidity selection from
universe.parquet, with akshare 流通市值 snapshot. Persisted to
data/ab_pool.parquet; static unless rebuilt by hand.

See docs/superpowers/specs/2026-06-06-ab-candidate-pool-design.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_serializer


class AbPoolConfig(BaseModel):
    """Build parameters for `python -m stockpool ab-pool build`.

    Defaults reproduce the spec's recipe exactly (28 SW-1 industries × 2 mcap
    + 2 liq ≈ 100 stocks). Section is fully optional in config.yaml.
    """
    model_config = ConfigDict(extra="forbid")

    cache_path: Path = Path("data/ab_pool.parquet")
    industry_source: Literal["auto", "baostock", "akshare"] = "auto"
    min_listing_days: int = 252
    min_avg_amount_20d: float = 5.0e7
    per_industry_top_mcap: int = 2
    per_industry_top_liq: int = 2
    exclude_st: bool = True
    include_unknown_industry: bool = True

    # Path serializes to str so downstream ``model_dump(mode="python")`` →
    # ``yaml.safe_dump`` round-trip works (used by ab/config.py to compute
    # content_hash). WindowsPath is not directly YAML-serializable.
    @field_serializer("cache_path")
    def _serialize_cache_path(self, value: Path) -> str:
        return str(value)
