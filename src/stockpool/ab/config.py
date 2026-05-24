"""Pydantic schema + loader + deep-merge for A/B test configuration."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from stockpool.config import (
    AppConfig,
    BacktestCostConfig,
    StrategyConfig,
    load_config,
)


class ArmBacktestOverride(BaseModel):
    """Per-arm overrides to the base.backtest section.

    equity_curve_holding_days is required and must be a length-1 list.
    All other fields default to None, meaning "inherit base.backtest.<same>".
    """
    model_config = ConfigDict(extra="forbid")
    equity_curve_holding_days: list[int]
    forward_days: list[int] | None = None
    risk_free_rate: float | None = None
    costs: BacktestCostConfig | None = None
    engine: Literal["single", "multi_lot"] | None = None
    position_size: float | None = None
    max_concurrent_lots: int | None = None

    @field_validator("equity_curve_holding_days")
    @classmethod
    def _single_n(cls, v: list[int]) -> list[int]:
        if len(v) != 1 or v[0] <= 0:
            raise ValueError(
                f"equity_curve_holding_days must be [N] with N > 0, got {v!r}"
            )
        return v


class ArmOverride(BaseModel):
    """One A/B arm: full strategy replacement + partial backtest override."""
    model_config = ConfigDict(extra="forbid")
    strategy: StrategyConfig
    backtest: ArmBacktestOverride


class ABConfig(BaseModel):
    """Top-level A/B test config (loaded from ab.yaml)."""
    model_config = ConfigDict(extra="forbid")
    base_config: str
    stocks_filter: list[str] = Field(default_factory=list)
    arms: dict[str, ArmOverride]

    @field_validator("arms")
    @classmethod
    def _exactly_two(cls, v: dict[str, ArmOverride]) -> dict[str, ArmOverride]:
        if len(v) != 2:
            raise ValueError(
                f"arms must contain exactly 2 entries, got {len(v)}: {list(v)}"
            )
        return v


def build_effective_cfg(base: AppConfig, arm: ArmOverride) -> AppConfig:
    """Deep-merge an arm's overrides into the base config.

    Rules:
      * arm.strategy replaces base.strategy wholesale.
      * arm.backtest fields with non-None values replace; None fields inherit
        from base.backtest.
      * All other top-level fields pass through unchanged.

    Returns a fresh AppConfig with content_hash recomputed; does not mutate base.

    Note: ``content_hash`` is recomputed from the dumped merged dict (canonical
    sorted-key yaml), which is intentionally a different canonicalisation from
    ``load_config``'s raw-bytes hash. The hashes are only comparable across
    effective_cfgs produced by this function — they will not match the hash of
    a plain ``load_config(<base_yaml>)``. This is fine for the only consumer
    (ML monthly fit cache keyed by sig) because both arms route through here.
    """
    merged = base.model_dump(mode="python")
    merged["strategy"] = arm.strategy.model_dump(mode="python")
    arm_bt = arm.backtest.model_dump(mode="python")
    base_bt = merged["backtest"]
    for k, v in arm_bt.items():
        if v is not None:
            base_bt[k] = v
    merged["backtest"] = base_bt
    out = AppConfig.model_validate(merged)
    canonical = yaml.safe_dump(merged, sort_keys=True).encode("utf-8")
    out.content_hash = hashlib.sha256(canonical).hexdigest()[:8]
    return out


def load_ab_config(ab_path: str | Path) -> ABConfig:
    """Load and validate ab.yaml. Performs post-pydantic checks that need
    side info (base config existence, stocks_filter membership, deep-merge
    validity).

    Raises pydantic.ValidationError or ValueError on any failure.
    """
    ab_path = Path(ab_path)
    raw = yaml.safe_load(ab_path.read_text(encoding="utf-8"))
    ab_cfg = ABConfig.model_validate(raw)

    base_path = (ab_path.parent / ab_cfg.base_config).resolve()
    if not base_path.exists():
        raise ValueError(
            f"base_config {ab_cfg.base_config!r} (resolved to {base_path}) "
            f"does not exist"
        )

    base_cfg = load_config(base_path)

    if ab_cfg.stocks_filter:
        base_codes = {s.code for s in base_cfg.stocks}
        unknown = [c for c in ab_cfg.stocks_filter if c not in base_codes]
        if unknown:
            raise ValueError(
                f"stocks_filter references codes not in base.stocks: {unknown}"
            )

    for name, arm in ab_cfg.arms.items():
        try:
            build_effective_cfg(base_cfg, arm)
        except ValidationError as e:
            raise ValueError(
                f"arm {name!r} fails effective-config validation: {e}"
            ) from e

    return ab_cfg
