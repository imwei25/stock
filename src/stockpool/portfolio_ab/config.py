"""Portfolio AB config schema + loader + deep-merge.

Mirrors ``stockpool.ab.config`` in style but allows a different override
surface: only ``strategy`` (whole-replace) and ``portfolio_backtest``
(field-level merge). All other top-level fields inherit from base.

Why a separate subpackage rather than extending ``ab/``:
  * Different override fields → schemas would have to fork anyway
  * Report layout is completely different (no per-stock scatter; instead
    equity overlay + contribution decomposition)
  * Evolution stays decoupled (per-stock AB and portfolio AB will likely
    diverge further as both add features)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from stockpool.config import AppConfig, load_config


class PortfolioArmOverride(BaseModel):
    """One A/B arm: strategy (whole-replace) + portfolio_backtest (field merge).

    Two dict fields (rather than typed sub-models) because:
      * ``strategy``: AppConfig's StrategyConfig has many nested sub-models
        (ml_factor, selectors, weighters, thresholds, ...). Re-typing all of
        them here would duplicate hundreds of lines; passing as ``dict`` and
        merging into the base AppConfig then re-validating is simpler.
      * ``portfolio_backtest``: same reason, plus we need field-level merge
        for partial overrides (e.g. only override ``portfolio.top_k``).
    Validation happens on the merged result in ``build_effective_cfg``.
    """
    model_config = ConfigDict(extra="forbid")
    strategy: dict | None = None
    portfolio_backtest: dict | None = None


class PortfolioABConfig(BaseModel):
    """Top-level portfolio AB config (loaded from portfolio_ab.yaml)."""
    model_config = ConfigDict(extra="forbid")
    base_config: str
    arms: dict[str, PortfolioArmOverride] = Field(..., min_length=2, max_length=2)

    @model_validator(mode="after")
    def _check_arms_count(self) -> "PortfolioABConfig":
        if len(self.arms) != 2:
            raise ValueError(
                f"portfolio AB requires exactly 2 arms, got {len(self.arms)}: "
                f"{list(self.arms)}"
            )
        return self


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursive field-level merge: override values replace base, but nested
    dicts are merged recursively. Lists are replaced (not appended)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def build_effective_cfg(
    base: AppConfig, arm: PortfolioArmOverride,
) -> AppConfig:
    """Deep-merge an arm's overrides into the base config.

    Rules:
      * ``arm.strategy`` (if set) replaces ``base.strategy`` *wholesale*.
      * ``arm.portfolio_backtest`` (if set) field-merges into
        ``base.portfolio_backtest`` (recursive for nested dicts).
      * All other top-level fields pass through unchanged.

    Returns a fresh ``AppConfig`` with ``content_hash`` recomputed from a
    canonical sorted-key yaml dump of the merged dict. ⚠ Same caveat as
    ``stockpool.ab.build_effective_cfg``: this hash is intentionally a
    different canonicalisation from ``load_config``'s raw-bytes hash, so it
    only compares meaningfully across other effective_cfgs produced by this
    function (used for per-arm score-panel cache isolation).
    """
    merged = base.model_dump(mode="python")
    if arm.strategy is not None:
        merged["strategy"] = dict(arm.strategy)
    if arm.portfolio_backtest is not None:
        merged["portfolio_backtest"] = _deep_merge_dict(
            merged.get("portfolio_backtest", {}) or {},
            arm.portfolio_backtest,
        )
    out = AppConfig.model_validate(merged)
    canonical = yaml.safe_dump(merged, sort_keys=True).encode("utf-8")
    out.content_hash = hashlib.sha256(canonical).hexdigest()[:8]
    return out


def load_portfolio_ab_config(path: str | Path) -> PortfolioABConfig:
    """Load and validate portfolio_ab.yaml.

    Performs:
      1. Pydantic schema validation (arms count, extra=forbid)
      2. base_config file existence check
      3. Per-arm effective-config build to surface deep-merge errors early
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ab_cfg = PortfolioABConfig.model_validate(raw)

    base_path = (path.parent / ab_cfg.base_config).resolve()
    if not base_path.exists():
        raise ValueError(
            f"base_config {ab_cfg.base_config!r} (resolved to {base_path}) "
            f"does not exist"
        )

    base_cfg = load_config(base_path)
    for name, arm in ab_cfg.arms.items():
        try:
            build_effective_cfg(base_cfg, arm)
        except ValidationError as e:
            raise ValueError(
                f"arm {name!r} fails effective-config validation: {e}"
            ) from e
    return ab_cfg
