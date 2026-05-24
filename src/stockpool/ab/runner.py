"""A/B test runner: pool sharing decision + arm execution + ABResult.

Two public entry points (added in Task 6):
  * run_ab(...) → ABResult (always 2 arms)
  * run_single_arm(...) → ArmResult (debug helper for --arm flag)

Task 5 only adds the pool-sharing decision logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from stockpool.backtest_composite import EquityResult
from stockpool.config import AppConfig, Stock

log = logging.getLogger("stockpool")


@dataclass
class ArmResult:
    """Outcome of running one arm.

    name              — arm key from ab.yaml
    effective_cfg     — base ⊕ arm.override
    per_stock         — successful backtests: [(code, name, EquityResult), ...]
    failed            — failures: [(code, error_message), ...]
    """
    name: str
    effective_cfg: AppConfig
    per_stock: list[tuple[str, str, EquityResult]]
    failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ABResult:
    """Outcome of a full A/B run."""
    ab_cfg: object              # ABConfig — kept untyped here to avoid cycle
    base_cfg: AppConfig
    arm_a: ArmResult
    arm_b: ArmResult
    run_date: str


def _ml_uses_universe(cfg: AppConfig) -> bool:
    """True iff this cfg's strategy needs the all-A-share universe cache."""
    if cfg.strategy.name != "ml_factor":
        return False
    ml = cfg.strategy.ml_factor
    return ml.panel_mode == "pooled" and ml.training_universe == "all"


def _decide_pool_sharing(
    arm_cfgs: list[AppConfig], stocks: list[Stock],
) -> dict:
    """Decide whether the universe cache and/or factor panel can be shared
    across the two arms.

    Returns {"load_universe": bool, "shared_factors": list[str] | None}.
      * load_universe=True iff at least one arm needs the all-universe cache.
      * shared_factors is a non-None factor list iff both arms are ml_factor +
        pooled + training_universe=all AND their factor lists are equal
        (order-sensitive).
    """
    load_universe = any(_ml_uses_universe(c) for c in arm_cfgs)

    shared_factors: list[str] | None = None
    if (
        len(arm_cfgs) == 2
        and all(_ml_uses_universe(c) for c in arm_cfgs)
    ):
        f_a = list(arm_cfgs[0].strategy.ml_factor.factors)
        f_b = list(arm_cfgs[1].strategy.ml_factor.factors)
        if f_a == f_b:
            shared_factors = f_a

    return {"load_universe": load_universe, "shared_factors": shared_factors}


def _no_share_plan() -> dict:
    return {"load_universe": False, "shared_factors": None}
