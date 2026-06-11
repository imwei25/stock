"""A/B test runner: pool sharing decision + arm execution + ABResult.

Two public entry points (added in Task 6):
  * run_ab(...) → ABResult (always 2 arms)
  * run_single_arm(...) → ArmResult (debug helper for --arm flag)

Task 5 only adds the pool-sharing decision logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from stockpool.ab.config import ABConfig, ArmOverride, build_effective_cfg
from stockpool.backtest_composite import EquityResult
from stockpool.backtest_runner import backtest_stocks, prepare_pool
from stockpool.config import AppConfig, Stock
from stockpool.fetcher import fetch_daily, load_universe_cache
from stockpool.strategy_factory import build_close_panel, load_or_build_factor_panel

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
    ab_cfg: ABConfig
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


def _decide_pool_sharing(arm_cfgs: list[AppConfig]) -> dict:
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
        # Mask is not a sharing barrier — factor panels are mask-agnostic
        # (mask only affects labels downstream of factor computation).
        # Preprocess IS a barrier — different preprocess produces different
        # panel values; sharing would silently apply arm_a's preprocess to arm_b.
        p_a = arm_cfgs[0].strategy.ml_factor.preprocess.model_dump()
        p_b = arm_cfgs[1].strategy.ml_factor.preprocess.model_dump()
        if f_a == f_b and p_a == p_b:
            shared_factors = f_a

    return {"load_universe": load_universe, "shared_factors": shared_factors}


def _no_share_plan() -> dict:
    return {"load_universe": False, "shared_factors": None}


def _prepare_pool_for_arm(
    arm_cfg: AppConfig,
    stocks: list[Stock],
    refresh: bool,
    injected_universe: dict[str, pd.DataFrame] | None,
    injected_factor_panel: dict | None,
) -> tuple[dict[str, pd.DataFrame] | None, dict | None, pd.DataFrame | None]:
    """Per-arm pool prep with optional shared inputs from run_ab.

    If ``injected_universe`` is provided, skip ``load_universe_cache`` and
    use it directly (merging per-stock fetches on top, same as ``prepare_pool``).
    If ``injected_factor_panel`` is provided, skip ``build_factor_panel``.

    For non-ml_factor or non-pooled arms, returns ``(None, None)`` — same as
    ``prepare_pool``.
    """
    if (
        arm_cfg.strategy.name != "ml_factor"
        or arm_cfg.strategy.ml_factor.panel_mode != "pooled"
    ):
        return None, None, None

    # If neither shared input is provided, delegate entirely to prepare_pool.
    if injected_universe is None and injected_factor_panel is None:
        return prepare_pool(arm_cfg, stocks, refresh)

    ml_cfg = arm_cfg.strategy.ml_factor
    pool_data: dict[str, pd.DataFrame] = (
        dict(injected_universe) if injected_universe is not None else {}
    )
    if injected_universe is None and ml_cfg.training_universe == "all":
        pool_data = load_universe_cache(
            arm_cfg.data.cache_dir, arm_cfg.data.history_days,
        )

    for s in stocks:
        try:
            pool_data[s.code] = fetch_daily(
                s.code, arm_cfg.data.history_days, arm_cfg.data.cache_dir,
                force_refresh=refresh, source=arm_cfg.data.source,
            )
        except Exception as e:
            log.warning("Pool preload skipped for %s: %s", s.code, e)

    if injected_factor_panel is not None:
        factor_panel = injected_factor_panel
        close_panel = build_close_panel(pool_data)
    else:
        # Inject sector_map for industry-aware factors (WQ101 + custom
        # industry_relative_strength_N) — needed when prepare_pool was
        # skipped (shared-universe path).
        from stockpool.factors.context import set_sector_map
        from stockpool.industry_map import load_or_build_industry_map
        from stockpool.strategy_factory import maybe_inject_mcap_panel
        sector_map = load_or_build_industry_map(arm_cfg.data.cache_dir, source="auto")
        set_sector_map(sector_map)
        # Inject log-mcap panel for market_cap_neutralize (no-op unless enabled).
        maybe_inject_mcap_panel(ml_cfg.preprocess, pool_data, arm_cfg.data.cache_dir)

        factor_panel, close_panel = load_or_build_factor_panel(
            ml_cfg.factors, pool_data, arm_cfg.data.cache_dir,
            preprocess_cfg=ml_cfg.preprocess,
        )
    return pool_data, factor_panel, close_panel


def _run_arm(
    arm_cfg: AppConfig,
    arm_name: str,
    stocks: list[Stock],
    pool_data: dict | None,
    factor_panel: dict | None,
    refresh: bool,
    close_panel: pd.DataFrame | None = None,
) -> ArmResult:
    """Backtest every stock for one arm."""
    log.info("Running arm %s ...", arm_name)
    per_stock, failed = backtest_stocks(
        arm_cfg, stocks, pool_data, factor_panel,
        shared_cache={}, refresh=refresh, close_panel=close_panel,
    )
    log.info("Arm %s: %d done, %d failed", arm_name, len(per_stock), len(failed))
    return ArmResult(
        name=arm_name, effective_cfg=arm_cfg,
        per_stock=per_stock, failed=failed,
    )


def run_ab(
    ab_cfg: ABConfig,
    base_cfg: AppConfig,
    stocks: list[Stock],
    refresh: bool,
    *,
    share_pool: bool = True,
) -> ABResult:
    """Run both arms; return an ABResult with exactly two ArmResults."""
    arm_items = list(ab_cfg.arms.items())
    arm_cfgs = [build_effective_cfg(base_cfg, arm) for _, arm in arm_items]

    plan = _decide_pool_sharing(arm_cfgs) if share_pool else _no_share_plan()

    # P3-18: --refresh 时先统一拉一次数据,两 arm 都用 refresh=False 读同一
    # 份缓存 —— 否则两 arm 各自拉取,数据切片可能不一致(盘中 bar 防护已
    # 消掉最大头,这里保证严格同源)。
    if refresh:
        from stockpool.fetcher import fetch_daily
        for s in stocks:
            try:
                fetch_daily(
                    s.code, base_cfg.data.history_days, base_cfg.data.cache_dir,
                    force_refresh=True, source=base_cfg.data.source,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("AB pre-fetch failed for %s: %s", s.code, e)
        refresh = False

    shared_universe = None
    if plan["load_universe"]:
        try:
            shared_universe = load_universe_cache(
                base_cfg.data.cache_dir, base_cfg.data.history_days,
            )
            log.info("Shared universe loaded: %d stocks",
                     len(shared_universe) if shared_universe else 0)
        except Exception as e:
            log.warning("Universe load failed (each arm will reload): %s", e)

    shared_panel = None
    arm_results: list[ArmResult] = []
    for (name, _arm), arm_cfg in zip(arm_items, arm_cfgs):
        # Only inject the shared universe into arms that actually want it.
        # Without this gate, an arm with training_universe=pool gets the
        # other arm's all-A-share data dumped into its pool_data, silently
        # converting it into a training_universe=all run. See P3-2 bug
        # in docs/ab_validation_results.md §3.7.
        arm_wants_universe = _ml_uses_universe(arm_cfg)
        pool_data, factor_panel, close_panel = _prepare_pool_for_arm(
            arm_cfg, stocks, refresh,
            injected_universe=(shared_universe if arm_wants_universe else None),
            injected_factor_panel=(shared_panel if plan["shared_factors"] else None),
        )
        if plan["shared_factors"] and shared_panel is None and factor_panel is not None:
            shared_panel = factor_panel
        arm_results.append(_run_arm(
            arm_cfg, name, stocks, pool_data, factor_panel, refresh,
            close_panel=close_panel,
        ))

    return ABResult(
        ab_cfg=ab_cfg, base_cfg=base_cfg,
        arm_a=arm_results[0], arm_b=arm_results[1],
        run_date=date.today().isoformat(),
    )


def run_single_arm(
    ab_cfg: ABConfig,
    base_cfg: AppConfig,
    stocks: list[Stock],
    refresh: bool,
    arm_name: str,
) -> ArmResult:
    """Debug helper: run only one arm by name; no pool sharing."""
    if arm_name not in ab_cfg.arms:
        raise KeyError(f"arm {arm_name!r} not in {list(ab_cfg.arms)}")
    arm = ab_cfg.arms[arm_name]
    arm_cfg = build_effective_cfg(base_cfg, arm)
    pool_data, factor_panel, close_panel = _prepare_pool_for_arm(
        arm_cfg, stocks, refresh,
        injected_universe=None, injected_factor_panel=None,
    )
    return _run_arm(
        arm_cfg, arm_name, stocks, pool_data, factor_panel, refresh,
        close_panel=close_panel,
    )
