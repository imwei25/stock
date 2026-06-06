"""Backtest orchestration helpers shared by cli.cmd_backtest and ab.runner.

Provides:
  * prepare_pool(cfg, stocks, force_refresh) — pool data + factor panel for
    ml_factor in pooled mode (or (None, None) for other configurations).
  * backtest_stocks(cfg, stocks, pool_data, factor_panel, shared_cache,
    refresh) — per-stock backtest loop with failure isolation.

Both functions are extracted from cli.py to break the reverse-dependency
that ab/runner.py would otherwise have on cli.
"""
from __future__ import annotations

import logging
import traceback

import pandas as pd

from stockpool.backtest_composite import EquityResult, simulate_equity_curve, walk_forward_verdicts
from stockpool.backtesting.sizing import build_lot_sizer
from stockpool.config import AppConfig, Stock
from stockpool.fetcher import fetch_daily, load_universe_cache
from stockpool.strategy_factory import (
    build_strategy,
    load_or_build_factor_panel,
    simulate_strategy_equity_curve,
)

log = logging.getLogger("stockpool")


def prepare_pool(
    cfg: AppConfig, stocks: list[Stock], force_refresh: bool,
    refresh_factor_panel: bool = False,
) -> tuple[dict[str, pd.DataFrame] | None, dict | None, pd.DataFrame | None]:
    """Build (pool_data, factor_panel, close_panel) for ml_factor or (None,None,None).

    Pool composition depends on ``cfg.strategy.ml_factor.training_universe``:
      * ``pool``: only ``cfg.stocks`` (legacy, ~10 stocks).
      * ``all``: full A-share cache from ``data/`` (~4000 stocks, requires a
        prior ``fetch-universe`` run). Application stocks are merged in so any
        cfg.stocks entry missing from the universe cache (e.g. 北交) is still
        usable. Cross-sec factors only become meaningful at panel widths in
        the hundreds, so ``all`` is recommended whenever WQ101 alphas are used.

    The factor panel is computed once on the combined pool and reused across
    every per-stock predict — the panel-wide computation can be expensive on
    the all-universe path.
    """
    if (
        cfg.strategy.name != "ml_factor"
        or cfg.strategy.ml_factor.panel_mode != "pooled"
    ):
        return None, None, None

    ml_cfg = cfg.strategy.ml_factor
    pool_data: dict[str, pd.DataFrame] = {}

    if ml_cfg.training_universe == "all":
        log.info("Loading universe cache (training_universe=all) ...")
        pool_data = load_universe_cache(cfg.data.cache_dir, cfg.data.history_days, warmup_days=cfg.data.warmup_days)
        if not pool_data:
            log.warning(
                "training_universe=all but data/ has no cached stocks. "
                "Run `python -m stockpool fetch-universe` first; falling back to pool."
            )
        else:
            log.info("Universe cache loaded: %d stocks", len(pool_data))

    # Ensure every application stock is in the pool (fetch fresh if missing
    # or stale; also picks up today's bar for already-cached stocks).
    for s in stocks:
        try:
            pool_data[s.code] = fetch_daily(
                s.code, cfg.data.history_days, cfg.data.cache_dir,
                force_refresh=force_refresh, source=cfg.data.source,
                warmup_days=cfg.data.warmup_days,
            )
        except Exception as e:
            log.warning("Pool preload skipped for %s: %s", s.code, e)

    # Inject sector_map so industry_neutral factors (WQ101 + custom
    # industry_relative_strength_N) get group context. Empty map → factors
    # fall back to cross-sec demean / NaN.
    from stockpool.factors.context import set_sector_map
    from stockpool.industry_map import load_or_build_industry_map

    sector_map = load_or_build_industry_map(cfg.data.cache_dir, source="auto")
    set_sector_map(sector_map)

    factor_panel, close_panel = load_or_build_factor_panel(
        ml_cfg.factors, pool_data, cfg.data.cache_dir,
        refresh=refresh_factor_panel,
        preprocess_cfg=ml_cfg.preprocess,
    )
    return pool_data, factor_panel, close_panel


def backtest_stocks(
    cfg: AppConfig,
    stocks: list[Stock],
    pool_data: dict[str, pd.DataFrame] | None,
    factor_panel: dict | None,
    shared_cache: dict,
    refresh: bool,
    close_panel: pd.DataFrame | None = None,
) -> tuple[list[tuple[str, str, EquityResult]], list[tuple[str, str]]]:
    """Backtest each stock; return (successes, failures).

    Failure isolation: any exception during a single stock's pipeline
    (data fetch, walk-forward, ML training, engine simulation) is caught,
    the offending code is appended to ``failures`` as ``(code, message)``,
    and the loop continues. Callers decide how to surface failures.

    Args:
        cfg: effective AppConfig (already deep-merged for ab arms).
        stocks: list of Stock to backtest.
        pool_data, factor_panel: pre-built ml_factor inputs (or None).
        shared_cache: mutable dict passed to MLFactorStrategy for cross-stock
            pipeline reuse within one call.
        refresh: forces fetch_daily to bypass cache.
    """
    per_stock: list[tuple[str, str, EquityResult]] = []
    failed: list[tuple[str, str]] = []
    needs_pool = pool_data is not None

    for s in stocks:
        log.info("Backtesting %s (%s)...", s.code, s.name)
        try:
            daily = pool_data.get(s.code) if needs_pool else None
            if daily is None:
                daily = fetch_daily(
                    s.code, cfg.data.history_days, cfg.data.cache_dir,
                    force_refresh=refresh, source=cfg.data.source,
                    warmup_days=cfg.data.warmup_days,
                )
            # P2: trim warmup region — factor panel keeps full data for training,
            # but per-stock backtest iteration only spans the history_days window.
            if (
                daily is not None
                and cfg.data.warmup_days > 0
                and len(daily) > cfg.data.warmup_days
            ):
                daily = daily.iloc[cfg.data.warmup_days:].reset_index(drop=True)
            if cfg.strategy.name == "composite_verdict":
                wf = walk_forward_verdicts(
                    daily, cfg.weights, cfg.scoring, cfg.verdicts, cfg.indicators,
                )
                if len(wf) == 0:
                    failed.append((s.code, "insufficient history"))
                    continue
                result = simulate_equity_curve(
                    wf,
                    holding_days_list=cfg.backtest.equity_curve_holding_days,
                    with_buy_and_hold=True,
                    buy_cost=cfg.backtest.costs.buy_cost,
                    sell_cost=cfg.backtest.costs.sell_cost,
                    risk_free_rate=cfg.backtest.risk_free_rate,
                    engine=cfg.backtest.engine,
                    lot_sizer=build_lot_sizer(cfg.backtest.sizing),
                    max_concurrent_lots=cfg.backtest.max_concurrent_lots,
                )
            else:
                strategy = build_strategy(
                    cfg,
                    pool_data=pool_data if needs_pool else None,
                    current_stock_code=s.code,
                    factor_panel=factor_panel,
                    close_panel=close_panel,
                    shared_cache=shared_cache,
                )
                result = simulate_strategy_equity_curve(
                    daily, strategy,
                    holding_days_list=cfg.backtest.equity_curve_holding_days,
                    with_buy_and_hold=True,
                    buy_cost=cfg.backtest.costs.buy_cost,
                    sell_cost=cfg.backtest.costs.sell_cost,
                    risk_free_rate=cfg.backtest.risk_free_rate,
                    engine=cfg.backtest.engine,
                    lot_sizer=build_lot_sizer(cfg.backtest.sizing),
                    max_concurrent_lots=cfg.backtest.max_concurrent_lots,
                )
            per_stock.append((s.code, s.name, result))
        except Exception as e:
            log.error("Backtest failed for %s: %s\n%s", s.code, e, traceback.format_exc())
            failed.append((s.code, str(e)))

    return per_stock, failed
