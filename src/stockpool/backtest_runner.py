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
from stockpool.config import AppConfig, Stock
from stockpool.fetcher import fetch_daily, load_universe_cache
from stockpool.strategy_factory import build_factor_panel, build_strategy, simulate_strategy_equity_curve

log = logging.getLogger("stockpool")


def prepare_pool(
    cfg: AppConfig, stocks: list[Stock], force_refresh: bool,
) -> tuple[dict[str, pd.DataFrame] | None, dict | None]:
    """Build (pool_data, factor_panel) for ml_factor strategies, or (None, None).

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
        return None, None

    ml_cfg = cfg.strategy.ml_factor
    pool_data: dict[str, pd.DataFrame] = {}

    if ml_cfg.training_universe == "all":
        log.info("Loading universe cache (training_universe=all) ...")
        pool_data = load_universe_cache(cfg.data.cache_dir, cfg.data.history_days)
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
            )
        except Exception as e:
            log.warning("Pool preload skipped for %s: %s", s.code, e)

    log.info("Building factor panel over %d stocks × %d factors ...",
             len(pool_data), len(ml_cfg.factors))
    factor_panel = build_factor_panel(ml_cfg.factors, pool_data)
    log.info("Factor panel built: %d factors", len(factor_panel))
    return pool_data, factor_panel


def backtest_stocks(
    cfg: AppConfig,
    stocks: list[Stock],
    pool_data: dict[str, pd.DataFrame] | None,
    factor_panel: dict | None,
    shared_cache: dict,
    refresh: bool,
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
                )
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
                    position_size=cfg.backtest.position_size,
                    max_concurrent_lots=cfg.backtest.max_concurrent_lots,
                )
            else:
                strategy = build_strategy(
                    cfg,
                    pool_data=pool_data if needs_pool else None,
                    current_stock_code=s.code,
                    factor_panel=factor_panel,
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
                    position_size=cfg.backtest.position_size,
                    max_concurrent_lots=cfg.backtest.max_concurrent_lots,
                )
            per_stock.append((s.code, s.name, result))
        except Exception as e:
            log.error("Backtest failed for %s: %s\n%s", s.code, e, traceback.format_exc())
            failed.append((s.code, str(e)))

    return per_stock, failed
