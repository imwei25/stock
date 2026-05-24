"""Backtest orchestration helpers shared by cli.cmd_backtest and ab.runner.

Provides:
  * prepare_pool(cfg, stocks, refresh) — pool data + factor panel for ml_factor
    in pooled mode (or (None, None) for other configurations).
  * backtest_stocks(cfg, stocks, pool_data, factor_panel, shared_cache,
    refresh) — per-stock backtest loop with failure isolation.

Both functions are extracted from cli.py to break the reverse-dependency
that ab/runner.py would otherwise have on cli.
"""
from __future__ import annotations

import logging

import pandas as pd

from stockpool.config import AppConfig, Stock
from stockpool.fetcher import fetch_daily, load_universe_cache
from stockpool.strategy_factory import build_factor_panel

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
