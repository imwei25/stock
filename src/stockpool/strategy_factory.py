"""Strategy factory + ML-strategy backtest helper.

Decouples the CLI from concrete strategy classes:

  * ``build_strategy(cfg, pool_data=None, current_stock_code=None)`` —
    return the strategy implementation selected by ``cfg.strategy.name``.
  * ``simulate_ml_equity_curve(...)`` — mirror of ``backtest_composite.
    simulate_equity_curve``'s output shape but driven by ``MLFactorStrategy``;
    lets ``cmd_backtest`` route ML runs without touching the report renderer.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.backtest_composite import EquityResult
from stockpool.backtesting import (
    BacktestEngine,
    CompositeVerdictStrategy,
    MLFactorStrategy,
    MultiLotBacktestEngine,
    Strategy,
    TradeCosts,
    buy_and_hold_baseline,
)
from stockpool.config import AppConfig


def build_strategy(
    cfg: AppConfig,
    pool_data: Mapping[str, pd.DataFrame] | None = None,
    current_stock_code: str | None = None,
) -> Strategy:
    """Construct the strategy referenced by ``cfg.strategy.name``.

    Args:
        cfg: loaded ``AppConfig``.
        pool_data: required for ``ml_factor`` strategies in pooled mode. Pass
            the full daily-history dict; the strategy applies per-bar
            look-ahead-safe truncation internally.
        current_stock_code: which stock in ``pool_data`` is the one being
            backtested — excluded from pool truncation to avoid double-counting.
    """
    name = cfg.strategy.name
    if name == "composite_verdict":
        return CompositeVerdictStrategy(
            weights=cfg.weights,
            scoring=cfg.scoring,
            verdicts_cfg=cfg.verdicts,
            indicators_cfg=cfg.indicators,
        )
    if name == "ml_factor":
        return MLFactorStrategy(
            cfg=cfg.strategy.ml_factor,
            pool_data=pool_data,
            current_stock_code=current_stock_code,
        )
    raise ValueError(f"unknown strategy: {name!r}")


def simulate_strategy_equity_curve(
    daily_df: pd.DataFrame,
    strategy: Strategy,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    risk_free_rate: float = 0.02,
    engine: str = "single",
    position_size: float = 0.1,
    max_concurrent_lots: int | None = None,
) -> EquityResult:
    """Generic equity-curve simulator: runs ``strategy`` for each holding-day cap.

    Output shape matches ``backtest_composite.simulate_equity_curve`` so the
    HTML renderer accepts both.
    """
    costs = TradeCosts(buy_cost=buy_cost, sell_cost=sell_cost)
    if engine == "single":
        bt = BacktestEngine(strategy, costs=costs, risk_free_rate=risk_free_rate)
    elif engine == "multi_lot":
        bt = MultiLotBacktestEngine(
            strategy, position_size=position_size, costs=costs,
            risk_free_rate=risk_free_rate, max_concurrent_lots=max_concurrent_lots,
        )
    else:
        raise ValueError(f"engine must be 'single' or 'multi_lot', got {engine!r}")

    # Generate signals once; reuse across holding-day sweeps.
    signals = strategy.generate_signals(daily_df)
    curves: dict[int, pd.DataFrame] = {}
    metrics: dict[int, dict] = {}
    for N in holding_days_list:
        result = bt.run_on_signals(signals, max_holding_days=N)
        curves[N] = result.curve
        metrics[N] = result.metrics

    bh_curve = None
    bh_metrics = None
    if with_buy_and_hold and len(daily_df) > 0:
        bh = buy_and_hold_baseline(daily_df, risk_free_rate=risk_free_rate)
        bh_curve = bh.curve[["date", "equity"]].reset_index(drop=True)
        bh_metrics = bh.metrics

    return EquityResult(
        curves=curves,
        metrics=metrics,
        buy_and_hold=bh_curve,
        buy_and_hold_metrics=bh_metrics,
    )
