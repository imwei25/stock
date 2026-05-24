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
from stockpool.backtesting.sizing import FixedLotSizer, LotSizer
from stockpool.config import AppConfig


def build_strategy(
    cfg: AppConfig,
    pool_data: Mapping[str, pd.DataFrame] | None = None,
    current_stock_code: str | None = None,
    factor_panel: Mapping[str, pd.DataFrame] | None = None,
    shared_cache: dict | None = None,
) -> Strategy:
    """Construct the strategy referenced by ``cfg.strategy.name``.

    Args:
        cfg: loaded ``AppConfig``.
        pool_data: required for ``ml_factor`` strategies in pooled mode. Pass
            the full daily-history dict; the strategy applies per-bar
            look-ahead-safe truncation internally.
        current_stock_code: which stock in ``pool_data`` is the one being
            backtested — excluded from pool truncation to avoid double-counting.
        factor_panel: precomputed ``{factor_name: T×N DataFrame}``. Pass this
            when iterating many stocks against the same pool, so the (potentially
            expensive) panel-wide factor computation runs once. If omitted and
            ``pool_data`` is provided in pooled mode, the panel is built here.
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
        # 在 pooled 模式 + 有 pool_data 时,预算因子面板,让 WQ101 cross-sec
        # 因子在 predict 阶段拿到真实横截面值(否则会退化为 1-stock 常数)。
        if (
            factor_panel is None
            and cfg.strategy.ml_factor.panel_mode == "pooled"
            and pool_data
        ):
            factor_panel = build_factor_panel(cfg.strategy.ml_factor.factors, pool_data)
        return MLFactorStrategy(
            cfg=cfg.strategy.ml_factor,
            pool_data=pool_data,
            current_stock_code=current_stock_code,
            factor_panel=factor_panel,
            cache_dir=cfg.data.cache_dir,
            shared_cache=shared_cache,
        )
    raise ValueError(f"unknown strategy: {name!r}")


def build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """从 ``{code: daily_df}`` 装一个 OHLCV Panel,在 Panel 上算所有因子,
    返回 ``{factor_name: T×N DataFrame}``。

    Look-ahead 安全:因子在第 i 行只用 ``[:i+1]`` 数据(由 Factor 契约保证),
    所以一次性预算整段历史不会泄露未来。
    """
    from stockpool.ml.dataset import compute_factor_panel

    # 1) 把每股 daily_df → date-indexed,按列拼成宽表
    per_stock: dict[str, pd.DataFrame] = {}
    for code, df in pool_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        per_stock[code] = d.set_index("date").sort_index()
    if not per_stock:
        return {}
    all_dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")
    panel: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        panel[field] = pd.DataFrame(
            {code: d[field].reindex(idx) for code, d in per_stock.items()},
            index=idx,
        )
    return compute_factor_panel(panel, factor_names)


def simulate_strategy_equity_curve(
    daily_df: pd.DataFrame,
    strategy: Strategy,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    risk_free_rate: float = 0.02,
    engine: str = "single",
    position_size: float | None = None,
    lot_sizer: LotSizer | None = None,
    max_concurrent_lots: int | None = None,
) -> EquityResult:
    """Generic equity-curve simulator: runs ``strategy`` for each holding-day cap.

    Output shape matches ``backtest_composite.simulate_equity_curve`` so the
    HTML renderer accepts both.

    Args:
        engine: ``"single"`` or ``"multi_lot"``.
        lot_sizer: only used when ``engine="multi_lot"`` — a ``LotSizer``
                   callable (e.g. ``FixedLotSizer(0.1)`` or ``VolTargetLotSizer(...)``)
                   that determines lot size per buy. Preferred over the
                   deprecated ``position_size``.
        position_size: deprecated alias of ``lot_sizer=FixedLotSizer(position_size)``;
                       kept for backwards compat (existing tests pass it as a
                       kwarg). Mutually exclusive with ``lot_sizer`` — passing
                       both raises ValueError. If both are None, defaults to
                       ``FixedLotSizer(0.1)``.
        max_concurrent_lots: only used when ``engine="multi_lot"`` — cap on
                             simultaneous open lots; None = uncapped by count
                             (cash still self-caps).
    """
    costs = TradeCosts(buy_cost=buy_cost, sell_cost=sell_cost)
    if engine == "single":
        bt = BacktestEngine(strategy, costs=costs, risk_free_rate=risk_free_rate)
    elif engine == "multi_lot":
        if lot_sizer is None:
            size = position_size if position_size is not None else 0.1
            lot_sizer = FixedLotSizer(size)
        elif position_size is not None:
            raise ValueError(
                "Pass either lot_sizer or position_size, not both"
            )
        bt = MultiLotBacktestEngine(
            strategy, lot_sizer=lot_sizer, costs=costs,
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
