"""Strategy-agnostic backtesting framework.

Top-level imports for typical use:

    from stockpool.backtesting import (
        BacktestEngine, TradeCosts, buy_and_hold_baseline,
        CompositeVerdictStrategy,
    )

    engine = BacktestEngine(
        CompositeVerdictStrategy(weights, scoring, verdicts, indicators),
        costs=TradeCosts(buy_cost=0.0008, sell_cost=0.0013),
        risk_free_rate=0.02,
    )
    results = engine.sweep_holding_days(daily_df, [5, 10, 20])
    baseline = buy_and_hold_baseline(daily_df)

See ``docs/backtesting_framework.md`` for the full guide.
"""
from stockpool.backtesting.framework import (
    BacktestEngine,
    BacktestResult,
    BarContext,
    MultiLotBacktestEngine,
    PositionContext,
    Strategy,
    Trade,
    TradeCosts,
    buy_and_hold_baseline,
)
from stockpool.backtesting.metrics import (
    TRADING_DAYS_PER_YEAR,
    compute_metrics,
)
from stockpool.backtesting.strategies import (
    CompositeVerdictStrategy,
    MLFactorStrategy,
    SMACrossStrategy,
    VerdictExecution,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BarContext",
    "MultiLotBacktestEngine",
    "PositionContext",
    "Strategy",
    "Trade",
    "TradeCosts",
    "buy_and_hold_baseline",
    "compute_metrics",
    "TRADING_DAYS_PER_YEAR",
    "CompositeVerdictStrategy",
    "MLFactorStrategy",
    "VerdictExecution",
    "SMACrossStrategy",
]
