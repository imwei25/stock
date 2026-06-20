"""Result types for portfolio-level backtests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class PortfolioTrade:
    """One closed (or still-open at end-of-backtest) position."""
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    weight_at_entry: float
    ret: float            # net of buy_cost and sell_cost (0 if still open)
    days_held: int
    exit_reason: Literal[
        "rebalance_drop",          # PR-1 uses this for any non-survivor
        "no_longer_eligible",      # PR-2 hook
        "end_of_backtest",         # forced close at last bar for metrics
    ]


@dataclass
class PortfolioBacktestResult:
    """One run of ``PortfolioEngine``."""
    curve: pd.DataFrame                # date / equity / num_positions / cash_ratio
    trades: list[PortfolioTrade]
    rebalance_log: pd.DataFrame        # date / target_codes / num_target
    metrics: dict
    strategy_name: str
    config_hash: str = ""
