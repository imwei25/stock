"""Portfolio-level backtest framework (spec 2026-05-24).

Public surface:
  * ``PortfolioStrategy`` — cross-sectional ABC (per-bar scoring)
  * ``PrecomputedScoreStrategy`` — wrap a pre-built (T×N) score panel
  * ``precompute_scores_from_legacy`` — adapter from per-stock ``Strategy``
  * ``PortfolioEngine`` — top-K equal-weight engine with periodic rebalance
  * ``PortfolioRunConfig`` (config) / ``PortfolioTrade`` / ``PortfolioBacktestResult``
"""
from stockpool.portfolio.eligibility import EligibilityFilter
from stockpool.portfolio.engine import PortfolioEngine
from stockpool.portfolio.ensemble import EnsembleResult, StaggeredRunner
from stockpool.portfolio.result import PortfolioBacktestResult, PortfolioTrade
from stockpool.portfolio.scoring import precompute_scores_from_legacy
from stockpool.portfolio.strategy import (
    PortfolioStrategy,
    PrecomputedScoreStrategy,
)

__all__ = [
    "PortfolioStrategy",
    "PrecomputedScoreStrategy",
    "precompute_scores_from_legacy",
    "PortfolioEngine",
    "PortfolioBacktestResult",
    "PortfolioTrade",
    "EligibilityFilter",
    "StaggeredRunner",
    "EnsembleResult",
]
