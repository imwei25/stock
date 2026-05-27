"""Portfolio-level A/B testing (spec 2026-05-24, PR-4).

Parallel to ``stockpool.ab`` (per-stock A/B) but with a different override
surface: only ``strategy`` (whole-replace) and ``portfolio_backtest``
(field-level merge) are allowed.
"""
from stockpool.portfolio_ab.config import (
    PortfolioABConfig,
    PortfolioArmOverride,
    build_effective_cfg,
    load_portfolio_ab_config,
)
from stockpool.portfolio_ab.report import render_portfolio_ab_report
from stockpool.portfolio_ab.runner import (
    ABResult,
    ArmResult,
    run_portfolio_ab,
    run_single_arm,
)

__all__ = [
    "PortfolioABConfig",
    "PortfolioArmOverride",
    "load_portfolio_ab_config",
    "build_effective_cfg",
    "ABResult",
    "ArmResult",
    "run_portfolio_ab",
    "run_single_arm",
    "render_portfolio_ab_report",
]
