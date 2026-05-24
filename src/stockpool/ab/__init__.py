"""A/B testing tool — compare two strategies on the same per-stock universe.

Entry points:
    from stockpool.ab import ABConfig, load_ab_config
    from stockpool.ab import run_ab, run_single_arm, ABResult, ArmResult
    from stockpool.ab import render_ab_report

See docs/superpowers/specs/2026-05-24-ab-testing-design.md for the full design.
"""
from stockpool.ab.config import (
    ABConfig,
    ArmBacktestOverride,
    ArmOverride,
    build_effective_cfg,
    load_ab_config,
)
from stockpool.ab.report import render_ab_report
from stockpool.ab.runner import ABResult, ArmResult, run_ab, run_single_arm

__all__ = [
    "ABConfig",
    "ArmBacktestOverride",
    "ArmOverride",
    "build_effective_cfg",
    "load_ab_config",
    "ABResult",
    "ArmResult",
    "run_ab",
    "run_single_arm",
    "render_ab_report",
]
