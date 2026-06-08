"""Shared factor context (sector_map et al.).

Lifted from ``factors/wq101.py`` to make sector-aware factors outside wq101
share the same injection point. ``factors/wq101.py`` re-exports
``set_sector_map`` and ``get_sector_map`` for backward compatibility.
"""
from __future__ import annotations

from typing import ClassVar, Mapping

import pandas as pd

from stockpool.factors import ops


class _FactorContext:
    """Module-wide context for sector-aware factors.

    Set via ``set_sector_map`` at the strategy / analysis entry point;
    factors read via ``get_sector_map`` (returns a copy).

    ``mcap_panel`` holds an optional T×N ``log(total_market_cap)`` wide-frame
    used by the ``market_cap_neutralize`` preprocess step (Phase 2). It is set
    by the pool-prep entry point (``backtest_runner.prepare_pool`` /
    ``ab.runner``) and read by ``strategy_factory.build_factor_panel``.
    """
    sector_map: ClassVar[dict[str, str]] = {}
    mcap_panel: ClassVar[pd.DataFrame | None] = None


def set_sector_map(mapping: Mapping[str, str]) -> None:
    """Inject ``{code: sector_name}`` for downstream factors."""
    _FactorContext.sector_map = dict(mapping)


def get_sector_map() -> dict[str, str]:
    """Return a snapshot of the current sector_map (empty if unset)."""
    return dict(_FactorContext.sector_map)


def set_mcap_panel(panel: "pd.DataFrame | None") -> None:
    """Inject a T×N ``log(total_market_cap)`` panel for market-cap neutralize.

    Pass ``None`` to clear. Stored by reference (read-only consumer); callers
    should not mutate the frame after injecting it.
    """
    _FactorContext.mcap_panel = panel


def get_mcap_panel() -> "pd.DataFrame | None":
    """Return the current log-mcap panel (``None`` if unset)."""
    return _FactorContext.mcap_panel


def indneutralize_with_context(x: pd.DataFrame) -> pd.DataFrame:
    """Industry-neutralise ``x`` (T×N) using current sector context.

    If sector_map is empty, falls back to cross-sectional demean
    (subtract per-day row mean, equivalent to ``ops.cs_demean``).
    """
    if _FactorContext.sector_map:
        return ops.indneutralize(x, _FactorContext.sector_map)
    return ops.cs_demean(x)
