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
    """
    sector_map: ClassVar[dict[str, str]] = {}


def set_sector_map(mapping: Mapping[str, str]) -> None:
    """Inject ``{code: sector_name}`` for downstream factors."""
    _FactorContext.sector_map = dict(mapping)


def get_sector_map() -> dict[str, str]:
    """Return a snapshot of the current sector_map (empty if unset)."""
    return dict(_FactorContext.sector_map)


def indneutralize_with_context(x: pd.DataFrame) -> pd.DataFrame:
    """Industry-neutralise ``x`` (T×N) using current sector context.

    If sector_map is empty, falls back to cross-sectional demean
    (subtract per-day row mean, equivalent to ``ops.cs_demean``).
    """
    if _FactorContext.sector_map:
        return ops.indneutralize(x, _FactorContext.sector_map)
    return ops.cs_demean(x)
