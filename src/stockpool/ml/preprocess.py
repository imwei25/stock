"""Cross-sectional preprocessing pipeline for ML factor panels.

Three stateless steps, applied per-day (cross-sectional):

  * ``winsorize_panel(df, lo, hi)``  — clip to per-day [lo, hi] quantiles
  * ``cs_zscore_panel(df)``           — per-day (x - μ_t) / σ_t
  * ``industry_neutralize_panel(df, sector_map)``
                                      — per-day within-industry demean

Wrapped by ``apply_preprocess_pipeline`` which honors a ``PreprocessConfig``.

Look-ahead safe: each function consumes only per-day cross-sectional info,
never references other rows. See spec
``docs/superpowers/specs/2026-06-06-factor-preprocessing-phase1-design.md``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stockpool.config import PreprocessConfig

log = logging.getLogger(__name__)


def winsorize_panel(
    df: pd.DataFrame, lower: float, upper: float,
) -> pd.DataFrame:
    """Per-day cross-sectional clip to ``[lower quantile, upper quantile]``.

    Args:
        df: T × N factor wide-frame (date index, code columns).
        lower: lower quantile bound, e.g. ``0.01``.
        upper: upper quantile bound, e.g. ``0.99``.

    Returns:
        Same-shape DataFrame with values outside [q_lo(t), q_hi(t)] clipped.
        All-NaN rows are returned unchanged (shape preserved).

    Raises:
        ValueError: if not ``0 < lower < upper < 1``.
    """
    if not (0 < lower < upper < 1):
        raise ValueError(
            f"winsorize bounds must satisfy 0 < lower < upper < 1, "
            f"got ({lower}, {upper})"
        )
    lo_q = df.quantile(lower, axis=1)
    hi_q = df.quantile(upper, axis=1)
    out = df.clip(lower=lo_q, upper=hi_q, axis=0)
    return out
