"""log(market_cap) panel construction for OLS-based neutralization.

mcap = close × totalShare, PIT-aligned by pubDate via the same helper that
``factors.fundamentals`` uses. Reuses the 30-day baostock balance-table parquet
cache — no new fetch path.

See: docs/superpowers/specs/2026-06-06-mcap-neutralization-design.md
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def build_log_mcap_panel(
    panel: Mapping[str, pd.DataFrame],
    cache_dir,
) -> pd.DataFrame:
    """Build T×N log(market_cap) panel aligned to ``panel["close"]``.

    Args:
        panel: must contain a ``"close"`` T×N wide-frame.
        cache_dir: passed to ``load_or_build_fundamentals`` for the balance
            parquet cache. ``None`` skips caching (live fetch each call).

    Returns:
        T×N DataFrame, same index/columns as ``panel["close"]``. Cells where
        ``totalShare`` is missing or ``mcap <= 0`` are NaN — the per-day OLS
        downstream drops those rows.
    """
    from stockpool.fundamentals_loader import load_or_build_fundamentals
    from stockpool.factors.fundamentals import _pit_align

    close = panel["close"]
    balance = load_or_build_fundamentals("balance", cache_dir=cache_dir)
    if balance is None or balance.empty:
        return pd.DataFrame(
            np.nan, index=close.index, columns=close.columns,
        )
    shares_panel = _pit_align(balance, "totalShare", close)
    mcap = close * shares_panel
    return np.log(mcap.where(mcap > 0))
