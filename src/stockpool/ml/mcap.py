"""log(market_cap) panel construction for OLS-based neutralization.

mcap = close × totalShare, PIT-aligned by pubDate via the same helper that
``factors.fundamentals`` uses.

NOTE on data source: baostock 的 "balance" 表(``query_balance_data``)是
*偿债能力指标*(currentRatio / quickRatio / liabilityToAsset 等),并不含
``totalShare``。``totalShare`` 在 baostock 的 *profit* 表
(``query_profit_data``)里 —— 那个表除了 ROE / 利润率以外还附带每股股本与
流通股本。原 spec 把它写成 balance 是受 baostock 表名误导。

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
        cache_dir: passed to ``load_or_build_fundamentals`` for the profit
            parquet cache. ``None`` skips caching (live fetch each call).

    Returns:
        T×N DataFrame, same index/columns as ``panel["close"]``. Cells where
        ``totalShare`` is missing or ``mcap <= 0`` are NaN — the per-day OLS
        downstream drops those rows.
    """
    from stockpool.fundamentals_loader import load_or_build_fundamentals
    from stockpool.factors.fundamentals import _pit_align

    close = panel["close"]
    profit = load_or_build_fundamentals("profit", cache_dir=cache_dir)
    if profit is None or profit.empty:
        return pd.DataFrame(
            np.nan, index=close.index, columns=close.columns,
        )
    shares_panel = _pit_align(profit, "totalShare", close)
    mcap = close * shares_panel
    return np.log(mcap.where(mcap > 0))
