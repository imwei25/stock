"""A-share specific custom factors (panel-native).

补 WQ101 没覆盖的 A 股专属信号:同业超额收益、涨停频次、异常活跃度。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.context import get_sector_map
from stockpool.factors.registry import register


@register(
    "industry_relative_strength",
    sources=("custom",),
    types=("momentum", "industry_neutral", "cross_sectional"),
    description="N 日动量减去同行业中位动量 (sector_map 通过 factors.context 注入)",
)
class IndustryRelativeStrengthFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"industry_relative_strength_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ret = close.pct_change(self.n, fill_method=None)  # T×N

        sector_map = get_sector_map()
        if not sector_map:
            return pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)

        # Label every column with its sector ("__unknown__" for codes not in map).
        sector_series = pd.Series(
            {code: sector_map.get(code, "__unknown__") for code in ret.columns},
            name="sector",
        )

        # groupby column → sector; transform within each sector daily.
        groups = ret.T.groupby(sector_series)
        sector_median = groups.transform("median").T          # T×N
        sector_count = groups.transform("count").T            # T×N count of non-NaN
        # singleton sector (count<2 on that day) → NaN to avoid self-minus-self
        sector_median = sector_median.where(sector_count >= 2, np.nan)

        result = ret - sector_median

        # codes not in sector_map → entire column NaN
        unknown_cols = [
            c for c in result.columns
            if sector_map.get(c, "__unknown__") == "__unknown__"
        ]
        if unknown_cols:
            result.loc[:, unknown_cols] = np.nan
        return result
