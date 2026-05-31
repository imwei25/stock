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


# ── IndustryRelativeStrengthFactor ──────────────────────────────────────────

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
            # Empty sector_map produces an all-NaN factor, which silently
            # poisons the factor_panel disk cache (the sig hash doesn't track
            # sector_map state). Fail loud so callers must inject a real
            # map via factors.context.set_sector_map before building.
            raise RuntimeError(
                "IndustryRelativeStrengthFactor requires a non-empty sector_map. "
                "Call stockpool.factors.context.set_sector_map(...) before "
                "computing this factor (e.g. via "
                "stockpool.industry_map.load_or_build_industry_map)."
            )

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


# ── LimitUpCountFactor ──────────────────────────────────────────────────────

@register(
    "limit_up_count",
    sources=("custom",),
    types=("momentum", "time_series"),
    description="近 N 日触及涨停 (close > prev_close × 1.099) 的次数",
)
class LimitUpCountFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"limit_up_count_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ret = close.pct_change(fill_method=None)
        # 主板涨停 10%, 留 0.1% tolerance 免 round-off
        # ST/科创/北交已被 fetch-universe 过滤, 此处不区分
        is_limit_up = (ret > 0.099).astype(float)
        # bar 0's pct_change is NaN → astype(float) → 0.0; force to NaN
        # so rolling.sum with min_periods=n properly warmups
        is_limit_up.iloc[0] = np.nan
        return is_limit_up.rolling(self.n, min_periods=self.n).sum()


# ── TurnoverZScoreFactor ────────────────────────────────────────────────────

@register(
    "turnover_zscore",
    sources=("custom",),
    types=("volume", "time_series"),
    description="log(volume) 的 N 日时间序列 z-score, 反映异常活跃度",
)
class TurnoverZScoreFactor(Factor):
    def __init__(self, n: int = 60):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"turnover_zscore_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.rolling(self.n, min_periods=self.n).mean()
        std = lv.rolling(self.n, min_periods=self.n).std(ddof=0)
        std = std.replace(0.0, np.nan)
        return (lv - mean) / std
