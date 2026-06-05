"""VWAP 偏离族 (论文 B better_* 28 个的精神复现).

VWAP proxy: (high + low + close) / 3 (复用 ops.vwap)。
4 个 base class × 5 窗口 ∈ {3, 5, 10, 20, 60} = ~20 变体。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "vwap_dev",
    sources=("builtin",),
    types=("trend", "volume", "time_series"),
    description="(close - vwap) / vwap 的 N 日均值",
)
class VWAPDevFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vwap_dev_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        vwap = ops.vwap(panel)
        dev = (panel["close"] - vwap) / vwap
        return dev.rolling(self.n).mean()


@register(
    "vwap_weighted_mom",
    sources=("builtin",),
    types=("momentum", "volume", "time_series"),
    description="量加权偏离动量: sum_d((close-vwap)*volume) / sum_d(volume) / vwap[t]",
)
class VWAPWeightedMomFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vwap_weighted_mom_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        vwap = ops.vwap(panel)
        v = panel["volume"]
        weighted_dev = (panel["close"] - vwap) * v
        num = weighted_dev.rolling(self.n).sum()
        den = v.rolling(self.n).sum()
        return num / den / vwap


@register(
    "vwap_above_ratio",
    sources=("builtin",),
    types=("trend", "time_series"),
    description="N 日内 close > vwap 的天数比例 ∈ [0, 1]",
)
class VWAPAboveRatioFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vwap_above_ratio_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        vwap = ops.vwap(panel)
        above = (panel["close"] > vwap).astype(float)
        return above.rolling(self.n).mean()


@register(
    "vwap_dev_std",
    sources=("builtin",),
    types=("volatility", "volume", "time_series"),
    description="(close - vwap) / vwap 的 N 日 std",
)
class VWAPDevStdFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vwap_dev_std_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        vwap = ops.vwap(panel)
        dev = (panel["close"] - vwap) / vwap
        return dev.rolling(self.n).std(ddof=0)
