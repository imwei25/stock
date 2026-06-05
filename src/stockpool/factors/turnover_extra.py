"""短窗换手族 (论文 B extra_* 14 个的精神复现).

补 custom.py:turnover_zscore_60 (长窗) 之外的短/中窗换手指标。
v.replace(0.0, np.nan) 防停牌日 log(0) 污染,与 custom.py 一致。
3 个 base class × 4 窗口 = ~12 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "turnover_z",
    sources=("custom",),
    types=("volume", "time_series"),
    description="log(volume) 短窗 z-score (短 vs custom.turnover_zscore_60 长窗)",
)
class TurnoverZShortFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"turnover_z_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.rolling(self.n).mean()
        std = lv.rolling(self.n).std(ddof=0).replace(0.0, np.nan)
        return (lv - mean) / std


@register(
    "amount_z",
    sources=("custom",),
    types=("volume", "time_series"),
    description="log(volume*close) 短窗 z-score (成交额)",
)
class AmountZFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"amount_z_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        amount = (panel["volume"] * panel["close"]).replace(0.0, np.nan)
        la = np.log(amount)
        mean = la.rolling(self.n).mean()
        std = la.rolling(self.n).std(ddof=0).replace(0.0, np.nan)
        return (la - mean) / std


@register(
    "volume_ratio_short",
    sources=("custom",),
    types=("volume", "time_series"),
    description="volume / mean(volume, n).shift(1) - 1,短窗放/缩量",
)
class VolumeRatioShortFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"volume_ratio_short_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        mean = v.rolling(self.n).mean().shift(1)
        return v / mean - 1.0
