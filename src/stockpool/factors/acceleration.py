"""加速度族 (论文 B change_* 5 个的精神复现).

动量/换手的二阶差分,捕获趋势变速。3 个 base × 3 窗口 = ~9 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "mom_accel",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="动量的“加速度”:N 日涨幅减去 N 日前的同期 N 日涨幅。正值 = 涨势在加速;负值 = 涨势在减速/反转。",
)
class MomAccelFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"mom_accel_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mom = panel["close"].pct_change(self.n, fill_method=None)
        return mom - mom.shift(self.n)


@register(
    "vol_accel",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="log 成交量的二阶差分,捕捉成交活跃度由热到冷或由冷到热的“转折速度”。",
)
class VolAccelFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vol_accel_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        return lv - 2.0 * lv.shift(self.n) + lv.shift(2 * self.n)


@register(
    "turnover_accel",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="换手 z 分数的 N 日变化,识别活跃度趋势的拐点(由热转冷或由冷转热)。",
)
class TurnoverAccelFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"turnover_accel_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.rolling(self.n).mean()
        std = lv.rolling(self.n).std(ddof=0).replace(0.0, np.nan)
        tz = (lv - mean) / std
        return tz - tz.shift(self.n)
