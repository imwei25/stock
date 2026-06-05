"""收盘位置动量族 (论文 B best_* 21 个的精神复现).

pos_raw = (close - low) / (high - low),涨停封板日 range=0 时 NaN。
3 个 base × 多个窗口 = ~15 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _pos_raw(panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """(close - low) / (high - low),range=0 时 NaN。"""
    rng = (panel["high"] - panel["low"]).replace(0.0, np.nan)
    return (panel["close"] - panel["low"]) / rng


@register(
    "close_pos",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="(close - low) / (high - low) 的 N 日均值 ∈ [0, 1]",
)
class ClosePositionFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_pos_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return _pos_raw(panel).rolling(self.n).mean()


@register(
    "close_pos_cum",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="(pos - 0.5) 的 N 日累积偏离,正多空",
)
class ClosePositionCumFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_pos_cum_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return (_pos_raw(panel) - 0.5).rolling(self.n).sum()


@register(
    "close_pos_ema",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="收盘位置 pos 的 EMA(span=n) 平滑",
)
class ClosePositionEMAFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_pos_ema_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return _pos_raw(panel).ewm(span=self.n, adjust=False).mean()
