"""rolling 直接统计量因子族 (论文 B original_* 28 个的精神复现).

公式都是 close / volume / range 的 rolling mean/std/skew/kurt 直接量,
不做 rank、不做归一化(除自身比例外)。

变体数 ~25,7 个 base class × 多个窗口参数。

ddof 选择:本族 ``std`` 类因子统一用 pandas ``.rolling(N).std()`` 默认的
``ddof=1``(样本标准差),与测试中的对照公式保持一致。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "close_std",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="收盘价 N 日波动率(用当前价归一化)。值越大代表股价摆动越剧烈,常用来识别震荡 vs 趋势状态。",
)
class CloseStdFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_std_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        return c.rolling(self.n).std() / c


@register(
    "close_skew",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="收盘价 N 日分布的偏度。正偏 = 多数日小涨偶尔大涨,负偏 = 多数日小跌偶尔大跌,极值常预示拐点。",
)
class CloseSkewFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_skew_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["close"].rolling(self.n).skew()


@register(
    "close_kurt",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="收盘价 N 日峰度。值越高代表“平时温吞、偶有极端波动”,可识别“看似平静实有暗涌”的票。",
)
class CloseKurtFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_kurt_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["close"].rolling(self.n).kurt()


@register(
    "volume_skew",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="成交量 N 日分布偏度。正偏 = 多数日量平稳偶尔放巨量(突发关注),可作为关注度异动信号。",
)
class VolumeSkewFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"volume_skew_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["volume"].rolling(self.n).skew()


@register(
    "volume_kurt",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="成交量 N 日峰度。值高 = 日常量小但常有偶发尖峰,体现脉冲式资金流入流出。",
)
class VolumeKurtFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"volume_kurt_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["volume"].rolling(self.n).kurt()


@register(
    "range_std",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="N 日日内振幅的波动率。衡量“波动率的波动率”,可分辨“持续宽幅震荡”与“偶尔宽幅”两种行情。",
)
class RangeStdFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"range_std_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        rng = panel["high"] - panel["low"]
        return rng.rolling(self.n).std() / panel["close"]


@register(
    "volume_std",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="成交量变异系数(N 日 std 除以均值)。值越大说明成交量分布越不稳定,可能正经历题材轮动。",
)
class VolumeStdFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"volume_std_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"]
        mean = v.rolling(self.n).mean()
        std = v.rolling(self.n).std()
        return std / mean.replace(0.0, np.nan)
