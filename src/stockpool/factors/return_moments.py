"""日收益率分布矩因子(已实现偏度 / 峰度 / 下行波动)。

⚠️ 与 ``original_stats.py`` 的区别:那里的 ``close_skew`` / ``close_kurt`` 是对
**价格水平**取偏度/峰度,业界标准(以及高频"已实现矩"派生到日频的版本)是对
**日收益率**取矩。本模块补齐收益率口径的分布矩:

  * ret_skew  —— 收益率偏度。A 股负偏(左尾厚)的股票常有更高预期收益(偏度厌恶)。
  * ret_kurt  —— 收益率峰度。尾部肥瘦,刻画跳跃风险。
  * downside_vol —— 下行半离差(只算负收益的二阶矩,LPM2 around 0)。下行风险在
                    A 股是较稳健的定价因子。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "ret_skew",
    sources=("custom",),
    types=("volatility", "time_series"),
    description="近 N 日日收益率偏度。负偏(左尾厚)股票常有偏度溢价;区别于 close_skew 的价格偏度。",
)
class ReturnSkewFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 1:
            raise ValueError(f"window must be > 1, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"ret_skew_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        return ret.rolling(self.n, min_periods=self.n).skew()


@register(
    "ret_kurt",
    sources=("custom",),
    types=("volatility", "time_series"),
    description="近 N 日日收益率峰度。尾部肥瘦/跳跃风险度量;区别于 close_kurt 的价格峰度。",
)
class ReturnKurtFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 3:
            raise ValueError(f"window must be > 3, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"ret_kurt_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        return ret.rolling(self.n, min_periods=self.n).kurt()


@register(
    "downside_vol",
    sources=("custom",),
    types=("volatility", "time_series"),
    description="近 N 日下行半波动:仅负收益的二阶矩开方 (LPM2)。只惩罚下跌,比对称 std 更贴近真实风险厌恶。",
)
class DownsideVolFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"downside_vol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        # 正收益日记 0(clip upper=0),负收益保留 → 半离差;
        # 第一行 pct_change 为 NaN,clip 后仍 NaN,rolling min_periods 自然 warmup。
        neg = ret.clip(upper=0.0)
        msq = neg.pow(2).rolling(self.n, min_periods=self.n).mean()
        return np.sqrt(msq)
