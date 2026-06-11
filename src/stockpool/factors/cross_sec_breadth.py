"""截面市场宽度族 (论文 B cs_rank_* 6 个的精神复现).

全市场标量(T×1)广播到 T×N。涨停股 / 停牌股 **不过滤**,
与 mask config 无关(spec §6.1.2)。

5 个 base class。breadth_above_ma 带窗口参数。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _broadcast(scalar: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """T×1 series → T×N DataFrame,广播到 like 的所有列。"""
    return pd.DataFrame(
        np.broadcast_to(scalar.to_numpy()[:, None], like.shape).copy(),
        index=like.index, columns=like.columns,
    )


@register(
    "breadth_above_ma",
    sources=("builtin",),
    types=("cross_sectional", "time_series", "broadcast"),
    description="全市场有多少比例股票当日收盘高于自己的 N 日均线。读数 > 0.5 多数股偏强;< 0.3 大盘普遍弱势。",
)
class BreadthAboveMAFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"breadth_above_ma_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        ma = c.rolling(self.n).mean()
        scalar = (c > ma).mean(axis=1)
        return _broadcast(scalar, c)


@register(
    "breadth_advance",
    sources=("builtin",),
    types=("cross_sectional", "time_series", "broadcast"),
    description="当天全市场涨股占比。> 0.6 全面普涨,< 0.4 普跌,中间值意味着分化。",
)
class BreadthAdvanceFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "breadth_advance"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        scalar = (c.pct_change(fill_method=None) > 0).mean(axis=1)
        return _broadcast(scalar, c)


@register(
    "breadth_limit_up",
    sources=("builtin",),
    types=("cross_sectional", "time_series", "broadcast"),
    description="当天触涨停股票占比。同步突高 = 题材热点炒作进入高峰,可作为情绪温度计。",
)
class BreadthLimitUpFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "breadth_limit_up"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        ret = c.pct_change(fill_method=None)
        scalar = (ret > 0.099).mean(axis=1)
        return _broadcast(scalar, c)


@register(
    "breadth_dispersion",
    sources=("builtin",),
    types=("cross_sectional", "volatility", "time_series", "broadcast"),
    description="当日个股收益率的横截面 std。大 = 个股分化严重(题材轮动);小 = 齐涨齐跌(系统性行情)。",
)
class BreadthDispersionFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "breadth_dispersion"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        scalar = c.pct_change(fill_method=None).std(axis=1)
        return _broadcast(scalar, c)


@register(
    "breadth_pos_skew",
    sources=("builtin",),
    types=("cross_sectional", "time_series", "broadcast"),
    description="当日横截面收益率的偏度。正偏 = 少数股大涨拉高指数(头部领涨);负偏 = 少数股大跌拖累指数。",
)
class BreadthPosSkewFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "breadth_pos_skew"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        scalar = c.pct_change(fill_method=None).skew(axis=1)
        return _broadcast(scalar, c)
