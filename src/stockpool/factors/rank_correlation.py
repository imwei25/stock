"""秩相关合成族 (论文 B old_* 50 个的精神复现).

基于 ops.correlation 和 ops.rank,产出价格秩 × 成交量秩等组合的滚动相关。
5 个 base × 4 窗口 = ~20 变体。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "corr_pv",
    sources=("builtin",),
    types=("cross_sectional", "volume", "time_series"),
    description="近 N 日“价格横截面秩”与“成交量横截面秩”的滚动相关。捕捉“领涨股是否也是放量股”。",
)
class CorrPVFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_pv_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        r_c = ops.rank(panel["close"])
        r_v = ops.rank(panel["volume"])
        return ops.correlation(r_c, r_v, self.n)


@register(
    "corr_high_low",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="N 日内 high 与 low 的滚动相关。值接近 1 = K 线上下沿同步运动(整体趋势盘);接近 0 = 形态混乱。",
)
class CorrHighLowFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_high_low_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return ops.correlation(panel["high"], panel["low"], self.n)


@register(
    "corr_close_vwap",
    sources=("builtin",),
    types=("trend", "time_series"),
    description="收盘价与 VWAP 在 N 日内的相关性,刻画“收盘价是否一直跟随日内均价”。",
)
class CorrCloseVWAPFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_close_vwap_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return ops.correlation(panel["close"], ops.vwap(panel), self.n)


@register(
    "corr_mom_vol",
    sources=("builtin",),
    types=("momentum", "volume", "time_series"),
    description="日收益率与日量变化的滚动相关。正 = 涨升量涨/跌升量降(同向);负 = 涨升量降(背离,常见为分歧)。",
)
class CorrMomVolFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_mom_vol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        vchg = panel["volume"].pct_change(fill_method=None)
        return ops.correlation(ret, vchg, self.n)


@register(
    "corr_close_close_lag",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="今日 close 与上一日 close 在 N 日窗口的相关性,本质是滚动一阶自相关。值高 = 惯性强,低 = 反转多。",
)
class CorrCloseCloseLagFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_close_close_lag_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        return ops.correlation(c, c.shift(1), self.n)
