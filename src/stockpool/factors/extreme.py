"""极值 / 彩票 (lottery) 因子。

Bali, Cakici & Whitelaw (2011) MAX 因子:过去 N 日的单日最大收益。投资者对
"彩票型"股票(偶尔暴涨)有偏好、愿意溢价买入,导致这类股票随后预期收益偏低
(MAX 与未来收益负相关)。A 股散户占比高、投机性强,该效应通常更明显。

对称补一个 MIN(过去 N 日单日最小收益,即最大单日跌幅)。两者纯 OHLCV 可算。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "max_ret",
    sources=("custom",),
    types=("reversal", "time_series"),
    description="近 N 日单日最大涨幅 (Bali MAX 彩票因子)。值大=有暴涨/投机属性,随后预期收益常偏低(负向)。",
)
class MaxReturnFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"max_ret_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        return ret.rolling(self.n, min_periods=self.n).max()


@register(
    "min_ret",
    sources=("custom",),
    types=("reversal", "time_series"),
    description="近 N 日单日最大跌幅(MAX 的对称面)。极端负收益日的幅度,刻画下行尾部冲击。",
)
class MinReturnFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"min_ret_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        return ret.rolling(self.n, min_periods=self.n).min()
