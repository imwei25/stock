"""流动性 / 非流动性因子。

Amihud (2002) 非流动性 = 平均(|日收益| / 日成交额)。衡量"单位成交额能把
价格推动多少",值越大越不流动。A 股流动性溢价显著(非流动股要求更高预期
收益),是商业模型(Barra Liquidity 的反面)和学界都重视的一类,本库此前只有
换手 z 分数、缺一个量纲清晰的非流动性度量,这里补上。

成交额用 ``volume * close`` 估算(panel 无独立 amount 字段,与 ``amount_z``
口径一致)。停牌/0 成交额日 → NaN,不污染窗口。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "amihud",
    sources=("custom",),
    types=("liquidity", "volume", "time_series"),
    description="Amihud 非流动性:近 N 日均值(|日收益|/成交额),×1e8 调到可读量纲。值大=不流动,通常要求更高预期收益。",
)
class AmihudIlliquidityFactor(Factor):
    """ILLIQ = mean_N( |ret_t| / amount_t ),amount = volume×close。"""

    # 乘一个常数把量纲抬到 O(1) 附近(成交额以元计,原始值极小);
    # 截面 zscore 会再归一,这个尺度只为可读性 + 防下溢。
    _SCALE = 1e8

    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"amihud_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ret = close.pct_change(fill_method=None).abs()
        amount = (panel["volume"] * close).replace(0.0, np.nan)
        daily = ret / amount
        return daily.rolling(self.n, min_periods=self.n).mean() * self._SCALE
