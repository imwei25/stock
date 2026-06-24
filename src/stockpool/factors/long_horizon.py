"""长周期因子:长期反转 + 52 周高点占比。

补两个长 horizon 的经典风格信号,本库此前的动量/反转都集中在短中窗:

  * long_term_reversal —— De Bondt-Thaler 长期反转。过去 [t−N, t−skip] 的累计
    收益(默认 N=240≈12 月,skip=21≈1 月,**跳过最近一个月**以避开短期动量/
    微观结构噪声)。CNE6 专门有 Long-Term Reversal 风格因子。原始值即过去长期
    收益,方向(反转为负相关)由下游 IC / 选因子自动定;符号无需在因子内写死。

  * high_proximity —— George & Hwang (2004) 52 周高点。close / 近 N 日最高价,
    越接近 1 表示越靠近阶段高点。"离高点近"的股票随后常继续走强(锚定效应下
    的动量),是与传统动量低相关的另一条动量代理。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "long_term_reversal",
    sources=("custom",),
    types=("reversal", "time_series"),
    description="长期反转:过去 [t−N, t−21] 的累计收益(默认 N=240,跳过最近一月)。长期赢家随后常跑输;方向由 IC 自动定。",
)
class LongTermReversalFactor(Factor):
    # 跳过最近 ~1 个月,避开短期动量/反转污染长期信号
    _SKIP = 21

    def __init__(self, n: int = 240):
        if n <= self._SKIP:
            raise ValueError(f"window must be > skip({self._SKIP}), got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"long_term_reversal_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        # [t−N, t−skip] 区间收益:close[t−skip]/close[t−N] − 1。两端皆历史值,
        # look-ahead 安全。
        return close.shift(self._SKIP) / close.shift(self.n) - 1.0


@register(
    "high_proximity",
    sources=("custom",),
    types=("momentum", "time_series"),
    description="52 周高点占比 (George-Hwang):close / 近 N 日最高价(默认 N=240)。越接近 1 越靠阶段高点,随后常延续走强。",
)
class HighProximityFactor(Factor):
    def __init__(self, n: int = 240):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"high_proximity_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        rolling_high = panel["high"].rolling(self.n, min_periods=self.n).max()
        return close / rolling_high
