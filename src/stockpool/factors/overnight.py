"""隔夜/日内收益分解族 (overnight vs intraday return decomposition).

A 股 T+1 结构下有文献记载的隔夜异象:高隔夜收益股倾向反转(散户集合竞价
情绪压力),隔夜-日内收益差携带与 close-to-close 动量正交的信息。本族从
已有 OHLCV 直接构造,无需任何新数据:

  * ``overnight_mom_N``  — N 日累计隔夜 log 收益 Σ ln(open_t / close_{t-1})
  * ``intraday_mom_N``   — N 日累计日内 log 收益 Σ ln(close_t / open_t)
  * ``oi_spread_N``      — 两者之差(隔夜 − 日内),分解视角的核心信号
  * ``overnight_vol_N``  — 隔夜收益的 N 日波动(跳空风险 proxy)

均为 time_series 因子,看 raw 价格(与 mask 设计一致);open/prev-close
非正或缺失 → NaN。2026-07-18 审计循环方向⑤(scout 提案:library 中此前
不存在任何 gap/overnight 家族)。须经 `factors analyze` 覆盖率 gate +
pick-by-ic + AB 验证后才可能进生产 selection。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _log_overnight(panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """ln(open_t / close_{t-1}),非正价格 → NaN。"""
    o = panel["open"].where(panel["open"] > 0)
    c_prev = panel["close"].where(panel["close"] > 0).shift(1)
    return np.log(o / c_prev)


def _log_intraday(panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """ln(close_t / open_t),非正价格 → NaN。"""
    o = panel["open"].where(panel["open"] > 0)
    c = panel["close"].where(panel["close"] > 0)
    return np.log(c / o)


@register(
    "overnight_mom",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="N 日累计隔夜 log 收益 Σ ln(open/prev_close)。捕获集合竞价段的持续跳空方向;A 股文献中高隔夜收益常伴随后续反转。",
)
class OvernightMomFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"overnight_mom_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return _log_overnight(panel).rolling(self.n, min_periods=self.n).sum()


@register(
    "intraday_mom",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="N 日累计日内 log 收益 Σ ln(close/open)。剥离隔夜跳空后的盘中趋势成分,与 close-to-close 动量互补。",
)
class IntradayMomFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"intraday_mom_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return _log_intraday(panel).rolling(self.n, min_periods=self.n).sum()


@register(
    "oi_spread",
    sources=("builtin",),
    types=("momentum", "reversal", "time_series"),
    description="N 日隔夜累计收益 − 日内累计收益。正值 = 涨幅主要来自跳空(情绪驱动,倾向回吐);负值 = 涨幅来自盘中(资金推动,更可持续)。",
)
class OvernightIntradaySpreadFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"oi_spread_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        on = _log_overnight(panel).rolling(self.n, min_periods=self.n).sum()
        intra = _log_intraday(panel).rolling(self.n, min_periods=self.n).sum()
        return on - intra


@register(
    "overnight_vol",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="隔夜 log 收益的 N 日滚动标准差。跳空风险 proxy:高隔夜波动 = 信息多在盘外释放(公告/情绪),与盘中波动族(ATR/Parkinson)正交。",
)
class OvernightVolFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"overnight_vol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return _log_overnight(panel).rolling(self.n, min_periods=self.n).std(ddof=0)
