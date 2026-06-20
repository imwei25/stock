"""EWMA 平滑因子族 (本 spec 自主补,论文 B 无对应).

5 个 base class × 半衰期 ∈ {5, 10, 20} = ~15 变体。
命名:``ewma_<signal>_hl<h>``,h 是 halflife。

后缀解析:from_suffix_args 把 ["hl10"] 解析成 halflife=10。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _parse_hl(args: list[str]) -> int:
    """suffix 形如 ["hl10"] → 10。"""
    if len(args) != 1 or not args[0].startswith("hl"):
        raise ValueError(f"expected ['hl<n>'], got {args!r}")
    return int(args[0][2:])


@register(
    "ewma_momentum",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="收盘价相对其指数平滑均线的偏离比例。正值 = 现价高于近期均价(强势上行);半衰期 h 控制远近权重。",
)
class EWMAMomentumFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_momentum_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        ema = c.ewm(halflife=self.halflife).mean()
        return (c - ema) / ema

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMAMomentumFactor":
        return cls(halflife=_parse_hl(args))


@register(
    "ewma_vol",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="用 RiskMetrics 风格指数平滑算出的近期收益波动率。比简单 rolling std 反应更快,半衰期短=对最新波动更敏感。",
)
class EWMAVolFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_vol_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        return ret.ewm(halflife=self.halflife).std()

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMAVolFactor":
        return cls(halflife=_parse_hl(args))


@register(
    "ewma_turnover_z",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="今日 log 成交量相对其指数平滑均值的 z 分数,衡量“今天有多反常”。极值 = 异常活跃或异常清淡。",
)
class EWMATurnoverZFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_turnover_z_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.ewm(halflife=self.halflife).mean()
        std = lv.ewm(halflife=self.halflife).std().replace(0.0, np.nan)
        return (lv - mean) / std

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMATurnoverZFactor":
        return cls(halflife=_parse_hl(args))


@register(
    "ewma_close_dev",
    sources=("builtin",),
    types=("trend", "time_series"),
    description="收盘价相对 EMA 均线的 z 分数(用 EWM std 标准化)。同时考虑偏离方向与偏离幅度,绝对值大代表显著偏离常态。",
)
class EWMACloseDevFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_close_dev_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        ema = c.ewm(halflife=self.halflife).mean()
        std = c.ewm(halflife=self.halflife).std()
        return (c - ema) / std

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMACloseDevFactor":
        return cls(halflife=_parse_hl(args))


@register(
    "ewma_volume_ratio",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="今日量除以昨日(及之前)指数平滑均量。正值放量,负值缩量;比简单 rolling 版本反应更快。",
)
class EWMAVolumeRatioFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_volume_ratio_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"]
        ema = v.ewm(halflife=self.halflife).mean().shift(1)
        return v / ema - 1.0

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMAVolumeRatioFactor":
        return cls(halflife=_parse_hl(args))
