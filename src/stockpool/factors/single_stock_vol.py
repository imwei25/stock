"""单股波动 / 振幅族 (论文 B stock_* 22 个的精神复现).

ATR / CCI / 日内振幅 / Parkinson vol / Garman-Klass vol。
5 个 base × 4 窗口 = ~20 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "atr",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="Wilder 经典 ATR:N 日真实波幅的指数平滑值。衡量“一根 K 线平均能走多远”,常用于止损/仓位计算。",
)
class ATRFactor(Factor):
    def __init__(self, n: int = 14):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"atr_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        h, l, c = panel["high"], panel["low"], panel["close"]
        c_prev = c.shift(1)
        tr1 = (h - l).to_numpy()
        tr2 = (h - c_prev).abs().to_numpy()
        tr3 = (l - c_prev).abs().to_numpy()
        # fmax 是 NaN-safe: 第一行 c_prev=NaN → tr2/tr3 NaN → 取 tr1
        tr_arr = np.fmax(np.fmax(tr1, tr2), tr3)
        tr = pd.DataFrame(tr_arr, index=h.index, columns=h.columns)
        # Wilder smoothing: alpha=1/n EWM
        return tr.ewm(alpha=1.0 / self.n, adjust=False).mean()


@register(
    "cci",
    sources=("builtin",),
    types=("reversal", "time_series"),
    description="顺势指标 CCI。值远离 0(如 ±100)代表“典型价”偏离常态过度,常用来抓超买/超卖反转。",
)
class CCIFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"cci_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        tp = (panel["high"] + panel["low"] + panel["close"]) / 3.0
        sma = tp.rolling(self.n).mean()
        # MAD = mean(|tp - SMA|)
        mad = (tp - sma).abs().rolling(self.n).mean().replace(0.0, np.nan)
        return (tp - sma) / (0.015 * mad)


@register(
    "amp",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="N 日日内振幅 (高-低)/收盘 的均值。波动率代理,数值越大说明这只票每天来回幅度越大。",
)
class AmpFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"amp_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        amp = (panel["high"] - panel["low"]) / panel["close"]
        return amp.rolling(self.n).mean()


@register(
    "park_vol",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="Parkinson 波动率,只用 high/low 估算,比仅看收盘价的 std 更稳健、对真实波动更敏感。",
)
class ParkinsonVolFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"park_vol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ratio = panel["high"] / panel["low"].replace(0.0, np.nan)
        x = (np.log(ratio)) ** 2 / (4.0 * np.log(2.0))
        return np.sqrt(x.rolling(self.n).mean())


@register(
    "gk_vol",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="Garman-Klass 波动率,综合 OHLC 算出来,既比 close-only 准也比 high-low 准;对极端跳空更敏感。",
)
class GarmanKlassVolFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"gk_vol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        h, l, c, o = panel["high"], panel["low"], panel["close"], panel["open"]
        l_safe = l.replace(0.0, np.nan)
        o_safe = o.replace(0.0, np.nan)
        log_hl = np.log(h / l_safe)
        log_co = np.log(c / o_safe)
        x = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
        return np.sqrt(x.rolling(self.n).mean())
