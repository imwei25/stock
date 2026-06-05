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
    description="Wilder ATR: EMA(alpha=1/n) of true range max(h-l, |h-c_prev|, |l-c_prev|)",
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
    description="CCI = (tp - SMA(tp, n)) / (0.015 * MAD(tp, n)),tp=(H+L+C)/3",
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
    description="日内振幅 (high-low)/close 的 N 日均值",
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
    description="Parkinson vol = sqrt(mean(ln(H/L)^2 / (4 ln 2), n))",
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
    description="Garman-Klass vol: 综合 OHLC 的极差估计",
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
