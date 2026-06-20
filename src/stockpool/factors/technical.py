"""Built-in technical factors (panel-native).

每个因子 ``compute(panel) -> DataFrame``,在 T × N 宽表上直接做向量化运算,
不需要 per-stock loop。原来基于 ``stockpool.indicators`` 的公式被内联,因为
indicators 是 long-form / 单股 API。

types 至少含 ``"time_series"``;有趋势/反转/波动/量等子标签便于 HTML 粗筛。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.registry import register


# ─────────────────────────────────────────────────────────────────────────────
# Momentum 系
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "momentum",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="N 日涨跌幅,过去 N 天股价的累计涨幅。最经典的趋势/动量信号。",
)
class MomentumFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"momentum window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"momentum_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["close"].pct_change(self.n, fill_method=None)


@register(
    "macd_hist",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="MACD 红柱/绿柱的高度。正且变大 = 多头加速;负且变大 = 空头加速;在 0 附近 = 动能枯竭。",
)
class MACDHistFactor(Factor):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return "macd_hist"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ema_f = close.ewm(span=self.fast, adjust=False).mean()
        ema_s = close.ewm(span=self.slow, adjust=False).mean()
        dif = ema_f - ema_s
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        return 2.0 * (dif - dea)


@register(
    "macd_dif_norm",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="MACD DIF 线除以收盘价,把不同价位的票放在同一尺度,方便横向比较“谁更强势”。",
)
class MACDDifNormFactor(Factor):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return "macd_dif_norm"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ema_f = close.ewm(span=self.fast, adjust=False).mean()
        ema_s = close.ewm(span=self.slow, adjust=False).mean()
        return (ema_f - ema_s) / close


# ─────────────────────────────────────────────────────────────────────────────
# Reversal / 超买超卖
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "rsi_centered",
    sources=("builtin",),
    types=("reversal", "time_series"),
    description="RSI 减去 50 的中心化值。正 = 多头主导,负 = 空头主导;接近 ±50 = 极端超买/超卖。",
)
class RSICenteredFactor(Factor):
    def __init__(self, n: int = 14):
        if n <= 0:
            raise ValueError(f"rsi window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"rsi_centered_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        diff = close.diff()
        gain = diff.clip(lower=0.0)
        loss = (-diff).clip(lower=0.0)
        # SMMA / Wilder's smoothing 等价于 alpha=1/n 的 EWM
        avg_g = gain.ewm(alpha=1.0 / self.n, adjust=False).mean()
        avg_l = loss.ewm(alpha=1.0 / self.n, adjust=False).mean()
        # loss=0 时 rs=inf → rsi=100 (持续上涨), 不要置 NaN
        rs = avg_g / avg_l.replace(0.0, np.inf)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        # warmup: 第一行 diff 是 NaN,顺势把 RSI 也置 NaN
        rsi.iloc[0] = np.nan
        return rsi - 50.0


@register(
    "kdj_j",
    sources=("builtin",),
    types=("reversal", "time_series"),
    description="KDJ 的 J 线减去 50。比 KDJ 本身更敏感,J 远离 50 常预警短线极值反转。",
)
class KDJJFactor(Factor):
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        self.n = n
        self.m1 = m1
        self.m2 = m2

    @property
    def name(self) -> str:
        return "kdj_j"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        high_n = panel["high"].rolling(self.n).max()
        low_n = panel["low"].rolling(self.n).min()
        rng = (high_n - low_n).replace(0.0, np.nan)
        rsv = (panel["close"] - low_n) / rng * 100.0
        rsv = rsv.fillna(50.0)
        k = rsv.ewm(alpha=1.0 / self.m1, adjust=False).mean()
        d = k.ewm(alpha=1.0 / self.m2, adjust=False).mean()
        j = 3.0 * k - 2.0 * d
        # warmup 前 n-1 行置 NaN
        if self.n > 1:
            j.iloc[: self.n - 1] = np.nan
        return j - 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Trend (距均线) 系
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "ma_distance",
    sources=("builtin",),
    types=("trend", "time_series"),
    description="收盘价距离 N 日均线的相对距离。正且大 = 显著高于均线可能透支;负且大 = 跌破均线深可能见底。",
)
class MADistanceFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"ma window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"ma_distance_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ma = close.rolling(self.n).mean()
        return (close - ma) / ma


@register(
    "ma_slope",
    sources=("builtin",),
    types=("trend", "time_series"),
    description="N 日均线的近期斜率(相对值)。正陡 = 均线明显走高(强趋势);0 附近 = 走平(整理期)。",
)
class MASlopeFactor(Factor):
    def __init__(self, n: int = 20, k: int = 5):
        if n <= 0 or k <= 0:
            raise ValueError(f"n and k must be > 0, got n={n}, k={k}")
        self.n = n
        self.k = k

    @property
    def name(self) -> str:
        return f"ma_slope_{self.n}_{self.k}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ma = panel["close"].rolling(self.n).mean()
        prev = ma.shift(self.k)
        return (ma - prev) / prev


# ─────────────────────────────────────────────────────────────────────────────
# Volume / Volatility
# ─────────────────────────────────────────────────────────────────────────────

@register(
    "vol_ratio",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="今日成交量相对近 N 日均量的偏离比,大于 0 放量、小于 0 缩量,衡量交易热度。",
)
class VolumeRatioFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"vol window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vol_ratio_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"]
        ma_v = v.rolling(self.n).mean().shift(1)
        return v / ma_v - 1.0


@register(
    "boll_position",
    sources=("builtin",),
    types=("reversal", "volatility", "time_series"),
    description="收盘价在布林轨道中的相对位置 ∈ [-1, +1]。+1 触上轨可能超买;-1 触下轨可能超卖。",
)
class BollPositionFactor(Factor):
    def __init__(self, n: int = 20, k: float = 2.0):
        self.n = n
        self.k = k

    @property
    def name(self) -> str:
        return f"boll_position_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        mid = close.rolling(self.n).mean()
        std = close.rolling(self.n).std(ddof=0)
        width = self.k * std
        width = width.replace(0.0, np.nan)
        return (close - mid) / width


@register(
    "hl_range",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="近 N 日日均振幅相对收盘价的比值,简单波动率代理。",
)
class HLRangeFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"hl window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"hl_range_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        high = panel["high"].rolling(self.n).max()
        low = panel["low"].rolling(self.n).min()
        return (high - low) / panel["close"]
