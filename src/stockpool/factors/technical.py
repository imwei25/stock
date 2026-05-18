"""Built-in technical factors.

These are continuous reformulations of the discrete triggers in
``stockpool.signals``: momentum, MACD histogram, RSI deviation,
distance from MA, volume ratio, Bollinger position, breakout proximity.

Each factor's ``compute`` reuses ``stockpool.indicators`` so the underlying
formulas stay in one place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register
from stockpool.indicators import (
    add_boll,
    add_kdj,
    add_ma,
    add_macd,
    add_rsi,
    add_volume_ratio,
)


@register("momentum")
class MomentumFactor(Factor):
    """N-day momentum: close[t] / close[t-n] - 1."""

    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"momentum window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"momentum_{self.n}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(self.n)


@register("macd_hist")
class MACDHistFactor(Factor):
    """Raw MACD histogram value (2 × (DIF - DEA))."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return "macd_hist"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return add_macd(df, self.fast, self.slow, self.signal)["macd_hist"]


@register("macd_dif_norm")
class MACDDifNormFactor(Factor):
    """MACD DIF normalised by close price — comparable across stocks."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return "macd_dif_norm"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        out = add_macd(df, self.fast, self.slow, self.signal)
        return out["macd_dif"] / out["close"]


@register("rsi_centered")
class RSICenteredFactor(Factor):
    """RSI - 50: positive = bullish, negative = bearish."""

    def __init__(self, n: int = 14):
        if n <= 0:
            raise ValueError(f"rsi window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"rsi_centered_{self.n}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return add_rsi(df, [self.n])[f"rsi{self.n}"] - 50.0


@register("ma_distance")
class MADistanceFactor(Factor):
    """(close - MA_n) / MA_n — relative distance to N-day MA."""

    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"ma window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"ma_distance_{self.n}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        out = add_ma(df, [self.n])
        ma = out[f"ma{self.n}"]
        return (out["close"] - ma) / ma


@register("ma_slope")
class MASlopeFactor(Factor):
    """K-bar slope of MA_n, normalised by current MA — trend strength."""

    def __init__(self, n: int = 20, k: int = 5):
        if n <= 0 or k <= 0:
            raise ValueError(f"n and k must be > 0, got n={n}, k={k}")
        self.n = n
        self.k = k

    @property
    def name(self) -> str:
        return f"ma_slope_{self.n}_{self.k}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = add_ma(df, [self.n])[f"ma{self.n}"]
        return (ma - ma.shift(self.k)) / ma.shift(self.k)


@register("vol_ratio")
class VolumeRatioFactor(Factor):
    """volume / MA_n(volume).shift(1) - 1, centered at 0."""

    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"vol window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vol_ratio_{self.n}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return add_volume_ratio(df, self.n)[f"vol_ratio{self.n}"] - 1.0


@register("boll_position")
class BollPositionFactor(Factor):
    """Bollinger position in [-1, +1]: (close - mid) / (up - mid)."""

    def __init__(self, n: int = 20, k: float = 2.0):
        self.n = n
        self.k = k

    @property
    def name(self) -> str:
        return f"boll_position_{self.n}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        b = add_boll(df, self.n, self.k)
        width = b["boll_up"] - b["boll_mid"]
        # Where width is 0 (flat market), factor is undefined → NaN.
        width = width.replace(0.0, np.nan)
        return (b["close"] - b["boll_mid"]) / width


@register("kdj_j")
class KDJJFactor(Factor):
    """KDJ J-line centered: J - 50."""

    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        self.n = n
        self.m1 = m1
        self.m2 = m2

    @property
    def name(self) -> str:
        return "kdj_j"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return add_kdj(df, self.n, self.m1, self.m2)["kdj_j"] - 50.0


@register("hl_range")
class HLRangeFactor(Factor):
    """N-bar high-low range / close — volatility proxy."""

    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"hl window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"hl_range_{self.n}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"].rolling(self.n).max()
        low = df["low"].rolling(self.n).min()
        return (high - low) / df["close"]
