"""Pure indicator functions: DataFrame in → DataFrame out (with added columns).

Each function NEVER mutates input — always returns a copy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_ma(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Simple moving averages on close."""
    out = df.copy()
    for p in periods:
        out[f"ma{p}"] = out["close"].rolling(p).mean()
    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD: DIF = EMA_fast - EMA_slow; DEA = EMA(DIF, signal); HIST = 2*(DIF-DEA)."""
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd_dif"] = ema_fast - ema_slow
    out["macd_dea"] = out["macd_dif"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = 2 * (out["macd_dif"] - out["macd_dea"])
    return out


def add_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ (China-market convention): RSV → SMA → K/D/J."""
    out = df.copy()
    low_n = out["low"].rolling(n).min()
    high_n = out["high"].rolling(n).max()
    rsv = (out["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d

    k.iloc[: n - 1] = np.nan
    d.iloc[: n - 1] = np.nan
    j.iloc[: n - 1] = np.nan

    out["kdj_k"] = k
    out["kdj_d"] = d
    out["kdj_j"] = j
    return out


def add_rsi(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Wilder's RSI: 100 - 100/(1 + RS), RS = avg_gain / avg_loss (SMMA/EWMA)."""
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for p in periods:
        avg_gain = gain.ewm(alpha=1 / p, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / p, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        # P3-14: avg_loss==0(窗口内纯涨)语义上 RSI=100,不是中性 50;
        # avg_gain 也为 0(完全无变动)才填 50。
        pure_gain = (avg_loss == 0) & (avg_gain > 0)
        rsi = rsi.where(~pure_gain, 100.0).fillna(50)
        rsi.iloc[:p] = np.nan
        out[f"rsi{p}"] = rsi
    return out


def add_boll(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: mid = MA(n), up/low = mid ± k × stddev."""
    out = df.copy()
    mid = out["close"].rolling(n).mean()
    std = out["close"].rolling(n).std(ddof=0)
    out["boll_mid"] = mid
    out["boll_up"] = mid + k * std
    out["boll_low"] = mid - k * std
    return out


def add_volume_ratio(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """vol_ratio_N = volume / MA_N(volume).shift(1) — today's volume vs N-day avg."""
    out = df.copy()
    avg = out["volume"].rolling(window).mean().shift(1)
    out[f"vol_ratio{window}"] = out["volume"] / avg
    return out


def add_breakout_markers(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Close == past N-day max → new high, vice versa for new low."""
    out = df.copy()
    rolling_high = out["close"].rolling(window).max()
    rolling_low = out["close"].rolling(window).min()
    out["is_breakout_high"] = out["close"] >= rolling_high
    out["is_breakout_low"] = out["close"] <= rolling_low
    if len(out) >= window - 1:
        out.loc[: window - 2, "is_breakout_high"] = False
        out.loc[: window - 2, "is_breakout_low"] = False
    return out


def add_all(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """One-stop: apply every indicator according to IndicatorsConfig."""
    out = df
    out = add_ma(out, cfg.ma_periods)
    out = add_macd(out, cfg.macd.fast, cfg.macd.slow, cfg.macd.signal)
    out = add_kdj(out, cfg.kdj.n, cfg.kdj.m1, cfg.kdj.m2)
    out = add_rsi(out, cfg.rsi_periods)
    out = add_boll(out, cfg.boll.n, cfg.boll.k)
    out = add_volume_ratio(out, cfg.volume_ratio_window)
    out = add_breakout_markers(out, cfg.breakout_window)
    return out
