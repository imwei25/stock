"""WorldQuant 101 风格算子库,作用在 T × N 宽表上。

所有函数 ``f(x: DataFrame, ...) -> DataFrame``,保持 ``x`` 的 index / columns。
NaN 安全:窗口期不足或除零返回 NaN。

约定:
  * ``ts_*`` 系列是时间序列算子(沿 axis=0)
  * ``rank`` / ``scale`` / ``signedpower`` 是横截面算子(沿 axis=1)
  * ``indneutralize`` 按分组在横截面内 demean

WQ101 论文里 ``rank(x)`` 默认就是横截面 (每天对所有股票打分),与本库一致。

实现拆分:7 个 hot op(correlation / ts_rank / decay_linear / ts_std /
ts_argmax / ts_argmin / rank / indneutralize)的 pandas oracle 在
``_ops_py.py`` —— 后续 PR 会接 Rust 加速,本模块的公开 API 不变。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ._ops_py import _min_periods  # re-used by light rolling ops below
from ._ops_py import (
    correlation,
    decay_linear,
    indneutralize,
    rank,
    ts_argmax,
    ts_argmin,
    ts_rank,
    ts_std,
)

__all__ = [
    # hot ops (delegated to _ops_py)
    "correlation",
    "decay_linear",
    "indneutralize",
    "rank",
    "ts_argmax",
    "ts_argmin",
    "ts_rank",
    "ts_std",
    # light ops (defined inline below)
    "adv",
    "covariance",
    "cs_demean",
    "delay",
    "delta",
    "returns",
    "safe_div",
    "scale",
    "signedpower",
    "stddev",
    "ts_max",
    "ts_mean",
    "ts_min",
    "ts_product",
    "ts_sum",
    "vwap",
]


# ─────────────────────────────────────────────────────────────────────────────
# 时间序列算子(light)
# ─────────────────────────────────────────────────────────────────────────────

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """t-d 的值。等价 ``x.shift(d)``。"""
    return x.shift(d)


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """x[t] - x[t-d]。"""
    return x - x.shift(d)


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).sum()


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).mean()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).max()


def ts_product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).apply(
        lambda s: float(np.nanprod(s)) if np.isfinite(np.nanprod(s)) else np.nan,
        raw=True,
    )


def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).cov(y)


def stddev(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return ts_std(x, d)


# ─────────────────────────────────────────────────────────────────────────────
# 横截面算子(light)
# ─────────────────────────────────────────────────────────────────────────────

def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """横截面 L1 归一化: 每行 / (|x|.sum() / a)。"""
    denom = x.abs().sum(axis=1).replace(0.0, np.nan)
    return x.div(denom, axis=0) * a


def signedpower(x: pd.DataFrame, a: float) -> pd.DataFrame:
    """sign(x) * |x|^a 。"""
    return np.sign(x) * x.abs().pow(a)


def cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    """横截面 demean: 每行减去当天均值。"""
    return x.sub(x.mean(axis=1), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """分母为 0 / NaN 时返回 NaN。"""
    d = den.where(den != 0)
    return num / d


def vwap(panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """近似 vwap = (high + low + close) / 3。WQ101 论文里没有真实分钟成交,
    这是行业惯用的 daily proxy。"""
    return (panel["high"] + panel["low"] + panel["close"]) / 3.0


def returns(close: pd.DataFrame) -> pd.DataFrame:
    """简单日收益。"""
    return close.pct_change(fill_method=None)


def adv(volume: pd.DataFrame, d: int) -> pd.DataFrame:
    """平均日成交量 (Average Daily Volume) over d days。WQ101 的 ``adv{d}``。"""
    return volume.rolling(d, min_periods=d).mean()
