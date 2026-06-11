"""WorldQuant 101 风格算子库,作用在 T × N 宽表上。

所有函数 ``f(x: DataFrame, ...) -> DataFrame``,保持 ``x`` 的 index / columns。
NaN 安全:窗口期不足或除零返回 NaN。

约定:
  * ``ts_*`` 系列是时间序列算子(沿 axis=0)
  * ``rank`` / ``scale`` / ``signedpower`` 是横截面算子(沿 axis=1)
  * ``indneutralize`` 按分组在横截面内 demean

WQ101 论文里 ``rank(x)`` 默认就是横截面 (每天对所有股票打分),与本库一致。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 时间序列算子
# ─────────────────────────────────────────────────────────────────────────────

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """t-d 的值。等价 ``x.shift(d)``。"""
    return x.shift(d)


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """x[t] - x[t-d]。"""
    return x - x.shift(d)


def _min_periods(d: int) -> int:
    """放宽 min_periods 到 60% 窗口长度,使 mask=False 引入的 NaN 不会
    整段杀掉因子值。``max(1, ...)`` 防 d<2 时退化。"""
    return max(1, int(d * 0.6))


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """滚动求和。P3-5:部分窗口(有效值 < d)按 ``d / count`` 重标定 ——
    否则历史短/停牌多的股票部分和系统性偏小,进 ``rank()`` 后截面秩被
    结构性压低(alpha_019/039/052 的 ts_sum(ret, 250) 受害最深)。
    mean × d 数学上等价于"重标定后的 sum"。"""
    return x.rolling(d, min_periods=_min_periods(d)).mean() * d


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).mean()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).max()


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).std(ddof=0)


def ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """过去 d 期内最大值出现的位置(0=今天,d-1=最远)。

    ⚠ 方向约定与主流公开复现(yli188 等,返回"窗口内位置",越大越近)
    **相反**(P3-6)。原文 "which day ts_max occurred on" 本就含糊;本项目
    的 IC 加权会自动纠符号,对 ML 路径无实际影响,但与文献对比 alpha_001/
    057/060/096/098/100 的 IC 时符号会反,比较时注意。"""
    def _arg(s: pd.Series) -> float:
        a = s.values
        if np.isnan(a).any():
            return np.nan
        return float(len(a) - 1 - int(np.argmax(a)))
    return x.rolling(d, min_periods=d).apply(_arg, raw=False)


def ts_argmin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    def _arg(s: pd.Series) -> float:
        a = s.values
        if np.isnan(a).any():
            return np.nan
        return float(len(a) - 1 - int(np.argmin(a)))
    return x.rolling(d, min_periods=d).apply(_arg, raw=False)


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """过去 d 期内的时间序列分位排名 ∈ [0, 1]。"""
    def _rank(s: pd.Series) -> float:
        a = s.values
        if np.isnan(a).any():
            return np.nan
        # 当前值在过去 d 个值里的 rank(取最后一个元素的位置)
        last = a[-1]
        return float((a <= last).sum()) / float(len(a))
    return x.rolling(d, min_periods=d).apply(_rank, raw=False)


def ts_product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """滚动连乘。P3-5:要求满窗(min_periods=d)—— nanprod 把 NaN 当 1,
    部分窗口的"部分积"无法像 sum 那样线性重标定,缺值直接 NaN 更诚实。"""
    return x.rolling(d, min_periods=d).apply(
        lambda s: np.nan if np.isnan(s).any() else float(np.prod(s)),
        raw=True,
    )


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """加权移动平均,权重 1, 2, ..., d 归一化。WQ101 ``decay_linear``。

    NaN-safe:窗口内 NaN 位置同步从分子/分母剔除,余下权重重归一化。
    全 NaN 窗口返回 NaN。
    """
    weights = np.arange(1, d + 1, dtype=float)

    def _wmean(a: np.ndarray) -> float:
        # Rolling may pass arrays shorter than d when min_periods < d;
        # align weights to the tail of the full weight vector.
        w_slice = weights[-len(a):]
        valid = ~np.isnan(a)
        if not valid.any():
            return np.nan
        w = w_slice[valid]
        v = a[valid]
        return float(np.dot(v, w) / w.sum())

    # raw=True: 直接收 ndarray,绕过 pandas 在大宽表上构造 Series 时触发的
    # closure-cell 路径 bug (TypeError: 'cell' object is not callable)。
    return x.rolling(d, min_periods=_min_periods(d)).apply(_wmean, raw=True)


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """每列分别滚动 d 期相关系数。"""
    return x.rolling(d, min_periods=d).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).cov(y)


def stddev(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return ts_std(x, d)


# ─────────────────────────────────────────────────────────────────────────────
# 横截面算子(每个 t 沿 axis=1)
# ─────────────────────────────────────────────────────────────────────────────

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """横截面排名,归一化到 [0, 1]。WQ101 默认 ``rank``。"""
    return x.rank(axis=1, pct=True, method="average")


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


def indneutralize(
    x: pd.DataFrame,
    group: Mapping[str, str] | pd.Series,
) -> pd.DataFrame:  # noqa: D401
    """按行业分组,在每天的横截面内 demean。

    Args:
        x: T × N 因子值宽表。
        group: ``code -> sector_name`` 映射(dict 或 Series)。**未出现的
               code 输出 NaN**(P3-7)——旧实现给独立组,x − 自身均值恰好
               = 0,以"完美中性"的假值参与后续 rank,静默且危险;与
               ``custom.py`` 的同情形置 NaN 语义统一。

    NaN 安全:组内不参与计算。
    """
    g = pd.Series(group) if not isinstance(group, pd.Series) else group
    mapped = [c in g.index for c in x.columns] if isinstance(group, pd.Series) \
        else [c in group for c in x.columns]
    sectors = [g.get(c, "__unmapped__") for c in x.columns]
    sec_series = pd.Series(sectors, index=x.columns, name="sector")
    # group-mean: 对每行,按列分组求均值,再 broadcast 回去
    means = x.T.groupby(sec_series).transform("mean").T
    out = x - means
    unmapped_cols = [c for c, ok in zip(x.columns, mapped) if not ok]
    if unmapped_cols:
        out[unmapped_cols] = float("nan")
    return out


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
