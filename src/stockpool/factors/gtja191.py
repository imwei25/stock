"""国泰君安 191 Alpha 因子(GTJA191)—— A 股本土化短周期量价因子族(验证子集).

来源:国泰君安《基于短周期价量特征的多因子选股体系——数量化专题之九十三》(2017)。
这是 WorldQuant Alpha101 的 A 股对应物,公式与窗口为 A 股市场设计。

本模块是**经人工核对公式的验证子集**(非全 191):只收录能用本项目现有算子 +
``ops.sma`` 忠实移植的 alpha,刻意跳过依赖 WMA / REGBETA / REGRESI / SEQUENCE /
SMEAN 等尚未实现或语义有歧义算子的公式,避免隐错污染 IC 结论。后续可在补齐算子并
逐一校验后扩展。

命名:``gtja_NNN``(3 位补零,NNN = 原研报编号),``sources=("gtja191",)``。
RANK = 截面百分位(``ops.rank``);VWAP = ``(H+L+C)/3`` proxy(``ops.vwap``);
RET = 日收益(``ops.returns``);MEAN/SUM/STD/TSMAX/TSMIN/TSRANK/CORR/DELAY/DELTA/
DECAYLINEAR 一一对应 ``ops`` 同名算子。除零(如涨跌停 H==L)用 ``ops.safe_div`` 置 NaN。
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.registry import register

_safe = ops.safe_div


class GtjaAlpha(Factor):
    """GTJA191 alpha 基类。子类设 ``NUM`` 即可。"""
    NUM: ClassVar[int] = 0
    sources = ("gtja191",)

    @property
    def name(self) -> str:
        return f"gtja_{self.NUM:03d}"

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "Factor":
        return cls()


def _gtja(num: int, types: tuple[str, ...], description: str):
    name = f"gtja_{num:03d}"

    def _wrap(cls):
        cls.NUM = num
        return register(name, sources=("gtja191",), types=types,
                        description=description)(cls)
    return _wrap


@_gtja(1, ("cross_sectional", "volume"),
       "成交量对数变化秩与日内涨幅秩的 6 日相关性取负:量价背离反转。")
class Gtja001(GtjaAlpha):
    def compute(self, panel):
        vol, close, open_ = panel["volume"], panel["close"], panel["open"]
        a = ops.rank(ops.delta(np.log(vol), 1))
        b = ops.rank((close - open_) / open_)
        return -1.0 * ops.correlation(a, b, 6)


@_gtja(2, ("time_series", "reversal"),
       "收盘在当日高低区间内相对位置的一阶差分取负:日内位置反转。")
class Gtja002(GtjaAlpha):
    def compute(self, panel):
        high, low, close = panel["high"], panel["low"], panel["close"]
        pos = _safe((close - low) - (high - close), high - low)
        return -1.0 * ops.delta(pos, 1)


@_gtja(5, ("cross_sectional", "volume"),
       "量秩与高价秩 5 日时序相关性的 3 日滚动最大值取负。")
class Gtja005(GtjaAlpha):
    def compute(self, panel):
        vol, high = panel["volume"], panel["high"]
        c = ops.correlation(ops.ts_rank(vol, 5), ops.ts_rank(high, 5), 5)
        return -1.0 * ops.ts_max(c, 3)


@_gtja(6, ("cross_sectional", "momentum"),
       "开高加权价 4 日变化方向的截面秩取负。")
class Gtja006(GtjaAlpha):
    def compute(self, panel):
        open_, high = panel["open"], panel["high"]
        return ops.rank(np.sign(ops.delta(open_ * 0.85 + high * 0.15, 4))) * -1.0


@_gtja(7, ("cross_sectional", "volume"),
       "VWAP 与收盘差的 3 日极值秩之和,乘以成交量变化秩。")
class Gtja007(GtjaAlpha):
    def compute(self, panel):
        vw, close, vol = ops.vwap(panel), panel["close"], panel["volume"]
        d = vw - close
        return (ops.rank(ops.ts_max(d, 3)) + ops.rank(ops.ts_min(d, 3))) \
            * ops.rank(ops.delta(vol, 3))


@_gtja(8, ("cross_sectional", "reversal"),
       "中价与 VWAP 加权值 4 日变化取负的截面秩。")
class Gtja008(GtjaAlpha):
    def compute(self, panel):
        high, low, vw = panel["high"], panel["low"], ops.vwap(panel)
        return ops.rank(ops.delta((high + low) / 2 * 0.2 + vw * 0.8, 4) * -1.0)


@_gtja(9, ("time_series", "volume"),
       "中价动量×振幅/成交量的 SMA(7,2) 平滑:量能加权趋势。")
class Gtja009(GtjaAlpha):
    def compute(self, panel):
        high, low, vol = panel["high"], panel["low"], panel["volume"]
        mid_mom = (high + low) / 2 - (ops.delay(high, 1) + ops.delay(low, 1)) / 2
        x = _safe(mid_mom * (high - low), vol)
        return ops.sma(x, 7, 2)


@_gtja(11, ("time_series", "volume"),
       "收盘在高低区间相对位置×成交量的 6 日累加:量能确认的位置强度。")
class Gtja011(GtjaAlpha):
    def compute(self, panel):
        high, low, close, vol = panel["high"], panel["low"], panel["close"], panel["volume"]
        x = _safe((close - low) - (high - close), high - low) * vol
        return ops.ts_sum(x, 6)


@_gtja(12, ("cross_sectional", "reversal"),
       "开盘相对 10 日 VWAP 均值的秩,乘以收盘偏离 VWAP 绝对值秩取负。")
class Gtja012(GtjaAlpha):
    def compute(self, panel):
        open_, close, vw = panel["open"], panel["close"], ops.vwap(panel)
        return ops.rank(open_ - ops.ts_sum(vw, 10) / 10) \
            * (-1.0 * ops.rank((close - vw).abs()))


@_gtja(13, ("time_series",),
       "几何中价(√(高×低))与 VWAP 之差:日内价格结构。")
class Gtja013(GtjaAlpha):
    def compute(self, panel):
        high, low, vw = panel["high"], panel["low"], ops.vwap(panel)
        return (high * low) ** 0.5 - vw


@_gtja(14, ("time_series", "momentum"),
       "收盘相对 5 日前的绝对变化:中短期动量。")
class Gtja014(GtjaAlpha):
    def compute(self, panel):
        close = panel["close"]
        return close - ops.delay(close, 5)


@_gtja(15, ("time_series", "reversal"),
       "开盘相对昨收的跳空收益。")
class Gtja015(GtjaAlpha):
    def compute(self, panel):
        open_, close = panel["open"], panel["close"]
        return _safe(open_, ops.delay(close, 1)) - 1.0


@_gtja(16, ("cross_sectional", "volume"),
       "量秩与 VWAP 秩 5 日相关性的截面秩的 5 日滚动最大值取负。")
class Gtja016(GtjaAlpha):
    def compute(self, panel):
        vol, vw = panel["volume"], ops.vwap(panel)
        c = ops.rank(ops.correlation(ops.rank(vol), ops.rank(vw), 5))
        return -1.0 * ops.ts_max(c, 5)


@_gtja(18, ("time_series", "momentum"),
       "收盘相对 5 日前的比值:中期动量。")
class Gtja018(GtjaAlpha):
    def compute(self, panel):
        close = panel["close"]
        return _safe(close, ops.delay(close, 5))


@_gtja(20, ("time_series", "momentum"),
       "收盘 6 日变化率(百分比)。")
class Gtja020(GtjaAlpha):
    def compute(self, panel):
        close = panel["close"]
        return _safe(close - ops.delay(close, 6), ops.delay(close, 6)) * 100.0


@_gtja(24, ("time_series", "momentum"),
       "收盘 5 日变化的 SMA(5,1) 平滑动量。")
class Gtja024(GtjaAlpha):
    def compute(self, panel):
        close = panel["close"]
        return ops.sma(close - ops.delay(close, 5), 5, 1)


@_gtja(25, ("cross_sectional", "volume"),
       "收盘 7 日变化×(1-量比衰减秩)的秩取负,再乘 250 日收益秩调制。")
class Gtja025(GtjaAlpha):
    def compute(self, panel):
        close, vol = panel["close"], panel["volume"]
        r = ops.returns(close)
        volratio = _safe(vol, ops.ts_mean(vol, 20))
        inner = ops.delta(close, 7) * (1 - ops.rank(ops.decay_linear(volratio, 9)))
        return (-1.0 * ops.rank(inner)) * (1 + ops.rank(ops.ts_sum(r, 250)))


@_gtja(26, ("time_series",),
       "7 日均价相对现价之差,加 VWAP 与 5 日前收盘的 230 日相关性。")
class Gtja026(GtjaAlpha):
    def compute(self, panel):
        close, vw = panel["close"], ops.vwap(panel)
        return (ops.ts_sum(close, 7) / 7 - close) \
            + ops.correlation(vw, ops.delay(close, 5), 230)


@_gtja(29, ("time_series", "volume"),
       "收盘 6 日变化率×成交量:量能加权动量。")
class Gtja029(GtjaAlpha):
    def compute(self, panel):
        close, vol = panel["close"], panel["volume"]
        return _safe(close - ops.delay(close, 6), ops.delay(close, 6)) * vol


@_gtja(31, ("time_series", "reversal"),
       "收盘相对 12 日均线的偏离率(百分比)。")
class Gtja031(GtjaAlpha):
    def compute(self, panel):
        close = panel["close"]
        m = ops.ts_mean(close, 12)
        return _safe(close - m, m) * 100.0


@_gtja(32, ("cross_sectional", "volume"),
       "高价秩与量秩 3 日相关性的截面秩 3 日累加取负。")
class Gtja032(GtjaAlpha):
    def compute(self, panel):
        high, vol = panel["high"], panel["volume"]
        c = ops.rank(ops.correlation(ops.rank(high), ops.rank(vol), 3))
        return -1.0 * ops.ts_sum(c, 3)


@_gtja(33, ("cross_sectional", "reversal"),
       "5 日最低价反转×长短期收益差秩×成交量时序秩的复合反转因子。")
class Gtja033(GtjaAlpha):
    def compute(self, panel):
        low, vol, close = panel["low"], panel["volume"], panel["close"]
        r = ops.returns(close)
        tsmin5 = ops.ts_min(low, 5)
        term = (-1.0 * tsmin5 + ops.delay(tsmin5, 5)) \
            * ops.rank((ops.ts_sum(r, 240) - ops.ts_sum(r, 20)) / 220)
        return term * ops.ts_rank(vol, 5)


@_gtja(34, ("time_series", "reversal"),
       "12 日均线相对现价的比值:均值回归。")
class Gtja034(GtjaAlpha):
    def compute(self, panel):
        close = panel["close"]
        return _safe(ops.ts_mean(close, 12), close)


@_gtja(37, ("cross_sectional", "momentum"),
       "5 日开盘和×5 日收益和的 10 日变化的截面秩取负。")
class Gtja037(GtjaAlpha):
    def compute(self, panel):
        open_, close = panel["open"], panel["close"]
        r = ops.returns(close)
        prod = ops.ts_sum(open_, 5) * ops.ts_sum(r, 5)
        return -1.0 * ops.rank(prod - ops.delay(prod, 10))


@_gtja(40, ("time_series", "volume"),
       "26 日上涨日成交量和 / 下跌日成交量和(百分比):量能多空比。")
class Gtja040(GtjaAlpha):
    def compute(self, panel):
        close, vol = panel["close"], panel["volume"]
        prev = ops.delay(close, 1)
        up = vol.where(close > prev, 0.0)
        down = vol.where(close <= prev, 0.0)
        return _safe(ops.ts_sum(up, 26), ops.ts_sum(down, 26)) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 第二批(alpha 41-90 子集,2026-06-24)。仅收录逐字核对、用现有算子 + ops.sma /
# ops.count 可忠实移植的公式;跳过依赖 DTM/DBM、benchmark、残缺公式(62/73/74)、
# STD 缺窗口(54)、巨型嵌套条件(55)、WMA/REGBETA(留待后续批)。
# ─────────────────────────────────────────────────────────────────────────────

def _typ(panel):
    """典型价 (H+L+C)/3。"""
    return (panel["high"] + panel["low"] + panel["close"]) / 3.0


def _signed_vol(close, vol):
    """close>昨? +vol; close<昨? -vol; 相等? 0。输入 NaN 处保持 NaN。"""
    prev = ops.delay(close, 1)
    sign = np.where(close > prev, 1.0, np.where(close < prev, -1.0, 0.0))
    s = pd.DataFrame(sign, index=close.index, columns=close.columns)
    s = s.where(close.notna() & prev.notna())
    return s * vol


@_gtja(41, ("cross_sectional", "reversal"),
       "VWAP 3 日变化的 5 日滚动最大值的截面秩取负。")
class Gtja041(GtjaAlpha):
    def compute(self, panel):
        return ops.rank(ops.ts_max(ops.delta(ops.vwap(panel), 3), 5)) * -1.0


@_gtja(42, ("cross_sectional", "volume"),
       "高价 10 日波动率秩取负,乘以高价与量 10 日相关性。")
class Gtja042(GtjaAlpha):
    def compute(self, panel):
        high, vol = panel["high"], panel["volume"]
        return (-1.0 * ops.rank(ops.stddev(high, 10))) * ops.correlation(high, vol, 10)


@_gtja(43, ("time_series", "volume"),
       "6 日按涨跌方向的带符号成交量累加:量能净多空。")
class Gtja043(GtjaAlpha):
    def compute(self, panel):
        return ops.ts_sum(_signed_vol(panel["close"], panel["volume"]), 6)


@_gtja(45, ("cross_sectional", "volume"),
       "收开加权价 1 日变化秩,乘以 VWAP 与 150 日均量 15 日相关性秩。")
class Gtja045(GtjaAlpha):
    def compute(self, panel):
        close, open_, vol = panel["close"], panel["open"], panel["volume"]
        a = ops.rank(ops.delta(close * 0.6 + open_ * 0.4, 1))
        b = ops.rank(ops.correlation(ops.vwap(panel), ops.ts_mean(vol, 150), 15))
        return a * b


@_gtja(46, ("time_series", "reversal"),
       "3/6/12/24 日均价之和相对 4 倍现价:多周期均值回归。")
class Gtja046(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        s = ops.ts_mean(c, 3) + ops.ts_mean(c, 6) + ops.ts_mean(c, 12) + ops.ts_mean(c, 24)
        return _safe(s, 4.0 * c)


@_gtja(47, ("time_series", "reversal"),
       "6 日高点相对收盘在高低区间的位置的 SMA(9,1) 平滑(超买超卖)。")
class Gtja047(GtjaAlpha):
    def compute(self, panel):
        high, low, close = panel["high"], panel["low"], panel["close"]
        hh = ops.ts_max(high, 6)
        x = _safe(hh - close, hh - ops.ts_min(low, 6)) * 100.0
        return ops.sma(x, 9, 1)


@_gtja(48, ("cross_sectional", "volume"),
       "近 3 日收盘方向符号和的截面秩,乘以 5 日/20 日量比,取负。")
class Gtja048(GtjaAlpha):
    def compute(self, panel):
        c, vol = panel["close"], panel["volume"]
        s = (np.sign(c - ops.delay(c, 1)) + np.sign(ops.delay(c, 1) - ops.delay(c, 2))
             + np.sign(ops.delay(c, 2) - ops.delay(c, 3)))
        return -1.0 * ops.rank(s) * _safe(ops.ts_sum(vol, 5), ops.ts_sum(vol, 20))


@_gtja(52, ("time_series", "momentum"),
       "26 日上行动量和 / 下行动量和(相对典型价):多空动能比。")
class Gtja052(GtjaAlpha):
    def compute(self, panel):
        high, low = panel["high"], panel["low"]
        dtyp = ops.delay(_typ(panel), 1)
        up = (high - dtyp).clip(lower=0.0)
        dn = (dtyp - low).clip(lower=0.0)
        return _safe(ops.ts_sum(up, 26), ops.ts_sum(dn, 26)) * 100.0


@_gtja(53, ("time_series", "momentum"),
       "12 日内上涨天数占比(百分比)。")
class Gtja053(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return ops.count(c > ops.delay(c, 1), 12) / 12.0 * 100.0


@_gtja(57, ("time_series", "reversal"),
       "9 日 Stochastic %K 的 SMA(3,1) 平滑(KDJ K 值)。")
class Gtja057(GtjaAlpha):
    def compute(self, panel):
        high, low, close = panel["high"], panel["low"], panel["close"]
        ll = ops.ts_min(low, 9)
        x = _safe(close - ll, ops.ts_max(high, 9) - ll) * 100.0
        return ops.sma(x, 3, 1)


@_gtja(58, ("time_series", "momentum"),
       "20 日内上涨天数占比(百分比)。")
class Gtja058(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return ops.count(c > ops.delay(c, 1), 20) / 20.0 * 100.0


@_gtja(59, ("time_series", "momentum"),
       "20 日方向性收盘动量累加(涨用 min(low,昨收)、跌用 max(high,昨收) 为基)。")
class Gtja059(GtjaAlpha):
    def compute(self, panel):
        close, high, low = panel["close"], panel["high"], panel["low"]
        prev = ops.delay(close, 1)
        base = np.minimum(low, prev).where(close > prev, np.maximum(high, prev))
        raw = (close - base).where(close != prev, 0.0)
        raw = raw.where(close.notna() & prev.notna())
        return ops.ts_sum(raw, 20)


@_gtja(60, ("time_series", "volume"),
       "收盘在高低区间位置×成交量的 20 日累加(同 gtja_011 但 20 日)。")
class Gtja060(GtjaAlpha):
    def compute(self, panel):
        high, low, close, vol = panel["high"], panel["low"], panel["close"], panel["volume"]
        x = _safe((close - low) - (high - close), high - low) * vol
        return ops.ts_sum(x, 20)


@_gtja(63, ("time_series", "momentum"),
       "6 日 RSI:上行幅度 SMA / 总幅度 SMA(百分比)。")
class Gtja063(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        chg = c - ops.delay(c, 1)
        return _safe(ops.sma(chg.clip(lower=0.0), 6, 1), ops.sma(chg.abs(), 6, 1)) * 100.0


@_gtja(65, ("time_series", "reversal"),
       "6 日均线相对现价比值:均值回归。")
class Gtja065(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return _safe(ops.ts_mean(c, 6), c)


@_gtja(66, ("time_series", "reversal"),
       "收盘相对 6 日均线偏离率(百分比)。")
class Gtja066(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        m = ops.ts_mean(c, 6)
        return _safe(c - m, m) * 100.0


@_gtja(67, ("time_series", "momentum"),
       "24 日 RSI。")
class Gtja067(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        chg = c - ops.delay(c, 1)
        return _safe(ops.sma(chg.clip(lower=0.0), 24, 1), ops.sma(chg.abs(), 24, 1)) * 100.0


@_gtja(68, ("time_series", "volume"),
       "中价动量×振幅/成交量的 SMA(15,2) 平滑(同 gtja_009 但 15,2)。")
class Gtja068(GtjaAlpha):
    def compute(self, panel):
        high, low, vol = panel["high"], panel["low"], panel["volume"]
        mid_mom = (high + low) / 2 - (ops.delay(high, 1) + ops.delay(low, 1)) / 2
        return ops.sma(_safe(mid_mom * (high - low), vol), 15, 2)


@_gtja(71, ("time_series", "reversal"),
       "收盘相对 24 日均线偏离率(百分比)。")
class Gtja071(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        m = ops.ts_mean(c, 24)
        return _safe(c - m, m) * 100.0


@_gtja(72, ("time_series", "reversal"),
       "6 日高点超买度的 SMA(15,1) 平滑。")
class Gtja072(GtjaAlpha):
    def compute(self, panel):
        high, low, close = panel["high"], panel["low"], panel["close"]
        hh = ops.ts_max(high, 6)
        x = _safe(hh - close, hh - ops.ts_min(low, 6)) * 100.0
        return ops.sma(x, 15, 1)


@_gtja(76, ("time_series", "volume"),
       "单位成交量收益波动 / 其均值的 20 日变异系数:量价稳定性。")
class Gtja076(GtjaAlpha):
    def compute(self, panel):
        close, vol = panel["close"], panel["volume"]
        u = _safe((_safe(close, ops.delay(close, 1)) - 1.0).abs(), vol)
        return _safe(ops.stddev(u, 20), ops.ts_mean(u, 20))


@_gtja(78, ("time_series", "reversal"),
       "典型价相对 12 日均值的 CCI 型偏离。")
class Gtja078(GtjaAlpha):
    def compute(self, panel):
        typ = _typ(panel)
        mt = ops.ts_mean(typ, 12)
        mad = ops.ts_mean((panel["close"] - mt).abs(), 12)
        return _safe(typ - mt, 0.015 * mad)


@_gtja(79, ("time_series", "momentum"),
       "12 日 RSI。")
class Gtja079(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        chg = c - ops.delay(c, 1)
        return _safe(ops.sma(chg.clip(lower=0.0), 12, 1), ops.sma(chg.abs(), 12, 1)) * 100.0


@_gtja(80, ("time_series", "volume"),
       "成交量 5 日变化率(百分比)。")
class Gtja080(GtjaAlpha):
    def compute(self, panel):
        v = panel["volume"]
        return _safe(v - ops.delay(v, 5), ops.delay(v, 5)) * 100.0


@_gtja(81, ("time_series", "volume"),
       "成交量 SMA(21,2) 平滑。")
class Gtja081(GtjaAlpha):
    def compute(self, panel):
        return ops.sma(panel["volume"], 21, 2)


@_gtja(82, ("time_series", "reversal"),
       "6 日高点超买度的 SMA(20,1) 平滑。")
class Gtja082(GtjaAlpha):
    def compute(self, panel):
        high, low, close = panel["high"], panel["low"], panel["close"]
        hh = ops.ts_max(high, 6)
        x = _safe(hh - close, hh - ops.ts_min(low, 6)) * 100.0
        return ops.sma(x, 20, 1)


@_gtja(83, ("cross_sectional", "volume"),
       "高价秩与量秩 5 日协方差的截面秩取负。")
class Gtja083(GtjaAlpha):
    def compute(self, panel):
        high, vol = panel["high"], panel["volume"]
        return -1.0 * ops.rank(ops.covariance(ops.rank(high), ops.rank(vol), 5))


@_gtja(84, ("time_series", "volume"),
       "20 日按涨跌方向的带符号成交量累加(同 gtja_043 但 20 日)。")
class Gtja084(GtjaAlpha):
    def compute(self, panel):
        return ops.ts_sum(_signed_vol(panel["close"], panel["volume"]), 20)


@_gtja(85, ("time_series", "volume"),
       "20 日量比时序秩 × 7 日反向收盘动量时序秩。")
class Gtja085(GtjaAlpha):
    def compute(self, panel):
        close, vol = panel["close"], panel["volume"]
        a = ops.ts_rank(_safe(vol, ops.ts_mean(vol, 20)), 20)
        b = ops.ts_rank(-1.0 * ops.delta(close, 7), 8)
        return a * b


@_gtja(88, ("time_series", "momentum"),
       "收盘 20 日变化率(百分比)。")
class Gtja088(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return _safe(c - ops.delay(c, 20), ops.delay(c, 20)) * 100.0


@_gtja(89, ("time_series", "momentum"),
       "SMA 双指数 MACD 柱状(13/27/10,平滑系数 2)。")
class Gtja089(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        dif = ops.sma(c, 13, 2) - ops.sma(c, 27, 2)
        return 2.0 * (dif - ops.sma(dif, 10, 2))


@_gtja(90, ("cross_sectional", "volume"),
       "VWAP 秩与量秩 5 日相关性的截面秩取负。")
class Gtja090(GtjaAlpha):
    def compute(self, panel):
        vw, vol = ops.vwap(panel), panel["volume"]
        return ops.rank(ops.correlation(ops.rank(vw), ops.rank(vol), 5)) * -1.0


# ─────────────────────────────────────────────────────────────────────────────
# 第三批(alpha 91-191 子集,2026-06-24)。新增算子 ops.highday / lowday / regslope。
# 仍只收录逐字核对、可忠实移植的;跳过 AMOUNT(无成交额字段)外的 SELF/SUMAC/FILTER/
# benchmark/DMI(TR-LD-HD)/残缺公式。AMOUNT 出现处用 close*volume 代理(IC 对常数尺度不变)。
# ─────────────────────────────────────────────────────────────────────────────

def _tr(panel):
    """真实波幅 TR = max(high-low, |昨收-high|, |昨收-low|)。"""
    high, low, close = panel["high"], panel["low"], panel["close"]
    pc = ops.delay(close, 1)
    return np.maximum(np.maximum(high - low, (pc - high).abs()), (pc - low).abs())


@_gtja(91, ("cross_sectional", "volume"),
       "收盘距 5 日高点秩 × 40 日均量与低价 5 日相关性秩,取负。")
class Gtja091(GtjaAlpha):
    def compute(self, panel):
        close, low, vol = panel["close"], panel["low"], panel["volume"]
        return (ops.rank(close - ops.ts_max(close, 5))
                * ops.rank(ops.correlation(ops.ts_mean(vol, 40), low, 5))) * -1.0


@_gtja(93, ("time_series", "momentum"),
       "20 日开盘下行缺口动量累加。")
class Gtja093(GtjaAlpha):
    def compute(self, panel):
        open_, low = panel["open"], panel["low"]
        po = ops.delay(open_, 1)
        val = np.maximum(open_ - low, open_ - po)
        x = val.where(open_ < po, 0.0).where(open_.notna() & po.notna())
        return ops.ts_sum(x, 20)


@_gtja(94, ("time_series", "volume"),
       "30 日按涨跌方向带符号成交量累加。")
class Gtja094(GtjaAlpha):
    def compute(self, panel):
        return ops.ts_sum(_signed_vol(panel["close"], panel["volume"]), 30)


@_gtja(96, ("time_series", "reversal"),
       "9 日 Stochastic %K 的双重 SMA(3,1) 平滑(KDJ D 值)。")
class Gtja096(GtjaAlpha):
    def compute(self, panel):
        high, low, close = panel["high"], panel["low"], panel["close"]
        ll = ops.ts_min(low, 9)
        k = _safe(close - ll, ops.ts_max(high, 9) - ll) * 100.0
        return ops.sma(ops.sma(k, 3, 1), 3, 1)


@_gtja(97, ("time_series", "volume"),
       "10 日成交量波动率。")
class Gtja097(GtjaAlpha):
    def compute(self, panel):
        return ops.stddev(panel["volume"], 10)


@_gtja(99, ("cross_sectional", "volume"),
       "收盘秩与量秩 5 日协方差的截面秩取负。")
class Gtja099(GtjaAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.covariance(
            ops.rank(panel["close"]), ops.rank(panel["volume"]), 5))


@_gtja(100, ("time_series", "volume"),
       "20 日成交量波动率。")
class Gtja100(GtjaAlpha):
    def compute(self, panel):
        return ops.stddev(panel["volume"], 20)


@_gtja(102, ("time_series", "volume"),
       "成交量 6 日 RSI(量能上行强度)。")
class Gtja102(GtjaAlpha):
    def compute(self, panel):
        v = panel["volume"]
        ch = v - ops.delay(v, 1)
        return _safe(ops.sma(ch.clip(lower=0.0), 6, 1), ops.sma(ch.abs(), 6, 1)) * 100.0


@_gtja(103, ("time_series", "reversal"),
       "20 日内最低价出现的近期度(LOWDAY)。")
class Gtja103(GtjaAlpha):
    def compute(self, panel):
        return (20 - ops.lowday(panel["low"], 20)) / 20.0 * 100.0


@_gtja(104, ("cross_sectional", "volume"),
       "高量相关性 5 日变化 × 收盘 20 日波动率秩,取负。")
class Gtja104(GtjaAlpha):
    def compute(self, panel):
        high, vol, close = panel["high"], panel["volume"], panel["close"]
        return -1.0 * (ops.delta(ops.correlation(high, vol, 5), 5)
                       * ops.rank(ops.stddev(close, 20)))


@_gtja(105, ("cross_sectional", "volume"),
       "开盘秩与量秩 10 日相关性取负。")
class Gtja105(GtjaAlpha):
    def compute(self, panel):
        return -1.0 * ops.correlation(
            ops.rank(panel["open"]), ops.rank(panel["volume"]), 10)


@_gtja(106, ("time_series", "momentum"),
       "收盘 20 日绝对变化。")
class Gtja106(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return c - ops.delay(c, 20)


@_gtja(107, ("cross_sectional", "reversal"),
       "开盘相对昨高/昨收/昨低三个跳空秩之积,取负。")
class Gtja107(GtjaAlpha):
    def compute(self, panel):
        open_, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        return ((-1.0 * ops.rank(open_ - ops.delay(high, 1)))
                * ops.rank(open_ - ops.delay(close, 1))
                * ops.rank(open_ - ops.delay(low, 1)))


@_gtja(109, ("time_series",),
       "10 日振幅 SMA 相对其再平滑的比值。")
class Gtja109(GtjaAlpha):
    def compute(self, panel):
        hl = panel["high"] - panel["low"]
        s = ops.sma(hl, 10, 2)
        return _safe(s, ops.sma(s, 10, 2))


@_gtja(110, ("time_series", "momentum"),
       "20 日上行幅度和 / 下行幅度和(相对昨收):多空动能比。")
class Gtja110(GtjaAlpha):
    def compute(self, panel):
        high, low, close = panel["high"], panel["low"], panel["close"]
        pc = ops.delay(close, 1)
        up = (high - pc).clip(lower=0.0)
        dn = (pc - low).clip(lower=0.0)
        return _safe(ops.ts_sum(up, 20), ops.ts_sum(dn, 20)) * 100.0


@_gtja(111, ("time_series", "volume"),
       "量能加权收盘位置的 SMA(11,2) 与 SMA(4,2) 之差。")
class Gtja111(GtjaAlpha):
    def compute(self, panel):
        high, low, close, vol = panel["high"], panel["low"], panel["close"], panel["volume"]
        x = vol * _safe((close - low) - (high - close), high - low)
        return ops.sma(x, 11, 2) - ops.sma(x, 4, 2)


@_gtja(112, ("time_series", "momentum"),
       "12 日上行幅度和与下行幅度和的相对强弱(百分比)。")
class Gtja112(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        ch = c - ops.delay(c, 1)
        up = ops.ts_sum(ch.clip(lower=0.0), 12)
        dn = ops.ts_sum((-ch).clip(lower=0.0), 12)
        return _safe(up - dn, up + dn) * 100.0


@_gtja(116, ("time_series", "momentum"),
       "收盘对时间 20 日回归斜率(趋势强度)。")
class Gtja116(GtjaAlpha):
    def compute(self, panel):
        return ops.regslope(panel["close"], 20)


@_gtja(118, ("time_series",),
       "20 日上影和 / 下影和(百分比):多空影线比。")
class Gtja118(GtjaAlpha):
    def compute(self, panel):
        high, low, open_ = panel["high"], panel["low"], panel["open"]
        return _safe(ops.ts_sum(high - open_, 20), ops.ts_sum(open_ - low, 20)) * 100.0


@_gtja(120, ("cross_sectional",),
       "VWAP 与收盘之差秩 / 其和秩。")
class Gtja120(GtjaAlpha):
    def compute(self, panel):
        vw, close = ops.vwap(panel), panel["close"]
        return _safe(ops.rank(vw - close), ops.rank(vw + close))


@_gtja(126, ("time_series",),
       "典型价 (C+H+L)/3。")
class Gtja126(GtjaAlpha):
    def compute(self, panel):
        return (panel["close"] + panel["high"] + panel["low"]) / 3.0


@_gtja(128, ("time_series", "volume"),
       "14 日资金流量指标 MFI(典型价×量的上下行比)。")
class Gtja128(GtjaAlpha):
    def compute(self, panel):
        typ = _typ(panel)
        m = typ * panel["volume"]
        prev = ops.delay(typ, 1)
        up = ops.ts_sum(m.where(typ > prev, 0.0), 14)
        dn = ops.ts_sum(m.where(typ < prev, 0.0), 14)
        return 100.0 - _safe(100.0, 1.0 + _safe(up, dn))


@_gtja(129, ("time_series", "reversal"),
       "12 日下行幅度累加。")
class Gtja129(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        ch = c - ops.delay(c, 1)
        return ops.ts_sum((-ch).clip(lower=0.0), 12)


@_gtja(133, ("time_series", "reversal"),
       "20 日高点近期度 - 低点近期度(HIGHDAY/LOWDAY 差)。")
class Gtja133(GtjaAlpha):
    def compute(self, panel):
        hd = (20 - ops.highday(panel["high"], 20)) / 20.0 * 100.0
        ld = (20 - ops.lowday(panel["low"], 20)) / 20.0 * 100.0
        return hd - ld


@_gtja(134, ("time_series", "volume"),
       "收盘 12 日变化率 × 成交量。")
class Gtja134(GtjaAlpha):
    def compute(self, panel):
        c, v = panel["close"], panel["volume"]
        return _safe(c - ops.delay(c, 12), ops.delay(c, 12)) * v


@_gtja(135, ("time_series", "momentum"),
       "20 日收益的滞后 SMA(20,1) 平滑。")
class Gtja135(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return ops.sma(ops.delay(_safe(c, ops.delay(c, 20)), 1), 20, 1)


@_gtja(136, ("cross_sectional", "volume"),
       "收益 3 日变化秩 × 开量 10 日相关性,取负。")
class Gtja136(GtjaAlpha):
    def compute(self, panel):
        return (-1.0 * ops.rank(ops.delta(ops.returns(panel["close"]), 3))
                * ops.correlation(panel["open"], panel["volume"], 10))


@_gtja(139, ("cross_sectional", "volume"),
       "开盘与量 10 日相关性取负。")
class Gtja139(GtjaAlpha):
    def compute(self, panel):
        return -1.0 * ops.correlation(panel["open"], panel["volume"], 10)


@_gtja(141, ("cross_sectional", "volume"),
       "高价秩与 15 日均量秩 9 日相关性的截面秩取负。")
class Gtja141(GtjaAlpha):
    def compute(self, panel):
        high, vol = panel["high"], panel["volume"]
        return ops.rank(ops.correlation(
            ops.rank(high), ops.rank(ops.ts_mean(vol, 15)), 9)) * -1.0


@_gtja(142, ("cross_sectional", "volume"),
       "收盘时序秩 × 二阶动量秩 × 量比时序秩,取负。")
class Gtja142(GtjaAlpha):
    def compute(self, panel):
        close, vol = panel["close"], panel["volume"]
        return (-1.0 * ops.rank(ops.ts_rank(close, 10))
                * ops.rank(ops.delta(ops.delta(close, 1), 1))
                * ops.rank(ops.ts_rank(_safe(vol, ops.ts_mean(vol, 20)), 5)))


@_gtja(145, ("time_series", "volume"),
       "量能均线差(9-26)/12 日均量(百分比):量能趋势。")
class Gtja145(GtjaAlpha):
    def compute(self, panel):
        v = panel["volume"]
        return _safe(ops.ts_mean(v, 9) - ops.ts_mean(v, 26), ops.ts_mean(v, 12)) * 100.0


@_gtja(147, ("time_series", "momentum"),
       "12 日均线对时间 12 日回归斜率(趋势)。")
class Gtja147(GtjaAlpha):
    def compute(self, panel):
        return ops.regslope(ops.ts_mean(panel["close"], 12), 12)


@_gtja(150, ("time_series", "volume"),
       "典型价 × 成交量(资金流强度)。")
class Gtja150(GtjaAlpha):
    def compute(self, panel):
        return _typ(panel) * panel["volume"]


@_gtja(151, ("time_series", "momentum"),
       "收盘 20 日变化的 SMA(20,1) 平滑。")
class Gtja151(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return ops.sma(c - ops.delay(c, 20), 20, 1)


@_gtja(153, ("time_series", "reversal"),
       "3/6/12/24 日均价均值(多周期均线)。")
class Gtja153(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return (ops.ts_mean(c, 3) + ops.ts_mean(c, 6)
                + ops.ts_mean(c, 12) + ops.ts_mean(c, 24)) / 4.0


@_gtja(155, ("time_series", "volume"),
       "成交量 SMA-MACD 柱(13/27/10)。")
class Gtja155(GtjaAlpha):
    def compute(self, panel):
        v = panel["volume"]
        dif = ops.sma(v, 13, 2) - ops.sma(v, 27, 2)
        return dif - ops.sma(dif, 10, 2)


@_gtja(158, ("time_series",),
       "高低相对 SMA(15,2) 均线的展开度 / 收盘。")
class Gtja158(GtjaAlpha):
    def compute(self, panel):
        high, low, close = panel["high"], panel["low"], panel["close"]
        ma = ops.sma(close, 15, 2)
        return _safe((high - ma) - (low - ma), close)


@_gtja(161, ("time_series",),
       "12 日平均真实波幅(ATR)。")
class Gtja161(GtjaAlpha):
    def compute(self, panel):
        return ops.ts_mean(_tr(panel), 12)


@_gtja(167, ("time_series", "momentum"),
       "12 日上行幅度累加。")
class Gtja167(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return ops.ts_sum((c - ops.delay(c, 1)).clip(lower=0.0), 12)


@_gtja(168, ("time_series", "volume"),
       "当日量相对 20 日均量比,取负(放量反向)。")
class Gtja168(GtjaAlpha):
    def compute(self, panel):
        v = panel["volume"]
        return -1.0 * _safe(v, ops.ts_mean(v, 20))


@_gtja(171, ("time_series",),
       "下影×开盘^5 / 上影×收盘^5,取负:日内多空结构。")
class Gtja171(GtjaAlpha):
    def compute(self, panel):
        open_, high, low, close = panel["open"], panel["high"], panel["low"], panel["close"]
        num = -1.0 * ((low - close) * (open_ ** 5))
        return _safe(num, (close - high) * (close ** 5))


@_gtja(173, ("time_series", "momentum"),
       "三重 SMA 平滑趋势合成(TRIX 型)。")
class Gtja173(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        s1 = ops.sma(c, 13, 2)
        return (3.0 * s1 - 2.0 * ops.sma(s1, 13, 2)
                + ops.sma(ops.sma(ops.sma(np.log(c), 13, 2), 13, 2), 13, 2))


@_gtja(175, ("time_series",),
       "6 日平均真实波幅。")
class Gtja175(GtjaAlpha):
    def compute(self, panel):
        return ops.ts_mean(_tr(panel), 6)


@_gtja(176, ("cross_sectional", "volume"),
       "12 日 Stochastic 位置秩与量秩 6 日相关性。")
class Gtja176(GtjaAlpha):
    def compute(self, panel):
        high, low, close, vol = panel["high"], panel["low"], panel["close"], panel["volume"]
        ll = ops.ts_min(low, 12)
        stoch = _safe(close - ll, ops.ts_max(high, 12) - ll)
        return ops.correlation(ops.rank(stoch), ops.rank(vol), 6)


@_gtja(177, ("time_series", "reversal"),
       "20 日最高价出现的近期度(HIGHDAY)。")
class Gtja177(GtjaAlpha):
    def compute(self, panel):
        return (20 - ops.highday(panel["high"], 20)) / 20.0 * 100.0


@_gtja(178, ("time_series", "volume"),
       "日收益 × 成交量(量能加权动量)。")
class Gtja178(GtjaAlpha):
    def compute(self, panel):
        c, v = panel["close"], panel["volume"]
        return _safe(c - ops.delay(c, 1), ops.delay(c, 1)) * v


@_gtja(179, ("cross_sectional", "volume"),
       "VWAP-量 4 日相关性秩 × 低价秩-50 日均量秩 12 日相关性秩。")
class Gtja179(GtjaAlpha):
    def compute(self, panel):
        vw, vol, low = ops.vwap(panel), panel["volume"], panel["low"]
        return (ops.rank(ops.correlation(vw, vol, 4))
                * ops.rank(ops.correlation(ops.rank(low), ops.rank(ops.ts_mean(vol, 50)), 12)))


@_gtja(180, ("time_series", "volume"),
       "放量时按 7 日动量反向定价,缩量时按量反向。")
class Gtja180(GtjaAlpha):
    def compute(self, panel):
        close, vol = panel["close"], panel["volume"]
        cond = ops.ts_mean(vol, 20) < vol
        active = (-1.0 * ops.ts_rank(ops.delta(close, 7).abs(), 60)
                  * np.sign(ops.delta(close, 7)))
        return active.where(cond, -1.0 * vol)


@_gtja(184, ("cross_sectional", "reversal"),
       "昨日开收差与收盘 200 日相关性秩 + 当日开收差秩。")
class Gtja184(GtjaAlpha):
    def compute(self, panel):
        open_, close = panel["open"], panel["close"]
        return (ops.rank(ops.correlation(ops.delay(open_ - close, 1), close, 200))
                + ops.rank(open_ - close))


@_gtja(185, ("cross_sectional", "reversal"),
       "开收比偏离平方的截面秩取负。")
class Gtja185(GtjaAlpha):
    def compute(self, panel):
        open_, close = panel["open"], panel["close"]
        return ops.rank(-1.0 * (1.0 - _safe(open_, close)) ** 2)


@_gtja(187, ("time_series", "momentum"),
       "20 日开盘上行缺口动量累加。")
class Gtja187(GtjaAlpha):
    def compute(self, panel):
        open_, high = panel["open"], panel["high"]
        po = ops.delay(open_, 1)
        val = np.maximum(high - open_, open_ - po)
        x = val.where(open_ > po, 0.0).where(open_.notna() & po.notna())
        return ops.ts_sum(x, 20)


@_gtja(188, ("time_series",),
       "振幅相对 SMA(11,2) 均线的偏离(百分比)。")
class Gtja188(GtjaAlpha):
    def compute(self, panel):
        hl = panel["high"] - panel["low"]
        ma = ops.sma(hl, 11, 2)
        return _safe(hl - ma, ma) * 100.0


@_gtja(189, ("time_series",),
       "收盘相对 6 日均线的 6 日平均绝对偏离。")
class Gtja189(GtjaAlpha):
    def compute(self, panel):
        c = panel["close"]
        return ops.ts_mean((c - ops.ts_mean(c, 6)).abs(), 6)


@_gtja(191, ("cross_sectional", "volume"),
       "20 日均量与低价 5 日相关性 + 中价 - 收盘。")
class Gtja191(GtjaAlpha):
    def compute(self, panel):
        high, low, close, vol = panel["high"], panel["low"], panel["close"], panel["volume"]
        return (ops.correlation(ops.ts_mean(vol, 20), low, 5)
                + (high + low) / 2.0 - close)
