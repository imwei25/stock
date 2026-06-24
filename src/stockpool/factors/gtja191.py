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
