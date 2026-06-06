"""WorldQuant 101 Formulaic Alphas (Kakushadze, 2015).

每个 Alpha 一个 ``Factor`` 子类,名字 ``alpha_NNN`` (3 位补零)。

约定:
  * ``sources=("wq101",)``;``types`` 至少含 ``"cross_sectional"`` 或
    ``"time_series"`` 之一,部分含 ``"momentum"`` / ``"reversal"`` /
    ``"volume"`` / ``"industry_neutral"`` 等细分标签。
  * ``IndClass.subindustry`` 在我们的数据里没有 → 一律退化到 ``sector`` 一级。
    `indneutralize` 用 ``ctx.sector_map`` 当分组键(若未提供则跳过中性化)。
  * 部分 Alpha 依赖 ``cap``(总市值);项目无该字段 → 这些 Alpha 在 description
    里标注 "需 cap (未实现)",``compute`` 返回全 NaN。

行业中性化:本项目把 sector_map 通过 ``Factor`` 子类的 ``set_context`` 注入。
默认 sector_map 为空时,``indneutralize`` 退化为 horizontal demean。

公式来源: Kakushadze, "101 Formulaic Alphas" (2015),
arxiv.org/abs/1601.00991。
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.context import (  # noqa: F401 — re-export for back-compat
    _FactorContext,
    get_sector_map,
    indneutralize_with_context as _indneutralize,
    set_sector_map,
)
from stockpool.factors.registry import register


# Legacy alias for any code that imported _Wq101Context directly.
# All references now go through _FactorContext via factors.context.
_Wq101Context = _FactorContext


# ─────────────────────────────────────────────────────────────────────────────
# WqAlpha base: 统一 name / sources, 子类只需实现 compute + 类属性 NUM
# ─────────────────────────────────────────────────────────────────────────────

class WqAlpha(Factor):
    """WQ101 alpha 基类。子类设 ``NUM`` (1-101) 即可。"""
    NUM: ClassVar[int] = 0
    sources = ("wq101",)

    @property
    def name(self) -> str:
        return f"alpha_{self.NUM:03d}"

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "Factor":
        # WQ alpha 不带后缀参数
        return cls()


def _wq(num: int, types: tuple[str, ...], description: str):
    """装饰器: 简化 WQ alpha 注册。"""
    name = f"alpha_{num:03d}"
    def _wrap(cls):
        cls.NUM = num
        # 用 register 来登记 sources/types/description
        decorated = register(
            name, sources=("wq101",), types=types, description=description
        )(cls)
        return decorated
    return _wrap


# ─────────────────────────────────────────────────────────────────────────────
# 公用快捷量
# ─────────────────────────────────────────────────────────────────────────────

def _ret(panel):
    return panel["close"].pct_change(fill_method=None)


def _vwap(panel):
    return ops.vwap(panel)


def _adv(panel, d: int) -> pd.DataFrame:
    return panel["volume"].rolling(d, min_periods=d).mean()


def _nan_like(panel) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=panel["close"].index, columns=panel["close"].columns)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 001-010
# ─────────────────────────────────────────────────────────────────────────────

@_wq(1, ("cross_sectional", "reversal"),
     "横截面反转因子,先把负收益日替换为近 20 日收益率波动率、正收益日保留收盘价,平方后取近 5 日最大值出现的位置做横截面秩,衡量波动峰值在窗口内的相对位置。")
class Alpha001(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        cond = ret < 0
        x = ret.where(cond, panel["close"])  # default: close
        x = x.where(~cond, ops.ts_std(ret, 20))  # if ret<0 -> stddev
        sp = ops.signedpower(x, 2)
        ta = ops.ts_argmax(sp, 5)
        return ops.rank(ta) - 0.5


@_wq(2, ("cross_sectional", "volume"),
     "量价背离因子,对成交量对数差分的横截面秩与日内涨跌幅的横截面秩做近 6 日滚动相关再取负,捕捉量价不同步的反转信号。")
class Alpha002(WqAlpha):
    def compute(self, panel):
        log_v = np.log(panel["volume"].replace(0, np.nan))
        a = ops.rank(ops.delta(log_v, 2))
        b = ops.rank((panel["close"] - panel["open"]) / panel["open"])
        return -1.0 * ops.correlation(a, b, 6)


@_wq(3, ("cross_sectional", "volume"),
     "经典的开盘价与成交量横截面秩的负向 10 日滚动相关,是 WQ101 里最直观的量价反向因子,相关性越强信号越弱。")
class Alpha003(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.correlation(ops.rank(panel["open"]), ops.rank(panel["volume"]), 10)


@_wq(4, ("cross_sectional", "reversal"),
     "对最低价做横截面秩后再算近 9 日时序排名并取负,本质是短期反转因子:近期低点抬升越快越看空。")
class Alpha004(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.ts_rank(ops.rank(panel["low"]), 9)


@_wq(5, ("cross_sectional",),
     "开盘价相对近 10 日 vwap 均值的横截面秩,乘以收盘价与 vwap 偏离秩的绝对值取负,反映日内强弱与近期均价的背离。")
class Alpha005(WqAlpha):
    def compute(self, panel):
        vwap = _vwap(panel)
        a = ops.rank(panel["open"] - ops.ts_sum(vwap, 10) / 10.0)
        b = -1.0 * ops.rank(panel["close"] - vwap).abs()
        return a * b


@_wq(6, ("cross_sectional", "volume"),
     "开盘价与成交量的 10 日滚动相关取负,典型量价反向因子,跟 Alpha003 思路相近但不做秩归一。")
class Alpha006(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.correlation(panel["open"], panel["volume"], 10)


@_wq(7, ("cross_sectional",),
     "当成交量放大超过 20 日均量时,用近 7 日收盘差分的绝对值时序排名乘以方向取负,否则强制返回 -1,带成交量门控的趋势反转。")
class Alpha007(WqAlpha):
    def compute(self, panel):
        adv20 = _adv(panel, 20)
        d7 = ops.delta(panel["close"], 7)
        a = -1.0 * ops.ts_rank(d7.abs(), 60) * np.sign(d7)
        # 默认 -1,条件成立用 a
        out = pd.DataFrame(-1.0, index=panel["close"].index, columns=panel["close"].columns)
        return out.where(~(adv20 < panel["volume"]), a)


@_wq(8, ("cross_sectional", "momentum"),
     "近 5 日开盘价之和乘以 5 日收益率之和,与其 10 日前的同一组合做差,再做横截面秩并取负,捕捉量价动量的周期性反复。")
class Alpha008(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        prod = ops.ts_sum(panel["open"], 5) * ops.ts_sum(ret, 5)
        return -1.0 * ops.rank(prod - ops.delay(prod, 10))


@_wq(9, ("time_series",),
     "近 5 日日内涨跌均同号时延续动量,否则取反,纯时序的小周期动量/反转切换信号。")
class Alpha009(WqAlpha):
    def compute(self, panel):
        d = ops.delta(panel["close"], 1)
        out = -1.0 * d
        out = out.where(~(ops.ts_max(d, 5) < 0), d)
        out = out.where(~(ops.ts_min(d, 5) > 0), d)
        return out


@_wq(10, ("cross_sectional", "time_series"),
     "Alpha009 的横截面秩版本,4 日窗口判断方向一致性后再做横截面归一,把时序信号转成横截面排名。")
class Alpha010(WqAlpha):
    def compute(self, panel):
        d = ops.delta(panel["close"], 1)
        inner = -1.0 * d
        inner = inner.where(~(ops.ts_max(d, 4) < 0), d)
        inner = inner.where(~(ops.ts_min(d, 4) > 0), d)
        return ops.rank(inner)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 011-020
# ─────────────────────────────────────────────────────────────────────────────

@_wq(11, ("cross_sectional",),
     "vwap 与收盘价之差在近 3 日最大/最小值的横截面秩之和,再乘以成交量 3 日差分秩,把日内强弱与放量结合。")
class Alpha011(WqAlpha):
    def compute(self, panel):
        d = _vwap(panel) - panel["close"]
        return (ops.rank(ops.ts_max(d, 3)) + ops.rank(ops.ts_min(d, 3))) * ops.rank(ops.delta(panel["volume"], 3))


@_wq(12, ("cross_sectional", "volume"),
     "成交量变化方向乘以收盘价单日变化的反向,放量上涨偏空、放量下跌偏多,经典的量价短期反转。")
class Alpha012(WqAlpha):
    def compute(self, panel):
        return np.sign(ops.delta(panel["volume"], 1)) * (-1.0 * ops.delta(panel["close"], 1))


@_wq(13, ("cross_sectional", "volume"),
     "收盘价秩与成交量秩 5 日协方差再取横截面秩并取负,衡量价量同步性的横截面短期反转。")
class Alpha013(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.covariance(ops.rank(panel["close"]), ops.rank(panel["volume"]), 5))


@_wq(14, ("cross_sectional", "volume"),
     "近 3 日收益率差分的横截面秩取负,乘以开盘价与成交量的 10 日滚动相关,把动量减速与量价关系叠加。")
class Alpha014(WqAlpha):
    def compute(self, panel):
        return (-1.0 * ops.rank(ops.delta(_ret(panel), 3))) * ops.correlation(panel["open"], panel["volume"], 10)


@_wq(15, ("cross_sectional", "volume"),
     "最高价秩与成交量秩的 3 日滚动相关,横截面秩后再做 3 日时序累加并取负,衡量高位放量的持续性。")
class Alpha015(WqAlpha):
    def compute(self, panel):
        c = ops.correlation(ops.rank(panel["high"]), ops.rank(panel["volume"]), 3)
        return -1.0 * ops.ts_sum(ops.rank(c), 3)


@_wq(16, ("cross_sectional", "volume"),
     "最高价秩与成交量秩的 5 日协方差横截面秩并取负,跟 Alpha013 同款思路换成 high 价,捕捉冲高放量后的反转。")
class Alpha016(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.covariance(ops.rank(panel["high"]), ops.rank(panel["volume"]), 5))


@_wq(17, ("cross_sectional",),
     "把收盘价的 10 日时序排名秩、二阶差分秩、相对 20 日均量的近 5 日时序排名秩三者相乘,WQ101 数据挖掘风格的复合反转信号。")
class Alpha017(WqAlpha):
    def compute(self, panel):
        a = -1.0 * ops.rank(ops.ts_rank(panel["close"], 10))
        b = ops.rank(ops.delta(ops.delta(panel["close"], 1), 1))
        adv20 = _adv(panel, 20)
        c = ops.rank(ops.ts_rank(panel["volume"] / adv20, 5))
        return a * b * c


@_wq(18, ("cross_sectional",),
     "近 5 日日内振幅波动、当日涨跌、收开 10 日相关性相加后做横截面秩取负,综合波动与日内强弱的反向因子。")
class Alpha018(WqAlpha):
    def compute(self, panel):
        diff = panel["close"] - panel["open"]
        inside = ops.ts_std(diff.abs(), 5) + diff + ops.correlation(panel["close"], panel["open"], 10)
        return -1.0 * ops.rank(inside)


@_wq(19, ("cross_sectional", "momentum"),
     "判断近 7 日是涨是跌的方向取负,乘以近 250 日累计收益的横截面秩加一,把短期动量方向和长期收益位置反向耦合。")
class Alpha019(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        a = -1.0 * np.sign((c - ops.delay(c, 7)) + ops.delta(c, 7))
        ret = _ret(panel)
        b = 1.0 + ops.rank(1.0 + ops.ts_sum(ret, 250))
        return a * b


@_wq(20, ("cross_sectional", "reversal"),
     "开盘价分别相对前一日 high/close/low 偏离的横截面秩相乘,首项取负,典型的隔夜跳空反转因子。")
class Alpha020(WqAlpha):
    def compute(self, panel):
        op = panel["open"]
        a = -1.0 * ops.rank(op - ops.delay(panel["high"], 1))
        b = ops.rank(op - ops.delay(panel["close"], 1))
        c = ops.rank(op - ops.delay(panel["low"], 1))
        return a * b * c


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 021-030
# ─────────────────────────────────────────────────────────────────────────────

@_wq(21, ("time_series", "volume"),
     "用近 2 日均价与近 8 日均价加减一倍标准差比较,再叠加放量条件,产生 -1/+1 的三态信号,带波动通道的时序均值回归。")
class Alpha021(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        m8 = ops.ts_sum(c, 8) / 8.0
        s8 = ops.ts_std(c, 8)
        m2 = ops.ts_sum(c, 2) / 2.0
        adv20 = _adv(panel, 20)
        out = pd.DataFrame(-1.0, index=c.index, columns=c.columns)
        out = out.where(~(panel["volume"] / adv20 >= 1.0), 1.0)
        out = out.where(~(m2 < m8 - s8), 1.0)
        out = out.where(~(m8 + s8 < m2), -1.0)
        return out


@_wq(22, ("cross_sectional", "volume"),
     "high 与 volume 5 日滚动相关再做 5 日差分,乘以收盘价 20 日波动率的横截面秩并取负,量价关系恶化叠加高波动时的反转信号。")
class Alpha022(WqAlpha):
    def compute(self, panel):
        a = ops.delta(ops.correlation(panel["high"], panel["volume"], 5), 5)
        b = ops.rank(ops.ts_std(panel["close"], 20))
        return -1.0 * a * b


@_wq(23, ("time_series", "reversal"),
     "仅当 high 突破近 20 日均高时触发,信号为近 2 日 high 差分取负,其余日子为零,是带条件的突破后回落反转。")
class Alpha023(WqAlpha):
    def compute(self, panel):
        h = panel["high"]
        cond = (ops.ts_sum(h, 20) / 20.0) < h
        out = -1.0 * ops.delta(h, 2)
        return out.where(cond, 0.0)


@_wq(24, ("time_series", "reversal"),
     "如果近 100 日均价相对 100 日前的涨幅极小,则发出距离 100 日低点的负偏离;否则用近 3 日收盘差分取负,慢牛/震荡市切换的反转信号。")
class Alpha024(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        ma100 = ops.ts_sum(c, 100) / 100.0
        cond = ops.delta(ma100, 100) / ops.delay(c, 100) <= 0.05
        a = -1.0 * (c - ops.ts_min(c, 100))
        b = -1.0 * ops.delta(c, 3)
        return a.where(cond, b)


@_wq(25, ("cross_sectional", "volume"),
     "把负收益、20 日均量、vwap 和当日上影线长度连乘后做横截面秩,综合反转、流动性和日内弱势,典型 WQ101 数据挖掘合成因子。")
class Alpha025(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        return ops.rank(((-1.0 * ret) * _adv(panel, 20) * _vwap(panel)) * (panel["high"] - panel["close"]))


@_wq(26, ("cross_sectional", "volume"),
     "近 5 日成交量时序排名与最高价时序排名做 5 日滚动相关,再取近 3 日最大值并取负,捕捉量价同步走强后的短期反转。")
class Alpha026(WqAlpha):
    def compute(self, panel):
        c = ops.correlation(ops.ts_rank(panel["volume"], 5), ops.ts_rank(panel["high"], 5), 5)
        return -1.0 * ops.ts_max(c, 3)


@_wq(27, ("cross_sectional", "volume"),
     "横截面下成交量秩与 vwap 秩的 6 日滚动相关,值高发出做空、低发出做多的反向信号,典型量价相关反转。")
class Alpha027(WqAlpha):
    def compute(self, panel):
        c = ops.correlation(ops.rank(panel["volume"]), ops.rank(_vwap(panel)), 6)
        r = ops.rank(ops.ts_sum(c, 2) / 2.0)
        out = pd.DataFrame(1.0, index=r.index, columns=r.columns)
        return out.where(~(r > 0.5), -1.0)


@_wq(28, ("cross_sectional",),
     "20 日均量与最低价的 5 日滚动相关,叠加中间价偏离收盘的部分,再做横截面标准化的复合因子。")
class Alpha028(WqAlpha):
    def compute(self, panel):
        return ops.scale(
            ops.correlation(_adv(panel, 20), panel["low"], 5)
            + (panel["high"] + panel["low"]) / 2.0
            - panel["close"]
        )


@_wq(29, ("cross_sectional",),
     "WQ101 数据挖掘风格的横截面复合,包含多层 rank、log、scale 与 6 日延迟收益率时序排名,无清晰金融直觉。")
class Alpha029(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        ret = _ret(panel)
        inner = -1.0 * ops.rank(ops.delta(c - 1.0, 5))
        inner = ops.rank(ops.rank(inner))
        inner = ops.ts_min(inner, 2)
        inner = ops.ts_sum(inner, 1)
        inner = np.log(inner.where(inner > 0, np.nan))
        inner = ops.rank(ops.rank(ops.scale(inner)))
        a = ops.ts_min(ops.ts_product(inner, 1), 5)
        b = ops.ts_rank(ops.delay(-1.0 * ret, 6), 5)
        return a + b


@_wq(30, ("cross_sectional", "volume", "reversal"),
     "用最近 3 日收盘方向是否一致衡量趋势连续性,连续同向时减仓,再用近 5 日与 20 日成交量比值放大,属于量价反转因子。")
class Alpha030(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        s = np.sign(c - ops.delay(c, 1)) + np.sign(ops.delay(c, 1) - ops.delay(c, 2)) + np.sign(ops.delay(c, 2) - ops.delay(c, 3))
        return ((1.0 - ops.rank(s)) * ops.ts_sum(panel["volume"], 5)) / ops.ts_sum(panel["volume"], 20)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 031-040
# ─────────────────────────────────────────────────────────────────────────────

@_wq(31, ("cross_sectional",),
     "三段叠加:10 日动量的衰减加权 rank、3 日反向动量 rank、以及 20 日均量与最低价相关的符号,综合横截面打分。")
class Alpha031(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        a = ops.rank(ops.rank(ops.rank(ops.decay_linear(-1.0 * ops.rank(ops.rank(ops.delta(c, 10))), 10))))
        b = ops.rank(-1.0 * ops.delta(c, 3))
        d = np.sign(ops.scale(ops.correlation(_adv(panel, 20), panel["low"], 12)))
        return a + b + d


@_wq(32, ("cross_sectional",),
     "近 7 日均价相对当前收盘的偏离,叠加 vwap 与 5 日前收盘的长期 230 日相关,属于均值回归 + 长周期量价共振复合。")
class Alpha032(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        a = ops.scale(ops.ts_sum(c, 7) / 7.0 - c)
        b = 20.0 * ops.scale(ops.correlation(_vwap(panel), ops.delay(c, 5), 230))
        return a + b


@_wq(33, ("cross_sectional",),
     "横截面下对开收比 open/close 取负后做秩,本质是收盘相对开盘越强排名越靠前的简单日内动量。")
class Alpha033(WqAlpha):
    def compute(self, panel):
        return ops.rank(-1.0 * (1.0 - panel["open"] / panel["close"]))


@_wq(34, ("cross_sectional",),
     "把短期(2 日)与中期(5 日)收益波动比、以及 1 日收盘差分都做反向横截面秩相加,挑选低波动 + 短期下跌的票。")
class Alpha034(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        a = 1.0 - ops.rank(ops.ts_std(ret, 2) / ops.ts_std(ret, 5))
        b = 1.0 - ops.rank(ops.delta(panel["close"], 1))
        return ops.rank(a + b)


@_wq(35, ("time_series",),
     "32 日成交量时序排名乘以(1 - 16 日价格振幅时序排名)乘以(1 - 32 日收益时序排名),倾向于高量、低波动、低收益的票。")
class Alpha035(WqAlpha):
    def compute(self, panel):
        a = ops.ts_rank(panel["volume"], 32)
        b = 1.0 - ops.ts_rank(panel["close"] + panel["high"] - panel["low"], 16)
        c = 1.0 - ops.ts_rank(_ret(panel), 32)
        return a * b * c


@_wq(36, ("cross_sectional", "volume"),
     "5 个子项加权求和的横截面因子,涵盖日内涨跌与昨量相关、开收差、滞后收益时序排名、vwap-均量相关绝对值、200 日均价偏离等。")
class Alpha036(WqAlpha):
    def compute(self, panel):
        c, o = panel["close"], panel["open"]
        ret = _ret(panel)
        a = 2.21 * ops.rank(ops.correlation(c - o, ops.delay(panel["volume"], 1), 15))
        b = 0.7 * ops.rank(o - c)
        d = 0.73 * ops.rank(ops.ts_rank(ops.delay(-1.0 * ret, 6), 5))
        e = ops.rank(ops.correlation(_vwap(panel), _adv(panel, 20), 6).abs())
        f = 0.6 * ops.rank((ops.ts_sum(c, 200) / 200.0 - o) * (c - o))
        return a + b + d + e + f


@_wq(37, ("cross_sectional",),
     "昨日开收差与今日收盘的 200 日长周期滚动相关,加上当日开收差的横截面秩,捕捉日内反转的持续模式。")
class Alpha037(WqAlpha):
    def compute(self, panel):
        c, o = panel["close"], panel["open"]
        return ops.rank(ops.correlation(ops.delay(o - c, 1), c, 200)) + ops.rank(o - c)


@_wq(38, ("cross_sectional", "reversal"),
     "10 日收盘时序排名与日内收开比的横截面秩相乘后取负,属于短期反转因子:涨得久且当日强的票给负分。")
class Alpha038(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_rank(panel["close"], 10)) * ops.rank(panel["close"] / panel["open"])


@_wq(39, ("cross_sectional", "momentum"),
     "7 日动量乘以(1 - 衰减加权的相对成交量秩),再乘以 250 日累计收益的秩(+1),把短期动量按长期动量和量能调权。")
class Alpha039(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        ret = _ret(panel)
        dvol = panel["volume"] / _adv(panel, 20)
        a = -1.0 * ops.rank(ops.delta(c, 7) * (1.0 - ops.rank(ops.decay_linear(dvol, 9))))
        b = 1.0 + ops.rank(ops.ts_sum(ret, 250))
        return a * b


@_wq(40, ("cross_sectional", "volatility"),
     "10 日最高价波动的横截面秩乘以最高价与成交量的 10 日相关,再取负,本质是高波动+量价同步走强的反向信号。")
class Alpha040(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_std(panel["high"], 10)) * ops.correlation(panel["high"], panel["volume"], 10)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 041-050
# ─────────────────────────────────────────────────────────────────────────────

@_wq(41, ("cross_sectional",),
     "高低价几何均值减去 vwap,衡量当日中枢相对成交均价的偏离,是一个简单的日内价位偏离因子。")
class Alpha041(WqAlpha):
    def compute(self, panel):
        return (panel["high"] * panel["low"]) ** 0.5 - _vwap(panel)


@_wq(42, ("cross_sectional",),
     "vwap 减收盘的横截面秩除以 vwap 加收盘的秩,捕捉收盘相对成交均价的相对位置,数值小代表收盘偏强。")
class Alpha042(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        return ops.rank(vw - panel["close"]) / ops.rank(vw + panel["close"])


@_wq(43, ("time_series",),
     "相对成交量(volume/adv20)的 20 日时序排名乘以 7 日反向动量的 8 日时序排名,典型量价共振反转。")
class Alpha043(WqAlpha):
    def compute(self, panel):
        return ops.ts_rank(panel["volume"] / _adv(panel, 20), 20) * ops.ts_rank(-1.0 * ops.delta(panel["close"], 7), 8)


@_wq(44, ("cross_sectional", "volume"),
     "最高价与横截面成交量秩的 5 日滚动相关取负,放大量价背离信号的反向因子。")
class Alpha044(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.correlation(panel["high"], ops.rank(panel["volume"]), 5)


@_wq(45, ("cross_sectional",),
     "5 日前收盘的 20 日均值秩、收盘成交量的 2 日相关、以及 5/20 日累计收盘的相关秩,三者相乘取负的复合因子。")
class Alpha045(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        a = ops.rank(ops.ts_sum(ops.delay(c, 5), 20) / 20.0)
        b = ops.correlation(c, panel["volume"], 2)
        d = ops.rank(ops.correlation(ops.ts_sum(c, 5), ops.ts_sum(c, 20), 2))
        return -1.0 * a * b * d


@_wq(46, ("time_series", "reversal"),
     "通过比较最近 20-10 日与 10-0 日的趋势加速度,加速向上时做空、向下时做多,否则按 1 日反向动量,典型时序反转。")
class Alpha046(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        diff = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 1))
        out = out.where(~(diff < 0), 1.0)
        out = out.where(~(diff > 0.25), -1.0)
        return out


@_wq(47, ("cross_sectional", "volume"),
     "把收盘倒数秩 × 成交量 / 均量 乘以高价相对 5 日均高的偏离,再减去 vwap 的 5 日动量秩,属于量价加权的反转因子。")
class Alpha047(WqAlpha):
    def compute(self, panel):
        c, h = panel["close"], panel["high"]
        adv20 = _adv(panel, 20)
        vw = _vwap(panel)
        a = (ops.rank(1.0 / c) * panel["volume"]) / adv20
        b = (h * ops.rank(h - c)) / (ops.ts_sum(h, 5) / 5.0)
        d = ops.rank(vw - ops.delay(vw, 5))
        return a * b - d


@_wq(48, ("cross_sectional", "industry_neutral"),
     "对一阶差分自相关与价格变动的复合做行业中性化,再除以 250 日累计平方收益,捕捉行业内动量持续性的归一化版本。")
class Alpha048(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        d1 = ops.delta(c, 1)
        corr = ops.correlation(d1, ops.delta(ops.delay(c, 1), 1), 250)
        numer = _indneutralize(corr * d1 / c)
        denom = ops.ts_sum((d1 / ops.delay(c, 1)) ** 2, 250)
        return numer / denom


@_wq(49, ("time_series",),
     "类似 Alpha046 的趋势加速度判断,但阈值更宽松(-0.1):加速向下时做多,否则按 1 日反向动量,时序反转因子。")
class Alpha049(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        diff = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 1))
        return out.where(~(diff < -0.1), 1.0)


@_wq(50, ("cross_sectional", "volume"),
     "成交量秩与 vwap 秩的 5 日滚动相关再取横截面秩,取 5 日最大值后取负,惩罚量价同步走强的票,典型反转信号。")
class Alpha050(WqAlpha):
    def compute(self, panel):
        c = ops.correlation(ops.rank(panel["volume"]), ops.rank(_vwap(panel)), 5)
        return -1.0 * ops.ts_max(ops.rank(c), 5)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 051-060
# ─────────────────────────────────────────────────────────────────────────────

@_wq(51, ("time_series",),
     "看 20 日到 10 日均速与近 10 日均速的差异,若长期减速明显则给 +1 多头,否则按昨日涨跌取反向,是一个偏反转的时序信号。")
class Alpha051(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        diff = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 1))
        return out.where(~(diff < -0.05), 1.0)


@_wq(52, ("cross_sectional",),
     "5 日最低价的差分动量、长短期累计收益率横截面秩、近 5 日成交量时序排名三者相乘,长期-短期收益率差是核心。")
class Alpha052(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        a = -1.0 * ops.ts_min(panel["low"], 5) + ops.delay(ops.ts_min(panel["low"], 5), 5)
        b = ops.rank((ops.ts_sum(ret, 240) - ops.ts_sum(ret, 20)) / 220.0)
        d = ops.ts_rank(panel["volume"], 5)
        return a * b * d


@_wq(53, ("time_series",),
     "对(收盘相对当日区间的位置)做 9 日差分取负,本质是反转因子,区间内位置回落则给出多头信号。")
class Alpha053(WqAlpha):
    def compute(self, panel):
        c, h, l = panel["close"], panel["high"], panel["low"]
        denom = (c - l).replace(0.0, np.nan)
        x = ((c - l) - (h - c)) / denom
        return -1.0 * ops.delta(x, 9)


@_wq(54, ("cross_sectional",),
     "用开盘价五次方与收盘价五次方的比值放大区间位置差异,属于横截面价格形态因子,无明确金融直觉。")
class Alpha054(WqAlpha):
    def compute(self, panel):
        c, o, h, l = panel["close"], panel["open"], panel["high"], panel["low"]
        num = -1.0 * ((l - c) * (o ** 5))
        den = ((l - h) * (c ** 5)).replace(0.0, np.nan)
        return num / den


@_wq(55, ("cross_sectional", "volume"),
     "12 日通道内收盘相对位置的秩与成交量秩的 6 日滚动相关取负,属于典型量价背离类反转因子。")
class Alpha055(WqAlpha):
    def compute(self, panel):
        c, h, l = panel["close"], panel["high"], panel["low"]
        lo12 = ops.ts_min(l, 12)
        hi12 = ops.ts_max(h, 12)
        rng = (hi12 - lo12).replace(0.0, np.nan)
        r = (c - lo12) / rng
        return -1.0 * ops.correlation(ops.rank(r), ops.rank(panel["volume"]), 6)


@_wq(56, ("cross_sectional", "industry_neutral"),
     "原版需要总市值(cap)字段,本项目未实现市值字段,直接返回 NaN,仅作占位。")
class Alpha056(WqAlpha):
    def compute(self, panel):
        # 项目无 cap 字段;返回 NaN
        return _nan_like(panel)


@_wq(57, ("cross_sectional",),
     "收盘价相对 vwap 的偏离除以 30 日最高位置秩的衰减权重并取负,价格偏离均价时给反向信号。")
class Alpha057(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        return -1.0 * ((c - _vwap(panel)) / ops.decay_linear(ops.rank(ops.ts_argmax(c, 30)), 2))


@_wq(58, ("cross_sectional", "industry_neutral"),
     "对行业中性化的 vwap 与成交量做滚动相关,再线性衰减加权后做时序排名取负,属于行业中性的量价背离因子。")
class Alpha058(WqAlpha):
    def compute(self, panel):
        vw_n = _indneutralize(_vwap(panel))
        c = ops.correlation(vw_n, panel["volume"], 4)
        dl = ops.decay_linear(c, 8)
        return -1.0 * ops.ts_rank(dl, 6)


@_wq(59, ("cross_sectional", "industry_neutral"),
     "与 58 号几乎一致,区别只在 vwap 用 0.728/0.272 加权(化简后仍是 vwap),同样是行业中性量价背离的反向因子。")
class Alpha059(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        weighted = vw * 0.728317 + vw * (1 - 0.728317)  # 简化为 vwap
        c = ops.correlation(_indneutralize(weighted), panel["volume"], 4)
        return -1.0 * ops.ts_rank(ops.decay_linear(c, 16), 8)


@_wq(60, ("cross_sectional",),
     "把(区间位置×成交量)的横截面秩与 10 日最高位置秩相减,构造方向偏空的横截面合成,WQ101 数据挖掘风格。")
class Alpha060(WqAlpha):
    def compute(self, panel):
        c, h, l = panel["close"], panel["high"], panel["low"]
        rng = (h - l).replace(0.0, np.nan)
        a = ops.scale(ops.rank(((c - l) - (h - c)) / rng * panel["volume"]))
        b = ops.scale(ops.rank(ops.ts_argmax(c, 10)))
        return -1.0 * (2.0 * a - b)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 061-070
# ─────────────────────────────────────────────────────────────────────────────

@_wq(61, ("cross_sectional",),
     "比较 vwap 距 16 日低点的秩与 vwap 和 180 日均额相关性的秩,前者较小时给多头,是量价配合类信号。")
class Alpha061(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(vw - ops.ts_min(vw, 16))
        b = ops.rank(ops.correlation(vw, _adv(panel, 180), 18))
        return (a < b).astype(float)


@_wq(62, ("cross_sectional",),
     "vwap 与 22 日累计 adv20 的相关性秩,对比开盘与高低均价秩的不等式比较,结果取负,量价结构类反转。")
class Alpha062(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        adv20 = _adv(panel, 20)
        a = ops.rank(ops.correlation(vw, ops.ts_sum(adv20, 22), 10))
        b = ops.rank(((ops.rank(panel["open"]) + ops.rank(panel["open"]))
                      < (ops.rank((panel["high"] + panel["low"]) / 2.0) + ops.rank(panel["high"]))).astype(float))
        return (a < b).astype(float) * -1.0


@_wq(63, ("cross_sectional", "industry_neutral"),
     "行业中性后收盘的近期差分衰减秩,减去(vwap 与开盘加权)与长期 adv180 累计的相关性秩,再取负,行业中性量价合成。")
class Alpha063(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        vw = _vwap(panel)
        x = _indneutralize(c)
        a = ops.rank(ops.decay_linear(ops.delta(x, 2), 8))
        adv180 = _adv(panel, 180)
        proxy = vw * 0.318108 + panel["open"] * (1 - 0.318108)
        b = ops.rank(ops.decay_linear(ops.correlation(proxy, ops.ts_sum(adv180, 37), 13), 12))
        return -1.0 * (a - b)


@_wq(64, ("cross_sectional",),
     "开盘与最低加权后的 12 日累计与 adv120 累计的相关性秩,对比(高低中价与 vwap 加权)的 3 日差分秩,反向信号。")
class Alpha064(WqAlpha):
    def compute(self, panel):
        adv120 = _adv(panel, 120)
        a_term = panel["open"] * 0.178404 + panel["low"] * (1 - 0.178404)
        a = ops.rank(ops.correlation(ops.ts_sum(a_term, 12), ops.ts_sum(adv120, 12), 16))
        b_term = (panel["high"] + panel["low"]) / 2.0 * 0.178404 + _vwap(panel) * (1 - 0.178404)
        b = ops.rank(ops.delta(b_term, 3))
        return (a < b).astype(float) * -1.0


@_wq(65, ("cross_sectional",),
     "开盘与 vwap 加权后与 adv60 累计的相关性秩,对比开盘距 13 日最低位置秩,前者较小则给反向信号。")
class Alpha065(WqAlpha):
    def compute(self, panel):
        adv60 = _adv(panel, 60)
        a_term = panel["open"] * 0.00817205 + _vwap(panel) * (1 - 0.00817205)
        a = ops.rank(ops.correlation(a_term, ops.ts_sum(adv60, 8), 6))
        b = ops.rank(panel["open"] - ops.ts_min(panel["open"], 13))
        return (a < b).astype(float) * -1.0


@_wq(66, ("cross_sectional",),
     "vwap 的 3 日差分线性衰减秩,加上(最低相对 vwap 偏离 / 开盘相对中价偏离)的衰减时序排名,整体取负。")
class Alpha066(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 3), 7))
        denom = (panel["open"] - (panel["high"] + panel["low"]) / 2.0).replace(0.0, np.nan)
        inner = (panel["low"] - vw) / denom
        b = ops.ts_rank(ops.decay_linear(inner, 11), 6)
        return -1.0 * (a + b)


@_wq(67, ("cross_sectional", "industry_neutral"),
     "高点距 2 日最低的秩,按(行业中性 vwap 与行业中性 adv20 相关性的秩)做幂次复合,结果取负,属典型 WQ 复合形式。")
class Alpha067(WqAlpha):
    def compute(self, panel):
        a = ops.rank(panel["high"] - ops.ts_min(panel["high"], 2))
        b = ops.rank(ops.correlation(_indneutralize(_vwap(panel)),
                                     _indneutralize(_adv(panel, 20)), 6))
        # WQ 公式里的 ^ 是 power;为数值稳定限制底数为非负
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@_wq(68, ("cross_sectional",),
     "高点秩与 adv15 秩的 9 日相关性时序排名,对比(收盘与最低加权)的 1 日差分秩,前者较小给反向信号。")
class Alpha068(WqAlpha):
    def compute(self, panel):
        adv15 = _adv(panel, 15)
        a = ops.ts_rank(ops.correlation(ops.rank(panel["high"]), ops.rank(adv15), 9), 14)
        b_term = panel["close"] * 0.518371 + panel["low"] * (1 - 0.518371)
        b = ops.rank(ops.delta(b_term, 1))
        return (a < b).astype(float) * -1.0


@_wq(69, ("cross_sectional", "industry_neutral"),
     "行业中性 vwap 差分 5 日最大值的秩,按收盘与 vwap 加权和 adv20 相关性的时序排名做幂次复合,取负。")
class Alpha069(WqAlpha):
    def compute(self, panel):
        vw_n = _indneutralize(_vwap(panel))
        a = ops.rank(ops.ts_max(ops.delta(vw_n, 3), 5))
        adv20 = _adv(panel, 20)
        inner = (panel["close"] * 0.490655 + _vwap(panel) * (1 - 0.490655))
        b = ops.ts_rank(ops.correlation(inner, adv20, 5), 9)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@_wq(70, ("cross_sectional", "industry_neutral"),
     "vwap 的 1 日差分秩,按(行业中性收盘与 adv50 的 18 日相关性的时序排名)做幂次复合,最后取负。")
class Alpha070(WqAlpha):
    def compute(self, panel):
        a = ops.rank(ops.delta(_vwap(panel), 1))
        adv50 = _adv(panel, 50)
        b = ops.ts_rank(ops.correlation(_indneutralize(panel["close"]), adv50, 18), 18)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 071-080
# ─────────────────────────────────────────────────────────────────────────────

@_wq(71, ("cross_sectional",),
     "收盘时序秩与 adv180 时序秩相关性的衰减再排名,与(低开相对 vwap 偏离秩平方)的衰减再排名取最大值,属横截面复合。")
class Alpha071(WqAlpha):
    def compute(self, panel):
        adv180 = _adv(panel, 180)
        a = ops.ts_rank(ops.decay_linear(ops.correlation(ops.ts_rank(panel["close"], 3),
                                                         ops.ts_rank(adv180, 12), 18), 4), 15)
        vw = _vwap(panel)
        inner = ops.rank((panel["low"] + panel["open"]) - (vw + vw))
        b = ops.ts_rank(ops.decay_linear(inner ** 2, 16), 4)
        return pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(72, ("cross_sectional",),
     "中价与 adv40 相关性的衰减秩,除以 vwap 时序秩与成交量时序秩相关性的衰减秩,量价相关强弱比值。")
class Alpha072(WqAlpha):
    def compute(self, panel):
        hl = (panel["high"] + panel["low"]) / 2.0
        adv40 = _adv(panel, 40)
        a = ops.rank(ops.decay_linear(ops.correlation(hl, adv40, 9), 10))
        b = ops.rank(ops.decay_linear(ops.correlation(ops.ts_rank(_vwap(panel), 4),
                                                      ops.ts_rank(panel["volume"], 19), 7), 3))
        return a / b


@_wq(73, ("cross_sectional",),
     "vwap 的 5 日差分衰减秩,与(开盘最低加权的 2 日相对差分取负)的衰减时序排名取最大值,再取负,反转复合。")
class Alpha073(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 5), 3))
        proxy = panel["open"] * 0.147155 + panel["low"] * 0.852845
        inner = ops.delta(proxy, 2) / proxy.replace(0.0, np.nan) * -1.0
        b = ops.ts_rank(ops.decay_linear(inner, 3), 17)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(74, ("cross_sectional",),
     "收盘与 37 日累计 adv30 相关性的秩,对比(高点与 vwap 加权)秩与成交量秩 11 日相关性的秩,前者较小给反向。")
class Alpha074(WqAlpha):
    def compute(self, panel):
        adv30 = _adv(panel, 30)
        a = ops.rank(ops.correlation(panel["close"], ops.ts_sum(adv30, 37), 15))
        proxy = panel["high"] * 0.0261661 + _vwap(panel) * (1 - 0.0261661)
        b = ops.rank(ops.correlation(ops.rank(proxy), ops.rank(panel["volume"]), 11))
        return (a < b).astype(float) * -1.0


@_wq(75, ("cross_sectional",),
     "vwap 与成交量 4 日相关性的秩,对比最低秩与 adv50 秩的 12 日相关性的秩,前者较小给多头,属量价配合因子。")
class Alpha075(WqAlpha):
    def compute(self, panel):
        adv50 = _adv(panel, 50)
        a = ops.rank(ops.correlation(_vwap(panel), panel["volume"], 4))
        b = ops.rank(ops.correlation(ops.rank(panel["low"]), ops.rank(adv50), 12))
        return (a < b).astype(float)


@_wq(76, ("cross_sectional", "industry_neutral"),
     "VWAP 短期差分秩与行业中性化后 low 与成交额相关性的合成,取较大值后取负,典型 WQ101 反转风格挖掘因子。")
class Alpha076(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 1), 12))
        adv81 = _adv(panel, 81)
        inner = ops.ts_rank(ops.correlation(_indneutralize(panel["low"]), adv81, 8), 20)
        b = ops.ts_rank(ops.decay_linear(inner, 17), 19)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(77, ("cross_sectional",),
     "结合中价高于 VWAP 的程度与中价同成交额的相关性,取两路秩衰减的较小值,偏短期反转的横截面信号。")
class Alpha077(WqAlpha):
    def compute(self, panel):
        hl = (panel["high"] + panel["low"]) / 2.0
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear((hl + panel["high"]) - (vw + panel["high"]), 20))
        adv40 = _adv(panel, 40)
        b = ops.rank(ops.decay_linear(ops.correlation(hl, adv40, 3), 6))
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(78, ("cross_sectional",),
     "20 日加权低价均值与成交额均值的相关性,再用 VWAP 与成交量秩相关作为指数幂调制,量价共振挖掘因子。")
class Alpha078(WqAlpha):
    def compute(self, panel):
        adv40 = _adv(panel, 40)
        proxy = panel["low"] * 0.352233 + _vwap(panel) * (1 - 0.352233)
        a = ops.rank(ops.correlation(ops.ts_sum(proxy, 20), ops.ts_sum(adv40, 20), 7))
        b = ops.rank(ops.correlation(ops.rank(_vwap(panel)), ops.rank(panel["volume"]), 6))
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return ops.signedpower(a_safe, b)


@_wq(79, ("cross_sectional", "industry_neutral"),
     "行业中性化后的收开盘加权价单日差分秩,与 VWAP 时序排名同成交额时序排名相关性的横截面比较,偏动量方向因子。")
class Alpha079(WqAlpha):
    def compute(self, panel):
        proxy = panel["close"] * 0.60733 + panel["open"] * (1 - 0.60733)
        a = ops.rank(ops.delta(_indneutralize(proxy), 1))
        adv150 = _adv(panel, 150)
        b = ops.rank(ops.correlation(ops.ts_rank(_vwap(panel), 4), ops.ts_rank(adv150, 9), 15))
        return (a < b).astype(float)


@_wq(80, ("cross_sectional", "industry_neutral"),
     "行业中性化后的开高加权价 4 日动量方向,与高价同成交额相关性时序排名的指数复合,再取负,反转挖掘因子。")
class Alpha080(WqAlpha):
    def compute(self, panel):
        proxy = panel["open"] * 0.868128 + panel["high"] * (1 - 0.868128)
        a = ops.rank(np.sign(ops.delta(_indneutralize(proxy), 4)))
        adv10 = _adv(panel, 10)
        b = ops.ts_rank(ops.correlation(panel["high"], adv10, 5), 6)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 081-090
# ─────────────────────────────────────────────────────────────────────────────

@_wq(81, ("cross_sectional",),
     "VWAP 与成交额累计的高阶相关性秩积取对数,与 VWAP 和成交量秩相关比较后取负,典型 WQ101 量价挖掘合成。")
class Alpha081(WqAlpha):
    def compute(self, panel):
        adv10 = _adv(panel, 10)
        vw = _vwap(panel)
        inner = ops.rank(ops.rank(ops.correlation(vw, ops.ts_sum(adv10, 50), 8)) ** 4)
        prod = ops.ts_product(inner, 15)
        a = ops.rank(np.log(prod.where(prod > 0, np.nan)))
        b = ops.rank(ops.correlation(ops.rank(vw), ops.rank(panel["volume"]), 5))
        return (a < b).astype(float) * -1.0


@_wq(82, ("cross_sectional", "industry_neutral"),
     "开盘价 1 日动量的秩衰减,与行业中性化成交量同开盘价相关性的时序排名,取较小值后取负,短期反转因子。")
class Alpha082(WqAlpha):
    def compute(self, panel):
        a = ops.rank(ops.decay_linear(ops.delta(panel["open"], 1), 15))
        proxy = panel["open"] * 0.634196 + panel["open"] * (1 - 0.634196)
        b = ops.ts_rank(ops.decay_linear(ops.correlation(_indneutralize(panel["volume"]), proxy, 17), 7), 13)
        return -1.0 * pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(83, ("cross_sectional",),
     "前两日波动幅度比例乘以成交量秩,除以当前波幅与 VWAP 收盘价差的比,量价波动复合的横截面因子。")
class Alpha083(WqAlpha):
    def compute(self, panel):
        c, h, l = panel["close"], panel["high"], panel["low"]
        vw = _vwap(panel)
        ratio = (h - l) / (ops.ts_sum(c, 5) / 5.0)
        a = ops.rank(ops.delay(ratio, 2)) * ops.rank(ops.rank(panel["volume"]))
        denom = (ratio / (vw - c).replace(0.0, np.nan))
        return a / denom


@_wq(84, ("cross_sectional",),
     "VWAP 偏离 15 日高点的时序排名,以 5 日 close 动量为指数做符号幂运算,无明确金融解释的数据挖掘因子。")
class Alpha084(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.ts_rank(vw - ops.ts_max(vw, 15), 21)
        b = ops.delta(panel["close"], 5)
        return ops.signedpower(a, b)


@_wq(85, ("cross_sectional",),
     "高收加权价与成交额相关性秩,以中价时序排名和成交量时序排名相关性为指数做符号幂调制,量价相关挖掘因子。")
class Alpha085(WqAlpha):
    def compute(self, panel):
        adv30 = _adv(panel, 30)
        proxy = panel["high"] * 0.876703 + panel["close"] * (1 - 0.876703)
        a = ops.rank(ops.correlation(proxy, adv30, 10))
        hl = (panel["high"] + panel["low"]) / 2.0
        b = ops.rank(ops.correlation(ops.ts_rank(hl, 4), ops.ts_rank(panel["volume"], 10), 7))
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return ops.signedpower(a_safe, b)


@_wq(86, ("cross_sectional",),
     "close 与 20 日成交额累计的相关性时序排名,与开收 VWAP 差异秩的横截面比较,取负后输出,反转挖掘因子。")
class Alpha086(WqAlpha):
    def compute(self, panel):
        adv20 = _adv(panel, 20)
        a = ops.ts_rank(ops.correlation(panel["close"], ops.ts_sum(adv20, 15), 6), 20)
        vw = _vwap(panel)
        b = ops.rank((panel["open"] + panel["close"]) - (vw + panel["open"]))
        return (a < b).astype(float) * -1.0


@_wq(87, ("cross_sectional", "industry_neutral"),
     "收 VWAP 加权价 2 日动量秩衰减,与行业中性化成交额同 close 相关性绝对值的时序排名,取较大值再取负。")
class Alpha087(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        proxy = panel["close"] * 0.369701 + vw * (1 - 0.369701)
        a = ops.rank(ops.decay_linear(ops.delta(proxy, 2), 3))
        adv81 = _adv(panel, 81)
        inner = ops.correlation(_indneutralize(adv81), panel["close"], 13).abs()
        b = ops.ts_rank(ops.decay_linear(inner, 5), 14)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(88, ("cross_sectional",),
     "开低秩之和减高收秩之和的衰减秩,与 close 时序排名同成交额时序排名相关性衰减的时序排名,取较小值。")
class Alpha088(WqAlpha):
    def compute(self, panel):
        a = ops.rank(ops.decay_linear((ops.rank(panel["open"]) + ops.rank(panel["low"]))
                                       - (ops.rank(panel["high"]) + ops.rank(panel["close"])), 8))
        adv60 = _adv(panel, 60)
        inner = ops.correlation(ops.ts_rank(panel["close"], 8), ops.ts_rank(adv60, 21), 8)
        b = ops.ts_rank(ops.decay_linear(inner, 7), 3)
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(89, ("cross_sectional", "industry_neutral"),
     "低价与成交额相关性的时序排名,减去行业中性化 VWAP 动量的时序排名,量价信号减去动量的合成因子。")
class Alpha089(WqAlpha):
    def compute(self, panel):
        adv10 = _adv(panel, 10)
        proxy = panel["low"] * 0.967285 + panel["low"] * (1 - 0.967285)
        a = ops.ts_rank(ops.decay_linear(ops.correlation(proxy, adv10, 7), 6), 4)
        inner = ops.delta(_indneutralize(_vwap(panel)), 3)
        b = ops.ts_rank(ops.decay_linear(inner, 10), 15)
        return a - b


@_wq(90, ("cross_sectional", "industry_neutral"),
     "close 偏离 5 日高点的秩,以行业中性化成交额同 low 相关性时序排名为指数做符号幂,再取负,反转因子。")
class Alpha090(WqAlpha):
    def compute(self, panel):
        a = ops.rank(panel["close"] - ops.ts_max(panel["close"], 5))
        adv40 = _adv(panel, 40)
        b = ops.ts_rank(ops.correlation(_indneutralize(adv40), panel["low"], 5), 3)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 091-101
# ─────────────────────────────────────────────────────────────────────────────

@_wq(91, ("cross_sectional", "industry_neutral"),
     "行业中性化 close 与成交量相关性两次衰减后的时序排名,减 VWAP 与成交额相关性衰减秩,再取负。")
class Alpha091(WqAlpha):
    def compute(self, panel):
        inner = ops.correlation(_indneutralize(panel["close"]), panel["volume"], 10)
        a = ops.ts_rank(ops.decay_linear(ops.decay_linear(inner, 16), 4), 5)
        adv30 = _adv(panel, 30)
        b = ops.rank(ops.decay_linear(ops.correlation(_vwap(panel), adv30, 4), 3))
        return -1.0 * (a - b)


@_wq(92, ("cross_sectional",),
     "中价加 close 是否低于低开之和的衰减时序排名,与低价秩同成交额秩相关性衰减的时序排名,取较小值。")
class Alpha092(WqAlpha):
    def compute(self, panel):
        cond = (((panel["high"] + panel["low"]) / 2.0 + panel["close"])
                < (panel["low"] + panel["open"])).astype(float)
        a = ops.ts_rank(ops.decay_linear(cond, 15), 19)
        adv30 = _adv(panel, 30)
        b = ops.ts_rank(ops.decay_linear(ops.correlation(ops.rank(panel["low"]), ops.rank(adv30), 8), 7), 7)
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(93, ("cross_sectional", "industry_neutral"),
     "行业中性化 VWAP 与成交额相关性的衰减时序排名,除以收 VWAP 加权价 3 日动量秩衰减,趋势与动量比值因子。")
class Alpha093(WqAlpha):
    def compute(self, panel):
        adv81 = _adv(panel, 81)
        a = ops.ts_rank(ops.decay_linear(ops.correlation(_indneutralize(_vwap(panel)), adv81, 17), 20), 8)
        proxy = panel["close"] * 0.524434 + _vwap(panel) * (1 - 0.524434)
        b = ops.rank(ops.decay_linear(ops.delta(proxy, 3), 16))
        return a / b


@_wq(94, ("cross_sectional",),
     "VWAP 偏离 12 日低点的秩,以 VWAP 与成交额时序排名相关性的时序排名为指数做符号幂,再取负,反转因子。")
class Alpha094(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(vw - ops.ts_min(vw, 12))
        adv60 = _adv(panel, 60)
        b = ops.ts_rank(ops.correlation(ops.ts_rank(vw, 20), ops.ts_rank(adv60, 4), 18), 3)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@_wq(95, ("cross_sectional",),
     "open 偏离 12 日低点的秩,小于中价累计与成交额累计相关性五次幂的时序排名时输出 1,横截面反转触发因子。")
class Alpha095(WqAlpha):
    def compute(self, panel):
        a = ops.rank(panel["open"] - ops.ts_min(panel["open"], 12))
        hl = (panel["high"] + panel["low"]) / 2.0
        adv40 = _adv(panel, 40)
        inner = ops.rank(ops.correlation(ops.ts_sum(hl, 19), ops.ts_sum(adv40, 19), 13)) ** 5
        b = ops.ts_rank(inner, 12)
        return (a < b).astype(float)


@_wq(96, ("cross_sectional",),
     "VWAP 与成交量秩相关性的双层衰减时序排名,与 close 同成交额相关性极值位置衰减的时序排名,取较大值再取负。")
class Alpha096(WqAlpha):
    def compute(self, panel):
        a = ops.ts_rank(ops.decay_linear(ops.correlation(ops.rank(_vwap(panel)),
                                                          ops.rank(panel["volume"]), 4), 4), 8)
        adv60 = _adv(panel, 60)
        inner = ops.ts_argmax(ops.correlation(ops.ts_rank(panel["close"], 7), ops.ts_rank(adv60, 4), 4), 13)
        b = ops.ts_rank(ops.decay_linear(inner, 14), 13)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(97, ("cross_sectional", "industry_neutral"),
     "行业中性化的低价 VWAP 加权 3 日动量秩衰减,减去低价与成交额嵌套时序排名相关性的多层衰减,再取负。")
class Alpha097(WqAlpha):
    def compute(self, panel):
        proxy = panel["low"] * 0.721001 + _vwap(panel) * (1 - 0.721001)
        a = ops.rank(ops.decay_linear(ops.delta(_indneutralize(proxy), 3), 20))
        adv60 = _adv(panel, 60)
        inner = ops.ts_rank(ops.correlation(ops.ts_rank(panel["low"], 8), ops.ts_rank(adv60, 17), 5), 19)
        b = ops.ts_rank(ops.decay_linear(inner, 16), 7)
        return -1.0 * (a - b)


@_wq(98, ("cross_sectional",),
     "VWAP 与短期成交额累计相关性的衰减秩,减去开盘秩与成交额秩相关性极小值位置的衰减秩,量价合成因子。")
class Alpha098(WqAlpha):
    def compute(self, panel):
        adv5 = _adv(panel, 5)
        a = ops.rank(ops.decay_linear(ops.correlation(_vwap(panel), ops.ts_sum(adv5, 26), 5), 7))
        adv15 = _adv(panel, 15)
        inner = ops.ts_argmin(ops.correlation(ops.rank(panel["open"]), ops.rank(adv15), 21), 9)
        b = ops.rank(ops.decay_linear(ops.ts_rank(inner, 7), 8))
        return a - b


@_wq(99, ("cross_sectional",),
     "中价累计与成交额累计相关性秩小于低价同成交量相关性秩时输出负 1,典型量价相关触发型反转因子。")
class Alpha099(WqAlpha):
    def compute(self, panel):
        hl = (panel["high"] + panel["low"]) / 2.0
        adv60 = _adv(panel, 60)
        a = ops.rank(ops.correlation(ops.ts_sum(hl, 20), ops.ts_sum(adv60, 20), 9))
        b = ops.rank(ops.correlation(panel["low"], panel["volume"], 6))
        return (a < b).astype(float) * -1.0


@_wq(100, ("cross_sectional", "industry_neutral"),
     "((close-low)-(high-close))/(high-low)*volume 二次行业中性化标准化,减去 close 与成交额相关性等的中性化标准化,再按相对成交量缩放,Money Flow 风格的复杂合成。")
class Alpha100(WqAlpha):
    def compute(self, panel):
        c, h, l = panel["close"], panel["high"], panel["low"]
        rng = (h - l).replace(0.0, np.nan)
        x = ((c - l) - (h - c)) / rng * panel["volume"]
        a = 1.5 * ops.scale(_indneutralize(_indneutralize(ops.rank(x))))
        adv20 = _adv(panel, 20)
        inner = ops.correlation(c, ops.rank(adv20), 5) - ops.rank(ops.ts_argmin(c, 30))
        b = ops.scale(_indneutralize(inner))
        return -1.0 * (a - b) * (panel["volume"] / adv20)


@_wq(101, ("cross_sectional",),
     "当日涨跌幅相对于日内振幅的比例,等价于 K 线实体占总波幅比重,简洁直观的日内强弱因子。")
class Alpha101(WqAlpha):
    def compute(self, panel):
        return (panel["close"] - panel["open"]) / ((panel["high"] - panel["low"]) + 0.001)
