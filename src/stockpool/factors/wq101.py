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
     "rank(Ts_ArgMax(SignedPower((returns<0)?stddev(returns,20):close, 2), 5)) - 0.5")
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
     "-1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)")
class Alpha002(WqAlpha):
    def compute(self, panel):
        log_v = np.log(panel["volume"].replace(0, np.nan))
        a = ops.rank(ops.delta(log_v, 2))
        b = ops.rank((panel["close"] - panel["open"]) / panel["open"])
        return -1.0 * ops.correlation(a, b, 6)


@_wq(3, ("cross_sectional", "volume"),
     "-1 * correlation(rank(open), rank(volume), 10)")
class Alpha003(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.correlation(ops.rank(panel["open"]), ops.rank(panel["volume"]), 10)


@_wq(4, ("cross_sectional", "reversal"),
     "-1 * Ts_Rank(rank(low), 9)")
class Alpha004(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.ts_rank(ops.rank(panel["low"]), 9)


@_wq(5, ("cross_sectional",),
     "rank(open - sum(vwap,10)/10) * (-1 * abs(rank(close - vwap)))")
class Alpha005(WqAlpha):
    def compute(self, panel):
        vwap = _vwap(panel)
        a = ops.rank(panel["open"] - ops.ts_sum(vwap, 10) / 10.0)
        b = -1.0 * ops.rank(panel["close"] - vwap).abs()
        return a * b


@_wq(6, ("cross_sectional", "volume"),
     "-1 * correlation(open, volume, 10)")
class Alpha006(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.correlation(panel["open"], panel["volume"], 10)


@_wq(7, ("cross_sectional",),
     "adv20<volume ? -1*ts_rank(abs(delta(close,7)),60)*sign(delta(close,7)) : -1")
class Alpha007(WqAlpha):
    def compute(self, panel):
        adv20 = _adv(panel, 20)
        d7 = ops.delta(panel["close"], 7)
        a = -1.0 * ops.ts_rank(d7.abs(), 60) * np.sign(d7)
        # 默认 -1,条件成立用 a
        out = pd.DataFrame(-1.0, index=panel["close"].index, columns=panel["close"].columns)
        return out.where(~(adv20 < panel["volume"]), a)


@_wq(8, ("cross_sectional", "momentum"),
     "-1 * rank((sum(open,5)*sum(returns,5)) - delay(sum(open,5)*sum(returns,5), 10))")
class Alpha008(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        prod = ops.ts_sum(panel["open"], 5) * ops.ts_sum(ret, 5)
        return -1.0 * ops.rank(prod - ops.delay(prod, 10))


@_wq(9, ("time_series",),
     "((0<ts_min(delta(close,1),5))?delta(close,1):((ts_max(delta(close,1),5)<0)?delta(close,1):-1*delta(close,1)))")
class Alpha009(WqAlpha):
    def compute(self, panel):
        d = ops.delta(panel["close"], 1)
        out = -1.0 * d
        out = out.where(~(ops.ts_max(d, 5) < 0), d)
        out = out.where(~(ops.ts_min(d, 5) > 0), d)
        return out


@_wq(10, ("cross_sectional", "time_series"),
     "rank(((0<ts_min(delta(close,1),4))?delta(close,1):((ts_max(delta(close,1),4)<0)?delta(close,1):-1*delta(close,1))))")
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
     "(rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))")
class Alpha011(WqAlpha):
    def compute(self, panel):
        d = _vwap(panel) - panel["close"]
        return (ops.rank(ops.ts_max(d, 3)) + ops.rank(ops.ts_min(d, 3))) * ops.rank(ops.delta(panel["volume"], 3))


@_wq(12, ("cross_sectional", "volume"),
     "sign(delta(volume,1)) * (-1 * delta(close,1))")
class Alpha012(WqAlpha):
    def compute(self, panel):
        return np.sign(ops.delta(panel["volume"], 1)) * (-1.0 * ops.delta(panel["close"], 1))


@_wq(13, ("cross_sectional", "volume"),
     "-1 * rank(covariance(rank(close), rank(volume), 5))")
class Alpha013(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.covariance(ops.rank(panel["close"]), ops.rank(panel["volume"]), 5))


@_wq(14, ("cross_sectional", "volume"),
     "(-1 * rank(delta(returns,3))) * correlation(open, volume, 10)")
class Alpha014(WqAlpha):
    def compute(self, panel):
        return (-1.0 * ops.rank(ops.delta(_ret(panel), 3))) * ops.correlation(panel["open"], panel["volume"], 10)


@_wq(15, ("cross_sectional", "volume"),
     "-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)")
class Alpha015(WqAlpha):
    def compute(self, panel):
        c = ops.correlation(ops.rank(panel["high"]), ops.rank(panel["volume"]), 3)
        return -1.0 * ops.ts_sum(ops.rank(c), 3)


@_wq(16, ("cross_sectional", "volume"),
     "-1 * rank(covariance(rank(high), rank(volume), 5))")
class Alpha016(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.covariance(ops.rank(panel["high"]), ops.rank(panel["volume"]), 5))


@_wq(17, ("cross_sectional",),
     "((-1*rank(ts_rank(close,10))) * rank(delta(delta(close,1),1))) * rank(ts_rank(volume/adv20,5))")
class Alpha017(WqAlpha):
    def compute(self, panel):
        a = -1.0 * ops.rank(ops.ts_rank(panel["close"], 10))
        b = ops.rank(ops.delta(ops.delta(panel["close"], 1), 1))
        adv20 = _adv(panel, 20)
        c = ops.rank(ops.ts_rank(panel["volume"] / adv20, 5))
        return a * b * c


@_wq(18, ("cross_sectional",),
     "-1*rank(stddev(abs(close-open),5) + (close-open) + correlation(close,open,10))")
class Alpha018(WqAlpha):
    def compute(self, panel):
        diff = panel["close"] - panel["open"]
        inside = ops.ts_std(diff.abs(), 5) + diff + ops.correlation(panel["close"], panel["open"], 10)
        return -1.0 * ops.rank(inside)


@_wq(19, ("cross_sectional", "momentum"),
     "(-1*sign(close-delay(close,7)+delta(close,7))) * (1+rank(1+sum(returns,250)))")
class Alpha019(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        a = -1.0 * np.sign((c - ops.delay(c, 7)) + ops.delta(c, 7))
        ret = _ret(panel)
        b = 1.0 + ops.rank(1.0 + ops.ts_sum(ret, 250))
        return a * b


@_wq(20, ("cross_sectional", "reversal"),
     "((-1*rank(open-delay(high,1))) * rank(open-delay(close,1))) * rank(open-delay(low,1))")
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
     "if ((sum(close,8)/8+stddev(close,8))<(sum(close,2)/2)) then -1; elif ((sum(close,2)/2)<(sum(close,8)/8-stddev(close,8))) then 1; elif (1<=(volume/adv20)) then 1 else -1")
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
     "-1 * (delta(correlation(high,volume,5),5) * rank(stddev(close,20)))")
class Alpha022(WqAlpha):
    def compute(self, panel):
        a = ops.delta(ops.correlation(panel["high"], panel["volume"], 5), 5)
        b = ops.rank(ops.ts_std(panel["close"], 20))
        return -1.0 * a * b


@_wq(23, ("time_series", "reversal"),
     "((sum(high,20)/20)<high) ? -1*delta(high,2) : 0")
class Alpha023(WqAlpha):
    def compute(self, panel):
        h = panel["high"]
        cond = (ops.ts_sum(h, 20) / 20.0) < h
        out = -1.0 * ops.delta(h, 2)
        return out.where(cond, 0.0)


@_wq(24, ("time_series", "reversal"),
     "if delta(sum(close,100)/100,100)/delay(close,100)<=0.05 then -1*(close-ts_min(close,100)) else -1*delta(close,3)")
class Alpha024(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        ma100 = ops.ts_sum(c, 100) / 100.0
        cond = ops.delta(ma100, 100) / ops.delay(c, 100) <= 0.05
        a = -1.0 * (c - ops.ts_min(c, 100))
        b = -1.0 * ops.delta(c, 3)
        return a.where(cond, b)


@_wq(25, ("cross_sectional", "volume"),
     "rank(((-1*returns)*adv20*vwap)*(high-close))")
class Alpha025(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        return ops.rank(((-1.0 * ret) * _adv(panel, 20) * _vwap(panel)) * (panel["high"] - panel["close"]))


@_wq(26, ("cross_sectional", "volume"),
     "-1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)")
class Alpha026(WqAlpha):
    def compute(self, panel):
        c = ops.correlation(ops.ts_rank(panel["volume"], 5), ops.ts_rank(panel["high"], 5), 5)
        return -1.0 * ops.ts_max(c, 3)


@_wq(27, ("cross_sectional", "volume"),
     "(0.5<rank(sum(correlation(rank(volume),rank(vwap),6),2)/2)) ? -1 : 1")
class Alpha027(WqAlpha):
    def compute(self, panel):
        c = ops.correlation(ops.rank(panel["volume"]), ops.rank(_vwap(panel)), 6)
        r = ops.rank(ops.ts_sum(c, 2) / 2.0)
        out = pd.DataFrame(1.0, index=r.index, columns=r.columns)
        return out.where(~(r > 0.5), -1.0)


@_wq(28, ("cross_sectional",),
     "scale(correlation(adv20, low, 5) + (high+low)/2 - close)")
class Alpha028(WqAlpha):
    def compute(self, panel):
        return ops.scale(
            ops.correlation(_adv(panel, 20), panel["low"], 5)
            + (panel["high"] + panel["low"]) / 2.0
            - panel["close"]
        )


@_wq(29, ("cross_sectional",),
     "min(product(rank(rank(scale(log(sum(ts_min(rank(rank(-1*rank(delta(close-1,5)))),2),1))))),1),5) + ts_rank(delay(-1*returns,6),5)")
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
     "((1.0-rank(sign(close-delay(close,1)) + sign(delay(close,1)-delay(close,2)) + sign(delay(close,2)-delay(close,3)))) * sum(volume,5)) / sum(volume,20)")
class Alpha030(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        s = np.sign(c - ops.delay(c, 1)) + np.sign(ops.delay(c, 1) - ops.delay(c, 2)) + np.sign(ops.delay(c, 2) - ops.delay(c, 3))
        return ((1.0 - ops.rank(s)) * ops.ts_sum(panel["volume"], 5)) / ops.ts_sum(panel["volume"], 20)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 031-040
# ─────────────────────────────────────────────────────────────────────────────

@_wq(31, ("cross_sectional",),
     "rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10)))) + rank(-1*delta(close,3)) + sign(scale(correlation(adv20,low,12)))")
class Alpha031(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        a = ops.rank(ops.rank(ops.rank(ops.decay_linear(-1.0 * ops.rank(ops.rank(ops.delta(c, 10))), 10))))
        b = ops.rank(-1.0 * ops.delta(c, 3))
        d = np.sign(ops.scale(ops.correlation(_adv(panel, 20), panel["low"], 12)))
        return a + b + d


@_wq(32, ("cross_sectional",),
     "scale(sum(close,7)/7 - close) + 20*scale(correlation(vwap, delay(close,5), 230))")
class Alpha032(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        a = ops.scale(ops.ts_sum(c, 7) / 7.0 - c)
        b = 20.0 * ops.scale(ops.correlation(_vwap(panel), ops.delay(c, 5), 230))
        return a + b


@_wq(33, ("cross_sectional",),
     "rank(-1 * ((1 - open/close)^1))")
class Alpha033(WqAlpha):
    def compute(self, panel):
        return ops.rank(-1.0 * (1.0 - panel["open"] / panel["close"]))


@_wq(34, ("cross_sectional",),
     "rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))")
class Alpha034(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        a = 1.0 - ops.rank(ops.ts_std(ret, 2) / ops.ts_std(ret, 5))
        b = 1.0 - ops.rank(ops.delta(panel["close"], 1))
        return ops.rank(a + b)


@_wq(35, ("time_series",),
     "ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))")
class Alpha035(WqAlpha):
    def compute(self, panel):
        a = ops.ts_rank(panel["volume"], 32)
        b = 1.0 - ops.ts_rank(panel["close"] + panel["high"] - panel["low"], 16)
        c = 1.0 - ops.ts_rank(_ret(panel), 32)
        return a * b * c


@_wq(36, ("cross_sectional", "volume"),
     "2.21*rank(corr(close-open, delay(volume,1), 15)) + 0.7*rank(open-close) + 0.73*rank(ts_rank(delay(-1*returns,6),5)) + rank(abs(corr(vwap,adv20,6))) + 0.6*rank((sum(close,200)/200-open)*(close-open))")
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
     "rank(correlation(delay(open-close,1), close, 200)) + rank(open - close)")
class Alpha037(WqAlpha):
    def compute(self, panel):
        c, o = panel["close"], panel["open"]
        return ops.rank(ops.correlation(ops.delay(o - c, 1), c, 200)) + ops.rank(o - c)


@_wq(38, ("cross_sectional", "reversal"),
     "-1 * rank(ts_rank(close,10)) * rank(close/open)")
class Alpha038(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_rank(panel["close"], 10)) * ops.rank(panel["close"] / panel["open"])


@_wq(39, ("cross_sectional", "momentum"),
     "(-1 * rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9))))) * (1 + rank(sum(returns,250)))")
class Alpha039(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        ret = _ret(panel)
        dvol = panel["volume"] / _adv(panel, 20)
        a = -1.0 * ops.rank(ops.delta(c, 7) * (1.0 - ops.rank(ops.decay_linear(dvol, 9))))
        b = 1.0 + ops.rank(ops.ts_sum(ret, 250))
        return a * b


@_wq(40, ("cross_sectional", "volatility"),
     "-1 * rank(stddev(high,10)) * correlation(high, volume, 10)")
class Alpha040(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_std(panel["high"], 10)) * ops.correlation(panel["high"], panel["volume"], 10)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 041-050
# ─────────────────────────────────────────────────────────────────────────────

@_wq(41, ("cross_sectional",),
     "(high * low)^0.5 - vwap")
class Alpha041(WqAlpha):
    def compute(self, panel):
        return (panel["high"] * panel["low"]) ** 0.5 - _vwap(panel)


@_wq(42, ("cross_sectional",),
     "rank(vwap - close) / rank(vwap + close)")
class Alpha042(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        return ops.rank(vw - panel["close"]) / ops.rank(vw + panel["close"])


@_wq(43, ("time_series",),
     "ts_rank(volume/adv20, 20) * ts_rank(-1*delta(close,7), 8)")
class Alpha043(WqAlpha):
    def compute(self, panel):
        return ops.ts_rank(panel["volume"] / _adv(panel, 20), 20) * ops.ts_rank(-1.0 * ops.delta(panel["close"], 7), 8)


@_wq(44, ("cross_sectional", "volume"),
     "-1 * correlation(high, rank(volume), 5)")
class Alpha044(WqAlpha):
    def compute(self, panel):
        return -1.0 * ops.correlation(panel["high"], ops.rank(panel["volume"]), 5)


@_wq(45, ("cross_sectional",),
     "-1 * (rank(sum(delay(close,5),20)/20) * correlation(close,volume,2) * rank(correlation(sum(close,5),sum(close,20),2)))")
class Alpha045(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        a = ops.rank(ops.ts_sum(ops.delay(c, 5), 20) / 20.0)
        b = ops.correlation(c, panel["volume"], 2)
        d = ops.rank(ops.correlation(ops.ts_sum(c, 5), ops.ts_sum(c, 20), 2))
        return -1.0 * a * b * d


@_wq(46, ("time_series", "reversal"),
     "if 0.25<((delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10) then -1; elif (...)<0 then 1; else -1*(close - delay(close,1))")
class Alpha046(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        diff = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 1))
        out = out.where(~(diff < 0), 1.0)
        out = out.where(~(diff > 0.25), -1.0)
        return out


@_wq(47, ("cross_sectional", "volume"),
     "((rank(1/close)*volume)/adv20) * ((high*rank(high-close))/(sum(high,5)/5)) - rank(vwap - delay(vwap,5))")
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
     "indneutralize(((correlation(delta(close,1), delta(delay(close,1),1), 250)*delta(close,1))/close), IndClass.subindustry) / sum((delta(close,1)/delay(close,1))^2, 250)")
class Alpha048(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        d1 = ops.delta(c, 1)
        corr = ops.correlation(d1, ops.delta(ops.delay(c, 1), 1), 250)
        numer = _indneutralize(corr * d1 / c)
        denom = ops.ts_sum((d1 / ops.delay(c, 1)) ** 2, 250)
        return numer / denom


@_wq(49, ("time_series",),
     "if delta(delay(close,10),10)/10 - delta(close,10)/10 < -0.1 then 1 else -1*(close - delay(close,1))")
class Alpha049(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        diff = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 1))
        return out.where(~(diff < -0.1), 1.0)


@_wq(50, ("cross_sectional", "volume"),
     "-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)")
class Alpha050(WqAlpha):
    def compute(self, panel):
        c = ops.correlation(ops.rank(panel["volume"]), ops.rank(_vwap(panel)), 5)
        return -1.0 * ops.ts_max(ops.rank(c), 5)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha 051-060
# ─────────────────────────────────────────────────────────────────────────────

@_wq(51, ("time_series",),
     "if delta(delay(close,10),10)/10 - delta(close,10)/10 < -0.05 then 1 else -1*(close - delay(close,1))")
class Alpha051(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        diff = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 1))
        return out.where(~(diff < -0.05), 1.0)


@_wq(52, ("cross_sectional",),
     "(-1*ts_min(low,5)+delay(ts_min(low,5),5)) * rank((sum(returns,240)-sum(returns,20))/220) * ts_rank(volume,5)")
class Alpha052(WqAlpha):
    def compute(self, panel):
        ret = _ret(panel)
        a = -1.0 * ops.ts_min(panel["low"], 5) + ops.delay(ops.ts_min(panel["low"], 5), 5)
        b = ops.rank((ops.ts_sum(ret, 240) - ops.ts_sum(ret, 20)) / 220.0)
        d = ops.ts_rank(panel["volume"], 5)
        return a * b * d


@_wq(53, ("time_series",),
     "-1 * delta(((close-low)-(high-close))/(close-low), 9)")
class Alpha053(WqAlpha):
    def compute(self, panel):
        c, h, l = panel["close"], panel["high"], panel["low"]
        denom = (c - l).replace(0.0, np.nan)
        x = ((c - l) - (h - c)) / denom
        return -1.0 * ops.delta(x, 9)


@_wq(54, ("cross_sectional",),
     "(-1 * ((low-close)*(open^5))) / ((low-high)*(close^5))")
class Alpha054(WqAlpha):
    def compute(self, panel):
        c, o, h, l = panel["close"], panel["open"], panel["high"], panel["low"]
        num = -1.0 * ((l - c) * (o ** 5))
        den = ((l - h) * (c ** 5)).replace(0.0, np.nan)
        return num / den


@_wq(55, ("cross_sectional", "volume"),
     "-1 * correlation(rank((close - ts_min(low,12))/(ts_max(high,12) - ts_min(low,12))), rank(volume), 6)")
class Alpha055(WqAlpha):
    def compute(self, panel):
        c, h, l = panel["close"], panel["high"], panel["low"]
        lo12 = ops.ts_min(l, 12)
        hi12 = ops.ts_max(h, 12)
        rng = (hi12 - lo12).replace(0.0, np.nan)
        r = (c - lo12) / rng
        return -1.0 * ops.correlation(ops.rank(r), ops.rank(panel["volume"]), 6)


@_wq(56, ("cross_sectional", "industry_neutral"),
     "0 - (1 * (rank(sum(returns,10)/sum(sum(returns,2),3)) * rank(returns*cap)))  // 需 cap (未实现)")
class Alpha056(WqAlpha):
    def compute(self, panel):
        # 项目无 cap 字段;返回 NaN
        return _nan_like(panel)


@_wq(57, ("cross_sectional",),
     "0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close,30)),2)))")
class Alpha057(WqAlpha):
    def compute(self, panel):
        c = panel["close"]
        return -1.0 * ((c - _vwap(panel)) / ops.decay_linear(ops.rank(ops.ts_argmax(c, 30)), 2))


@_wq(58, ("cross_sectional", "industry_neutral"),
     "-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322)")
class Alpha058(WqAlpha):
    def compute(self, panel):
        vw_n = _indneutralize(_vwap(panel))
        c = ops.correlation(vw_n, panel["volume"], 4)
        dl = ops.decay_linear(c, 8)
        return -1.0 * ops.ts_rank(dl, 6)


@_wq(59, ("cross_sectional", "industry_neutral"),
     "-1 * Ts_Rank(decay_linear(correlation(IndNeutralize((vwap*0.728317)+(vwap*(1-0.728317)), IndClass.industry), volume, 4.25197), 16.2289), 8.19648)")
class Alpha059(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        weighted = vw * 0.728317 + vw * (1 - 0.728317)  # 简化为 vwap
        c = ops.correlation(_indneutralize(weighted), panel["volume"], 4)
        return -1.0 * ops.ts_rank(ops.decay_linear(c, 16), 8)


@_wq(60, ("cross_sectional",),
     "0 - (1 * ((2*scale(rank(((close-low)-(high-close))/(high-low)*volume))) - scale(rank(ts_argmax(close,10)))))")
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
     "rank(vwap - ts_min(vwap,16.1219)) < rank(correlation(vwap, adv180, 17.9282))")
class Alpha061(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(vw - ops.ts_min(vw, 16))
        b = ops.rank(ops.correlation(vw, _adv(panel, 180), 18))
        return (a < b).astype(float)


@_wq(62, ("cross_sectional",),
     "rank(correlation(vwap, sum(adv20,22), 10)) < rank(((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high))))")
class Alpha062(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        adv20 = _adv(panel, 20)
        a = ops.rank(ops.correlation(vw, ops.ts_sum(adv20, 22), 10))
        b = ops.rank(((ops.rank(panel["open"]) + ops.rank(panel["open"]))
                      < (ops.rank((panel["high"] + panel["low"]) / 2.0) + ops.rank(panel["high"]))).astype(float))
        return (a < b).astype(float) * -1.0


@_wq(63, ("cross_sectional", "industry_neutral"),
     "需 IndClass.industry,退化到 sector")
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
     "rank(corr(sum((open*0.178404)+(low*(1-0.178404)),12), sum(adv120,12), 16)) < rank(delta((((high+low)/2)*0.178404)+(vwap*(1-0.178404)),3))")
class Alpha064(WqAlpha):
    def compute(self, panel):
        adv120 = _adv(panel, 120)
        a_term = panel["open"] * 0.178404 + panel["low"] * (1 - 0.178404)
        a = ops.rank(ops.correlation(ops.ts_sum(a_term, 12), ops.ts_sum(adv120, 12), 16))
        b_term = (panel["high"] + panel["low"]) / 2.0 * 0.178404 + _vwap(panel) * (1 - 0.178404)
        b = ops.rank(ops.delta(b_term, 3))
        return (a < b).astype(float) * -1.0


@_wq(65, ("cross_sectional",),
     "rank(corr((open*0.00817205)+(vwap*(1-0.00817205)), sum(adv60,8), 6)) < rank(open - ts_min(open,13))")
class Alpha065(WqAlpha):
    def compute(self, panel):
        adv60 = _adv(panel, 60)
        a_term = panel["open"] * 0.00817205 + _vwap(panel) * (1 - 0.00817205)
        a = ops.rank(ops.correlation(a_term, ops.ts_sum(adv60, 8), 6))
        b = ops.rank(panel["open"] - ops.ts_min(panel["open"], 13))
        return (a < b).astype(float) * -1.0


@_wq(66, ("cross_sectional",),
     "(rank(decay_linear(delta(vwap,3),7)) + Ts_Rank(decay_linear(((low*0.96633)+(low*(1-0.96633))-vwap)/(open-(high+low)/2), 11), 6)) * -1")
class Alpha066(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 3), 7))
        denom = (panel["open"] - (panel["high"] + panel["low"]) / 2.0).replace(0.0, np.nan)
        inner = (panel["low"] - vw) / denom
        b = ops.ts_rank(ops.decay_linear(inner, 11), 6)
        return -1.0 * (a + b)


@_wq(67, ("cross_sectional", "industry_neutral"),
     "(rank(high - ts_min(high,2))^rank(corr(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20,IndClass.subindustry), 6))) * -1")
class Alpha067(WqAlpha):
    def compute(self, panel):
        a = ops.rank(panel["high"] - ops.ts_min(panel["high"], 2))
        b = ops.rank(ops.correlation(_indneutralize(_vwap(panel)),
                                     _indneutralize(_adv(panel, 20)), 6))
        # WQ 公式里的 ^ 是 power;为数值稳定限制底数为非负
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@_wq(68, ("cross_sectional",),
     "(Ts_Rank(corr(rank(high), rank(adv15), 9), 14) < rank(delta((close*0.518371)+(low*(1-0.518371)),1))) * -1")
class Alpha068(WqAlpha):
    def compute(self, panel):
        adv15 = _adv(panel, 15)
        a = ops.ts_rank(ops.correlation(ops.rank(panel["high"]), ops.rank(adv15), 9), 14)
        b_term = panel["close"] * 0.518371 + panel["low"] * (1 - 0.518371)
        b = ops.rank(ops.delta(b_term, 1))
        return (a < b).astype(float) * -1.0


@_wq(69, ("cross_sectional", "industry_neutral"),
     "需 IndClass.industry,退化到 sector")
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
     "(rank(delta(vwap,1))^Ts_Rank(corr(IndNeutralize(close,IndClass.industry), adv50, 18), 18)) * -1")
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
     "max(Ts_Rank(decay_linear(corr(Ts_Rank(close,3), Ts_Rank(adv180,12), 18), 4), 15), Ts_Rank(decay_linear((rank((low+open)-(vwap+vwap)))^2, 16), 4))")
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
     "rank(decay_linear(corr((high+low)/2, adv40, 9), 10)) / rank(decay_linear(corr(Ts_Rank(vwap,4), Ts_Rank(volume,19), 7), 3))")
class Alpha072(WqAlpha):
    def compute(self, panel):
        hl = (panel["high"] + panel["low"]) / 2.0
        adv40 = _adv(panel, 40)
        a = ops.rank(ops.decay_linear(ops.correlation(hl, adv40, 9), 10))
        b = ops.rank(ops.decay_linear(ops.correlation(ops.ts_rank(_vwap(panel), 4),
                                                      ops.ts_rank(panel["volume"], 19), 7), 3))
        return a / b


@_wq(73, ("cross_sectional",),
     "max(rank(decay_linear(delta(vwap,5),3)), Ts_Rank(decay_linear((delta((open*0.147155+low*0.852845),2)/(open*0.147155+low*0.852845))*-1, 3), 17)) * -1")
class Alpha073(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 5), 3))
        proxy = panel["open"] * 0.147155 + panel["low"] * 0.852845
        inner = ops.delta(proxy, 2) / proxy.replace(0.0, np.nan) * -1.0
        b = ops.ts_rank(ops.decay_linear(inner, 3), 17)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(74, ("cross_sectional",),
     "(rank(corr(close, sum(adv30,37), 15)) < rank(corr(rank((high*0.0261661)+(vwap*(1-0.0261661))), rank(volume), 11))) * -1")
class Alpha074(WqAlpha):
    def compute(self, panel):
        adv30 = _adv(panel, 30)
        a = ops.rank(ops.correlation(panel["close"], ops.ts_sum(adv30, 37), 15))
        proxy = panel["high"] * 0.0261661 + _vwap(panel) * (1 - 0.0261661)
        b = ops.rank(ops.correlation(ops.rank(proxy), ops.rank(panel["volume"]), 11))
        return (a < b).astype(float) * -1.0


@_wq(75, ("cross_sectional",),
     "rank(corr(vwap, volume, 4)) < rank(corr(rank(low), rank(adv50), 12))")
class Alpha075(WqAlpha):
    def compute(self, panel):
        adv50 = _adv(panel, 50)
        a = ops.rank(ops.correlation(_vwap(panel), panel["volume"], 4))
        b = ops.rank(ops.correlation(ops.rank(panel["low"]), ops.rank(adv50), 12))
        return (a < b).astype(float)


@_wq(76, ("cross_sectional", "industry_neutral"),
     "max(rank(decay_linear(delta(vwap,1),12)), Ts_Rank(decay_linear(Ts_Rank(corr(IndNeutralize(low,IndClass.sector), adv81, 8), 20), 17), 19)) * -1")
class Alpha076(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 1), 12))
        adv81 = _adv(panel, 81)
        inner = ops.ts_rank(ops.correlation(_indneutralize(panel["low"]), adv81, 8), 20)
        b = ops.ts_rank(ops.decay_linear(inner, 17), 19)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(77, ("cross_sectional",),
     "min(rank(decay_linear((((high+low)/2)+high)-(vwap+high), 20)), rank(decay_linear(corr((high+low)/2, adv40, 3), 6)))")
class Alpha077(WqAlpha):
    def compute(self, panel):
        hl = (panel["high"] + panel["low"]) / 2.0
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear((hl + panel["high"]) - (vw + panel["high"]), 20))
        adv40 = _adv(panel, 40)
        b = ops.rank(ops.decay_linear(ops.correlation(hl, adv40, 3), 6))
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(78, ("cross_sectional",),
     "rank(corr(sum((low*0.352233)+(vwap*(1-0.352233)),20), sum(adv40,20), 7))^rank(corr(rank(vwap), rank(volume), 6))")
class Alpha078(WqAlpha):
    def compute(self, panel):
        adv40 = _adv(panel, 40)
        proxy = panel["low"] * 0.352233 + _vwap(panel) * (1 - 0.352233)
        a = ops.rank(ops.correlation(ops.ts_sum(proxy, 20), ops.ts_sum(adv40, 20), 7))
        b = ops.rank(ops.correlation(ops.rank(_vwap(panel)), ops.rank(panel["volume"]), 6))
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return ops.signedpower(a_safe, b)


@_wq(79, ("cross_sectional", "industry_neutral"),
     "(rank(delta(IndNeutralize((close*0.60733)+(open*(1-0.60733)), IndClass.sector), 1)) < rank(corr(Ts_Rank(vwap,4), Ts_Rank(adv150,9), 15)))")
class Alpha079(WqAlpha):
    def compute(self, panel):
        proxy = panel["close"] * 0.60733 + panel["open"] * (1 - 0.60733)
        a = ops.rank(ops.delta(_indneutralize(proxy), 1))
        adv150 = _adv(panel, 150)
        b = ops.rank(ops.correlation(ops.ts_rank(_vwap(panel), 4), ops.ts_rank(adv150, 9), 15))
        return (a < b).astype(float)


@_wq(80, ("cross_sectional", "industry_neutral"),
     "(rank(sign(delta(IndNeutralize((open*0.868128)+(high*(1-0.868128)), IndClass.industry), 4)))^Ts_Rank(corr(high, adv10, 5), 6)) * -1")
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
     "(rank(log(product(rank(rank(corr(vwap, sum(adv10,50), 8))^4),15))) < rank(corr(rank(vwap), rank(volume), 5))) * -1")
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
     "min(rank(decay_linear(delta(open,1),15)), Ts_Rank(decay_linear(corr(IndNeutralize(volume,IndClass.sector), (open*0.634196)+(open*(1-0.634196)), 17), 7), 13)) * -1")
class Alpha082(WqAlpha):
    def compute(self, panel):
        a = ops.rank(ops.decay_linear(ops.delta(panel["open"], 1), 15))
        proxy = panel["open"] * 0.634196 + panel["open"] * (1 - 0.634196)
        b = ops.ts_rank(ops.decay_linear(ops.correlation(_indneutralize(panel["volume"]), proxy, 17), 7), 13)
        return -1.0 * pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(83, ("cross_sectional",),
     "(rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume))) / (((high-low)/(sum(close,5)/5)) / (vwap-close))")
class Alpha083(WqAlpha):
    def compute(self, panel):
        c, h, l = panel["close"], panel["high"], panel["low"]
        vw = _vwap(panel)
        ratio = (h - l) / (ops.ts_sum(c, 5) / 5.0)
        a = ops.rank(ops.delay(ratio, 2)) * ops.rank(ops.rank(panel["volume"]))
        denom = (ratio / (vw - c).replace(0.0, np.nan))
        return a / denom


@_wq(84, ("cross_sectional",),
     "SignedPower(Ts_Rank(vwap-ts_max(vwap,15),21), delta(close,5))")
class Alpha084(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.ts_rank(vw - ops.ts_max(vw, 15), 21)
        b = ops.delta(panel["close"], 5)
        return ops.signedpower(a, b)


@_wq(85, ("cross_sectional",),
     "rank(corr((high*0.876703)+(close*(1-0.876703)), adv30, 10))^rank(corr(Ts_Rank((high+low)/2,4), Ts_Rank(volume,10), 7))")
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
     "(Ts_Rank(corr(close, sum(adv20,15), 6), 20) < rank((open+close)-(vwap+open))) * -1")
class Alpha086(WqAlpha):
    def compute(self, panel):
        adv20 = _adv(panel, 20)
        a = ops.ts_rank(ops.correlation(panel["close"], ops.ts_sum(adv20, 15), 6), 20)
        vw = _vwap(panel)
        b = ops.rank((panel["open"] + panel["close"]) - (vw + panel["open"]))
        return (a < b).astype(float) * -1.0


@_wq(87, ("cross_sectional", "industry_neutral"),
     "max(rank(decay_linear(delta((close*0.369701)+(vwap*(1-0.369701)),2),3)), Ts_Rank(decay_linear(abs(corr(IndNeutralize(adv81,IndClass.industry), close, 13)), 5), 14)) * -1")
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
     "min(rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)),8)), Ts_Rank(decay_linear(corr(Ts_Rank(close,8), Ts_Rank(adv60,21), 8), 7), 3))")
class Alpha088(WqAlpha):
    def compute(self, panel):
        a = ops.rank(ops.decay_linear((ops.rank(panel["open"]) + ops.rank(panel["low"]))
                                       - (ops.rank(panel["high"]) + ops.rank(panel["close"])), 8))
        adv60 = _adv(panel, 60)
        inner = ops.correlation(ops.ts_rank(panel["close"], 8), ops.ts_rank(adv60, 21), 8)
        b = ops.ts_rank(ops.decay_linear(inner, 7), 3)
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(89, ("cross_sectional", "industry_neutral"),
     "Ts_Rank(decay_linear(corr((low*0.967285)+(low*(1-0.967285)), adv10, 7), 6), 4) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3), 10), 15)")
class Alpha089(WqAlpha):
    def compute(self, panel):
        adv10 = _adv(panel, 10)
        proxy = panel["low"] * 0.967285 + panel["low"] * (1 - 0.967285)
        a = ops.ts_rank(ops.decay_linear(ops.correlation(proxy, adv10, 7), 6), 4)
        inner = ops.delta(_indneutralize(_vwap(panel)), 3)
        b = ops.ts_rank(ops.decay_linear(inner, 10), 15)
        return a - b


@_wq(90, ("cross_sectional", "industry_neutral"),
     "(rank(close - ts_max(close,5))^Ts_Rank(corr(IndNeutralize(adv40,IndClass.subindustry), low, 5), 3)) * -1")
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
     "(Ts_Rank(decay_linear(decay_linear(corr(IndNeutralize(close,IndClass.industry), volume, 10), 16), 4), 5) - rank(decay_linear(corr(vwap, adv30, 4), 3))) * -1")
class Alpha091(WqAlpha):
    def compute(self, panel):
        inner = ops.correlation(_indneutralize(panel["close"]), panel["volume"], 10)
        a = ops.ts_rank(ops.decay_linear(ops.decay_linear(inner, 16), 4), 5)
        adv30 = _adv(panel, 30)
        b = ops.rank(ops.decay_linear(ops.correlation(_vwap(panel), adv30, 4), 3))
        return -1.0 * (a - b)


@_wq(92, ("cross_sectional",),
     "min(Ts_Rank(decay_linear((((high+low)/2)+close) < (low+open), 15), 19), Ts_Rank(decay_linear(corr(rank(low), rank(adv30), 8), 7), 7))")
class Alpha092(WqAlpha):
    def compute(self, panel):
        cond = (((panel["high"] + panel["low"]) / 2.0 + panel["close"])
                < (panel["low"] + panel["open"])).astype(float)
        a = ops.ts_rank(ops.decay_linear(cond, 15), 19)
        adv30 = _adv(panel, 30)
        b = ops.ts_rank(ops.decay_linear(ops.correlation(ops.rank(panel["low"]), ops.rank(adv30), 8), 7), 7)
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(93, ("cross_sectional", "industry_neutral"),
     "Ts_Rank(decay_linear(corr(IndNeutralize(vwap,IndClass.industry), adv81, 17), 20), 8) / rank(decay_linear(delta((close*0.524434)+(vwap*(1-0.524434)),3), 16))")
class Alpha093(WqAlpha):
    def compute(self, panel):
        adv81 = _adv(panel, 81)
        a = ops.ts_rank(ops.decay_linear(ops.correlation(_indneutralize(_vwap(panel)), adv81, 17), 20), 8)
        proxy = panel["close"] * 0.524434 + _vwap(panel) * (1 - 0.524434)
        b = ops.rank(ops.decay_linear(ops.delta(proxy, 3), 16))
        return a / b


@_wq(94, ("cross_sectional",),
     "(rank(vwap - ts_min(vwap,12))^Ts_Rank(corr(Ts_Rank(vwap,20), Ts_Rank(adv60,4), 18), 3)) * -1")
class Alpha094(WqAlpha):
    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(vw - ops.ts_min(vw, 12))
        adv60 = _adv(panel, 60)
        b = ops.ts_rank(ops.correlation(ops.ts_rank(vw, 20), ops.ts_rank(adv60, 4), 18), 3)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@_wq(95, ("cross_sectional",),
     "rank(open - ts_min(open,12)) < Ts_Rank(rank(corr(sum((high+low)/2,19), sum(adv40,19), 13))^5, 12)")
class Alpha095(WqAlpha):
    def compute(self, panel):
        a = ops.rank(panel["open"] - ops.ts_min(panel["open"], 12))
        hl = (panel["high"] + panel["low"]) / 2.0
        adv40 = _adv(panel, 40)
        inner = ops.rank(ops.correlation(ops.ts_sum(hl, 19), ops.ts_sum(adv40, 19), 13)) ** 5
        b = ops.ts_rank(inner, 12)
        return (a < b).astype(float)


@_wq(96, ("cross_sectional",),
     "max(Ts_Rank(decay_linear(corr(rank(vwap), rank(volume), 4), 4), 8), Ts_Rank(decay_linear(Ts_ArgMax(corr(Ts_Rank(close,7), Ts_Rank(adv60,4), 4), 13), 14), 13)) * -1")
class Alpha096(WqAlpha):
    def compute(self, panel):
        a = ops.ts_rank(ops.decay_linear(ops.correlation(ops.rank(_vwap(panel)),
                                                          ops.rank(panel["volume"]), 4), 4), 8)
        adv60 = _adv(panel, 60)
        inner = ops.ts_argmax(ops.correlation(ops.ts_rank(panel["close"], 7), ops.ts_rank(adv60, 4), 4), 13)
        b = ops.ts_rank(ops.decay_linear(inner, 14), 13)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@_wq(97, ("cross_sectional", "industry_neutral"),
     "(rank(decay_linear(delta(IndNeutralize((low*0.721001)+(vwap*(1-0.721001)),IndClass.industry),3),20)) - Ts_Rank(decay_linear(Ts_Rank(corr(Ts_Rank(low,8), Ts_Rank(adv60,17), 5), 19), 16), 7)) * -1")
class Alpha097(WqAlpha):
    def compute(self, panel):
        proxy = panel["low"] * 0.721001 + _vwap(panel) * (1 - 0.721001)
        a = ops.rank(ops.decay_linear(ops.delta(_indneutralize(proxy), 3), 20))
        adv60 = _adv(panel, 60)
        inner = ops.ts_rank(ops.correlation(ops.ts_rank(panel["low"], 8), ops.ts_rank(adv60, 17), 5), 19)
        b = ops.ts_rank(ops.decay_linear(inner, 16), 7)
        return -1.0 * (a - b)


@_wq(98, ("cross_sectional",),
     "rank(decay_linear(corr(vwap, sum(adv5,26), 5), 7)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(corr(rank(open), rank(adv15), 21), 9), 7), 8))")
class Alpha098(WqAlpha):
    def compute(self, panel):
        adv5 = _adv(panel, 5)
        a = ops.rank(ops.decay_linear(ops.correlation(_vwap(panel), ops.ts_sum(adv5, 26), 5), 7))
        adv15 = _adv(panel, 15)
        inner = ops.ts_argmin(ops.correlation(ops.rank(panel["open"]), ops.rank(adv15), 21), 9)
        b = ops.rank(ops.decay_linear(ops.ts_rank(inner, 7), 8))
        return a - b


@_wq(99, ("cross_sectional",),
     "(rank(corr(sum((high+low)/2,20), sum(adv60,20), 9)) < rank(corr(low, volume, 6))) * -1")
class Alpha099(WqAlpha):
    def compute(self, panel):
        hl = (panel["high"] + panel["low"]) / 2.0
        adv60 = _adv(panel, 60)
        a = ops.rank(ops.correlation(ops.ts_sum(hl, 20), ops.ts_sum(adv60, 20), 9))
        b = ops.rank(ops.correlation(panel["low"], panel["volume"], 6))
        return (a < b).astype(float) * -1.0


@_wq(100, ("cross_sectional", "industry_neutral"),
     "0 - (1*(1.5*scale(IndNeutralize(IndNeutralize(rank(((close-low)-(high-close))/(high-low)*volume), IndClass.subindustry), IndClass.subindustry)) - scale(IndNeutralize(corr(close, rank(adv20), 5) - rank(ts_argmin(close,30)), IndClass.subindustry)))) * (volume/adv20)")
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
     "(close - open) / ((high - low) + 0.001)")
class Alpha101(WqAlpha):
    def compute(self, panel):
        return (panel["close"] - panel["open"]) / ((panel["high"] - panel["low"]) + 0.001)
