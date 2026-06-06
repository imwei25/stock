"""复合补充族 (论文 B add_* 30 个的精神复现).

用现有 ops 拼装的混合信号: rank * sign / decay_linear / scale 等。
4 个 base × 3 窗口 = ~12 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "rank_signed_mom",
    sources=("builtin",),
    types=("cross_sectional", "momentum", "time_series"),
    description="横截面动量秩 × 量变方向。所有股票按 N 日涨幅排名,再乘以“量是放还是缩”,同时筛“涨且放量”的股。",
)
class RankSignedMomFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"rank_signed_mom_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mom = panel["close"].pct_change(self.n, fill_method=None)
        vol_chg = panel["volume"].pct_change(self.n, fill_method=None)
        return ops.rank(mom) * np.sign(vol_chg)


@register(
    "decay_corr_pv",
    sources=("builtin",),
    types=("cross_sectional", "volume", "time_series"),
    description="价格秩与量秩相关性的线性衰减加权(近期权重更大),刻画“近期量价是否一致变动”。",
)
class DecayCorrPVFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"decay_corr_pv_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        r_c = ops.rank(panel["close"])
        r_v = ops.rank(panel["volume"])
        corr = ops.correlation(r_c, r_v, self.n)
        return ops.decay_linear(corr, self.n)


@register(
    "scale_decay_mom",
    sources=("builtin",),
    types=("cross_sectional", "momentum", "time_series"),
    description="动量先 N 日线性衰减加权再做截面归一化,平滑短期噪音,适合做横截面排序选股。",
)
class ScaleDecayMomFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"scale_decay_mom_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mom = panel["close"].pct_change(self.n, fill_method=None)
        return ops.scale(ops.decay_linear(mom, self.n))


@register(
    "mom_vol_interact",
    sources=("builtin",),
    types=("momentum", "volume", "time_series"),
    description="动量 × 放量比的乘积。两者同号(涨+放量 或 跌+缩量)时数值最大,纯涨或纯放都不算极强。",
)
class MomVolInteractFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"mom_vol_interact_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mom = panel["close"].pct_change(self.n, fill_method=None)
        v = panel["volume"].replace(0.0, np.nan)
        v_ratio = v / v.rolling(self.n).mean().shift(1) - 1.0
        return mom * v_ratio
