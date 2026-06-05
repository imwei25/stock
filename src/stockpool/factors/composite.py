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
    description="rank(close.pct_change(d)) * sign(volume.pct_change(d))",
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
    description="decay_linear(ts_corr(rank(close), rank(volume), d), d)",
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
    description="scale(decay_linear(close.pct_change(d), d))",
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
    description="动量与放量的乘积: mom_d * (volume / mean(volume, d).shift(1) - 1)",
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
