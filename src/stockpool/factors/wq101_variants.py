"""Auto-generated WQ101 variants for A-share localization (Round 1).

Do not edit by hand; regenerate via:
    python scripts/generate_wq101_variants.py \
        --baseline reports/factor_analysis/<NEW>.json --top-n 30
"""
from __future__ import annotations

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from stockpool.factors import ops  # noqa: F401
from stockpool.factors.base import Factor  # noqa: F401
from stockpool.factors.registry import register
from stockpool.factors.wq101 import (
    WqAlpha, _ret, _vwap, _adv, _nan_like, _indneutralize,
)

@register("alpha_040_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_040 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha040_compress(WqAlpha):
    NUM = 40
    @property
    def name(self):
        return "alpha_040_compress"

    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_std(panel['high'], 5)) * ops.correlation(panel['high'], panel['volume'], 5)


@register("alpha_040_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_040 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha040_rev_short(WqAlpha):
    NUM = 40
    @property
    def name(self):
        return "alpha_040_rev_short"

    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_std(panel['high'], 5)) * ops.correlation(panel['high'], panel['volume'], 5)


@register("alpha_040_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_040 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha040_expand_long(WqAlpha):
    NUM = 40
    @property
    def name(self):
        return "alpha_040_expand_long"

    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_std(panel['high'], 10)) * ops.correlation(panel['high'], panel['volume'], 10)


@register("alpha_042_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_042 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha042_compress(WqAlpha):
    NUM = 42
    @property
    def name(self):
        return "alpha_042_compress"

    def compute(self, panel):
        vw = _vwap(panel)
        return ops.rank(vw - panel['close']) / ops.rank(vw + panel['close'])


@register("alpha_042_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_042 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha042_rev_short(WqAlpha):
    NUM = 42
    @property
    def name(self):
        return "alpha_042_rev_short"

    def compute(self, panel):
        vw = _vwap(panel)
        return ops.rank(vw - panel['close']) / ops.rank(vw + panel['close'])


@register("alpha_042_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_042 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha042_expand_long(WqAlpha):
    NUM = 42
    @property
    def name(self):
        return "alpha_042_expand_long"

    def compute(self, panel):
        vw = _vwap(panel)
        return ops.rank(vw - panel['close']) / ops.rank(vw + panel['close'])


@register("alpha_033_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_033 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha033_compress(WqAlpha):
    NUM = 33
    @property
    def name(self):
        return "alpha_033_compress"

    def compute(self, panel):
        return ops.rank(-1.0 * (1.0 - panel['open'] / panel['close']))


@register("alpha_033_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_033 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha033_rev_short(WqAlpha):
    NUM = 33
    @property
    def name(self):
        return "alpha_033_rev_short"

    def compute(self, panel):
        return ops.rank(-1.0 * (1.0 - panel['open'] / panel['close']))


@register("alpha_033_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_033 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha033_expand_long(WqAlpha):
    NUM = 33
    @property
    def name(self):
        return "alpha_033_expand_long"

    def compute(self, panel):
        return ops.rank(-1.0 * (1.0 - panel['open'] / panel['close']))


@register("alpha_041_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_041 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha041_compress(WqAlpha):
    NUM = 41
    @property
    def name(self):
        return "alpha_041_compress"

    def compute(self, panel):
        return (panel['high'] * panel['low']) ** 0.5 - _vwap(panel)


@register("alpha_041_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_041 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha041_rev_short(WqAlpha):
    NUM = 41
    @property
    def name(self):
        return "alpha_041_rev_short"

    def compute(self, panel):
        return (panel['high'] * panel['low']) ** 0.5 - _vwap(panel)


@register("alpha_041_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_041 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha041_expand_long(WqAlpha):
    NUM = 41
    @property
    def name(self):
        return "alpha_041_expand_long"

    def compute(self, panel):
        return (panel['high'] * panel['low']) ** 0.5 - _vwap(panel)


@register("alpha_094_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_094 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha094_compress(WqAlpha):
    NUM = 94
    @property
    def name(self):
        return "alpha_094_compress"

    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(vw - ops.ts_min(vw, 6))
        adv60 = _adv(panel, 30)
        b = ops.ts_rank(ops.correlation(ops.ts_rank(vw, 10), ops.ts_rank(adv60, 2), 9), 2)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@register("alpha_094_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_094 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha094_rev_short(WqAlpha):
    NUM = 94
    @property
    def name(self):
        return "alpha_094_rev_short"

    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(vw - ops.ts_min(vw, 12))
        adv60 = _adv(panel, 60)
        b = ops.ts_rank(ops.correlation(ops.ts_rank(vw, 20), ops.ts_rank(adv60, 2), 18), 2)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@register("alpha_094_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_094 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha094_expand_long(WqAlpha):
    NUM = 94
    @property
    def name(self):
        return "alpha_094_expand_long"

    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(vw - ops.ts_min(vw, 12))
        adv60 = _adv(panel, 90)
        b = ops.ts_rank(ops.correlation(ops.ts_rank(vw, 20), ops.ts_rank(adv60, 4), 18), 3)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@register("alpha_101_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_101 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha101_compress(WqAlpha):
    NUM = 101
    @property
    def name(self):
        return "alpha_101_compress"

    def compute(self, panel):
        return (panel['close'] - panel['open']) / (panel['high'] - panel['low'] + 0.001)


@register("alpha_101_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_101 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha101_rev_short(WqAlpha):
    NUM = 101
    @property
    def name(self):
        return "alpha_101_rev_short"

    def compute(self, panel):
        return (panel['close'] - panel['open']) / (panel['high'] - panel['low'] + 0.001)


@register("alpha_101_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_101 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha101_expand_long(WqAlpha):
    NUM = 101
    @property
    def name(self):
        return "alpha_101_expand_long"

    def compute(self, panel):
        return (panel['close'] - panel['open']) / (panel['high'] - panel['low'] + 0.001)


@register("alpha_038_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_038 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha038_compress(WqAlpha):
    NUM = 38
    @property
    def name(self):
        return "alpha_038_compress"

    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_rank(panel['close'], 5)) * ops.rank(panel['close'] / panel['open'])


@register("alpha_038_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_038 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha038_rev_short(WqAlpha):
    NUM = 38
    @property
    def name(self):
        return "alpha_038_rev_short"

    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_rank(panel['close'], 5)) * ops.rank(panel['close'] / panel['open'])


@register("alpha_038_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_038 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha038_expand_long(WqAlpha):
    NUM = 38
    @property
    def name(self):
        return "alpha_038_expand_long"

    def compute(self, panel):
        return -1.0 * ops.rank(ops.ts_rank(panel['close'], 10)) * ops.rank(panel['close'] / panel['open'])


@register("alpha_083_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_083 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha083_compress(WqAlpha):
    NUM = 83
    @property
    def name(self):
        return "alpha_083_compress"

    def compute(self, panel):
        c, h, l = (panel['close'], panel['high'], panel['low'])
        vw = _vwap(panel)
        ratio = (h - l) / (ops.ts_sum(c, 3) / 5.0)
        a = ops.rank(ops.delay(ratio, 2)) * ops.rank(ops.rank(panel['volume']))
        denom = ratio / (vw - c).replace(0.0, np.nan)
        return a / denom


@register("alpha_083_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_083 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha083_rev_short(WqAlpha):
    NUM = 83
    @property
    def name(self):
        return "alpha_083_rev_short"

    def compute(self, panel):
        c, h, l = (panel['close'], panel['high'], panel['low'])
        vw = _vwap(panel)
        ratio = (h - l) / (ops.ts_sum(c, 3) / 5.0)
        a = ops.rank(ops.delay(ratio, 2)) * ops.rank(ops.rank(panel['volume']))
        denom = ratio / (vw - c).replace(0.0, np.nan)
        return a / denom


@register("alpha_083_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_083 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha083_expand_long(WqAlpha):
    NUM = 83
    @property
    def name(self):
        return "alpha_083_expand_long"

    def compute(self, panel):
        c, h, l = (panel['close'], panel['high'], panel['low'])
        vw = _vwap(panel)
        ratio = (h - l) / (ops.ts_sum(c, 5) / 5.0)
        a = ops.rank(ops.delay(ratio, 2)) * ops.rank(ops.rank(panel['volume']))
        denom = ratio / (vw - c).replace(0.0, np.nan)
        return a / denom


@register("alpha_025_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_025 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha025_compress(WqAlpha):
    NUM = 25
    @property
    def name(self):
        return "alpha_025_compress"

    def compute(self, panel):
        ret = _ret(panel)
        return ops.rank(-1.0 * ret * _adv(panel, 10) * _vwap(panel) * (panel['high'] - panel['close']))


@register("alpha_025_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_025 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha025_rev_short(WqAlpha):
    NUM = 25
    @property
    def name(self):
        return "alpha_025_rev_short"

    def compute(self, panel):
        ret = _ret(panel)
        return ops.rank(-1.0 * ret * _adv(panel, 20) * _vwap(panel) * (panel['high'] - panel['close']))


@register("alpha_025_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_025 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha025_expand_long(WqAlpha):
    NUM = 25
    @property
    def name(self):
        return "alpha_025_expand_long"

    def compute(self, panel):
        ret = _ret(panel)
        return ops.rank(-1.0 * ret * _adv(panel, 20) * _vwap(panel) * (panel['high'] - panel['close']))


@register("alpha_008_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_008 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha008_compress(WqAlpha):
    NUM = 8
    @property
    def name(self):
        return "alpha_008_compress"

    def compute(self, panel):
        ret = _ret(panel)
        prod = ops.ts_sum(panel['open'], 3) * ops.ts_sum(ret, 3)
        return -1.0 * ops.rank(prod - ops.delay(prod, 5))


@register("alpha_008_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_008 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha008_rev_short(WqAlpha):
    NUM = 8
    @property
    def name(self):
        return "alpha_008_rev_short"

    def compute(self, panel):
        ret = _ret(panel)
        prod = ops.ts_sum(panel['open'], 3) * ops.ts_sum(ret, 3)
        return -1.0 * ops.rank(prod - ops.delay(prod, 5))


@register("alpha_008_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_008 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha008_expand_long(WqAlpha):
    NUM = 8
    @property
    def name(self):
        return "alpha_008_expand_long"

    def compute(self, panel):
        ret = _ret(panel)
        prod = ops.ts_sum(panel['open'], 5) * ops.ts_sum(ret, 5)
        return -1.0 * ops.rank(prod - ops.delay(prod, 10))


@register("alpha_005_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_005 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha005_compress(WqAlpha):
    NUM = 5
    @property
    def name(self):
        return "alpha_005_compress"

    def compute(self, panel):
        vwap = _vwap(panel)
        a = ops.rank(panel['open'] - ops.ts_sum(vwap, 5) / 10.0)
        b = -1.0 * ops.rank(panel['close'] - vwap).abs()
        return a * b


@register("alpha_005_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_005 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha005_rev_short(WqAlpha):
    NUM = 5
    @property
    def name(self):
        return "alpha_005_rev_short"

    def compute(self, panel):
        vwap = _vwap(panel)
        a = ops.rank(panel['open'] - ops.ts_sum(vwap, 5) / 10.0)
        b = -1.0 * ops.rank(panel['close'] - vwap).abs()
        return a * b


@register("alpha_005_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_005 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha005_expand_long(WqAlpha):
    NUM = 5
    @property
    def name(self):
        return "alpha_005_expand_long"

    def compute(self, panel):
        vwap = _vwap(panel)
        a = ops.rank(panel['open'] - ops.ts_sum(vwap, 10) / 10.0)
        b = -1.0 * ops.rank(panel['close'] - vwap).abs()
        return a * b


@register("alpha_018_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_018 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha018_compress(WqAlpha):
    NUM = 18
    @property
    def name(self):
        return "alpha_018_compress"

    def compute(self, panel):
        diff = panel['close'] - panel['open']
        inside = ops.ts_std(diff.abs(), 3) + diff + ops.correlation(panel['close'], panel['open'], 5)
        return -1.0 * ops.rank(inside)


@register("alpha_018_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_018 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha018_rev_short(WqAlpha):
    NUM = 18
    @property
    def name(self):
        return "alpha_018_rev_short"

    def compute(self, panel):
        diff = panel['close'] - panel['open']
        inside = ops.ts_std(diff.abs(), 3) + diff + ops.correlation(panel['close'], panel['open'], 5)
        return -1.0 * ops.rank(inside)


@register("alpha_018_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_018 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha018_expand_long(WqAlpha):
    NUM = 18
    @property
    def name(self):
        return "alpha_018_expand_long"

    def compute(self, panel):
        diff = panel['close'] - panel['open']
        inside = ops.ts_std(diff.abs(), 5) + diff + ops.correlation(panel['close'], panel['open'], 10)
        return -1.0 * ops.rank(inside)


@register("alpha_088_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_088 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha088_compress(WqAlpha):
    NUM = 88
    @property
    def name(self):
        return "alpha_088_compress"

    def compute(self, panel):
        a = ops.rank(ops.decay_linear(ops.rank(panel['open']) + ops.rank(panel['low']) - (ops.rank(panel['high']) + ops.rank(panel['close'])), 4))
        adv60 = _adv(panel, 30)
        inner = ops.correlation(ops.ts_rank(panel['close'], 4), ops.ts_rank(adv60, 11), 4)
        b = ops.ts_rank(ops.decay_linear(inner, 4), 2)
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@register("alpha_088_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_088 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha088_rev_short(WqAlpha):
    NUM = 88
    @property
    def name(self):
        return "alpha_088_rev_short"

    def compute(self, panel):
        a = ops.rank(ops.decay_linear(ops.rank(panel['open']) + ops.rank(panel['low']) - (ops.rank(panel['high']) + ops.rank(panel['close'])), 4))
        adv60 = _adv(panel, 60)
        inner = ops.correlation(ops.ts_rank(panel['close'], 4), ops.ts_rank(adv60, 21), 4)
        b = ops.ts_rank(ops.decay_linear(inner, 4), 2)
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@register("alpha_088_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_088 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha088_expand_long(WqAlpha):
    NUM = 88
    @property
    def name(self):
        return "alpha_088_expand_long"

    def compute(self, panel):
        a = ops.rank(ops.decay_linear(ops.rank(panel['open']) + ops.rank(panel['low']) - (ops.rank(panel['high']) + ops.rank(panel['close'])), 8))
        adv60 = _adv(panel, 90)
        inner = ops.correlation(ops.ts_rank(panel['close'], 8), ops.ts_rank(adv60, 21), 8)
        b = ops.ts_rank(ops.decay_linear(inner, 7), 3)
        return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


@register("alpha_057_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_057 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha057_compress(WqAlpha):
    NUM = 57
    @property
    def name(self):
        return "alpha_057_compress"

    def compute(self, panel):
        c = panel['close']
        return -1.0 * ((c - _vwap(panel)) / ops.decay_linear(ops.rank(ops.ts_argmax(c, 15)), 2))


@register("alpha_057_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_057 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha057_rev_short(WqAlpha):
    NUM = 57
    @property
    def name(self):
        return "alpha_057_rev_short"

    def compute(self, panel):
        c = panel['close']
        return -1.0 * ((c - _vwap(panel)) / ops.decay_linear(ops.rank(ops.ts_argmax(c, 30)), 2))


@register("alpha_057_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_057 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha057_expand_long(WqAlpha):
    NUM = 57
    @property
    def name(self):
        return "alpha_057_expand_long"

    def compute(self, panel):
        c = panel['close']
        return -1.0 * ((c - _vwap(panel)) / ops.decay_linear(ops.rank(ops.ts_argmax(c, 30)), 2))


@register("alpha_073_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_073 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha073_compress(WqAlpha):
    NUM = 73
    @property
    def name(self):
        return "alpha_073_compress"

    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 3), 2))
        proxy = panel['open'] * 0.147155 + panel['low'] * 0.852845
        inner = ops.delta(proxy, 2) / proxy.replace(0.0, np.nan) * -1.0
        b = ops.ts_rank(ops.decay_linear(inner, 2), 9)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@register("alpha_073_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_073 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha073_rev_short(WqAlpha):
    NUM = 73
    @property
    def name(self):
        return "alpha_073_rev_short"

    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 3), 2))
        proxy = panel['open'] * 0.147155 + panel['low'] * 0.852845
        inner = ops.delta(proxy, 2) / proxy.replace(0.0, np.nan) * -1.0
        b = ops.ts_rank(ops.decay_linear(inner, 2), 17)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@register("alpha_073_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_073 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha073_expand_long(WqAlpha):
    NUM = 73
    @property
    def name(self):
        return "alpha_073_expand_long"

    def compute(self, panel):
        vw = _vwap(panel)
        a = ops.rank(ops.decay_linear(ops.delta(vw, 5), 3))
        proxy = panel['open'] * 0.147155 + panel['low'] * 0.852845
        inner = ops.delta(proxy, 2) / proxy.replace(0.0, np.nan) * -1.0
        b = ops.ts_rank(ops.decay_linear(inner, 3), 17)
        return -1.0 * pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


@register("alpha_009_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_009 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha009_compress(WqAlpha):
    NUM = 9
    @property
    def name(self):
        return "alpha_009_compress"

    def compute(self, panel):
        d = ops.delta(panel['close'], 2)
        out = -1.0 * d
        out = out.where(~(ops.ts_max(d, 3) < 0), d)
        out = out.where(~(ops.ts_min(d, 3) > 0), d)
        return out


@register("alpha_009_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_009 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha009_rev_short(WqAlpha):
    NUM = 9
    @property
    def name(self):
        return "alpha_009_rev_short"

    def compute(self, panel):
        d = ops.delta(panel['close'], 2)
        out = -1.0 * d
        out = out.where(~(ops.ts_max(d, 3) < 0), d)
        out = out.where(~(ops.ts_min(d, 3) > 0), d)
        return out


@register("alpha_009_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_009 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha009_expand_long(WqAlpha):
    NUM = 9
    @property
    def name(self):
        return "alpha_009_expand_long"

    def compute(self, panel):
        d = ops.delta(panel['close'], 1)
        out = -1.0 * d
        out = out.where(~(ops.ts_max(d, 5) < 0), d)
        out = out.where(~(ops.ts_min(d, 5) > 0), d)
        return out


@register("alpha_052_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_052 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha052_compress(WqAlpha):
    NUM = 52
    @property
    def name(self):
        return "alpha_052_compress"

    def compute(self, panel):
        ret = _ret(panel)
        a = -1.0 * ops.ts_min(panel['low'], 3) + ops.delay(ops.ts_min(panel['low'], 3), 3)
        b = ops.rank((ops.ts_sum(ret, 120) - ops.ts_sum(ret, 10)) / 220.0)
        d = ops.ts_rank(panel['volume'], 3)
        return a * b * d


@register("alpha_052_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_052 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha052_rev_short(WqAlpha):
    NUM = 52
    @property
    def name(self):
        return "alpha_052_rev_short"

    def compute(self, panel):
        ret = _ret(panel)
        a = -1.0 * ops.ts_min(panel['low'], 3) + ops.delay(ops.ts_min(panel['low'], 3), 3)
        b = ops.rank((ops.ts_sum(ret, 240) - ops.ts_sum(ret, 20)) / 220.0)
        d = ops.ts_rank(panel['volume'], 3)
        return a * b * d


@register("alpha_052_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_052 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha052_expand_long(WqAlpha):
    NUM = 52
    @property
    def name(self):
        return "alpha_052_expand_long"

    def compute(self, panel):
        ret = _ret(panel)
        a = -1.0 * ops.ts_min(panel['low'], 5) + ops.delay(ops.ts_min(panel['low'], 5), 5)
        b = ops.rank((ops.ts_sum(ret, 360) - ops.ts_sum(ret, 20)) / 220.0)
        d = ops.ts_rank(panel['volume'], 5)
        return a * b * d


@register("alpha_090_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_090 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha090_compress(WqAlpha):
    NUM = 90
    @property
    def name(self):
        return "alpha_090_compress"

    def compute(self, panel):
        a = ops.rank(panel['close'] - ops.ts_max(panel['close'], 3))
        adv40 = _adv(panel, 20)
        b = ops.ts_rank(ops.correlation(_indneutralize(adv40), panel['low'], 3), 2)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@register("alpha_090_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_090 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha090_rev_short(WqAlpha):
    NUM = 90
    @property
    def name(self):
        return "alpha_090_rev_short"

    def compute(self, panel):
        a = ops.rank(panel['close'] - ops.ts_max(panel['close'], 3))
        adv40 = _adv(panel, 40)
        b = ops.ts_rank(ops.correlation(_indneutralize(adv40), panel['low'], 3), 2)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@register("alpha_090_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_090 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha090_expand_long(WqAlpha):
    NUM = 90
    @property
    def name(self):
        return "alpha_090_expand_long"

    def compute(self, panel):
        a = ops.rank(panel['close'] - ops.ts_max(panel['close'], 5))
        adv40 = _adv(panel, 40)
        b = ops.ts_rank(ops.correlation(_indneutralize(adv40), panel['low'], 5), 3)
        a_safe = a.clip(lower=0.0).fillna(0.0)
        return -1.0 * ops.signedpower(a_safe, b)


@register("alpha_049_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_049 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha049_compress(WqAlpha):
    NUM = 49
    @property
    def name(self):
        return "alpha_049_compress"

    def compute(self, panel):
        c = panel['close']
        diff = (ops.delay(c, 10) - ops.delay(c, 5)) / 10.0 - (ops.delay(c, 5) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 2))
        return out.where(~(diff < -0.1), 1.0)


@register("alpha_049_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_049 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha049_rev_short(WqAlpha):
    NUM = 49
    @property
    def name(self):
        return "alpha_049_rev_short"

    def compute(self, panel):
        c = panel['close']
        diff = (ops.delay(c, 20) - ops.delay(c, 5)) / 10.0 - (ops.delay(c, 5) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 2))
        return out.where(~(diff < -0.1), 1.0)


@register("alpha_049_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_049 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha049_expand_long(WqAlpha):
    NUM = 49
    @property
    def name(self):
        return "alpha_049_expand_long"

    def compute(self, panel):
        c = panel['close']
        diff = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 1))
        return out.where(~(diff < -0.1), 1.0)


@register("alpha_054_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_054 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha054_compress(WqAlpha):
    NUM = 54
    @property
    def name(self):
        return "alpha_054_compress"

    def compute(self, panel):
        c, o, h, l = (panel['close'], panel['open'], panel['high'], panel['low'])
        num = -1.0 * ((l - c) * o ** 5)
        den = ((l - h) * c ** 5).replace(0.0, np.nan)
        return num / den


@register("alpha_054_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_054 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha054_rev_short(WqAlpha):
    NUM = 54
    @property
    def name(self):
        return "alpha_054_rev_short"

    def compute(self, panel):
        c, o, h, l = (panel['close'], panel['open'], panel['high'], panel['low'])
        num = -1.0 * ((l - c) * o ** 5)
        den = ((l - h) * c ** 5).replace(0.0, np.nan)
        return num / den


@register("alpha_054_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_054 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha054_expand_long(WqAlpha):
    NUM = 54
    @property
    def name(self):
        return "alpha_054_expand_long"

    def compute(self, panel):
        c, o, h, l = (panel['close'], panel['open'], panel['high'], panel['low'])
        num = -1.0 * ((l - c) * o ** 5)
        den = ((l - h) * c ** 5).replace(0.0, np.nan)
        return num / den


@register("alpha_047_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_047 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha047_compress(WqAlpha):
    NUM = 47
    @property
    def name(self):
        return "alpha_047_compress"

    def compute(self, panel):
        c, h = (panel['close'], panel['high'])
        adv20 = _adv(panel, 10)
        vw = _vwap(panel)
        a = ops.rank(1.0 / c) * panel['volume'] / adv20
        b = h * ops.rank(h - c) / (ops.ts_sum(h, 3) / 5.0)
        d = ops.rank(vw - ops.delay(vw, 3))
        return a * b - d


@register("alpha_047_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_047 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha047_rev_short(WqAlpha):
    NUM = 47
    @property
    def name(self):
        return "alpha_047_rev_short"

    def compute(self, panel):
        c, h = (panel['close'], panel['high'])
        adv20 = _adv(panel, 20)
        vw = _vwap(panel)
        a = ops.rank(1.0 / c) * panel['volume'] / adv20
        b = h * ops.rank(h - c) / (ops.ts_sum(h, 3) / 5.0)
        d = ops.rank(vw - ops.delay(vw, 3))
        return a * b - d


@register("alpha_047_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_047 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha047_expand_long(WqAlpha):
    NUM = 47
    @property
    def name(self):
        return "alpha_047_expand_long"

    def compute(self, panel):
        c, h = (panel['close'], panel['high'])
        adv20 = _adv(panel, 20)
        vw = _vwap(panel)
        a = ops.rank(1.0 / c) * panel['volume'] / adv20
        b = h * ops.rank(h - c) / (ops.ts_sum(h, 5) / 5.0)
        d = ops.rank(vw - ops.delay(vw, 5))
        return a * b - d


@register("alpha_019_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_019 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha019_compress(WqAlpha):
    NUM = 19
    @property
    def name(self):
        return "alpha_019_compress"

    def compute(self, panel):
        c = panel['close']
        a = -1.0 * np.sign(c - ops.delay(c, 4) + ops.delta(c, 4))
        ret = _ret(panel)
        b = 1.0 + ops.rank(1.0 + ops.ts_sum(ret, 125))
        return a * b


@register("alpha_019_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_019 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha019_rev_short(WqAlpha):
    NUM = 19
    @property
    def name(self):
        return "alpha_019_rev_short"

    def compute(self, panel):
        c = panel['close']
        a = -1.0 * np.sign(c - ops.delay(c, 4) + ops.delta(c, 4))
        ret = _ret(panel)
        b = 1.0 + ops.rank(1.0 + ops.ts_sum(ret, 250))
        return a * b


@register("alpha_019_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_019 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha019_expand_long(WqAlpha):
    NUM = 19
    @property
    def name(self):
        return "alpha_019_expand_long"

    def compute(self, panel):
        c = panel['close']
        a = -1.0 * np.sign(c - ops.delay(c, 7) + ops.delta(c, 7))
        ret = _ret(panel)
        b = 1.0 + ops.rank(1.0 + ops.ts_sum(ret, 375))
        return a * b


@register("alpha_010_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_010 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha010_compress(WqAlpha):
    NUM = 10
    @property
    def name(self):
        return "alpha_010_compress"

    def compute(self, panel):
        d = ops.delta(panel['close'], 2)
        inner = -1.0 * d
        inner = inner.where(~(ops.ts_max(d, 2) < 0), d)
        inner = inner.where(~(ops.ts_min(d, 2) > 0), d)
        return ops.rank(inner)


@register("alpha_010_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_010 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha010_rev_short(WqAlpha):
    NUM = 10
    @property
    def name(self):
        return "alpha_010_rev_short"

    def compute(self, panel):
        d = ops.delta(panel['close'], 2)
        inner = -1.0 * d
        inner = inner.where(~(ops.ts_max(d, 2) < 0), d)
        inner = inner.where(~(ops.ts_min(d, 2) > 0), d)
        return ops.rank(inner)


@register("alpha_010_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_010 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha010_expand_long(WqAlpha):
    NUM = 10
    @property
    def name(self):
        return "alpha_010_expand_long"

    def compute(self, panel):
        d = ops.delta(panel['close'], 1)
        inner = -1.0 * d
        inner = inner.where(~(ops.ts_max(d, 4) < 0), d)
        inner = inner.where(~(ops.ts_min(d, 4) > 0), d)
        return ops.rank(inner)


@register("alpha_051_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_051 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha051_compress(WqAlpha):
    NUM = 51
    @property
    def name(self):
        return "alpha_051_compress"

    def compute(self, panel):
        c = panel['close']
        diff = (ops.delay(c, 10) - ops.delay(c, 5)) / 10.0 - (ops.delay(c, 5) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 2))
        return out.where(~(diff < -0.05), 1.0)


@register("alpha_051_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_051 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha051_rev_short(WqAlpha):
    NUM = 51
    @property
    def name(self):
        return "alpha_051_rev_short"

    def compute(self, panel):
        c = panel['close']
        diff = (ops.delay(c, 20) - ops.delay(c, 5)) / 10.0 - (ops.delay(c, 5) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 2))
        return out.where(~(diff < -0.05), 1.0)


@register("alpha_051_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_051 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha051_expand_long(WqAlpha):
    NUM = 51
    @property
    def name(self):
        return "alpha_051_expand_long"

    def compute(self, panel):
        c = panel['close']
        diff = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
        out = -1.0 * (c - ops.delay(c, 1))
        return out.where(~(diff < -0.05), 1.0)


@register("alpha_039_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_039 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha039_compress(WqAlpha):
    NUM = 39
    @property
    def name(self):
        return "alpha_039_compress"

    def compute(self, panel):
        c = panel['close']
        ret = _ret(panel)
        dvol = panel['volume'] / _adv(panel, 10)
        a = -1.0 * ops.rank(ops.delta(c, 4) * (1.0 - ops.rank(ops.decay_linear(dvol, 5))))
        b = 1.0 + ops.rank(ops.ts_sum(ret, 125))
        return a * b


@register("alpha_039_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_039 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha039_rev_short(WqAlpha):
    NUM = 39
    @property
    def name(self):
        return "alpha_039_rev_short"

    def compute(self, panel):
        c = panel['close']
        ret = _ret(panel)
        dvol = panel['volume'] / _adv(panel, 20)
        a = -1.0 * ops.rank(ops.delta(c, 4) * (1.0 - ops.rank(ops.decay_linear(dvol, 5))))
        b = 1.0 + ops.rank(ops.ts_sum(ret, 250))
        return a * b


@register("alpha_039_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_039 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha039_expand_long(WqAlpha):
    NUM = 39
    @property
    def name(self):
        return "alpha_039_expand_long"

    def compute(self, panel):
        c = panel['close']
        ret = _ret(panel)
        dvol = panel['volume'] / _adv(panel, 20)
        a = -1.0 * ops.rank(ops.delta(c, 7) * (1.0 - ops.rank(ops.decay_linear(dvol, 9))))
        b = 1.0 + ops.rank(ops.ts_sum(ret, 375))
        return a * b


@register("alpha_029_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_029 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha029_compress(WqAlpha):
    NUM = 29
    @property
    def name(self):
        return "alpha_029_compress"

    def compute(self, panel):
        c = panel['close']
        ret = _ret(panel)
        inner = -1.0 * ops.rank(ops.delta(c - 1.0, 3))
        inner = ops.rank(ops.rank(inner))
        inner = ops.ts_min(inner, 2)
        inner = ops.ts_sum(inner, 2)
        inner = np.log(inner.where(inner > 0, np.nan))
        inner = ops.rank(ops.rank(ops.scale(inner)))
        a = ops.ts_min(ops.ts_product(inner, 2), 3)
        b = ops.ts_rank(ops.delay(-1.0 * ret, 3), 3)
        return a + b


@register("alpha_029_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_029 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha029_rev_short(WqAlpha):
    NUM = 29
    @property
    def name(self):
        return "alpha_029_rev_short"

    def compute(self, panel):
        c = panel['close']
        ret = _ret(panel)
        inner = -1.0 * ops.rank(ops.delta(c - 1.0, 3))
        inner = ops.rank(ops.rank(inner))
        inner = ops.ts_min(inner, 2)
        inner = ops.ts_sum(inner, 2)
        inner = np.log(inner.where(inner > 0, np.nan))
        inner = ops.rank(ops.rank(ops.scale(inner)))
        a = ops.ts_min(ops.ts_product(inner, 2), 3)
        b = ops.ts_rank(ops.delay(-1.0 * ret, 3), 3)
        return a + b


@register("alpha_029_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_029 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha029_expand_long(WqAlpha):
    NUM = 29
    @property
    def name(self):
        return "alpha_029_expand_long"

    def compute(self, panel):
        c = panel['close']
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


@register("alpha_024_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_024 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha024_compress(WqAlpha):
    NUM = 24
    @property
    def name(self):
        return "alpha_024_compress"

    def compute(self, panel):
        c = panel['close']
        ma100 = ops.ts_sum(c, 50) / 100.0
        cond = ops.delta(ma100, 50) / ops.delay(c, 50) <= 0.05
        a = -1.0 * (c - ops.ts_min(c, 50))
        b = -1.0 * ops.delta(c, 2)
        return a.where(cond, b)


@register("alpha_024_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_024 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha024_rev_short(WqAlpha):
    NUM = 24
    @property
    def name(self):
        return "alpha_024_rev_short"

    def compute(self, panel):
        c = panel['close']
        ma100 = ops.ts_sum(c, 100) / 100.0
        cond = ops.delta(ma100, 100) / ops.delay(c, 100) <= 0.05
        a = -1.0 * (c - ops.ts_min(c, 100))
        b = -1.0 * ops.delta(c, 2)
        return a.where(cond, b)


@register("alpha_024_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_024 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha024_expand_long(WqAlpha):
    NUM = 24
    @property
    def name(self):
        return "alpha_024_expand_long"

    def compute(self, panel):
        c = panel['close']
        ma100 = ops.ts_sum(c, 150) / 100.0
        cond = ops.delta(ma100, 150) / ops.delay(c, 150) <= 0.05
        a = -1.0 * (c - ops.ts_min(c, 150))
        b = -1.0 * ops.delta(c, 3)
        return a.where(cond, b)


@register("alpha_060_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_060 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha060_compress(WqAlpha):
    NUM = 60
    @property
    def name(self):
        return "alpha_060_compress"

    def compute(self, panel):
        c, h, l = (panel['close'], panel['high'], panel['low'])
        rng = (h - l).replace(0.0, np.nan)
        a = ops.scale(ops.rank((c - l - (h - c)) / rng * panel['volume']))
        b = ops.scale(ops.rank(ops.ts_argmax(c, 5)))
        return -1.0 * (2.0 * a - b)


@register("alpha_060_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_060 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha060_rev_short(WqAlpha):
    NUM = 60
    @property
    def name(self):
        return "alpha_060_rev_short"

    def compute(self, panel):
        c, h, l = (panel['close'], panel['high'], panel['low'])
        rng = (h - l).replace(0.0, np.nan)
        a = ops.scale(ops.rank((c - l - (h - c)) / rng * panel['volume']))
        b = ops.scale(ops.rank(ops.ts_argmax(c, 5)))
        return -1.0 * (2.0 * a - b)


@register("alpha_060_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_060 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha060_expand_long(WqAlpha):
    NUM = 60
    @property
    def name(self):
        return "alpha_060_expand_long"

    def compute(self, panel):
        c, h, l = (panel['close'], panel['high'], panel['low'])
        rng = (h - l).replace(0.0, np.nan)
        a = ops.scale(ops.rank((c - l - (h - c)) / rng * panel['volume']))
        b = ops.scale(ops.rank(ops.ts_argmax(c, 10)))
        return -1.0 * (2.0 * a - b)


@register("alpha_037_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_037 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha037_compress(WqAlpha):
    NUM = 37
    @property
    def name(self):
        return "alpha_037_compress"

    def compute(self, panel):
        c, o = (panel['close'], panel['open'])
        return ops.rank(ops.correlation(ops.delay(o - c, 2), c, 100)) + ops.rank(o - c)


@register("alpha_037_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_037 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha037_rev_short(WqAlpha):
    NUM = 37
    @property
    def name(self):
        return "alpha_037_rev_short"

    def compute(self, panel):
        c, o = (panel['close'], panel['open'])
        return ops.rank(ops.correlation(ops.delay(o - c, 2), c, 200)) + ops.rank(o - c)


@register("alpha_037_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_037 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha037_expand_long(WqAlpha):
    NUM = 37
    @property
    def name(self):
        return "alpha_037_expand_long"

    def compute(self, panel):
        c, o = (panel['close'], panel['open'])
        return ops.rank(ops.correlation(ops.delay(o - c, 1), c, 300)) + ops.rank(o - c)


@register("alpha_017_compress",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_017 with rule=compress applied to its window literals (A-share localization, Round 1).')
class Alpha017_compress(WqAlpha):
    NUM = 17
    @property
    def name(self):
        return "alpha_017_compress"

    def compute(self, panel):
        a = -1.0 * ops.rank(ops.ts_rank(panel['close'], 5))
        b = ops.rank(ops.delta(ops.delta(panel['close'], 2), 2))
        adv20 = _adv(panel, 10)
        c = ops.rank(ops.ts_rank(panel['volume'] / adv20, 3))
        return a * b * c


@register("alpha_017_rev_short",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_017 with rule=rev_short applied to its window literals (A-share localization, Round 1).')
class Alpha017_rev_short(WqAlpha):
    NUM = 17
    @property
    def name(self):
        return "alpha_017_rev_short"

    def compute(self, panel):
        a = -1.0 * ops.rank(ops.ts_rank(panel['close'], 5))
        b = ops.rank(ops.delta(ops.delta(panel['close'], 2), 2))
        adv20 = _adv(panel, 20)
        c = ops.rank(ops.ts_rank(panel['volume'] / adv20, 3))
        return a * b * c


@register("alpha_017_expand_long",
          sources=("wq101", "wq101_localized"),
          types=("cross_sectional",),
          description='WQ101 alpha_017 with rule=expand_long applied to its window literals (A-share localization, Round 1).')
class Alpha017_expand_long(WqAlpha):
    NUM = 17
    @property
    def name(self):
        return "alpha_017_expand_long"

    def compute(self, panel):
        a = -1.0 * ops.rank(ops.ts_rank(panel['close'], 10))
        b = ops.rank(ops.delta(ops.delta(panel['close'], 1), 1))
        adv20 = _adv(panel, 20)
        c = ops.rank(ops.ts_rank(panel['volume'] / adv20, 5))
        return a * b * c

