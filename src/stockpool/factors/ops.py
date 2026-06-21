"""WorldQuant 101 风格算子库,作用在 T × N 宽表上。

所有函数 ``f(x: DataFrame, ...) -> DataFrame``,保持 ``x`` 的 index / columns。
NaN 安全:窗口期不足或除零返回 NaN。

约定:
  * ``ts_*`` 系列是时间序列算子(沿 axis=0)
  * ``rank`` / ``scale`` / ``signedpower`` 是横截面算子(沿 axis=1)
  * ``indneutralize`` 按分组在横截面内 demean

WQ101 论文里 ``rank(x)`` 默认就是横截面 (每天对所有股票打分),与本库一致。

实现拆分:hot op 的 pandas oracle 在 ``_ops_py.py`` ——
Rust 加速通过 dispatcher 透明接入,本模块的公开 API 不变。

当前 Rust dispatch 状态:
  - rank / ts_std / ts_argmax / ts_argmin / ts_rank: Rust (PR-T1.1)
  - decay_linear / indneutralize: Rust (PR-T1.2)
  - ts_min / ts_max: Rust (PR-B2); bit-exact with pandas oracle
  - ts_sum / ts_mean: pandas oracle only — ~1 ULP FP diff in Rust
    cascades through downstream rank() into O(1) divergence in
    composed factors. Same deferral precedent as correlation.
  - correlation: pandas oracle only (PR-3 deferral; see note below)
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

import os as _os

from . import _ops_py as _py_ops

# Rust dispatch hook. PR-2 ports `rank` only; later PRs will add more
# wrappers below. STOCKPOOL_USE_PYTHON_OPS=1 forces the pandas oracle
# regardless of whether the Rust module is importable.
_USE_RUST = False
_rust = None
if _os.environ.get("STOCKPOOL_USE_PYTHON_OPS") != "1":
    try:
        import stockpool_ops_rs as _rust  # type: ignore
        _USE_RUST = True
    except ImportError:
        _USE_RUST = False


def rank(x):
    """Cross-sectional pct-rank per row.

    Dispatches to the Rust ``stockpool_ops_rs.rank`` when available, else
    falls back to the pandas oracle (``_ops_py.rank``).
    """
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        # Rust path: enforce float64 C-contiguous numpy view; rewrap to DataFrame.
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.rank(arr)
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.rank(x)


def ts_std(x, d):
    """Rolling population stddev (ddof=0). NaN-skip."""
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.ts_std(arr, int(d))
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.ts_std(x, d)


def ts_argmax(x, d):
    """Position of max in trailing-d window (0=today, d-1=oldest)."""
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.ts_argmax(arr, int(d))
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.ts_argmax(x, d)


def ts_argmin(x, d):
    """Position of min in trailing-d window (0=today, d-1=oldest)."""
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.ts_argmin(arr, int(d))
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.ts_argmin(x, d)


def ts_rank(x, d):
    """Time-series quantile rank within trailing-d window, ∈ (0, 1]."""
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.ts_rank(arr, int(d))
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.ts_rank(x, d)


def decay_linear(x, d):
    """Linearly-weighted moving average; weights 1..=d normalised, NaN-safe."""
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.decay_linear(arr, int(d))
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.decay_linear(x, d)


def indneutralize(x, group):
    """Cross-sectional demean within sector groups (NaN-safe).

    ``group`` 是 ``{code -> sector_name}`` dict 或 Series — dispatcher 内部
    会编成 ``int32`` sector_id 数组喂给 Rust 端;缺 map 的 code 编 -1。
    """
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        # 编码:连续 0..K-1;缺 map -> -1。
        if hasattr(group, "to_dict"):
            gmap = group.to_dict()
        else:
            gmap = dict(group)
        label_to_id: dict[str, int] = {}
        sector_ids = _np.empty(len(x.columns), dtype=_np.int32)
        for i, c in enumerate(x.columns):
            s = gmap.get(c)
            if s is None:
                sector_ids[i] = -1
                continue
            if s not in label_to_id:
                label_to_id[s] = len(label_to_id)
            sector_ids[i] = label_to_id[s]
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.indneutralize(arr, sector_ids)
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.indneutralize(x, group)


def correlation(x, y, d):
    """Per-column trailing-d Pearson correlation (cleaned for FP noise).

    NOTE: Always uses the pandas oracle path (``_ops_py.correlation``),
    even when Rust is available. The Rust Welford accumulator and pandas'
    internal rolling formula have slightly different FP overflow behaviour
    for near-±1 inputs (window × 2-3 elements with tied rank values), which
    cascades through downstream ``rank()`` calls into large NaN-placement
    differences in factors like ``alpha_015`` / ``alpha_045``. The snapshot
    test is generated against the pandas oracle, so Rust dispatch for
    ``correlation`` would require regenerating the snapshot — a larger scope
    change outside this PR. ``decay_linear`` and ``indneutralize`` are
    dispatched to Rust as planned.
    """
    return _py_ops.correlation(x, y, d)


__all__ = [
    # hot ops (delegated to _ops_py; Rust dispatch when available)
    "correlation",
    "decay_linear",
    "indneutralize",
    "rank",
    "ts_argmax",
    "ts_argmin",
    "ts_max",
    "ts_mean",
    "ts_min",
    "ts_rank",
    "ts_std",
    "ts_sum",
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
    "ts_product",
    "vwap",
]


def _min_periods(d: int) -> int:
    """Relax min_periods to 60% of the window so that mask-induced NaN
    runs don't kill a whole factor. ``max(1, ...)`` defends against d<2.

    Duplicated from ``_ops_py.py`` to avoid a cross-module private import;
    must stay in sync."""
    return max(1, int(d * 0.6))


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
    """Rolling sum. NaN/inf-skip; min_periods = max(1, int(d*0.6)).

    NOTE: Always uses the pandas oracle path, even when Rust is available.
    The Rust per-window recomputation and pandas' internal Cython path
    differ by ~1 ULP (~2e-13) for windows of large values (e.g. price *
    3-field average ~70-80). This tiny diff, when fed through a downstream
    rank() call, can flip the rank order of nearly-equal stocks by exactly
    1/N ≈ 0.01-0.02, producing O(1) divergence in composed factors
    (alpha_005, alpha_023, alpha_024, alpha_045, alpha_083). The snapshot
    test was generated against the pandas oracle so Rust dispatch requires
    regenerating the snapshot — same scope issue as correlation. Deferred
    to a follow-up (same precedent as the correlation dispatcher).
    """
    return _py_ops.ts_sum(x, d)


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling mean. NaN/inf-skip; min_periods = max(1, int(d*0.6)).

    NOTE: Always uses the pandas oracle (same FP-cascade reasoning as
    ts_sum — ts_mean = ts_sum / n_finite, so the FP diff propagates
    identically into downstream rank() operations).
    """
    return _py_ops.ts_mean(x, d)


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling min. Strict min_periods = d; any NaN/inf in window → NaN."""
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.ts_min(arr, int(d))
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.ts_min(x, d)


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling max. Strict min_periods = d; any NaN/inf in window → NaN."""
    if _USE_RUST:
        import numpy as _np
        import pandas as _pd
        arr = _np.ascontiguousarray(x.to_numpy(), dtype=_np.float64)
        out_arr = _rust.ts_max(arr, int(d))
        return _pd.DataFrame(out_arr, index=x.index, columns=x.columns)
    return _py_ops.ts_max(x, d)


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
