"""Pandas oracle implementations for the 7 hot ops (8 functions).

These implementations are the *contract* that the Rust acceleration in
PR-2..5 must reproduce element-wise within ``atol=1e-9, rtol=1e-7``.

``ops.py`` re-exports every public name from here, so callers
(``wq101.py``, custom factors, etc.) keep working unchanged. Direct
imports from this module are only intended for tests and the
snapshot generator.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "correlation",
    "decay_linear",
    "indneutralize",
    "rank",
    "ts_argmax",
    "ts_argmin",
    "ts_rank",
    "ts_std",
]


def _min_periods(d: int) -> int:
    """Relax min_periods to 60% of the window so that mask-induced NaN
    runs don't kill a whole factor. ``max(1, ...)`` defends against d<2."""
    return max(1, int(d * 0.6))


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).std(ddof=0)


def ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Position of the max within the trailing ``d`` window
    (0 = today, d-1 = oldest)."""
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
    """Time-series quantile rank within the trailing ``d`` window, ∈ (0, 1]."""
    def _rank(s: pd.Series) -> float:
        a = s.values
        if np.isnan(a).any():
            return np.nan
        last = a[-1]
        return float((a <= last).sum()) / float(len(a))
    return x.rolling(d, min_periods=d).apply(_rank, raw=False)


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Linearly weighted moving average, weights 1, 2, ..., d, normalized.

    NaN-safe: NaN positions drop from numerator AND denominator; the
    remaining weights are renormalized. All-NaN window → NaN.
    """
    weights = np.arange(1, d + 1, dtype=float)

    def _wmean(a: np.ndarray) -> float:
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
    """Per-column trailing-``d`` Pearson correlation.

    Cleanup of pandas ``Rolling.corr`` FP path:
      * windows where ``std(x) < 1e-7`` or ``std(y) < 1e-7`` (effectively
        constant input) → NaN, because correlation is mathematically
        undefined for constant inputs;
      * any ``±inf`` or ``|x|>1`` result (FP garbage from ``cov/tiny_denom``)
        → NaN, since correlation is bounded on ``[-1, 1]``.

    Without this cleanup, factors that compose correlation with rank-style
    inputs (``alpha_044`` / ``alpha_050`` / ``alpha_088`` / ``corr_pv_20``
    / ``alpha_026`` …) emit ``±inf`` and FP-noise values that propagate
    through ``ts_max``/``rank``/``decay_linear`` into 36-87% NaN downstream
    when computed on small (e.g. 16-stock) cross-sections.
    """
    raw = x.rolling(d, min_periods=d).corr(y)
    sx = x.rolling(d, min_periods=d).std(ddof=0)
    sy = y.rolling(d, min_periods=d).std(ddof=0)
    constant = (sx < 1e-7) | (sy < 1e-7)
    valid = np.isfinite(raw) & (raw.abs() <= 1.0) & ~constant
    return raw.where(valid, other=np.nan)


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank per row, normalized to [0, 1].

    Mirrors WQ101's default ``rank``. Ties get average rank.
    NaN cells stay NaN and are excluded from the ranking.
    """
    return x.rank(axis=1, pct=True, method="average")


def indneutralize(
    x: pd.DataFrame,
    group: Mapping[str, str] | pd.Series,
) -> pd.DataFrame:
    """Cross-sectional demean within each sector group, per row.

    Args:
        x: T × N factor wide-frame.
        group: ``code -> sector_name`` mapping (dict or Series). Codes
               missing from the map become a unique solo group
               (self - self = 0).

    NaN cells are excluded from the per-group mean computation; the
    cell itself remains NaN in the output.
    """
    g = pd.Series(group) if not isinstance(group, pd.Series) else group
    sectors = [g.get(c, f"__solo__{c}") for c in x.columns]
    sec_series = pd.Series(sectors, index=x.columns, name="sector")
    means = x.T.groupby(sec_series).transform("mean").T
    return x - means
