"""Factor analysis library: rolling IC / IR / half-life / correlation / regime.

The pipeline is intentionally panel-first — every analytic function takes the
already-built OHLCV Panel and factor name list. This keeps the heavy lifting
(panel construction, factor computation) at the call site and makes the core
testable on synthetic data without touching the cache.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


def _scrub_float(v):
    """Map NaN/inf to None so json.dumps produces RFC-8259-compliant output."""
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def _series_to_json_dict(s: pd.Series) -> dict:
    return {k: _scrub_float(float(v)) for k, v in s.items()}


def _series_from_json_dict(d: dict) -> pd.Series:
    return pd.Series({k: (float("nan") if v is None else float(v)) for k, v in d.items()})


@dataclass
class FactorAnalysisResult:
    """Aggregate output of ``analyze_factors``.

    All Series are indexed by factor name (in input order).
    ``daily_ic`` and ``regime_ic`` keys are factor names / regime names.
    """
    factor_names: list[str]
    daily_ic: dict[str, pd.Series]
    mean_ic: pd.Series
    ic_ir: pd.Series
    abs_ic_mean: pd.Series
    half_life: pd.Series
    ic_correlation: pd.DataFrame
    regime_ic: dict[str, pd.Series]
    horizon: int
    ic_window: int
    n_stocks: int
    n_days: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "factor_names": list(self.factor_names),
            "daily_ic": {
                k: {
                    "index": [d.isoformat() for d in v.index],
                    "values": [_scrub_float(float(x)) for x in v.tolist()],
                } for k, v in self.daily_ic.items()
            },
            "mean_ic": _series_to_json_dict(self.mean_ic),
            "ic_ir": _series_to_json_dict(self.ic_ir),
            "abs_ic_mean": _series_to_json_dict(self.abs_ic_mean),
            "half_life": _series_to_json_dict(self.half_life),
            "ic_correlation": {
                "index": list(self.ic_correlation.index),
                "columns": list(self.ic_correlation.columns),
                "values": [[_scrub_float(float(v)) for v in row]
                           for row in self.ic_correlation.values],
            },
            "regime_ic": {
                k: v.to_dict() for k, v in self.regime_ic.items()
            },
            "horizon": self.horizon,
            "ic_window": self.ic_window,
            "n_stocks": self.n_stocks,
            "n_days": self.n_days,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, d: dict) -> "FactorAnalysisResult":
        ic_corr_values = [[float("nan") if v is None else float(v) for v in row]
                          for row in d["ic_correlation"]["values"]]
        ic_corr = pd.DataFrame(
            ic_corr_values,
            index=d["ic_correlation"]["index"],
            columns=d["ic_correlation"]["columns"],
        )
        return cls(
            factor_names=list(d["factor_names"]),
            daily_ic={
                k: pd.Series(
                    [float("nan") if x is None else float(x) for x in v["values"]],
                    index=pd.to_datetime(v["index"]),
                )
                for k, v in d["daily_ic"].items()
            },
            mean_ic=_series_from_json_dict(d["mean_ic"]),
            ic_ir=_series_from_json_dict(d["ic_ir"]),
            abs_ic_mean=_series_from_json_dict(d["abs_ic_mean"]),
            half_life=_series_from_json_dict(d["half_life"]),
            ic_correlation=ic_corr,
            regime_ic={k: pd.Series(v) for k, v in d["regime_ic"].items()},
            horizon=int(d["horizon"]),
            ic_window=int(d["ic_window"]),
            n_stocks=int(d["n_stocks"]),
            n_days=int(d["n_days"]),
            start_date=pd.Timestamp(d["start_date"]),
            end_date=pd.Timestamp(d["end_date"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FactorAnalysisResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Placeholder forward declarations — implemented in later tasks.
def compute_daily_ic(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.Series:
    """Per-day cross-sectional correlation between factor and forward return.

    Args:
        factor:      T × N wide DataFrame of factor values.
        forward_ret: T × N wide DataFrame of forward returns (same shape/index).
        method:      "spearman" (rank IC, default) or "pearson".

    Returns:
        T-indexed Series of daily IC. Days where either side has <2 valid
        cross-sectional observations or one side is constant are NaN.
    """
    if not factor.index.equals(forward_ret.index):
        raise ValueError("factor and forward_ret must share the same index")
    if not factor.columns.equals(forward_ret.columns):
        raise ValueError("factor and forward_ret must share the same columns")
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    out = pd.Series(np.nan, index=factor.index, name="ic")
    for date in factor.index:
        x = factor.loc[date]
        y = forward_ret.loc[date]
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            continue
        xv = x[mask]
        yv = y[mask]
        if method == "spearman":
            xr = xv.rank()
            yr = yv.rank()
            if xr.std(ddof=0) < 1e-12 or yr.std(ddof=0) < 1e-12:
                continue
            out.loc[date] = float(xr.corr(yr))
        else:
            if xv.std(ddof=0) < 1e-12 or yv.std(ddof=0) < 1e-12:
                continue
            out.loc[date] = float(xv.corr(yv))
    return out


def classify_regimes(
    index_close: pd.Series,
    sma_window: int = 60,
    slope_lookback: int = 5,
) -> pd.Series:
    """Label each day as 'bull' / 'bear' / 'sideways' from an index close series.

    A day is:
      * **bull** if close > SMA(sma_window) and SMA is rising over `slope_lookback`;
      * **bear** if close < SMA(sma_window) and SMA is falling over `slope_lookback`;
      * **sideways** otherwise.

    The first ``sma_window + slope_lookback - 1`` rows are NaN (warmup).
    """
    if not isinstance(index_close, pd.Series):
        raise TypeError("index_close must be a pd.Series")
    if sma_window < 2 or slope_lookback < 1:
        raise ValueError("sma_window >= 2 and slope_lookback >= 1")

    sma = index_close.rolling(sma_window, min_periods=sma_window).mean()
    slope = sma - sma.shift(slope_lookback)

    out = pd.Series(np.nan, index=index_close.index, dtype=object, name="regime")
    above = index_close > sma
    below = index_close < sma
    rising = slope > 0
    falling = slope < 0

    out.loc[above & rising] = "bull"
    out.loc[below & falling] = "bear"
    # Anything else with non-NaN sma+slope is sideways:
    valid = sma.notna() & slope.notna()
    sideways_mask = valid & ~(above & rising) & ~(below & falling)
    out.loc[sideways_mask] = "sideways"
    return out


def analyze_factors(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 5")


def pick_top_factors(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 6")
