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

from stockpool.factors.registry import make_factor
from stockpool.ml.dataset import forward_return_panel


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


def _half_life_from_acf(series: pd.Series, max_half_life: float = 252.0) -> float:
    """Half-life of a series via AR(1) lag-1 autocorrelation.

    Returns ``log(0.5) / log(ρ_1)`` if ``ρ_1`` is in ``(0, 1)``; ``NaN`` otherwise.
    Clipped at ``max_half_life`` to avoid blow-up near unit-root.
    """
    s = series.dropna()
    if len(s) < 10:
        return float("nan")
    s_centered = s - s.mean()
    s_shifted = s_centered.shift(1).dropna()
    s_current = s_centered.iloc[1:]
    denom = (s_shifted ** 2).sum()
    if denom < 1e-12:
        return float("nan")
    rho = float((s_current * s_shifted).sum() / denom)
    if rho <= 0 or rho >= 1:
        return float("nan")
    hl = float(np.log(0.5) / np.log(rho))
    return min(hl, max_half_life)


def analyze_factors(
    panel: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str],
    horizon: int = 3,
    ic_window: int = 252,
    regime_index_close: pd.Series | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
) -> FactorAnalysisResult:
    """End-to-end factor analysis on a panel.

    Args:
        panel:       OHLCV wide-frame panel (output of ``build_panel_from_cache``).
        factor_names: registered factor names (e.g. ``["momentum_20", "alpha_001"]``).
        horizon:     forward-return horizon (bars).
        ic_window:   reserved for future rolling-IC variants. Currently only
                     affects the metadata stored on the result; daily IC is
                     computed across the full available window.
        regime_index_close: optional pd.Series of an index close (e.g. sh000001)
                     to split daily IC into bull/bear/sideways regimes.
        method:      "spearman" (rank IC, default) or "pearson".

    Returns:
        ``FactorAnalysisResult`` with per-factor metrics and pairwise IC correlation.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    factor_names = list(factor_names)
    if not factor_names:
        raise ValueError("factor_names must be non-empty")

    # Resolve base names (e.g. "momentum", "boll_position") to canonical names
    # ("momentum_20", "boll_position_20"). Pre-resolved names round-trip unchanged.
    factor_names = [make_factor(n).name for n in factor_names]

    fwd = forward_return_panel(panel["close"], horizon)

    # Stream factor compute -> IC -> discard the T×N factor panel. The full
    # accumulated dict in compute_factor_panel costs ~17 MB * len(factor_names)
    # at 4357 stocks * 500 days; at 167 factors that is ~3 GB of held panels,
    # which can OOM C extensions even when raw RAM looks fine (ACCESS_VIOLATION
    # on Windows). daily_ic alone is a T-length Series per factor (~4 KB).
    daily_ic: dict[str, pd.Series] = {}
    try:
        from tqdm import tqdm
        factor_iter = tqdm(
            factor_names, desc="analyze_factors",
            unit="factor", mininterval=1.0,
        )
    except ImportError:
        factor_iter = factor_names
    for name in factor_iter:
        if hasattr(factor_iter, "set_postfix_str"):
            factor_iter.set_postfix_str(name)
        f = make_factor(name)
        fp_one = f.compute(panel)
        daily_ic[name] = compute_daily_ic(fp_one, fwd, method=method)
        del fp_one

    mean_ic = pd.Series(
        {n: daily_ic[n].mean(skipna=True) for n in factor_names}, name="mean_ic",
    )
    std_ic = pd.Series(
        {n: daily_ic[n].std(skipna=True, ddof=0) for n in factor_names}, name="std_ic",
    )
    ic_ir = pd.Series(
        {
            n: (mean_ic[n] / std_ic[n]) if std_ic[n] > 1e-12 else float("nan")
            for n in factor_names
        },
        name="ic_ir",
    )
    abs_ic_mean = pd.Series(
        {n: daily_ic[n].abs().mean(skipna=True) for n in factor_names},
        name="abs_ic_mean",
    )
    half_life = pd.Series(
        {n: _half_life_from_acf(daily_ic[n]) for n in factor_names},
        name="half_life",
    )

    ic_corr_df = pd.DataFrame(daily_ic)[factor_names]
    ic_correlation = ic_corr_df.corr(method="pearson").fillna(0.0)
    # Force diagonal to exactly 1 (NaN columns get filled with 0; fix that).
    for i, n in enumerate(factor_names):
        ic_correlation.iloc[i, i] = 1.0

    regime_ic: dict[str, pd.Series] = {}
    if regime_index_close is not None:
        regimes = classify_regimes(regime_index_close).reindex(
            ic_corr_df.index
        )
        for regime in ("bull", "bear", "sideways"):
            mask = regimes == regime
            if mask.sum() < 5:
                continue
            sliced = ic_corr_df.loc[mask]
            regime_ic[regime] = pd.Series(
                {n: sliced[n].mean(skipna=True) for n in factor_names},
                name=f"ic_{regime}",
            )

    valid_dates = ic_corr_df.dropna(how="all").index
    return FactorAnalysisResult(
        factor_names=factor_names,
        daily_ic=daily_ic,
        mean_ic=mean_ic,
        ic_ir=ic_ir,
        abs_ic_mean=abs_ic_mean,
        half_life=half_life,
        ic_correlation=ic_correlation,
        regime_ic=regime_ic,
        horizon=horizon,
        ic_window=ic_window,
        n_stocks=panel["close"].shape[1],
        n_days=panel["close"].shape[0],
        start_date=valid_dates.min() if len(valid_dates) else panel["close"].index.min(),
        end_date=valid_dates.max() if len(valid_dates) else panel["close"].index.max(),
    )


def pick_top_factors(
    result: FactorAnalysisResult,
    top_n: int = 20,
    max_correlation: float = 0.6,
    min_ir: float = 0.05,
    score_by: Literal["ir", "mean_ic", "abs_ic"] = "ir",
) -> list[str]:
    """Greedy de-correlation selection on a FactorAnalysisResult.

    Algorithm:
      1. Score = |result.ic_ir|  (or |mean_ic|, or abs_ic_mean — picked by ``score_by``).
      2. Drop factors with |ic_ir| < min_ir up front.
      3. Sort survivors by score descending.
      4. Walk the list; accept a factor iff its absolute IC-correlation with
         every already-accepted factor is < max_correlation.
      5. Stop when ``top_n`` factors accepted.

    Returns the picked factor names in selection order (highest-scored first).
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    if not (0 < max_correlation <= 1):
        raise ValueError(f"max_correlation must be in (0, 1], got {max_correlation}")
    if score_by == "ir":
        score = result.ic_ir.abs()
    elif score_by == "mean_ic":
        score = result.mean_ic.abs()
    elif score_by == "abs_ic":
        score = result.abs_ic_mean
    else:
        raise ValueError(f"unknown score_by: {score_by!r}")

    eligible = [
        n for n in result.factor_names
        if not pd.isna(score[n]) and abs(result.ic_ir.get(n, 0.0)) >= min_ir
    ]
    eligible.sort(key=lambda n: float(score[n]), reverse=True)

    picked: list[str] = []
    for name in eligible:
        if len(picked) >= top_n:
            break
        if any(abs(result.ic_correlation.loc[name, p]) >= max_correlation for p in picked):
            continue
        picked.append(name)
    return picked
