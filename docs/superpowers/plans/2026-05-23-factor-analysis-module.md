# Factor Analysis Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `factors_analysis` module + HTML/JSON report + `factors analyze` / `factors pick-by-ic` CLI subcommands that compute per-factor rolling IC / IR / half-life / pairwise correlation / regime IC on the pooled all-A panel, and pick a de-correlated top-N selection that can be written as a `factors_file`-compatible JSON.

**Architecture:**
- New module `stockpool.factors_analysis` — pure-function library on Panel + factor names → `FactorAnalysisResult` dataclass; `pick_top_factors` greedy de-correlation selector.
- New module `stockpool.factors_analysis_report` — pyecharts HTML renderer over `FactorAnalysisResult`.
- `stockpool.cli` — two new subcommands wired under existing `factors` parser.
- All numeric work delegates to existing `panel.build_panel_from_cache`, `ml.dataset.compute_factor_panel`, `ml.dataset.forward_return_panel`. No new external dependency.

**Tech Stack:** Python 3.10+, pandas, numpy, pyecharts (existing), pydantic (existing), pytest. No new dependencies.

---

## File Structure

| File | Role |
|------|------|
| `src/stockpool/factors_analysis.py` (new) | Core: `FactorAnalysisResult` dataclass, `compute_daily_ic`, `classify_regimes`, `_half_life_from_acf`, `analyze_factors`, `pick_top_factors`, JSON I/O |
| `src/stockpool/factors_analysis_report.py` (new) | pyecharts HTML render: ranking table, IC time-series multi-line, correlation heatmap, regime breakdown |
| `src/stockpool/cli.py` (modify, ~30 lines added) | Register `factors analyze` + `factors pick-by-ic` subcommands |
| `tests/test_factors_analysis.py` (new) | Unit tests for core module |
| `tests/test_factors_analysis_report.py` (new) | Smoke test for HTML render |
| `tests/test_cli_factors_analyze.py` (new) | CLI smoke tests |
| `CLAUDE.md` (modify) | Add `factors_analysis` to module map + commands |
| `README.md` (modify) | Add `factors analyze` + `factors pick-by-ic` to commands |

**Out of scope for this plan (deferred to plan-2):**
- `src/stockpool/factors/custom.py` (industry_relative_strength_20, limit_up_count_20, turnover_zscore_60)
- End-to-end A/B backtest comparison (old vs new factor set)

---

## Task 1: `FactorAnalysisResult` dataclass + JSON I/O

**Files:**
- Create: `src/stockpool/factors_analysis.py`
- Test: `tests/test_factors_analysis.py`

- [ ] **Step 1: Write the failing test for the dataclass**

Create `tests/test_factors_analysis.py`:

```python
"""Tests for stockpool.factors_analysis core."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpool.factors_analysis import (
    FactorAnalysisResult,
    analyze_factors,
    classify_regimes,
    compute_daily_ic,
    pick_top_factors,
)


def _make_result_fixture() -> FactorAnalysisResult:
    factor_names = ["f1", "f2", "f3"]
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    return FactorAnalysisResult(
        factor_names=factor_names,
        daily_ic={
            "f1": pd.Series([0.1] * 20, index=dates),
            "f2": pd.Series([-0.05] * 20, index=dates),
            "f3": pd.Series([0.0] * 20, index=dates),
        },
        mean_ic=pd.Series({"f1": 0.1, "f2": -0.05, "f3": 0.0}),
        ic_ir=pd.Series({"f1": 2.0, "f2": -1.0, "f3": 0.0}),
        abs_ic_mean=pd.Series({"f1": 0.1, "f2": 0.05, "f3": 0.0}),
        half_life=pd.Series({"f1": 10.0, "f2": 5.0, "f3": float("nan")}),
        ic_correlation=pd.DataFrame(
            [[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=factor_names, columns=factor_names,
        ),
        regime_ic={
            "bull": pd.Series({"f1": 0.15, "f2": -0.05, "f3": 0.0}),
            "bear": pd.Series({"f1": 0.05, "f2": -0.10, "f3": 0.0}),
            "sideways": pd.Series({"f1": 0.10, "f2": 0.0, "f3": 0.0}),
        },
        horizon=3,
        ic_window=20,
        n_stocks=10,
        n_days=20,
        start_date=dates[0],
        end_date=dates[-1],
    )


def test_factor_analysis_result_to_dict_roundtrip(tmp_path):
    res = _make_result_fixture()
    out_path = tmp_path / "result.json"
    res.to_json(out_path)
    loaded = FactorAnalysisResult.from_json(out_path)
    assert loaded.factor_names == res.factor_names
    assert loaded.horizon == 3
    assert loaded.n_stocks == 10
    pd.testing.assert_series_equal(loaded.mean_ic, res.mean_ic)
    pd.testing.assert_frame_equal(loaded.ic_correlation, res.ic_correlation)
    assert set(loaded.regime_ic.keys()) == {"bull", "bear", "sideways"}
    pd.testing.assert_series_equal(loaded.regime_ic["bull"], res.regime_ic["bull"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_factors_analysis.py::test_factor_analysis_result_to_dict_roundtrip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stockpool.factors_analysis'`

- [ ] **Step 3: Implement the dataclass + JSON I/O**

Create `src/stockpool/factors_analysis.py`:

```python
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
                    "values": v.tolist(),
                } for k, v in self.daily_ic.items()
            },
            "mean_ic": self.mean_ic.to_dict(),
            "ic_ir": self.ic_ir.to_dict(),
            "abs_ic_mean": self.abs_ic_mean.to_dict(),
            "half_life": self.half_life.to_dict(),
            "ic_correlation": {
                "index": list(self.ic_correlation.index),
                "columns": list(self.ic_correlation.columns),
                "values": self.ic_correlation.values.tolist(),
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
        ic_corr = pd.DataFrame(
            d["ic_correlation"]["values"],
            index=d["ic_correlation"]["index"],
            columns=d["ic_correlation"]["columns"],
        )
        return cls(
            factor_names=list(d["factor_names"]),
            daily_ic={
                k: pd.Series(v["values"], index=pd.to_datetime(v["index"]))
                for k, v in d["daily_ic"].items()
            },
            mean_ic=pd.Series(d["mean_ic"]),
            ic_ir=pd.Series(d["ic_ir"]),
            abs_ic_mean=pd.Series(d["abs_ic_mean"]),
            half_life=pd.Series(d["half_life"]),
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
def compute_daily_ic(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 2")


def classify_regimes(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 3")


def analyze_factors(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 5")


def pick_top_factors(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 6")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_factors_analysis.py::test_factor_analysis_result_to_dict_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors_analysis.py tests/test_factors_analysis.py
git commit -m "feat(factors): FactorAnalysisResult dataclass + JSON roundtrip"
```

---

## Task 2: `compute_daily_ic` (per-day cross-sectional IC)

**Files:**
- Modify: `src/stockpool/factors_analysis.py`
- Test: `tests/test_factors_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_factors_analysis.py`:

```python
def _synth_panel(n_days: int = 60, n_stocks: int = 8, seed: int = 0):
    """Build a deterministic OHLCV panel for unit tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    codes = [f"s{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, (n_days, n_stocks)), axis=0),
        index=dates, columns=codes,
    )
    panel = {
        "open":  close * 0.998,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": pd.DataFrame(
            rng.integers(1_000_000, 5_000_000, (n_days, n_stocks)).astype(float),
            index=dates, columns=codes,
        ),
    }
    return panel


def test_compute_daily_ic_perfect_negative_correlation():
    panel = _synth_panel(n_days=30, n_stocks=10, seed=1)
    # Forward return is close.pct_change(3).shift(-3).
    fwd = panel["close"].pct_change(3).shift(-3)
    # Factor = -forward_return → daily Spearman IC == -1 on rows where data is complete.
    factor = -fwd
    ic = compute_daily_ic(factor, fwd, method="spearman")
    # Drop rows where either side is all-NaN (head/tail).
    valid = ic.dropna()
    assert len(valid) >= 10
    assert (valid < -0.999).all(), f"expected IC ≈ -1, got {valid.head()}"


def test_compute_daily_ic_zero_for_random_factor():
    panel = _synth_panel(n_days=200, n_stocks=20, seed=2)
    fwd = panel["close"].pct_change(3).shift(-3)
    rng = np.random.default_rng(99)
    factor = pd.DataFrame(
        rng.normal(0, 1, fwd.shape), index=fwd.index, columns=fwd.columns,
    )
    ic = compute_daily_ic(factor, fwd, method="spearman").dropna()
    # Mean IC should be small (|μ| < 0.1) over 200 days.
    assert abs(ic.mean()) < 0.1, f"expected mean IC ≈ 0, got {ic.mean()}"


def test_compute_daily_ic_skips_constant_rows():
    """A day where the factor is constant across stocks must yield NaN, not 0/error."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    codes = ["a", "b", "c"]
    factor = pd.DataFrame(
        [[1.0, 1.0, 1.0],  # constant — IC should be NaN
         [1.0, 2.0, 3.0],
         [3.0, 2.0, 1.0],
         [1.0, 2.0, 3.0],
         [1.0, 2.0, 3.0]],
        index=dates, columns=codes,
    )
    fwd = pd.DataFrame(
        [[0.01, 0.02, 0.03]] * 5, index=dates, columns=codes,
    )
    ic = compute_daily_ic(factor, fwd, method="spearman")
    assert pd.isna(ic.iloc[0])
    assert ic.iloc[1] == pytest.approx(1.0)
    assert ic.iloc[2] == pytest.approx(-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_factors_analysis.py::test_compute_daily_ic_perfect_negative_correlation -v`
Expected: FAIL with `NotImplementedError: implemented in Task 2`

- [ ] **Step 3: Implement `compute_daily_ic`**

Replace the `compute_daily_ic` placeholder in `src/stockpool/factors_analysis.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "compute_daily_ic"`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors_analysis.py tests/test_factors_analysis.py
git commit -m "feat(factors): per-day cross-sectional IC computation"
```

---

## Task 3: `classify_regimes` (index-based bull/bear/sideways)

**Files:**
- Modify: `src/stockpool/factors_analysis.py`
- Test: `tests/test_factors_analysis.py`

**Design**: Take index close series, compute SMA over `sma_window` (default 60), label each day:
- `bull` if `close > sma` AND `sma` is rising (today's sma > sma 5 days ago)
- `bear` if `close < sma` AND `sma` is falling
- `sideways` otherwise

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_factors_analysis.py`:

```python
def test_classify_regimes_pure_uptrend_is_bull():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    # Linear uptrend → close always above SMA, SMA rising.
    close = pd.Series(np.linspace(100, 200, 120), index=dates)
    regimes = classify_regimes(close, sma_window=60)
    # First sma_window-ish days are warmup (NaN); after that should be "bull".
    tail = regimes.iloc[80:]
    assert (tail == "bull").all(), f"got {tail.value_counts()}"


def test_classify_regimes_pure_downtrend_is_bear():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    close = pd.Series(np.linspace(200, 100, 120), index=dates)
    regimes = classify_regimes(close, sma_window=60)
    tail = regimes.iloc[80:]
    assert (tail == "bear").all()


def test_classify_regimes_choppy_is_sideways():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    # Oscillation around 100 → SMA flat, close crosses repeatedly.
    close = pd.Series(100.0 + 5.0 * np.sin(np.linspace(0, 12 * np.pi, 120)), index=dates)
    regimes = classify_regimes(close, sma_window=60)
    tail = regimes.iloc[80:]
    sideways_share = (tail == "sideways").sum() / len(tail)
    assert sideways_share > 0.3, f"sideways share {sideways_share} too low"


def test_classify_regimes_warmup_is_nan():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    close = pd.Series(np.linspace(100, 200, 120), index=dates)
    regimes = classify_regimes(close, sma_window=60)
    assert regimes.iloc[0:30].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "classify_regimes"`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `classify_regimes`**

Replace the `classify_regimes` placeholder in `src/stockpool/factors_analysis.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "classify_regimes"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors_analysis.py tests/test_factors_analysis.py
git commit -m "feat(factors): regime classifier from index close"
```

---

## Task 4: `_half_life_from_acf` (IC autocorrelation half-life)

**Files:**
- Modify: `src/stockpool/factors_analysis.py`
- Test: `tests/test_factors_analysis.py`

**Design**: Given an IC time series, fit AR(1) and report ln(0.5) / ln(|ρ_1|). If ρ_1 ≤ 0 or NaN, return NaN. Cap at `max_half_life` (default 252) to avoid wild values from near-unit-root.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_factors_analysis.py`:

```python
from stockpool.factors_analysis import _half_life_from_acf


def test_half_life_ar1_known_decay():
    """AR(1) with ρ=0.5 → half-life = log(0.5)/log(0.5) = 1.0"""
    n = 500
    rng = np.random.default_rng(7)
    rho = 0.5
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, 1)
    series = pd.Series(x)
    hl = _half_life_from_acf(series)
    assert 0.7 < hl < 1.5, f"expected half-life ≈ 1.0, got {hl}"


def test_half_life_ar1_slow_decay():
    """AR(1) with ρ=0.9 → half-life = log(0.5)/log(0.9) ≈ 6.58"""
    n = 2000
    rng = np.random.default_rng(11)
    rho = 0.9
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, 1)
    series = pd.Series(x)
    hl = _half_life_from_acf(series)
    assert 5.0 < hl < 9.0, f"expected half-life ≈ 6.58, got {hl}"


def test_half_life_white_noise_is_nan_or_zero():
    """White noise has ρ ≈ 0, so half-life is either NaN (rho ≤ 0) or tiny."""
    rng = np.random.default_rng(13)
    series = pd.Series(rng.normal(0, 1, 500))
    hl = _half_life_from_acf(series)
    assert pd.isna(hl) or hl < 0.5, f"got {hl}"


def test_half_life_handles_nan_input():
    series = pd.Series([np.nan] * 10)
    hl = _half_life_from_acf(series)
    assert pd.isna(hl)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "half_life"`
Expected: FAIL with `ImportError: cannot import name '_half_life_from_acf'`

- [ ] **Step 3: Implement `_half_life_from_acf`**

Add to `src/stockpool/factors_analysis.py` (above the placeholder `analyze_factors`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "half_life"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors_analysis.py tests/test_factors_analysis.py
git commit -m "feat(factors): AR(1) half-life helper"
```

---

## Task 5: `analyze_factors` orchestrator

**Files:**
- Modify: `src/stockpool/factors_analysis.py`
- Test: `tests/test_factors_analysis.py`

**Design**: Take a panel + factor names, drive `compute_factor_panel`, `forward_return_panel`, then loop factors to compute daily IC → aggregate mean / IR / half-life / abs-mean → pairwise correlation matrix of daily IC series → optional regime split.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_factors_analysis.py`:

```python
def test_analyze_factors_end_to_end_synthetic():
    """One strong factor + one anti-factor + one noise factor on a synthetic panel."""
    panel = _synth_panel(n_days=120, n_stocks=15, seed=42)
    # Inject deterministic factors via the registry by computing them ourselves
    # and stubbing compute_factor_panel — but to keep things simple here we
    # use real registered factors and just verify the API.
    factor_names = ["momentum_20", "rsi_centered_14", "vol_ratio_5"]
    result = analyze_factors(
        panel=panel,
        factor_names=factor_names,
        horizon=3,
        ic_window=60,
    )
    assert result.factor_names == factor_names
    assert set(result.daily_ic.keys()) == set(factor_names)
    assert len(result.mean_ic) == 3
    assert len(result.ic_ir) == 3
    assert result.ic_correlation.shape == (3, 3)
    assert result.horizon == 3
    assert result.n_stocks == 15
    # No regime data when no index series given.
    assert result.regime_ic == {}


def test_analyze_factors_with_regime_index():
    panel = _synth_panel(n_days=200, n_stocks=10, seed=5)
    # Build a 200-day uptrending "index" so all post-warmup days are "bull".
    idx_close = pd.Series(
        np.linspace(100, 300, 200), index=panel["close"].index, name="sh000001",
    )
    result = analyze_factors(
        panel=panel,
        factor_names=["momentum_20", "rsi_centered_14"],
        horizon=3,
        ic_window=60,
        regime_index_close=idx_close,
    )
    # Should have at least the "bull" regime (and possibly only bull).
    assert "bull" in result.regime_ic
    assert len(result.regime_ic["bull"]) == 2


def test_analyze_factors_rejects_unknown_factor():
    panel = _synth_panel(n_days=60, n_stocks=5, seed=0)
    with pytest.raises(KeyError):
        analyze_factors(
            panel=panel, factor_names=["this_factor_does_not_exist"], horizon=3,
        )


def test_analyze_factors_ic_correlation_diagonal_is_one():
    panel = _synth_panel(n_days=120, n_stocks=10, seed=8)
    result = analyze_factors(
        panel=panel, factor_names=["momentum_20", "rsi_centered_14"],
        horizon=3, ic_window=60,
    )
    diag = np.diag(result.ic_correlation.values)
    assert np.allclose(diag, 1.0, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "analyze_factors"`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `analyze_factors`**

Replace the `analyze_factors` placeholder in `src/stockpool/factors_analysis.py`. Also add the import for the dataset helpers at the top of the file:

```python
# At the top of factors_analysis.py, with other imports:
from stockpool.ml.dataset import compute_factor_panel, forward_return_panel
```

Then replace the placeholder with:

```python
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

    fp = compute_factor_panel(panel, factor_names)
    fwd = forward_return_panel(panel["close"], horizon)

    daily_ic: dict[str, pd.Series] = {}
    for name in factor_names:
        daily_ic[name] = compute_daily_ic(fp[name], fwd, method=method)

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "analyze_factors"`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full test file to check no regressions**

Run: `python -m pytest tests/test_factors_analysis.py -v`
Expected: PASS (all tests in the file so far)

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/factors_analysis.py tests/test_factors_analysis.py
git commit -m "feat(factors): analyze_factors orchestrator + regime IC"
```

---

## Task 6: `pick_top_factors` (greedy de-correlation selector)

**Files:**
- Modify: `src/stockpool/factors_analysis.py`
- Test: `tests/test_factors_analysis.py`

**Design**: Sort by `|score_by|` desc, walk down the list, accept each factor iff its `|correlation|` with every already-accepted factor is < `max_correlation` AND `|ic_ir| >= min_ir`. Stop at `top_n`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_factors_analysis.py`:

```python
def _build_pick_fixture(factor_names, ir_values, corr_pairs=None):
    """Build a minimal FactorAnalysisResult for pick_top_factors tests."""
    n = len(factor_names)
    corr = pd.DataFrame(
        np.eye(n), index=factor_names, columns=factor_names,
    )
    for (a, b, v) in (corr_pairs or []):
        corr.loc[a, b] = v
        corr.loc[b, a] = v
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    return FactorAnalysisResult(
        factor_names=list(factor_names),
        daily_ic={n: pd.Series([0.0] * 10, index=dates) for n in factor_names},
        mean_ic=pd.Series(dict(zip(factor_names, ir_values))),
        ic_ir=pd.Series(dict(zip(factor_names, ir_values))),
        abs_ic_mean=pd.Series(dict(zip(factor_names, [abs(v) for v in ir_values]))),
        half_life=pd.Series(dict(zip(factor_names, [10.0] * n))),
        ic_correlation=corr,
        regime_ic={},
        horizon=3, ic_window=60, n_stocks=5, n_days=10,
        start_date=dates[0], end_date=dates[-1],
    )


def test_pick_top_factors_drops_correlated():
    res = _build_pick_fixture(
        factor_names=["a", "b", "c", "d"],
        ir_values=[0.5, 0.45, 0.4, 0.3],
        corr_pairs=[("a", "b", 0.8)],  # a and b too correlated
    )
    picked = pick_top_factors(res, top_n=3, max_correlation=0.6, min_ir=0.0)
    assert picked == ["a", "c", "d"]   # b is dropped, c/d are independent


def test_pick_top_factors_respects_min_ir():
    res = _build_pick_fixture(
        factor_names=["a", "b", "c", "d"],
        ir_values=[0.5, 0.4, 0.03, 0.01],
    )
    picked = pick_top_factors(res, top_n=4, max_correlation=0.6, min_ir=0.05)
    assert picked == ["a", "b"]


def test_pick_top_factors_uses_absolute_score():
    """Negative IR is still informative — sort by |ir|, not raw ir."""
    res = _build_pick_fixture(
        factor_names=["a", "b", "c"],
        ir_values=[0.1, -0.5, 0.3],
    )
    picked = pick_top_factors(res, top_n=2, max_correlation=0.99, min_ir=0.0)
    assert picked == ["b", "c"]


def test_pick_top_factors_returns_empty_when_all_below_threshold():
    res = _build_pick_fixture(
        factor_names=["a", "b"],
        ir_values=[0.01, 0.02],
    )
    picked = pick_top_factors(res, top_n=5, max_correlation=0.6, min_ir=0.1)
    assert picked == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "pick_top_factors"`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `pick_top_factors`**

Replace the `pick_top_factors` placeholder in `src/stockpool/factors_analysis.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_factors_analysis.py -v -k "pick_top_factors"`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full test file**

Run: `python -m pytest tests/test_factors_analysis.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/factors_analysis.py tests/test_factors_analysis.py
git commit -m "feat(factors): greedy de-correlation pick_top_factors"
```

---

## Task 7: HTML report renderer

**Files:**
- Create: `src/stockpool/factors_analysis_report.py`
- Test: `tests/test_factors_analysis_report.py`

**Design**: pyecharts-based HTML. Sections in order:
1. **Summary header**: # factors, n_stocks, n_days, horizon, date range
2. **Ranking table**: factor / mean_ic / ic_ir / abs_ic_mean / half_life / regime_ic columns
3. **IC time-series chart**: multi-line, one line per top-K factors (K=10)
4. **Correlation heatmap**: F × F (use pyecharts HeatMap)
5. **Regime breakdown table**: factor × regime grid (if regimes present)
6. **Picked selection box**: shows what `pick_top_factors(...)` returns with default params

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_factors_analysis_report.py`:

```python
"""Smoke test for stockpool.factors_analysis_report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpool.factors_analysis import FactorAnalysisResult
from stockpool.factors_analysis_report import render_factor_analysis_report


def _build_result_for_report():
    factor_names = ["alpha_001", "momentum_20", "rsi_centered_14"]
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    rng = np.random.default_rng(0)
    return FactorAnalysisResult(
        factor_names=factor_names,
        daily_ic={
            n: pd.Series(rng.normal(0.05, 0.1, 50), index=dates)
            for n in factor_names
        },
        mean_ic=pd.Series({"alpha_001": 0.08, "momentum_20": 0.05, "rsi_centered_14": -0.02}),
        ic_ir=pd.Series({"alpha_001": 0.6, "momentum_20": 0.3, "rsi_centered_14": -0.1}),
        abs_ic_mean=pd.Series({"alpha_001": 0.08, "momentum_20": 0.06, "rsi_centered_14": 0.04}),
        half_life=pd.Series({"alpha_001": 12.0, "momentum_20": 8.0, "rsi_centered_14": 3.0}),
        ic_correlation=pd.DataFrame(
            [[1.0, 0.3, -0.1], [0.3, 1.0, 0.2], [-0.1, 0.2, 1.0]],
            index=factor_names, columns=factor_names,
        ),
        regime_ic={
            "bull": pd.Series({"alpha_001": 0.10, "momentum_20": 0.08, "rsi_centered_14": -0.01}),
            "bear": pd.Series({"alpha_001": 0.05, "momentum_20": 0.02, "rsi_centered_14": -0.05}),
            "sideways": pd.Series({"alpha_001": 0.07, "momentum_20": 0.05, "rsi_centered_14": -0.02}),
        },
        horizon=3, ic_window=60, n_stocks=20, n_days=50,
        start_date=dates[0], end_date=dates[-1],
    )


def test_render_html_writes_file(tmp_path):
    result = _build_result_for_report()
    out = tmp_path / "report.html"
    render_factor_analysis_report(result, out_path=out, picked=["alpha_001", "momentum_20"])
    assert out.exists()
    assert out.stat().st_size > 2048
    html = out.read_text(encoding="utf-8")
    # Each factor name should appear at least once
    for n in result.factor_names:
        assert n in html, f"missing {n} in HTML"
    # Picked section should be visible
    assert "alpha_001" in html and "momentum_20" in html
    # Regime headers should be present
    for regime in ("bull", "bear", "sideways"):
        assert regime in html


def test_render_html_handles_empty_regimes(tmp_path):
    result = _build_result_for_report()
    result.regime_ic.clear()
    out = tmp_path / "report.html"
    render_factor_analysis_report(result, out_path=out, picked=[])
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # No regime section when empty — but the document still renders
    assert "alpha_001" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_factors_analysis_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stockpool.factors_analysis_report'`

- [ ] **Step 3: Implement the renderer**

Create `src/stockpool/factors_analysis_report.py`:

```python
"""HTML report for FactorAnalysisResult.

Renders a single self-contained HTML file with:
  * summary header (n_factors / n_stocks / n_days / date range / horizon)
  * ranking table (factor / mean_ic / ic_ir / abs_ic_mean / half_life)
  * IC time-series multi-line chart (top-10 by |ic_ir|)
  * correlation heatmap (F × F)
  * regime IC table (factor × regime)
  * picked selection box

Uses pyecharts (already a project dep) — no new dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import HeatMap, Line, Page

from stockpool.factors_analysis import FactorAnalysisResult


def _summary_html(result: FactorAnalysisResult) -> str:
    return (
        '<div style="font-family:sans-serif;padding:12px 18px;background:#f5f7fa;'
        'border-radius:6px;margin-bottom:18px">'
        f'<b>因子分析报告</b> &nbsp;|&nbsp; '
        f'因子数 {len(result.factor_names)} &nbsp;|&nbsp; '
        f'股票数 {result.n_stocks} &nbsp;|&nbsp; '
        f'交易日 {result.n_days} &nbsp;|&nbsp; '
        f'horizon {result.horizon} &nbsp;|&nbsp; '
        f'区间 {result.start_date.date()} → {result.end_date.date()}'
        '</div>'
    )


def _ranking_table_html(result: FactorAnalysisResult) -> str:
    rows = []
    for n in result.factor_names:
        rows.append({
            "factor": n,
            "mean_ic": float(result.mean_ic[n]),
            "ic_ir": float(result.ic_ir[n]),
            "abs_ic_mean": float(result.abs_ic_mean[n]),
            "half_life": float(result.half_life[n]),
        })
    df = pd.DataFrame(rows).sort_values("ic_ir", key=lambda s: s.abs(), ascending=False)
    return (
        '<h3 style="font-family:sans-serif">因子排名 (按 |IC IR| 降序)</h3>'
        + df.to_html(
            index=False, float_format="%.4f", border=0,
            classes="ranking-table",
        )
    )


def _ic_timeseries_chart(result: FactorAnalysisResult, top_k: int = 10) -> Line:
    top = result.ic_ir.abs().sort_values(ascending=False).head(top_k).index.tolist()
    line = Line(init_opts=opts.InitOpts(width="1100px", height="380px"))
    if not top:
        return line
    dates = result.daily_ic[top[0]].index
    line.add_xaxis([d.strftime("%Y-%m-%d") for d in dates])
    for n in top:
        smooth_ic = result.daily_ic[n].rolling(20, min_periods=5).mean()
        line.add_yaxis(
            n, [None if pd.isna(v) else round(v, 4) for v in smooth_ic.tolist()],
            is_symbol_show=False, is_smooth=True,
        )
    line.set_global_opts(
        title_opts=opts.TitleOpts(title=f"Top-{len(top)} 因子 20 日滚动 IC"),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(type_="value", min_=-0.3, max_=0.3),
        legend_opts=opts.LegendOpts(pos_top="bottom"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts(type_="inside")],
    )
    return line


def _correlation_heatmap(result: FactorAnalysisResult) -> HeatMap:
    corr = result.ic_correlation
    names = list(corr.index)
    data = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            data.append([i, j, round(float(corr.iloc[i, j]), 3)])
    hm = HeatMap(init_opts=opts.InitOpts(width="900px", height="650px"))
    hm.add_xaxis(names)
    hm.add_yaxis("IC corr", names, data)
    hm.set_global_opts(
        title_opts=opts.TitleOpts(title="因子 IC 相关性热图"),
        visualmap_opts=opts.VisualMapOpts(
            min_=-1, max_=1, range_color=["#3060cf", "#ffffff", "#c4463a"],
            pos_left="right",
        ),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60)),
    )
    return hm


def _regime_table_html(result: FactorAnalysisResult) -> str:
    if not result.regime_ic:
        return ""
    df = pd.DataFrame(result.regime_ic)[list(result.regime_ic.keys())]
    df.index.name = "factor"
    return (
        '<h3 style="font-family:sans-serif">不同 regime 下的均值 IC</h3>'
        + df.reset_index().to_html(
            index=False, float_format="%.4f", border=0,
            classes="regime-table",
        )
    )


def _picked_box_html(picked: Sequence[str]) -> str:
    if not picked:
        return ""
    items = ", ".join(f"<code>{n}</code>" for n in picked)
    return (
        '<div style="font-family:sans-serif;padding:12px 18px;'
        'background:#fffceb;border-left:4px solid #f5c518;margin:18px 0">'
        f'<b>pick_top_factors 默认参数下选出</b>({len(picked)} 个):<br>{items}'
        '</div>'
    )


def render_factor_analysis_report(
    result: FactorAnalysisResult,
    out_path: str | Path,
    picked: Sequence[str] = (),
) -> Path:
    """Render `result` to a single self-contained HTML file at `out_path`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page = Page(layout=Page.SimplePageLayout)
    page.add(_ic_timeseries_chart(result))
    page.add(_correlation_heatmap(result))

    html_chunks = [
        _summary_html(result),
        _picked_box_html(picked),
        _ranking_table_html(result),
        _regime_table_html(result),
    ]

    base_html = page.render_embed()
    head = (
        '<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"/>'
        '<title>因子分析报告</title>'
        '<style>'
        'body{font-family:Segoe UI, "PingFang SC", sans-serif;margin:24px;}'
        'table{border-collapse:collapse;font-size:13px;margin:8px 0 24px;}'
        'th{background:#f0f3f7;padding:6px 10px;text-align:left;}'
        'td{padding:4px 10px;border-bottom:1px solid #e6e9ed;}'
        '</style></head><body>'
    )
    out_path.write_text(
        head + "\n".join(html_chunks) + base_html + "</body></html>",
        encoding="utf-8",
    )
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_factors_analysis_report.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors_analysis_report.py tests/test_factors_analysis_report.py
git commit -m "feat(factors): HTML report renderer for FactorAnalysisResult"
```

---

## Task 8: CLI `factors analyze` subcommand

**Files:**
- Modify: `src/stockpool/cli.py` (add subcommand + handler)
- Test: `tests/test_cli_factors_analyze.py`

**Design**: New `cmd_factors_analyze(args)` function. Loads config, builds the panel from cache (universe = `all` reads from `data/universe.parquet`, else from `cfg.stocks`), runs `analyze_factors`, writes both JSON and HTML to `--output` dir, returns 0.

- [ ] **Step 1: Write the failing CLI smoke test**

Create `tests/test_cli_factors_analyze.py`:

```python
"""Smoke tests for `python -m stockpool factors analyze` and `pick-by-ic`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_cache(tmp_path):
    """Seed the cache with 3 synthetic stocks so factor_panel has columns to rank across."""
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    rng = np.random.default_rng(7)
    n = 200
    for code in ("605589", "603986", "000528"):
        returns = rng.normal(0.0005, 0.02, n)
        close = 100.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "open": close * 0.998,
            "high": close * 1.005,
            "low":  close * 0.995,
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        })
        df.to_parquet(cache_dir / f"{code}_daily.parquet", index=False)
    return cache_dir


def _make_config(tmp_path, cache_dir):
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(cache_dir)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_file


def test_factors_analyze_cli_writes_outputs(tmp_path, isolated_cache):
    cfg_file = _make_config(tmp_path, isolated_cache)
    out_dir = tmp_path / "factor_analysis"
    rc = main([
        "factors", "analyze",
        "--config", str(cfg_file),
        "--universe", "pool",
        "--factors", "momentum_20", "rsi_centered_14", "vol_ratio_5",
        "--horizon", "3",
        "--output", str(out_dir),
    ])
    assert rc == 0
    html_files = list(out_dir.glob("*.html"))
    json_files = list(out_dir.glob("*.json"))
    assert len(html_files) == 1
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["factor_names"] == ["momentum_20", "rsi_centered_14", "vol_ratio_5"]
    assert payload["n_stocks"] == 3
    assert payload["horizon"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_factors_analyze.py::test_factors_analyze_cli_writes_outputs -v`
Expected: FAIL with `argparse.ArgumentError` or `invalid choice: 'analyze'`

- [ ] **Step 3: Implement the CLI handler + subcommand registration**

In `src/stockpool/cli.py`, add a new function near `cmd_fetch_universe` (around line 358):

```python
def cmd_factors_analyze(args: argparse.Namespace) -> int:
    """Analyze factors on the pooled panel and write HTML + JSON reports."""
    from datetime import date
    from stockpool.factors import list_factors
    from stockpool.factors_analysis import analyze_factors
    from stockpool.factors_analysis_report import render_factor_analysis_report
    from stockpool.panel import build_panel_from_cache

    cfg = load_config(args.config)
    cache_dir = Path(cfg.data.cache_dir)

    if args.universe == "all":
        codes = list_universe(cache_dir)
        if not codes:
            log.error("universe=all but data/universe.parquet is empty; "
                      "run `python -m stockpool fetch-universe` first")
            return 1
    else:
        codes = [s.code for s in cfg.stocks]

    factor_names = list(args.factors) if args.factors else list_factors()
    log.info("Analyzing %d factors over %d stocks (universe=%s)",
             len(factor_names), len(codes), args.universe)

    panel = build_panel_from_cache(codes, cfg.data.history_days, cache_dir)

    regime_close = None
    if not args.no_regime:
        idx_code = cfg.context.indices[0].code if cfg.context.indices else None
        if idx_code:
            idx_path = cache_dir / f"idx_{idx_code}.parquet"
            if idx_path.exists():
                idx_df = pd.read_parquet(idx_path)
                idx_df["date"] = pd.to_datetime(idx_df["date"])
                regime_close = idx_df.set_index("date").sort_index()["close"]
            else:
                log.warning("regime index cache missing (%s); skipping regime split", idx_path)

    result = analyze_factors(
        panel=panel,
        factor_names=factor_names,
        horizon=args.horizon,
        ic_window=args.ic_window,
        regime_index_close=regime_close,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    json_path = out_dir / f"{stamp}.json"
    html_path = out_dir / f"{stamp}.html"
    latest_html = out_dir / "latest.html"

    result.to_json(json_path)
    render_factor_analysis_report(result, html_path)
    if latest_html.exists() or latest_html.is_symlink():
        latest_html.unlink()
    latest_html.write_bytes(html_path.read_bytes())

    log.info("Wrote %s and %s", json_path, html_path)
    return 0
```

Now register the subcommand. Find the existing `# `factors` sub-tree: list / show / pick` block (around line 505) and append (after the `p_pick.set_defaults(func=cli_pick)` line, before `args = parser.parse_args(argv)`):

```python
    p_analyze = fsub.add_parser(
        "analyze",
        help="Compute rolling IC / IR / half-life / correlation across factors",
    )
    p_analyze.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_analyze.add_argument(
        "--universe", choices=["pool", "all"], default="pool",
        help="pool = cfg.stocks; all = data/universe.parquet (needs fetch-universe first)",
    )
    p_analyze.add_argument(
        "--factors", nargs="*", default=None,
        help="Factor names (default: all registered factors)",
    )
    p_analyze.add_argument("--horizon", type=int, default=3)
    p_analyze.add_argument("--ic-window", type=int, default=252,
                           help="Metadata only — daily IC uses the full window")
    p_analyze.add_argument("--no-regime", action="store_true",
                           help="Skip the bull/bear/sideways regime split")
    p_analyze.add_argument(
        "--output", default="reports/factor_analysis",
        help="Output directory (HTML + JSON written here)",
    )
    p_analyze.set_defaults(func=cmd_factors_analyze)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_cli_factors_analyze.py::test_factors_analyze_cli_writes_outputs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/cli.py tests/test_cli_factors_analyze.py
git commit -m "feat(cli): factors analyze subcommand"
```

---

## Task 9: CLI `factors pick-by-ic` subcommand

**Files:**
- Modify: `src/stockpool/cli.py`
- Test: `tests/test_cli_factors_analyze.py`

**Design**: Read a `FactorAnalysisResult` JSON, call `pick_top_factors`, write a `selection.json` compatible with the existing `factors_file` loader (`{"factors": [...]}` shape).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_factors_analyze.py`:

```python
def test_factors_pick_by_ic_writes_selection(tmp_path, isolated_cache):
    cfg_file = _make_config(tmp_path, isolated_cache)
    analyze_dir = tmp_path / "factor_analysis"
    rc = main([
        "factors", "analyze",
        "--config", str(cfg_file),
        "--universe", "pool",
        "--factors", "momentum_20", "rsi_centered_14", "vol_ratio_5",
        "--horizon", "3",
        "--output", str(analyze_dir),
    ])
    assert rc == 0
    json_files = list(analyze_dir.glob("[0-9]*.json"))
    assert len(json_files) == 1
    input_json = json_files[0]

    selection_path = tmp_path / "selection.json"
    rc = main([
        "factors", "pick-by-ic",
        "--input", str(input_json),
        "--output", str(selection_path),
        "--top-n", "2",
        "--max-corr", "0.99",
        "--min-ir", "0.0",
    ])
    assert rc == 0
    assert selection_path.exists()
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    assert "factors" in payload
    assert isinstance(payload["factors"], list)
    assert 0 < len(payload["factors"]) <= 2
    for n in payload["factors"]:
        assert n in {"momentum_20", "rsi_centered_14", "vol_ratio_5"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_cli_factors_analyze.py::test_factors_pick_by_ic_writes_selection -v`
Expected: FAIL with `invalid choice: 'pick-by-ic'`

- [ ] **Step 3: Implement the handler + subcommand**

Add to `src/stockpool/cli.py` after `cmd_factors_analyze`:

```python
def cmd_factors_pick_by_ic(args: argparse.Namespace) -> int:
    """Pick a de-correlated top-N from a FactorAnalysisResult JSON."""
    import json
    from stockpool.factors_analysis import FactorAnalysisResult, pick_top_factors

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("input JSON not found: %s", input_path)
        return 1

    result = FactorAnalysisResult.from_json(input_path)
    picked = pick_top_factors(
        result,
        top_n=args.top_n,
        max_correlation=args.max_corr,
        min_ir=args.min_ir,
        score_by=args.score_by,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"factors": picked}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Picked %d factors → %s", len(picked), output_path)
    return 0
```

Register the subcommand. Append after the `p_analyze.set_defaults(func=cmd_factors_analyze)` line:

```python
    p_pick_ic = fsub.add_parser(
        "pick-by-ic",
        help="From a factors-analyze JSON, pick a top-N de-correlated selection.json",
    )
    p_pick_ic.add_argument(
        "--input", required=True,
        help="Path to a factors-analyze JSON (e.g. reports/factor_analysis/2026-05-23.json)",
    )
    p_pick_ic.add_argument(
        "--output", default="reports/selection.json",
        help="Output selection.json (consumed by MLFactorConfig.factors_file)",
    )
    p_pick_ic.add_argument("--top-n", type=int, default=20)
    p_pick_ic.add_argument("--max-corr", type=float, default=0.6)
    p_pick_ic.add_argument("--min-ir", type=float, default=0.05)
    p_pick_ic.add_argument(
        "--score-by", choices=["ir", "mean_ic", "abs_ic"], default="ir",
    )
    p_pick_ic.set_defaults(func=cmd_factors_pick_by_ic)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_cli_factors_analyze.py::test_factors_pick_by_ic_writes_selection -v`
Expected: PASS

- [ ] **Step 5: Run the full CLI test file**

Run: `python -m pytest tests/test_cli_factors_analyze.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/cli.py tests/test_cli_factors_analyze.py
git commit -m "feat(cli): factors pick-by-ic subcommand"
```

---

## Task 10: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

Per the project rule (CLAUDE.md "改动后更新文档"):
> 新增 / 删除 / 重命名顶层模块或公开 API → 同一次改动里把两份文档一并改完

- [ ] **Step 1: Update CLAUDE.md module map**

In `CLAUDE.md`, find the **模块地图** table and add new rows (alphabetical insertion within the relevant block):

```
| `src/stockpool/factors_analysis.py` | **因子分析**: 滚动 IC / IR / half-life / 相关性 / regime 切片;`analyze_factors` + `pick_top_factors` |
| `src/stockpool/factors_analysis_report.py` | pyecharts HTML 报告: 排名表 + IC 时序 + 相关性 heatmap + regime 拆分 |
```

In the **快速命令** section, add:

```bash
# 因子分析 (跑一次输出 HTML + JSON 报告)
python -m stockpool factors analyze --universe all --output reports/factor_analysis

# 从分析 JSON 自动选 top-N 去相关因子,写成 ml_factor.factors_file 兼容的 selection.json
python -m stockpool factors pick-by-ic \
  --input reports/factor_analysis/<日期>.json \
  --output reports/selection.json \
  --top-n 20 --max-corr 0.6 --min-ir 0.05
```

In the **测试** section table, add:

```
| `test_factors_analysis.py` | FactorAnalysisResult / compute_daily_ic / classify_regimes / half-life / analyze_factors / pick_top_factors |
| `test_factors_analysis_report.py` | HTML 渲染烟雾 + 空 regime 处理 |
| `test_cli_factors_analyze.py` | `factors analyze` 与 `factors pick-by-ic` CLI 烟雾 |
```

- [ ] **Step 2: Update README.md**

In `README.md`, find the **常用命令** section and add the two `factors analyze` / `pick-by-ic` examples (use the same snippet from Step 1).

In the **快速开始** section, add a one-line example after the existing `factors` usage if present, or add a new bullet point: "**选因子**: `python -m stockpool factors analyze --universe all && python -m stockpool factors pick-by-ic --input reports/factor_analysis/$(date +%F).json --output reports/selection.json`"

- [ ] **Step 3: Verify both files render correctly**

Run: `head -50 CLAUDE.md` and `head -50 README.md` — confirm no syntax breakage.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: factors analyze / pick-by-ic in CLAUDE.md and README.md"
```

---

## Task 11: Full test suite + smoke check

**Files:** (none modified)

- [ ] **Step 1: Run the entire test suite to confirm zero regressions**

Run: `python -m pytest tests/ -q`
Expected: PASS — count should be **previous count + 24** (Task 1: 1, Task 2: 3, Task 3: 4, Task 4: 4, Task 5: 4, Task 6: 4, Task 7: 2, Task 8: 1, Task 9: 1).

- [ ] **Step 2: End-to-end smoke against a fresh cache**

If the user has previously run `python -m stockpool fetch-universe`, do:

```bash
python -m stockpool factors analyze --universe all --output reports/factor_analysis
```

Confirm two output files exist:
```bash
ls reports/factor_analysis/
```
Expected: `<YYYY-MM-DD>.html`, `<YYYY-MM-DD>.json`, `latest.html`

Then:
```bash
python -m stockpool factors pick-by-ic \
  --input reports/factor_analysis/<YYYY-MM-DD>.json \
  --output reports/selection.json \
  --top-n 20 --max-corr 0.6 --min-ir 0.05
```

Confirm `reports/selection.json` exists and parses as `{"factors": [...]}`.

- [ ] **Step 3: (Optional) Wire the new selection into config and run a backtest**

Edit `config.yaml`:

```yaml
strategy:
  name: ml_factor
  ml_factor:
    factors: []                       # leave empty so factors_file takes over
    factors_file: reports/selection.json
```

Run: `python -m stockpool backtest --config config.yaml`
Expected: backtest produces `reports/backtest/latest.html` using the new factor set. Compare net equity / sharpe with the previous (default factor) run.

**Note:** Step 3 is optional within this plan — the A/B comparison itself is plan-2's verification gate.

- [ ] **Step 4: Final commit (if any uncommitted state)**

```bash
git status
# If clean: no commit needed.
# If something is uncommitted (e.g. config tweak), discuss with reviewer first.
```

---

## Self-Review Notes (filled during plan write)

**Spec coverage** (against `docs/strategy_improvement_2026.md` §3.3):
- ✅ `factors_analysis.py` module — Tasks 1-6
- ✅ HTML report — Task 7
- ✅ CLI subcommands `factors analyze` + `factors pick-by-ic` — Tasks 8-9
- ⏭️ `factors/custom.py` (industry_relative_strength_20 etc) — **deferred to plan-2**
- ⏭️ A/B 对比回测 — **deferred to plan-2** (Task 11 step 3 is optional preview only)

**Placeholders:** none.

**Type consistency check:**
- `FactorAnalysisResult.daily_ic` is `dict[str, pd.Series]` — used consistently in all tasks.
- `FactorAnalysisResult.ic_correlation` is `pd.DataFrame` with both row + col indexed by factor name — used consistently.
- `analyze_factors(panel=..., factor_names=..., horizon=..., ic_window=..., regime_index_close=..., method=...)` — all callers (Task 5, 8) match this signature.
- `pick_top_factors(result, top_n, max_correlation, min_ir, score_by)` — Tasks 6, 9 match.
- `render_factor_analysis_report(result, out_path, picked=())` — Tasks 7, 8 match (Task 8 passes `picked` as positional from a derived list — verify in Step 3 of Task 8 if needed; current code does NOT pass picked to render_factor_analysis_report in Task 8, which is intentional: the analyze CLI doesn't pick; the user runs pick-by-ic separately).

---
