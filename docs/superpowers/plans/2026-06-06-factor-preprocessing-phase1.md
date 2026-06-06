# Factor Preprocessing Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cross-sectional preprocessing pipeline (winsorize / cs-zscore / industry-neutralize) to ML factor panels behind a default-off config switch, then validate via A/B test.

**Architecture:** New stateless module `src/stockpool/ml/preprocess.py` with 3 step functions + 1 pipeline driver. Wired into `build_factor_panel` after `compute_factor_panel` and before `load_or_build_factor_panel` caches output. Preprocess config is part of factor-panel cache key (when not all-off); when all-off, cache key is byte-identical to pre-PR baseline → existing caches stay valid.

**Tech Stack:** Python 3.11, pandas, pydantic, pytest. Conforms to existing stockpool patterns: `.venv/Scripts/python.exe` for commands, `ConfigDict(extra="forbid")` for pydantic models, panel-first APIs.

**Spec:** `docs/superpowers/specs/2026-06-06-factor-preprocessing-phase1-design.md`

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/stockpool/ml/preprocess.py` | Create | 3 stateless preprocess functions + pipeline driver |
| `src/stockpool/config.py` | Modify (add ~30 lines) | New `PreprocessConfig` class; `MLFactorConfig.preprocess` field |
| `src/stockpool/strategy_factory.py` | Modify (~40 lines) | Thread `preprocess_cfg` through `build_factor_panel`, `_factor_panel_sig`, `load_or_build_factor_panel`; add `_is_all_off()` helper |
| `src/stockpool/cli.py` | Modify (1 line) | `load_or_build_factor_panel(..., preprocess_cfg=...)` at line 451 |
| `src/stockpool/backtest_runner.py` | Modify (1 line) | Same at line 90 |
| `tests/test_ml_preprocess.py` | Create (~250 lines) | 18 unit cases for the new module |
| `tests/test_factor_panel_cache.py` | Modify (+4 cases) | Cache-sig backwards-compat + invalidation |
| `tests/test_config.py` | Modify (+3 cases) | PreprocessConfig validation |
| `tests/test_ml_strategy.py` | Modify (+1 case) | End-to-end smoke |
| `ab_preprocess.yaml` | Create | A/B config: baseline vs three-step preprocess |
| `CLAUDE.md` | Modify | Document new module, config field, test file (per "改动后更新文档" rule) |
| `README.md` | Modify | Mention `preprocess` field in ml_factor example |
| `docs/ab_validation_results.md` | Modify (post-AB) | Append Phase 1 verdict |
| `docs/research/2026-06-06-factor-preprocessing-and-orthogonalization.md` | Modify (post-AB) | Note Phase 1 outcome |

---

## Task 1: Add `PreprocessConfig` to config.py

**Files:**
- Modify: `src/stockpool/config.py` (insert after line 365, before `class MLFactorConfig`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Open `tests/test_config.py` and append:

```python
def test_preprocess_config_defaults_all_off():
    """PreprocessConfig with no args has all three steps disabled."""
    from stockpool.config import PreprocessConfig
    cfg = PreprocessConfig()
    assert cfg.winsorize is None
    assert cfg.zscore is False
    assert cfg.industry_neutralize is False


def test_preprocess_winsorize_invalid_bounds_raises():
    """winsorize bounds must satisfy 0 < lo < hi < 1."""
    import pytest
    from pydantic import ValidationError
    from stockpool.config import PreprocessConfig

    with pytest.raises(ValidationError):
        PreprocessConfig(winsorize=(0.99, 0.01))  # reversed
    with pytest.raises(ValidationError):
        PreprocessConfig(winsorize=(0.0, 0.99))   # lo <= 0
    with pytest.raises(ValidationError):
        PreprocessConfig(winsorize=(0.01, 1.0))   # hi >= 1
    with pytest.raises(ValidationError):
        PreprocessConfig(winsorize=(0.5, 0.5))    # lo == hi


def test_preprocess_extra_field_forbidden():
    """Extra fields rejected (extra=forbid)."""
    import pytest
    from pydantic import ValidationError
    from stockpool.config import PreprocessConfig

    with pytest.raises(ValidationError):
        PreprocessConfig(winsorize=(0.01, 0.99), unknown_field=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py::test_preprocess_config_defaults_all_off tests/test_config.py::test_preprocess_winsorize_invalid_bounds_raises tests/test_config.py::test_preprocess_extra_field_forbidden -v`
Expected: 3 errors with `ImportError: cannot import name 'PreprocessConfig'`.

- [ ] **Step 3: Implement `PreprocessConfig`**

In `src/stockpool/config.py`, insert immediately before `class MLFactorConfig` (currently at line 367):

```python
class PreprocessConfig(BaseModel):
    """Cross-sectional preprocessing pipeline for ML factor panels.

    Applied at ``build_factor_panel()`` output, before disk caching. Affects
    ml_factor training, predict, and downstream Pool B consumers identically.
    Default = all off → fully backwards compatible (cache sig unchanged).

    See: docs/superpowers/specs/2026-06-06-factor-preprocessing-phase1-design.md
    """
    model_config = ConfigDict(extra="forbid")

    winsorize: tuple[float, float] | None = None
    zscore: bool = False
    industry_neutralize: bool = False

    @field_validator("winsorize")
    @classmethod
    def _check_winsorize_bounds(cls, v):
        if v is None:
            return None
        lo, hi = v
        if not (0 < lo < hi < 1):
            raise ValueError(
                f"winsorize bounds must satisfy 0 < lo < hi < 1, got ({lo}, {hi})"
            )
        return (float(lo), float(hi))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k preprocess`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "feat(config): add PreprocessConfig with bounds validation

PreprocessConfig holds the 3 cross-sectional preprocess switches
(winsorize / zscore / industry_neutralize). Default = all off.
Pydantic field_validator enforces 0 < lo < hi < 1 for winsorize.

Part of factor-preprocessing Phase 1 (see spec
2026-06-06-factor-preprocessing-phase1-design.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire `PreprocessConfig` into `MLFactorConfig`

**Files:**
- Modify: `src/stockpool/config.py:428` (after the `mask:` line)
- Test: `tests/test_config.py` (add 1 case)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_ml_factor_config_includes_preprocess_default():
    """MLFactorConfig has preprocess sub-config defaulting to all-off."""
    from stockpool.config import MLFactorConfig, PreprocessConfig
    cfg = MLFactorConfig()
    assert isinstance(cfg.preprocess, PreprocessConfig)
    assert cfg.preprocess.winsorize is None
    assert cfg.preprocess.zscore is False
    assert cfg.preprocess.industry_neutralize is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py::test_ml_factor_config_includes_preprocess_default -v`
Expected: `AttributeError: 'MLFactorConfig' object has no attribute 'preprocess'`.

- [ ] **Step 3: Add field**

In `src/stockpool/config.py`, find the line in `MLFactorConfig`:

```python
    mask: MaskConfig = Field(default_factory=MaskConfig)
```

Insert immediately after:

```python
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
```

- [ ] **Step 4: Run test + full config tests to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: all green (existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "feat(config): wire PreprocessConfig into MLFactorConfig

Adds MLFactorConfig.preprocess field with default_factory so
all existing yaml configs continue to work unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Implement `winsorize_panel` in `ml/preprocess.py`

**Files:**
- Create: `src/stockpool/ml/preprocess.py`
- Create: `tests/test_ml_preprocess.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ml_preprocess.py`:

```python
"""Unit tests for cross-sectional factor preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_panel(n_days=5, n_stocks=30, seed=0):
    """Synthetic factor panel: T × N DataFrame, dates index, codes columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    values = rng.standard_normal((n_days, n_stocks))
    return pd.DataFrame(values, index=dates, columns=codes)


def test_winsorize_clips_to_quantile():
    """Values outside [lo quantile, hi quantile] are clipped per day."""
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel(n_days=3, n_stocks=100, seed=1)
    out = winsorize_panel(df, 0.05, 0.95)
    for d in df.index:
        row = df.loc[d]
        lo_q = row.quantile(0.05)
        hi_q = row.quantile(0.95)
        assert out.loc[d].min() >= lo_q - 1e-9
        assert out.loc[d].max() <= hi_q + 1e-9


def test_winsorize_all_nan_row_passthrough():
    """A day with all-NaN cross-section is returned unchanged."""
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel(n_days=3, n_stocks=10, seed=2)
    df.iloc[1] = np.nan
    out = winsorize_panel(df, 0.01, 0.99)
    assert out.iloc[1].isna().all()
    assert out.shape == df.shape


def test_winsorize_invalid_bounds_raises():
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel()
    with pytest.raises(ValueError):
        winsorize_panel(df, 0.99, 0.01)
    with pytest.raises(ValueError):
        winsorize_panel(df, 0.5, 0.5)


def test_winsorize_preserves_index_columns():
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel()
    out = winsorize_panel(df, 0.01, 0.99)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -v`
Expected: `ImportError: cannot import name 'winsorize_panel'`.

- [ ] **Step 3: Create the module + winsorize_panel**

Create `src/stockpool/ml/preprocess.py`:

```python
"""Cross-sectional preprocessing pipeline for ML factor panels.

Three stateless steps, applied per-day (cross-sectional):

  * ``winsorize_panel(df, lo, hi)``  — clip to per-day [lo, hi] quantiles
  * ``cs_zscore_panel(df)``           — per-day (x - μ_t) / σ_t
  * ``industry_neutralize_panel(df, sector_map)``
                                      — per-day within-industry demean

Wrapped by ``apply_preprocess_pipeline`` which honors a ``PreprocessConfig``.

Look-ahead safe: each function consumes only per-day cross-sectional info,
never references other rows. See spec
``docs/superpowers/specs/2026-06-06-factor-preprocessing-phase1-design.md``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stockpool.config import PreprocessConfig

log = logging.getLogger(__name__)


def winsorize_panel(
    df: pd.DataFrame, lower: float, upper: float,
) -> pd.DataFrame:
    """Per-day cross-sectional clip to ``[lower quantile, upper quantile]``.

    Args:
        df: T × N factor wide-frame (date index, code columns).
        lower: lower quantile bound, e.g. ``0.01``.
        upper: upper quantile bound, e.g. ``0.99``.

    Returns:
        Same-shape DataFrame with values outside [q_lo(t), q_hi(t)] clipped.
        All-NaN rows are returned unchanged (shape preserved).

    Raises:
        ValueError: if not ``0 < lower < upper < 1``.
    """
    if not (0 < lower < upper < 1):
        raise ValueError(
            f"winsorize bounds must satisfy 0 < lower < upper < 1, "
            f"got ({lower}, {upper})"
        )
    lo_q = df.quantile(lower, axis=1)
    hi_q = df.quantile(upper, axis=1)
    out = df.clip(lower=lo_q, upper=hi_q, axis=0)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess.py
git commit -m "feat(ml): add winsorize_panel cross-sectional clipper

Per-day cross-sectional quantile clip. All-NaN rows pass through
unchanged; bounds validated 0 < lo < hi < 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Implement `cs_zscore_panel`

**Files:**
- Modify: `src/stockpool/ml/preprocess.py`
- Modify: `tests/test_ml_preprocess.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ml_preprocess.py`:

```python
def test_cs_zscore_mean_zero_std_one():
    """After per-day cs zscore, each row has μ ≈ 0 and σ ≈ 1."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=3, n_stocks=50, seed=3)
    out = cs_zscore_panel(df)
    for d in df.index:
        row = out.loc[d].dropna()
        assert abs(row.mean()) < 1e-9
        assert abs(row.std(ddof=0) - 1.0) < 1e-9


def test_cs_zscore_constant_row_returns_zero():
    """A day where every stock has identical value → returns zeros (σ < 1e-12)."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=3, n_stocks=10, seed=4)
    df.iloc[1] = 7.5  # constant row
    out = cs_zscore_panel(df)
    assert (out.iloc[1] == 0.0).all()


def test_cs_zscore_handles_nan():
    """Partial NaN row: zscore computed on non-NaN values, NaN positions stay NaN."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=2, n_stocks=10, seed=5)
    df.iloc[0, :3] = np.nan
    out = cs_zscore_panel(df)
    assert out.iloc[0, :3].isna().all()
    valid = out.iloc[0, 3:]
    assert abs(valid.mean()) < 1e-9
    assert abs(valid.std(ddof=0) - 1.0) < 1e-9


def test_cs_zscore_preserves_index_columns():
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel()
    out = cs_zscore_panel(df)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -v -k cs_zscore`
Expected: 4 errors with ImportError.

- [ ] **Step 3: Implement `cs_zscore_panel`**

Append to `src/stockpool/ml/preprocess.py`:

```python
def cs_zscore_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day cross-sectional z-score: ``(x - μ_t) / σ_t``.

    Args:
        df: T × N factor wide-frame.

    Returns:
        Same-shape DataFrame. Rows where ``σ_t < 1e-12`` (constant
        cross-section, all-NaN, or single non-NaN cell) return 0 — this
        deterministically neutralizes a degenerate day rather than producing
        ``±inf``/``NaN``. NaN cells stay NaN.

        ``σ`` uses ``ddof=0`` (matches ``standardize_fit`` upstream).
    """
    mu = df.mean(axis=1, skipna=True)
    sigma = df.std(axis=1, ddof=0, skipna=True)
    # Avoid div-by-zero: replace tiny σ with 1, then zero those rows out.
    sigma_safe = sigma.where(sigma >= 1e-12, 1.0)
    out = df.sub(mu, axis=0).div(sigma_safe, axis=0)
    degenerate = sigma < 1e-12
    if degenerate.any():
        # For degenerate rows, force non-NaN cells to 0 (NaN cells stay NaN).
        for d in df.index[degenerate]:
            out.loc[d] = out.loc[d].where(df.loc[d].isna(), 0.0)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -v -k cs_zscore`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess.py
git commit -m "feat(ml): add cs_zscore_panel per-day cross-sectional zscore

Per-day mean/std (ddof=0). Constant or all-NaN rows return 0 for
non-NaN cells; NaN cells preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement `industry_neutralize_panel`

**Files:**
- Modify: `src/stockpool/ml/preprocess.py`
- Modify: `tests/test_ml_preprocess.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ml_preprocess.py`:

```python
def test_industry_neutralize_within_group_mean_zero():
    """Per-day, within each industry group, mean of values is 0."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=3, n_stocks=12, seed=6)
    # 3 industries × 4 stocks each.
    sector_map = {f"S{i:03d}": f"ind_{i // 4}" for i in range(12)}
    out = industry_neutralize_panel(df, sector_map)
    for d in df.index:
        row = out.loc[d]
        for ind in {"ind_0", "ind_1", "ind_2"}:
            members = [c for c, s in sector_map.items() if s == ind]
            assert abs(row[members].mean()) < 1e-9


def test_industry_neutralize_unknown_code_bucket():
    """Codes not in sector_map go to '_unknown_' bucket and are demeaned together."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=6, seed=7)
    sector_map = {"S000": "A", "S001": "A"}  # only 2 of 6 mapped
    out = industry_neutralize_panel(df, sector_map)
    unknown_cols = [f"S{i:03d}" for i in range(2, 6)]
    for d in df.index:
        assert abs(out.loc[d, unknown_cols].mean()) < 1e-9


def test_industry_neutralize_empty_sector_map_raises():
    """Empty sector_map raises — caller (apply_pipeline) wraps in try/skip."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel()
    with pytest.raises(ValueError):
        industry_neutralize_panel(df, {})


def test_industry_neutralize_preserves_index_columns():
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=8)
    sector_map = {f"S{i:03d}": f"ind_{i % 2}" for i in range(8)}
    out = industry_neutralize_panel(df, sector_map)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -v -k industry`
Expected: 4 errors.

- [ ] **Step 3: Implement `industry_neutralize_panel`**

Append to `src/stockpool/ml/preprocess.py`:

```python
def industry_neutralize_panel(
    df: pd.DataFrame, sector_map: Mapping[str, str],
) -> pd.DataFrame:
    """Per-day within-industry demean.

    Args:
        df: T × N factor wide-frame (columns = codes).
        sector_map: ``{code: industry_label}``. Codes absent from the map
            fall into a single ``"_unknown_"`` bucket and are demeaned together.

    Returns:
        Same-shape DataFrame, each cell ``= x - mean(x within industry on day)``.

    Raises:
        ValueError: if ``sector_map`` is empty (caller catches and skips).
    """
    if not sector_map:
        raise ValueError("sector_map is empty; cannot industry-neutralize")
    industries = pd.Series(
        {c: sector_map.get(c, "_unknown_") for c in df.columns},
        name="industry",
    )
    # Transpose so each industry is contiguous rows; groupby + transform demean.
    transposed = df.T.copy()
    transposed["__industry__"] = industries
    # For each day column, subtract per-industry mean.
    date_cols = [c for c in transposed.columns if c != "__industry__"]
    demeaned = transposed.groupby("__industry__")[date_cols].transform(
        lambda s: s - s.mean()
    )
    return demeaned.T
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -v -k industry`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess.py
git commit -m "feat(ml): add industry_neutralize_panel within-group demean

Per-day per-industry demean (simplified Barra-style without log_mcap).
Unmapped codes pool into '_unknown_' bucket. Empty sector_map raises
for explicit caller handling.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement `apply_preprocess_pipeline` + `_is_all_off` helper

**Files:**
- Modify: `src/stockpool/ml/preprocess.py`
- Modify: `tests/test_ml_preprocess.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ml_preprocess.py`:

```python
def test_pipeline_all_off_returns_input():
    """All-off cfg returns a shallow copy with same values (no transform)."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    fp = {"f1": _make_panel(n_days=2, n_stocks=6),
          "f2": _make_panel(n_days=2, n_stocks=6, seed=99)}
    out = apply_preprocess_pipeline(fp, PreprocessConfig())
    assert set(out.keys()) == {"f1", "f2"}
    for k in out:
        pd.testing.assert_frame_equal(out[k], fp[k])
    # Shallow-copy semantics: dict is a fresh object, but frames are the same.
    assert out is not fp


def test_pipeline_all_on_three_steps_order():
    """Pipeline runs winsorize → zscore → neutralize in that order."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _make_panel(n_days=3, n_stocks=20, seed=10)
    fp = {"x": df}
    sector_map = {f"S{i:03d}": f"ind_{i % 4}" for i in range(20)}
    cfg = PreprocessConfig(
        winsorize=(0.01, 0.99), zscore=True, industry_neutralize=True,
    )
    out = apply_preprocess_pipeline(fp, cfg, sector_map=sector_map)
    # After neutralize, within-industry mean = 0 each day.
    for d in df.index:
        row = out["x"].loc[d]
        for ind in {f"ind_{i}" for i in range(4)}:
            members = [c for c, s in sector_map.items() if s == ind]
            assert abs(row[members].mean()) < 1e-9


def test_pipeline_skips_neutralize_when_no_sector_map(caplog):
    """industry_neutralize=true but empty sector_map → warning, skip step."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _make_panel(n_days=2, n_stocks=10, seed=11)
    fp = {"x": df}
    cfg = PreprocessConfig(industry_neutralize=True)
    import logging
    with caplog.at_level(logging.WARNING, logger="stockpool.ml.preprocess"):
        out = apply_preprocess_pipeline(fp, cfg, sector_map=None)
    # No neutralize done → values unchanged (no other step enabled).
    pd.testing.assert_frame_equal(out["x"], df)
    assert any("sector_map" in r.message.lower() for r in caplog.records)


def test_pipeline_skips_neutralize_for_fundamental_types():
    """Factors tagged 'fundamental' bypass industry_neutralize."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _make_panel(n_days=2, n_stocks=10, seed=12)
    fp = {"pe_ratio": df, "momentum_20": df.copy()}
    factor_types = {
        "pe_ratio": ("fundamental",),
        "momentum_20": ("momentum", "time_series"),
    }
    sector_map = {f"S{i:03d}": f"ind_{i % 2}" for i in range(10)}
    cfg = PreprocessConfig(industry_neutralize=True)
    out = apply_preprocess_pipeline(
        fp, cfg, sector_map=sector_map, factor_types=factor_types,
    )
    # pe_ratio untouched
    pd.testing.assert_frame_equal(out["pe_ratio"], df)
    # momentum_20 demeaned per industry
    for d in df.index:
        row = out["momentum_20"].loc[d]
        for ind in {"ind_0", "ind_1"}:
            members = [c for c, s in sector_map.items() if s == ind]
            assert abs(row[members].mean()) < 1e-9


def test_pipeline_partial_steps_independent():
    """Each step independently togglable: zscore-only doesn't trigger others."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline, cs_zscore_panel
    df = _make_panel(n_days=2, n_stocks=10, seed=13)
    fp = {"x": df}
    cfg = PreprocessConfig(zscore=True)
    out = apply_preprocess_pipeline(fp, cfg)
    pd.testing.assert_frame_equal(out["x"], cs_zscore_panel(df))


def test_pipeline_preserves_factor_keys_and_shapes():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    fp = {"a": _make_panel(seed=20), "b": _make_panel(seed=21), "c": _make_panel(seed=22)}
    cfg = PreprocessConfig(winsorize=(0.05, 0.95), zscore=True)
    out = apply_preprocess_pipeline(fp, cfg)
    assert set(out.keys()) == {"a", "b", "c"}
    for k in out:
        assert out[k].shape == fp[k].shape
```

Also add the `_is_all_off` helper test:

```python
def test_is_all_off_true_for_defaults():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    assert _is_all_off(PreprocessConfig()) is True


def test_is_all_off_false_for_any_step_enabled():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    assert _is_all_off(PreprocessConfig(zscore=True)) is False
    assert _is_all_off(PreprocessConfig(winsorize=(0.01, 0.99))) is False
    assert _is_all_off(PreprocessConfig(industry_neutralize=True)) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -v -k "pipeline or is_all_off"`
Expected: 8 errors with ImportError.

- [ ] **Step 3: Implement pipeline + helper**

Append to `src/stockpool/ml/preprocess.py`:

```python
def _is_all_off(cfg: "PreprocessConfig") -> bool:
    """True when every step is disabled (cfg semantically a no-op)."""
    return (
        cfg.winsorize is None
        and cfg.zscore is False
        and cfg.industry_neutralize is False
    )


def apply_preprocess_pipeline(
    factor_panel: dict[str, pd.DataFrame],
    cfg: "PreprocessConfig",
    sector_map: Mapping[str, str] | None = None,
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run winsorize → cs_zscore → industry_neutralize on each factor.

    Args:
        factor_panel: ``{factor_name: T × N DataFrame}``.
        cfg: ``PreprocessConfig`` controlling which steps run.
        sector_map: ``{code: industry}``. Required when
            ``cfg.industry_neutralize=True``; if missing/empty, that step is
            skipped with a warning (other steps still run).
        factor_types: ``{factor_name: (type_tag, ...)}``. Factors whose tag
            tuple includes ``"fundamental"`` skip industry neutralize
            (preserves sector-intrinsic signal like bank-low-PE).

    Returns:
        New dict with same keys; values are transformed (or shallow-copied
        if cfg is all-off). Original input is never mutated.
    """
    if _is_all_off(cfg):
        return dict(factor_panel)

    out: dict[str, pd.DataFrame] = {}
    do_neutralize = cfg.industry_neutralize and bool(sector_map)
    if cfg.industry_neutralize and not sector_map:
        log.warning(
            "industry_neutralize=True but sector_map is empty/None; "
            "skipping that step (winsorize/zscore still applied if enabled)"
        )

    for name, df in factor_panel.items():
        work = df
        if cfg.winsorize is not None:
            lo, hi = cfg.winsorize
            work = winsorize_panel(work, lo, hi)
        if cfg.zscore:
            work = cs_zscore_panel(work)
        if do_neutralize:
            tags = factor_types.get(name, ()) if factor_types else ()
            if "fundamental" not in tags:
                work = industry_neutralize_panel(work, sector_map)
        out[name] = work
    return out
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -v`
Expected: 18 passed total (4 winsorize + 4 zscore + 4 industry + 6 pipeline).

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess.py
git commit -m "feat(ml): add apply_preprocess_pipeline + _is_all_off helper

Pipeline runs winsorize → cs_zscore → industry_neutralize in order,
honoring per-step toggles. Skips industry neutralize when sector_map
is empty (warning) and for factors tagged 'fundamental'. All-off cfg
short-circuits to a shallow copy of the input dict.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Wire `build_factor_panel` + `_factor_panel_sig` with backwards-compat guarantee

**Files:**
- Modify: `src/stockpool/strategy_factory.py:117-207` (build_factor_panel + _factor_panel_sig)
- Modify: `tests/test_factor_panel_cache.py` (add 4 cases)

- [ ] **Step 1: Write the failing tests**

Open `tests/test_factor_panel_cache.py` and append:

```python
def test_cache_sig_all_off_backwards_compat():
    """Default PreprocessConfig sig matches pre-PR baseline (preprocess=None in dict)."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    pool = _pool(["S001"])  # _stock_df / _pool helpers already in the file
    sig_no_arg, _ = _factor_panel_sig(["momentum_20"], pool)
    sig_default, _ = _factor_panel_sig(["momentum_20"], pool, preprocess_cfg=PreprocessConfig())
    assert sig_no_arg == sig_default


def test_cache_sig_with_preprocess_isolated_from_baseline():
    """Enabling preprocess changes the sig (cache key)."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    pool = _pool(["S001"])
    sig_off, _ = _factor_panel_sig(["momentum_20"], pool, preprocess_cfg=PreprocessConfig())
    sig_on, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(zscore=True),
    )
    assert sig_off != sig_on


def test_cache_invalidates_on_preprocess_change():
    """Two different preprocess settings produce distinct sigs."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    pool = _pool(["S001"])
    sig_a, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(zscore=True),
    )
    sig_b, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(winsorize=(0.01, 0.99)),
    )
    assert sig_a != sig_b


def test_build_factor_panel_passes_preprocess(monkeypatch):
    """build_factor_panel routes preprocess_cfg through apply_preprocess_pipeline."""
    from stockpool.config import PreprocessConfig
    from stockpool import strategy_factory
    from stockpool.ml import preprocess as preproc_mod

    pool = _pool(["S001", "S002"])

    called = {}
    real_apply = preproc_mod.apply_preprocess_pipeline

    def spy(fp, cfg, sector_map=None, factor_types=None):
        called["cfg"] = cfg
        called["n_factors"] = len(fp)
        return real_apply(fp, cfg, sector_map=sector_map, factor_types=factor_types)

    monkeypatch.setattr(preproc_mod, "apply_preprocess_pipeline", spy)

    # Spy is set on the module; build_factor_panel imports inside the function
    # so it'll see the patched version.
    cfg = PreprocessConfig(zscore=True)
    strategy_factory.build_factor_panel(["momentum_20"], pool, preprocess_cfg=cfg)
    assert called["cfg"] is cfg
    assert called["n_factors"] == 1

    # All-off should NOT invoke the pipeline (short-circuit before call).
    called.clear()
    strategy_factory.build_factor_panel(
        ["momentum_20"], pool, preprocess_cfg=PreprocessConfig(),
    )
    assert called == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py -v -k "preprocess or cache_sig"`
Expected: errors — `_factor_panel_sig` doesn't accept `preprocess_cfg` kwarg yet.

- [ ] **Step 3: Modify `_factor_panel_sig` to accept preprocess_cfg**

In `src/stockpool/strategy_factory.py`, replace the existing `_factor_panel_sig` (currently lines ~179-207) with:

```python
def _factor_panel_sig(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> tuple[str, str]:
    """Return (12-char sig, last_date_iso) identifying a (factor list, universe,
    history range, preprocess config) tuple.

    Universe = sorted code list. last_date = max of any stock's max date.

    ``preprocess_cfg`` is included only when non-None **and** not all-off — an
    all-off cfg maps to ``"preprocess": None`` in the sig dict so the hash is
    byte-identical to the pre-PR baseline (no orphan cache files).

    Mask config is **not** part of the key — factor panels are mask-
    independent (mask only affects labels downstream of factor computation).
    """
    from stockpool.ml.preprocess import _is_all_off

    codes = sorted(pool_data.keys())
    last_date = pd.Timestamp.min
    for df in pool_data.values():
        if len(df) > 0:
            d = pd.to_datetime(df["date"]).max()
            if d > last_date:
                last_date = d
    last_iso = "" if last_date is pd.Timestamp.min else last_date.date().isoformat()
    preprocess_part = None
    if preprocess_cfg is not None and not _is_all_off(preprocess_cfg):
        preprocess_part = preprocess_cfg.model_dump()
    blob = json.dumps(
        {
            "factors": sorted(factor_names),
            "codes": codes,
            "last_date": last_iso,
            "preprocess": preprocess_part,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12], last_iso
```

- [ ] **Step 4: Modify `build_factor_panel` to apply preprocess**

In the same file, replace the existing `build_factor_panel` (currently lines ~117-154) with:

```python
def build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> dict[str, pd.DataFrame]:
    """从 ``{code: daily_df}`` 装一个 OHLCV Panel,在 Panel 上算所有因子,
    返回 ``{factor_name: T×N DataFrame}``。

    Look-ahead 安全:因子在第 i 行只用 ``[:i+1]`` 数据(由 Factor 契约保证),
    所以一次性预算整段历史不会泄露未来。

    **不应用 tradability mask** — 时间序列因子需要看真实价格(包括涨停日)。
    Mask 仅在标签 (``forward_return_panel``) 与训练样本筛选上生效,详见
    ``compute_factor_panel`` docstring。

    Args:
        factor_names: 因子名列表。
        pool_data: ``{code: daily_df}``.
        preprocess_cfg: 可选的 ``PreprocessConfig``。非 None 且非全关时,
            对原始因子 panel 运行 winsorize / cs_zscore / industry_neutralize
            流水线(见 ``ml/preprocess.py``)。sector_map 从
            ``factors.context.get_sector_map()`` 读取(caller 责任注入)。
    """
    from stockpool.ml.dataset import compute_factor_panel
    from stockpool.ml import preprocess as preproc_mod

    # 1) Build OHLCV panel
    per_stock: dict[str, pd.DataFrame] = {}
    for code, df in pool_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        per_stock[code] = d.set_index("date").sort_index()
    if not per_stock:
        return {}
    all_dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")
    panel: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        panel[field] = pd.DataFrame(
            {code: d[field].reindex(idx) for code, d in per_stock.items()},
            index=idx,
        )

    raw = compute_factor_panel(panel, factor_names)
    if preprocess_cfg is None or preproc_mod._is_all_off(preprocess_cfg):
        return raw

    from stockpool.factors.context import get_sector_map
    from stockpool.factors.registry import list_specs
    sector_map = get_sector_map() or None
    types_map = {
        s.name: s.types for s in list_specs() if s.name in factor_names
    }
    return preproc_mod.apply_preprocess_pipeline(
        raw, preprocess_cfg, sector_map=sector_map, factor_types=types_map,
    )
```

- [ ] **Step 5: Run all factor-panel cache tests + ml_preprocess + config**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py tests/test_ml_preprocess.py tests/test_config.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/strategy_factory.py tests/test_factor_panel_cache.py
git commit -m "feat(factor-panel): thread preprocess_cfg through build + sig

build_factor_panel and _factor_panel_sig now accept an optional
PreprocessConfig. All-off (default) maps to 'preprocess': None in the
sig dict, byte-identical to pre-PR baseline → existing factor_panels/
caches remain valid.

When non-None and non-empty, the raw factor panel is run through
apply_preprocess_pipeline before return, pulling sector_map from
factors.context.get_sector_map() and factor type tags from the registry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire `load_or_build_factor_panel` to forward preprocess_cfg

**Files:**
- Modify: `src/stockpool/strategy_factory.py:210-298` (load_or_build_factor_panel)
- Modify: `tests/test_factor_panel_cache.py` (verify no regression on existing tests)

- [ ] **Step 1: Modify `load_or_build_factor_panel`**

In `src/stockpool/strategy_factory.py`, replace the signature line and key internal calls:

Find:
```python
def load_or_build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    cache_dir: str | Path,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
```

Replace with:
```python
def load_or_build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    cache_dir: str | Path,
    refresh: bool = False,
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
```

Then find the line:
```python
    sig, last_iso = _factor_panel_sig(factor_names, pool_data)
```
Replace with:
```python
    sig, last_iso = _factor_panel_sig(factor_names, pool_data, preprocess_cfg=preprocess_cfg)
```

Then find the line:
```python
    factor_panel = build_factor_panel(factor_names, pool_data)
```
Replace with:
```python
    factor_panel = build_factor_panel(factor_names, pool_data, preprocess_cfg=preprocess_cfg)
```

- [ ] **Step 2: Run existing tests to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py -v`
Expected: all green (existing tests pass with default `preprocess_cfg=None`).

- [ ] **Step 3: Add integration test for preprocess threading**

Append to `tests/test_factor_panel_cache.py`:

```python
def test_load_or_build_factor_panel_threads_preprocess(tmp_path):
    """When preprocess_cfg is on, two calls produce different sig dirs."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import load_or_build_factor_panel
    pool = _pool(["S001", "S002"])

    # Off
    fp_off, _ = load_or_build_factor_panel(
        ["momentum_20"], pool, str(tmp_path),
        preprocess_cfg=PreprocessConfig(),
    )
    # On
    fp_on, _ = load_or_build_factor_panel(
        ["momentum_20"], pool, str(tmp_path),
        preprocess_cfg=PreprocessConfig(zscore=True),
    )
    # Two distinct sig dirs created
    sig_dirs = list((tmp_path / "factor_panels").iterdir())
    assert len(sig_dirs) == 2
```

- [ ] **Step 4: Run new test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py::test_load_or_build_factor_panel_threads_preprocess -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/strategy_factory.py tests/test_factor_panel_cache.py
git commit -m "feat(factor-panel): forward preprocess_cfg in load_or_build wrapper

load_or_build_factor_panel now takes an optional preprocess_cfg and
threads it into both _factor_panel_sig (cache key) and build_factor_panel
(actual computation). Default None → behavior unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Wire callers (`cli.py` + `backtest_runner.py`)

**Files:**
- Modify: `src/stockpool/cli.py:451`
- Modify: `src/stockpool/backtest_runner.py:90-93`

- [ ] **Step 1: Update `backtest_runner.prepare_pool`**

In `src/stockpool/backtest_runner.py`, find:
```python
    factor_panel, close_panel = load_or_build_factor_panel(
        ml_cfg.factors, pool_data, cfg.data.cache_dir,
        refresh=refresh_factor_panel,
    )
```

Replace with:
```python
    factor_panel, close_panel = load_or_build_factor_panel(
        ml_cfg.factors, pool_data, cfg.data.cache_dir,
        refresh=refresh_factor_panel,
        preprocess_cfg=ml_cfg.preprocess,
    )
```

- [ ] **Step 2: Update `cli.py:_analyze_one` path**

In `src/stockpool/cli.py`, find line 451:
```python
        factor_panel, close_panel = load_or_build_factor_panel(
            cfg.strategy.ml_factor.factors, pool_data, cfg.data.cache_dir,
        )
```

Replace with:
```python
        factor_panel, close_panel = load_or_build_factor_panel(
            cfg.strategy.ml_factor.factors, pool_data, cfg.data.cache_dir,
            preprocess_cfg=cfg.strategy.ml_factor.preprocess,
        )
```

- [ ] **Step 3: Run smoke + cli + backtest test suites**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_backtest.py tests/test_report_smoke.py tests/test_factor_panel_cache.py -v`
Expected: all green.

- [ ] **Step 4: Run the full test suite once to confirm no broad regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all green (~640 passed). Investigate any failure before continuing.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/cli.py src/stockpool/backtest_runner.py
git commit -m "feat(wiring): pass ml_factor.preprocess into factor-panel loaders

Two caller sites updated to forward the new preprocess config:
- backtest_runner.prepare_pool (used by both run and backtest paths)
- cli.cmd_run's direct load_or_build_factor_panel call (line 451)

Default-off preprocess means existing yaml + caches stay valid.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: End-to-end smoke test for ml_factor + preprocess

**Files:**
- Modify: `tests/test_ml_strategy.py` (append 1 case)

- [ ] **Step 1: Add the smoke test**

Append to `tests/test_ml_strategy.py`:

```python
def test_ml_factor_with_preprocess_runs_end_to_end():
    """ml_factor strategy with three-step preprocess produces signals end-to-end.

    8 synthetic stocks × 200 bars; pooled mode; full train + predict cycle.
    Just verifies the pipeline doesn't crash and signals come out non-NaN.
    """
    import numpy as np
    import pandas as pd
    from stockpool.config import PreprocessConfig
    from stockpool.factors.context import set_sector_map
    from stockpool.strategy_factory import build_factor_panel, build_close_panel
    from stockpool.backtesting.strategies import MLFactorStrategy

    rng = np.random.default_rng(0)
    n_stocks = 8
    n_bars = 200
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    pool_data = {}
    for i in range(n_stocks):
        close = 100 * np.exp(np.cumsum(rng.standard_normal(n_bars) * 0.02))
        pool_data[f"S{i:03d}"] = pd.DataFrame({
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n_bars),
        })

    sector_map = {f"S{i:03d}": f"ind_{i % 3}" for i in range(n_stocks)}
    set_sector_map(sector_map)

    factor_names = ["momentum_20", "rsi_centered_14", "vol_ratio_5"]
    preprocess_cfg = PreprocessConfig(
        winsorize=(0.01, 0.99), zscore=True, industry_neutralize=True,
    )
    factor_panel = build_factor_panel(
        factor_names, pool_data, preprocess_cfg=preprocess_cfg,
    )
    close_panel = build_close_panel(pool_data)
    assert set(factor_panel.keys()) == set(factor_names)
    for df in factor_panel.values():
        assert df.shape == (n_bars, n_stocks)
        # After zscore, per-day mean should be ~0 on non-NaN cells.
        last_row = df.iloc[-1].dropna()
        if len(last_row) >= 5:
            # Industry-demeaned + zscored → within-industry sum should be ~0.
            for ind in {"ind_0", "ind_1", "ind_2"}:
                members = [c for c, s in sector_map.items() if s == ind]
                non_nan = last_row[last_row.index.intersection(members)].dropna()
                if len(non_nan) >= 2:
                    assert abs(non_nan.mean()) < 1.0  # loose bound, just sanity

    # Build strategy and run predict on each stock — just verify no crash.
    strategy = MLFactorStrategy(
        factors=factor_names,
        horizon=3,
        train_window=100,
        min_train_samples=30,
        refit_every=20,
        panel_mode="pooled",
        factor_panel=factor_panel,
        close_panel=close_panel,
    )
    for code in pool_data:
        result = strategy.predict_latest(pool_data[code])
        # Result is a dict with 'signal' and 'final_score' (may be NaN
        # on warmup or insufficient training data; just must not raise).
        assert "signal" in result
```

Note: This test depends on existing `MLFactorStrategy` accepting `factor_panel` and `close_panel` kwargs (already true per PR-3). If the constructor signature differs in your branch, adapt to the actual kwargs.

- [ ] **Step 2: Run the smoke test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py::test_ml_factor_with_preprocess_runs_end_to_end -v`
Expected: PASS.

If it fails for non-preprocess reasons (e.g. MLFactorStrategy kwarg mismatch), read the actual signature and adjust the test. Do NOT skip — the smoke test is the only end-to-end coverage for the wiring.

- [ ] **Step 3: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ml_strategy.py
git commit -m "test(ml): add ml_factor + preprocess end-to-end smoke

8 synthetic stocks × 200 bars; three-step preprocess on; verifies
build_factor_panel → MLFactorStrategy.predict_latest runs without crash.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Create `ab_preprocess.yaml`

**Files:**
- Create: `ab_preprocess.yaml` (project root)

- [ ] **Step 1: Create the AB config**

Create `ab_preprocess.yaml` at project root with this content (matches spec §4.2, modeled on `ab_mask_small.yaml`):

```yaml
base_config: config.yaml
arms:
  baseline:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        selector: &id001
          type: lasso
          lasso:
            alpha: 0.001
            max_iter: 1000
            tol: 1.0e-06
        weighter: &id002
          type: ic
          ic:
            use_rank: true
            min_abs_ic: 0.0
          ir:
            n_chunks: 6
            use_rank: true
            min_abs_ir: 0.0
        thresholds: &id003
          strong_buy: 0.9
          buy: 0.7
          sell: 0.3
          strong_sell: 0.1
        buy_verdicts: &id004
        - buy
        - strong_buy
        sell_verdicts: &id005
        - sell
        - strong_sell
        refresh_verdicts: &id006
        - strong_buy
        mask:
          enabled: false
        preprocess:
          winsorize: null
          zscore: false
          industry_neutralize: false
    backtest:
      equity_curve_holding_days:
      - 10

  with_preprocess:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        selector: *id001
        weighter: *id002
        thresholds: *id003
        buy_verdicts: *id004
        sell_verdicts: *id005
        refresh_verdicts: *id006
        mask:
          enabled: false
        preprocess:
          winsorize: [0.01, 0.99]
          zscore: true
          industry_neutralize: true
    backtest:
      equity_curve_holding_days:
      - 10
```

- [ ] **Step 2: Validate yaml parses + AB cfg loads**

Run: `.venv/Scripts/python.exe -c "from stockpool.ab.config import load_ab_config; cfg = load_ab_config('ab_preprocess.yaml'); print('arms:', list(cfg.arms.keys()))"`
Expected: `arms: ['baseline', 'with_preprocess']`. Any pydantic error → fix the yaml.

- [ ] **Step 3: Commit**

```bash
git add ab_preprocess.yaml
git commit -m "test(ab): add ab_preprocess.yaml — baseline vs three-step preprocess

Single A/B per Q1 decision: arm A all-off, arm B winsorize+zscore+
industry_neutralize all on. Inherits config.yaml stock pool, uses
training_universe=pool (per P3-2 verdict).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Run the A/B and capture verdict

**Files:**
- (Generates) `docs/ab_runs/preprocess_phase1.html` (or default ab output path)
- Modify: `docs/ab_validation_results.md`
- Modify: `docs/research/2026-06-06-factor-preprocessing-and-orthogonalization.md`

- [ ] **Step 1: Run the A/B**

Run: `.venv/Scripts/python.exe -m stockpool ab --config ab_preprocess.yaml`
Expected: completes in ~5-15 min depending on training_universe and stock count. Output HTML path printed to stdout.

If it fails on missing `reports/selection.json`, the user has not run `factors pick` yet — fall back to inline factors list. Edit `ab_preprocess.yaml` both arms: replace `factors_file: reports/selection.json` with `factors: ["momentum_20", "macd_hist", "rsi_centered_14", "ma_distance_20", "vol_ratio_5", "boll_position_20"]` and re-run.

If it fails on missing universe cache (training_universe was `all`), it's already `pool` here so should not happen.

- [ ] **Step 2: Read the HTML report and capture metrics**

Open the generated HTML. Extract the following into a worksheet/notes:

```
Stock count (denominator for "stocks won"):
Mean Δsharpe (with_preprocess − baseline):
Median Δsharpe:
Stocks won (B sharpe > A sharpe):
Mean Δtotal_return:
Max drawdown delta (worst case across stocks):
Trade count A vs B:
```

- [ ] **Step 3: Apply pass-criteria decision tree**

Per spec §7.3:

| Outcome | Action |
|---|---|
| ✅ Pass (`Δsharpe ≥ +0.05`, total_return same direction, stocks won > n/2, drawdown not worse by >3pp) | Proceed to Step 4a |
| ⚠️ Indecisive (`\|Δsharpe\| < 0.05`) | Note in verdict; flag for follow-up ablation in Phase 1.5 |
| ❌ Regression (`Δsharpe < -0.05`) | Proceed to Step 4b |

- [ ] **Step 4a: If PASS — write positive verdict + enable default**

Append to `docs/ab_validation_results.md`:

```markdown
## P4-1: Preprocess Phase 1 (winsorize + cs_zscore + industry_neutralize)

**Date**: <fill in run date>
**Config**: `ab_preprocess.yaml` (training_universe=pool, lasso+ic baseline, holding_days=10)
**Stock count**: <n>
**Sharpe baseline**: <a>  → **with_preprocess**: <b>  **Δ = <b-a>**
**Total return**: A=<x%>  B=<y%>  Δ=<y-x>%
**Stocks won (B>A)**: <k>/<n>
**Max drawdown**: A=<dd_a>%  B=<dd_b>%

**Verdict: ✅ Pass** — three-step cross-sectional preprocess delivers
Δsharpe=<b-a> with consistent total-return direction. Default flipped on
in config.yaml. Proceed to Phase 2 (market-cap neutralization).
```

Then enable in `config.yaml` — find the `strategy.ml_factor` block and add:

```yaml
        preprocess:
          winsorize: [0.01, 0.99]
          zscore: true
          industry_neutralize: true
```

- [ ] **Step 4b: If FAIL — write negative verdict, leave defaults**

Append to `docs/ab_validation_results.md` a negative section (mirror format above with **❌ Regression** verdict). Do NOT change `config.yaml` defaults. Add a root-cause hypothesis section listing the three candidates from spec §7.3.

- [ ] **Step 4c: If INDECISIVE — write neutral verdict, plan ablation**

Append a `⚠️ Indecisive` section. Plan a Phase 1.5 ablation: 3 sub-A/Bs each toggling one step alone, to be done in a follow-up plan.

- [ ] **Step 5: Update the research doc**

Append to `docs/research/2026-06-06-factor-preprocessing-and-orthogonalization.md`:

```markdown
---

## Phase 1 Outcome (<run date>)

Validated via `ab_preprocess.yaml`. Verdict: <Pass / Regression / Indecisive>.
See `docs/ab_validation_results.md` P4-1 for metrics.
<one-line summary of next-step impact>
```

- [ ] **Step 6: Update CLAUDE.md + README.md (doc rule)**

In `CLAUDE.md`:

- In the factor library section, add a row:
  ```
  | `src/stockpool/ml/preprocess.py` | 截面预处理流水线 (winsorize / cs_zscore / industry_neutralize) |
  ```
- In the 配置 section under `strategy.ml_factor`, add a bullet:
  ```
  **`preprocess.{winsorize, zscore, industry_neutralize}`**(2026-06-06 Phase 1):
  截面预处理三步,winsorize 默认 null,zscore + industry_neutralize 默认 false。
  baked 进 factor_panel cache → 改 preprocess 自动 sig 失效。industry_neutralize
  跳过 factor types 含 "fundamental" 的因子。
  ```
- In the 测试 table, add:
  ```
  | `test_ml_preprocess.py` | 截面预处理三函数 + apply_preprocess_pipeline + _is_all_off |
  ```

In `README.md`, find the ml_factor config example block and append a `preprocess:` sub-block matching the spec §4.1 yaml shape.

- [ ] **Step 7: Commit verdict + doc updates**

```bash
git add docs/ab_validation_results.md docs/research/2026-06-06-factor-preprocessing-and-orthogonalization.md CLAUDE.md README.md
# if PASS, also add: config.yaml
git commit -m "docs(factor-preprocess): record Phase 1 AB verdict + update guides

P4-1 outcome captured in ab_validation_results.md and research doc
appendix. CLAUDE.md + README.md updated per 'docs after change' rule.
<one-line outcome summary>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final Validation Checklist

Before declaring the plan complete, confirm:

- [ ] All Task steps committed individually
- [ ] `.venv/Scripts/python.exe -m pytest tests/ -q` reports ~640 passed (up from 615)
- [ ] `.venv/Scripts/python.exe -m stockpool ab --config ab_preprocess.yaml` ran and HTML generated
- [ ] `docs/ab_validation_results.md` has the P4-1 entry with metrics
- [ ] `CLAUDE.md` + `README.md` updated per docs rule
- [ ] No accidentally-modified files (`git status` clean)
