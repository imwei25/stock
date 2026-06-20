# Market-Cap Neutralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional preprocessing step that residualises factor panels against `log(market_cap)` (either alone or jointly with industry dummies), validated via A/B comparison on the 4000+ stock training universe.

**Architecture:** New `src/stockpool/ml/mcap.py` builds a daily `log(market_cap)` panel from existing baostock `balance.totalShare × close` (PIT). `ml/preprocess.py` gains `mcap_neutralize_panel(df, log_mcap)` and a `log_mcap` kwarg on `industry_neutralize_panel` that switches to per-day OLS residualisation via `numpy.linalg.lstsq`. `PreprocessConfig.mcap_neutralize: bool = False` toggles the path and is baked into the `factor_panels/<sig>/` cache key, auto-invalidating stale caches.

**Tech Stack:** Python, pandas, numpy, pytest, baostock fundamentals cache, existing `factor_panels/` parquet cache.

**Spec reference:** `docs/superpowers/specs/2026-06-06-mcap-neutralization-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/stockpool/ml/mcap.py` | Build `log(market_cap)` T×N panel from balance + close | Create |
| `src/stockpool/ml/preprocess.py` | Add `mcap_neutralize_panel`, extend `industry_neutralize_panel`, route via `apply_preprocess_pipeline` | Modify |
| `src/stockpool/config.py:372-402` | `PreprocessConfig.mcap_neutralize: bool = False` | Modify |
| `src/stockpool/strategy_factory.py:117-173` | `build_factor_panel` builds `log_mcap_panel` when toggled, plumbs through | Modify |
| `src/stockpool/strategy_factory.py:302` | `load_or_build_factor_panel` passes `cache_dir` to `build_factor_panel` | Modify |
| `src/stockpool/factors/fundamentals.py` | Register `MarketCapFactor` / `LogMarketCapFactor`, add `"contains_mcap"` tag to PE/PB | Modify |
| `tests/test_ml_preprocess_mcap.py` | Unit + integration tests for new neutralizers | Create |
| `tests/test_factors_fundamentals.py` | Tests for `market_cap` / `log_market_cap` factors + PE/PB tag | Modify |
| `tests/test_config.py` | `mcap_neutralize` default + content_hash sensitivity | Modify |
| `tests/test_factor_panel_cache.py` | Cache sig changes when `mcap_neutralize` flips | Modify |
| `tests/test_strategy_factory.py` or new file | `build_factor_panel` builds + passes log_mcap when toggled | Modify/Create |
| `ab_mcap.yaml` | A/B config: `preprocess_only` vs `preprocess_plus_mcap` | Create |
| `docs/ab_validation_results.md` | Append P5-mcap section with verdict (after AB run) | Modify |
| `CLAUDE.md` | Doc preprocessing config + new factors + test file | Modify |
| `README.md` | Document `mcap_neutralize` flag if README has preprocess example | Modify (if applicable) |

---

## Task Sequence (PR-1: code + tests)

### Task 1: Add `mcap_neutralize` field to `PreprocessConfig`

**Files:**
- Modify: `src/stockpool/config.py:372-402`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test in `tests/test_config.py`**

Add this at the end of the file:

```python
def test_preprocess_config_mcap_neutralize_default_false():
    from stockpool.config import PreprocessConfig
    cfg = PreprocessConfig()
    assert cfg.mcap_neutralize is False


def test_preprocess_config_mcap_neutralize_explicit_true():
    from stockpool.config import PreprocessConfig
    cfg = PreprocessConfig(mcap_neutralize=True)
    assert cfg.mcap_neutralize is True


def test_preprocess_config_extra_forbid_still_rejects_typos():
    """Regression: adding mcap_neutralize must not loosen extra='forbid'."""
    import pytest
    from pydantic import ValidationError
    from stockpool.config import PreprocessConfig
    with pytest.raises(ValidationError):
        PreprocessConfig(mcap_neutralise=True)  # British spelling typo


def test_ml_factor_config_hash_changes_when_mcap_neutralize_flips():
    from stockpool.config import MLFactorConfig, PreprocessConfig
    cfg_off = MLFactorConfig(preprocess=PreprocessConfig(mcap_neutralize=False))
    cfg_on = MLFactorConfig(preprocess=PreprocessConfig(mcap_neutralize=True))
    assert cfg_off.content_hash != cfg_on.content_hash
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py::test_preprocess_config_mcap_neutralize_default_false -v
```

Expected: FAIL with pydantic error "Extra inputs are not permitted" (because field doesn't exist yet).

- [ ] **Step 3: Add the field to `PreprocessConfig`**

In `src/stockpool/config.py`, find the existing `PreprocessConfig` class (around line 372) and insert `mcap_neutralize` after `industry_neutralize`:

```python
class PreprocessConfig(BaseModel):
    """Cross-sectional preprocessing pipeline for ML factor panels.

    Applied at ``build_factor_panel()`` output, before disk caching. Affects
    ml_factor training, predict, and downstream Pool B consumers identically.
    Default = all off → fully backwards compatible (cache sig unchanged).

    See: docs/superpowers/specs/2026-06-06-factor-preprocessing-phase1-design.md
         docs/superpowers/specs/2026-06-06-mcap-neutralization-design.md
    """
    model_config = ConfigDict(extra="forbid")

    winsorize: tuple[float, float] | None = None
    zscore: bool = False
    industry_neutralize: bool = False
    mcap_neutralize: bool = False
    # 当 True 时,apply_preprocess_pipeline 需要 caller 传 log_mcap_panel:
    #   - mcap_neutralize=True, industry_neutralize=False → 单变量 OLS Y ~ log_mcap
    #   - mcap_neutralize=True, industry_neutralize=True  → 联合 OLS Y ~ industry_dummies + log_mcap
    # PE/PB(打 contains_mcap tag)跳过 mcap_neutralize 防止与 close × shares 强共线。
    # 详见 docs/superpowers/specs/2026-06-06-mcap-neutralization-design.md
    min_pool_size: int = Field(default=200, ge=0)
    # n_codes < min_pool_size 时 winsorize / cs_zscore / industry_neutralize /
    # mcap_neutralize 全部跳过(估计不稳)。industry_neutralize 即使在大池子也建议
    # 保持默认 false:单成员细分行业会触发 silent demean-to-zero bug(P4-1 verdict)。
    # Phase 1.5 全市场参照设计落地前不推荐启用 industry_neutralize。

    @field_validator("winsorize")
    @classmethod
    def _check_winsorize_bounds(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is None:
            return None
        lo, hi = v
        if not (0 < lo < hi < 1):
            raise ValueError(
                f"winsorize bounds must satisfy 0 < lo < hi < 1, got ({lo}, {hi})"
            )
        return (float(lo), float(hi))
```

- [ ] **Step 4: Run the new tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py -k "mcap_neutralize or extra_forbid_still_rejects" -v
```

Expected: all PASS.

- [ ] **Step 5: Run full test suite to confirm no regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py -q
```

Expected: all PASS (existing tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "feat(config): add PreprocessConfig.mcap_neutralize field (default False)"
```

---

### Task 2: Update `_is_all_off` to include `mcap_neutralize`

**Files:**
- Modify: `src/stockpool/ml/preprocess.py:118-124`
- Test: `tests/test_ml_preprocess.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ml_preprocess.py`:

```python
def test_is_all_off_treats_mcap_neutralize_as_active():
    """mcap_neutralize=True alone must NOT short-circuit the pipeline."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
    )
    assert _is_all_off(cfg) is False


def test_is_all_off_true_when_everything_off_including_mcap():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=False,
    )
    assert _is_all_off(cfg) is True
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py::test_is_all_off_treats_mcap_neutralize_as_active -v
```

Expected: FAIL (current `_is_all_off` returns True even when mcap_neutralize=True).

- [ ] **Step 3: Update `_is_all_off` in `src/stockpool/ml/preprocess.py`**

Replace the function:

```python
def _is_all_off(cfg: "PreprocessConfig") -> bool:
    """True when every step is disabled (cfg semantically a no-op)."""
    return (
        cfg.winsorize is None
        and cfg.zscore is False
        and cfg.industry_neutralize is False
        and cfg.mcap_neutralize is False
    )
```

- [ ] **Step 4: Run the new tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -k "is_all_off" -v
```

Expected: PASS.

- [ ] **Step 5: Run full preprocess + config tests to confirm no regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py tests/test_config.py -q
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess.py
git commit -m "feat(preprocess): include mcap_neutralize in _is_all_off short-circuit"
```

---

### Task 3: Implement `mcap_neutralize_panel` (standalone OLS Y ~ log_mcap)

**Files:**
- Modify: `src/stockpool/ml/preprocess.py`
- Test: `tests/test_ml_preprocess_mcap.py` (new)

- [ ] **Step 1: Create `tests/test_ml_preprocess_mcap.py` with the failing happy-path test**

```python
"""Unit tests for market-cap neutralization preprocessing."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


def _panel(n_days, n_stocks, seed):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(
        rng.standard_normal((n_days, n_stocks)), index=dates, columns=codes,
    )


def test_mcap_neutralize_removes_log_mcap_loading():
    """Y = 2 * log_mcap + noise → residuals should have ~zero correlation with log_mcap."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=4, freq="B")
    codes = [f"S{i:03d}" for i in range(50)]
    log_mcap = pd.DataFrame(
        rng.standard_normal((4, 50)) * 0.5 + 10.0,
        index=dates, columns=codes,
    )
    noise = pd.DataFrame(
        rng.standard_normal((4, 50)) * 0.1, index=dates, columns=codes,
    )
    y = 2.0 * log_mcap + noise

    resid = mcap_neutralize_panel(y, log_mcap)

    # Per-day OLS residual should be ~noise (corr with log_mcap ~0)
    for d in dates:
        r = resid.loc[d]
        m = log_mcap.loc[d]
        corr = np.corrcoef(r.values, m.values)[0, 1]
        assert abs(corr) < 0.05, f"residual still correlated with log_mcap on {d}: corr={corr}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_mcap.py::test_mcap_neutralize_removes_log_mcap_loading -v
```

Expected: FAIL with `ImportError: cannot import name 'mcap_neutralize_panel'`.

- [ ] **Step 3: Add a shared per-day OLS helper plus `mcap_neutralize_panel` to `src/stockpool/ml/preprocess.py`**

Add to the imports at the top of `src/stockpool/ml/preprocess.py`:

```python
# (already present): import logging, numpy as np, pandas as pd
# nothing new needed
```

Then add **above** `industry_neutralize_panel`:

```python
def _per_day_ols_residual(
    y: pd.Series, X: pd.DataFrame,
) -> tuple[pd.Series, bool]:
    """Per-day OLS residualisation. Returns (residual, used_ols).

    ``y`` and ``X`` share the same index (codes for one date). Drops rows where
    y is NaN or any X column is NaN. Requires ≥ 10 valid rows AND X to be
    strictly tall (more rows than columns) — otherwise returns y unchanged with
    used_ols=False so callers can fall back / count degenerate days.

    Single-member dummy columns (column-sum == 1) are dropped along with the
    corresponding row to avoid those codes being demeaned to their own value
    (a silent zero-out, see Phase 1.5 incident).
    """
    valid_mask = y.notna() & X.notna().all(axis=1)
    y_v = y[valid_mask]
    X_v = X.loc[valid_mask]

    if len(y_v) < 10:
        return y, False

    # Drop dummies that have only one member among the valid rows;
    # drop those rows too (those single-member codes get y unchanged).
    col_sums = X_v.sum(axis=0)
    single_member_cols = col_sums.index[col_sums == 1].tolist()
    if single_member_cols:
        # The row(s) where the single-member dummy is hot — these codes are not
        # represented in the regression and keep their original y.
        single_member_rows = X_v.index[
            X_v[single_member_cols].any(axis=1)
        ].tolist()
        X_v = X_v.drop(columns=single_member_cols).drop(index=single_member_rows)
        y_v = y_v.drop(index=single_member_rows)

    if len(y_v) < 10 or X_v.shape[0] <= X_v.shape[1]:
        return y, False

    coef, *_ = np.linalg.lstsq(X_v.values, y_v.values, rcond=None)
    resid_v = y_v.values - X_v.values @ coef

    out = y.copy()
    out.loc[y_v.index] = resid_v
    return out, True


def mcap_neutralize_panel(
    df: pd.DataFrame, log_mcap: pd.DataFrame,
) -> pd.DataFrame:
    """Per-day residualise Y ~ 1 + log_mcap (no industry).

    Args:
        df: T × N factor wide-frame (date index, code columns).
        log_mcap: T × N log-market-cap aligned to df's index. Codes that appear
            in ``df.columns`` but not in ``log_mcap.columns`` are treated as NaN
            and dropped per day.

    Returns:
        Same shape as df. NaN cells stay NaN. Days that fail the OLS preconditions
        (< 10 valid codes or rank deficiency) return their original df rows
        unchanged; aggregate fallback count is logged at WARNING level once per call.
    """
    if df.empty:
        return df.copy()
    log_mcap_aligned = log_mcap.reindex(index=df.index, columns=df.columns)
    out = df.copy()
    fallback_days = 0
    for date in df.index:
        y = df.loc[date]
        m = log_mcap_aligned.loc[date]
        X = pd.DataFrame({"intercept": 1.0, "log_mcap": m.values}, index=y.index)
        resid, used_ols = _per_day_ols_residual(y, X)
        if used_ols:
            out.loc[date] = resid
        else:
            fallback_days += 1
    if fallback_days:
        log.warning(
            "mcap_neutralize_panel: fallback on %d / %d days "
            "(degenerate cross-section: < 10 valid codes or rank deficient)",
            fallback_days, len(df.index),
        )
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_mcap.py::test_mcap_neutralize_removes_log_mcap_loading -v
```

Expected: PASS.

- [ ] **Step 5: Add coverage tests for NaN handling, degenerate days, and shape preservation**

Append to `tests/test_ml_preprocess_mcap.py`:

```python
def test_mcap_neutralize_preserves_shape_and_nan_cells():
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(3, 30, seed=1)
    log_mcap = _panel(3, 30, seed=2)
    df.iloc[0, 5] = np.nan
    log_mcap.iloc[1, 7] = np.nan
    out = mcap_neutralize_panel(df, log_mcap)
    assert out.shape == df.shape
    assert np.isnan(out.iloc[0, 5])  # original NaN in y stays NaN


def test_mcap_neutralize_falls_back_when_too_few_codes(caplog):
    """A day with < 10 valid codes returns original row + emits warning count."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(2, 8, seed=3)  # only 8 codes — below hard minimum 10
    log_mcap = _panel(2, 8, seed=4)
    with caplog.at_level(logging.WARNING, logger="stockpool.ml.preprocess"):
        out = mcap_neutralize_panel(df, log_mcap)
    # All days should fall back → df returned unchanged
    pd.testing.assert_frame_equal(out, df)
    assert any("fallback on 2 / 2 days" in rec.message for rec in caplog.records)


def test_mcap_neutralize_handles_all_nan_log_mcap_day():
    """A day where log_mcap is fully NaN falls back to original df row."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(3, 30, seed=5)
    log_mcap = _panel(3, 30, seed=6)
    log_mcap.iloc[1] = np.nan
    out = mcap_neutralize_panel(df, log_mcap)
    pd.testing.assert_series_equal(out.iloc[1], df.iloc[1])


def test_mcap_neutralize_days_are_independent():
    """Mutating log_mcap on one day must not change residuals on other days."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(3, 30, seed=7)
    log_mcap = _panel(3, 30, seed=8)
    out1 = mcap_neutralize_panel(df, log_mcap)
    log_mcap_mod = log_mcap.copy()
    log_mcap_mod.iloc[1] = log_mcap_mod.iloc[1] * 100
    out2 = mcap_neutralize_panel(df, log_mcap_mod)
    pd.testing.assert_series_equal(out1.iloc[0], out2.iloc[0])
    pd.testing.assert_series_equal(out1.iloc[2], out2.iloc[2])
```

- [ ] **Step 6: Run all new tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_mcap.py -v
```

Expected: 5 PASS.

- [ ] **Step 7: Run existing preprocess tests to confirm no regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess_mcap.py
git commit -m "feat(preprocess): add mcap_neutralize_panel (per-day OLS Y ~ log_mcap)"
```

---

### Task 4: Extend `industry_neutralize_panel` with optional `log_mcap` joint OLS

**Files:**
- Modify: `src/stockpool/ml/preprocess.py:85-115`
- Test: `tests/test_ml_preprocess_mcap.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ml_preprocess_mcap.py`:

```python
def test_industry_neutralize_legacy_behavior_unchanged_when_log_mcap_none():
    """log_mcap=None must produce bit-for-bit identical output to the pre-PR code path."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _panel(3, 20, seed=10)
    sector_map = {c: f"IND{i % 4}" for i, c in enumerate(df.columns)}
    legacy = industry_neutralize_panel(df, sector_map)  # no log_mcap kwarg
    explicit_none = industry_neutralize_panel(df, sector_map, log_mcap=None)
    pd.testing.assert_frame_equal(legacy, explicit_none)


def test_industry_neutralize_joint_ols_residual_orthogonal_to_inputs():
    """Y = 3 * log_mcap + 1.5 * industry_effect + noise →
    residuals ~ noise, ~uncorrelated with log_mcap and industry membership."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    rng = np.random.default_rng(11)
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    codes = [f"S{i:03d}" for i in range(60)]
    sector_map = {c: f"IND{i % 4}" for i, c in enumerate(codes)}
    industry_offset = pd.Series(
        {c: float({"IND0": -1.0, "IND1": 0.0, "IND2": 1.0, "IND3": 2.0}[sector_map[c]])
         for c in codes}
    )
    log_mcap = pd.DataFrame(
        rng.standard_normal((3, 60)) * 0.5 + 10.0, index=dates, columns=codes,
    )
    noise = pd.DataFrame(
        rng.standard_normal((3, 60)) * 0.1, index=dates, columns=codes,
    )
    y = 3.0 * log_mcap + industry_offset.values[None, :] * 1.5 + noise

    resid = industry_neutralize_panel(y, sector_map, log_mcap=log_mcap)

    # Residual should be uncorrelated with log_mcap per day
    for d in dates:
        r = resid.loc[d]
        m = log_mcap.loc[d]
        corr = np.corrcoef(r.values, m.values)[0, 1]
        assert abs(corr) < 0.1, f"residual ~ log_mcap on {d}: corr={corr}"

    # Per-industry mean of residual should be ~0 (industry demeaned)
    for d in dates:
        for ind in {"IND0", "IND1", "IND2", "IND3"}:
            members = [c for c in codes if sector_map[c] == ind]
            assert abs(resid.loc[d, members].mean()) < 0.1


def test_industry_neutralize_single_member_industry_keeps_original_value():
    """A single-member industry must NOT be silently demeaned to 0;
    its code is excluded from the regression and the original y is kept."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    rng = np.random.default_rng(12)
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    codes = [f"S{i:03d}" for i in range(30)]
    sector_map = {c: "BIG" for c in codes[:-1]}
    sector_map[codes[-1]] = "LONELY"
    log_mcap = pd.DataFrame(
        rng.standard_normal((2, 30)) * 0.5 + 10.0, index=dates, columns=codes,
    )
    df = pd.DataFrame(
        rng.standard_normal((2, 30)), index=dates, columns=codes,
    )
    out = industry_neutralize_panel(df, sector_map, log_mcap=log_mcap)
    # The lonely code should keep its original value (NOT silently zeroed)
    pd.testing.assert_series_equal(out[codes[-1]], df[codes[-1]])
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_mcap.py -k "industry_neutralize" -v
```

Expected: FAIL with `TypeError: industry_neutralize_panel() got an unexpected keyword argument 'log_mcap'`.

- [ ] **Step 3: Extend `industry_neutralize_panel` in `src/stockpool/ml/preprocess.py`**

Replace the existing function with this:

```python
def industry_neutralize_panel(
    df: pd.DataFrame,
    sector_map: Mapping[str, str],
    log_mcap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-day within-industry demean OR joint OLS Y ~ industry + log_mcap.

    Args:
        df: T × N factor wide-frame (columns = codes).
        sector_map: ``{code: industry_label}``. Codes absent from the map
            fall into a single ``"_unknown_"`` bucket and are demeaned together.
        log_mcap: optional T × N log(market_cap). When provided, the per-day
            transform switches from group demean to OLS residualisation against
            ``[industry_dummies(drop_first), log_mcap]``. Days that fail OLS
            preconditions (< 10 valid codes or rank deficient) fall back to
            the legacy group-demean path for that day.

    Returns:
        Same-shape DataFrame.

    Raises:
        ValueError: if ``sector_map`` is empty (caller catches and skips).
    """
    if not sector_map:
        raise ValueError("sector_map is empty; cannot industry-neutralize")

    if log_mcap is None:
        # Legacy fast path — group demean (bit-for-bit unchanged).
        industries = pd.Series(
            {c: sector_map.get(c, "_unknown_") for c in df.columns},
            name="industry",
        )
        transposed = df.T.copy()
        transposed["__industry__"] = industries
        date_cols = [c for c in transposed.columns if c != "__industry__"]
        demeaned = transposed.groupby("__industry__")[date_cols].transform(
            lambda s: s - s.mean()
        )
        return demeaned.T

    # OLS path: build one-hot industry dummies (drop first to avoid singularity)
    industries = pd.Series(
        {c: sector_map.get(c, "_unknown_") for c in df.columns},
    )
    dummies = pd.get_dummies(industries, prefix="ind", drop_first=True, dtype=float)
    # dummies index = codes; columns = ind_<label> minus reference

    log_mcap_aligned = log_mcap.reindex(index=df.index, columns=df.columns)
    out = df.copy()
    fallback_days = 0

    # Pre-compute group-demean output once for fallback rows
    legacy_fallback = industry_neutralize_panel(df, sector_map, log_mcap=None)

    for date in df.index:
        y = df.loc[date]
        m = log_mcap_aligned.loc[date]
        X = dummies.copy()
        X["intercept"] = 1.0
        X["log_mcap"] = m.values
        resid, used_ols = _per_day_ols_residual(y, X)
        if used_ols:
            out.loc[date] = resid
        else:
            out.loc[date] = legacy_fallback.loc[date]
            fallback_days += 1

    if fallback_days:
        log.warning(
            "industry_neutralize_panel(log_mcap=...): OLS fallback on %d / %d days "
            "(degenerate cross-section); used group demean for those days",
            fallback_days, len(df.index),
        )
    return out
```

- [ ] **Step 4: Run the new tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_mcap.py -k "industry_neutralize" -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run all preprocess tests for regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py tests/test_ml_preprocess_mcap.py -q
```

Expected: all PASS (legacy behavior preserved).

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess_mcap.py
git commit -m "feat(preprocess): industry_neutralize_panel accepts log_mcap for joint OLS"
```

---

### Task 5: Create `src/stockpool/ml/mcap.py` with `build_log_mcap_panel`

**Files:**
- Create: `src/stockpool/ml/mcap.py`
- Test: `tests/test_ml_mcap.py` (new)

- [ ] **Step 1: Write the failing test in `tests/test_ml_mcap.py`**

```python
"""Unit tests for log(market_cap) panel construction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _close_panel():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    return pd.DataFrame(
        {"600000": [10.0, 11.0, 12.0], "000001": [20.0, 22.0, 24.0]},
        index=dates,
    )


def test_build_log_mcap_panel_uses_close_times_total_share(monkeypatch, tmp_path):
    """mcap = close × totalShare → log(mcap), PIT-aligned by pubDate."""
    from stockpool.ml.mcap import build_log_mcap_panel

    # Fake balance table: 6e8 shares for 600000 announced 2024-12-15;
    #                     1e9 shares for 000001 announced 2024-12-20.
    fake_balance = pd.DataFrame({
        "code": ["600000", "000001"],
        "pubDate": pd.to_datetime(["2024-12-15", "2024-12-20"]),
        "statDate": pd.to_datetime(["2024-09-30", "2024-09-30"]),
        "totalShare": [6e8, 1e9],
    })

    def fake_loader(table, cache_dir=None):
        assert table == "balance"
        return fake_balance

    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        fake_loader,
    )

    close = _close_panel()
    panel = {"close": close}
    log_mcap = build_log_mcap_panel(panel, cache_dir=str(tmp_path))

    # Expected: mcap[date, code] = close × totalShare (ffill from pubDate)
    expected_mcap = close.copy()
    expected_mcap["600000"] = close["600000"] * 6e8
    expected_mcap["000001"] = close["000001"] * 1e9
    expected_log = np.log(expected_mcap)

    pd.testing.assert_frame_equal(log_mcap, expected_log, check_dtype=False)


def test_build_log_mcap_panel_returns_nan_when_shares_missing(monkeypatch, tmp_path):
    """No totalShare row for a code → NaN log_mcap (so per-day OLS dropna handles it)."""
    from stockpool.ml.mcap import build_log_mcap_panel

    fake_balance = pd.DataFrame({
        "code": ["600000"],  # 000001 missing
        "pubDate": pd.to_datetime(["2024-12-15"]),
        "statDate": pd.to_datetime(["2024-09-30"]),
        "totalShare": [6e8],
    })
    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: fake_balance,
    )
    close = _close_panel()
    log_mcap = build_log_mcap_panel({"close": close}, cache_dir=str(tmp_path))
    assert log_mcap["000001"].isna().all()
    assert log_mcap["600000"].notna().all()


def test_build_log_mcap_panel_handles_empty_balance(monkeypatch, tmp_path):
    """Empty balance table → all-NaN log_mcap panel of correct shape."""
    from stockpool.ml.mcap import build_log_mcap_panel

    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: pd.DataFrame(),
    )
    close = _close_panel()
    log_mcap = build_log_mcap_panel({"close": close}, cache_dir=str(tmp_path))
    assert log_mcap.shape == close.shape
    assert log_mcap.isna().all().all()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_mcap.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'stockpool.ml.mcap'`.

- [ ] **Step 3: Create `src/stockpool/ml/mcap.py`**

```python
"""log(market_cap) panel construction for OLS-based neutralization.

mcap = close × totalShare, PIT-aligned by pubDate via the same helper that
``factors.fundamentals`` uses. Reuses the 30-day baostock balance-table parquet
cache — no new fetch path.

See: docs/superpowers/specs/2026-06-06-mcap-neutralization-design.md
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def build_log_mcap_panel(
    panel: Mapping[str, pd.DataFrame],
    cache_dir,
) -> pd.DataFrame:
    """Build T×N log(market_cap) panel aligned to ``panel["close"]``.

    Args:
        panel: must contain a ``"close"`` T×N wide-frame.
        cache_dir: passed to ``load_or_build_fundamentals`` for the balance
            parquet cache. ``None`` skips caching (live fetch each call).

    Returns:
        T×N DataFrame, same index/columns as ``panel["close"]``. Cells where
        ``totalShare`` is missing or ``mcap <= 0`` are NaN — the per-day OLS
        downstream drops those rows.
    """
    from stockpool.fundamentals_loader import load_or_build_fundamentals
    from stockpool.factors.fundamentals import _pit_align

    close = panel["close"]
    balance = load_or_build_fundamentals("balance", cache_dir=cache_dir)
    if balance is None or balance.empty:
        return pd.DataFrame(
            np.nan, index=close.index, columns=close.columns,
        )
    shares_panel = _pit_align(balance, "totalShare", close)
    mcap = close * shares_panel
    return np.log(mcap.where(mcap > 0))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_mcap.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ml/mcap.py tests/test_ml_mcap.py
git commit -m "feat(ml): add build_log_mcap_panel (close × totalShare, PIT)"
```

---

### Task 6: Route `log_mcap_panel` through `apply_preprocess_pipeline`

**Files:**
- Modify: `src/stockpool/ml/preprocess.py` (`apply_preprocess_pipeline` signature + body)
- Test: `tests/test_ml_preprocess_mcap.py`

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_ml_preprocess_mcap.py`:

```python
def test_apply_pipeline_skips_mcap_when_log_mcap_none(caplog):
    """mcap_neutralize=True without log_mcap_panel → log warning, skip mcap step."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _panel(3, 30, seed=20)
    cfg = PreprocessConfig(
        winsorize=None, zscore=True,
        industry_neutralize=False, mcap_neutralize=True,
    )
    with caplog.at_level(logging.WARNING, logger="stockpool.ml.preprocess"):
        out = apply_preprocess_pipeline({"f1": df}, cfg, log_mcap_panel=None)
    assert "mcap_neutralize=True" in " ".join(rec.message for rec in caplog.records)
    # zscore still applied (rows mean ~0)
    assert abs(out["f1"].iloc[0].mean()) < 1e-9


def test_apply_pipeline_runs_mcap_neutralize_when_enabled():
    """mcap_neutralize=True with valid log_mcap → factor residualised."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    rng = np.random.default_rng(21)
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    codes = [f"S{i:03d}" for i in range(50)]
    log_mcap = pd.DataFrame(
        rng.standard_normal((3, 50)) * 0.5 + 10.0, index=dates, columns=codes,
    )
    df = 2.0 * log_mcap + pd.DataFrame(
        rng.standard_normal((3, 50)) * 0.1, index=dates, columns=codes,
    )
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
    )
    out = apply_preprocess_pipeline(
        {"f1": df}, cfg, log_mcap_panel=log_mcap, n_codes=50,
    )
    for d in dates:
        corr = np.corrcoef(out["f1"].loc[d].values, log_mcap.loc[d].values)[0, 1]
        assert abs(corr) < 0.05


def test_apply_pipeline_size_guard_short_circuits_mcap():
    """n_codes < min_pool_size → mcap step also skipped along with others."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _panel(3, 5, seed=22)
    log_mcap = _panel(3, 5, seed=23)
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
        min_pool_size=200,
    )
    out = apply_preprocess_pipeline(
        {"f1": df}, cfg, log_mcap_panel=log_mcap, n_codes=5,
    )
    pd.testing.assert_frame_equal(out["f1"], df)


def test_apply_pipeline_fundamental_factor_skip_industry_but_runs_mcap():
    """ROE (fundamental, no contains_mcap tag) skips industry, runs mcap."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    rng = np.random.default_rng(24)
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    codes = [f"S{i:03d}" for i in range(60)]
    log_mcap = pd.DataFrame(
        rng.standard_normal((2, 60)) * 0.5 + 10.0, index=dates, columns=codes,
    )
    roe_df = 3.0 * log_mcap + pd.DataFrame(
        rng.standard_normal((2, 60)) * 0.1, index=dates, columns=codes,
    )
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=True, mcap_neutralize=True,
    )
    sector_map = {c: f"IND{i % 4}" for i, c in enumerate(codes)}
    factor_types = {"roe": ("fundamental", "cross_sectional")}
    out = apply_preprocess_pipeline(
        {"roe": roe_df}, cfg,
        sector_map=sector_map, factor_types=factor_types,
        log_mcap_panel=log_mcap, n_codes=60,
    )
    # mcap should still have been removed
    for d in dates:
        corr = np.corrcoef(out["roe"].loc[d].values, log_mcap.loc[d].values)[0, 1]
        assert abs(corr) < 0.1


def test_apply_pipeline_pe_with_contains_mcap_skips_both():
    """PE (contains_mcap) skips both industry AND mcap neutralization."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _panel(2, 50, seed=25)
    log_mcap = _panel(2, 50, seed=26)
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=True, mcap_neutralize=True,
    )
    sector_map = {c: f"IND{i % 4}" for i, c in enumerate(df.columns)}
    factor_types = {"pe": ("fundamental", "cross_sectional", "contains_mcap")}
    out = apply_preprocess_pipeline(
        {"pe": df}, cfg,
        sector_map=sector_map, factor_types=factor_types,
        log_mcap_panel=log_mcap, n_codes=50,
    )
    pd.testing.assert_frame_equal(out["pe"], df)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_mcap.py -k "apply_pipeline" -v
```

Expected: FAIL with `TypeError: apply_preprocess_pipeline() got an unexpected keyword argument 'log_mcap_panel'`.

- [ ] **Step 3: Update `apply_preprocess_pipeline` in `src/stockpool/ml/preprocess.py`**

Replace the existing function with this:

```python
def apply_preprocess_pipeline(
    factor_panel: dict[str, pd.DataFrame],
    cfg: "PreprocessConfig",
    sector_map: Mapping[str, str] | None = None,
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
    n_codes: int | None = None,
    log_mcap_panel: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run winsorize → cs_zscore → (industry/mcap) neutralize per factor.

    Args:
        factor_panel: ``{factor_name: T × N DataFrame}``.
        cfg: ``PreprocessConfig`` controlling which steps run.
        sector_map: ``{code: industry}``. Required when ``cfg.industry_neutralize``;
            empty/missing skips that step with a warning.
        factor_types: ``{factor_name: (type_tag, ...)}``.
            * ``"fundamental"`` → skip industry neutralize (legacy bank-low-PE rule).
            * ``"contains_mcap"`` → also skip mcap neutralize (PE/PB are
              collinear with close × shares).
        n_codes: actual panel width. When below ``cfg.min_pool_size`` every step
            is skipped with a warning (Phase 1.5 size guard).
        log_mcap_panel: T × N log(market_cap). Required when
            ``cfg.mcap_neutralize``; ``None`` → skip mcap with a warning.

    Returns:
        New dict with same keys; values transformed (or shallow-copied if cfg
        all-off or size guard tripped). Original input is never mutated.
    """
    if _is_all_off(cfg):
        return dict(factor_panel)

    if n_codes is not None and n_codes < cfg.min_pool_size:
        log.warning(
            "preprocess pipeline skipped: n_codes=%d < min_pool_size=%d "
            "(cross-sectional preprocessing requires a wider panel)",
            n_codes, cfg.min_pool_size,
        )
        return dict(factor_panel)

    do_industry = cfg.industry_neutralize and bool(sector_map)
    if cfg.industry_neutralize and not sector_map:
        log.warning(
            "industry_neutralize=True but sector_map is empty/None; "
            "skipping that step (winsorize/zscore/mcap still applied if enabled)"
        )

    do_mcap = cfg.mcap_neutralize and log_mcap_panel is not None
    if cfg.mcap_neutralize and log_mcap_panel is None:
        log.warning(
            "mcap_neutralize=True but log_mcap_panel is None; skipping mcap step "
            "(caller must build log(market_cap) and pass it in)"
        )

    out: dict[str, pd.DataFrame] = {}
    for name, df in factor_panel.items():
        work = df
        if cfg.winsorize is not None:
            lo, hi = cfg.winsorize
            work = winsorize_panel(work, lo, hi)
        if cfg.zscore:
            work = cs_zscore_panel(work)

        tags = factor_types.get(name, ()) if factor_types else ()
        is_fundamental = "fundamental" in tags
        is_contains_mcap = "contains_mcap" in tags

        # industry: legacy rule — skip on fundamental tag.
        run_industry = do_industry and not is_fundamental
        # mcap: skip only on contains_mcap tag (PE/PB). Other fundamentals OK.
        run_mcap = do_mcap and not is_contains_mcap

        if run_industry and run_mcap:
            work = industry_neutralize_panel(
                work, sector_map, log_mcap=log_mcap_panel,
            )
        elif run_industry:
            work = industry_neutralize_panel(work, sector_map)
        elif run_mcap:
            work = mcap_neutralize_panel(work, log_mcap_panel)

        out[name] = work
    return out
```

- [ ] **Step 4: Run new tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_mcap.py -v
```

Expected: all PASS (new + existing in this file).

- [ ] **Step 5: Run full preprocess + config tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py tests/test_ml_preprocess_mcap.py tests/test_config.py -q
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess_mcap.py
git commit -m "feat(preprocess): route log_mcap_panel through apply_preprocess_pipeline"
```

---

### Task 7: Wire `cache_dir` into `build_factor_panel` and build `log_mcap_panel`

**Files:**
- Modify: `src/stockpool/strategy_factory.py:117-173, 302`
- Test: `tests/test_strategy_factory_mcap.py` (new)

- [ ] **Step 1: Write the failing test in `tests/test_strategy_factory_mcap.py`**

```python
"""Integration tests for log_mcap_panel wiring in build_factor_panel."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pool_data(n_stocks=300, n_days=80, seed=0):
    """Synthetic pool_data: {code: daily_df with date/open/high/low/close/volume}."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-09-01", periods=n_days, freq="B")
    pool = {}
    for i in range(n_stocks):
        code = f"{600000 + i:06d}"
        close = 10.0 + rng.standard_normal(n_days).cumsum() * 0.5
        close = np.clip(close, 1.0, None)
        df = pd.DataFrame({
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.97,
            "close": close,
            "volume": rng.integers(1e5, 1e7, size=n_days),
        })
        pool[code] = df
    return pool


def test_build_factor_panel_builds_log_mcap_when_enabled(monkeypatch, tmp_path):
    """mcap_neutralize=True triggers build_log_mcap_panel + passes to pipeline."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import build_factor_panel

    pool = _pool_data(n_stocks=300, n_days=80, seed=1)
    # Stub balance fundamentals (300 stocks × 1 quarter each)
    fake_balance = pd.DataFrame({
        "code": list(pool.keys()),
        "pubDate": pd.to_datetime(["2024-06-30"] * len(pool)),
        "statDate": pd.to_datetime(["2024-03-31"] * len(pool)),
        "totalShare": [1e9] * len(pool),
    })
    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: fake_balance,
    )

    call_log = {"log_mcap_panel": None}
    original_apply = None

    from stockpool.ml import preprocess as preproc_mod
    original_apply = preproc_mod.apply_preprocess_pipeline

    def spy_apply(factor_panel, cfg, **kwargs):
        call_log["log_mcap_panel"] = kwargs.get("log_mcap_panel")
        return original_apply(factor_panel, cfg, **kwargs)

    monkeypatch.setattr(preproc_mod, "apply_preprocess_pipeline", spy_apply)

    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
        min_pool_size=200,
    )
    result = build_factor_panel(
        ["momentum_20"], pool, preprocess_cfg=cfg, cache_dir=str(tmp_path),
    )
    assert call_log["log_mcap_panel"] is not None
    assert call_log["log_mcap_panel"].shape[1] == len(pool)
    assert "momentum_20" in result


def test_build_factor_panel_skips_log_mcap_when_disabled(monkeypatch, tmp_path):
    """mcap_neutralize=False → log_mcap_panel must be None (no balance fetch)."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import build_factor_panel

    pool = _pool_data(n_stocks=300, n_days=80, seed=2)

    # Should NOT be called at all
    fetch_calls = {"count": 0}

    def fake_loader(table, cache_dir=None):
        fetch_calls["count"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals", fake_loader,
    )

    cfg = PreprocessConfig(
        winsorize=None, zscore=True,
        industry_neutralize=False, mcap_neutralize=False,
        min_pool_size=200,
    )
    build_factor_panel(
        ["momentum_20"], pool, preprocess_cfg=cfg, cache_dir=str(tmp_path),
    )
    assert fetch_calls["count"] == 0


def test_build_factor_panel_warns_when_mcap_on_but_cache_dir_none(monkeypatch, caplog):
    """mcap_neutralize=True with cache_dir=None → warning, skips mcap (no crash)."""
    import logging
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import build_factor_panel

    pool = _pool_data(n_stocks=300, n_days=80, seed=3)
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
        min_pool_size=200,
    )
    with caplog.at_level(logging.WARNING):
        result = build_factor_panel(
            ["momentum_20"], pool, preprocess_cfg=cfg, cache_dir=None,
        )
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "mcap_neutralize=True" in msgs and "cache_dir" in msgs
    assert "momentum_20" in result
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/Scripts/python.exe -m pytest tests/test_strategy_factory_mcap.py -v
```

Expected: FAIL with `TypeError: build_factor_panel() got an unexpected keyword argument 'cache_dir'`.

- [ ] **Step 3: Update `build_factor_panel` in `src/stockpool/strategy_factory.py`**

Find the existing function (line 117) and replace it with this version that adds `cache_dir` parameter and builds `log_mcap_panel` when toggled:

```python
def build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    preprocess_cfg: "PreprocessConfig | None" = None,
    cache_dir: str | Path | None = None,
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
            / mcap_neutralize 流水线(见 ``ml/preprocess.py``)。sector_map
            从 ``factors.context.get_sector_map()`` 读取(caller 责任注入)。
        cache_dir: 用于 ``build_log_mcap_panel`` 取 baostock balance 缓存的
            根目录。``mcap_neutralize=True`` 且 ``cache_dir=None`` 时 log 一条
            warning 并跳过 mcap 步骤(其他步骤照常)。
    """
    from stockpool.ml.dataset import compute_factor_panel
    from stockpool.ml import preprocess as preproc_mod

    # 1) 把每股 daily_df → date-indexed,按列拼成宽表
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
        s.base_name: s.types for s in list_specs() if s.base_name in factor_names
    }

    log_mcap_panel = None
    if preprocess_cfg.mcap_neutralize:
        if cache_dir is None:
            log.warning(
                "mcap_neutralize=True but cache_dir=None; cannot load baostock "
                "balance — skipping mcap step (winsorize/zscore/industry still applied)"
            )
        else:
            from stockpool.ml.mcap import build_log_mcap_panel
            log_mcap_panel = build_log_mcap_panel(panel, cache_dir=cache_dir)

    return preproc_mod.apply_preprocess_pipeline(
        raw, preprocess_cfg, sector_map=sector_map, factor_types=types_map,
        n_codes=len(pool_data), log_mcap_panel=log_mcap_panel,
    )
```

Also update **the existing call inside `build_strategy`** (around line 75) to pass `cache_dir`:

```python
        if (
            factor_panel is None
            and cfg.strategy.ml_factor.panel_mode == "pooled"
            and pool_data
        ):
            factor_panel = build_factor_panel(
                cfg.strategy.ml_factor.factors, pool_data,
                cache_dir=cfg.data.cache_dir,
            )
```

And **the existing call inside `load_or_build_factor_panel`** (line 302) to pass `cache_dir`:

```python
    factor_panel = build_factor_panel(
        factor_names, pool_data,
        preprocess_cfg=preprocess_cfg,
        cache_dir=cache_dir,
    )
```

- [ ] **Step 4: Run the new tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_strategy_factory_mcap.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run the factor-panel cache regression suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py tests/test_strategy_factory_mcap.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/strategy_factory.py tests/test_strategy_factory_mcap.py
git commit -m "feat(strategy_factory): wire cache_dir + build_log_mcap_panel in build_factor_panel"
```

---

### Task 8: Add `mcap_neutralize` to cache sig + verify cache invalidation

**Files:**
- (No code change needed — `_factor_panel_sig` already dumps the full `PreprocessConfig`)
- Test: `tests/test_factor_panel_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_factor_panel_cache.py`:

```python
def test_factor_panel_sig_changes_when_mcap_neutralize_flips(tmp_path):
    """Toggling mcap_neutralize must produce a different cache sig."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    import pandas as pd

    pool = {
        "600000": pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "close": [1.0, 1.1, 1.2, 1.3, 1.4],
        }),
    }
    cfg_off = PreprocessConfig(
        winsorize=None, zscore=True,
        industry_neutralize=False, mcap_neutralize=False,
    )
    cfg_on = PreprocessConfig(
        winsorize=None, zscore=True,
        industry_neutralize=False, mcap_neutralize=True,
    )
    sig_off, _ = _factor_panel_sig(["momentum_20"], pool, preprocess_cfg=cfg_off)
    sig_on, _ = _factor_panel_sig(["momentum_20"], pool, preprocess_cfg=cfg_on)
    assert sig_off != sig_on
```

- [ ] **Step 2: Run the test to verify it passes immediately**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py::test_factor_panel_sig_changes_when_mcap_neutralize_flips -v
```

Expected: PASS (the existing `_factor_panel_sig` already JSON-dumps `cfg.model_dump()` so the new field auto-changes hash). This is a regression guard.

- [ ] **Step 3: Run the full cache test suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_factor_panel_cache.py
git commit -m "test(factor_panel_cache): assert mcap_neutralize toggle changes sig"
```

---

### Task 9: Register `market_cap` and `log_market_cap` as factors + tag PE/PB with `contains_mcap`

**Files:**
- Modify: `src/stockpool/factors/fundamentals.py`
- Test: `tests/test_factors_fundamentals.py`

- [ ] **Step 1: Write the failing tests in `tests/test_factors_fundamentals.py`**

Append to the file:

```python
def test_market_cap_factor_registered_and_computes(monkeypatch, tmp_path):
    """market_cap factor: close × totalShare PIT-aligned by pubDate."""
    from stockpool.factors.registry import make_factor
    import pandas as pd
    import numpy as np

    fake_balance = pd.DataFrame({
        "code": ["600000"],
        "pubDate": pd.to_datetime(["2024-12-15"]),
        "statDate": pd.to_datetime(["2024-09-30"]),
        "totalShare": [6e8],
    })
    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: fake_balance,
    )

    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    close = pd.DataFrame({"600000": [10.0, 11.0, 12.0]}, index=dates)
    panel = {"close": close, "open": close, "high": close, "low": close, "volume": close}

    factor = make_factor("market_cap")
    out = factor.compute(panel)
    expected = close["600000"] * 6e8
    np.testing.assert_allclose(out["600000"].values, expected.values)


def test_log_market_cap_factor_registered_and_computes(monkeypatch):
    """log_market_cap = log(close × totalShare), NaN where mcap ≤ 0 / missing."""
    from stockpool.factors.registry import make_factor
    import pandas as pd
    import numpy as np

    fake_balance = pd.DataFrame({
        "code": ["600000"],
        "pubDate": pd.to_datetime(["2024-12-15"]),
        "statDate": pd.to_datetime(["2024-09-30"]),
        "totalShare": [6e8],
    })
    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: fake_balance,
    )

    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    close = pd.DataFrame({"600000": [10.0, 11.0, 12.0]}, index=dates)
    panel = {"close": close, "open": close, "high": close, "low": close, "volume": close}

    factor = make_factor("log_market_cap")
    out = factor.compute(panel)
    expected = np.log(close["600000"] * 6e8)
    np.testing.assert_allclose(out["600000"].values, expected.values)


def test_pe_factor_has_contains_mcap_tag():
    """PE registration tag tuple must include 'contains_mcap'."""
    from stockpool.factors.registry import list_specs
    pe_spec = next(s for s in list_specs() if s.base_name == "pe")
    assert "contains_mcap" in pe_spec.types


def test_pb_factor_has_contains_mcap_tag():
    from stockpool.factors.registry import list_specs
    pb_spec = next(s for s in list_specs() if s.base_name == "pb")
    assert "contains_mcap" in pb_spec.types


def test_market_cap_factor_has_size_tag():
    from stockpool.factors.registry import list_specs
    spec = next(s for s in list_specs() if s.base_name == "market_cap")
    assert "size" in spec.types
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factors_fundamentals.py -k "market_cap or contains_mcap" -v
```

Expected: FAIL with `KeyError: 'market_cap is not registered'` for the factor tests and `AssertionError` for the tag tests.

- [ ] **Step 3: Update `src/stockpool/factors/fundamentals.py`**

Add `"contains_mcap"` to the existing PE and PB `@register` decorators. Find these blocks and edit the `types=` tuple:

```python
@register(
    "pe",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "contains_mcap"),
    description="市盈率(总市值 / 滚动 4 季净利润)。值越低越便宜,但要警惕周期顶反转;亏损公司返回 NaN。",
)
class PEFactor(Factor):
    ...
```

```python
@register(
    "pb",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "contains_mcap"),
    description="市净率(总市值 / 股东权益)。低 PB 常见于金融/周期股,需要配合 ROE 才能判断是否真便宜。",
)
class PBFactor(Factor):
    ...
```

Then at the bottom of the file, add the two new factor classes:

```python
@register(
    "market_cap",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "size"),
    description="总市值 (close × 总股本)。规模因子,小盘溢价 / 大盘稳定的常用代理。严格按公告日 PIT 对齐。",
)
class MarketCapFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "market_cap"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        balance = load_or_build_fundamentals("balance", cache_dir=_default_cache_dir())
        if balance is None or balance.empty:
            return pd.DataFrame(
                np.nan, index=panel["close"].index, columns=panel["close"].columns,
            )
        shares_panel = _pit_align(balance, "totalShare", panel["close"])
        return panel["close"] * shares_panel


@register(
    "log_market_cap",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "size"),
    description="log(总市值)。剥离市值 β 时常用;线性回归更稳定。NaN 出现在停牌 / 无股本数据 / mcap ≤ 0 时。",
)
class LogMarketCapFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "log_market_cap"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mcap = MarketCapFactor().compute(panel)
        return np.log(mcap.where(mcap > 0))
```

- [ ] **Step 4: Run new tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factors_fundamentals.py -k "market_cap or contains_mcap" -v
```

Expected: 5 PASS.

- [ ] **Step 5: Run full factors-fundamentals + factor registry tests for regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factors_fundamentals.py tests/test_factors.py -q
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/factors/fundamentals.py tests/test_factors_fundamentals.py
git commit -m "feat(factors): register market_cap / log_market_cap, tag PE/PB contains_mcap"
```

---

### Task 10: Full test suite + documentation update for PR-1

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md` (if it documents `preprocess:` block)

- [ ] **Step 1: Run the entire test suite to check for any regression**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: all PASS (615 + ~17 new ≈ 632 tests). If anything fails, fix before continuing.

- [ ] **Step 2: Update `CLAUDE.md` — preprocess config row**

Find the existing line(s) describing `preprocess.{winsorize, zscore, industry_neutralize, min_pool_size}` and update to include `mcap_neutralize`:

```markdown
- **`preprocess.{winsorize, zscore, industry_neutralize, mcap_neutralize, min_pool_size}`**(2026-06-06, Phase 1+1.5+2):截面预处理流水线。Phase 1 三步 + Phase 2 `mcap_neutralize`(per-day OLS Y ~ log(market_cap),或与 industry 联合 OLS)。`mcap_neutralize` default `false`,需 baostock `balance.totalShare` parquet 缓存(`fundamentals_loader` 30 天),验证见 `docs/ab_validation_results.md` P5-mcap 段。`min_pool_size: int = 200` runtime guard 同时守 mcap 步。PE/PB 因子打 `contains_mcap` tag,自动跳过 mcap 神经化(避免与 close × shares 强共线)。`mcap_neutralize=True` 时 `build_factor_panel` 接 `cache_dir` 参数后调 `ml.mcap.build_log_mcap_panel` 现 build,不入 panel cache。baked 进 `factor_panels/<sig>/` cache → 改 preprocess 自动新 sig 重算。
```

Find the "**`src/stockpool/ml/preprocess.py`** | **截面预处理流水线**" row in the module table — append `+ mcap_neutralize` to the description. Also add a new row for the new module:

```markdown
| `src/stockpool/ml/mcap.py` | **log(market_cap) 面板构造**(Phase 2):`build_log_mcap_panel(panel, cache_dir)` 用 baostock `balance.totalShare × close` PIT-aligned。不入 cache,每次现 build(< 100ms on 4000 stocks × 250 days)。 |
```

Find the test table — add new rows:

```markdown
| `test_ml_preprocess_mcap.py` | `mcap_neutralize_panel` OLS 残差化 / `industry_neutralize_panel(log_mcap=...)` 联合 OLS / `apply_preprocess_pipeline` 路由与 skip 规则 / 单成员 industry 不被零化 / fallback 计数 |
| `test_ml_mcap.py` | `build_log_mcap_panel`:close × totalShare、PIT 对齐、缺数据 NaN、空 balance 全 NaN 输出 |
| `test_strategy_factory_mcap.py` | `build_factor_panel(cache_dir=...)` 在 `mcap_neutralize=True` 时调 `build_log_mcap_panel` 并传给 pipeline / disabled 时不 fetch balance / `cache_dir=None` 时 warning + skip |
```

Find the factors table — add a row for the new module *no*, `market_cap` lives in existing `factors/fundamentals.py`. Instead update the existing fundamentals row to mention size factors and the new tag:

```markdown
| `src/stockpool/factors/fundamentals.py` | 基本面因子(PE/PB/ROE/ROA/毛利率/净利率/营收 YOY)+ **size 因子**(market_cap / log_market_cap),baostock 5 张季度表,严格 PIT。PE/PB 打 `contains_mcap` tag 防 mcap 神经化共线。 |
```

- [ ] **Step 3: Check whether `README.md` documents `preprocess:` block**

```bash
.venv/Scripts/python.exe -c "import pathlib; print('preprocess' in pathlib.Path('README.md').read_text(encoding='utf-8'))"
```

If True, update README with a `mcap_neutralize: false  # Phase 2` line in the yaml example and a one-sentence explanation. If False, skip this step.

- [ ] **Step 4: Re-run all tests to confirm nothing broke from doc-only commits**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document mcap_neutralize (Phase 2) + log_market_cap factor + new modules"
```

---

## Task Sequence (PR-2: A/B yaml + run + verdict)

### Task 11: Create `ab_mcap.yaml`

**Files:**
- Create: `ab_mcap.yaml` (project root, next to `ab_preprocess.yaml`)

- [ ] **Step 1: Confirm `reports/selection.json` exists (prerequisite)**

```bash
.venv/Scripts/python.exe -c "import pathlib, json; p=pathlib.Path('reports/selection.json'); print('exists' if p.exists() else 'MISSING', 'factors:', len(json.loads(p.read_text())['factors']) if p.exists() else 'n/a')"
```

Expected: `exists factors: <some count>`. If MISSING, stop and ask the user to regenerate via `python -m stockpool factors pick-by-ic` first.

- [ ] **Step 2: Create `ab_mcap.yaml`**

```yaml
base_config: config.yaml

arms:
  preprocess_only:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: all
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
          winsorize: [0.01, 0.99]
          zscore: true
          industry_neutralize: false
          mcap_neutralize: false
    backtest:
      equity_curve_holding_days:
      - 10

  preprocess_plus_mcap:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: all
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
          industry_neutralize: false
          mcap_neutralize: true
    backtest:
      equity_curve_holding_days:
      - 10
```

- [ ] **Step 3: Verify yaml parses + passes ABConfig validation**

```bash
.venv/Scripts/python.exe -c "from stockpool.ab.config import load_ab_config; cfg = load_ab_config('ab_mcap.yaml'); print('arms:', list(cfg.arms.keys()))"
```

Expected: `arms: ['preprocess_only', 'preprocess_plus_mcap']`. If it raises, fix the yaml.

- [ ] **Step 4: Commit**

```bash
git add ab_mcap.yaml
git commit -m "feat(ab): add ab_mcap.yaml comparing preprocess vs preprocess+mcap_neutralize"
```

---

### Task 12: Dry-run single arm to verify OLS warnings stay sane

**Files:**
- (None — runtime only)

- [ ] **Step 1: Run treatment arm in single-arm debug mode**

```bash
.venv/Scripts/python.exe -m stockpool ab --config ab_mcap.yaml --arm preprocess_plus_mcap 2>&1 | tee /tmp/ab_mcap_dry.log
```

Expected:
- Prints per-stock metrics to stdout
- Logs from `mcap_neutralize_panel` / `industry_neutralize_panel` should show OLS fallback < ~5% of days (i.e. message form: `fallback on N / M days` with N/M < 0.05)
- Process exits 0
- No HTML written (single-arm mode)

- [ ] **Step 2: Inspect log for red flags**

```bash
grep -E "fallback|skipping|skipped" /tmp/ab_mcap_dry.log
```

Look for:
- `fallback on X / Y days` — X should be small (< 10% of Y)
- Should NOT see `mcap_neutralize=True but log_mcap_panel is None` (would mean cache_dir wiring is broken)
- Should NOT see `preprocess pipeline skipped: n_codes=` (would mean universe < 200)

If any red flag appears, stop and debug before committing the full AB run.

---

### Task 13: Run full A/B + record verdict

**Files:**
- Modify: `docs/ab_validation_results.md`

- [ ] **Step 1: Run the full A/B**

```bash
.venv/Scripts/python.exe -m stockpool ab --config ab_mcap.yaml 2>&1 | tee /tmp/ab_mcap_full.log
```

Expected:
- Both arms complete
- HTML report written under `reports/ab/` (path printed at end)
- Process exits 0

- [ ] **Step 2: Extract per-arm aggregate Sharpe + annualized return from stdout**

The CLI prints aggregate metrics for each arm. Capture:

| Metric | preprocess_only | preprocess_plus_mcap | Δ |
|---|---|---|---|
| mean Sharpe | A | B | B - A |
| median Sharpe | A | B | B - A |
| mean ann_return | A | B | B - A |
| median ann_return | A | B | B - A |
| #stocks where treatment beats baseline (Sharpe) | — | N/total | — |

Open the HTML to copy values if stdout truncates.

- [ ] **Step 3: Apply verdict rule from the spec**

| Verdict | Rule |
|---|---|
| PASS | `Δ mean Sharpe ≥ +0.10` **AND** `Δ mean ann_return ≥ +1%` |
| HOLD | `|Δ mean Sharpe| < 0.05` |
| REJECT | `Δ mean Sharpe ≤ -0.10` **OR** `Δ mean ann_return ≤ -1%` |

(For values in between, lean conservative — HOLD unless clearly above PASS line.)

- [ ] **Step 4: Append a section to `docs/ab_validation_results.md`**

Find the bottom of the file and add:

```markdown
## P5-mcap: Phase 2 market-cap neutralization (2026-06-06)

**Config:** `ab_mcap.yaml`
- Baseline (`preprocess_only`): winsorize [0.01, 0.99] + cs_zscore + industry_neutralize=false + mcap_neutralize=false
- Treatment (`preprocess_plus_mcap`): same + mcap_neutralize=true (per-day OLS Y ~ log(market_cap))

**Universe:** 4000+ A-share training pool (training_universe=all), `factors_file: reports/selection.json`,
walk-forward horizon=3, refit_every=20, holding_days=10, pooled panel mode.

**Results:**

| Metric | preprocess_only | preprocess_plus_mcap | Δ |
|---|---|---|---|
| Mean Sharpe | TBD-fill | TBD-fill | TBD-fill |
| Median Sharpe | TBD-fill | TBD-fill | TBD-fill |
| Mean annualized return | TBD-fill | TBD-fill | TBD-fill |
| Median annualized return | TBD-fill | TBD-fill | TBD-fill |
| Treatment wins (Sharpe, per-stock) | — | TBD-fill / TBD-fill | — |
| OLS fallback days | — | TBD-fill / total | — |

**Verdict:** TBD-PASS / HOLD / REJECT

**Decision:** TBD-fill (e.g. "default left at `mcap_neutralize: false`; flag retained for users who opt in via yaml")

**Notes:**
- OLS fallback rate stayed < TBD-fill% (acceptable, < 5% target)
- TBD-fill any surprising regime behaviour

**Follow-up:**
- TBD-fill (only if PASS): consider second AB with `industry_neutralize=true` baseline + treatment to test joint OLS
```

Fill in the TBD-* placeholders with the actual numbers from steps 2-3 and the verdict from step 3.

- [ ] **Step 5: Commit**

```bash
git add docs/ab_validation_results.md
git commit -m "docs(ab): record P5-mcap verdict (mcap_neutralize phase 2 A/B)"
```

---

## Self-Review (run before declaring complete)

After implementing all tasks, verify against the spec:

**Spec coverage check:**
- §3.1 mcap_panel build → Task 5
- §3.2 market_cap / log_market_cap factors → Task 9
- §4.1-4.3 OLS math + fallback → Tasks 3, 4
- §5.1-5.5 PreprocessConfig + pipeline routing + PE/PB tag → Tasks 1, 2, 6, 9
- §6 cache invalidation → Task 8
- §7 A/B yaml + verdict → Tasks 11, 12, 13
- §8 test matrix all covered → Tasks 1-9 cumulatively
- §9 PR sequencing → PR-1 = Tasks 1-10, PR-2 = Tasks 11-13
- §10 documentation → Task 10

**Type / name consistency check:**
- `mcap_neutralize_panel(df, log_mcap)` — same signature in Tasks 3, 6
- `build_log_mcap_panel(panel, cache_dir)` — Task 5 + Task 7 + Task 9 docstrings agree
- `apply_preprocess_pipeline(..., log_mcap_panel=None)` — Tasks 6, 7 agree on kwarg name `log_mcap_panel`
- `_per_day_ols_residual` — used in Tasks 3 + 4 with same return tuple shape `(Series, bool)`
- `contains_mcap` tag string — Tasks 6 (pipeline) + 9 (registration) agree
- `size` tag string — Task 9 (factor registration) + factor type filter (existing infra) agree

If anything diverges, fix it before invoking executing-plans.
