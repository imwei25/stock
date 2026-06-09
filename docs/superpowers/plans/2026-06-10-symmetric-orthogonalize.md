# Symmetric (Löwdin) Orthogonalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in per-day cross-sectional symmetric (Löwdin) orthogonalization step to the factor-panel preprocessing pipeline that decorrelates the selected factors, then A/B test whether it improves the backtest.

**Architecture:** A new *joint* transform `symmetric_orthogonalize_panel` in `ml/preprocess.py` runs as the final pipeline step (after the existing per-factor steps). It's gated by a new `PreprocessConfig.symmetric_orthogonalize` bool, baked into the existing `factor_panels/<sig>/` cache (the sig already hashes the full preprocess cfg), so train/predict consistency and A/B cache isolation come for free. Stateless per day → look-ahead safe. IC/IR weighting runs on the decorrelated panel unchanged.

**Tech Stack:** Python, numpy (`np.linalg.eigh`), pandas, pydantic v2, pytest. Run python via `.venv/Scripts/python.exe`.

**Reference spec:** `docs/superpowers/specs/2026-06-10-symmetric-orthogonalize-design.md`

---

## File Structure

- **Modify** `src/stockpool/config.py` — add `symmetric_orthogonalize: bool = False` to `PreprocessConfig`.
- **Modify** `src/stockpool/ml/preprocess.py` — add `symmetric_orthogonalize_panel(...)`, wire into `apply_preprocess_pipeline`, update `_is_all_off`.
- **Create** `tests/test_ml_preprocess_orthogonalize.py` — unit tests for the new transform.
- **Modify** `tests/test_ml_preprocess.py` — update `_is_all_off` cases (no new field breaks them, but add an explicit case).
- **Modify** `tests/test_config.py` — config field parse/default/toggle.
- **Modify** `tests/test_factor_panel_cache.py` — sig changes when flag flips.
- **Create** `ab_orthogonalize_small.yaml`, `ab_orthogonalize.yaml` — A/B configs.
- **Modify** `CLAUDE.md`, `README.md`, `docs/ab_validation_results.md` — docs.

---

## Task 1: Config field

**Files:**
- Modify: `src/stockpool/config.py:367-394` (`PreprocessConfig`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py` (after `test_preprocess_market_cap_neutralize_togglable`, ~line 685):

```python
def test_preprocess_symmetric_orthogonalize_default_off():
    """symmetric_orthogonalize defaults to False."""
    from stockpool.config import PreprocessConfig
    cfg = PreprocessConfig()
    assert cfg.symmetric_orthogonalize is False


def test_preprocess_symmetric_orthogonalize_togglable():
    """symmetric_orthogonalize is an independent bool switch."""
    from stockpool.config import PreprocessConfig
    cfg = PreprocessConfig(symmetric_orthogonalize=True)
    assert cfg.symmetric_orthogonalize is True
    assert cfg.market_cap_neutralize is False
```

Also extend `test_preprocess_config_defaults_all_off` (line ~669) by adding one assertion line at the end:

```python
    assert cfg.symmetric_orthogonalize is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -k symmetric_orthogonalize -q`
Expected: FAIL — `PreprocessConfig` has no field `symmetric_orthogonalize` (pydantic `extra=forbid` raises ValidationError).

- [ ] **Step 3: Add the field**

In `src/stockpool/config.py`, inside `PreprocessConfig`, after the `min_pool_size` field (line ~382) add:

```python
    symmetric_orthogonalize: bool = False
    # symmetric_orthogonalize (2026-06-10): per-day cross-sectional Löwdin
    # 对称正交化 — jointly decorrelates the non-fundamental factors so the
    # weighter sees mutually-orthogonal inputs. Runs as the FINAL preprocess
    # step (after winsorize/zscore/neutralize). Order-independent, stateless
    # per day → look-ahead safe. Default False (opt-in). Fundamental-tagged
    # factors pass through untouched (same as the neutralize steps).
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -k "preprocess" -q`
Expected: PASS (all preprocess config tests including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "feat(config): add preprocess.symmetric_orthogonalize flag"
```

---

## Task 2: `_is_all_off` includes the new flag

**Files:**
- Modify: `src/stockpool/ml/preprocess.py:177-184` (`_is_all_off`)
- Test: `tests/test_ml_preprocess.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ml_preprocess.py` (after `test_is_all_off_false_for_market_cap_neutralize`, ~line 412):

```python
def test_is_all_off_false_for_symmetric_orthogonalize():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    assert _is_all_off(PreprocessConfig(symmetric_orthogonalize=True)) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py::test_is_all_off_false_for_symmetric_orthogonalize -q`
Expected: FAIL — `_is_all_off` returns `True` because it doesn't check the new flag yet.

- [ ] **Step 3: Update `_is_all_off`**

In `src/stockpool/ml/preprocess.py`, change `_is_all_off` to:

```python
def _is_all_off(cfg: "PreprocessConfig") -> bool:
    """True when every step is disabled (cfg semantically a no-op)."""
    return (
        cfg.winsorize is None
        and cfg.zscore is False
        and cfg.industry_neutralize is False
        and cfg.market_cap_neutralize is False
        and cfg.symmetric_orthogonalize is False
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -k "is_all_off" -q`
Expected: PASS (all `is_all_off` cases).

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess.py
git commit -m "feat(preprocess): _is_all_off accounts for symmetric_orthogonalize"
```

---

## Task 3: `symmetric_orthogonalize_panel` — core transform

**Files:**
- Modify: `src/stockpool/ml/preprocess.py` (add function after `market_cap_neutralize_panel`, ~line 175)
- Test: `tests/test_ml_preprocess_orthogonalize.py` (create)

The function decorrelates the **non-fundamental** factors jointly per day. Algorithm per day `t`: take the all-factors-valid stock subset; if `N_valid < K_nf` (or day empty) pass through unchanged; else z-score each column, form `M = F_stdᵀ F_std / N_valid`, eigendecompose (`eigh`), floor eigenvalues at `1e-10`, build `S = U diag(λ⁺^-1/2) Uᵀ`, write `F_std · S` back to the valid rows.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ml_preprocess_orthogonalize.py`:

```python
"""Unit tests for per-day symmetric (Löwdin) orthogonalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _panel(n_days=4, n_stocks=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    return rng.standard_normal((n_days, n_stocks)), dates, codes


def _correlated_pair(n_days=4, n_stocks=300, seed=0, rho=0.8):
    """Two factor panels f2 strongly correlated with f1 (per day)."""
    base, dates, codes = _panel(n_days, n_stocks, seed)
    noise, _, _ = _panel(n_days, n_stocks, seed + 100)
    f1 = pd.DataFrame(base, index=dates, columns=codes)
    f2 = pd.DataFrame(rho * base + np.sqrt(1 - rho**2) * noise,
                      index=dates, columns=codes)
    return {"f1": f1, "f2": f2}


def test_output_is_orthogonal_per_day():
    """After transform, the per-day cross-sectional Gram matrix is ~diagonal."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.85)
    out = symmetric_orthogonalize_panel(fp)
    for d in fp["f1"].index:
        a = out["f1"].loc[d].to_numpy()
        b = out["f2"].loc[d].to_numpy()
        # standardize for correlation
        ca = (a - a.mean()) / a.std(ddof=0)
        cb = (b - b.mean()) / b.std(ddof=0)
        corr = float((ca * cb).mean())
        assert abs(corr) < 1e-6, f"day {d} corr={corr}"


def test_order_independent():
    """Permuting input factor order yields the same per-column result (Löwdin)."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.7)
    out_ab = symmetric_orthogonalize_panel({"f1": fp["f1"], "f2": fp["f2"]})
    out_ba = symmetric_orthogonalize_panel({"f2": fp["f2"], "f1": fp["f1"]})
    pd.testing.assert_frame_equal(out_ab["f1"], out_ba["f1"])
    pd.testing.assert_frame_equal(out_ab["f2"], out_ba["f2"])


def test_close_to_original():
    """Each orthogonalized factor stays sign-aligned & correlated with original."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.6)
    out = symmetric_orthogonalize_panel(fp)
    for name in ("f1", "f2"):
        d = fp[name].index[0]
        orig = fp[name].loc[d].to_numpy()
        new = out[name].loc[d].to_numpy()
        co = (orig - orig.mean()) / orig.std(ddof=0)
        cn = (new - new.mean()) / new.std(ddof=0)
        assert float((co * cn).mean()) > 0.5  # still close to original


def test_degenerate_day_passthrough():
    """A day with fewer valid stocks than factors is returned unchanged."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(n_days=3, n_stocks=300, rho=0.7)
    # Day index 1: only 1 stock valid across both factors (< K=2).
    for name in fp:
        fp[name].iloc[1, 1:] = np.nan
    out = symmetric_orthogonalize_panel(fp)
    for name in fp:
        pd.testing.assert_series_equal(out[name].iloc[1], fp[name].iloc[1])


def test_nan_cells_stay_nan():
    """Stocks with any NaN factor stay NaN in the output."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(n_days=3, n_stocks=300, rho=0.7)
    fp["f1"].iloc[0, 5] = np.nan  # stock 5, day 0 invalid in f1
    out = symmetric_orthogonalize_panel(fp)
    assert np.isnan(out["f1"].iloc[0, 5])
    assert np.isnan(out["f2"].iloc[0, 5])  # dropped from valid subset both sides


def test_fundamental_factor_skipped():
    """A fundamental-tagged factor passes through byte-for-byte."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.7)
    pe = fp["f1"].copy() * 3.0 + 1.0
    fp["pe"] = pe
    factor_types = {"f1": ("momentum",), "f2": ("reversal",), "pe": ("fundamental",)}
    out = symmetric_orthogonalize_panel(fp, factor_types=factor_types)
    pd.testing.assert_frame_equal(out["pe"], pe)
    # non-fundamental still orthogonalized
    d = fp["f1"].index[0]
    a = out["f1"].loc[d].to_numpy(); b = out["f2"].loc[d].to_numpy()
    ca = (a - a.mean()) / a.std(ddof=0); cb = (b - b.mean()) / b.std(ddof=0)
    assert abs(float((ca * cb).mean())) < 1e-6


def test_single_non_fundamental_factor_no_crash():
    """K_nf == 1 → orthogonalization reduces to a per-day z-score, no crash."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.7)
    fp = {"f1": fp["f1"]}
    out = symmetric_orthogonalize_panel(fp)
    assert out["f1"].shape == fp["f1"].shape
    assert not out["f1"].isna().all().all()


def test_input_not_mutated():
    """Original input frames are unchanged after the call."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.7)
    snap = {k: v.copy() for k, v in fp.items()}
    _ = symmetric_orthogonalize_panel(fp)
    for k in fp:
        pd.testing.assert_frame_equal(fp[k], snap[k])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_orthogonalize.py -q`
Expected: FAIL — `ImportError: cannot import name 'symmetric_orthogonalize_panel'`.

- [ ] **Step 3: Implement the function**

In `src/stockpool/ml/preprocess.py`, add after `market_cap_neutralize_panel` (before `_is_all_off`):

```python
def symmetric_orthogonalize_panel(
    factor_panel: dict[str, pd.DataFrame],
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-day cross-sectional symmetric (Löwdin) orthogonalization.

    Jointly decorrelates the **non-fundamental** factors so that, on each day,
    the cross-sectional correlation between any two output factors is ~0, while
    each output factor stays maximally close to its (standardised) input
    (order-independent — unlike Gram-Schmidt). Fundamental-tagged factors pass
    through untouched (orthogonalising PE/PB against momentum muddies the
    intrinsic valuation signal — same rationale as the neutralize steps).

    Stateless per day (each day computes its own transform from that day's
    cross-section only) → look-ahead safe; the predict path reading the same
    cached panel is automatically consistent with training.

    Args:
        factor_panel: ``{factor_name: T × N DataFrame}`` (date index, code cols).
        factor_types: ``{factor_name: (type_tag, ...)}``; names whose tags include
            ``"fundamental"`` are excluded from orthogonalization (copied through).

    Returns:
        New dict, same keys. Non-fundamental factors decorrelated per day;
        fundamental factors and the input frames are never mutated.

        Per-day fallbacks (day returned unchanged for the affected factors):
          * fewer jointly-valid stocks than factors (``N_valid < K``) → cannot
            form a full-rank correlation matrix;
          * all-NaN / empty day.
        Cells where any non-fundamental factor is NaN stay NaN across all
        non-fundamental factors that day (they leave the valid subset).
    """
    types = factor_types or {}
    nf_names = [
        n for n in factor_panel
        if "fundamental" not in types.get(n, ())
    ]
    out: dict[str, pd.DataFrame] = {n: factor_panel[n].copy() for n in factor_panel}
    K = len(nf_names)
    if K == 0:
        return out

    ref = factor_panel[nf_names[0]]
    dates, codes = ref.index, ref.columns
    # Stack non-fundamental factors into a (T, N, K) cube aligned on (dates, codes).
    cube = np.stack(
        [factor_panel[n].reindex(index=dates, columns=codes).to_numpy(dtype=float)
         for n in nf_names],
        axis=-1,
    )  # shape (T, N, K)

    transformed = cube.copy()
    for ti in range(cube.shape[0]):
        day = cube[ti]                          # (N, K)
        valid = ~np.isnan(day).any(axis=1)      # stocks with all K factors present
        n_valid = int(valid.sum())
        if n_valid < K or n_valid == 0:
            continue                            # passthrough this day
        F = day[valid]                          # (n_valid, K)
        mu = F.mean(axis=0)
        sigma = F.std(axis=0, ddof=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        Fs = (F - mu) / sigma                   # per-day z-score
        M = (Fs.T @ Fs) / n_valid               # (K, K) correlation matrix
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-10)    # floor to keep S finite
        S = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
        transformed[ti][valid] = Fs @ S
        # Stocks missing any non-fundamental factor leave the valid subset →
        # NaN them across ALL non-fundamental factors so the day is consistent
        # (they are dropped downstream at stack_panel_to_xy anyway). Days that
        # pass through (the `continue` above) keep their raw values untouched.
        transformed[ti][~valid] = np.nan

    for k, name in enumerate(nf_names):
        out[name] = pd.DataFrame(transformed[:, :, k], index=dates, columns=codes)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess_orthogonalize.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess_orthogonalize.py
git commit -m "feat(preprocess): symmetric_orthogonalize_panel per-day Löwdin transform"
```

---

## Task 4: Wire into `apply_preprocess_pipeline`

**Files:**
- Modify: `src/stockpool/ml/preprocess.py:187-264` (`apply_preprocess_pipeline`)
- Test: `tests/test_ml_preprocess.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ml_preprocess.py` (near the other pipeline tests, e.g. after `test_pipeline_mcap_neutralize_skips_fundamental_and_warns`):

```python
def test_pipeline_runs_orthogonalize_last():
    """symmetric_orthogonalize=True → pipeline output factors are decorrelated."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    # Two correlated factors over a wide cross-section.
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    codes = [f"S{i:03d}" for i in range(300)]
    base = rng.standard_normal((3, 300))
    noise = rng.standard_normal((3, 300))
    f1 = pd.DataFrame(base, index=dates, columns=codes)
    f2 = pd.DataFrame(0.8 * base + 0.6 * noise, index=dates, columns=codes)
    cfg = PreprocessConfig(symmetric_orthogonalize=True)
    out = apply_preprocess_pipeline(
        {"f1": f1, "f2": f2}, cfg, n_codes=300,
    )
    d = dates[0]
    a = out["f1"].loc[d].to_numpy(); b = out["f2"].loc[d].to_numpy()
    ca = (a - a.mean()) / a.std(ddof=0); cb = (b - b.mean()) / b.std(ddof=0)
    assert abs(float((ca * cb).mean())) < 1e-6


def test_pipeline_orthogonalize_skipped_by_size_guard():
    """min_pool_size guard skips orthogonalization on a tiny pool."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _make_panel(n_days=3, n_stocks=10, seed=8)
    cfg = PreprocessConfig(symmetric_orthogonalize=True, min_pool_size=200)
    out = apply_preprocess_pipeline({"f1": df.copy(), "f2": df.copy()},
                                    cfg, n_codes=10)
    pd.testing.assert_frame_equal(out["f1"], df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -k "orthogonalize_last or size_guard" -q`
Expected: `test_pipeline_runs_orthogonalize_last` FAILS (factors still correlated — step not wired); `test_pipeline_orthogonalize_skipped_by_size_guard` may already pass (guard short-circuits whole pipeline).

- [ ] **Step 3: Wire the step in**

In `apply_preprocess_pipeline`, after the `do_mcap` warning block (before the `for name, df in factor_panel.items()` loop, ~line 248), add:

```python
    do_ortho = cfg.symmetric_orthogonalize
```

Then after the per-factor loop builds `out` (right before `return out`, ~line 263), add:

```python
    if do_ortho:
        out = symmetric_orthogonalize_panel(out, factor_types=factor_types)
    return out
```

(Replace the existing bare `return out` with the block above.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ml_preprocess.py -q`
Expected: PASS (full preprocess suite).

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ml/preprocess.py tests/test_ml_preprocess.py
git commit -m "feat(preprocess): run symmetric_orthogonalize as final pipeline step"
```

---

## Task 5: Cache sig invalidation

**Files:**
- Test: `tests/test_factor_panel_cache.py` (no source change — verifying existing sig machinery picks up the new field)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_factor_panel_cache.py` (after `test_cache_invalidates_on_preprocess_change`, ~line 401):

```python
def test_cache_sig_changes_on_symmetric_orthogonalize():
    """Flipping symmetric_orthogonalize yields a distinct factor-panel sig."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    pool = _pool(["S001"])
    sig_off, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(zscore=True),
    )
    sig_on, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(zscore=True, symmetric_orthogonalize=True),
    )
    assert sig_off != sig_on
```

- [ ] **Step 2: Run test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py::test_cache_sig_changes_on_symmetric_orthogonalize -q`
Expected: PASS immediately — `_factor_panel_sig` dumps the full `PreprocessConfig` (via `model_dump()`) into the hash when not all-off, so the new field is already included. (If it FAILS, that means the sig isn't capturing the field — investigate `_factor_panel_sig` in `strategy_factory.py`.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_factor_panel_cache.py
git commit -m "test(cache): symmetric_orthogonalize flips factor-panel sig"
```

---

## Task 6: Full test sweep

- [ ] **Step 1: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: PASS (all prior + new tests; ~625 total).

- [ ] **Step 2: If anything fails, fix it before proceeding.** Do not move on with red tests.

---

## Task 7: A/B configs + smoke run

**Files:**
- Create: `ab_orthogonalize_small.yaml`
- Create: `ab_orthogonalize.yaml`

- [ ] **Step 1: Create the small smoke config**

Create `ab_orthogonalize_small.yaml`. It must use a pool ≥ 200 stocks (so the `min_pool_size` guard doesn't skip orthogonalization) — use `training_universe: all` so the cross-section is wide, and `stocks_filter` to keep the application/backtest set small for speed. Base it on `ab_neutralize_confirm.yaml`:

```yaml
# Smoke A/B for symmetric orthogonalization: confirms the module runs
# end-to-end on a wide cross-section and gives a first directional read.
# Both arms run training_universe=all (so n_codes >> min_pool_size and the
# orthogonalization step actually executes); stocks_filter keeps the backtest
# set small for speed.
base_config: config.yaml
stocks_filter:
  - "605589"
arms:
  base:
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
          lasso: {alpha: 0.001, max_iter: 1000, tol: 1.0e-06}
        weighter: &id002
          type: ic
          ic: {use_rank: true, min_abs_ic: 0.0}
          ir: {n_chunks: 6, use_rank: true, min_abs_ir: 0.0}
        thresholds: &id003 {strong_buy: 0.9, buy: 0.7, sell: 0.3, strong_sell: 0.1}
        buy_verdicts: &id004 [buy, strong_buy]
        sell_verdicts: &id005 [sell, strong_sell]
        refresh_verdicts: &id006 [strong_buy]
        mask: {enabled: false}
        preprocess:
          winsorize: [0.01, 0.99]
          zscore: true
          industry_neutralize: false
          market_cap_neutralize: true
          symmetric_orthogonalize: false
    backtest:
      equity_curve_holding_days: [10]
  ortho:
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
        mask: {enabled: false}
        preprocess:
          winsorize: [0.01, 0.99]
          zscore: true
          industry_neutralize: false
          market_cap_neutralize: true
          symmetric_orthogonalize: true
    backtest:
      equity_curve_holding_days: [10]
```

> NOTE: confirm `605589` (or whatever the smoke ticker is) exists in `config.yaml`'s `stocks` list; `stocks_filter` can only subset. If unsure, run `.venv/Scripts/python.exe -c "import yaml; print([s['code'] for s in yaml.safe_load(open('config.yaml'))['stocks']])"` and pick 1-2 codes from the output.

- [ ] **Step 2: Run the smoke A/B**

Run: `.venv/Scripts/python.exe -m stockpool ab --config ab_orthogonalize_small.yaml`
Expected: completes without error, writes an HTML report under `reports/ab/`. Note the per-arm metrics printed to stdout.

If it errors on missing `mcap_shares.parquet` or factor panel: the `market_cap_neutralize` step warns-and-skips on missing mcap (non-fatal). A factor-panel rebuild is expected (new sig). Allow it to run.

- [ ] **Step 3: Sanity-check the smoke result**

Confirm both arms produced trades and the `ortho` arm's factor panel was rebuilt (look for a new `data/factor_panels/<sig>/` dir). The smoke is directional only — do not draw conclusions yet. Commit the config.

```bash
git add ab_orthogonalize_small.yaml
git commit -m "test(ab): smoke A/B config for symmetric orthogonalization"
```

- [ ] **Step 4: Create the full-market confirm config**

Create `ab_orthogonalize.yaml` — identical to `ab_orthogonalize_small.yaml` but **remove the `stocks_filter` block** so the backtest runs over the full `config.yaml` stock set. Update the header comment to "Full-market confirm A/B".

```bash
git add ab_orthogonalize.yaml
git commit -m "test(ab): full-market confirm A/B config for symmetric orthogonalization"
```

---

## Task 8: Full-market A/B run + record results

**Files:**
- Modify: `docs/ab_validation_results.md`

- [ ] **Step 1: Run the full-market A/B**

Run: `.venv/Scripts/python.exe -m stockpool ab --config ab_orthogonalize.yaml`
Expected: completes (may take 10–40 min — the `ortho` arm rebuilds the factor panel over ~4357 stocks once, then caches). Writes `reports/ab/<date>.html`.

- [ ] **Step 2: Record results**

Append a new section to `docs/ab_validation_results.md` mirroring the P4-2 / P4-3 entries: arms, pool size, Δ Sharpe, Δ total return, per-stock win count, and a PASS/FAIL verdict against the same bar used for P4-3 (positive Δ Sharpe). Quote the actual numbers from the report — do not invent them.

- [ ] **Step 3: Commit**

```bash
git add docs/ab_validation_results.md
git commit -m "docs(ab): record symmetric orthogonalization A/B results"
```

---

## Task 9: Decide default + docs

**Files:**
- Modify (conditional): `src/stockpool/config.py` (only if A/B PASSES)
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Default decision**

If the full-market A/B PASSED (positive Δ Sharpe, consistent with the P4-3 bar): you MAY flip the `config.yaml` default to `symmetric_orthogonalize: true` — but **ask the user first** (this changes production behavior). Keep the `PreprocessConfig` field default `False` regardless (matches the conservative project convention). If the A/B was neutral/negative, leave everything off and note it in the docs.

- [ ] **Step 2: Update CLAUDE.md**

In the `src/stockpool/ml/preprocess.py` module-map row, add a sentence describing the joint orthogonalization step (final step, decorrelates non-fundamental factors per day, fundamental skip). In the `preprocess.{...}` config bullet, add `symmetric_orthogonalize` (default false, opt-in, with the A/B verdict). Add a row to the test table:

```
| `test_ml_preprocess_orthogonalize.py` | 对称正交化:逐日输出 Gram 矩阵正交 / order-independent / 接近原因子 / 退化日 passthrough / fundamental 跳过 / NaN 守护 / 不 mutate |
```

- [ ] **Step 3: Update README.md**

In the preprocess config example, add the `symmetric_orthogonalize: false` key with a one-line comment (per-day Löwdin decorrelation of selected factors, opt-in).

- [ ] **Step 4: Verify docs reference real commands**

Confirm the A/B command in both docs reads `python -m stockpool ab --config ab_orthogonalize.yaml`.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md src/stockpool/config.py
git commit -m "docs: document symmetric orthogonalization preprocess + A/B verdict"
```

---

## Done criteria

- `symmetric_orthogonalize_panel` implemented, wired as the final preprocess step, gated by `PreprocessConfig.symmetric_orthogonalize`.
- New unit tests + cache-sig test + config tests all green; full `pytest tests/ -q` passes.
- Smoke + full-market A/B run; results recorded in `docs/ab_validation_results.md`.
- `CLAUDE.md` + `README.md` updated.
- Default behavior unchanged unless the user approves flipping it on after a PASS.
