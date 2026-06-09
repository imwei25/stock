# Symmetric (Löwdin) Orthogonalization — Design

Date: 2026-06-10
Status: Design (pending implementation)

## Goal

Add an opt-in **symmetric orthogonalization** (对称正交 / Löwdin) preprocessing
step that decorrelates the selected factors at each cross-section, while keeping
each orthogonalized factor maximally close to its original. Downstream IC / IR /
equal / LightGBM weighting then runs on the decorrelated panel with **no pipeline
changes**. Validate with an A/B test; ship default-off until proven.

## Why symmetric (not Gram-Schmidt)

Gram-Schmidt orthogonalization is **order-dependent**: the first factor is kept
intact, later factors lose whatever they share with earlier ones, so the result
depends on the (arbitrary) factor ordering and distorts the meaning of late
factors. Symmetric (Löwdin) orthogonalization minimizes the total displacement
`Σ‖f_i^orth − f_i‖²` subject to orthonormality, is **order-independent**, and
keeps every orthogonalized factor as close as possible to its original — so the
factor names (and hence per-factor IC/IR weights) stay interpretable.

## Math (per day `t`)

1. Build the cross-section `F` = (N stocks × K factors) for day `t`, using only
   stocks where **all K factors are non-NaN** ("valid subset", `N_valid` rows).
2. Standardize each column (per-day z-score) → `F_std`, so that
   `M = F_stdᵀ F_std / N_valid` is the K×K **correlation** matrix.
3. Eigendecompose the symmetric PSD matrix `M = U Λ Uᵀ` (`np.linalg.eigh`).
4. Floor eigenvalues: `Λ⁺ = max(Λ, 1e-10)`; form
   `S = U · diag(Λ⁺^(-1/2)) · Uᵀ` (symmetric inverse square root).
5. `F_orth = F_std · S`. Columns of `F_orth` are mutually orthogonal
   (`F_orthᵀ F_orth / N_valid ≈ I`). Write these back to the panel for the valid
   rows; NaN rows stay NaN.

**Look-ahead safety**: every quantity is computed from day `t`'s cross-section
only — no other rows referenced. This is exactly the stateless-per-day property
of the existing `cs_zscore_panel`, so the predict path (which reads the same
cached panel) is consistent with training automatically; no fitted state.

## Architecture

### Placement

A new **joint** step appended to the preprocessing pipeline:

```
winsorize → cs_zscore → industry_neutralize → market_cap_neutralize → symmetric_orthogonalize
```

The existing four steps loop **per factor** (each transform is independent across
factors). Orthogonalization is fundamentally **joint** — it mixes all factors at
once per day. So `apply_preprocess_pipeline` keeps its per-factor loop for the
first four steps, then applies orthogonalization to the *resulting* panel as a
single final pass over the whole `{factor_name: T×N}` dict.

### New function (`src/stockpool/ml/preprocess.py`)

```python
def symmetric_orthogonalize_panel(
    factor_panel: dict[str, pd.DataFrame],
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-day cross-sectional symmetric (Löwdin) orthogonalization.

    Decorrelates the non-fundamental factors jointly per day; fundamental
    factors pass through untouched (orthogonalizing PE/PB against momentum
    muddies the intrinsic valuation signal — same rationale as the neutralize
    steps skipping `fundamental`-tagged factors).
    """
```

Implementation notes:
- Operate on the **non-fundamental** subset of factors only. Build a 3-D view
  (T days × N stocks × K_nf factors) by reindexing all non-fundamental factors
  onto a common (dates, codes) frame. Fundamental factors copied through.
- Per day: extract the all-factors-valid stock subset; if `N_valid < K_nf` or
  the day is empty, **pass that day through unchanged** (debug log, no crash) —
  cannot form a full-rank `M`. Otherwise z-score columns, build `S` via `eigh`
  + eigenvalue floor, apply `F_std · S`, scatter back to valid rows.
- Vectorize across days where practical; a per-day python loop over `eigh` is
  acceptable (K is ~20, T is ~250–500). Days share no state.
- Returns a **new** dict; never mutates input (matches existing pipeline
  contract).

### Pipeline wiring (`apply_preprocess_pipeline`)

- Add `do_ortho = cfg.symmetric_orthogonalize` after the existing `do_mcap`
  block.
- After the per-factor `for name, df` loop builds `out`, if `do_ortho`:
  `out = symmetric_orthogonalize_panel(out, factor_types=factor_types)`.
- The `min_pool_size` guard already short-circuits the whole pipeline when
  `n_codes < min_pool_size`, so orthogonalization inherits that protection
  (it's degenerate on tiny pools — `M` rank-deficient).

### Config (`src/stockpool/config.py`)

Add to `PreprocessConfig`:

```python
symmetric_orthogonalize: bool = False
```

- Default `False` → fully backward compatible; `_is_all_off` returns `True` only
  when this is also `False`, so an all-off cfg still omits the `preprocess` key
  from the factor-panel sig (cache unchanged for existing users).
- `_is_all_off` updated:
  `... and cfg.market_cap_neutralize is False and cfg.symmetric_orthogonalize is False`.

### Cache / sig

No new wiring: `_factor_panel_sig` already dumps the full `PreprocessConfig`
into the sig hash when not all-off, so flipping `symmetric_orthogonalize`
auto-invalidates `factor_panels/<sig>/` and `ml_models/<sig>_*.pkl`. A/B arms
get isolated caches via `effective_cfg.content_hash` as today.

### Predict path

Unchanged. `predict_latest` / `generate_signals` read the cached (now
orthogonalized) factor panel via `slice_stock_factor_matrix` / `_build_x_full`,
so the decorrelated factors flow into prediction identically to training. No
code touches the strategy classes.

## Edge cases

| Case | Behavior |
|---|---|
| Day with `N_valid < K_nf` | pass through unchanged (cannot form full-rank `M`) |
| All-NaN day | pass through (no valid rows) |
| Collinear factors (singular `M`) | eigenvalue floor `1e-10` keeps `S` finite |
| Single non-fundamental factor (K_nf=1) | orthogonalization is a no-op z-score → still safe; effectively returns standardized single column |
| All factors fundamental | no-op (nothing to orthogonalize) |
| `n_codes < min_pool_size` | whole pipeline skipped (existing guard) |
| Input never mutated | new dict returned |

## Testing

New `tests/test_ml_preprocess_orthogonalize.py`:
- **Orthogonality**: synthetic correlated panel → output per-day Gram matrix
  off-diagonals ≈ 0.
- **Order-independence**: permuting input factor columns yields the same
  per-column result (up to the permutation) — distinguishes from Gram-Schmidt.
- **Closeness**: orthogonalized columns correlate strongly (sign-aligned) with
  their originals (Löwdin keeps them maximally close).
- **Degenerate day passthrough**: a day with `N_valid < K` is returned unchanged.
- **Fundamental skip**: a `fundamental`-tagged factor passes through byte-for-byte.
- **NaN handling**: NaN cells stay NaN; valid rows transformed.
- **No mutation**: input dict/frames unchanged after call.
- **`_is_all_off`**: `True` only when the new flag is also `False`.

Extend `tests/test_factor_panel_cache.py`: assert flipping
`symmetric_orthogonalize` produces a different sig (cache invalidation).

Extend `tests/test_config.py`: `symmetric_orthogonalize` parses, defaults
`False`, rejected as top-level extra.

Existing `tests/test_ml_preprocess.py` `_is_all_off` cases updated for the new
field.

## A/B validation

Two-stage (per design decision):

1. **Smoke (small pool)**: `ab_orthogonalize_small.yaml` on a modest pool —
   confirms the module runs end-to-end and gives a first directional signal,
   fast. **Caveat**: the `min_pool_size` guard (default 200) skips the *entire*
   preprocess pipeline — including orthogonalization — when `n_codes <
   min_pool_size`. So the smoke config must either keep the pool ≥ 200 stocks or
   set `preprocess.min_pool_size` low enough that orthogonalization actually runs
   (otherwise the smoke silently tests the no-op path and proves nothing). Also
   need `n_codes > K` (number of factors) per day for a full-rank `M`; with ~20
   factors any ≥200-stock pool is safe.
2. **Confirm (full market)**: `ab_orthogonalize.yaml`, both arms
   `training_universe: all` (~4357 stocks), on top of the **current production
   default** preprocess (`winsorize [0.01, 0.99] + zscore + market_cap_neutralize`):
   - arm `base`: `symmetric_orthogonalize: false`
   - arm `ortho`: `symmetric_orthogonalize: true`

Run `python -m stockpool ab --config ab_orthogonalize.yaml`; record
Δ Sharpe / Δ return / win-count in `docs/ab_validation_results.md` (new section,
mirroring the P4-2 / P4-3 entries). **Default stays `false`** unless the full-
market A/B shows a clear improvement (positive Δ Sharpe, consistent with the
P4-x acceptance bar).

## Docs to update (same change)

- `CLAUDE.md`: PreprocessConfig field list + `ml/preprocess.py` module-map entry
  (mention the joint orthogonalization step + fundamental skip) + test-table row
  for `test_ml_preprocess_orthogonalize.py` + the A/B config in the config
  section.
- `README.md`: preprocess config example gains the `symmetric_orthogonalize`
  key with a one-line explanation.
- `docs/ab_validation_results.md`: results section after the A/B runs.

## Out of scope (YAGNI)

- Gram-Schmidt / sequential orthogonalization (rejected — order-dependent).
- Orthogonalizing only the Lasso-selected survivors inside `TwoStepPipeline`
  (rejected — bigger lift, predict-path complications; orthogonalizing the full
  `factors_file` selection before Lasso is cleaner and Lasso benefits from the
  decorrelation).
- Weighted/regularized Löwdin variants, shrinkage on `M` (can revisit if the
  eigenvalue floor proves insufficient on real data).
