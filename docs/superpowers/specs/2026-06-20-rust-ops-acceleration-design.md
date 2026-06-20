# Rust Backend for Hot Factor Ops — Design

**Status**: Approved (brainstormed 2026-06-20)
**Author/owner**: imwei25
**Estimated effort**: 5-8 working days across 5 PRs

## Context

`python -m stockpool factors analyze --universe all` (4357 stocks × 167
factors × ~500 days) reliably crashes with Windows `ACCESS_VIOLATION`
(rc=3221225477) after processing ~30-90 factors in a single process.
Streaming the per-factor compute (commit on `feat/ab-candidate-pool` branch,
`analyze_factors` no longer accumulates all 167 panels) bought us from ~34
to ~91 factors but did not eliminate the crash. Reducing MKL/OMP threads
to 2 actually regressed the crash to factor 17. A chunked subprocess
driver (`scripts/factors_analyze_chunked.py`, default chunk_size=10) is the
current workaround but still crashes inside chunks of the heavy
`alpha_076-100` region (each crashes within a fresh process at ~18
factors).

Diagnosis: C-level state buildup (most likely pandas internal allocator
fragmentation under heavy `.rolling(...).apply(...)` / `groupby` churn)
is exacerbated by the large panel size and the chained ops in WQ101
alphas. Rust ops with `numpy` zero-copy I/O sidestep the pandas/numpy
allocator path entirely for the hot ops and additionally exploit per-stock
parallelism via `rayon`.

## Goals

1. Make `factors analyze --universe all` complete reliably (no crash) on
   the production 4357-stock universe.
2. Speed up `compute_factor_panel` by 5-10× on the same hardware (target
   total analyze time ~3-5 min, down from ~30 min when it does complete).
3. Preserve **bit-near** numerical equivalence (`atol=1e-9, rtol=1e-7`)
   with the current pandas implementation; the existing pandas code
   becomes the **oracle** that Rust ops must reproduce.
4. Keep pandas implementations available as a documented fallback for
   environments without a Rust toolchain.

## Non-goals

- Distributing the Rust crate via PyPI (single-machine local builds only).
- Rewriting any factor in Rust. Only the 7 hot ops in `factors/ops.py` are
  ported; factor classes in `wq101.py` / `factors/*.py` are untouched.
- Adding CI for the Rust build. Single Windows developer machine; tests
  guarded with `@pytest.mark.skipif(not _RUST_AVAILABLE)`.
- Process-level parallelism for chunks. `rayon` within-op saturates 16
  cores; over-subscribing would risk new Windows thread-pool issues.

## Decisions (all chosen during brainstorm, see questions Q1-Q5)

| # | Decision | Choice |
|---|---|---|
| Q1 | Rewrite scope | **B**: hot 5-7 ops only; remaining 18 stay pandas |
| Q2 | Rust framework | **A**: PyO3 + `ndarray` + `rayon`, custom crate |
| Q3 | Validation strategy | **C**: per-op unit tests + mid-scale snapshot (`atol=1e-9 rtol=1e-7`) |
| Q4 | Parallelism | **A**: rayon within-op only; chunks remain serial |
| Q5 | Fallback policy | **B**: soft fallback, env `STOCKPOOL_USE_PYTHON_OPS=1` forces pandas |

## Architecture

```
stockpool/factors/wq101.py + builtin factors        (UNCHANGED)
        |
        v  imports
stockpool/factors/ops.py  (THIN WRAPPER)
   |
   |-- 18 light ops kept inline as pandas
   |   (delay, delta, ts_sum, ts_mean, ts_min, ts_max, scale,
   |    safe_div, vwap, adv, returns, cs_demean, stddev,
   |    ts_product, signedpower, covariance, _min_periods,
   |    plus aliases)
   |
   |-- 7 hot ops -> wrapper:
   |   def correlation(x, y, d):
   |       if _USE_RUST:
   |           return _rust.correlation(x, y, d)
   |       return _ops_py.correlation(x, y, d)
   |
   v
stockpool/factors/_ops_py.py  (NEW; pandas oracle for the 7 hot ops)
   correlation, ts_rank, decay_linear, ts_std,
   ts_argmin, ts_argmax, rank, indneutralize
   (verbatim copies of current ops.py implementations)
   |
   v  alternative path via try-import
stockpool_ops_rs  (NEW; PyO3 compiled module)
   rust/stockpool_ops/
   |-- Cargo.toml
   |-- pyproject.toml          (maturin entry)
   |-- src/
       |-- lib.rs              (#[pymodule] stockpool_ops_rs)
       |-- rolling.rs          (correlation, ts_std, ts_argmax, ts_argmin)
       |-- decay.rs            (decay_linear)
       |-- ts_rank.rs
       |-- cs.rs               (rank, indneutralize)
       |-- util.rs             (NaN-safe helpers, rolling_map_par<F>)
```

### Wrapper pattern

```python
# ops.py (sketch)
import os
import numpy as np
import pandas as pd
from . import _ops_py

_USE_RUST = False
if os.environ.get("STOCKPOOL_USE_PYTHON_OPS") != "1":
    try:
        import stockpool_ops_rs as _rust
        _USE_RUST = True
    except ImportError:
        _USE_RUST = False


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    if _USE_RUST:
        return _rust.correlation(x, y, d)
    return _ops_py.correlation(x, y, d)
```

### Rust ↔ Python data exchange

- Input: `numpy.ndarray[f64, 2]` view → Rust `ArrayView2<f64>`
  (zero-copy via PyO3 `numpy::PyReadonlyArray2`).
- Output: Rust allocates `Array2<f64>` → numpy via `to_pyarray`
  (zero-copy when ownership transfers).
- Python wrapper enforces `np.ascontiguousarray(df.to_numpy(),
  dtype=np.float64)` before passing in (one defensive copy if the
  DataFrame's backing array is not row-major float64; typical case is
  no-copy).
- DataFrame re-wrap reuses original `index` / `columns` objects (Python-
  side identity preserved).

### Parallelism

- Each Rust op iterates over the N column (stock) axis via `rayon`
  parallel iterator. Default rayon thread pool (one per logical core)
  saturates a 16-core box on typical analyze runs.
- The chunked driver (`scripts/factors_analyze_chunked.py`) keeps
  running chunks **serially**. Each chunk subprocess still uses rayon
  within ops. This avoids the "two thread pools fighting" problem and
  keeps the existing crash-resume logic intact.

## Op Equivalence Contracts

Every op must reproduce `_ops_py` output element-wise within
`atol=1e-9, rtol=1e-7`. The pandas implementations in
`src/stockpool/factors/_ops_py.py` (extracted verbatim from the
current `ops.py` in PR-1) are the contract.

| op | Signature | min_periods | NaN semantics | Notes |
|---|---|---|---|---|
| `correlation(x, y, d)` | (T×N, T×N, int) → T×N | `d` (strict) | window NaN → cell NaN | Pearson; mirrors `x.rolling(d, min_periods=d).corr(y)` |
| `ts_rank(x, d)` | (T×N, int) → T×N | `d` (strict) | window NaN → cell NaN | `rank = (window <= last).sum() / d`; ∈ (0, 1] |
| `decay_linear(x, d)` | (T×N, int) → T×N | `max(1, int(d*0.6))` | NaN positions drop from numerator + denominator, remaining weights renormalize; all-NaN window → NaN | weights `1..d`; partial windows take `weights[-len(a):]` |
| `ts_std(x, d)` | (T×N, int) → T×N | `max(1, int(d*0.6))` | NaN skip; <2 valid values → NaN | `ddof=0` (population) |
| `ts_argmax(x, d)` / `ts_argmin` | (T×N, int) → T×N | `d` (strict) | window NaN → cell NaN | return `d - 1 - first_arg`; 0 = today, d-1 = oldest; tie-break = oldest (numpy returns first index) |
| `rank(x)` | T×N → T×N | n/a | NaN stays NaN, excluded from rank | `axis=1, pct=True, method="average"` |
| `indneutralize(x, group)` | (T×N, dict/Series) → T×N | n/a | NaN excluded from group mean, NaN cell stays NaN | code not in group → solo group (self - self = 0); wrapper converts group dict to `np.int32` sector_id array |

**Rust-side conventions**:
- All input arrays float64; outputs float64.
- NaN detection via `f64::is_nan` — do NOT rely on `total_cmp` (NaN
  ordering differs).
- Iterate columns in parallel via `rayon` (`par_iter_mut` over output
  columns or `ndarray::Zip` rayon variant).
- No global state. Each op call is fully self-contained.

## Project Layout

```
claude/
  pyproject.toml               (unchanged; stockpool main package)
  rust/
    stockpool_ops/             (NEW crate)
      Cargo.toml
      pyproject.toml           (maturin entry)
      src/
        lib.rs
        rolling.rs
        decay.rs
        ts_rank.rs
        cs.rs
        util.rs
  src/stockpool/factors/
    ops.py                     (thin wrapper for 7 hot ops; 18 light ops unchanged)
    _ops_py.py                 (NEW; pandas oracle for the 7 hot ops)
  tests/
    test_ops_rust_equivalence.py   (NEW; per-op layer A tests)
    test_ops_snapshot.py            (NEW; layer B snapshot test)
    fixtures/
      ops_snapshot.parquet          (NEW; ~5-10 MB committed to git)
  scripts/
    gen_ops_snapshot.py             (NEW; regenerates the snapshot)
```

### Cargo.toml essentials

```toml
[package]
name = "stockpool_ops"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]
name = "stockpool_ops_rs"

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py310"] }
numpy = "0.22"
ndarray = { version = "0.16", features = ["rayon"] }
rayon = "1.10"

[profile.release]
lto = true
codegen-units = 1
```

### .gitignore additions

```
rust/stockpool_ops/target/
rust/stockpool_ops/*.so
rust/stockpool_ops/*.pyd
```

### Build flow (single-machine local)

```bash
# one-time
.venv/Scripts/pip install maturin

# every time Rust changes
cd rust/stockpool_ops
../../.venv/Scripts/maturin develop --release
```

Release mode is the default (debug build is ~10× slower; only useful
for panic debugging).

## Tests

### Layer A: Per-op unit tests (`tests/test_ops_rust_equivalence.py`)

- Run only when Rust module loads (`@pytest.mark.skipif(not _RUST_AVAILABLE)`).
- Per op, parametrize over multiple window sizes (`d ∈ {3, 5, 20, 60}`).
- Synthetic input: 50 days × 20 stocks float64, seeded random.
- Required NaN patterns per op:
  - Normal random input
  - 5% scattered NaN
  - One fully-NaN column
  - A 5-day consecutive NaN burst in one column (halted trading simulation)
  - Smallest meaningful `d` (2) and a large `d` (60)
- `indneutralize` adds: code not in group map; full-NaN group; single-
  member group.
- Assertion: `np.testing.assert_allclose(rs.values, py.values,
  atol=1e-9, rtol=1e-7, equal_nan=True)`.
- Wrapper identity check: `rs_out.index is x.index`; column order
  preserved.

Expected ~30 test functions total (7 ops × 4-5 cases avg).

### Layer B: Mid-scale snapshot test (`tests/test_ops_snapshot.py`)

- Fixture: `tests/fixtures/ops_snapshot.parquet` — generated by
  `scripts/gen_ops_snapshot.py`:
  - 100 codes (alphabetical-first 100 from `data/universe.parquet`)
  - Last 250 trading days of union dates
  - All 167 registered factors computed via `STOCKPOOL_USE_PYTHON_OPS=1`
    (pandas-only)
  - Stacked into long form: columns `{factor, date, code, value}`
  - Parquet+snappy → ~5-10 MB; committed to git
- Test: load panel, compute every factor with Rust enabled, diff against
  snapshot at `atol=1e-9, rtol=1e-7` per factor. Any divergence flags
  the factor name in the error message.
- Runtime expectation: ~5-10s (100 stocks is light).
- Skipped when Rust module unavailable.

### When to regenerate the snapshot

The snapshot **is** the contract. Legitimate regeneration triggers:
1. New factor registered (`@_wq` or new file under `factors/`).
2. A pandas oracle implementation's semantics changed (must be called
   out in commit message).

**NOT** legitimate: a Rust implementation diverging. Rust must match
the snapshot; never the other way around.

Regen flow:
```bash
.venv/Scripts/python.exe scripts/gen_ops_snapshot.py
git add tests/fixtures/ops_snapshot.parquet
git commit -m "ops: regenerate snapshot for new factor X"
```

## Rollout Plan (5 PRs)

### PR-1: Pandas oracle extraction + snapshot fixture (2-3h)
- New file `src/stockpool/factors/_ops_py.py` with the 7 hot ops copied
  verbatim from current `ops.py`.
- `ops.py` becomes thin wrapper around `_ops_py` (no Rust path yet).
- The existing test suite must pass unchanged (pure refactor moving
  code between files; public API of `ops.py` unchanged).
- New `scripts/gen_ops_snapshot.py` + committed
  `tests/fixtures/ops_snapshot.parquet`.
- New `tests/test_ops_snapshot.py` — in this PR it self-validates the
  fixture against the pandas path (must pass).
- Risk: 0 (pure refactor + new fixture).

### PR-2: Rust crate scaffold + `rank` as first op (1 day)
- Full `rust/stockpool_ops/` directory + Cargo.toml + lib.rs frame.
- Implement `rank` (cross-sectional, no window — the simplest math).
- `ops.py.rank` becomes the try-import wrapper.
- `tests/test_ops_rust_equivalence.py` with `rank` Layer A cases.
- Snapshot test now passes with Rust path enabled (only `rank` goes
  Rust; rest still pandas).
- README + CLAUDE.md additions: Rust install steps + one-line note in
  factor library section.
- Risk: medium (toolchain + PyO3 ramp-up). This PR locks in the
  build / import / verification flow.

### PR-3: 4 rolling time-series ops (1.5 days)
- Implement `correlation`, `ts_std`, `ts_argmax`, `ts_argmin`. Share
  `util.rs::rolling_map_par<F>` abstraction.
- Each op gets Layer A unit tests.
- Snapshot test must pass.
- Risk: medium (rolling boundaries + min_periods + NaN handling are the
  bug-prone bits; Layer A catches them per-op).

### PR-4: `decay_linear`, `ts_rank`, `indneutralize` (2 days)
- `decay_linear`: NaN-aware weighted sum + partial-window weight tail
  alignment (most subtle semantics, saved for last).
- `ts_rank`: window-internal rank with tie handling — must match
  `(window <= last).sum() / d` exactly.
- `indneutralize`: Python wrapper encodes group dict as `np.int32`
  sector_id array; Rust does groupby mean + broadcast subtract.
- Each op gets Layer A unit tests.
- Snapshot test must pass with all 7 ops on Rust path. All 167 factors
  must round-trip element-equivalent.

### PR-5: Performance validation + documentation (0.5 day)
- Run `python -m stockpool factors analyze --universe all` (still via
  chunked driver) on the production 4357-stock universe.
- Record: total time, peak RSS, crash count.
- Expected: no crash; total time 3-5 min (vs current ~30 min if it
  completes at all).
- README gets a performance note.
- CLAUDE.md "已知不支持的能力" entry referencing the
  `factors analyze --universe all` segfault gets removed (or amended to
  "fixed by Rust ops in commit X").

### Total estimate

5-6 working days nominal, 7-8 with buffer.

## Rollback Strategy

| Scope | Rollback |
|---|---|
| Single op divergence | `STOCKPOOL_USE_PYTHON_OPS=1` env var forces pandas for everything; Rust module still imports but is never called. |
| Rust build broken | Delete `.so` / `.pyd` artifacts; wrapper falls back to `_ops_py.py` automatically. |
| Full revert | Revert PR-2 onward; code returns to PR-1 state (thin wrapper + snapshot fixture, all pandas). |

## Open Risks

1. **PyO3 + numpy version compatibility on Windows**: PyO3 0.22 +
   numpy 0.22 require matching numpy ABI. Pin in Cargo.toml; verify
   `maturin develop --release` succeeds before PR-2 ships.
2. **rayon thread pool + Python GIL**: Releasing the GIL inside Rust
   ops is necessary for true parallelism. Use `py.allow_threads(|| ...)`
   wrapper in every `#[pyfunction]`. If forgotten, rayon serializes.
3. **`decay_linear` partial-window semantics**: pandas `_min_periods(d)`
   relaxation interacts with the `weights[-len(a):]` tail alignment in
   subtle ways. Layer A must cover `d=20`, `len(a)=12` (mid-partial)
   explicitly.
4. **Snapshot drift across pandas versions**: regenerating the snapshot
   on a different pandas may produce tiny FP differences. Pin pandas
   minor version in `pyproject.toml` if drift seen; until then, accept
   `atol=1e-9` headroom.

## Out of scope (for future specs)

- Per-op rayon thread count tuning beyond default
- Distributing `stockpool_ops` as a prebuilt wheel
- Porting the remaining 18 light ops (only do this if profiling shows
  they're a bottleneck after PR-5)
- Replacing `pd.DataFrame` with a thinner array container in the
  factor compute pipeline (would yield more gains but is a separate,
  much larger refactor)
