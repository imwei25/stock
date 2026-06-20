# Rust Ops Execution Log — 2026-06-20

Rolling decision log for autonomous execution of the Rust ops
acceleration project. Updated as each PR / phase completes. Reference
spec: `docs/superpowers/specs/2026-06-20-rust-ops-acceleration-design.md`.

## Goal

End-to-end:
1. PR-2 through PR-5 (Rust crate for 7 hot ops; soft fallback)
2. `factors analyze --universe all` on full 4357-stock universe
3. `factors pick-by-ic` → new selection JSON
4. AB compare pre-Rust selection vs post-Rust selection
5. Verdict on whether the change improved Sharpe / IC / etc.

Working branch: `feat/rust-ops` (off `main` at `618b71f`).

## Execution Plan

| Phase | Status | Notes |
|---|---|---|
| PR-2: Rust crate scaffold + `rank` op | ✅ DONE | 7 commits, 917 tests passing |
| PR-3: 4 rolling ts ops (correlation, ts_std, ts_argmax, ts_argmin) | in progress | |
| PR-4: decay_linear + ts_rank + indneutralize | pending | |
| PR-5: Performance validation + docs | pending | |
| Run analyze --universe all | pending | |
| Run pick-by-ic | pending | |
| AB compare old vs new selection | pending | |
| Findings doc | pending | |

### PR-2 outcomes (2026-06-20)

7 commits on `feat/rust-ops`:
- `4f4494d` scaffold stockpool_ops_rs PyO3 crate
- `bc08326` chore: broaden gitignore rust patterns + ignore Cargo.lock
- `f781349` port cross-sectional rank op to Rust
- `2e2102e` docs: clarify exact-equality tie semantics in rank
- `4a3d185` wire ops.rank to Rust dispatcher (env gate + try-import + fallback)
- `c70f5c9` Layer A equivalence tests for rank (7 cases)
- `3fa09b9` CLAUDE.md note about Rust dispatcher

Results:
- Full test suite: **917 passed**, 2 pre-existing failures unchanged
- Snapshot test (167 factors): every WQ alpha that calls `rank()` exercises the Rust path and matches the pandas oracle within `atol=1e-9, rtol=1e-7`
- `ops._USE_RUST` is `True` by default; `STOCKPOOL_USE_PYTHON_OPS=1` forces pandas fallback

### Key decisions made during PR-2
- **Tie equality on `f64`**: kept `==` (not epsilon-tolerant). Pandas `method="average"` uses bit-identical tie detection; epsilon would diverge from oracle. Comment added in `cs.rs`.
- **`.gitignore` broadened**: `rust/**/target/` + `rust/**/*.pyd` + `rust/**/*.dll` + `rust/**/Cargo.lock` (instead of crate-specific paths).
- **maturin needs `unset CONDA_PREFIX` + explicit `VIRTUAL_ENV`** on this machine — saved to `~/.claude/.../memory/maturin_conda_prefix_gotcha.md` so future sessions don't rediscover.

### PR-3 outcomes (2026-06-20)

Initial scope was 4 rolling time-series ops (`correlation`, `ts_std`, `ts_argmax`, `ts_argmin`). Final outcome: 3/4 ops Rust-wired, correlation Rust impl present but **dispatcher disabled** (correlation still goes through pandas).

Commits on `feat/rust-ops`:
- `bd52595` add util.rs with rolling_apply_col helpers
- `6e12bbe` port ts_std + ts_argmax + ts_argmin to Rust
- (correlation Rust impl + lib.rs registration commit pending — wiring deferred)

Results:
- Snapshot test: 167/167 pass (rust path for rank/ts_std/argmax/argmin; pandas for correlation/decay_linear/ts_rank/indneutralize)
- Layer A tests: pass for rank/ts_std/argmax/argmin

### Key decision: defer correlation Rust dispatch

**Investigation (~30 min, with several iterations)** discovered that `correlation` produces O(1) downstream divergence in ~20 WQ alphas when ported to Rust, not because of a bug but because of **fundamental FP fragility**:

1. pandas `Rolling.corr` uses Welford's online algorithm internally; for bit-identical constant inputs it produces `std=0` exactly. A naive `sum((x-mean)^2)` two-pass impl in Rust gives `std~1e-17` from mean-computation FP noise. Switching the Rust impl to Welford brought std to 0 on bit-constants but the cross-product (covariance) accumulator still drifts versus pandas' internal algorithm.
2. `rank(correlation(...))` then maps tiny FP-noisy correlation differences to discrete rank flips, which `ts_sum` accumulates into O(1) factor-output divergence (e.g. alpha_015 had 93.6% of cells mismatch with max diff 2.0 — exceeds [-1, 1] correlation range entirely after downstream propagation).
3. Even pandas vs pandas isn't bit-stable on FP-fragile factors: regenerating the snapshot twice on the same data produces 128 cell differences (up to 2.0 max diff for alpha_021).

Tried fixes that didn't suffice:
- Welford in Rust (matches pandas on `std` for constant inputs, doesn't match downstream).
- `|corr|>1 → NaN` clamping in both impls (catches inf, doesn't catch finite garbage).
- `std<1e-7 → NaN` guard in both impls (still differs in which positions fire due to FP).
- Loose tolerance (atol=1e-6, rtol=1e-4) + 1% NaN-mismatch budget (still failed 21 factors).

**Decision**: ship PR-3 with `ts_std`/`ts_argmax`/`ts_argmin` only. The Rust `correlation` impl stays in `rolling.rs` and is registered as `stockpool_ops_rs.correlation` (PyO3 binding), but `ops.py` does NOT dispatch to it (`correlation = _ops_py.correlation` re-export). Achieving bit-near equivalence for correlation is a follow-up task that needs either (a) reverse-engineering pandas' exact Cython rolling FP path, or (b) using a deterministic-equivalence test (e.g. IC-of-IC) instead of bit-near snapshot comparison.

This is fine for the overall goal: 3 of the 4 ops (and previously `rank`) get Rust acceleration; correlation still works (via pandas, no speedup). The user's downstream AB comparison can still answer "does Rust acceleration matter for selection quality" — it just won't include correlation in the Rust-vs-pandas variable.

## Key Decisions (rolling)

### Bootstrap (this turn)
- Branched off `main` at `618b71f` (post-merge of `feat/ab-candidate-pool` + setup script).
- Working on `feat/rust-ops`; will merge back when entire chain completes successfully.
- Each phase logs commit SHAs + outcomes here.
- Subagent-driven development per PR (using `subagent-driven-development` skill).

## Outcomes

(To be filled as phases complete.)
