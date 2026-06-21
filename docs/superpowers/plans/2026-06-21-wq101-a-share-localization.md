# WQ101 A-Share Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify whether window-only localization of WQ101's top-30 alphas materially improves cross-sectional IC and per-stock Sharpe on A-share, by generating three rule-based parameter variants per alpha and gating winners through walk-forward + AB validation.

**Architecture:** Phase 0 fixes the IC diagnostic pipeline (winsorize + degenerate-day rejection) so subsequent IC comparisons are trustworthy. Phase 1 statically scans `factors/wq101.py` for window literals. Phase 2 generates a `factors/wq101_variants.py` module via AST-based source rewriting (no runtime monkey-patching). Phase 3 re-runs `factors analyze` on Round 1 candidates and walk-forward halves. Phase 4 picks winners that pass both halves. Phase 5 AB-tests winners against current `selection.json`. Phase 6 (Round 2 / bottom-30) deferred to follow-up plan.

**Tech Stack:** Python 3.10 / pandas / numpy / pytest / ast (stdlib) / existing `stockpool.factors` registry + `stockpool.factors_analysis` + `stockpool.ml.preprocess.winsorize_panel` + `stockpool.ab` framework.

---

## Spec Reference

`docs/superpowers/specs/2026-06-21-wq101-a-share-localization-design.md`

## File Structure

**Modify:**
- `src/stockpool/factors_analysis.py` — add winsorize + degenerate-day params to `analyze_factors`, add `degenerate_day_ratio` to `FactorAnalysisResult`
- `src/stockpool/factors_analysis_report.py` — surface `degenerate_day_ratio` column in HTML
- `src/stockpool/cli.py` — `cmd_factors_analyze`: new flags `--no-winsorize` / `--winsorize` / `--degenerate-threshold` / `--factors-file`
- `tests/test_factors_analysis.py` — extend with winsorize + degenerate tests
- `CLAUDE.md` — factor library section: add `wq101_localized` source line + new analyze flags

**Create:**
- `scripts/wq101_window_inventory.py` — Phase 1 static AST scanner
- `scripts/generate_wq101_variants.py` — Phase 2 source generator
- `src/stockpool/factors/wq101_variants.py` — Phase 2 generated module (committed)
- `tests/test_wq101_variants.py` — Phase 2 unit tests
- `reports/wq101_window_inventory.csv` — Phase 1 output (committed)
- `scripts/build_round1_factor_list.py` — Phase 3 select top-30 + variants
- `reports/wq101_round1_factors.json` — Phase 3 input (committed)
- `reports/factor_analysis/wq101_round1_h1/` and `wq101_round1_h2/` — Phase 3 walk-forward outputs (gitignore — too large)
- `scripts/pick_wq101_winners.py` — Phase 4 picker
- `reports/wq101_round1_winners.csv` — Phase 4 output (committed)
- `reports/selection_wq101_localized.json` — Phase 4 output (committed)
- `ab/wq101_localized.yaml` — Phase 5 AB config (committed)
- `reports/ab/wq101_localized/` — Phase 5 AB HTML output (gitignore)

---

## Task 1: Add winsorize parameter to `analyze_factors`

**Files:**
- Modify: `src/stockpool/factors_analysis.py:237-354`
- Test: `tests/test_factors_analysis.py`

**Goal:** When `winsorize=(lo, hi)` is passed, apply `ml.preprocess.winsorize_panel` to each factor's wide-frame **before** computing daily IC. Default = `(0.01, 0.99)` (matches training pipeline). `winsorize=None` disables.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_factors_analysis.py`:

```python
def test_analyze_factors_applies_winsorize(monkeypatch):
    """winsorize=(lo,hi) clips factor cross-section before IC."""
    import numpy as np
    import pandas as pd
    from stockpool.factors_analysis import analyze_factors
    from stockpool.factors.registry import _REGISTRY, register
    from stockpool.factors.base import Factor

    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    codes = [f"S{i:03d}" for i in range(20)]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        np.cumprod(1 + rng.normal(0, 0.01, (40, 20)), axis=0),
        index=dates, columns=codes,
    )
    panel = {"open": close, "high": close, "low": close, "close": close,
             "volume": pd.DataFrame(1.0, index=dates, columns=codes)}

    class _SpikeFactor(Factor):
        sources = ("test",); types = ("cross_sectional",)
        description = "factor with one giant outlier per day"
        @property
        def name(self): return "spike_test"
        def compute(self, panel):
            base = panel["close"].rank(axis=1)
            base.iloc[:, 0] = 1e6  # huge outlier in column 0 every day
            return base

    monkeypatch.setitem(_REGISTRY, "spike_test",
        type(_REGISTRY[list(_REGISTRY)[0]])(
            base_name="spike_test", cls=_SpikeFactor,
            sources=("test",), types=("cross_sectional",), description="",
        ))

    r_winsorized = analyze_factors(panel, ["spike_test"], horizon=2,
                                   winsorize=(0.05, 0.95))
    r_raw = analyze_factors(panel, ["spike_test"], horizon=2, winsorize=None)
    # With winsorize on, the outlier column is clipped → IC shape differs from raw.
    assert r_raw.daily_ic["spike_test"].std() > 0
    assert r_winsorized.daily_ic["spike_test"].std() > 0
    assert not r_winsorized.daily_ic["spike_test"].equals(
        r_raw.daily_ic["spike_test"]
    ), "winsorize=(0.05,0.95) must change daily IC vs winsorize=None"


def test_analyze_factors_winsorize_default_is_lenient():
    """Default winsorize=(0.01, 0.99) should NOT change healthy-factor IC by much."""
    import numpy as np
    import pandas as pd
    from stockpool.factors_analysis import analyze_factors

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = [f"S{i:03d}" for i in range(50)]
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        np.cumprod(1 + rng.normal(0, 0.01, (80, 50)), axis=0),
        index=dates, columns=codes,
    )
    panel = {"open": close, "high": close, "low": close, "close": close,
             "volume": pd.DataFrame(1.0, index=dates, columns=codes)}

    r_default = analyze_factors(panel, ["momentum_20"], horizon=3)
    r_none = analyze_factors(panel, ["momentum_20"], horizon=3, winsorize=None)
    diff = (r_default.abs_ic_mean["momentum_20"]
            - r_none.abs_ic_mean["momentum_20"])
    assert abs(diff) < 0.02, f"winsorize=(0.01,0.99) shifted abs_ic by {diff:.4f}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_analysis.py::test_analyze_factors_applies_winsorize tests/test_factors_analysis.py::test_analyze_factors_winsorize_default_is_lenient -v`
Expected: FAIL with `TypeError: analyze_factors() got an unexpected keyword argument 'winsorize'`

- [ ] **Step 3: Add the winsorize parameter to `analyze_factors`**

Edit `src/stockpool/factors_analysis.py` — modify the `analyze_factors` signature and the per-factor loop body:

```python
def analyze_factors(
    panel: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str],
    horizon: int = 3,
    ic_window: int = 252,
    regime_index_close: pd.Series | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
    winsorize: tuple[float, float] | None = (0.01, 0.99),
) -> FactorAnalysisResult:
    """End-to-end factor analysis on a panel.

    Args:
        panel:       OHLCV wide-frame panel.
        factor_names: registered factor names.
        horizon:     forward-return horizon (bars).
        ic_window:   reserved for future rolling-IC variants.
        regime_index_close: optional pd.Series for bull/bear/sideways split.
        method:      "spearman" (default) or "pearson".
        winsorize:   ``(lo, hi)`` per-day quantile clip applied to each
                     factor wide-frame before IC; pass ``None`` to disable.
                     Default ``(0.01, 0.99)`` matches the ML training
                     pipeline so IC numbers are comparable.
    """
```

Then inside the per-factor loop (around the existing `fp_one = f.compute(panel)` line), insert the clip:

```python
    if winsorize is not None:
        from stockpool.ml.preprocess import winsorize_panel
        wlo, whi = winsorize
    for name in factor_iter:
        if hasattr(factor_iter, "set_postfix_str"):
            factor_iter.set_postfix_str(name)
        f = make_factor(name)
        fp_one = f.compute(panel)
        if winsorize is not None:
            fp_one = winsorize_panel(fp_one, wlo, whi)
        daily_ic[name] = compute_daily_ic(fp_one, fwd, method=method)
        del fp_one
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_analysis.py::test_analyze_factors_applies_winsorize tests/test_factors_analysis.py::test_analyze_factors_winsorize_default_is_lenient -v`
Expected: PASS

- [ ] **Step 5: Run full test_factors_analysis suite to verify no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_analysis.py -v`
Expected: all PASS (new default `winsorize=(0.01,0.99)` shouldn't break existing tests since they use synthetic non-pathological inputs)

If any existing tests fail because the IC numbers shift, examine them. If the shift is within ±0.02 abs_ic, update the test's tolerance. If larger, the test was relying on an unwinsorized edge case — discuss with user before changing semantics.

- [ ] **Step 6: Commit**

```bash
cd C:/Users/Administrator/Desktop/claude
git add src/stockpool/factors_analysis.py tests/test_factors_analysis.py
git commit -m "$(cat <<'EOF'
feat(factors_analysis): wire winsorize into analyze_factors

Add winsorize=(lo,hi) kwarg (default (0.01,0.99)) that applies per-day
cross-sectional clip to each factor wide-frame before IC computation,
matching the ML training pipeline. Reuses ml.preprocess.winsorize_panel.

Without this, IC diagnostics ran on raw factor values while ML training
saw winsorized values, making the analyze-side numbers a poor proxy for
training-time signal. Pass winsorize=None to recover raw behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add degenerate-day detection

**Files:**
- Modify: `src/stockpool/factors_analysis.py` (analyze_factors body + FactorAnalysisResult dataclass + to_dict/from_dict)
- Test: `tests/test_factors_analysis.py`

**Goal:** Detect days where the factor cross-section has fewer than `degenerate_day_unique_ratio_threshold` unique values per non-NaN stock. On those days, the rank-IC is dominated by tie-breaking noise and must be marked NaN. Record the fraction of such days per factor as `degenerate_day_ratio`. Default threshold = `0.01` (less than 1% of stocks have unique ranks).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_factors_analysis.py`:

```python
def test_analyze_factors_marks_degenerate_days_nan():
    """Days where factor cross-section is near-constant produce NaN IC."""
    import numpy as np
    import pandas as pd
    from stockpool.factors_analysis import analyze_factors
    from stockpool.factors.registry import _REGISTRY, FactorSpec
    from stockpool.factors.base import Factor

    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    codes = [f"S{i:03d}" for i in range(100)]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        np.cumprod(1 + rng.normal(0, 0.01, (40, 100)), axis=0),
        index=dates, columns=codes,
    )
    panel = {"open": close, "high": close, "low": close, "close": close,
             "volume": pd.DataFrame(1.0, index=dates, columns=codes)}

    class _DegenerateFactor(Factor):
        sources = ("test",); types = ("cross_sectional",)
        description = "factor that is constant on first 20 days, varied after"
        @property
        def name(self): return "degenerate_test"
        def compute(self, panel):
            out = pd.DataFrame(7.0, index=panel["close"].index,
                               columns=panel["close"].columns)
            # vary the last 20 days only
            for i in range(20, len(out)):
                out.iloc[i] = panel["close"].iloc[i].rank()
            return out

    _REGISTRY["degenerate_test"] = FactorSpec(
        base_name="degenerate_test", cls=_DegenerateFactor,
        sources=("test",), types=("cross_sectional",), description="",
    )

    r = analyze_factors(
        panel, ["degenerate_test"], horizon=2,
        winsorize=None,
        degenerate_day_unique_ratio_threshold=0.01,
    )
    ic = r.daily_ic["degenerate_test"]
    # first 20 days have nunique=1 → flagged → NaN
    assert ic.iloc[:20].isna().all(), "constant days must have NaN IC"
    # at least some of the last 20 should be non-NaN
    assert ic.iloc[20:].notna().sum() > 5, "varied days should produce IC"
    # ratio recorded on the result
    assert hasattr(r, "degenerate_day_ratio")
    ratio = r.degenerate_day_ratio["degenerate_test"]
    assert 0.4 < ratio <= 0.55, f"expected ~50% degenerate, got {ratio}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_analysis.py::test_analyze_factors_marks_degenerate_days_nan -v`
Expected: FAIL with `TypeError: analyze_factors() got an unexpected keyword argument 'degenerate_day_unique_ratio_threshold'` (or `AttributeError: ... degenerate_day_ratio`).

- [ ] **Step 3: Add the parameter + recording**

Edit `analyze_factors` signature:

```python
def analyze_factors(
    panel: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str],
    horizon: int = 3,
    ic_window: int = 252,
    regime_index_close: pd.Series | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
    winsorize: tuple[float, float] | None = (0.01, 0.99),
    degenerate_day_unique_ratio_threshold: float = 0.01,
) -> FactorAnalysisResult:
```

After computing `daily_ic[name]` in the per-factor loop, compute the degenerate-day mask **on the post-winsorize panel** and overwrite IC + record ratio:

```python
    degenerate_day_ratio_d: dict[str, float] = {}
    ...
    for name in factor_iter:
        ...
        fp_one = f.compute(panel)
        if winsorize is not None:
            fp_one = winsorize_panel(fp_one, wlo, whi)
        daily_ic[name] = compute_daily_ic(fp_one, fwd, method=method)
        # Detect degenerate (near-constant) days.
        n_valid = fp_one.notna().sum(axis=1)
        # Use nunique conditional on the row having any non-NaN values.
        unique_counts = fp_one.apply(lambda row: row.dropna().nunique(), axis=1)
        # threshold = (unique values) / (valid stocks); mark NaN where ratio < threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_per_day = unique_counts / n_valid.replace(0, np.nan)
        degenerate = (ratio_per_day < degenerate_day_unique_ratio_threshold)
        if degenerate.any():
            daily_ic[name].loc[degenerate[degenerate].index] = float("nan")
        # ratio = fraction of (post-warmup) days that were degenerate
        valid_days = daily_ic[name].notna() | degenerate
        denom = int(valid_days.sum())
        degenerate_day_ratio_d[name] = (
            float(degenerate.sum()) / denom if denom > 0 else float("nan")
        )
        del fp_one
```

Add `degenerate_day_ratio` to `FactorAnalysisResult`:

```python
@dataclass
class FactorAnalysisResult:
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
    degenerate_day_ratio: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
```

In `to_dict`, add:

```python
    "degenerate_day_ratio": _series_to_json_dict(self.degenerate_day_ratio),
```

In `from_dict`, add:

```python
    degenerate_day_ratio=_series_from_json_dict(d.get("degenerate_day_ratio", {})),
```

In the construction at the end of `analyze_factors`:

```python
    degenerate_day_ratio = pd.Series(
        degenerate_day_ratio_d, name="degenerate_day_ratio",
    )
    return FactorAnalysisResult(
        ...,
        degenerate_day_ratio=degenerate_day_ratio,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_analysis.py::test_analyze_factors_marks_degenerate_days_nan -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_analysis.py -v`
Expected: all PASS

- [ ] **Step 6: Surface degenerate ratio in HTML report**

Edit `src/stockpool/factors_analysis_report.py`. Find where the per-factor metrics table is built (search for `abs_ic_mean` rendering). Add a new column showing `degenerate_day_ratio` formatted as percentage. Rows with ratio > 0.20 get a CSS class `warn` (red text).

If the report module already uses pandas `.style` or pyecharts table builder, follow the existing pattern. Show the actual change once you've read the file; keep this step under 30 lines of edit.

- [ ] **Step 7: Commit**

```bash
git add src/stockpool/factors_analysis.py src/stockpool/factors_analysis_report.py tests/test_factors_analysis.py
git commit -m "$(cat <<'EOF'
feat(factors_analysis): detect & NaN-out cross-sectionally degenerate days

Add degenerate_day_unique_ratio_threshold kwarg (default 0.01) that marks
factor-days with nunique/n_valid < threshold as NaN before IC aggregation.
Records degenerate_day_ratio on the result and surfaces it in the HTML.

Motivated by alpha_096 (abs_ic_mean=0.4773 in 2026-06-20 analysis): its
ts_argmax output is a 0-12 discrete integer, producing constant
cross-sections that drive Spearman rank IC to spurious ±1.0. Without
this guard the IC ranking is meaningless on degenerate factors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Plumb new params + `--factors-file` through CLI

**Files:**
- Modify: `src/stockpool/cli.py` (`cmd_factors_analyze` + its argparse setup around line 1319)
- Test: `tests/test_cli_factors_analyze.py`

**Goal:** Expose four new CLI flags so the rest of the plan can drive analyze from the shell:
- `--winsorize-low FLOAT` / `--winsorize-high FLOAT` — pair, both required if either given
- `--no-winsorize` — disable
- `--degenerate-threshold FLOAT` — pass through
- `--factors-file PATH` — read factor names from JSON (same format as picker output: `{"factors": [...]}`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_factors_analyze.py`:

```python
def test_cli_factors_analyze_reads_factors_file(tmp_path, monkeypatch, _stub_cache):
    """--factors-file restricts the analysis to the listed factors."""
    import json
    import sys
    from stockpool.cli import main

    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"factors": ["momentum_20", "rsi_14"]}))

    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "stockpool", "factors", "analyze",
        "--config", str(_stub_cache["config"]),
        "--factors-file", str(sel),
        "--output", str(out_dir),
        "--no-winsorize",
    ])
    rc = main()
    assert rc == 0
    j = json.loads((out_dir / sorted(out_dir.glob("*.json"))[-1].name).read_text(
        encoding="utf-8"
    ))
    assert set(j["factor_names"]) == {"momentum_20", "rsi_14"}


def test_cli_factors_analyze_no_winsorize_disables(tmp_path, monkeypatch, _stub_cache):
    """--no-winsorize passes winsorize=None to analyze_factors."""
    import sys
    from unittest.mock import patch
    from stockpool.cli import main

    captured = {}
    real_analyze = None
    def _spy(panel, factor_names, **kw):
        nonlocal real_analyze
        captured.update(kw)
        from stockpool.factors_analysis import analyze_factors as _r
        return _r(panel, factor_names, **kw)

    with patch("stockpool.cli.analyze_factors", side_effect=_spy):
        monkeypatch.setattr(sys, "argv", [
            "stockpool", "factors", "analyze",
            "--config", str(_stub_cache["config"]),
            "--factors", "momentum_20",
            "--output", str(tmp_path / "out"),
            "--no-winsorize",
        ])
        main()

    assert captured.get("winsorize") is None
```

(`_stub_cache` is a fixture you may need to add or borrow from `test_cli_factors_analyze.py` if it already exists. Check the existing test file first; reuse its synthetic-cache pattern.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_factors_analyze.py -v -k "factors_file or no_winsorize"`
Expected: FAIL — flags don't exist yet.

- [ ] **Step 3: Add the CLI flags**

In `src/stockpool/cli.py`, find the `factors analyze` argparse subparser block (search for `cmd_factors_analyze` then scroll to its `add_parser`/argument setup around line 1319). Add these arguments:

```python
    p_analyze.add_argument(
        "--factors-file", type=str, default=None,
        help="JSON file with {\"factors\": [...]} listing factor names "
             "(overrides --factors if both given). Matches `factors pick` output.",
    )
    p_analyze.add_argument(
        "--winsorize-low", type=float, default=0.01,
        help="Lower quantile for per-day cross-sec winsorize (default 0.01).",
    )
    p_analyze.add_argument(
        "--winsorize-high", type=float, default=0.99,
        help="Upper quantile for per-day cross-sec winsorize (default 0.99).",
    )
    p_analyze.add_argument(
        "--no-winsorize", action="store_true",
        help="Disable winsorize on the factor panel before IC.",
    )
    p_analyze.add_argument(
        "--degenerate-threshold", type=float, default=0.01,
        help="Mark factor-day NaN if nunique/n_valid < this (default 0.01).",
    )
```

In `cmd_factors_analyze` body, before calling `analyze_factors`:

```python
    # --factors-file overrides --factors when both present.
    if args.factors_file:
        import json
        with open(args.factors_file, "r", encoding="utf-8") as fh:
            factor_names = list(json.load(fh)["factors"])
    else:
        factor_names = list(args.factors) if args.factors else list_factors()

    winsorize_arg = None if args.no_winsorize else (args.winsorize_low, args.winsorize_high)
```

Then in the `analyze_factors(...)` call, pass through:

```python
    result = analyze_factors(
        panel=panel,
        factor_names=factor_names,
        horizon=args.horizon,
        ic_window=args.ic_window,
        regime_index_close=regime_close,
        winsorize=winsorize_arg,
        degenerate_day_unique_ratio_threshold=args.degenerate_threshold,
    )
```

Make sure `analyze_factors` is imported at the top of `cli.py` (or, if it's only imported inside `cmd_factors_analyze`, the spy in the test needs to patch the local-import name — check and adjust).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_factors_analyze.py -v -k "factors_file or no_winsorize"`
Expected: PASS

- [ ] **Step 5: Run full CLI test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_factors_analyze.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/cli.py tests/test_cli_factors_analyze.py
git commit -m "$(cat <<'EOF'
feat(cli): factors analyze gains --factors-file + winsorize/degenerate flags

Round out Phase 0 of the WQ101 localization plan by exposing the new
analyze_factors knobs at the shell. --factors-file accepts the
factors pick output JSON unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: WQ101 window inventory (Phase 1)

**Files:**
- Create: `scripts/wq101_window_inventory.py`
- Create: `reports/wq101_window_inventory.csv` (committed)
- Test: `tests/test_wq101_inventory.py`

**Goal:** Static AST scan of `src/stockpool/factors/wq101.py`: for each `Alpha*.compute` method, find every call to a window-bearing op and record the integer literal in its window position. Output a CSV with columns `alpha_id, op, window, count_in_alpha, category, transformable`.

`transformable=False` for alphas whose window args are non-literal (variables, method calls, etc.) — those cannot be auto-rewritten by Task 5.

**Whitelisted window-bearing ops** (read from `src/stockpool/factors/ops.py` to confirm signatures; this list reflects the design):

| op | window arg position |
|---|---|
| `ts_sum`, `ts_mean`, `ts_min`, `ts_max`, `ts_argmin`, `ts_argmax`, `ts_rank`, `ts_std`, `ts_product` | 2nd (index 1) |
| `delta`, `delay`, `decay_linear` | 2nd (index 1) |
| `correlation`, `covariance` | 3rd (index 2) |
| `_adv` (helper in wq101.py) | 2nd (index 1) |

- [ ] **Step 1: Write the failing test**

Create `tests/test_wq101_inventory.py`:

```python
def test_inventory_extracts_alpha002_windows(tmp_path):
    """alpha_002 has correlation(., ., 6) → inventory should report window=6."""
    import subprocess, sys, csv
    from pathlib import Path
    out_csv = tmp_path / "inv.csv"
    rc = subprocess.run(
        [sys.executable, "scripts/wq101_window_inventory.py",
         "--output", str(out_csv)],
        cwd=Path.cwd(),
    ).returncode
    assert rc == 0
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    a002 = [r for r in rows if r["alpha_id"] == "alpha_002"]
    # alpha_002: ops.delta(log_v, 2) + ops.correlation(a, b, 6)
    windows = sorted(int(r["window"]) for r in a002)
    assert windows == [2, 6], f"alpha_002 windows should be [2,6], got {windows}"
    # check op names roughly
    ops_seen = {r["op"] for r in a002}
    assert "correlation" in ops_seen
    assert "delta" in ops_seen


def test_inventory_classifies_categories(tmp_path):
    import subprocess, sys, csv
    from pathlib import Path
    out_csv = tmp_path / "inv.csv"
    subprocess.run(
        [sys.executable, "scripts/wq101_window_inventory.py",
         "--output", str(out_csv)],
        cwd=Path.cwd(), check=True,
    )
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    for r in rows:
        w = int(r["window"])
        cat = r["category"]
        if w <= 10: assert cat == "short", f"w={w} should be short"
        elif w <= 30: assert cat == "medium", f"w={w} should be medium"
        elif w >= 60: assert cat == "long", f"w={w} should be long"
        else: assert cat == "other", f"w={w} should be other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_wq101_inventory.py -v`
Expected: FAIL with `FileNotFoundError: scripts/wq101_window_inventory.py`

- [ ] **Step 3: Write the inventory script**

Create `scripts/wq101_window_inventory.py`:

```python
"""Static AST scan of factors/wq101.py for window literals in window-bearing ops.

Output CSV columns: alpha_id, op, window, count_in_alpha, category, transformable.

`transformable=False` flags alphas with non-literal window args; those cannot
be auto-rewritten by generate_wq101_variants.py.
"""
from __future__ import annotations
import argparse
import ast
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WQ101_SRC = REPO_ROOT / "src" / "stockpool" / "factors" / "wq101.py"

# op_name -> 0-indexed position of window arg
WINDOW_OPS = {
    "ts_sum": 1, "ts_mean": 1, "ts_min": 1, "ts_max": 1,
    "ts_argmin": 1, "ts_argmax": 1, "ts_rank": 1,
    "ts_std": 1, "ts_product": 1,
    "delta": 1, "delay": 1, "decay_linear": 1,
    "correlation": 2, "covariance": 2,
    "_adv": 1,
}


def _categorize(w: int) -> str:
    if w <= 10: return "short"
    if w <= 30: return "medium"
    if w >= 60: return "long"
    return "other"


def _is_alpha_class(node: ast.ClassDef) -> str | None:
    """Return alpha_NNN if node decorated with @_wq(N, ...), else None."""
    for dec in node.decorator_list:
        if (isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "_wq"):
            if dec.args and isinstance(dec.args[0], ast.Constant):
                num = dec.args[0].value
                if isinstance(num, int):
                    return f"alpha_{num:03d}"
    return None


def _scan_compute_body(compute_fn: ast.FunctionDef):
    """Yield (op_name, window_value_or_None, is_literal) for each whitelisted call."""
    for sub in ast.walk(compute_fn):
        if not isinstance(sub, ast.Call):
            continue
        # match ops.ts_sum(...) or _adv(...)
        op_name = None
        if isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name):
            if sub.func.value.id == "ops":
                op_name = sub.func.attr
        elif isinstance(sub.func, ast.Name):
            if sub.func.id == "_adv":
                op_name = "_adv"
        if op_name not in WINDOW_OPS:
            continue
        pos = WINDOW_OPS[op_name]
        if len(sub.args) <= pos:
            continue
        arg = sub.args[pos]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            yield op_name, arg.value, True
        else:
            yield op_name, None, False


def scan_file(path: Path) -> list[dict]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    rows: list[dict] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        alpha_id = _is_alpha_class(node)
        if alpha_id is None:
            continue
        # find compute method
        compute = next(
            (n for n in node.body
             if isinstance(n, ast.FunctionDef) and n.name == "compute"),
            None,
        )
        if compute is None:
            continue
        # collect (op, window) pairs
        literal_pairs: list[tuple[str, int]] = []
        has_non_literal = False
        for op_name, w, is_lit in _scan_compute_body(compute):
            if is_lit:
                literal_pairs.append((op_name, w))
            else:
                has_non_literal = True
        # de-duplicate + count
        counts = Counter(literal_pairs)
        for (op_name, w), cnt in sorted(counts.items()):
            rows.append({
                "alpha_id": alpha_id,
                "op": op_name,
                "window": w,
                "count_in_alpha": cnt,
                "category": _categorize(w),
                "transformable": not has_non_literal,
            })
        # if no literals at all, still emit one row with window blank
        if not counts:
            rows.append({
                "alpha_id": alpha_id, "op": "", "window": "",
                "count_in_alpha": 0, "category": "",
                "transformable": False,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/wq101_window_inventory.csv")
    args = ap.parse_args()
    rows = scan_file(WQ101_SRC)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "alpha_id", "op", "window", "count_in_alpha",
            "category", "transformable",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_wq101_inventory.py -v`
Expected: PASS

- [ ] **Step 5: Generate the committed inventory CSV**

Run: `.venv/Scripts/python.exe scripts/wq101_window_inventory.py`

Inspect briefly: `head reports/wq101_window_inventory.csv`. Confirm ≥ 200 rows. If `transformable=False` covers more than 20 of the 101 alphas, note this in the commit message — Task 5 will need to skip them.

- [ ] **Step 6: Commit**

```bash
git add scripts/wq101_window_inventory.py reports/wq101_window_inventory.csv tests/test_wq101_inventory.py
git commit -m "$(cat <<'EOF'
feat(scripts): WQ101 window inventory (Phase 1 of A-share localization)

Static AST scan of factors/wq101.py listing all integer-literal window
args to window-bearing ops (ts_*, delta, delay, decay_linear,
correlation, covariance, _adv). Flags alphas whose windows are
expressions rather than literals as non-transformable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Variant generator (Phase 2)

**Files:**
- Create: `scripts/generate_wq101_variants.py`
- Create: `src/stockpool/factors/wq101_variants.py` (generated, committed)
- Test: `tests/test_wq101_variants.py`

**Goal:** Read top-30 wq101 alphas from a baseline IC JSON, AST-rewrite each one's `compute` method body with three window transformation rules, emit a generated source file. The generated file imports nothing exotic and registers each variant via the existing `register` decorator from `stockpool.factors.registry`.

**Transformation rules** (operate on literal integer windows only):

| rule | transform | bounds |
|---|---|---|
| `_compress` | `N → max(2, ceil(N * 0.5))` | applied to all literals |
| `_rev_short` | `N ≤ 10 → max(2, ceil(N * 0.5))`; else `N` | preserves medium/long |
| `_expand_long` | `N ≥ 60 → ceil(N * 1.5)`; else `N` | preserves short/medium |

If two rules produce the same windows for a given alpha (e.g. an alpha has only short windows → `_compress` == `_rev_short`), still emit both classes — the picker in Task 8 will deduplicate.

**Baseline source for top-30 selection:** the JSON output produced by Phase 0's re-run of `factors analyze`. **Until that re-run exists** the generator falls back to the pre-Phase-0 `reports/factor_analysis/2026-06-20.json` and prints a warning. Task 7 re-runs the generator with the new baseline.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_wq101_variants.py`:

```python
def test_generator_rewrites_correlation_window(tmp_path):
    """alpha_002's correlation(., ., 6) → alpha_002_compress with window 3."""
    import subprocess, sys, json, importlib
    from pathlib import Path
    baseline = tmp_path / "baseline.json"
    # synthetic baseline ranking alpha_002 / alpha_003 / alpha_004 as top-3
    baseline.write_text(json.dumps({
        "factor_names": ["alpha_002", "alpha_003", "alpha_004"],
        "abs_ic_mean": {"alpha_002": 0.10, "alpha_003": 0.09, "alpha_004": 0.08},
    }))
    out_py = tmp_path / "wq101_variants_test.py"
    rc = subprocess.run(
        [sys.executable, "scripts/generate_wq101_variants.py",
         "--baseline", str(baseline),
         "--top-n", "3",
         "--output", str(out_py)],
        check=True,
    ).returncode
    assert rc == 0
    src = out_py.read_text(encoding="utf-8")
    # alpha_002_compress should rewrite correlation(., ., 6) -> correlation(., ., 3)
    assert "class Alpha002_compress" in src or "Alpha002Compress" in src
    # Direct literal check: the substring "correlation(a, b, 3)" must appear
    # (whitespace agnostic via normalization)
    norm = " ".join(src.split())
    assert "correlation(a, b, 3)" in norm or "correlation( a, b, 3 )" in norm


def test_generated_module_registers_and_computes(tmp_path):
    """After running generator and importing, alpha_002_compress is registered."""
    import subprocess, sys, json, importlib, shutil
    from pathlib import Path
    # Use the real output path so import via canonical module name works.
    out_py = Path("src/stockpool/factors/wq101_variants.py")
    backup = None
    if out_py.exists():
        backup = out_py.read_text(encoding="utf-8")
    try:
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({
            "factor_names": ["alpha_002"],
            "abs_ic_mean": {"alpha_002": 0.10},
        }))
        subprocess.run(
            [sys.executable, "scripts/generate_wq101_variants.py",
             "--baseline", str(baseline), "--top-n", "1",
             "--output", str(out_py)],
            check=True,
        )
        # Drop any cached module + clear the variant entries from registry.
        from stockpool.factors.registry import _REGISTRY
        for k in list(_REGISTRY):
            if k.endswith("_compress") or k.endswith("_rev_short") or k.endswith("_expand_long"):
                del _REGISTRY[k]
        # Re-import
        if "stockpool.factors.wq101_variants" in sys.modules:
            del sys.modules["stockpool.factors.wq101_variants"]
        importlib.import_module("stockpool.factors.wq101_variants")
        assert "alpha_002_compress" in _REGISTRY
    finally:
        if backup is not None:
            out_py.write_text(backup, encoding="utf-8")
        elif out_py.exists():
            out_py.unlink()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_wq101_variants.py -v`
Expected: FAIL — generator script + variants module don't exist.

- [ ] **Step 3: Write the generator script**

Create `scripts/generate_wq101_variants.py`:

```python
"""Generate factors/wq101_variants.py: top-N WQ101 alphas × 3 window rules.

Reads a baseline factor_analysis JSON (must contain abs_ic_mean dict), picks
the top-N wq101 names, AST-rewrites each alpha's compute method body with
three rules (_compress / _rev_short / _expand_long), and emits a Python file
with the generated classes already decorated for registration.
"""
from __future__ import annotations
import argparse
import ast
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WQ101_SRC = REPO_ROOT / "src" / "stockpool" / "factors" / "wq101.py"

WINDOW_OPS = {
    "ts_sum": 1, "ts_mean": 1, "ts_min": 1, "ts_max": 1,
    "ts_argmin": 1, "ts_argmax": 1, "ts_rank": 1,
    "ts_std": 1, "ts_product": 1,
    "delta": 1, "delay": 1, "decay_linear": 1,
    "correlation": 2, "covariance": 2,
    "_adv": 1,
}


def _transform(w: int, rule: str) -> int:
    if rule == "compress":
        return max(2, math.ceil(w * 0.5))
    if rule == "rev_short":
        return max(2, math.ceil(w * 0.5)) if w <= 10 else w
    if rule == "expand_long":
        return math.ceil(w * 1.5) if w >= 60 else w
    raise ValueError(rule)


class _WindowRewriter(ast.NodeTransformer):
    def __init__(self, rule: str):
        self.rule = rule

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        op_name = None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "ops":
                op_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            if node.func.id == "_adv":
                op_name = "_adv"
        if op_name not in WINDOW_OPS:
            return node
        pos = WINDOW_OPS[op_name]
        if len(node.args) <= pos:
            return node
        arg = node.args[pos]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            new_w = _transform(arg.value, self.rule)
            node.args[pos] = ast.Constant(value=new_w)
        return node


def _alpha_num(node: ast.ClassDef) -> int | None:
    for dec in node.decorator_list:
        if (isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "_wq"
                and dec.args
                and isinstance(dec.args[0], ast.Constant)):
            return int(dec.args[0].value)
    return None


def _is_transformable(compute: ast.FunctionDef) -> bool:
    """No non-literal window args anywhere in window-bearing op calls."""
    for sub in ast.walk(compute):
        if not isinstance(sub, ast.Call):
            continue
        op_name = None
        if isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name):
            if sub.func.value.id == "ops":
                op_name = sub.func.attr
        elif isinstance(sub.func, ast.Name):
            if sub.func.id == "_adv":
                op_name = "_adv"
        if op_name not in WINDOW_OPS:
            continue
        pos = WINDOW_OPS[op_name]
        if len(sub.args) <= pos:
            continue
        if not (isinstance(sub.args[pos], ast.Constant)
                and isinstance(sub.args[pos].value, int)):
            return False
    return True


def _pick_top_n_wq101(baseline_path: Path, top_n: int) -> list[int]:
    """Return list of alpha numbers (1..101) ordered by abs_ic_mean desc."""
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    abs_ic = data["abs_ic_mean"]
    pairs = []
    for name, v in abs_ic.items():
        if not name.startswith("alpha_"):
            continue
        try:
            n = int(name.split("_")[1])
        except (ValueError, IndexError):
            continue
        if v is None or v != v:  # NaN
            continue
        pairs.append((float(v), n))
    pairs.sort(reverse=True)
    return [n for _, n in pairs[:top_n]]


def generate(baseline: Path, top_n: int, output: Path) -> None:
    src = WQ101_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Map: alpha_num -> ast.ClassDef
    classes: dict[int, ast.ClassDef] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            num = _alpha_num(node)
            if num is not None:
                classes[num] = node

    top_nums = _pick_top_n_wq101(baseline, top_n)
    skipped: list[int] = []
    emitted: list[tuple[int, str]] = []  # (num, rule)

    header = [
        '"""Auto-generated WQ101 variants for A-share localization (Round 1).',
        "",
        "Do not edit by hand; regenerate via:",
        "    python scripts/generate_wq101_variants.py \\",
        "        --baseline reports/factor_analysis/<NEW>.json --top-n 30",
        '"""',
        "from __future__ import annotations",
        "",
        "import numpy as np  # noqa: F401",
        "import pandas as pd  # noqa: F401",
        "",
        "from stockpool.factors import ops  # noqa: F401",
        "from stockpool.factors.base import Factor",
        "from stockpool.factors.registry import register",
        "from stockpool.factors.wq101 import (",
        "    WqAlpha, _ret, _vwap, _adv, _nan_like, _indneutralize,",
        ")",
        "",
    ]
    body: list[str] = []

    for num in top_nums:
        cls_node = classes.get(num)
        if cls_node is None:
            skipped.append(num)
            continue
        compute = next(
            (n for n in cls_node.body
             if isinstance(n, ast.FunctionDef) and n.name == "compute"),
            None,
        )
        if compute is None or not _is_transformable(compute):
            skipped.append(num)
            continue
        for rule in ("compress", "rev_short", "expand_long"):
            new_compute = _WindowRewriter(rule).visit(
                ast.parse(ast.unparse(compute))
            ).body[0]
            ast.fix_missing_locations(new_compute)
            variant_cls_name = f"Alpha{num:03d}_{rule}"
            variant_factor_name = f"alpha_{num:03d}_{rule}"
            description = (
                f"WQ101 alpha_{num:03d} with rule={rule} applied to its window "
                "literals (A-share localization, Round 1)."
            )
            body.append(
                f'@register("{variant_factor_name}",\n'
                f'          sources=("wq101", "wq101_localized"),\n'
                f'          types=("cross_sectional",),\n'
                f'          description={description!r})\n'
                f"class {variant_cls_name}(WqAlpha):\n"
                f"    NUM = {num}\n"
                f"    @property\n"
                f"    def name(self):\n"
                f'        return "{variant_factor_name}"\n'
                f"{ast.unparse(new_compute)}\n"
            )
            emitted.append((num, rule))

    out = output
    out.parent.mkdir(parents=True, exist_ok=True)
    # Indent the unparsed compute under the class body.
    rendered_body = "\n\n".join(body)
    # ast.unparse emits top-level `def compute(...)`; indent it 4 spaces.
    rendered_body = "\n".join(
        ("    " + line if line and not line.startswith("@") and not line.startswith("class ")
         else line)
        for line in rendered_body.split("\n")
    )
    out.write_text("\n".join(header) + "\n" + rendered_body + "\n", encoding="utf-8")
    print(f"Emitted {len(emitted)} variants from {len(top_nums)} top alphas;"
          f" skipped {len(skipped)} non-transformable: {skipped}")
    print(f"Wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path,
                    help="factor_analysis JSON with abs_ic_mean dict")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    generate(args.baseline, args.top_n, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Note on indentation:** the simple `"    "` prefix above is a hack that may break if `ast.unparse` produces multi-line strings or comments. Before relying on it, run a small dry-run (next step) and **read the generated file** — if any compute body is malformed, switch to inserting the unparsed compute as a single multi-line indented string built via `textwrap.indent(ast.unparse(new_compute), "    ")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_wq101_variants.py -v`
Expected: PASS

If indentation issues fail the import in `test_generated_module_registers_and_computes`, switch to `textwrap.indent(ast.unparse(new_compute), "    ")` for the compute body interpolation. Don't generate complex template strings — keep it explicit.

- [ ] **Step 5: Smoke-check the generator manually**

Run: `.venv/Scripts/python.exe scripts/generate_wq101_variants.py --baseline reports/factor_analysis/2026-06-20.json --top-n 3 --output /tmp/_smoke.py`
Then `cat /tmp/_smoke.py | head -50` — verify the file parses by:
`.venv/Scripts/python.exe -c "import ast; ast.parse(open('/tmp/_smoke.py').read())"`
Expected: no error.

- [ ] **Step 6: Commit the script + tests (NOT the generated file yet)**

The real generated file is produced in Task 7 from the Phase 0 re-run baseline. We commit only the generator + tests here.

```bash
git add scripts/generate_wq101_variants.py tests/test_wq101_variants.py
git commit -m "$(cat <<'EOF'
feat(scripts): WQ101 variant generator (Phase 2 of A-share localization)

AST-rewriter that takes an analyze_factors baseline JSON, picks the top-N
WQ101 alphas, and emits a Python module with three rule-based variants
per alpha (_compress / _rev_short / _expand_long). Non-literal-window
alphas are skipped. The generated module imports cleanly and registers
each variant via the existing @register decorator.

The actual generated wq101_variants.py is produced in Task 7 from the
post-Phase-0 baseline; not committed here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Auto-import variants on stockpool.factors load

**Files:**
- Modify: `src/stockpool/factors/__init__.py`

**Goal:** Ensure the generated `wq101_variants` module gets imported when `stockpool.factors` is first loaded, so its variants are present in the registry without callers needing to import the module manually.

- [ ] **Step 1: Identify the existing wq101 import**

Run: `grep -n 'wq101' src/stockpool/factors/__init__.py`
Confirm there is a line like `from stockpool.factors import wq101 as _wq101  # noqa: F401`. The variants module needs the same treatment, but **guarded by file existence** so the import doesn't crash before Task 7 generates the file.

- [ ] **Step 2: Add the conditional import**

Edit `src/stockpool/factors/__init__.py`. Right after the existing `wq101` import, add:

```python
# WQ101 A-share-localized variants (Phase 2 of WQ101 localization plan).
# Generated by scripts/generate_wq101_variants.py from a factor_analysis
# baseline; absent before Phase 3 runs. Wrap import to keep stockpool
# importable when the generated file doesn't exist.
try:
    from stockpool.factors import wq101_variants as _wq101_variants  # noqa: F401
except ImportError:
    pass
```

- [ ] **Step 3: Verify nothing breaks now**

Run: `.venv/Scripts/python.exe -c "from stockpool.factors import list_factors; print(len(list_factors()))"`
Expected: prints a number around 165 (no `wq101_variants` yet, ImportError caught).

- [ ] **Step 4: Run the full factor test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors.py tests/test_wq101.py -v -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/__init__.py
git commit -m "$(cat <<'EOF'
feat(factors): auto-import wq101_variants if generated module is present

Adds a guarded import so the localized WQ101 variants register
automatically once Task 7 generates wq101_variants.py. The try/except
keeps stockpool importable when the file is absent (e.g. clean clone).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Phase 0 re-baseline + Phase 3 walk-forward analyze

**Files:**
- Create: `reports/wq101_round1_factors.json` (committed)
- Create: `reports/factor_analysis/wq101_round1_h1/<date>.json` and `wq101_round1_h2/<date>.json` (NOT committed — too large; add to `.gitignore`)
- Create / commit: `src/stockpool/factors/wq101_variants.py` (generated)

**Goal:** Re-run `factors analyze` on full universe with Phase 0 defaults to obtain a clean baseline. Then generate variants for the top-30 wq101 from that baseline. Build the Round 1 factor list (30 baseline + ~90 variants). Run analyze twice: first half of dates and second half.

- [ ] **Step 1: Re-run full analyze with Phase 0 defaults**

Run:
```bash
cd C:/Users/Administrator/Desktop/claude
.venv/Scripts/python.exe -m stockpool factors analyze \
    --universe all \
    --output reports/factor_analysis
```

This takes ~15-30 min (4357 stocks × 167 factors). The new JSON will be at `reports/factor_analysis/<today>.json`. Note the path; the rest of Task 7 uses it.

- [ ] **Step 2: Sanity check the new baseline against the spec acceptance**

```bash
.venv/Scripts/python.exe - <<'PY'
import json
p = sorted(__import__("pathlib").Path("reports/factor_analysis").glob("2026-*.json"))[-1]
d = json.load(open(p, encoding="utf-8"))
abs_ic = d["abs_ic_mean"]
degen = d.get("degenerate_day_ratio", {})
print(f"baseline: {p}")
print(f"alpha_096 abs_ic = {abs_ic.get('alpha_096')}")
print(f"alpha_096 degenerate_ratio = {degen.get('alpha_096')}")
print(f"ewma_vol_hl10 abs_ic = {abs_ic.get('ewma_vol_hl10')}")
PY
```

Expected per spec §3.0.2:
- `alpha_096 abs_ic` ≤ 0.10 (was 0.4773)
- `alpha_096 degenerate_ratio` ≥ 0.30
- `ewma_vol_hl10 abs_ic` ∈ [0.155, 0.195] (was 0.1758; tolerance ±0.02)

If any of these fail, **stop and investigate** — Phase 0 was buggy. Don't proceed.

- [ ] **Step 3: Generate the variants from the new baseline**

```bash
NEW_BASELINE=$(ls -t reports/factor_analysis/2026-*.json | head -1)
.venv/Scripts/python.exe scripts/generate_wq101_variants.py \
    --baseline "$NEW_BASELINE" \
    --top-n 30 \
    --output src/stockpool/factors/wq101_variants.py
```

Verify import works:
```bash
.venv/Scripts/python.exe -c "from stockpool.factors import list_factors; \
    all_f = list_factors(); \
    print(f'total={len(all_f)}'); \
    print(f'localized={sum(1 for f in all_f if any(s in f for s in [\"_compress\", \"_rev_short\", \"_expand_long\"]))}')"
```

Expected: total grows by ~70-90 (some alphas with non-literal windows are skipped); localized count ~70-90.

- [ ] **Step 4: Build the Round 1 factor list JSON**

Create `scripts/build_round1_factor_list.py`:

```python
"""Build reports/wq101_round1_factors.json from a baseline + generated variants.

Output is {"factors": [top-30 wq101 baseline names + all _compress/_rev_short/_expand_long variants of those names]}.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from stockpool.factors import list_factors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    d = json.loads(args.baseline.read_text(encoding="utf-8"))
    abs_ic = d["abs_ic_mean"]
    pairs = sorted(
        ((float(v), n) for n, v in abs_ic.items()
         if n.startswith("alpha_") and v is not None and v == v),
        reverse=True,
    )
    top_names = [n for _, n in pairs[:args.top_n]]

    all_registered = set(list_factors())
    factors: list[str] = []
    for base in top_names:
        factors.append(base)
        for rule in ("compress", "rev_short", "expand_long"):
            v = f"{base}_{rule}"
            if v in all_registered:
                factors.append(v)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"factors": factors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(factors)} factor names to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run it:
```bash
.venv/Scripts/python.exe scripts/build_round1_factor_list.py \
    --baseline "$NEW_BASELINE" \
    --top-n 30 \
    --output reports/wq101_round1_factors.json
```

Verify: `cat reports/wq101_round1_factors.json | head -5` — should show ~120 factor names.

- [ ] **Step 5: Run walk-forward analyze (two halves)**

We need to split the date range. The cleanest approach is to use the project's existing `data/<code>_daily.parquet` cache plus a small wrapper that limits the panel to a date range. Since `cmd_factors_analyze` doesn't currently accept date bounds, add a thin one-off wrapper script:

Create `scripts/run_walkforward_analyze.py`:

```python
"""Walk-forward wrapper around analyze_factors: splits the date range
in half, runs once per half, saves to <output>/h1/<date>.json and h2/."""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from stockpool.config import load_config
from stockpool.factors_analysis import analyze_factors
from stockpool.factors_analysis_report import render_factor_analysis_report
from stockpool.factors.context import set_sector_map
from stockpool.industry_map import load_or_build_industry_map
from stockpool.panel import build_panel_from_cache


def _slice_panel(panel: dict, start, end) -> dict:
    return {k: v.loc[start:end] for k, v in panel.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--factors-file", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--horizon", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(cfg.data.cache_dir)
    universe_file = cache_dir / "universe.parquet"
    all_codes = pd.read_parquet(universe_file)["code"].tolist()
    codes = [c for c in all_codes if (cache_dir / f"{c}_daily.parquet").exists()]

    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    set_sector_map(sector_map or {})

    panel = build_panel_from_cache(codes, cfg.data.history_days, cache_dir)
    factor_names = list(json.loads(
        args.factors_file.read_text(encoding="utf-8"))["factors"])

    dates = panel["close"].index
    mid = dates[len(dates) // 2]
    halves = [
        ("h1", dates.min(), mid),
        ("h2", mid + pd.Timedelta(days=1), dates.max()),
    ]
    stamp = date.today().isoformat()
    for tag, lo, hi in halves:
        sub = _slice_panel(panel, lo, hi)
        result = analyze_factors(
            panel=sub, factor_names=factor_names,
            horizon=args.horizon,
        )
        out_dir = args.output_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        result.to_json(out_dir / f"{stamp}.json")
        render_factor_analysis_report(result, out_dir / f"{stamp}.html")
        print(f"wrote {tag} to {out_dir}/{stamp}.{{json,html}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add the output directories to `.gitignore`:
```bash
echo "reports/factor_analysis/wq101_round1_h1/" >> .gitignore
echo "reports/factor_analysis/wq101_round1_h2/" >> .gitignore
```

Run:
```bash
.venv/Scripts/python.exe scripts/run_walkforward_analyze.py \
    --factors-file reports/wq101_round1_factors.json \
    --output-root reports/factor_analysis/wq101_round1 \
    --horizon 3
```

Expected: ~20-40 min total compute (120 factors × 2 halves). Both `h1/<date>.json` and `h2/<date>.json` exist.

- [ ] **Step 6: Commit (generator output + factor list, NOT the analyze outputs)**

```bash
git add src/stockpool/factors/wq101_variants.py \
        scripts/build_round1_factor_list.py \
        scripts/run_walkforward_analyze.py \
        reports/wq101_round1_factors.json \
        .gitignore
git commit -m "$(cat <<'EOF'
feat(factors): WQ101 Round 1 variants generated + walk-forward driver

Re-baseline with Phase 0 defaults, then emit wq101_variants.py covering
top-30 baseline alphas × 3 rules. Walk-forward analyze wrapper produces
h1 + h2 JSONs needed for Task 8's winner picker. Reports kept out of git.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Winner picker (Phase 4)

**Files:**
- Create: `scripts/pick_wq101_winners.py`
- Create: `reports/wq101_round1_winners.csv` (committed)
- Create: `reports/selection_wq101_localized.json` (committed)
- Test: `tests/test_pick_wq101_winners.py`

**Goal:** For each top-30 baseline alpha, evaluate its three variants against the baseline on BOTH walk-forward halves. Keep a variant only if it passes the dual-half criterion (spec §7.1):
- h1: `Δ abs_ic ≥ 0.02` AND `Δ |ir| ≥ 0.1` AND `degenerate_day_ratio ≤ 0.10`
- h2: same as h1
- When multiple variants pass for the same baseline, pick the one whose **minimum** of `(abs_ic_h1, abs_ic_h2)` is largest (conservative).

Output:
- `reports/wq101_round1_winners.csv` — `baseline_alpha, chosen_variant, abs_ic_baseline_h1, abs_ic_winner_h1, delta_abs_ic_h1, delta_ir_h1, abs_ic_baseline_h2, abs_ic_winner_h2, delta_abs_ic_h2, delta_ir_h2`
- `reports/selection_wq101_localized.json` — current `reports/selection.json` with top-30 baseline names swapped for winners (baselines without a winner are kept unchanged).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pick_wq101_winners.py`:

```python
def test_picker_passes_dual_half_threshold(tmp_path):
    """Variant passing both halves is selected over baseline."""
    import json, subprocess, sys
    from pathlib import Path

    # Synthetic baseline + variants. alpha_002_compress passes both halves.
    def _mkjson(d): return json.dumps(d, ensure_ascii=False)

    h1 = tmp_path / "h1.json"; h2 = tmp_path / "h2.json"
    base_template = {
        "factor_names": ["alpha_002", "alpha_002_compress",
                          "alpha_002_rev_short", "alpha_002_expand_long"],
        "abs_ic_mean": {"alpha_002": 0.08, "alpha_002_compress": 0.12,
                        "alpha_002_rev_short": 0.09,
                        "alpha_002_expand_long": 0.06},
        "ic_ir": {"alpha_002": 0.10, "alpha_002_compress": 0.30,
                  "alpha_002_rev_short": 0.15,
                  "alpha_002_expand_long": 0.05},
        "degenerate_day_ratio": {"alpha_002": 0.0, "alpha_002_compress": 0.0,
                                  "alpha_002_rev_short": 0.05,
                                  "alpha_002_expand_long": 0.0},
    }
    h1.write_text(_mkjson(base_template))
    h2.write_text(_mkjson(base_template))

    cur = tmp_path / "selection.json"
    cur.write_text(_mkjson({"factors": ["alpha_002", "momentum_20"]}))

    winners_csv = tmp_path / "winners.csv"
    new_sel = tmp_path / "new_sel.json"

    subprocess.run(
        [sys.executable, "scripts/pick_wq101_winners.py",
         "--h1", str(h1), "--h2", str(h2),
         "--current-selection", str(cur),
         "--winners-csv", str(winners_csv),
         "--output-selection", str(new_sel),
         "--baseline-top-n", "1"],
        check=True,
    )
    import csv
    rows = list(csv.DictReader(open(winners_csv, encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["chosen_variant"] == "alpha_002_compress"
    out = json.loads(new_sel.read_text(encoding="utf-8"))
    # baseline alpha_002 replaced by alpha_002_compress
    assert "alpha_002_compress" in out["factors"]
    assert "alpha_002" not in out["factors"]
    # unrelated factor kept
    assert "momentum_20" in out["factors"]


def test_picker_keeps_baseline_when_no_variant_passes(tmp_path):
    import json, subprocess, sys
    base = {
        "factor_names": ["alpha_002", "alpha_002_compress"],
        "abs_ic_mean": {"alpha_002": 0.08, "alpha_002_compress": 0.085},
        "ic_ir": {"alpha_002": 0.10, "alpha_002_compress": 0.11},
        "degenerate_day_ratio": {"alpha_002": 0.0, "alpha_002_compress": 0.0},
    }
    h1 = tmp_path / "h1.json"; h2 = tmp_path / "h2.json"
    h1.write_text(json.dumps(base)); h2.write_text(json.dumps(base))
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps({"factors": ["alpha_002"]}))
    winners = tmp_path / "win.csv"
    new = tmp_path / "new.json"
    subprocess.run(
        [sys.executable, "scripts/pick_wq101_winners.py",
         "--h1", str(h1), "--h2", str(h2),
         "--current-selection", str(cur),
         "--winners-csv", str(winners),
         "--output-selection", str(new),
         "--baseline-top-n", "1"],
        check=True,
    )
    import csv
    rows = list(csv.DictReader(open(winners, encoding="utf-8")))
    assert rows == []  # no winners
    out = json.loads(new.read_text(encoding="utf-8"))
    assert out["factors"] == ["alpha_002"]  # baseline preserved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pick_wq101_winners.py -v`
Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Write the picker script**

Create `scripts/pick_wq101_winners.py`:

```python
"""Pick WQ101 variants that beat their baseline on both walk-forward halves.

Criteria per spec §7.1:
  h1 & h2 each: Δabs_ic ≥ 0.02 AND Δ|ir| ≥ 0.1 AND degenerate ≤ 0.10.
  Pick variant maximizing min(abs_ic_h1, abs_ic_h2).
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _passes(base_row, var_row, *,
            min_dabs=0.02, min_dir=0.10, max_degen=0.10) -> bool:
    if var_row["degen"] > max_degen:
        return False
    if (var_row["abs"] - base_row["abs"]) < min_dabs:
        return False
    if (abs(var_row["ir"]) - abs(base_row["ir"])) < min_dir:
        return False
    return True


def _table(j: dict) -> dict[str, dict]:
    abs_ic = j["abs_ic_mean"]; ir = j["ic_ir"]; degen = j.get("degenerate_day_ratio", {})
    return {n: {
        "abs": float(abs_ic[n]) if abs_ic[n] is not None else float("nan"),
        "ir": float(ir[n]) if ir[n] is not None else float("nan"),
        "degen": float(degen.get(n, 0.0)) if degen.get(n) is not None else 0.0,
    } for n in j["factor_names"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1", required=True, type=Path)
    ap.add_argument("--h2", required=True, type=Path)
    ap.add_argument("--current-selection", required=True, type=Path)
    ap.add_argument("--winners-csv", required=True, type=Path)
    ap.add_argument("--output-selection", required=True, type=Path)
    ap.add_argument("--baseline-top-n", type=int, default=30,
                    help="Only consider this many baseline wq101 alphas, "
                         "ranked by min(abs_ic_h1, abs_ic_h2).")
    args = ap.parse_args()

    h1, h2 = _table(_load(args.h1)), _table(_load(args.h2))
    cur = _load(args.current_selection)
    # Baseline candidate set: alphas present in both tables, non-variant suffixed.
    suffixes = ("_compress", "_rev_short", "_expand_long")
    baselines = sorted(
        n for n in h1
        if n.startswith("alpha_")
        and not any(n.endswith(s) for s in suffixes)
        and n in h2
    )
    # Rank baselines by min(abs_ic_h1, abs_ic_h2) descending; take top N.
    baselines.sort(key=lambda n: -min(h1[n]["abs"], h2[n]["abs"]))
    baselines = baselines[:args.baseline_top_n]

    winners: list[dict] = []
    for base in baselines:
        candidates = []
        for s in suffixes:
            var = base + s
            if var not in h1 or var not in h2:
                continue
            if not _passes(h1[base], h1[var]):
                continue
            if not _passes(h2[base], h2[var]):
                continue
            candidates.append((min(h1[var]["abs"], h2[var]["abs"]), var))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        chosen = candidates[0][1]
        winners.append({
            "baseline_alpha": base,
            "chosen_variant": chosen,
            "abs_ic_baseline_h1": h1[base]["abs"],
            "abs_ic_winner_h1": h1[chosen]["abs"],
            "delta_abs_ic_h1": h1[chosen]["abs"] - h1[base]["abs"],
            "delta_ir_h1": abs(h1[chosen]["ir"]) - abs(h1[base]["ir"]),
            "abs_ic_baseline_h2": h2[base]["abs"],
            "abs_ic_winner_h2": h2[chosen]["abs"],
            "delta_abs_ic_h2": h2[chosen]["abs"] - h2[base]["abs"],
            "delta_ir_h2": abs(h2[chosen]["ir"]) - abs(h2[base]["ir"]),
        })

    args.winners_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.winners_csv.open("w", newline="", encoding="utf-8") as fh:
        if winners:
            w = csv.DictWriter(fh, fieldnames=list(winners[0].keys()))
            w.writeheader(); w.writerows(winners)
        else:
            fh.write("baseline_alpha,chosen_variant\n")

    # Build the new selection: swap baselines for winners.
    swap = {w_["baseline_alpha"]: w_["chosen_variant"] for w_ in winners}
    new_factors = [swap.get(f, f) for f in cur["factors"]]
    args.output_selection.parent.mkdir(parents=True, exist_ok=True)
    args.output_selection.write_text(
        json.dumps({"factors": new_factors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{len(winners)} winners written to {args.winners_csv}")
    print(f"Updated selection written to {args.output_selection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pick_wq101_winners.py -v`
Expected: PASS

- [ ] **Step 5: Run on real Round 1 data**

```bash
H1=$(ls -t reports/factor_analysis/wq101_round1/h1/*.json | head -1)
H2=$(ls -t reports/factor_analysis/wq101_round1/h2/*.json | head -1)
.venv/Scripts/python.exe scripts/pick_wq101_winners.py \
    --h1 "$H1" --h2 "$H2" \
    --current-selection reports/selection.json \
    --winners-csv reports/wq101_round1_winners.csv \
    --output-selection reports/selection_wq101_localized.json \
    --baseline-top-n 30
```

**Spec gate (§7.2):** `wc -l reports/wq101_round1_winners.csv` ≥ 7 (6 winners + 1 header). If fewer, the hypothesis is partially falsified — STOP, document the result in the spec's complaint section, and ask the user whether to proceed to Phase 5 anyway, abandon, or skip to Round 2.

- [ ] **Step 6: Commit**

```bash
git add scripts/pick_wq101_winners.py tests/test_pick_wq101_winners.py \
        reports/wq101_round1_winners.csv reports/selection_wq101_localized.json
git commit -m "$(cat <<'EOF'
feat(scripts): WQ101 Round 1 winner picker (Phase 4 of localization)

Dual-half threshold: Δabs_ic ≥ 0.02 AND Δ|ir| ≥ 0.1 AND degen ≤ 10% on
both halves. Picks the variant maximizing min(abs_ic_h1, abs_ic_h2).
Outputs winners CSV and a swapped-in selection.json compatible with
factors_file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: AB validation (Phase 5)

**Files:**
- Create: `ab/wq101_localized.yaml`
- Run: `reports/ab/wq101_localized/<date>.html` (NOT committed)

**Goal:** Compare the two `selection.json` variants under identical pipeline conditions on the ~100-stock AB pool. Acceptance per spec §8.3.

- [ ] **Step 1: Confirm AB pool exists**

Run: `ls -la data/ab_pool.parquet`
Expected: file present, recent. If missing, build it: `.venv/Scripts/python.exe -m stockpool ab-pool build`. This takes ~5 min.

- [ ] **Step 2: Write the AB config**

Create `ab/wq101_localized.yaml`:

```yaml
base_config: config.yaml
use_ab_pool: true
arms:
  baseline:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        horizon: 3
        train_window: 250
        refit_every: 20
        panel_mode: pooled
        training_universe: all
        share_pool_fit: true
        embargo_days: null
        label_type: return
        selector:
          lasso:
            alpha: 0.001
            max_iter: 5000
            tol: 1e-4
        weighter:
          ic:
            use_rank: true
            min_abs_ic: 0.02
        thresholds:
          strong_buy: 0.8
          buy: 0.6
          sell: 0.4
          strong_sell: 0.2
  localized:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection_wq101_localized.json
        horizon: 3
        train_window: 250
        refit_every: 20
        panel_mode: pooled
        training_universe: all
        share_pool_fit: true
        embargo_days: null
        label_type: return
        selector:
          lasso:
            alpha: 0.001
            max_iter: 5000
            tol: 1e-4
        weighter:
          ic:
            use_rank: true
            min_abs_ic: 0.02
        thresholds:
          strong_buy: 0.8
          buy: 0.6
          sell: 0.4
          strong_sell: 0.2
```

(Adjust the numeric strategy params to match whatever is currently in `config.yaml:strategy.ml_factor`. The point is identical strategy hyperparams across both arms, with only `factors_file` differing.)

- [ ] **Step 3: Run the AB**

```bash
.venv/Scripts/python.exe -m stockpool ab --config ab/wq101_localized.yaml
```

Expected: ~30-90 min. Output at `reports/ab/wq101_localized/<today>.html`.

- [ ] **Step 4: Read the AB outcome and decide**

Open the report (browser) or read its JSON summary if `stockpool.ab` writes one. Find the per-stock median Δ Sharpe (localized minus baseline).

Apply spec §8.3 thresholds:
- `≥ +0.10`: SHIP → continue to Task 10
- `∈ [-0.05, +0.10)`: NEUTRAL → archive, do not change `selection.json`, document in spec §14, ask user about Round 2
- `< -0.05`: FAIL → archive, leave `selection.json` alone, document in spec §14, mark spec failed

Do NOT modify `reports/selection.json` automatically — the swap is a user-driven decision once the AB outcome is on the table.

- [ ] **Step 5: Add AB report dir to .gitignore + commit**

```bash
echo "reports/ab/wq101_localized/" >> .gitignore
git add ab/wq101_localized.yaml .gitignore
git commit -m "$(cat <<'EOF'
feat(ab): WQ101 localized vs baseline arm config

Identical ml_factor strategy hyperparams across both arms; only the
factors_file differs (reports/selection.json vs the swap from Task 8).
AB report kept out of git.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Documentation + final commit (Phase 7)

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-06-21-wq101-a-share-localization-design.md` (fill §14)

**Goal:** Record the outcome and propagate user-facing notes.

- [ ] **Step 1: Update CLAUDE.md factor library section**

Add (under "WQ101" item in the modules table or in the source-tag list):

> WQ101 本土化变体(`wq101_localized` source tag):由 `scripts/generate_wq101_variants.py` 从 IC 基线生成,对 top-30 WQ101 alpha 应用三种窗口规则(`_compress` / `_rev_short` / `_expand_long`)。变体与原版共存,通过 `selection.json` 切换。

Also add a `factors analyze` flag mention to the relevant section:

> `factors analyze` 默认 `winsorize=(0.01, 0.99)` + `degenerate_day_unique_ratio_threshold=0.01`(2026-06-21,Phase 0 of WQ101 localization plan);旧报告 IC 数字不可与新版直接比较

- [ ] **Step 2: Fill spec §14 复盘**

Edit `docs/superpowers/specs/2026-06-21-wq101-a-share-localization-design.md` §14 with:
- # winners (from `wq101_round1_winners.csv`)
- AB per-stock median Δ Sharpe
- Outcome (ship / neutral / fail)
- Open follow-ups (e.g. "Round 2 deferred — bottom-30 wq101 might be hopeless")

- [ ] **Step 3: Final commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-06-21-wq101-a-share-localization-design.md
git commit -m "$(cat <<'EOF'
docs(wq101): close out A-share localization Round 1

Record AB outcome in spec §14 + add wq101_localized source tag note in
CLAUDE.md. Round 2 (bottom-30) decision deferred per spec §9.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

**Spec coverage:**
- §1 (motivation) → context, no implementation work needed
- §2 (scope) → enforced by all tasks
- §3 (Phase 0) → Tasks 1-3
- §4 (Phase 1) → Task 4
- §5 (Phase 2) → Tasks 5, 6
- §6 (Phase 3) → Task 7
- §7 (Phase 4) → Task 8
- §8 (Phase 5) → Task 9
- §9 (Phase 6 / Round 2) → **explicitly deferred to a follow-up plan**, not in this plan
- §10 (Phase 7 / docs) → Task 10
- §11 (risks) → addressed inline where relevant (e.g. Task 5 indentation fallback, Task 7 sanity gate)
- §12-14 → docs / handoff

**No placeholders:** every code step has runnable code or exact commands; no "TBD".

**Type consistency:** `winsorize` is `tuple[float, float] | None` everywhere; `degenerate_day_unique_ratio_threshold` is `float`; `FactorAnalysisResult.degenerate_day_ratio` is `pd.Series`; `scripts/generate_wq101_variants.py` uses `--baseline` / `--top-n` / `--output`, matching Tasks 5 and 7.

**Known fragile points:**
- Task 5 source-rendering (the `"    "` indent prefix); fallback documented
- Task 9 ab.yaml `strategy.ml_factor` block must match whatever `config.yaml` currently has; engineer instructed to mirror, not invent

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-21-wq101-a-share-localization.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
