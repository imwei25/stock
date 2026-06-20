# A/B Testing Tool — Design

**Status:** spec / awaiting review
**Date:** 2026-05-24
**Branch (intended):** `feat/ab-testing`

## Problem

The project has two backtestable strategies (`composite_verdict` and `ml_factor`,
the latter with multiple sub-variants — `selector ∈ {lasso, lightgbm}`,
`weighter ∈ {ic, ir, equal, lightgbm}`, etc.) and no first-class way to compare
two of them on the same universe under identical execution conditions.

The user iterates on strategy parameters (factors, selector, weighter,
holding-day caps), and currently has to hand-flip the main `config.yaml`,
re-run `python -m stockpool backtest`, and eyeball the two reports. That is
tedious, error-prone (forget to flip a knob back), and provides no
side-by-side aggregate view.

## Goals (MVP)

- A second YAML (`ab.yaml`) declares two named "arms" (e.g. `composite_default`
  vs `ml_lgbm_top20`); each arm overrides only `strategy:` and `backtest:`,
  everything else is inherited from a referenced `base_config: config.yaml`.
- A new CLI command `python -m stockpool ab --config ab.yaml` runs both arms
  against the same per-stock universe (currently `cfg.stocks`, optionally
  filtered) and produces a comparison HTML report in `reports/ab/`.
- The existing backtest framework is **not** re-engineered: `Strategy` ABC,
  `BacktestEngine`, `MultiLotBacktestEngine`, and `simulate_strategy_equity_curve`
  are used as-is. A small refactor moves the per-stock backtest loop out of
  `cli.cmd_backtest` into a reusable helper.

## Non-Goals (explicit follow-ups)

- **Portfolio-level A/B** — comparing strategies that operate on a cross-stock
  panel (one equity curve per strategy across the whole pool). Needs a new
  `PortfolioStrategy` ABC + new engine. Will be its own spec; the per-stock
  MVP here is not designed to be retrofitted into portfolio mode — that work
  will be a separate redesign.
- **Statistical significance tests** (paired t-test, Wilcoxon, bootstrap CI).
  At 8–30 stocks the sample is too small for meaningful p-values. Revisit
  when Pool B integration brings the sample to hundreds.
- **Comparing two `composite_verdict` configs that differ in `indicators` /
  `weights` / `verdicts` / `scoring`**. These fields live at the top level of
  `AppConfig` (historical asymmetry — `ml_factor` is cleanly subnested under
  `strategy.ml_factor.*`, `composite_verdict` is not). Allowing arms to
  override top-level fields would make the arm-config schema semantically
  messy. The correct fix is to first refactor `composite_verdict` params into
  `strategy.composite_verdict.*`, after which arms would automatically gain
  this capability via the existing `strategy:` override mechanism. That
  refactor is a separate PR.
- **More than 2 arms** (A/B/C/...). The pairwise diff tables, scatter plot,
  and "A wins / B wins" counts are built on a 2-element comparison.
- **CLI flags overriding `stocks_filter`** — A/B is "serious experiments";
  stock selection should be in the config file for reproducibility. Use
  `--arm` to debug a single side.

## Architecture

### New module layout

```
src/stockpool/ab/
  __init__.py                # public: ABConfig, load_ab_config, run_ab, ABResult
  config.py                  # ABConfig, ArmOverride, ArmBacktestOverride (pydantic)
                             #   + load_ab_config(path) + build_effective_cfg(base, arm)
  runner.py                  # run_ab(ab_cfg, base_cfg, stocks, refresh, share_pool=True)
                             #   _decide_pool_sharing(arm_cfgs, stocks) → plan
                             #   _run_arm(arm_cfg, name, stocks, pool, panel, refresh) → ArmResult
  report.py                  # render_ab_report(ab_result, output_dir) → Path
                             # compute_diff_table / scatter / histogram helpers

src/stockpool/backtest_runner.py    # NEW (extracted from cli.py)
                             # prepare_pool(cfg, stocks, refresh) → (pool_data, factor_panel)
                             # backtest_stocks(cfg, stocks, pool_data, factor_panel,
                             #                 shared_cache, refresh)
                             #     → (per_stock: list[(code, name, EquityResult)],
                             #        failed: list[(code, err)])
```

### Modified files

| File | Change |
|---|---|
| `src/stockpool/cli.py` | Move `_prepare_ml_pool` → `backtest_runner.prepare_pool`; move per-stock backtest loop from `cmd_backtest` → `backtest_runner.backtest_stocks`; `cmd_backtest` becomes a thin caller. Add `cmd_ab` + subparser registration. |
| `CLAUDE.md` | Add `ab/` and `backtest_runner.py` to module map; add `ab` CLI to quick commands; add `test_ab.py` / `test_cli_ab.py` to test table; note the composite_verdict subnesting follow-up. |
| `README.md` | Add `ab` to quick-start command list; add a worked end-to-end "compare two strategies" section with a sample `ab.yaml`. |

### Dependency direction

```
ab/runner.py
    │
    └──> backtest_runner.py <── cli.cmd_backtest
              │
              └──> strategy_factory + backtesting + recommend_pool + fetcher
                                                    (existing — unchanged)
```

`ab/` only depends downward on `backtest_runner`, never on `cli`. This is the
*reason* for extracting `backtest_runner.py`: avoid `ab/` reverse-importing
the CLI module.

### Data model

```python
@dataclass
class ArmResult:
    name: str                                                  # arm key from ab.yaml
    effective_cfg: AppConfig                                   # base ⊕ arm.override
    per_stock: list[tuple[str, str, EquityResult]]             # (code, name, result)
    failed: list[tuple[str, str]]                              # (code, error message)

@dataclass
class ABResult:
    ab_cfg: ABConfig
    base_cfg: AppConfig
    arm_a: ArmResult
    arm_b: ArmResult
    run_date: str
```

`EquityResult` is the existing dataclass from `backtest_composite.py`
(`curves: dict[int, DataFrame]`, `metrics: dict[int, dict]`, `buy_and_hold`,
`buy_and_hold_metrics`). Each arm's `equity_curve_holding_days` is forced to
length 1 (see schema), so `curves` and `metrics` always have exactly one key.

## Config schema (`ab.yaml`)

### Example

```yaml
base_config: config.yaml             # required; relative to ab.yaml's directory

stocks_filter: ["605589", "300750"]  # optional; empty → all base.stocks.
                                     # codes must exist in base.stocks (subset only)

arms:                                 # dict, exactly 2 keys
  composite_default:                  # arm name (free-form, shown in report)
    strategy:                         # required: replaces base.strategy wholesale
      name: composite_verdict
    backtest:                         # required; must contain equity_curve_holding_days: [N]
      equity_curve_holding_days: [10]

  ml_lgbm_top20:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        selector: {type: lightgbm}
        weighter: {type: lightgbm}
    backtest:
      equity_curve_holding_days: [10]
      # other backtest fields (engine / costs / position_size / max_concurrent_lots /
      # risk_free_rate / forward_days) inherit base.backtest
```

### Pydantic models

```python
class ArmBacktestOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    equity_curve_holding_days: list[int]                       # required; len == 1
    forward_days: list[int] | None = None
    risk_free_rate: float | None = None
    costs: BacktestCostConfig | None = None
    engine: Literal["single", "multi_lot"] | None = None
    position_size: float | None = None
    max_concurrent_lots: int | None = None

    @field_validator("equity_curve_holding_days")
    @classmethod
    def _single_n(cls, v):
        if len(v) != 1 or v[0] <= 0:
            raise ValueError("equity_curve_holding_days must be [N] with N > 0")
        return v


class ArmOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: StrategyConfig                                   # required, replaced wholesale
    backtest: ArmBacktestOverride                              # required


class ABConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_config: str
    stocks_filter: list[str] = Field(default_factory=list)
    arms: dict[str, ArmOverride]                               # exactly 2 keys

    @field_validator("arms")
    @classmethod
    def _exactly_two(cls, v):
        if len(v) != 2:
            raise ValueError(f"arms must have exactly 2 entries, got {len(v)}")
        return v
```

### Deep-merge semantics (`build_effective_cfg(base, arm) → AppConfig`)

| Field | Behaviour |
|---|---|
| `arm.strategy` | Replaces `base.strategy` **wholesale** (no field-level merge). |
| `arm.backtest.equity_curve_holding_days` | Replaces `base.backtest.equity_curve_holding_days` (forced length 1). |
| `arm.backtest.<other>` | If not `None`, replaces same-named field in `base.backtest`; if `None`, inherits. |
| `base.stocks` | If `stocks_filter` non-empty, filtered to those codes; else passed through. |
| All other top-level fields (`data`, `indicators`, `weights`, `scoring`, `verdicts`, `report`, `context`, `recommend_pool`) | Pass through from base, arm cannot override. |

Implementation: `base.model_dump()` → in-place update of the merged dict →
`AppConfig.model_validate(merged_dict)` (re-runs pydantic, re-computes
`content_hash`). The re-validation ensures any merged result that is malformed
fails fast at load time.

### Validation rules

Two-stage validation, both early-fail:

1. **Pydantic schema** (when `ABConfig.model_validate` runs inside
   `load_ab_config`): `arms` count == 2, `equity_curve_holding_days` is
   singleton with N > 0, `extra="forbid"` rejects typoed field names on
   every arm-level model, type coercion on all primitive fields.
2. **Post-load checks in `load_ab_config`** (require side info pydantic
   doesn't have):
   - `base_config` path resolves to an existing file → if not, raise.
   - Load the base config; for each `code` in `stocks_filter`, verify it
     appears in `base.stocks` → if not, raise listing the offending codes.
   - For each arm: `build_effective_cfg(base, arm)` is called once
     speculatively so that any deep-merge result that fails `AppConfig`
     re-validation (e.g. `position_size: 2.0` in an arm override) surfaces
     at load time, not at run time.

**Runtime errors** (per-stock fetch failure, ML training crash) go into
`ArmResult.failed` and never propagate across the arm boundary.

### Why this shape

- `strategy` replaced wholesale: it is a single semantic object (`name`
  determines which subsection — `ml_factor` — is valid); field-level merge
  produces dangling-reference foot-guns (`arm.strategy.name = "composite_verdict"`
  but inheriting `base.strategy.ml_factor.factors_file`).
- `backtest` field-level merge: typical arms only change `holding_days`;
  wholesale would force re-typing `engine`, `costs`, etc.
- `stocks_filter` subtract-only: ensures cache hits (every code in `base.stocks`
  has been seen by prior fetch runs and has a local parquet); avoids A/B
  triggering surprise network fetches.
- Length-1 `equity_curve_holding_days`: deliberate constraint matching the
  "A vs B at one N" report shape; sweeping N is a separate question already
  covered by `python -m stockpool backtest`.

## CLI

### Subcommand

```
python -m stockpool ab --config ab.yaml [--refresh] [--arm <name>] [--no-share-pool]
```

| Flag | Default | Purpose |
|---|---|---|
| `--config` | `ab.yaml` | A/B config file path (not the base config). |
| `--refresh` | `False` | Forwarded to `prepare_pool` / `backtest_stocks` (`force_refresh`). |
| `--arm <name>` | `None` | Debug: run only one arm by key; skips report rendering, prints metrics to stdout. |
| `--no-share-pool` | `False` | Escape hatch: force each arm to load its own universe / factor panel even when both could share. |

No `--stocks` override (use `stocks_filter` in YAML). No `--output-dir`
(uses `base_cfg.report.output_dir`).

### `cmd_ab` control flow

```python
def cmd_ab(args) -> int:
    ab_cfg = load_ab_config(args.config)                       # all static errors raise here

    base_cfg_path = Path(args.config).parent / ab_cfg.base_config
    base_cfg = load_config(base_cfg_path)

    run_date = date.today().isoformat()
    out_root = Path(base_cfg.report.output_dir) / "ab"
    _setup_logging(out_root / run_date)
    log.info("stockpool ab v%s for %s", __version__, run_date)

    stocks = _apply_stocks_filter(base_cfg.stocks, ab_cfg.stocks_filter)

    if args.arm:                                               # debug mode
        if args.arm not in ab_cfg.arms:
            log.error("--arm %r not in %s", args.arm, list(ab_cfg.arms))
            return 2
        arm_result = run_single_arm(ab_cfg, base_cfg, stocks, args.refresh, args.arm)
        _print_single_arm_stdout(arm_result)
        return 0

    result = run_ab(ab_cfg, base_cfg, stocks, args.refresh,
                    share_pool=not args.no_share_pool)
    if not result.arm_a.per_stock and not result.arm_b.per_stock:
        log.error("Both arms produced no results.")
        return 1
    out = render_ab_report(result, output_dir=out_root)
    log.info("AB report written: %s", out)
    log.info("Latest also at: %s", out_root / "latest.html")
    return 0
```

### Output

```
reports/ab/
  2026-05-24/run.log
  2026-05-24.html
  latest.html              # byte-copy of 2026-05-24.html
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (including partial failures inside arms) |
| 1 | Both arms produced zero successful stocks |
| 2 | Argument / config error (`--arm` unknown, ab.yaml validation failure, etc.) |

## Runner internals

### Pool sharing plan

`_decide_pool_sharing(arm_cfgs, stocks) → plan`:

| Condition | `load_universe` | `shared_factors` |
|---|---|---|
| Both arms ml_factor + both `panel_mode=pooled` + both `training_universe=all` | `True` | `arm_a.factors` if equal to `arm_b.factors` (order-sensitive), else `None` |
| One ml_factor + one composite_verdict, ml side wants universe | `True` (only used by ml side's `prepare_pool`) | `None` |
| Both composite_verdict | `False` | `None` |
| Any other mix | `False` | `None` |

`--no-share-pool` short-circuits to `load_universe=False, shared_factors=None`
regardless of config (each arm calls `prepare_pool` independently).

### `run_ab` outline

Two entry points — full 2-arm comparison and a single-arm debug helper —
keep `ABResult` semantics clean (always exactly 2 arms; no placeholder hacks):

```python
def run_ab(ab_cfg, base_cfg, stocks, refresh, *, share_pool=True) -> ABResult:
    """Run both arms; returns ABResult with exactly two ArmResults."""
    arm_items = list(ab_cfg.arms.items())                      # exactly 2 (schema)
    arm_cfgs = [build_effective_cfg(base_cfg, arm) for _, arm in arm_items]

    plan = _decide_pool_sharing(arm_cfgs, stocks) if share_pool else _no_share_plan()
    shared_universe = (
        load_universe_cache(base_cfg.data.cache_dir, base_cfg.data.history_days)
        if plan["load_universe"] else None
    )

    shared_panel = None
    arm_results = []
    for (name, _), arm_cfg in zip(arm_items, arm_cfgs):
        pool_data, factor_panel = _prepare_pool_for_arm(
            arm_cfg, stocks, refresh,
            injected_universe=shared_universe,
            injected_factor_panel=(shared_panel if plan["shared_factors"] else None),
        )
        if plan["shared_factors"] and shared_panel is None:
            shared_panel = factor_panel
        arm_results.append(_run_arm(arm_cfg, name, stocks,
                                    pool_data, factor_panel, refresh))

    return ABResult(
        ab_cfg=ab_cfg, base_cfg=base_cfg,
        arm_a=arm_results[0], arm_b=arm_results[1],
        run_date=date.today().isoformat(),
    )


def run_single_arm(ab_cfg, base_cfg, stocks, refresh, arm_name: str) -> ArmResult:
    """Debug helper: run only one arm by name; skips pool sharing entirely
    (no opposite arm to share with). Used by --arm flag."""
    arm = ab_cfg.arms[arm_name]
    arm_cfg = build_effective_cfg(base_cfg, arm)
    pool_data, factor_panel = _prepare_pool_for_arm(
        arm_cfg, stocks, refresh,
        injected_universe=None, injected_factor_panel=None,
    )
    return _run_arm(arm_cfg, arm_name, stocks, pool_data, factor_panel, refresh)
```

### `_run_arm`

```python
def _run_arm(arm_cfg, arm_name, stocks, pool_data, factor_panel, refresh) -> ArmResult:
    log.info("Running arm %s ...", arm_name)
    per_stock, failed = backtest_stocks(
        arm_cfg, stocks, pool_data, factor_panel,
        shared_cache={}, refresh=refresh,
    )
    log.info("Arm %s: %d done, %d failed", arm_name, len(per_stock), len(failed))
    return ArmResult(name=arm_name, effective_cfg=arm_cfg,
                     per_stock=per_stock, failed=failed)
```

`shared_cache` is fresh per arm (configs differ → cached pipeline reuse across
arms is incorrect).

### `backtest_runner.backtest_stocks`

Extracted from current `cli.cmd_backtest` per-stock loop, with one behaviour
change: returns `(success, failed)` tuple instead of logging and dropping
failures. `cmd_backtest` post-extraction:

```python
per_stock, failed = backtest_stocks(cfg, stocks, pool_data, factor_panel, {}, args.refresh)
for code, err in failed:
    log.warning("Skipped %s: %s", code, err)
if not per_stock:
    log.error("No stocks could be backtested.")
    return 1
# remaining rendering unchanged
```

### ML cache isolation

`MLFactorStrategy` writes monthly fits to `<cache_dir>/ml_models/<sig>_<code>.pkl`
where `sig = 8-char hash of MLFactorConfig`. Two arms with any differing
ml_factor field automatically get distinct `sig`s → distinct pkl paths → no
cross-arm contamination. Second runs of the same A/B re-use the on-disk
fits transparently.

### Failure isolation matrix

| Scenario | Behaviour |
|---|---|
| `fetch_daily(code)` raises | code → `failed`, loop continues. |
| ML training crashes for a code | Same — exception caught at the per-stock boundary, code → `failed`. |
| Arm has zero successful stocks | `ArmResult.per_stock = []`; opposite arm still runs; report renders with a banner. |
| Both arms empty | `run_ab` returns; `cmd_ab` detects and returns exit code 1. |
| `prepare_pool` itself crashes (e.g. universe parquet corrupted) | Exception propagates; `cmd_ab` returns 1 with traceback in log. |

## Report (`reports/ab/<date>.html`)

### Top: experiment metadata banner

Shows arm names, the differing fields each arm declared (their `strategy:` +
`backtest:` overrides — not the full effective_cfg, which is noise), base
config hash, stock count, per-arm success/failure counts.

### Top charts (above the per-stock fold)

1. **Aggregate diff table** — one row per metric, columns: A mean / A median /
   B mean / B median / `B − A` mean / A wins / B wins. Computed over
   stocks where **both arms succeeded**. `max_drawdown` annotated "less is
   better"; `trade_count` shown but not scored.
2. **Sharpe scatter** — x = A.sharpe, y = B.sharpe, one point per common
   stock (labelled by code); reference diagonal y = x; points above = B wins.
3. **Sharpe diff histogram** — `B.sharpe - A.sharpe` per common stock; bin
   width auto from data range (8–12 bins).

### Per-stock cards

One `<details>` per stock (first 3 expanded by default, rest collapsed):

- Equity curve chart with three series: Arm A (solid blue), Arm B (solid red),
  Buy & Hold (dashed grey). datazoom + tooltip mirroring existing
  `backtest_report._equity_chart` style.
- Side-by-side metric table (A / B / Δ), Δ coloured green ✓ when B wins,
  red ✗ when A wins (with "less is better" inversion on `max_drawdown`).
- Stocks where one arm failed: card title shows `[Arm A failed]`, only the
  successful side's curve and single-arm metrics rendered.

### Bottom

- Failure detail list (folded): both arms' `failed` entries with error messages.
- Full effective_cfg yaml dumps for both arms (folded), for reproducibility.

### Reuse vs new

| Element | Source |
|---|---|
| `_CSS`, `_optimize_html`, page chrome | Imported from `backtest_report.py`. |
| Equity chart (3-series flavour) | New `_ab_equity_chart(arm_a_curve, arm_b_curve, bh_curve, title)` in `ab/report.py`. |
| Aggregate table / scatter / histogram | New, in `ab/report.py`. |
| pyecharts axis/datazoom defaults | Extracted to a module-level constant `_DEFAULT_AXIS_OPTS` shared between `backtest_report` and `ab/report` (small refactor inside this spec). |

## Testing

### New file: `tests/test_ab.py`

#### Config (`ab/config.py`)

- `test_arm_override_requires_strategy_and_backtest`
- `test_arm_holding_days_must_be_singleton` (`[5,10]`, `[]`, `[0]` fail; `[10]` passes)
- `test_arms_must_be_exactly_two` (1 fails, 3 fail, 2 passes)
- `test_arm_extra_fields_forbidden` (`arm.indicators`, `arm.backtest.foo`)
- `test_stocks_filter_must_be_subset_of_base`
- `test_load_ab_config_resolves_base_path_relative_to_ab_yaml`
- `test_load_ab_config_missing_base_raises`

#### Deep-merge (`build_effective_cfg`)

- `test_merge_replaces_strategy_section_wholly`
- `test_merge_backtest_fields_inherit_when_none`
- `test_merge_backtest_fields_override_when_set`
- `test_merge_recomputes_content_hash` (precondition for ML cache isolation)
- `test_merge_does_not_mutate_base`
- `test_merge_revalidates_pydantic` (e.g. `position_size: 2.0`)

#### Pool sharing (`_decide_pool_sharing`)

- `test_pool_plan_both_ml_pooled_all_same_factors`
- `test_pool_plan_both_ml_pooled_all_different_factors`
- `test_pool_plan_ml_vs_composite`
- `test_pool_plan_both_composite`
- `test_pool_plan_one_ml_per_stock`
- `test_no_share_pool_flag_disables_sharing`

#### Runner failure isolation

- `test_backtest_stocks_returns_success_and_failed_tuple`
- `test_backtest_stocks_continues_after_per_stock_exception`
- `test_run_ab_one_arm_total_failure_other_still_runs`
- `test_run_ab_both_arms_empty_returns_empty_result`

#### ML cache isolation

- `test_ml_cache_sig_differs_across_arms_with_different_ml_cfg`

### New file: `tests/test_cli_ab.py`

- `test_cmd_ab_smoke_two_composite_arms` (synthetic OHLCV via monkeypatched `fetch_daily`)
- `test_cmd_ab_arm_flag_runs_only_one_arm`
- `test_cmd_ab_arm_unknown_returns_2`
- `test_cmd_ab_no_share_pool_flag_propagates`

### Report smoke (in `tests/test_ab.py`)

- `test_render_ab_report_smoke` (synthetic ABResult: 2 arms × 3 stocks)
- `test_render_ab_report_one_arm_empty`
- `test_render_ab_report_single_arm_failure_per_stock`
- `test_compute_diff_table_only_uses_common_stocks`

### Regression: `cmd_backtest` after extraction

- `test_cmd_backtest_logs_skipped_stocks_after_refactor` — guards the
  "middle-stock crash doesn't kill the loop" behaviour through the new
  `(success, failed)` return shape. All existing `test_cli_backtest.py` tests
  must continue to pass.

### Explicitly NOT tested

- Concrete pyecharts JSON structure (existing `backtest_report.py` also
  doesn't — too brittle). Only smoke "render doesn't raise + HTML contains
  arm names + stock codes".
- `_optimize_html` lib dedup / lazy load (already tested in `report.py`).
- ML pool-data injection horizontal slicing (covered by `test_ml_strategy_panel.py`).

### Test density

~25 new test cases across two new files, matching the project's per-file
density (5–15 tests per file).

## Documentation

### `CLAUDE.md` updates

- Module map: add `src/stockpool/ab/` row + `src/stockpool/backtest_runner.py` row
- Quick commands: add `python -m stockpool ab --config ab.yaml`
- Configuration section: add a note "A/B comparisons use a separate
  `ab.yaml`; see `docs/superpowers/specs/2026-05-24-ab-testing-design.md`"
- Test table: add `test_ab.py`, `test_cli_ab.py` rows
- Known unsupported: add bullet "Top-level `weights` / `verdicts` / `scoring`
  cannot be overridden per-arm (follow-up: subnest under
  `strategy.composite_verdict.*`)"

### `README.md` updates

- Quick commands: add `python -m stockpool ab --config ab.yaml`
- New worked example: "对比两个策略 — A/B testing" with a minimal sample
  `ab.yaml` and pointer to the report path

## Open questions resolved

| Question | Decision |
|---|---|
| ab.yaml shape — independent or shared base? | Shared base (`base_config:`) with arm-level `strategy:` + `backtest:` overrides. |
| Report richness | HTML with side-by-side curves, aggregate table, Sharpe scatter, diff histogram. No significance tests (follow-up). |
| Portfolio-level support | Out of scope. Per-stock MVP only; portfolio-level is a separate spec / redesign. |
| Multi-N holding days | Each arm picks exactly one N (`equity_curve_holding_days: [10]`, length-1 enforced by schema). |
| Allow overriding `indicators` / `weights` / `verdicts`? | No. Follow-up: refactor `composite_verdict` params into `strategy.composite_verdict.*` first. |
| Number of arms | Exactly 2 (schema-enforced). |
| `stocks_filter` semantics | Subset-only (subtract from base), validated as subset of base.stocks. |
| `--no-share-pool` escape hatch | Yes. |
| Pool sharing factors equality | Order-sensitive list equality (no `sorted == sorted`). |

## Implementation order (handoff to plan)

1. Extract `backtest_runner.py` (`prepare_pool`, `backtest_stocks`). Refactor
   `cli.cmd_backtest` to use it. Run existing test suite — must stay green.
2. Add `ab/config.py` (schema + `load_ab_config` + `build_effective_cfg`)
   with unit tests.
3. Add `ab/runner.py` (`_decide_pool_sharing`, `_run_arm`, `run_ab`,
   `run_single_arm`) with unit tests and failure-isolation tests.
4. Add `ab/report.py` (charts + tables + HTML assembly) with smoke tests.
5. Add `cmd_ab` to `cli.py` + subparser registration + `test_cli_ab.py`.
6. Update `CLAUDE.md` + `README.md`.
7. Full `pytest -q` run.
