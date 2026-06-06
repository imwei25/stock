# Factor Preprocessing — Phase 1 Design

**Date**: 2026-06-06
**Status**: Approved (brainstorming session completed; ready for implementation plan)
**Scope**: Cross-sectional preprocessing pipeline for ML factor panels (winsorize / cs-zscore / industry neutralize). Phase 1 of the multi-phase roadmap in `docs/research/2026-06-06-factor-preprocessing-and-orthogonalization.md`.
**Out of scope**: NaN imputation, market-cap neutralization, symmetric orthogonalization (all deferred to later phases).

---

## 1. Motivation

The accompanying research doc (`docs/research/2026-06-06-...`) identified that the project's selection/weighting backend is mature but the *front* of the pipeline — cross-sectional preprocessing — is missing. Concretely:

- No winsorize → single outlier flips daily IC sign
- No cross-sectional z-score → only pooled z-score in `_StandardisingMixin`, which mixes time-series drift into "factor signal"
- No default industry neutralization → strategy implicitly bets on sector rotation, not factor effect

Phase 1 closes those three gaps with the smallest possible surface area: 3 stateless functions, 1 config block, 1 wiring change in `build_factor_panel`. Default off — fully backwards compatible.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  build_factor_panel(factor_names, pool_data,                │
│                     preprocess_cfg=None)                    │
│                                                             │
│    ① compute_factor_panel(panel, factor_names)              │
│         ↓                                                    │
│      raw_factor_panel: dict[str, T×N DataFrame]             │
│         ↓                                                    │
│    ② apply_preprocess_pipeline(raw, cfg, sector_map, types) │
│         ↓                                                    │
│         ├─ winsorize_panel(df, lo, hi)                      │
│         ├─ cs_zscore_panel(df)                              │
│         └─ industry_neutralize_panel(df, sector_map)         │
│            (skip factors tagged "fundamental")               │
│         ↓                                                    │
│    preprocessed_factor_panel → caller (factor cache)         │
└─────────────────────────────────────────────────────────────┘
```

**Key invariants**:

- Preprocessing happens **after** `compute_factor_panel`, **before** `load_or_build_factor_panel` writes to disk. Both train and predict (and Pool B) consume the preprocessed values → no consistency hole.
- `Factor.compute(panel)` interface is **unchanged** — preprocessing is a panel-level pipeline step, not a factor concept.
- Tradability mask (existing 2026-05-31 design) is orthogonal: mask still acts only on labels (`forward_return_panel`); factor values still see real prices including limit-up days.
- All 3 preprocessing functions are **stateless** — no fit/transform split. They take a panel and return a panel. Look-ahead safe because they only consume cross-sectional information per day.

---

## 3. Module Layout

### 3.1 New file: `src/stockpool/ml/preprocess.py` (~150 lines)

Four functions, all stateless:

```python
def winsorize_panel(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """每日截面 clip 到 [lower 分位, upper 分位]。

    Raises:
        ValueError: if not 0 < lower < upper < 1.
    """

def cs_zscore_panel(df: pd.DataFrame) -> pd.DataFrame:
    """每日截面 (x - μ_t) / σ_t,σ_t 用 ddof=0;σ_t<1e-12 当日返回 0(中性化)。"""

def industry_neutralize_panel(
    df: pd.DataFrame,
    sector_map: Mapping[str, str],
) -> pd.DataFrame:
    """每日截面按行业 demean。未在 sector_map 的 code 进 "_unknown_" 桶。

    Raises:
        ValueError: if sector_map is empty (caller wraps in try/skip).
    """

def apply_preprocess_pipeline(
    factor_panel: dict[str, pd.DataFrame],
    cfg: "PreprocessConfig",
    sector_map: Mapping[str, str] | None = None,
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, pd.DataFrame]:
    """串行 winsorize → cs_zscore → industry_neutralize。
    industry_neutralize 跳过 factor_types[name] 含 "fundamental" 的因子。
    cfg 全关时返回浅拷贝。
    """
```

### 3.2 Modified file: `src/stockpool/config.py`

```python
class PreprocessConfig(BaseModel):
    """Cross-sectional preprocessing pipeline for ML factor panels.

    Applied at build_factor_panel() output before caching; affects
    ml_factor training, predict, and Pool B consumers identically.
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


class MLFactorConfig(BaseModel):
    ...
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
```

### 3.3 Modified file: `src/stockpool/strategy_factory.py`

```python
def build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> dict[str, pd.DataFrame]:
    # ... existing panel construction ...
    raw = compute_factor_panel(panel, factor_names)
    if preprocess_cfg is None or _is_all_off(preprocess_cfg):
        return raw

    from stockpool.factors.context import get_sector_map
    from stockpool.factors.registry import list_specs
    from stockpool.ml.preprocess import apply_preprocess_pipeline

    sector_map = get_sector_map() or {}
    types_map = {
        s.name: s.types for s in list_specs() if s.name in factor_names
    }
    return apply_preprocess_pipeline(raw, preprocess_cfg, sector_map, types_map)


def _factor_panel_sig(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> tuple[str, str]:
    # ... existing key build ...
    blob_dict = {
        "factors": sorted(factor_names),
        "codes": codes,
        "last_date": last_iso,
        "preprocess": (
            preprocess_cfg.model_dump()
            if preprocess_cfg and not _is_all_off(preprocess_cfg)
            else None
        ),
    }
    # ...


def load_or_build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    cache_dir: str | Path,
    refresh: bool = False,
    preprocess_cfg: "PreprocessConfig | None" = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    # threads preprocess_cfg into both sig and build_factor_panel
    ...
```

**Backwards-compat guarantee**: when `preprocess_cfg` is None or all-off, sig dict includes `"preprocess": None` (the explicit `None`, not the cfg dict). Old yaml without the `preprocess` field gets `PreprocessConfig()` default → `_is_all_off()` true → `None` in sig dict → identical hash to pre-PR baseline → existing factor_panels/ cache stays valid.

### 3.4 Modified callers (3 sites, 1 line each)

- `cli.py:_prepare_ml_pool` → pass `preprocess_cfg=cfg.strategy.ml_factor.preprocess`
- `backtest_runner.py:prepare_pool` → same
- `cli.py:cmd_portfolio_*` → same (portfolio path shares the same factor cache)

### 3.5 Files NOT changed

- `src/stockpool/ml/dataset.py` — preprocess is upstream; `stack_panel_to_xy` consumes whatever it gets
- `src/stockpool/ml/weighters.py` — `_StandardisingMixin` retained as defensive no-op; pooled z-score on already-cs-z-scored data is harmless rescaling
- `src/stockpool/ml/selectors.py` — same reasoning

---

## 4. Configuration Schema & AB Example

### 4.1 yaml shape

```yaml
strategy:
  ml_factor:
    ...
    preprocess:
      winsorize: [0.01, 0.99]   # null = off
      zscore: true              # per-day cross-sectional
      industry_neutralize: true # uses factors.context.get_sector_map()
```

### 4.2 AB config: `ab_preprocess.yaml`

Sibling of `ab_mask.yaml`; only difference between arms is the `preprocess` block.
Both arms use:

- `panel_mode: pooled`
- `training_universe: pool` (sticks with P3-2 verdict — `all` showed regression)
- `selector.lasso`, `weighter.ic` (project defaults, P0-1 verdict)
- `mask.enabled: false`
- `equity_curve_holding_days: [10]`

Arm A `baseline`: `preprocess: { winsorize: null, zscore: false, industry_neutralize: false }`
Arm B `with_preprocess`: `preprocess: { winsorize: [0.01, 0.99], zscore: true, industry_neutralize: true }`

---

## 5. Error Handling Matrix

| Scenario | Behavior |
|---|---|
| `sector_map` empty + `industry_neutralize=true` | log warning once, skip industry step, keep winsorize + zscore output |
| Cross-section all-NaN day | winsorize / zscore / neutralize all return original (all-NaN) row unchanged — shape preserved, no crash |
| `σ_t < 1e-12` (constant cross-section) | zscore returns 0 for that day (neutralizes deterministically) |
| `winsorize=[0.99, 0.01]` reversed | Pydantic raises at config load |
| `winsorize=[0, 1]` no-op | Pydantic raises (`0 < lo < hi < 1`) |
| Factor not in `factor_types` dict | treated as non-fundamental (goes through industry neutralize) |
| Factor with multiple type tags including `"fundamental"` | skipped from industry neutralize |
| `preprocess_cfg=None` passed explicitly | bypass entire pipeline; cache sig matches pre-PR baseline |

---

## 6. Test Plan

**Target**: 615 → ~640 cases, all green.

### 6.1 New: `tests/test_ml_preprocess.py` (~250 lines, ~18 cases)

Single-function tests (synthetic 30 stocks × 5 days panel):
- `test_winsorize_clips_to_quantile`
- `test_winsorize_all_nan_row_passthrough`
- `test_winsorize_invalid_bounds_raises`
- `test_winsorize_preserves_index_columns`
- `test_cs_zscore_mean_zero_std_one`
- `test_cs_zscore_constant_row_returns_zero`
- `test_cs_zscore_handles_nan`
- `test_cs_zscore_preserves_index_columns`
- `test_industry_neutralize_within_group_mean_zero`
- `test_industry_neutralize_unknown_code_bucket`
- `test_industry_neutralize_empty_sector_map_raises`
- `test_industry_neutralize_preserves_index_columns`

Pipeline tests:
- `test_pipeline_all_off_returns_input` (shallow-copy semantics)
- `test_pipeline_all_on_three_steps_order`
- `test_pipeline_skips_neutralize_when_no_sector_map` (warning + skip)
- `test_pipeline_skips_neutralize_for_fundamental_types`
- `test_pipeline_partial_steps_independent`
- `test_pipeline_preserves_factor_keys_and_shapes`

### 6.2 Extend: `tests/test_factor_panel_cache.py` (~4 cases)

- `test_cache_sig_with_preprocess_isolated_from_baseline`
- `test_cache_sig_all_off_backwards_compat` (sig identical to no-preprocess)
- `test_cache_invalidates_on_preprocess_change`
- `test_build_factor_panel_passes_preprocess` (spy via monkeypatch)

### 6.3 Extend: `tests/test_config.py` (~3 cases)

- `test_preprocess_config_defaults_all_off`
- `test_preprocess_winsorize_invalid_bounds_raises`
- `test_preprocess_extra_field_forbidden`

### 6.4 Extend: `tests/test_ml_strategy.py` (1 smoke case)

- `test_ml_factor_with_preprocess_runs_end_to_end` — 8 synthetic stocks × 100 bars, full train + predict, verify no crash and IC computed.

---

## 7. AB Validation Plan

### 7.1 Run

```bash
.venv/Scripts/python.exe -m stockpool ab --config ab_preprocess.yaml
```

Output: `docs/ab_runs/preprocess_phase1.html` (auto-generated by ab framework).

### 7.2 Pass criteria

Adopt thresholds from `docs/ab_validation_runbook.md`:

| Metric | Pass line | Notes |
|---|---|---|
| Δ sharpe | ≥ +0.05 | primary metric |
| Δ total return | same direction as sharpe | required for consistency |
| Stocks won | majority (> n_stocks/2) | trade-by-trade robustness; current `config.yaml` has 18 stocks → ≥ 10 |
| Max drawdown | not worse by > 3 pp | risk gate |

### 7.3 Decision tree

| Outcome | Action |
|---|---|
| ✅ Pass (all 4 gates) | Update `docs/ab_validation_results.md` with new section; turn on `preprocess` in `config.yaml` default; proceed to Phase 2 (market-cap neutralization, separate spec) |
| ⚠️ Indecisive (`\|Δsharpe\| < 0.05`) | Run ablation: 3 sub-A/Bs each toggling one step alone; investigate whether 2 helpful steps are being cancelled by a 3rd harmful one |
| ❌ Regression | Write negative verdict to `docs/ab_validation_results.md`; do NOT change `config.yaml` default; spec status updated to "Phase 1 validated negative"; root-cause analysis (likely candidates: (a) IC sign flip on z-scored factors, (b) `cfg.stocks` cherry-picked so sector neutralize destroys their alpha, (c) winsorize too tight for 8-stock cross-section) |

### 7.4 Telemetry to capture (manually from per-stock cards in HTML)

- Mean Δsharpe across all stocks (mean + std)
- Stocks-won count
- Per-arm trade count (large delta suggests preprocess changing the signal density)

---

## 8. Cache & Compatibility

**factor_panel cache**:
- Sig hash extended to include `preprocess.model_dump()` when not all-off
- All-off case maps to `"preprocess": None` in sig dict → byte-identical to pre-PR hash → no orphan cache files

**ml_models cache** (`<sig>_<code>.pkl`):
- Already keyed off full `MLFactorConfig` dump → automatic invalidation when `preprocess` changes

**Backwards compatibility check**:
- Existing `ab_mask.yaml`, `ab_mask_small.yaml`, `ab_sizing.yaml`, `config.yaml` have no `preprocess` field
- Pydantic loads → `PreprocessConfig()` default → all off → `_is_all_off()` true → cache sig matches pre-PR
- All 615 existing tests should pass without modification

---

## 9. Documentation Updates

After implementation lands (per CLAUDE.md "改动后更新文档" rule):

- `CLAUDE.md`:
  - Add row in factor library section: "新增 `ml/preprocess.py` — 截面 winsorize / zscore / 行业中性 流水线"
  - Add row in 配置 section under `strategy.ml_factor`: "`preprocess.{winsorize, zscore, industry_neutralize}`"
  - Add row in 测试 section: "`test_ml_preprocess.py`"
- `README.md`:
  - Add `preprocess` line in the ml_factor config example

After AB verdict written:
- `docs/ab_validation_results.md` — new section with metrics + verdict
- `docs/research/2026-06-06-factor-preprocessing-and-orthogonalization.md` — append Phase 1 outcome note

---

## 10. Out of Scope (Phase 2+)

- NaN imputation (cross-sectional median fill)
- Market-cap neutralization (needs new data pipeline)
- Symmetric / Gram-Schmidt orthogonalization
- Per-factor-family preprocess configuration (current design is global on/off)
- Adaptive winsorize bounds (e.g., MAD-based)
- Automatic sector_map loading inside `apply_preprocess_pipeline` (caller responsibility — keeps the module pure)

Each of these is a candidate for a follow-up spec if Phase 1 validates positively.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Pooled z-score in `_StandardisingMixin` double-standardizes preprocessed input | Pooled re-z on already-cs-z-scored data is a near-identity rescale (mean ~0, std ~1 already); kept as defensive no-op |
| `factors.context.get_sector_map()` not set in some code paths → neutralize silently no-ops | Log warning once per `apply_preprocess_pipeline` call; covered by `test_pipeline_skips_neutralize_when_no_sector_map` |
| Cross-section too thin (e.g., 8 cfg.stocks) makes winsorize a no-op (quantile range too narrow) | Expected — winsorize designed for cross-sections of 100+ stocks; AB on small pool may understate Phase 1 benefit |
| `industry_neutralize` for `cfg.stocks` (sector-concentrated picks) wipes alpha | Mitigated by `factor_types` skip on fundamental; for technical factors, AB verdict is the call |

---

## 12. References

- Research doc: `docs/research/2026-06-06-factor-preprocessing-and-orthogonalization.md`
- Prior AB verdict tradition: `docs/ab_validation_results.md`, `docs/ab_validation_runbook.md`
- Mask design precedent: `docs/superpowers/specs/2026-05-31-tradability-mask-design.md`
- Related research: `docs/research/2026-05-31-a-share-quant-survey-comparison.md` §3.4
