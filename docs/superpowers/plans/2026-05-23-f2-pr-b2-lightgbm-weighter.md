# F2 PR-B2 — LightGBM Weighter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship F2 PR-B2 from spec `docs/superpowers/specs/2026-05-23-f2-pr-b2-lightgbm-weighter-design.md`: add `LightGBMWeighter`, refactor `WeighterConfig` to subnested form, flip default `weighter.type` to `"lightgbm"`, and move `contributions()` from `TwoStepPipeline` down to the `FactorWeighter` ABC.

**Architecture:** Six ordered tasks. Task 1 refactors the config schema (subnested + factory + yaml migration) keeping default `"ic"` to preserve behavior. Task 2 introduces the `contributions()` polymorphism on linear weighters and simplifies `TwoStepPipeline`. Task 3 adds the `LightGBMWeighter` class. Task 4 flips defaults + wires the factory + patches legacy fixtures. Task 5 syncs docs. Task 6 verifies.

**Tech Stack:** Python 3.10+, lightgbm 4.x (already required dep from PR-B1), Pydantic, pandas, numpy, pytest. No new dependencies.

---

## File Structure

| File | Change |
|------|--------|
| `src/stockpool/config.py` | Add `ICWeighterConfig`, `IRWeighterConfig`, `EqualWeighterConfig`, `LightGBMWeighterConfig`; restructure `WeighterConfig` to subnested form with `extra="forbid"` |
| `src/stockpool/ml/weighters.py` | Add `contributions()` abstract method to `FactorWeighter`; add `_LinearWeighterContributionsMixin`; apply mixin to IC/IR/Equal; add `LightGBMWeighter` class |
| `src/stockpool/ml/pipeline.py` | Simplify `TwoStepPipeline.contributions()` to a delegate |
| `src/stockpool/backtesting/strategies.py` | `_build_weighter` factory reads from subnested fields + adds `"lightgbm"` case; import `LightGBMWeighter` |
| `config.yaml` | Migrate `weighter` block to subnested form |
| `tests/test_config.py` | New tests for WeighterConfig subnesting (~10 tests) |
| `tests/test_ml_pipeline.py` | No changes — existing contributions tests should still pass numerically |
| `tests/test_ml_weighter_lightgbm.py` (new) | 8 unit + integration tests for `LightGBMWeighter` |
| `tests/test_ml_strategy.py` | Patch fixtures with `weighter=WeighterConfig(type="ic")` to preserve old IC numerics |
| `tests/test_ml_strategy_panel.py` | Same |
| `tests/test_ml_strategy_embargo.py` | Same for the 3 `_try_fit`-exercising fixtures |
| `CLAUDE.md` | Update module map + `weighter.{ic|ir|equal|lightgbm}` config description |
| `README.md` | Extend LGB caveat with weighter info |

---

## Task 1: WeighterConfig subnested refactor (default kept `"ic"`)

**Files:**
- Modify: `src/stockpool/config.py` (the `WeighterConfig` class around line 154)
- Modify: `src/stockpool/backtesting/strategies.py` (`_build_weighter` around line 247)
- Modify: `config.yaml` (the `weighter` block under `strategy.ml_factor`)
- Test: `tests/test_config.py`

**Critical**: This task changes the config schema AND the factory AND the user's YAML in one shot. They must commit together because `_build_weighter` will fail mid-refactor if either alone is committed. Default `type` stays `"ic"` — the flip to `"lightgbm"` happens in Task 4.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
def test_weighter_default_type_is_still_ic_in_task1():
    """Task 1 keeps default type='ic' to preserve behavior. Task 4 flips it."""
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig()
    assert cfg.type == "ic"


def test_weighter_ic_subcfg_explicit():
    """selector.ic.use_rank parses from YAML."""
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({
        "type": "ic",
        "ic": {"use_rank": False, "min_abs_ic": 0.05},
    })
    assert cfg.type == "ic"
    assert cfg.ic.use_rank is False
    assert cfg.ic.min_abs_ic == 0.05


def test_weighter_ic_subcfg_defaults():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({"type": "ic"})
    assert cfg.ic.use_rank is True
    assert cfg.ic.min_abs_ic == 0.0


def test_weighter_ir_subcfg_explicit():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({
        "type": "ir",
        "ir": {"n_chunks": 4, "use_rank": False, "min_abs_ir": 0.1},
    })
    assert cfg.type == "ir"
    assert cfg.ir.n_chunks == 4
    assert cfg.ir.use_rank is False
    assert cfg.ir.min_abs_ir == 0.1


def test_weighter_ir_subcfg_defaults():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({"type": "ir"})
    assert cfg.ir.n_chunks == 6
    assert cfg.ir.use_rank is True
    assert cfg.ir.min_abs_ir == 0.0


def test_weighter_equal_subcfg_parses():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({"type": "equal"})
    # EqualWeighterConfig has no params; just confirm the subfield exists
    assert cfg.type == "equal"
    assert cfg.equal is not None


def test_weighter_lightgbm_subcfg_explicit():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({
        "type": "lightgbm",
        "lightgbm": {
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.1,
            "num_iterations": 100,
            "max_depth": 6,
            "random_state": 7,
            "verbose": 0,
        },
    })
    assert cfg.type == "lightgbm"
    assert cfg.lightgbm.num_leaves == 31
    assert cfg.lightgbm.learning_rate == 0.1


def test_weighter_lightgbm_subcfg_defaults():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({"type": "lightgbm"})
    assert cfg.lightgbm.num_leaves == 15
    assert cfg.lightgbm.min_data_in_leaf == 20
    assert cfg.lightgbm.learning_rate == 0.05
    assert cfg.lightgbm.num_iterations == 200
    assert cfg.lightgbm.max_depth == 4
    assert cfg.lightgbm.random_state == 42
    assert cfg.lightgbm.verbose == -1


def test_weighter_flat_use_rank_rejected():
    """Old flat 'use_rank' at WeighterConfig top level is rejected (extra='forbid')."""
    import pydantic
    from stockpool.config import WeighterConfig
    with pytest.raises(pydantic.ValidationError):
        WeighterConfig.model_validate({"type": "ic", "use_rank": True})


def test_weighter_unknown_type_rejected():
    import pydantic
    from stockpool.config import WeighterConfig
    with pytest.raises(pydantic.ValidationError):
        WeighterConfig.model_validate({"type": "catboost"})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k "weighter" --no-header
```
Expected: 10 FAILs — `cfg.ic.use_rank` AttributeError, `pydantic.ValidationError` not raised, etc.

- [ ] **Step 3: Refactor `WeighterConfig` in `src/stockpool/config.py`**

Find the existing `class WeighterConfig(BaseModel):` (around line 154). REPLACE the whole class and INSERT four new subconfig classes immediately before it. Final state:

```python
class ICWeighterConfig(BaseModel):
    """IC weighter hyperparameters (was flat fields on WeighterConfig pre-PR-B2)."""
    model_config = ConfigDict(extra="forbid")
    use_rank: bool = True
    min_abs_ic: float = Field(default=0.0, ge=0.0)


class IRWeighterConfig(BaseModel):
    """IR weighter hyperparameters.

    ``IRWeighter`` internally uses ``use_rank`` to choose Spearman vs Pearson
    when computing per-chunk IC; ``min_abs_ir`` filters factors by IR magnitude.
    """
    model_config = ConfigDict(extra="forbid")
    n_chunks: int = Field(default=6, gt=0)
    use_rank: bool = True
    min_abs_ir: float = Field(default=0.0, ge=0.0)


class EqualWeighterConfig(BaseModel):
    """Equal weighter has no hyperparameters; this is an empty placeholder
    so ``WeighterConfig.equal`` is a real Pydantic object (uniform structure)."""
    model_config = ConfigDict(extra="forbid")


class LightGBMWeighterConfig(BaseModel):
    """LightGBM weighter hyperparameters. Defaults match LightGBMSelectorConfig
    for symmetric YAML structure; tune ``num_iterations`` upward if prediction
    quality is the bottleneck."""
    model_config = ConfigDict(extra="forbid")
    num_leaves: int = Field(default=15, gt=1)
    min_data_in_leaf: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_iterations: int = Field(default=200, gt=0)
    max_depth: int = Field(default=4, gt=0)
    random_state: int = Field(default=42, ge=0)
    verbose: int = Field(default=-1)


class WeighterConfig(BaseModel):
    """Step-2 (factor weighting) settings.

    PR-B2 refactors this from flat fields to subnested per-type blocks
    (ic / ir / equal / lightgbm), parallel to PR-A's SelectorConfig.
    Default ``type`` is currently ``"ic"`` (Task 1); Task 4 flips it to
    ``"lightgbm"`` after LightGBMWeighter is implemented.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["ic", "ir", "equal", "lightgbm"] = "ic"
    ic: ICWeighterConfig = Field(default_factory=ICWeighterConfig)
    ir: IRWeighterConfig = Field(default_factory=IRWeighterConfig)
    equal: EqualWeighterConfig = Field(default_factory=EqualWeighterConfig)
    lightgbm: LightGBMWeighterConfig = Field(default_factory=LightGBMWeighterConfig)
```

- [ ] **Step 4: Update `_build_weighter` factory in `src/stockpool/backtesting/strategies.py`**

Find the existing `def _build_weighter(cfg) -> FactorWeighter:` (around line 247). REPLACE the body to read from subnested fields. **Do NOT add a `"lightgbm"` branch yet — that comes in Task 4.** Raise `NotImplementedError` if `cfg.type == "lightgbm"` reaches the factory now:

```python
def _build_weighter(cfg) -> FactorWeighter:
    """Translate WeighterConfig → concrete FactorWeighter.

    PR-B2 Task 1: reads from subnested fields.
    PR-B2 Task 4 will add the "lightgbm" case.
    """
    if cfg.type == "ic":
        return ICWeighter(use_rank=cfg.ic.use_rank, min_abs_ic=cfg.ic.min_abs_ic)
    if cfg.type == "ir":
        return IRWeighter(
            n_chunks=cfg.ir.n_chunks,
            use_rank=cfg.ir.use_rank,
            min_abs_ir=cfg.ir.min_abs_ir,
        )
    if cfg.type == "equal":
        return EqualWeighter()
    if cfg.type == "lightgbm":
        raise NotImplementedError(
            "weighter.type='lightgbm' arrives in PR-B2 Task 4 "
            "(LightGBMWeighter not implemented yet)"
        )
    raise ValueError(f"unknown weighter type: {cfg.type!r}")
```

- [ ] **Step 5: Migrate `config.yaml`**

Find the `weighter` block under `strategy.ml_factor` (currently flat: `weighter: {type: ic, use_rank: true, min_abs_ic: 0.0, n_chunks: 6, min_abs_ir: 0.0}`). REPLACE with:

```yaml
    weighter:
      type: ic                # ic | ir | equal | lightgbm (PR-B2 Task 4 改默认)
      ic:
        use_rank: true        # rank IC (Spearman) 更稳健
        min_abs_ic: 0.0
      ir:
        n_chunks: 6
        use_rank: true
        min_abs_ir: 0.0
      # equal 子段无超参,不写就用 EqualWeighterConfig() 默认空块
      # lightgbm:             # PR-B2 Task 4 启用时可调超参,默认见 LightGBMWeighterConfig
      #   num_leaves: 15
      #   min_data_in_leaf: 20
      #   learning_rate: 0.05
      #   num_iterations: 200
      #   max_depth: 4
      #   random_state: 42
```

- [ ] **Step 6: Verify config loads**

```bash
./.venv/Scripts/python.exe -c "
from stockpool.config import load_config
cfg = load_config('config.yaml')
w = cfg.strategy.ml_factor.weighter
print('type:', w.type)
print('ic.use_rank:', w.ic.use_rank)
print('ir.n_chunks:', w.ir.n_chunks)
print('lightgbm.num_leaves:', w.lightgbm.num_leaves)
"
```
Expected:
```
type: ic
ic.use_rank: True
ir.n_chunks: 6
lightgbm.num_leaves: 15
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k "weighter" --no-header
```
Expected: 10 PASS.

- [ ] **Step 8: Run full test suite for regression**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q --no-header 2>&1 | tail -3
```
Expected: All 277 + 10 = 287 pass. The default `type="ic"` and `ICWeighterConfig` defaults match the old flat behavior, so legacy tests should be unaffected.

- [ ] **Step 9: Commit**

```bash
git status
git add src/stockpool/config.py src/stockpool/backtesting/strategies.py config.yaml tests/test_config.py
git diff --staged --stat
git commit -m "feat(config): WeighterConfig subnested refactor (ic/ir/equal/lightgbm)

Mirrors PR-A's SelectorConfig structure. Old flat fields (use_rank,
min_abs_ic, n_chunks, min_abs_ir on WeighterConfig top level) are now
under per-type subblocks; extra='forbid' rejects them.

Default type='ic' kept here; Task 4 flips to 'lightgbm' after the new
weighter class exists. _build_weighter raises NotImplementedError on
type='lightgbm' for now.

User config.yaml migrated to nested form."
```

---

## Task 2: `contributions()` on `FactorWeighter` ABC + linear-weighter mixin + Pipeline simplification

**Files:**
- Modify: `src/stockpool/ml/weighters.py` (FactorWeighter ABC + IC/IR/Equal classes)
- Modify: `src/stockpool/ml/pipeline.py` (`TwoStepPipeline.contributions()`)

- [ ] **Step 1: Run existing pipeline tests as baseline**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_pipeline.py -v --no-header 2>&1 | tail -10
```
Note which tests pass currently — they MUST continue to pass after the refactor.

- [ ] **Step 2: Add `contributions()` abstract method to `FactorWeighter` ABC**

In `src/stockpool/ml/weighters.py`, find the `class FactorWeighter(ABC):` block (around line 17). REPLACE with:

```python
class FactorWeighter(ABC):
    """Assign a weight to each factor and produce a composite score."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def weights(self) -> pd.Series: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-bar per-factor contribution to ``predict(X)``.

        Linear weighters return ``standardised(X) * weights`` (row sums equal
        ``predict(X)`` by construction). Non-linear weighters (e.g. LightGBM)
        return their model-specific decomposition, e.g. SHAP values.
        """
```

- [ ] **Step 3: Add `_LinearWeighterContributionsMixin`**

Still in `src/stockpool/ml/weighters.py`, find the `class _StandardisingMixin:` block (around line 53). Insert a new mixin **after** it:

```python
class _LinearWeighterContributionsMixin:
    """Shared ``contributions()`` impl for linear-combination weighters
    (IC / IR / Equal). Returns ``standardised(X) * weights`` per cell.

    Depends on the class to provide:
      * ``self._weights`` — pd.Series of per-factor weights
      * ``self._feature_names`` — list[str] of fit-time feature names
      * ``self._apply_standardiser(X)`` — z-score apply method (from
        _StandardisingMixin)
    """

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._weights is None or self._weights.empty:
            return pd.DataFrame(index=X.index)
        Xs = self._apply_standardiser(X)
        w = self._weights.to_numpy()
        return pd.DataFrame(
            Xs * w, index=X.index, columns=self._feature_names,
        )
```

- [ ] **Step 4: Apply mixin to existing linear weighters**

In `src/stockpool/ml/weighters.py`, update the class declarations for `EqualWeighter`, `ICWeighter`, `IRWeighter` to include the new mixin. For each, change:

```python
class EqualWeighter(FactorWeighter, _StandardisingMixin):
```
to:
```python
class EqualWeighter(FactorWeighter, _StandardisingMixin, _LinearWeighterContributionsMixin):
```

Same pattern for `ICWeighter`:
```python
class ICWeighter(FactorWeighter, _StandardisingMixin, _LinearWeighterContributionsMixin):
```

And `IRWeighter`:
```python
class IRWeighter(FactorWeighter, _StandardisingMixin, _LinearWeighterContributionsMixin):
```

No method bodies change — the mixin provides `contributions()` and the FactorWeighter ABC's abstractmethod is satisfied via MRO.

- [ ] **Step 5: Simplify `TwoStepPipeline.contributions()` in `src/stockpool/ml/pipeline.py`**

Find the existing `def contributions(self, X: pd.DataFrame) -> pd.DataFrame:` (around line 83). REPLACE the method body:

```python
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-bar per-factor contribution. Delegates to ``weighter.contributions``.

        Row sums equal ``self.predict(X)`` for linear weighters; for non-linear
        weighters the convention is weighter-specific (see weighter docstring).
        """
        if self.fit_info_ is None:
            raise RuntimeError("Pipeline not fitted yet")
        selected = self.fit_info_.selected_factors
        if not selected:
            return pd.DataFrame(index=X.index)
        missing = [c for c in selected if c not in X.columns]
        if missing:
            raise KeyError(f"contributions() missing columns: {missing}")
        return self.weighter.contributions(X[selected])
```

- [ ] **Step 6: Run existing pipeline tests to verify zero regression**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_pipeline.py -v --no-header 2>&1 | tail -10
```
Expected: All previously-passing tests still pass — the math is unchanged, just moved.

- [ ] **Step 7: Run full ml/* tests**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_pipeline.py tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_embargo.py tests/test_ml_selector_lightgbm.py tests/test_ml_dataset_labels.py -q --no-header 2>&1 | tail -3
```
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git status
git add src/stockpool/ml/weighters.py src/stockpool/ml/pipeline.py
git diff --staged --stat
git commit -m "refactor(ml): move contributions() from TwoStepPipeline to FactorWeighter ABC

FactorWeighter ABC gains an abstract contributions(X) method. Linear
weighters (IC/IR/Equal) share a _LinearWeighterContributionsMixin that
provides the existing standardised(X) * weights logic. TwoStepPipeline
.contributions() collapses to a delegate to weighter.contributions().

This enables non-linear weighters (PR-B2 Task 3 LightGBMWeighter) to
plug in their own contribution semantics (SHAP) without Pipeline
needing isinstance dispatch."
```

---

## Task 3: Implement `LightGBMWeighter` class

**Files:**
- Modify: `src/stockpool/ml/weighters.py` (append the new class)
- Test: `tests/test_ml_weighter_lightgbm.py` (new)

- [ ] **Step 1: Write the test file**

Create `tests/test_ml_weighter_lightgbm.py`:

```python
"""LightGBMWeighter unit + integration tests (F2 PR-B2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LightGBMSelector
from stockpool.ml.weighters import LightGBMWeighter


def _linear_signal_xy(n: int = 500, n_signal: int = 3, n_noise: int = 2, seed: int = 0):
    """y = sum(x_i for i in range(n_signal)) + small noise."""
    rng = np.random.default_rng(seed)
    cols = {}
    y = np.zeros(n)
    for i in range(n_signal):
        col = rng.normal(0, 1, n)
        cols[f"sig{i}"] = col
        y += col
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.normal(0, 1, n)
    y += 0.1 * rng.normal(0, 1, n)
    return pd.DataFrame(cols), pd.Series(y)


def test_lightgbm_weighter_fit_predict_round_trip():
    """Trained LGB predicts y positively correlated on the training set."""
    X, y = _linear_signal_xy(n=500, seed=1)
    w = LightGBMWeighter(random_state=1)
    w.fit(X, y)
    preds = w.predict(X)
    assert len(preds) == len(X)
    corr = float(preds.corr(y, method="spearman"))
    assert corr > 0.3, f"expected Spearman corr > 0.3, got {corr}"


def test_lightgbm_weighter_weights_are_mean_abs_shap():
    """weights() returns mean|SHAP| — non-negative, sum > 0 in normal fit."""
    X, y = _linear_signal_xy(n=500, seed=2)
    w = LightGBMWeighter(random_state=2)
    w.fit(X, y)
    ws = w.weights()
    assert len(ws) == len(X.columns)
    assert (ws >= 0).all(), f"|SHAP| should be non-negative, got {ws}"
    assert ws.sum() > 0


def test_lightgbm_weighter_contributions_shape_and_columns():
    """contributions(X) returns DataFrame with row=X.index, col=fit-time features."""
    X, y = _linear_signal_xy(n=300, seed=3)
    w = LightGBMWeighter(random_state=3)
    w.fit(X, y)
    contribs = w.contributions(X)
    assert contribs.shape == X.shape
    assert list(contribs.columns) == list(X.columns)
    pd.testing.assert_index_equal(contribs.index, X.index)


def test_lightgbm_weighter_contributions_row_sums_track_predict():
    """SHAP convention: row sums + base_value ≈ predict. Strong correlation."""
    X, y = _linear_signal_xy(n=300, seed=4)
    w = LightGBMWeighter(random_state=4)
    w.fit(X, y)
    preds = w.predict(X)
    contribs = w.contributions(X)
    row_sums = contribs.sum(axis=1)
    corr = float(row_sums.corr(preds))
    assert corr > 0.95, f"expected row_sums ↔ predict corr > 0.95, got {corr}"


def test_lightgbm_weighter_deterministic_with_seed():
    X, y = _linear_signal_xy(n=400, seed=5)
    w1 = LightGBMWeighter(random_state=42)
    w2 = LightGBMWeighter(random_state=42)
    w1.fit(X, y); w2.fit(X, y)
    np.testing.assert_array_almost_equal(
        w1.predict(X).values, w2.predict(X).values,
    )


def test_lightgbm_weighter_empty_input():
    X = pd.DataFrame({"a": [], "b": []}, dtype=float)
    y = pd.Series([], dtype=float)
    w = LightGBMWeighter(random_state=6)
    w.fit(X, y)
    assert w.weights().empty
    preds = w.predict(X)
    assert len(preds) == 0


def test_lightgbm_weighter_predict_missing_columns_raises():
    X, y = _linear_signal_xy(n=200, seed=7)
    w = LightGBMWeighter(random_state=7)
    w.fit(X, y)
    X_missing = X.drop(columns=[X.columns[0]])
    with pytest.raises(KeyError):
        w.predict(X_missing)


def test_two_step_pipeline_lgb_selector_lgb_weighter():
    """Integration: LGB selector + LGB weighter end-to-end."""
    X, y = _linear_signal_xy(n=500, n_signal=3, n_noise=2, seed=8)
    pipeline = TwoStepPipeline(
        selector=LightGBMSelector(top_k_factors=3, random_state=8),
        weighter=LightGBMWeighter(random_state=8),
    )
    info = pipeline.fit(X, y)
    if info.selected_factors:
        preds = pipeline.predict(X)
        assert len(preds) == len(X)
        contribs = pipeline.contributions(X)
        assert contribs.shape == (len(X), len(info.selected_factors))
        assert list(contribs.columns) == info.selected_factors
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_weighter_lightgbm.py -v --no-header
```
Expected: 8 ImportError-style failures.

- [ ] **Step 3: Implement `LightGBMWeighter`**

Append to `src/stockpool/ml/weighters.py` (after `IRWeighter` class — keep all existing classes untouched):

```python
class LightGBMWeighter(FactorWeighter):
    """Tree-based weighter using LightGBM.

    ``fit(X, y)`` trains a regression LGB and caches mean|SHAP| as ``_weights``
    (computed once on training data, returned by ``weights()``).
    ``predict(X)`` runs ``booster.predict(X.values)``.
    ``contributions(X)`` runs ``booster.predict(X.values, pred_contrib=True)``
    and returns per-feature SHAP values (drops the trailing base-value column).

    Unlike linear weighters, this class does NOT inherit
    ``_StandardisingMixin`` — LightGBM is scale-invariant. Look-ahead safety
    rests on the same ABC contract: predict only consumes X, never y.

    Lazy import: ``import lightgbm`` happens inside ``fit()`` so the module
    can be imported without lightgbm installed.
    """

    def __init__(
        self,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        learning_rate: float = 0.05,
        num_iterations: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
        verbose: int = -1,
    ):
        if num_leaves <= 1:
            raise ValueError(f"num_leaves must be > 1, got {num_leaves}")
        if min_data_in_leaf <= 0:
            raise ValueError(f"min_data_in_leaf must be > 0, got {min_data_in_leaf}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be > 0, got {num_iterations}")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {max_depth}")

        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose

        self._booster = None
        self._feature_names: list[str] | None = None
        self._weights: pd.Series | None = None  # cached mean|SHAP|

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb  # lazy import — ImportError surfaces only at fit

        if X.empty or len(y) == 0:
            self._feature_names = list(X.columns)
            self._weights = pd.Series(dtype=float)
            self._booster = None
            return

        self._feature_names = list(X.columns)
        dataset = lgb.Dataset(
            X.values, label=y.values, feature_name=self._feature_names,
        )
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "seed": self.random_state,
            "verbose": self.verbose,
        }
        self._booster = lgb.train(
            params, dataset, num_boost_round=self.num_iterations,
        )

        # Cache mean|SHAP| as weights (per Q1+Q5 design decisions).
        # pred_contrib returns shape (n, n_features + 1); last col is base value.
        contribs = self._booster.predict(X.values, pred_contrib=True)
        feature_contribs = contribs[:, :-1]
        mean_abs = np.abs(feature_contribs).mean(axis=0)
        self._weights = pd.Series(
            mean_abs, index=self._feature_names, name="lgb_mean_abs_shap",
        )

    def weights(self) -> pd.Series:
        if self._weights is None:
            raise RuntimeError("Weighter not fitted yet")
        return self._weights.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._booster is None:
            return pd.Series(0.0, index=X.index)
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"predict() missing columns: {missing}")
        Xn = X[self._feature_names].values
        preds = self._booster.predict(Xn)
        return pd.Series(preds, index=X.index, name="score")

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._booster is None:
            return pd.DataFrame(index=X.index)
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"contributions() missing columns: {missing}")
        Xn = X[self._feature_names].values
        contribs = self._booster.predict(Xn, pred_contrib=True)
        return pd.DataFrame(
            contribs[:, :-1], index=X.index, columns=self._feature_names,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_weighter_lightgbm.py -v --no-header
```
Expected: 8 PASS.

- [ ] **Step 5: Run all weighter/pipeline tests**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_weighter_lightgbm.py tests/test_ml_pipeline.py tests/test_ml_selector_lightgbm.py -v --no-header 2>&1 | tail -5
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git status
git add src/stockpool/ml/weighters.py tests/test_ml_weighter_lightgbm.py
git diff --staged --stat
git commit -m "feat(ml): add LightGBMWeighter implementing FactorWeighter

Lazy-imports lightgbm in fit(). weights() returns mean|SHAP| cached at
fit time (sum > 0 in normal fit, empty Series in degenerate cases).
predict() returns booster output. contributions() returns SHAP values
(pred_contrib without base col).

LightGBMWeighter does NOT inherit _StandardisingMixin — LGB is scale-
invariant. ABC's contributions() abstractmethod is satisfied by this
class's own SHAP impl (not the linear mixin)."
```

---

## Task 4: Flip default + wire factory + patch legacy fixtures

**Files:**
- Modify: `src/stockpool/config.py` (`WeighterConfig.type` default)
- Modify: `src/stockpool/backtesting/strategies.py` (`_build_weighter` + import)
- Modify: `tests/test_config.py` (the default-is-ic test from Task 1)
- Modify: `tests/test_ml_strategy.py`
- Modify: `tests/test_ml_strategy_panel.py`
- Modify: `tests/test_ml_strategy_embargo.py`

- [ ] **Step 1: Flip WeighterConfig default**

In `src/stockpool/config.py`, find `class WeighterConfig(BaseModel):` and change the `type` field default from `"ic"` to `"lightgbm"`:

```python
class WeighterConfig(BaseModel):
    # ... docstring stays ...
    model_config = ConfigDict(extra="forbid")
    type: Literal["ic", "ir", "equal", "lightgbm"] = "lightgbm"   # ← change default
    ic: ICWeighterConfig = Field(default_factory=ICWeighterConfig)
    ir: IRWeighterConfig = Field(default_factory=IRWeighterConfig)
    equal: EqualWeighterConfig = Field(default_factory=EqualWeighterConfig)
    lightgbm: LightGBMWeighterConfig = Field(default_factory=LightGBMWeighterConfig)
```

- [ ] **Step 2: Update the Task 1 placeholder test**

In `tests/test_config.py`, find `test_weighter_default_type_is_still_ic_in_task1` and update:

```python
def test_weighter_default_type_is_lightgbm():
    """Default weighter.type flips to 'lightgbm' in PR-B2 Task 4."""
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig()
    assert cfg.type == "lightgbm"
```

(Rename the function and change the assertion. Other 9 weighter tests stay.)

- [ ] **Step 3: Wire `_build_weighter` lightgbm case**

In `src/stockpool/backtesting/strategies.py`, find `_build_weighter`. REPLACE the body to fully support all 4 types:

```python
def _build_weighter(cfg) -> FactorWeighter:
    """Translate WeighterConfig → concrete FactorWeighter (PR-B2 subnested)."""
    if cfg.type == "ic":
        return ICWeighter(use_rank=cfg.ic.use_rank, min_abs_ic=cfg.ic.min_abs_ic)
    if cfg.type == "ir":
        return IRWeighter(
            n_chunks=cfg.ir.n_chunks,
            use_rank=cfg.ir.use_rank,
            min_abs_ir=cfg.ir.min_abs_ir,
        )
    if cfg.type == "equal":
        return EqualWeighter()
    if cfg.type == "lightgbm":
        c = cfg.lightgbm
        return LightGBMWeighter(
            num_leaves=c.num_leaves,
            min_data_in_leaf=c.min_data_in_leaf,
            learning_rate=c.learning_rate,
            num_iterations=c.num_iterations,
            max_depth=c.max_depth,
            random_state=c.random_state,
            verbose=c.verbose,
        )
    raise ValueError(f"unknown weighter type: {cfg.type!r}")
```

Also update the imports at the top. Find the existing `from stockpool.ml.weighters import` line and extend:

```python
from stockpool.ml.weighters import (
    EqualWeighter, FactorWeighter, ICWeighter, IRWeighter, LightGBMWeighter,
)
```

(Keep any other names that were there.)

- [ ] **Step 4: Patch `tests/test_ml_strategy.py`**

Top-level import already has `WeighterConfig` (PR-B1 Task 5 didn't add it but `SelectorConfig` is there from PR-B1). Confirm + extend:

```bash
grep -n "from stockpool.config import" tests/test_ml_strategy.py
```

If `WeighterConfig` is missing, add it:
```python
from stockpool.config import MLFactorConfig, QuantileThresholds, SelectorConfig, WeighterConfig
```

(WeighterConfig was already used in test_ml_strategy.py:215 per pre-existing code — so import should be there. Verify.)

For every `MLFactorConfig(...)` call (10 sites already have `selector=SelectorConfig(type="lasso")` from PR-B1 Task 5), append `weighter=WeighterConfig(type="ic")` as another kwarg.

**Special case at line 215**: that call already has `weighter=WeighterConfig(type="equal")` — do NOT replace it. Skip that one.

So you're patching 9 of the 10 sites (the one with explicit `weighter=WeighterConfig(type="equal")` keeps that).

Transformation example:

```python
# Before (after PR-B1):
cfg = MLFactorConfig(
    train_window=120, refit_every=20, min_train_samples=60,
    embargo_days=0, selector=SelectorConfig(type="lasso"),
)
# After:
cfg = MLFactorConfig(
    train_window=120, refit_every=20, min_train_samples=60,
    embargo_days=0,
    selector=SelectorConfig(type="lasso"),
    weighter=WeighterConfig(type="ic"),
)
```

- [ ] **Step 5: Patch `tests/test_ml_strategy_panel.py`**

```bash
grep -n "MLFactorConfig(\|from stockpool.config import" tests/test_ml_strategy_panel.py
```

For each local `from stockpool.config import MLFactorConfig, SelectorConfig` line, extend with `WeighterConfig`:
```python
from stockpool.config import MLFactorConfig, SelectorConfig, WeighterConfig
```

For each `MLFactorConfig(...)` call (3 sites), append `weighter=WeighterConfig(type="ic")`.

- [ ] **Step 6: Patch `tests/test_ml_strategy_embargo.py`**

```bash
grep -n "MLFactorConfig(\|def test_\|from stockpool.config import" tests/test_ml_strategy_embargo.py
```

Extend the top-level import:
```python
from stockpool.config import MLFactorConfig, SelectorConfig, WeighterConfig
```

For only the **3** `_try_fit`-exercising fixtures (the ones already patched with `selector=SelectorConfig(type="lasso")` in PR-B1 Task 5), append `weighter=WeighterConfig(type="ic")`:
- `test_refit_with_default_embargo_returns_none_when_insufficient_history`
- `test_refit_with_legacy_no_embargo_runs_to_completion`
- `test_refit_with_default_embargo_long_history_also_runs_to_completion`

Leave the 8 helper-only fixtures untouched.

- [ ] **Step 7: Run all affected tests**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_embargo.py tests/test_config.py -v --no-header 2>&1 | tail -10
```
Expected: All PASS.

- [ ] **Step 8: Run full test suite**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q --no-header 2>&1 | tail -3
```
Expected: All 287 + 8 = 295 pass. (Task 1 added 10 config tests, Task 3 added 8 weighter tests, Task 4 renamed 1 test but didn't add — wait, Task 1 added the 10 then Task 4 modifies 1 in-place; Task 3 added 8 → net 277 + 10 + 8 = 295.)

- [ ] **Step 9: Commit**

```bash
git status
git add src/stockpool/config.py src/stockpool/backtesting/strategies.py tests/test_config.py tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_embargo.py
git diff --staged --stat
git commit -m "feat(strategy): flip weighter.type default to lightgbm + wire factory

WeighterConfig.type default flips from 'ic' to 'lightgbm' now that
LightGBMWeighter exists. _build_weighter gains the lightgbm branch.
Legacy fit-exercising fixtures opt out via weighter=WeighterConfig(
type='ic') to preserve pre-PR-B2 IC numerics. test_ml_strategy.py
fixture at line ~215 with explicit weighter=type='equal' is unchanged."
```

---

## Task 5: Sync CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md module map**

Find the ML module map row mentioning "Lasso 或 LightGBM selector / IC&IR weighter" (added in PR-B1). Replace its description text with:

```
**两步法 ML 组合**(dataset / Lasso 或 LightGBM selector / IC&IR&Equal&LightGBM weighter / TwoStepPipeline)
```

Locate with `grep -n "两步法 ML\|Lasso 或 LightGBM" CLAUDE.md`.

- [ ] **Step 2: Update CLAUDE.md strategy config description**

Find the `**strategy** —` bullet in the 配置 section. Locate the `**selector.{lasso|lightgbm}**` and the existing `weighter` mentions. Update around `weighter` to read:

```
... **`weighter.{ic|ir|equal|lightgbm}`**(F2 PR-B2 子段化):`type` 默认 `"lightgbm"`,`ic.{use_rank,min_abs_ic}` / `ir.{n_chunks,use_rank,min_abs_ir}` / `equal` (无参) / `lightgbm.{num_leaves,min_data_in_leaf,learning_rate,num_iterations,max_depth,random_state}` 子段四选一,顶层扁平字段被 Pydantic 拒绝。 ...
```

If the original bullet doesn't separately mention `weighter`, append this segment at the end of the strategy bullet.

- [ ] **Step 3: Update CLAUDE.md 测试 table**

Append after the PR-B1 row:

```markdown
| `test_ml_weighter_lightgbm.py` | LightGBMWeighter: fit→predict 通 / mean&#124;SHAP&#124; weights / SHAP contributions 行和接近 predict / 确定性 / 退化输入 / TwoStepPipeline 集成 |
```

- [ ] **Step 4: Update README.md LGB caveat section**

Open `README.md`. Find the "关于 LightGBM 默认 selector" section added in PR-B1. After it, append:

```markdown
**F2 PR-B2 起,`weighter.type` 默认也是 `"lightgbm"`**,完成完全非线性两步法。weighter 与 selector 各训练一次 LGB,refit 训练时间约为 PR-B1 的 1.8-2.2 倍。

**A/B 对照**:想回到 PR-B1 的"LGB selector + IC 加权"baseline 做对照,YAML 改一行 `weighter.type: ic` 即可(`weighter.ic.use_rank: true` 是默认值,可不写)。

**关于 `weighter.contributions()`**:在 LGB weighter 下,返回的是 SHAP 值(每行每因子的边际贡献);在 IC/IR/Equal 等线性 weighter 下,返回 `standardised(X) * weights`。两者形状一致(行 = 样本,列 = 因子),但 LGB 行和 ≈ `predict(X) - base_value`(SHAP convention)而非完全等于 `predict(X)`。
```

- [ ] **Step 5: Verify markdown renders**

```bash
grep -A 5 "F2 PR-B2 起" README.md
grep -A 2 "test_ml_weighter_lightgbm" CLAUDE.md
```

- [ ] **Step 6: Commit**

```bash
git status
git add CLAUDE.md README.md
git diff --staged --stat
git commit -m "docs: F2 PR-B2 — LightGBM weighter default + contributions polymorphism

CLAUDE.md updates module map and weighter.{ic|ir|equal|lightgbm}
subnesting description. README.md extends the PR-B1 LGB section to
cover the weighter default flip, A/B fallback to IC, and the SHAP
semantics of contributions() under LGB."
```

---

## Task 6: Final regression + spec acceptance

**Files:** (none modified)

- [ ] **Step 1: Run full test suite**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q --no-header 2>&1 | tail -3
```
Expected: 295 PASS.

- [ ] **Step 2: Manual config smoke**

```bash
./.venv/Scripts/python.exe -c "
from stockpool.config import load_config
cfg = load_config('config.yaml')
sel = cfg.strategy.ml_factor.selector
w = cfg.strategy.ml_factor.weighter
print('selector.type:', sel.type)
print('selector.lightgbm.num_leaves:', sel.lightgbm.num_leaves)
print('weighter.type:', w.type)
print('weighter.ic.use_rank:', w.ic.use_rank)
print('weighter.lightgbm.num_leaves:', w.lightgbm.num_leaves)
"
```
Expected output:
```
selector.type: lasso
selector.lightgbm.num_leaves: 15
weighter.type: ic
weighter.ic.use_rank: True
weighter.lightgbm.num_leaves: 15
```

(User's config.yaml has explicit `selector.type: lasso` and `weighter.type: ic` from PR-A/PR-B1/PR-B2 migrations — so their config still produces the conservative Lasso+IC pipeline. Code default is LGB+LGB.)

- [ ] **Step 3: Spec acceptance criteria check**

Re-read `docs/superpowers/specs/2026-05-23-f2-pr-b2-lightgbm-weighter-design.md` §5. For each item, find the implementing test/step:

| # | Criterion | Where |
|---|-----------|-------|
| 1 | 零回归 — all tests pass | Step 1 above (295 pass) |
| 2 | LGB weighter fit-predict 通 — Spearman > 0.3 | `test_lightgbm_weighter_fit_predict_round_trip` |
| 3 | SHAP 行和 ≈ predict — corr > 0.95 | `test_lightgbm_weighter_contributions_row_sums_track_predict` |
| 4 | WeighterConfig 硬切 — flat fields rejected | `test_weighter_flat_use_rank_rejected` |
| 5 | 可切换 — `weighter.type: ic` 一行 YAML 回到 PR-B1 baseline | Config smoke (Step 2) confirms |
| 6 | `contributions()` 多态生效 — no isinstance in Pipeline | Source review of `TwoStepPipeline.contributions()` (1-line delegate) |
| 7 | 缓存失效自然 — sig changes via `_strategy_signature` | Implicit (PR-A mechanism extends naturally) |
| 8 | docs 同步 | Task 5 commit |

Report PASS/FAIL for each.

- [ ] **Step 4: Optional smoke — LGB end-to-end on local cache**

If `data/<stock>_daily.parquet` files exist (user previously ran `fetch-universe` or `run`), exercise the LGB+LGB path:

a. Temporarily edit `config.yaml` and flip:
```yaml
strategy:
  ml_factor:
    selector:
      type: lightgbm    # was: lasso
    weighter:
      type: lightgbm    # was: ic
```

b. Run:
```bash
./.venv/Scripts/python.exe -m stockpool run --skip-trading-day-check 2>&1 | tail -10
```
Expected: runs without error; `data/ml_models/<new_sig>_*.pkl` files appear; new HTML report generated.

c. **Restore `config.yaml`** to PR-A/B1/B2-migrated state (selector.type: lasso, weighter.type: ic) — do NOT commit the type flip. Leaving baselines preserves test numerics for future PRs.

If no cache, skip this step.

- [ ] **Step 5: Confirm clean tree**

```bash
git status
```
Expected: clean.

---

## Self-Review Notes

**Spec coverage** (against spec §3 / §5):

- ✅ §3.1 WeighterConfig subnesting (ic/ir/equal/lightgbm + extra="forbid") — Task 1
- ✅ §3.2 contributions() abstract on FactorWeighter ABC + mixin + Pipeline simplification — Task 2
- ✅ §3.3 LightGBMWeighter class — Task 3
- ✅ §3.4 _build_weighter factory subnested + lightgbm — Tasks 1 (subnested without lgb) + 4 (lgb branch + default flip)
- ✅ §3.5 legacy fixture patches — Task 4
- ✅ §3.6 8 LightGBMWeighter tests — Task 3
- ✅ §3.7 10 WeighterConfig tests — Task 1
- ✅ §3.8 user YAML migration — Task 1
- ✅ §3.10 cache invalidation — implicit
- ✅ §3.11 docs sync — Task 5
- ✅ §3.12 LGB selector/weighter independent training — covered by spec, no code needed
- ✅ §5 acceptance — Task 6

**Placeholders**: none.

**Type consistency**:
- `LightGBMWeighter.__init__(num_leaves, min_data_in_leaf, learning_rate, num_iterations, max_depth, random_state, verbose)` — 7 params, no `top_k_factors`/`min_importance_ratio` (those are selector-specific). Matches Task 3 impl, Task 4 factory call, Task 3 tests.
- `LightGBMWeighterConfig` (Task 1) has the same 7 fields. Factory in Task 4 unpacks them one-to-one.
- `FactorWeighter.contributions(X) -> pd.DataFrame` — abstract method (Task 2), implemented by mixin for linear weighters (Task 2) and by LightGBMWeighter directly (Task 3). `TwoStepPipeline.contributions()` delegates (Task 2).
- `WeighterConfig` subfield names (`ic`, `ir`, `equal`, `lightgbm`) consistent across config (Task 1), factory (Tasks 1 + 4), and tests (Task 1).
