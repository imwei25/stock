# F2 PR-B1 — LightGBM Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship F2 PR-B1 from spec `docs/superpowers/specs/2026-05-23-f2-pr-b1-lightgbm-selector-design.md`: add a LightGBM-based factor selector, make it the new default, switch `lightgbm` to a required dependency, and preserve legacy numerics by pinning existing tests to `selector.type="lasso"`.

**Architecture:** New `LightGBMSelector` implements the existing `FactorSelector` ABC. `SelectorConfig` gains a `lightgbm: LightGBMSelectorConfig` subfield (parallel to `lasso`) and default `type` flips to `"lightgbm"`. `MLFactorStrategy._try_fit` switches from inline `LassoSelector(...)` to a new `_build_selector(cfg.selector)` factory that dispatches on `cfg.type`. Weighter path stays IC (unchanged from PR-A).

**Tech Stack:** Python 3.10+, lightgbm 4.x, Pydantic, pandas, numpy, pytest. New required dep: `lightgbm>=4.0`.

---

## File Structure

| File | Change |
|------|--------|
| `pyproject.toml` | Add `lightgbm>=4.0` to `dependencies` |
| `src/stockpool/config.py` | New `LightGBMSelectorConfig`; `SelectorConfig.type` Literal extends to `["lasso", "lightgbm"]` (default `"lightgbm"`); `SelectorConfig.lightgbm` subfield |
| `src/stockpool/ml/selectors.py` | New `LightGBMSelector(FactorSelector)` class |
| `src/stockpool/backtesting/strategies.py` | New `_build_selector(cfg)` module-level factory; `_try_fit` calls it instead of inlining `LassoSelector(...)` |
| `tests/test_config.py` | 5 new tests for `SelectorConfig.lightgbm` parsing + default + extra-rejection |
| `tests/test_ml_selector_lightgbm.py` (new) | 8 unit + integration tests for `LightGBMSelector` |
| `tests/test_ml_strategy.py` | Inject `selector=SelectorConfig(type="lasso")` into all `MLFactorConfig(...)` fixtures |
| `tests/test_ml_strategy_panel.py` | Same |
| `tests/test_ml_strategy_embargo.py` | Inject `selector=SelectorConfig(type="lasso")` into the 3 `MLFactorConfig(...)` fixtures that exercise `_try_fit` |
| `CLAUDE.md` | Add `LightGBMSelector` to module map; update `strategy.ml_factor` config description |
| `README.md` | Add LGB overfitting caveat + fallback section |

---

## Task 1: Add `lightgbm>=4.0` to dependencies

**Files:**
- Modify: `pyproject.toml:10-20`

- [ ] **Step 1: Edit `pyproject.toml`**

In `pyproject.toml`, find the `dependencies = [...]` array. Add `"lightgbm>=4.0"` as a new line (alphabetical order — place after `"baostock>=0.8"`):

```toml
dependencies = [
    "akshare>=1.12",
    "mootdx>=0.11",
    "baostock>=0.8",
    "lightgbm>=4.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "pyarrow>=14.0",
    "pyecharts>=2.0",
    "pyyaml>=6.0",
    "pydantic>=2.5",
]
```

- [ ] **Step 2: Install the dependency into the editable env**

Run:
```bash
./.venv/Scripts/python.exe -m pip install "lightgbm>=4.0"
```
Expected: lightgbm 4.x wheel installs without error (~6 MB on Windows).

- [ ] **Step 3: Verify import works**

Run:
```bash
./.venv/Scripts/python.exe -c "import lightgbm; print('lightgbm', lightgbm.__version__)"
```
Expected output: `lightgbm 4.X.Y` (some 4.x release).

- [ ] **Step 4: Run existing test suite to confirm zero regression**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -3
```
Expected: All 264 (or whatever the current count is) tests still pass — installing lightgbm doesn't change behavior because nothing imports it yet.

- [ ] **Step 5: Commit**

```bash
git status
git add pyproject.toml
git diff --staged --stat
git commit -m "build: add lightgbm>=4.0 to required dependencies

Required by F2 PR-B1 LightGBMSelector. Native wheel (~6 MB on Windows).
Not yet imported anywhere — install-time only at this commit."
```

---

## Task 2: `LightGBMSelectorConfig` + extend `SelectorConfig`

**Files:**
- Modify: `src/stockpool/config.py` (LassoConfig section, around line 142)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
def test_selector_default_type_is_lightgbm():
    """Default selector.type flips to 'lightgbm' in PR-B1."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig()
    assert cfg.type == "lightgbm"


def test_selector_lightgbm_subcfg_explicit():
    """selector.lightgbm.num_leaves and friends parse from YAML."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig.model_validate({
        "type": "lightgbm",
        "lightgbm": {
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.1,
            "num_iterations": 100,
            "max_depth": 6,
            "random_state": 7,
            "top_k_factors": 10,
            "min_importance_ratio": 0.05,
            "verbose": 0,
        },
    })
    assert cfg.type == "lightgbm"
    assert cfg.lightgbm.num_leaves == 31
    assert cfg.lightgbm.min_data_in_leaf == 50
    assert cfg.lightgbm.learning_rate == 0.1
    assert cfg.lightgbm.num_iterations == 100
    assert cfg.lightgbm.max_depth == 6
    assert cfg.lightgbm.random_state == 7
    assert cfg.lightgbm.top_k_factors == 10
    assert cfg.lightgbm.min_importance_ratio == 0.05
    assert cfg.lightgbm.verbose == 0


def test_selector_lightgbm_subcfg_defaults():
    """LightGBMSelectorConfig defaults match spec section 3.2."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig.model_validate({"type": "lightgbm"})
    assert cfg.lightgbm.num_leaves == 15
    assert cfg.lightgbm.min_data_in_leaf == 20
    assert cfg.lightgbm.learning_rate == 0.05
    assert cfg.lightgbm.num_iterations == 200
    assert cfg.lightgbm.max_depth == 4
    assert cfg.lightgbm.random_state == 42
    assert cfg.lightgbm.top_k_factors == 20
    assert cfg.lightgbm.min_importance_ratio == 0.01
    assert cfg.lightgbm.verbose == -1


def test_selector_lightgbm_flat_num_leaves_rejected():
    """Flat num_leaves at SelectorConfig level is rejected (extra='forbid')."""
    import pydantic
    from stockpool.config import SelectorConfig
    with pytest.raises(pydantic.ValidationError):
        SelectorConfig.model_validate({"type": "lightgbm", "num_leaves": 31})


def test_selector_unknown_type_rejected():
    """type='xgboost' is not in Literal['lasso','lightgbm'] → reject."""
    import pydantic
    from stockpool.config import SelectorConfig
    with pytest.raises(pydantic.ValidationError):
        SelectorConfig.model_validate({"type": "xgboost"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k "selector_default_type_is_lightgbm or selector_lightgbm or selector_unknown" --no-header
```
Expected: 5 FAILs — `assert cfg.type == "lightgbm"` fails because current default is `"lasso"`; `lightgbm` attribute doesn't exist yet.

- [ ] **Step 3: Update `SelectorConfig` in `src/stockpool/config.py`**

Find the `class LassoConfig(BaseModel):` and `class SelectorConfig(BaseModel):` block (PR-A introduced these around line 142). Replace `SelectorConfig` and insert `LightGBMSelectorConfig` immediately before it. Final state:

```python
class LassoConfig(BaseModel):
    """Lasso-specific hyperparameters for ``selector.type == 'lasso'``.

    ``alpha`` is the L1 penalty on standardised features (typical range 1e-4 — 1e-1).
    """
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(default=0.001, ge=0.0)
    max_iter: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-6, gt=0.0)


class LightGBMSelectorConfig(BaseModel):
    """LightGBM-based selector hyperparameters.

    Defaults are conservative for walk-forward training (small per-refit
    training set; the embedded forest is intentionally shallow). Tighten
    ``num_leaves`` / increase ``min_data_in_leaf`` if observed IC is unstable
    across refits.
    """
    model_config = ConfigDict(extra="forbid")
    num_leaves: int = Field(default=15, gt=1)
    min_data_in_leaf: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_iterations: int = Field(default=200, gt=0)
    max_depth: int = Field(default=4, gt=0)
    random_state: int = Field(default=42, ge=0)
    top_k_factors: int = Field(default=20, gt=0)
    min_importance_ratio: float = Field(default=0.01, ge=0, le=1)
    verbose: int = Field(default=-1)


class SelectorConfig(BaseModel):
    """Step-1 (factor selection) settings.

    PR-A introduced ``selector.lasso.*`` subnesting. PR-B1 adds
    ``selector.lightgbm.*`` as a parallel block and flips the default
    ``type`` to ``"lightgbm"``.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["lasso", "lightgbm"] = "lightgbm"
    lasso: LassoConfig = Field(default_factory=LassoConfig)
    lightgbm: LightGBMSelectorConfig = Field(default_factory=LightGBMSelectorConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k "selector" --no-header
```
Expected: ALL pass (the 3 from PR-A and the 5 new ones).

- [ ] **Step 5: Run full config tests for regression**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_config.py -q --no-header 2>&1 | tail -3
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git status
git add src/stockpool/config.py tests/test_config.py
git diff --staged --stat
git commit -m "feat(config): add LightGBMSelectorConfig + default selector.type=lightgbm

SelectorConfig.type now Literal['lasso','lightgbm'] (default lightgbm).
Hyperparameters live under selector.lightgbm (extra='forbid' on both
SelectorConfig and LightGBMSelectorConfig). Defaults are conservative
for walk-forward stability."
```

---

## Task 3: Implement `LightGBMSelector` class

**Files:**
- Modify: `src/stockpool/ml/selectors.py`
- Test: `tests/test_ml_selector_lightgbm.py` (new)

- [ ] **Step 1: Write the test file**

Create `tests/test_ml_selector_lightgbm.py`:

```python
"""LightGBMSelector unit + integration tests (F2 PR-B1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LassoSelector, LightGBMSelector
from stockpool.ml.weighters import ICWeighter


def _nonlinear_xy(n: int = 500, seed: int = 0):
    """y = x0 * sign(x1) + small noise; x2/x3/x4 are pure noise.

    x0 is a linear main effect (Lasso can find it);
    x1 modulates x0 via sign → non-linear interaction (LGB can find it).
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    x4 = rng.normal(0, 1, n)
    y = x0 * np.sign(x1) + 0.1 * rng.normal(0, 1, n)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4})
    return X, pd.Series(y)


def _linear_signal_xy(n: int = 400, n_signal: int = 5, n_noise: int = 0, seed: int = 0):
    """y = sum(x_i for i in range(n_signal)) + small noise.

    Used when we want every signal factor to have meaningful importance.
    """
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


def test_lightgbm_selector_picks_nonlinear_features():
    """LGB finds the x0 main effect AND the x1 sign-modulator."""
    X, y = _nonlinear_xy(n=800, seed=1)
    sel = LightGBMSelector(top_k_factors=2, min_importance_ratio=0.01, random_state=1)
    sel.fit(X, y)
    picked = sel.selected_factors()
    assert "x0" in picked, f"expected x0 in selection, got {picked}"
    assert "x1" in picked, f"expected x1 in selection, got {picked}"


def test_lightgbm_selector_top_k_truncates():
    """top_k_factors=2 → exactly 2 selected when all 5 factors have signal."""
    X, y = _linear_signal_xy(n=500, n_signal=5, seed=2)
    sel = LightGBMSelector(top_k_factors=2, min_importance_ratio=0.0, random_state=2)
    sel.fit(X, y)
    assert len(sel.selected_factors()) == 2


def test_lightgbm_selector_min_importance_filter():
    """Tight ratio (0.99) keeps only the single strongest factor."""
    X, y = _linear_signal_xy(n=500, n_signal=5, seed=3)
    sel = LightGBMSelector(top_k_factors=10, min_importance_ratio=0.99, random_state=3)
    sel.fit(X, y)
    # max factor always survives (ratio 1.0 == max); others must clear 0.99
    assert len(sel.selected_factors()) <= 1


def test_lightgbm_selector_deterministic_with_seed():
    """Same data + same random_state → identical selection."""
    X, y = _nonlinear_xy(n=500, seed=4)
    sel1 = LightGBMSelector(random_state=42)
    sel2 = LightGBMSelector(random_state=42)
    sel1.fit(X, y)
    sel2.fit(X, y)
    assert sel1.selected_factors() == sel2.selected_factors()


def test_lightgbm_selector_coef_normalized():
    """coef_ sums to ~1.0 in non-degenerate case (importance normalized)."""
    X, y = _linear_signal_xy(n=500, n_signal=4, seed=5)
    sel = LightGBMSelector(random_state=5)
    sel.fit(X, y)
    assert sel.coef_ is not None
    total = float(sel.coef_.sum())
    assert abs(total - 1.0) < 1e-6, f"expected sum ≈ 1.0, got {total}"


def test_lightgbm_selector_empty_when_y_constant():
    """Constant y → 0 gain on every split → empty selection."""
    X = pd.DataFrame({
        "a": np.linspace(0, 1, 50),
        "b": np.linspace(1, 0, 50),
        "c": np.random.default_rng(0).normal(0, 1, 50),
    })
    y = pd.Series([1.0] * 50)
    sel = LightGBMSelector(random_state=6)
    sel.fit(X, y)
    assert sel.selected_factors() == []


def test_lightgbm_selector_empty_input():
    """Empty X → empty selection (no crash)."""
    X = pd.DataFrame({"a": [], "b": []}, dtype=float)
    y = pd.Series([], dtype=float)
    sel = LightGBMSelector(random_state=7)
    sel.fit(X, y)
    assert sel.selected_factors() == []


def test_two_step_pipeline_with_lgb_selector_and_ic_weighter():
    """Integration: TwoStepPipeline(LGB selector + IC weighter) fit→predict round-trip."""
    X, y = _linear_signal_xy(n=500, n_signal=3, n_noise=2, seed=8)
    pipeline = TwoStepPipeline(
        selector=LightGBMSelector(top_k_factors=3, random_state=8),
        weighter=ICWeighter(use_rank=True),
    )
    info = pipeline.fit(X, y)
    assert len(info.selected_factors) <= 3
    if info.selected_factors:
        preds = pipeline.predict(X)
        assert len(preds) == len(X)
        # predictions should correlate positively with y (signal-rich data)
        corr = float(preds.corr(y, method="spearman"))
        assert corr > 0.1, f"expected positive Spearman corr, got {corr}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_selector_lightgbm.py -v --no-header
```
Expected: ALL 8 FAILs — `ImportError: cannot import name 'LightGBMSelector' from 'stockpool.ml.selectors'`.

- [ ] **Step 3: Implement `LightGBMSelector`**

Append to `src/stockpool/ml/selectors.py` (after the existing `LassoSelector` class — keep `LassoSelector` untouched):

```python
class LightGBMSelector(FactorSelector):
    """Tree-based selector using LightGBM gain importance.

    ``fit()`` trains a regression LightGBM on (X, y);
    ``selected_factors()`` returns columns whose normalized gain importance
    is in the top-K AND >= ``max_importance * min_importance_ratio``.

    Attributes after ``fit``:
      * ``coef_``: pd.Series of normalized gain importance (sums to 1.0 in
        non-degenerate fits; sums to 0 when y is constant or empty).
      * ``selected_``: list of factor names that passed the top-K + ratio gate.

    Lazy import: ``import lightgbm`` happens inside ``fit`` so this module
    can be imported without lightgbm installed (only fitting requires it).
    """

    def __init__(
        self,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        learning_rate: float = 0.05,
        num_iterations: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
        top_k_factors: int = 20,
        min_importance_ratio: float = 0.01,
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
        if top_k_factors <= 0:
            raise ValueError(f"top_k_factors must be > 0, got {top_k_factors}")
        if not (0 <= min_importance_ratio <= 1):
            raise ValueError(
                f"min_importance_ratio must be in [0, 1], got {min_importance_ratio}"
            )

        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.random_state = random_state
        self.top_k_factors = top_k_factors
        self.min_importance_ratio = min_importance_ratio
        self.verbose = verbose

        self.coef_: pd.Series | None = None
        self.selected_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb  # lazy import — ImportError surfaces only at fit

        if X.empty or len(y) == 0:
            self.coef_ = pd.Series(dtype=float)
            self.selected_ = []
            return

        feature_names = list(X.columns)
        dataset = lgb.Dataset(
            X.values, label=y.values, feature_name=feature_names,
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
        booster = lgb.train(params, dataset, num_boost_round=self.num_iterations)
        gain = booster.feature_importance(importance_type="gain").astype(float)

        total = float(gain.sum())
        if total < 1e-12:
            # Constant y or no learnable signal → no selection.
            self.coef_ = pd.Series(0.0, index=feature_names, name="lgb_importance")
            self.selected_ = []
            return

        norm = gain / total
        self.coef_ = pd.Series(norm, index=feature_names, name="lgb_importance")

        max_val = float(self.coef_.max())
        threshold = max_val * self.min_importance_ratio
        ranked = self.coef_.sort_values(ascending=False)
        eligible = ranked[ranked >= threshold].head(self.top_k_factors)
        self.selected_ = list(eligible.index)

    def selected_factors(self) -> list[str]:
        return list(self.selected_)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_selector_lightgbm.py -v --no-header
```
Expected: 8 PASS.

- [ ] **Step 5: Run ml/* tests for regression**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_pipeline.py tests/test_ml_dataset_labels.py tests/test_ml_selector_lightgbm.py -v --no-header 2>&1 | tail -5
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git status
git add src/stockpool/ml/selectors.py tests/test_ml_selector_lightgbm.py
git diff --staged --stat
git commit -m "feat(ml): add LightGBMSelector implementing FactorSelector

Lazy-imports lightgbm in fit() so the module stays importable without
the dep. Selection rule: top_k_factors after gating by importance >=
max_importance * min_importance_ratio. coef_ stores normalized gain
importance (sum = 1 in non-degenerate fits, sum = 0 when y is constant
or X is empty) — compatible with TwoStepPipeline.FitInfo.coef shape."
```

---

## Task 4: `_build_selector` factory in `strategies.py`

**Files:**
- Modify: `src/stockpool/backtesting/strategies.py`

- [ ] **Step 1: Inspect current state**

Run:
```bash
grep -n "_build_weighter\|LassoSelector(" src/stockpool/backtesting/strategies.py
```
You should see:
- `def _build_weighter(...)` — the existing weighter factory pattern to copy
- `selector=LassoSelector(alpha=cfg.selector.lasso.alpha, ...)` — the inline call to replace

- [ ] **Step 2: Add the `_build_selector` factory**

In `src/stockpool/backtesting/strategies.py`, find the existing `_build_weighter` function (module-level, before `class MLFactorStrategy`). Insert `_build_selector` **immediately after `_build_weighter`** (module-level, parallel function):

```python
def _build_selector(cfg) -> FactorSelector:
    """Translate SelectorConfig → concrete FactorSelector."""
    if cfg.type == "lasso":
        return LassoSelector(
            alpha=cfg.lasso.alpha,
            max_iter=cfg.lasso.max_iter,
            tol=cfg.lasso.tol,
        )
    if cfg.type == "lightgbm":
        c = cfg.lightgbm
        return LightGBMSelector(
            num_leaves=c.num_leaves,
            min_data_in_leaf=c.min_data_in_leaf,
            learning_rate=c.learning_rate,
            num_iterations=c.num_iterations,
            max_depth=c.max_depth,
            random_state=c.random_state,
            top_k_factors=c.top_k_factors,
            min_importance_ratio=c.min_importance_ratio,
            verbose=c.verbose,
        )
    raise ValueError(f"unknown selector type: {cfg.type!r}")
```

You must also add the import at the top of `strategies.py`. Find the existing `from stockpool.ml.selectors import LassoSelector` line (or `from stockpool.ml.selectors import ... LassoSelector`) and extend it:

```python
from stockpool.ml.selectors import FactorSelector, LassoSelector, LightGBMSelector
```

(Add `FactorSelector` if it's missing too — it's needed for the return-type annotation.)

- [ ] **Step 3: Wire `_try_fit` to use the factory**

In `MLFactorStrategy._try_fit`, find the block (PR-A introduced it around line 549):

```python
        pipeline = TwoStepPipeline(
            selector=LassoSelector(
                alpha=cfg.selector.lasso.alpha,
                max_iter=cfg.selector.lasso.max_iter,
                tol=cfg.selector.lasso.tol,
            ),
            weighter=_build_weighter(cfg.weighter),
        )
```

Replace the `selector=...` keyword argument so the whole block becomes:

```python
        pipeline = TwoStepPipeline(
            selector=_build_selector(cfg.selector),
            weighter=_build_weighter(cfg.weighter),
        )
```

- [ ] **Step 4: Sanity-import**

Run:
```bash
./.venv/Scripts/python.exe -c "from stockpool.backtesting.strategies import _build_selector; print('ok')"
```
Expected: `ok`.

- [ ] **Step 5: Run all tests** (some will FAIL — Task 5 fixes them)

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/ -q --no-header 2>&1 | tail -5
```
Expected: Several failures in `test_ml_strategy.py`, `test_ml_strategy_panel.py`, `test_ml_strategy_embargo.py` (because the default selector is now LGB and fixture assertions may not match). Note the count. Task 5 patches.

- [ ] **Step 6: Commit**

```bash
git status
git add src/stockpool/backtesting/strategies.py
git diff --staged --stat
git commit -m "refactor(strategy): _build_selector factory dispatches on selector.type

Mirrors the existing _build_weighter pattern. _try_fit now resolves the
selector via the factory so LightGBMSelector can be selected via YAML
(selector.type=lightgbm). Legacy fixture tests will need to opt out of
the new default — that follows in the next commit."
```

---

## Task 5: Patch existing test fixtures with `selector=SelectorConfig(type="lasso")`

**Files:**
- Modify: `tests/test_ml_strategy.py`
- Modify: `tests/test_ml_strategy_panel.py`
- Modify: `tests/test_ml_strategy_embargo.py` (3 of the 11 fixtures)

- [ ] **Step 1: Enumerate failures**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_embargo.py -v --no-header 2>&1 | tail -30
```
Note which tests fail and the file/line refs.

- [ ] **Step 2: Inspect existing fixtures and imports**

```bash
grep -n "MLFactorConfig(" tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_embargo.py
grep -n "from stockpool.config import" tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_embargo.py
```

`SelectorConfig` is in `stockpool.config`. Each test file that uses `MLFactorConfig` must add `SelectorConfig` to the import.

- [ ] **Step 3: Patch `tests/test_ml_strategy.py`**

Update the import line. Find:
```python
from stockpool.config import MLFactorConfig, QuantileThresholds, WeighterConfig
```
Add `SelectorConfig`:
```python
from stockpool.config import MLFactorConfig, QuantileThresholds, SelectorConfig, WeighterConfig
```

For every `MLFactorConfig(...)` call (10 sites, lines 34, 43, 55, 65, 76, 92, 105, 123, 166, 185 — recheck with grep), add `selector=SelectorConfig(type="lasso")` as a kwarg.

Transformation examples:

```python
# Before:
cfg = MLFactorConfig(train_window=120, refit_every=20, min_train_samples=60, embargo_days=0)
# After:
cfg = MLFactorConfig(train_window=120, refit_every=20, min_train_samples=60,
                     embargo_days=0, selector=SelectorConfig(type="lasso"))
```

For multi-line constructors, add `selector=SelectorConfig(type="lasso")` on its own line within the kwargs.

- [ ] **Step 4: Patch `tests/test_ml_strategy_panel.py`**

The current pattern in `test_ml_strategy_panel.py` uses local imports inside test functions. Verify with:
```bash
grep -n "MLFactorConfig\|SelectorConfig" tests/test_ml_strategy_panel.py
```

For each `MLFactorConfig(...)` call (3 sites at lines 46, 73 in PR-A; recheck after PR-A added embargo_days), do **both**:

a. Add the `SelectorConfig` import to the corresponding local `from stockpool.config import MLFactorConfig` line, e.g.:
   ```python
   from stockpool.config import MLFactorConfig, SelectorConfig
   ```

b. Append `selector=SelectorConfig(type="lasso")` to the kwargs.

- [ ] **Step 5: Patch `tests/test_ml_strategy_embargo.py` (selective)**

This file has 11 `MLFactorConfig(...)` calls. Only patch the ones that call `_try_fit` (which actually trains a model). Helper-only tests don't need the patch.

**Patch these (3 fixtures)**:

| Line | Test |
|------|------|
| 52   | `test_refit_with_default_embargo_returns_none_when_insufficient_history` |
| 68   | `test_refit_with_legacy_no_embargo_runs_to_completion` |
| 87   | `test_refit_with_default_embargo_long_history_also_runs_to_completion` |

**Leave alone (8 fixtures)**:
- Lines 27, 33, 39, 45 (helper-only `_embargoed_label_end` tests)
- Lines 104, 108 (helper-only)
- Lines 128, 132 (helper-only)

Add the import too:
```python
from stockpool.config import MLFactorConfig, SelectorConfig
```

Verify line numbers with:
```bash
grep -n "MLFactorConfig(\|def test_" tests/test_ml_strategy_embargo.py
```

- [ ] **Step 6: Run targeted tests for green**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_embargo.py -v --no-header 2>&1 | tail -20
```
Expected: All PASS.

- [ ] **Step 7: Run full suite for regression**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q --no-header 2>&1 | tail -3
```
Expected: All pass — count is (previous total + 13) (5 from Task 2 + 8 from Task 3).

- [ ] **Step 8: Commit**

```bash
git status
git add tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_embargo.py
git diff --staged --stat
git commit -m "test(ml): pin legacy fixtures to selector=lasso for PR-B1 default flip

PR-B1 flips SelectorConfig.type default to 'lightgbm'. Existing tests
that exercise _try_fit need to opt out explicitly to preserve their
pre-PR-B1 numerics. Helper-only tests in test_ml_strategy_embargo.py
(those that only call _embargoed_label_end) are not touched."
```

---

## Task 6: Sync `CLAUDE.md` and `README.md`

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md` module map**

Open `CLAUDE.md`. Find the line in the 模块地图 section that mentions `ml/` and `Lasso selector`. Replace its description text from:

```
**两步法 ML 组合**(dataset / Lasso selector / IC&IR weighter / TwoStepPipeline)
```

with:

```
**两步法 ML 组合**(dataset / Lasso 或 LightGBM selector / IC&IR weighter / TwoStepPipeline)
```

- [ ] **Step 2: Update `CLAUDE.md` strategy config description**

Find the bullet beginning with `**`strategy`** —` in the 配置 section (PR-A introduced the long-form description). Update the `selector.lasso` mention to read `selector.{lasso|lightgbm}`. Concretely, find:

```
... **`selector.lasso`**(F2 PR-A 子段化):`alpha` / `max_iter` / `tol` 现在嵌在 `selector.lasso` 下,顶层扁平字段已被 Pydantic 拒绝;后续 PR-B 会加 `selector.lightgbm` 同级。 ...
```

Replace with:

```
... **`selector.{lasso|lightgbm}`**(F2 PR-A 子段化 + PR-B1 加 LGB):`type` 默认 `"lightgbm"`,`lasso.{alpha,max_iter,tol}` 或 `lightgbm.{num_leaves,min_data_in_leaf,learning_rate,num_iterations,max_depth,random_state,top_k_factors,min_importance_ratio}` 子段二选一,顶层扁平字段被 Pydantic 拒绝。改 `selector` 任一字段后旧 ml_models pkl 自动失效。 ...
```

If the exact original string differs, just update around `selector.lasso` to mention both types and `type` default `"lightgbm"`.

- [ ] **Step 3: Update `CLAUDE.md` 测试 table**

Append after the existing PR-A test rows:

```markdown
| `test_ml_selector_lightgbm.py` | LightGBMSelector: 非线性选 / top_k / min_importance_ratio / 确定性 / 退化输入 / TwoStepPipeline 集成 |
```

- [ ] **Step 4: Add README.md LGB caveat section**

Open `README.md`. After the existing 快速开始 / 常用命令 sections (find a sensible insertion point — e.g., before the "已知不支持" or at the end of the strategy-related section), add this new section verbatim:

```markdown
### 关于 LightGBM 默认 selector

F2 PR-B1 起,`strategy.ml_factor.selector.type` 默认为 `"lightgbm"`,用 LightGBM 在 walk-forward 训练窗口上选因子。这是非线性选 + IC 线性加的两步法。

**过拟合提示**:每次 refit 训练集只有 ~250 bars × N 股,LGB 在小样本上容易过拟合。当前默认参数(`num_leaves=15`、`min_data_in_leaf=20`、`learning_rate=0.05`、`num_iterations=200`)已为这个规模做了保守化,但仍然 *依赖* "walk-forward 每次重训,单次过拟合无伤大雅" 这个假设。

**观测指标**:跑回测后看 `reports/backtest/latest.html` 里的 trade 分布;如果 IC 跨 refit 不稳、净值曲线锯齿明显,先调小 `num_leaves` 或调大 `min_data_in_leaf`;还不行就 `selector.type: lasso` 回到 PR-A 的线性 baseline 做对照。

**不做** holdout + early stopping(留给 F2 PR-B2 或更后)。
```

- [ ] **Step 5: Verify markdown renders cleanly**

```bash
head -80 CLAUDE.md
grep -A 30 "LightGBM 默认 selector" README.md
```
Spot-check that no formatting got broken.

- [ ] **Step 6: Commit**

```bash
git status
git add CLAUDE.md README.md
git diff --staged --stat
git commit -m "docs: F2 PR-B1 — LightGBM selector default + overfitting caveat

CLAUDE.md updates module map and selector.{lasso|lightgbm} config
description. README.md adds an LGB overfitting note with the
selector.type: lasso fallback recipe."
```

---

## Task 7: Final regression + spec acceptance check

**Files:** (none modified)

- [ ] **Step 1: Run the entire test suite**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q --no-header 2>&1 | tail -5
```
Expected: PASS — `previous count + 13` (5 from Task 2 + 8 from Task 3). If pre-PR-B1 count was 264, expect 277.

- [ ] **Step 2: Manual config-load smoke**

```bash
./.venv/Scripts/python.exe -c "
from stockpool.config import load_config
cfg = load_config('config.yaml')
sel = cfg.strategy.ml_factor.selector
print('selector.type:', sel.type)
print('selector.lasso.alpha:', sel.lasso.alpha)
print('selector.lightgbm.num_leaves:', sel.lightgbm.num_leaves)
"
```
Expected output:
```
selector.type: lasso
selector.lasso.alpha: 0.001
selector.lightgbm.num_leaves: 15
```
(Note: user's existing config.yaml has `type: lasso` explicit from PR-A, so this stays lasso — but the LGB sub-block with defaults is still parseable.)

- [ ] **Step 3: lightgbm import + version**

```bash
./.venv/Scripts/python.exe -c "import lightgbm; print('lightgbm', lightgbm.__version__)"
```
Expected: `lightgbm 4.X.Y`.

- [ ] **Step 4: Spec acceptance criteria check**

Re-read `docs/superpowers/specs/2026-05-23-f2-pr-b1-lightgbm-selector-design.md` §5. For each item, find the implementing test or step:

| # | Criterion | Where proved |
|---|-----------|--------------|
| 1 | 零回归 — all existing tests pass with `selector=lasso` pinned | Step 1 above |
| 2 | LGB 非线性增益 — picks both x0 and x1 in `_nonlinear_xy` | `test_lightgbm_selector_picks_nonlinear_features` in `tests/test_ml_selector_lightgbm.py` |
| 3 | 依赖装上 — lightgbm imports | Step 3 above |
| 4 | 可切换 — `selector.type: lasso` 一行 YAML 回到 PR-A | Confirmed by current config.yaml + Step 2 |
| 5 | 配置硬切 — flat `selector.num_leaves` rejected | `test_selector_lightgbm_flat_num_leaves_rejected` in `tests/test_config.py` |
| 6 | 缓存失效自然 — sig changes via `_strategy_signature` hashing `model_dump()` | Implicit (PR-A's mechanism extends naturally) |
| 7 | README 提醒 — LGB overfitting section present | Task 6 step 4 |

Report PASS/FAIL for each.

- [ ] **Step 5: Optional smoke — try LGB on the local cache**

If `data/<stock>_daily.parquet` files exist (user previously ran `fetch-universe` or `run`), exercise the LGB path end-to-end:

a. Temporarily edit `config.yaml` and flip `selector.type` from `lasso` to `lightgbm` (or just remove the `type:` line to use the default).

b. Run:
```bash
./.venv/Scripts/python.exe -m stockpool run --skip-trading-day-check 2>&1 | tail -10
```
Expected: runs without error; ml_models cache rebuilds under `data/ml_models/<sig>_*.pkl`; reports HTML regenerated.

c. **Restore `config.yaml`** to the pre-edit state (do NOT commit the type flip; leaving it on lasso preserves baseline numerics).

If no cache, skip this step.

- [ ] **Step 6: Confirm clean working tree**

```bash
git status
```
Expected: clean.

---

## Self-Review Notes

**Spec coverage** (against spec §3 / §5):

- ✅ §3.1 lightgbm dep — Task 1
- ✅ §3.2 LightGBMSelectorConfig + SelectorConfig extension — Task 2
- ✅ §3.3 LightGBMSelector class — Task 3
- ✅ §3.4 `_build_selector` factory — Task 4
- ✅ §3.5 legacy fixture patches — Task 5
- ✅ §3.6 LightGBMSelector tests (8) — Task 3
- ✅ §3.7 config schema tests (5) — Task 2
- ✅ §3.8 cache invalidation — implicit via existing `_strategy_signature`
- ✅ §3.9 user YAML behavior (no required change) — implicit
- ✅ §3.10 docs sync — Task 6
- ✅ §5.1 zero regression — Tasks 5 + 7
- ✅ §5.2 LGB 非线性增益 — Task 3 test 1
- ✅ §5.3 lightgbm installed — Tasks 1 + 7 step 3
- ✅ §5.4 selector.type=lasso fallback — implicit (config.yaml has explicit type:lasso)
- ✅ §5.5 extra='forbid' enforcement — Task 2 test 4
- ✅ §5.6 cache failure — implicit
- ✅ §5.7 README caveat — Task 6

**Placeholders**: none. Every code block is concrete.

**Type consistency**: `LightGBMSelector.__init__` signature matches Task 3 impl, Task 4 factory call, and Task 3 tests. `SelectorConfig.lightgbm: LightGBMSelectorConfig` referenced consistently across Task 2 (definition), Task 4 (factory access via `cfg.lightgbm`), Task 5 (default flip awareness).
