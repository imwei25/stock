# F2 PR-A — Embargo + label_type + Lasso 子段化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship F2 PR-A from spec `docs/superpowers/specs/2026-05-23-f2-pr-a-embargo-and-config-subnesting-design.md`: add walk-forward embargo (default auto = horizon), introduce `label_type` interface, and subnest `selector.lasso.*` to make room for PR-B's LightGBM.

**Architecture:** Pure config + walk-forward logic changes. No new files except two test files. Config schema gains a `LassoConfig` subcfg, two new `MLFactorConfig` fields, and one new helper in `MLFactorStrategy`. Forward-return label computation gains a `label_type` parameter that only implements the existing `"return"` path; others raise `NotImplementedError` as interface stubs for future PRs.

**Tech Stack:** Python 3.10+, Pydantic, pandas, numpy, pytest. No new dependencies.

---

## File Structure

| File | Change |
|------|--------|
| `src/stockpool/config.py` | New `LassoConfig`; `SelectorConfig` gains `lasso` subcfg + `extra="forbid"`; `MLFactorConfig` gains `embargo_days` + `label_type` fields |
| `src/stockpool/ml/dataset.py` | `forward_return_panel` and `forward_return` accept `label_type` kwarg (only `"return"` implemented) |
| `src/stockpool/backtesting/strategies.py` | New `_embargoed_label_end` helper on `MLFactorStrategy`; `_refit` uses it for both `per_stock` and `pooled`; `_build_truncated_pool` truncates pool stocks to `label_end_date`; `LassoSelector` instantiation reads from `cfg.selector.lasso.*` |
| `config.yaml` | Migrate `selector` block to nested form |
| `tests/test_config.py` | New tests for `LassoConfig`, `extra="forbid"` rejection, `embargo_days`, `label_type` |
| `tests/test_ml_dataset_labels.py` (new) | `forward_return_panel` `label_type` interface contracts |
| `tests/test_ml_strategy_embargo.py` (new) | Embargo defaults, explicit overrides, label-leak elimination on synthetic data |
| `tests/test_ml_strategy.py` | Existing fixtures gain `embargo_days=0` to preserve old IC numerics |
| `tests/test_ml_strategy_panel.py` | Existing fixtures gain `embargo_days=0` |
| `tests/test_ml_pipeline.py` | Existing fixtures gain `embargo_days=0` if used |
| `CLAUDE.md` | Update `strategy.ml_factor` config description + ML 模块地图 |
| `README.md` | Update `config.yaml` example snippet if present |

---

## Task 1: `LassoConfig` + subnest under `SelectorConfig`

**Files:**
- Modify: `src/stockpool/config.py:142-151`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for new schema**

Append to `tests/test_config.py`:

```python
def test_selector_lasso_subcfg_explicit():
    """New form: selector.lasso.alpha works."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig.model_validate({
        "type": "lasso",
        "lasso": {"alpha": 0.01, "max_iter": 500, "tol": 1e-5},
    })
    assert cfg.type == "lasso"
    assert cfg.lasso.alpha == 0.01
    assert cfg.lasso.max_iter == 500
    assert cfg.lasso.tol == 1e-5


def test_selector_lasso_subcfg_defaults():
    """selector: {type: lasso} uses LassoConfig defaults."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig.model_validate({"type": "lasso"})
    assert cfg.lasso.alpha == 0.001
    assert cfg.lasso.max_iter == 1000
    assert cfg.lasso.tol == 1e-6


def test_selector_flat_alpha_rejected():
    """Legacy flat alpha field on SelectorConfig must raise ValidationError."""
    import pydantic
    from stockpool.config import SelectorConfig
    with pytest.raises(pydantic.ValidationError) as exc:
        SelectorConfig.model_validate({"type": "lasso", "alpha": 0.01})
    assert "extra" in str(exc.value).lower() or "forbid" in str(exc.value).lower()
```

Add to the top of `tests/test_config.py` if missing:
```python
import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k "selector" --no-header`
Expected: FAILs — `AttributeError: 'SelectorConfig' object has no attribute 'lasso'` and the flat-alpha test passes (because current schema has flat `alpha` field, no `extra="forbid"`).

- [ ] **Step 3: Implement schema change**

Edit `src/stockpool/config.py:142-151`. Replace:

```python
class SelectorConfig(BaseModel):
    """Step-1 (factor selection) settings.

    Currently only ``type: lasso`` is supported; ``alpha`` is the L1 penalty
    strength on standardised features (typical range 1e-4 — 1e-1).
    """
    type: Literal["lasso"] = "lasso"
    alpha: float = Field(default=0.001, ge=0.0)
    max_iter: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-6, gt=0.0)
```

With:

```python
class LassoConfig(BaseModel):
    """Lasso-specific hyperparameters for ``selector.type == 'lasso'``.

    ``alpha`` is the L1 penalty on standardised features (typical range 1e-4 — 1e-1).
    """
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(default=0.001, ge=0.0)
    max_iter: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-6, gt=0.0)


class SelectorConfig(BaseModel):
    """Step-1 (factor selection) settings.

    PR-A only supports ``type: lasso``; PR-B will add ``"lightgbm"``.
    Hyperparameters live in the per-type subsection (``lasso.alpha`` etc.) so
    new selector types can add their own block without flattening into this one.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["lasso"] = "lasso"
    lasso: LassoConfig = Field(default_factory=LassoConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k "selector" --no-header`
Expected: 3 PASS.

- [ ] **Step 5: Run all config tests for regression**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -v --no-header`
Expected: ALL pass except any that referenced `selector.alpha` flat path (none expected — we already grepped). If any fail, fix them by switching to `selector.lasso.alpha`.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "feat(config): introduce LassoConfig + selector.lasso subnesting

SelectorConfig now uses extra='forbid' and exposes lasso as a sub-block;
flat alpha/max_iter/tol on SelectorConfig are rejected.

Sets up the namespace for PR-B's LightGBM selector."
```

---

## Task 2: Update `LassoSelector` instantiation in `MLFactorStrategy`

**Files:**
- Modify: `src/stockpool/backtesting/strategies.py:549-554`

- [ ] **Step 1: Identify the call site**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py -v --no-header 2>&1 | tail -10`
Expected: Several FAILs with messages like `AttributeError: 'SelectorConfig' object has no attribute 'alpha'` because Task 1 removed the flat fields.

- [ ] **Step 2: Update the path**

Edit `src/stockpool/backtesting/strategies.py:549-554`. Replace:

```python
        pipeline = TwoStepPipeline(
            selector=LassoSelector(
                alpha=cfg.selector.alpha,
                max_iter=cfg.selector.max_iter,
                tol=cfg.selector.tol,
            ),
            weighter=_build_weighter(cfg.weighter),
        )
```

With:

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

- [ ] **Step 3: Run ML strategy tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_pipeline.py -v --no-header`
Expected: PASS for everything except possibly tests where embargo behavior changes (Task 3 will guard those — for now, default `embargo_days` is not yet introduced, so behavior is unchanged).

- [ ] **Step 4: Commit**

```bash
git add src/stockpool/backtesting/strategies.py
git commit -m "refactor(strategy): read Lasso hyperparams from selector.lasso subcfg"
```

---

## Task 3: Migrate `config.yaml`

**Files:**
- Modify: `config.yaml` (the `selector` block under `strategy.ml_factor`)

- [ ] **Step 1: Inspect current config**

Run: `grep -A 5 "selector:" config.yaml`
Expected: A `selector:` block with flat `alpha`/`max_iter`/`tol`.

- [ ] **Step 2: Migrate to nested form**

Edit `config.yaml`. Find the `selector` block under `strategy.ml_factor`. Replace:

```yaml
    selector:
      type: lasso             # 目前唯一支持的 selector
      alpha: 0.001            # L1 强度; 越大越稀疏 (1e-4 ~ 1e-1 合理)
      max_iter: 1000
      tol: 1.0e-6
```

With:

```yaml
    selector:
      type: lasso             # 目前唯一支持的 selector (PR-B 后增加 lightgbm)
      lasso:
        alpha: 0.001          # L1 强度; 越大越稀疏 (1e-4 ~ 1e-1 合理)
        max_iter: 1000
        tol: 1.0e-6
```

- [ ] **Step 3: Verify config still loads**

Run: `./.venv/Scripts/python.exe -c "from stockpool.config import load_config; cfg = load_config('config.yaml'); print('selector:', cfg.strategy.ml_factor.selector); print('lasso.alpha:', cfg.strategy.ml_factor.selector.lasso.alpha)"`
Expected: `lasso.alpha: 0.001`

- [ ] **Step 4: Commit**

```bash
git add config.yaml
git commit -m "config: migrate selector block to nested lasso form"
```

---

## Task 4: Add `embargo_days` + `label_type` fields to `MLFactorConfig`

**Files:**
- Modify: `src/stockpool/config.py` (`MLFactorConfig`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
def test_mlfactor_embargo_days_default_is_none():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig()
    assert cfg.embargo_days is None


def test_mlfactor_embargo_days_explicit_zero():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig(embargo_days=0)
    assert cfg.embargo_days == 0


def test_mlfactor_embargo_days_explicit_positive():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig(embargo_days=5)
    assert cfg.embargo_days == 5


def test_mlfactor_embargo_days_negative_rejected():
    import pydantic
    from stockpool.config import MLFactorConfig
    with pytest.raises(pydantic.ValidationError):
        MLFactorConfig(embargo_days=-1)


def test_mlfactor_label_type_default_is_return():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig()
    assert cfg.label_type == "return"


def test_mlfactor_label_type_accepts_all_documented():
    from stockpool.config import MLFactorConfig
    for label in ("return", "vol_adjusted", "cross_sec_rank"):
        cfg = MLFactorConfig(label_type=label)
        assert cfg.label_type == label


def test_mlfactor_label_type_unknown_rejected():
    import pydantic
    from stockpool.config import MLFactorConfig
    with pytest.raises(pydantic.ValidationError):
        MLFactorConfig(label_type="momentum")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k "embargo or label_type" --no-header`
Expected: 7 FAILs with `unexpected keyword argument 'embargo_days'` / `'label_type'`.

- [ ] **Step 3: Add fields**

In `src/stockpool/config.py`, find `MLFactorConfig` (around line 191). Add these two field declarations just before the `selector` field (preserve all existing fields):

```python
    # Walk-forward embargo: extra gap (in bars) between train window end and
    # the test bar, to prevent horizon-day forward returns from leaking into
    # training labels. ``None`` means "auto = horizon" (recommended default).
    # Set to ``0`` to opt out and reproduce pre-PR-A behavior.
    embargo_days: int | None = Field(default=None, ge=0)

    # Training-label transform. PR-A only implements "return" (the legacy
    # absolute forward return). "vol_adjusted" and "cross_sec_rank" are
    # interface placeholders — calls into the corresponding code path will
    # raise NotImplementedError until a later PR fills them in.
    label_type: Literal["return", "vol_adjusted", "cross_sec_rank"] = "return"
```

Make sure `Literal` is already imported at the top of `config.py` (it is — used by `data.source`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -v -k "embargo or label_type" --no-header`
Expected: 7 PASS.

- [ ] **Step 5: Run full config tests for regression**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -v --no-header`
Expected: ALL pass.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "feat(config): add MLFactorConfig.embargo_days and label_type

embargo_days defaults to None (auto=horizon, fixes walk-forward label
leak); set to 0 to opt out.

label_type defaults to 'return' (legacy behavior); 'vol_adjusted' and
'cross_sec_rank' are interface placeholders for a future PR."
```

---

## Task 5: `forward_return_panel` and `forward_return` accept `label_type`

**Files:**
- Modify: `src/stockpool/ml/dataset.py:48-52` and `:155` (single-stock `forward_return`)
- Test: `tests/test_ml_dataset_labels.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/test_ml_dataset_labels.py`:

```python
"""Tests for forward_return_panel/forward_return label_type interface (F2 PR-A)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml.dataset import forward_return, forward_return_panel


def _close_panel(n_days: int = 20, n_stocks: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    codes = [f"s{i:02d}" for i in range(n_stocks)]
    return pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, (n_days, n_stocks)), axis=0),
        index=dates, columns=codes,
    )


def _close_series(n_days: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_days))
    return pd.DataFrame({"date": dates, "close": close})


def test_forward_return_panel_label_type_return_default():
    close = _close_panel()
    out_default = forward_return_panel(close, horizon=3)
    out_explicit = forward_return_panel(close, horizon=3, label_type="return")
    pd.testing.assert_frame_equal(out_default, out_explicit)
    # Legacy formula sanity
    expected = close.shift(-3) / close - 1.0
    pd.testing.assert_frame_equal(out_default, expected)


def test_forward_return_panel_label_type_vol_adjusted_not_implemented():
    close = _close_panel()
    with pytest.raises(NotImplementedError, match="vol_adjusted"):
        forward_return_panel(close, horizon=3, label_type="vol_adjusted")


def test_forward_return_panel_label_type_cross_sec_rank_not_implemented():
    close = _close_panel()
    with pytest.raises(NotImplementedError, match="cross_sec_rank"):
        forward_return_panel(close, horizon=3, label_type="cross_sec_rank")


def test_forward_return_panel_label_type_unknown_rejected():
    close = _close_panel()
    with pytest.raises(ValueError, match="label_type"):
        forward_return_panel(close, horizon=3, label_type="nonsense")


def test_forward_return_panel_horizon_must_be_positive():
    close = _close_panel()
    with pytest.raises(ValueError):
        forward_return_panel(close, horizon=0)


def test_forward_return_single_stock_label_type_return_default():
    df = _close_series()
    out_default = forward_return(df, horizon=3)
    out_explicit = forward_return(df, horizon=3, label_type="return")
    pd.testing.assert_series_equal(out_default, out_explicit)


def test_forward_return_single_stock_label_type_not_implemented_paths():
    df = _close_series()
    with pytest.raises(NotImplementedError, match="vol_adjusted"):
        forward_return(df, horizon=3, label_type="vol_adjusted")
    with pytest.raises(NotImplementedError, match="cross_sec_rank"):
        forward_return(df, horizon=3, label_type="cross_sec_rank")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_ml_dataset_labels.py -v --no-header`
Expected: FAILs — `forward_return_panel() got an unexpected keyword argument 'label_type'`.

- [ ] **Step 3: Add `label_type` parameter**

Edit `src/stockpool/ml/dataset.py`. Replace the existing `forward_return_panel`:

```python
def forward_return_panel(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """T×N forward return: ``close[t+h] / close[t] - 1``,末 h 行 NaN。"""
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    return close.shift(-horizon) / close - 1.0
```

with:

```python
def forward_return_panel(
    close: pd.DataFrame,
    horizon: int,
    label_type: str = "return",
) -> pd.DataFrame:
    """T×N forward-return panel with configurable label transform.

    Args:
        close: T × N 收盘价宽表 (date index, code columns).
        horizon: 前瞻天数 h。
        label_type:
            "return"          — close[t+h] / close[t] - 1 (legacy, default).
            "vol_adjusted"    — NotImplementedError (placeholder for future PR).
            "cross_sec_rank"  — NotImplementedError (placeholder for future PR).
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if label_type == "return":
        return close.shift(-horizon) / close - 1.0
    if label_type in ("vol_adjusted", "cross_sec_rank"):
        raise NotImplementedError(
            f"label_type={label_type!r} is not implemented in PR-A; "
            f"interface stub only."
        )
    raise ValueError(
        f"unknown label_type={label_type!r}; "
        f"expected one of: return, vol_adjusted, cross_sec_rank"
    )
```

Then replace the existing `forward_return` (single-stock, around line 155):

```python
def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """单股 forward return,行结构是 ``df['date']`` 索引的 Series。"""
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    close = df["close"].reset_index(drop=True)
    return close.shift(-horizon) / close - 1.0
```

with:

```python
def forward_return(
    df: pd.DataFrame,
    horizon: int,
    label_type: str = "return",
) -> pd.Series:
    """单股 forward return,带 label_type 接口(与 forward_return_panel 一致)。

    Only ``label_type='return'`` is implemented in PR-A; other documented
    options raise NotImplementedError as interface placeholders.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if label_type == "return":
        close = df["close"].reset_index(drop=True)
        return close.shift(-horizon) / close - 1.0
    if label_type in ("vol_adjusted", "cross_sec_rank"):
        raise NotImplementedError(
            f"label_type={label_type!r} is not implemented in PR-A; "
            f"interface stub only."
        )
    raise ValueError(
        f"unknown label_type={label_type!r}; "
        f"expected one of: return, vol_adjusted, cross_sec_rank"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_ml_dataset_labels.py -v --no-header`
Expected: 7 PASS.

- [ ] **Step 5: Run ml/dataset regression**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_ml_pipeline.py tests/test_ml_strategy.py tests/test_ml_strategy_panel.py -v --no-header 2>&1 | tail -10`
Expected: All current tests still pass (no caller passes `label_type`, default `"return"` matches legacy behavior).

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/ml/dataset.py tests/test_ml_dataset_labels.py
git commit -m "feat(ml): forward_return label_type interface (return-only impl)

forward_return_panel and forward_return now accept label_type with
documented values 'return' (impl), 'vol_adjusted' / 'cross_sec_rank'
(NotImplementedError stubs). Unknown values raise ValueError.

PR-A only ships the 'return' path; future PR fills the others in."
```

---

## Task 6: `_embargoed_label_end` helper + embargo in `_refit` and `_build_truncated_pool`

**Files:**
- Modify: `src/stockpool/backtesting/strategies.py:511-591` (`_refit` and `_build_truncated_pool`)
- Test: `tests/test_ml_strategy_embargo.py` (new — see Task 7)

- [ ] **Step 1: Add the helper method**

Edit `src/stockpool/backtesting/strategies.py`. Find the `_refit` method (starts ~line 503). Insert the new helper method **immediately before** `_refit` (so it sits within the `MLFactorStrategy` class):

```python
    def _embargoed_label_end(self, current_bar: int) -> int:
        """Return the bar index where training labels must stop, accounting for
        ``cfg.embargo_days``.

        Without embargo, labels are valid up to ``current_bar - horizon``.
        With embargo ``E``, push another ``E`` bars back so the most recent
        training label's forward-return window ends at least ``E`` bars before
        the test bar — eliminating overlap when E ≥ horizon.

        ``embargo_days = None`` means "auto = horizon" (the default).
        ``embargo_days = 0`` reproduces pre-PR-A behavior.
        """
        cfg = self.cfg
        effective_embargo = (
            cfg.embargo_days if cfg.embargo_days is not None else cfg.horizon
        )
        return current_bar - cfg.horizon - effective_embargo
```

- [ ] **Step 2: Update `_refit` to use the helper**

In `_refit` (around line 513), replace:

```python
        label_end = current_bar - cfg.horizon
        if label_end <= 0:
            return None
```

with:

```python
        label_end = self._embargoed_label_end(current_bar)
        if label_end <= 0:
            return None
```

- [ ] **Step 3: Update `_build_truncated_pool` to honor embargo**

In `_build_truncated_pool` (around line 570), the current signature is:

```python
    def _build_truncated_pool(
        self, daily_df: pd.DataFrame, current_date, current_bar: int,
    ) -> dict[str, pd.DataFrame]:
```

Replace the body. Replace:

```python
        sharing = self._is_sharing()
        out: dict[str, pd.DataFrame] = {}
        for code, df in self.pool_data.items():
            if not sharing and code == self._current_stock_code:
                continue
            mask = df["date"] < current_date
            sub = df.loc[mask].reset_index(drop=True)
            if len(sub) > 0:
                out[code] = sub
        if not sharing:
            host_key = self._current_stock_code or "_self_"
            out[host_key] = daily_df.iloc[:current_bar].reset_index(drop=True)
        return out
```

with:

```python
        sharing = self._is_sharing()

        # Embargo: truncate pool stocks to data older than the host's
        # label_end date so labels can't reach into the embargo gap.
        label_end = self._embargoed_label_end(current_bar)
        # label_end may be <= 0 if there isn't enough history; caller (_refit)
        # already guards by returning None in that case, but be defensive.
        host_slice_end = max(0, label_end + self.cfg.horizon)  # bars where forward
                                                                # return is still
                                                                # observable
        cutoff_date = (
            daily_df["date"].iloc[host_slice_end - 1]
            if host_slice_end > 0 else
            daily_df["date"].iloc[0]
        )

        out: dict[str, pd.DataFrame] = {}
        for code, df in self.pool_data.items():
            if not sharing and code == self._current_stock_code:
                continue
            mask = df["date"] <= cutoff_date
            sub = df.loc[mask].reset_index(drop=True)
            if len(sub) > 0:
                out[code] = sub
        if not sharing:
            host_key = self._current_stock_code or "_self_"
            out[host_key] = daily_df.iloc[:host_slice_end].reset_index(drop=True)
        return out
```

**Why `host_slice_end = label_end + horizon`**: `label_end` is the bar where labels stop being valid. We still need `horizon` more bars in the slice so that the bars in `[label_end - horizon, label_end)` (where labels are computed from `[label_end, label_end + horizon)`) have observable forward returns. The pool dataframe is then handed to `build_panel` which computes `forward_return_panel` and drops NaN labels; the bars in `[label_end, host_slice_end)` have NaN labels and get dropped automatically.

- [ ] **Step 4: Run tests to verify the changes compile and existing behavior is preserved when ``embargo_days=0``**

At this point, existing tests will see `embargo_days=None` default → auto = horizon, which **does** shift behavior. Some tests will fail. Task 7 patches those fixtures. For now, run targeted tests:

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py::test_walk_forward_pipeline_runs_to_completion -v --no-header
```
Expected: Either PASS (if the test is robust to small training-window shifts) or FAIL (if it asserts specific bar counts). Don't fix here — Task 7 is the right place.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/backtesting/strategies.py
git commit -m "feat(strategy): embargo gap in walk-forward training

Add _embargoed_label_end helper; _refit and _build_truncated_pool both
honor cfg.embargo_days. Default behavior with embargo_days=None pushes
training labels back another horizon bars from the test bar, eliminating
the horizon-day forward-return leak into training samples.

Existing tests must pass embargo_days=0 to preserve legacy numerics —
that fix-up follows in the next commit."
```

---

## Task 7: Add `embargo_days=0` to existing test fixtures

**Files:**
- Modify: `tests/test_ml_strategy.py` (10 `MLFactorConfig(...)` calls per earlier grep)
- Modify: `tests/test_ml_strategy_panel.py` (2 `MLFactorConfig(...)` calls)
- Modify: `tests/test_ml_pipeline.py` (any `MLFactorConfig(...)` calls — verify with grep)

- [ ] **Step 1: Run the full ML test suite to enumerate failures**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_pipeline.py -v --no-header 2>&1 | tail -40
```
Expected: Some failures due to auto-embargo shifting training windows.

- [ ] **Step 2: Patch `tests/test_ml_strategy.py`**

Run: `grep -n "MLFactorConfig(" tests/test_ml_strategy.py`

For **every** `MLFactorConfig(...)` call in that file, add `embargo_days=0` as a kwarg. If the call already passes other kwargs, add it; if it's a bare `MLFactorConfig()`, change to `MLFactorConfig(embargo_days=0)`.

Example transformations:

```python
# Before:
cfg = MLFactorConfig(train_window=120, refit_every=20, min_train_samples=60)
# After:
cfg = MLFactorConfig(train_window=120, refit_every=20, min_train_samples=60, embargo_days=0)
```

Apply to all 10 locations identified earlier (lines 34, 43, 55, 65, 76, 92, 105, 123, 166, 185). Run grep after to verify the count went from 10 to 10 with `embargo_days=0` present in each.

- [ ] **Step 3: Patch `tests/test_ml_strategy_panel.py`**

Run: `grep -n "MLFactorConfig(" tests/test_ml_strategy_panel.py`

Apply the same transformation to both call sites (lines 46, 73).

- [ ] **Step 4: Patch `tests/test_ml_pipeline.py`**

Run: `grep -n "MLFactorConfig(" tests/test_ml_pipeline.py`

If any sites print, add `embargo_days=0` to them. If none, skip.

- [ ] **Step 5: Run all ML tests for green**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_pipeline.py -v --no-header
```
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_pipeline.py
git commit -m "test(ml): set embargo_days=0 on legacy fixtures to preserve numerics

PR-A's new default (embargo_days=None → auto=horizon) shifts walk-forward
training windows. Existing fixtures opt out explicitly so their stored
IC/scoring numerics match pre-PR-A behavior. New embargo-aware tests live
in test_ml_strategy_embargo.py."
```

---

## Task 8: New `tests/test_ml_strategy_embargo.py`

**Files:**
- Create: `tests/test_ml_strategy_embargo.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_ml_strategy_embargo.py`:

```python
"""Embargo behavior tests for MLFactorStrategy (F2 PR-A)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.strategies import MLFactorStrategy
from stockpool.config import MLFactorConfig


def _synthetic_daily(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_days))
    return pd.DataFrame({
        "date": dates,
        "open":  close * 0.998,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, n_days).astype(float),
    })


def test_embargoed_label_end_default_uses_horizon():
    cfg = MLFactorConfig(horizon=3)  # embargo_days defaults to None → auto = 3
    strat = MLFactorStrategy(cfg=cfg)
    # current_bar=100, horizon=3, embargo=3 → label_end = 100 - 3 - 3 = 94
    assert strat._embargoed_label_end(100) == 94


def test_embargoed_label_end_explicit_zero_matches_legacy():
    cfg = MLFactorConfig(horizon=5, embargo_days=0)
    strat = MLFactorStrategy(cfg=cfg)
    # No embargo → label_end = current_bar - horizon
    assert strat._embargoed_label_end(100) == 95


def test_embargoed_label_end_explicit_positive_overrides_horizon():
    cfg = MLFactorConfig(horizon=3, embargo_days=10)
    strat = MLFactorStrategy(cfg=cfg)
    # current_bar=100, horizon=3, embargo=10 → 87
    assert strat._embargoed_label_end(100) == 87


def test_embargoed_label_end_can_go_negative_when_history_short():
    cfg = MLFactorConfig(horizon=3, embargo_days=5)
    strat = MLFactorStrategy(cfg=cfg)
    assert strat._embargoed_label_end(5) == 5 - 3 - 5  # = -3 — _refit guards


def test_refit_with_default_embargo_returns_none_when_insufficient_history():
    """Short history + default embargo + 20-bar factor warmup → _refit refuses.

    With n_days=30, horizon=3, embargo=horizon=3, label_end = 30 - 3 - 3 = 24.
    Default factors need ~20 bars of warmup (momentum_20 etc), so only bars
    20..23 are usable — 4 < min_train_samples=20 → return None.
    """
    cfg = MLFactorConfig(
        horizon=3, train_window=50, min_train_samples=20,
        refit_every=10, panel_mode="per_stock",
        # embargo_days=None → 3
    )
    strat = MLFactorStrategy(cfg=cfg)
    df = _synthetic_daily(n_days=30)
    from stockpool.ml.dataset import forward_return, build_factor_matrix
    X = build_factor_matrix(df, cfg.factors)
    y = forward_return(df, cfg.horizon)
    result = strat._refit(df, X, y, current_bar=30)
    assert result is None


def test_refit_with_legacy_no_embargo_runs_to_completion():
    """Long history + embargo_days=0 → fit succeeds and returns quantiles."""
    cfg = MLFactorConfig(
        horizon=3, train_window=120, min_train_samples=60,
        refit_every=20, panel_mode="per_stock",
        embargo_days=0,
    )
    strat = MLFactorStrategy(cfg=cfg)
    df = _synthetic_daily(n_days=300)
    from stockpool.ml.dataset import forward_return, build_factor_matrix
    X = build_factor_matrix(df, cfg.factors)
    y = forward_return(df, cfg.horizon)
    result = strat._refit(df, X, y, current_bar=300)
    assert result is not None
    pipeline, quantiles = result
    assert set(quantiles) == {"strong_buy", "buy", "sell", "strong_sell"}


def test_refit_with_default_embargo_long_history_also_runs_to_completion():
    """Sanity: with plenty of history, default auto-embargo still leaves enough
    samples to fit. Just train_window worth of bars shifts back by `horizon`."""
    cfg = MLFactorConfig(
        horizon=3, train_window=120, min_train_samples=60,
        refit_every=20, panel_mode="per_stock",
        # embargo_days=None → 3
    )
    strat = MLFactorStrategy(cfg=cfg)
    df = _synthetic_daily(n_days=300)
    from stockpool.ml.dataset import forward_return, build_factor_matrix
    X = build_factor_matrix(df, cfg.factors)
    y = forward_return(df, cfg.horizon)
    result = strat._refit(df, X, y, current_bar=300)
    assert result is not None
    pipeline, quantiles = result
    assert set(quantiles) == {"strong_buy", "buy", "sell", "strong_sell"}


def test_default_embargo_shifts_training_label_end_vs_legacy():
    """Constructive: same data, embargo=None (=horizon) trains on fewer bars
    than embargo=0."""
    cfg_default = MLFactorConfig(
        horizon=3, train_window=200, min_train_samples=30,
        refit_every=20, panel_mode="per_stock",
    )
    cfg_legacy = MLFactorConfig(
        horizon=3, train_window=200, min_train_samples=30,
        refit_every=20, panel_mode="per_stock",
        embargo_days=0,
    )
    strat_default = MLFactorStrategy(cfg=cfg_default)
    strat_legacy = MLFactorStrategy(cfg=cfg_legacy)
    assert strat_default._embargoed_label_end(100) == 94  # 100 - 3 - 3
    assert strat_legacy._embargoed_label_end(100) == 97   # 100 - 3 - 0
    assert (
        strat_default._embargoed_label_end(100)
        < strat_legacy._embargoed_label_end(100)
    )


def test_embargo_eliminates_label_leak_on_synthetic():
    """Construct an AR(horizon) series whose forward return at t is strongly
    correlated with the forward return at t - horizon (the "label overlap"
    failure mode). With embargo_days=0, train-set IC on this series will be
    inflated relative to a holdout; with embargo_days=horizon, both should
    align.

    This is a contractual test — we don't measure IC numerics precisely;
    we check that the *training-label set* under the default embargo
    excludes the overlapping bars.
    """
    cfg_legacy = MLFactorConfig(
        horizon=3, train_window=100, min_train_samples=20,
        refit_every=20, panel_mode="per_stock", embargo_days=0,
    )
    cfg_embargo = MLFactorConfig(
        horizon=3, train_window=100, min_train_samples=20,
        refit_every=20, panel_mode="per_stock",
        # embargo_days=None → auto=3
    )
    strat_legacy = MLFactorStrategy(cfg=cfg_legacy)
    strat_embargo = MLFactorStrategy(cfg=cfg_embargo)

    # With current_bar=200, horizon=3:
    #   legacy   label_end = 197 (bars 0..196 may have labels)
    #   embargo  label_end = 194 (bars 0..193 may have labels)
    # The 3 bars (194, 195, 196) — which have labels computed from closes at
    # (197, 198, 199), i.e. closes adjacent to the test bar (200) — must NOT
    # be in the embargoed training set.
    assert strat_legacy._embargoed_label_end(200) == 197
    assert strat_embargo._embargoed_label_end(200) == 194
    leak_bars = set(range(194, 197))
    embargo_train_bars = set(range(0, strat_embargo._embargoed_label_end(200)))
    legacy_train_bars = set(range(0, strat_legacy._embargoed_label_end(200)))
    assert leak_bars.issubset(legacy_train_bars), \
        "legacy mode is supposed to include leaky bars"
    assert leak_bars.isdisjoint(embargo_train_bars), \
        "embargo mode must exclude leaky bars"
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_embargo.py -v --no-header
```
Expected: 9 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ml_strategy_embargo.py
git commit -m "test(strategy): F2 PR-A embargo contracts

Verify _embargoed_label_end semantics under default (None → horizon),
explicit zero, and explicit positive; show that default embargo excludes
the leaky bars (in [current_bar - 2*horizon, current_bar - horizon))
from training while legacy embargo_days=0 includes them."
```

---

## Task 9: Sync `CLAUDE.md` and `README.md`

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

Per the project's "改动后更新文档" rule (CLAUDE.md), schema changes + default behavior changes both require docs sync.

- [ ] **Step 1: Update `CLAUDE.md` strategy.ml_factor description**

Open `CLAUDE.md`. Find the `## 配置 (config.yaml)` section, locate the line describing `strategy.ml_factor` sub-fields. Replace the existing list with:

```markdown
- **`strategy`** — `name` (`composite_verdict` 默认 / `ml_factor`) + `ml_factor` 子配置(`factors` 或 **`factors_file`** / `horizon` / `train_window` / `refit_every` / `panel_mode` / **`training_universe`** / **`share_pool_fit`** / **`embargo_days`** / **`label_type`** / `selector.lasso` / `weighter` / `thresholds` / `*_verdicts`)。`factors_file` 指向 HTML picker 导出的 JSON,与 `factors` 列表二选一。**`training_universe`**: `pool`(默认,只用 cfg.stocks)/ `all`(全市场 cache,需先 `fetch-universe`;仅在 `panel_mode=pooled` 时生效)。**`share_pool_fit`**(默认 `true`,仅 `panel_mode=pooled` 生效):跨股共享 fit,缓存键 `(sig, year, month)`,同月内所有股、所有 refit_bar 复用同一 pipeline;训练集不再剔除 host,host 自己以 ~1/N 权重进入自己的训练。**`embargo_days`**(默认 `null` = auto = `horizon`,F2 PR-A 新增):walk-forward 训练集与测试集之间的额外间隔,消除 horizon 日前向收益的标签泄露;设 `0` 回到 pre-PR-A 行为。**`label_type`**(默认 `"return"`,F2 PR-A 接口位):训练标签变换 — `"return"` 已实装,`"vol_adjusted"` / `"cross_sec_rank"` 是占位 raise `NotImplementedError`,后续 PR 实装。**`selector.lasso`**(F2 PR-A 子段化):`alpha` / `max_iter` / `tol` 现在嵌在 `selector.lasso` 下,顶层扁平字段已被 Pydantic 拒绝;后续 PR-B 会加 `selector.lightgbm` 同级。切到 `all` 或翻 `share_pool_fit`、改 `embargo_days` / `label_type` / `selector` 任一项后旧的 ml_models pkl 会因 sig 变化自动失效。
```

- [ ] **Step 2: Update `CLAUDE.md` ML 模块地图 row(if present)**

If the module map mentions `selector` config wiring, update its description to reference `selector.lasso.*`. Otherwise, no change needed.

Search for it:
```bash
grep -n "selector" CLAUDE.md
```

If any line says "Lasso selector / IC&IR weighter" without the subnesting, update to: "Lasso selector(`selector.lasso.*`) / IC&IR weighter".

- [ ] **Step 3: Update `CLAUDE.md` 测试 table**

Add a row for the new test files:

```markdown
| `test_ml_dataset_labels.py` | forward_return / forward_return_panel 的 label_type 接口(只 "return" 已实装) |
| `test_ml_strategy_embargo.py` | walk-forward embargo: 默认 auto=horizon,explicit 0 恢复旧行为,泄露 bar 被排除 |
```

- [ ] **Step 4: Update `README.md` config example (if any)**

Run: `grep -n "selector:" README.md`

If README shows a config snippet with `selector: {type: lasso, alpha: ...}`, update it to the nested form:

```yaml
    selector:
      type: lasso
      lasso:
        alpha: 0.001
```

If README has no inline config example, skip.

- [ ] **Step 5: Verify both docs render cleanly**

```bash
head -80 CLAUDE.md
head -50 README.md
```
Spot-check for broken markdown.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: F2 PR-A — embargo, label_type, selector.lasso subnesting"
```

---

## Task 10: Final regression + spec-acceptance check

**Files:** (none modified)

- [ ] **Step 1: Run the entire test suite**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: PASS — `previous count + 26` tests (Task 1: 3, Task 4: 7, Task 5: 7, Task 8: 9 = 26).

Record the final count.

- [ ] **Step 2: Manual config-load smoke**

```bash
./.venv/Scripts/python.exe -c "
from stockpool.config import load_config
cfg = load_config('config.yaml')
ml = cfg.strategy.ml_factor
print('embargo_days:', ml.embargo_days)
print('label_type:', ml.label_type)
print('selector.lasso.alpha:', ml.selector.lasso.alpha)
print('signature placeholder ok')
"
```
Expected:
```
embargo_days: None
label_type: return
selector.lasso.alpha: 0.001
signature placeholder ok
```

- [ ] **Step 3: Daily-report sanity (optional, requires data cache)**

If the user has `data/<some_stock>_daily.parquet` files cached:

```bash
./.venv/Scripts/python.exe -m stockpool run --skip-trading-day-check
```
Expected: Runs to completion; report regenerated under `reports/`. The `ml_models/*.pkl` cache rebuilds (signature changed). Record any warnings.

If no cache, skip this step.

- [ ] **Step 4: Confirm spec acceptance criteria**

Re-read `docs/superpowers/specs/2026-05-23-f2-pr-a-embargo-and-config-subnesting-design.md` §5 (验收标准). For each item:

1. **零回归**: Step 1 above proves it.
2. **embargo 真生效**: `test_embargo_eliminates_label_leak_on_synthetic` (Task 8) proves it.
3. **配置硬切干净**: `test_selector_flat_alpha_rejected` (Task 1) proves it.
4. **接口承诺**: `test_forward_return_panel_label_type_vol_adjusted_not_implemented` (Task 5) proves it.
5. **缓存失效自然**: Step 3 above (if exercised) demonstrates.
6. **docs 同步**: Task 9 committed both files.

Confirm each. If anything is missing, file as a follow-up before considering PR-A done.

- [ ] **Step 5: No final commit unless there is uncommitted state**

```bash
git status
```
Expected: Clean working tree on `feat/composite-backtest` with PR-A commits ahead of `28afb43` (the base for the entire F1+F2 feature branch).

---

## Self-Review Notes

**Spec coverage** (against spec §3 / §5):

- ✅ §3.1 schema changes — Task 1 (LassoConfig + subnest) + Task 4 (embargo_days + label_type)
- ✅ §3.2 label_type in forward_return_panel — Task 5
- ✅ §3.3 embargo in `_refit` + `_build_truncated_pool` — Task 6, with helper at strategies.py
- ✅ §3.4 cache invalidation — implicit via `_strategy_signature` hashing `model_dump()`; Task 10 step 2 sanity-loads new config
- ✅ §3.5 tests — Task 4 (config), Task 5 (dataset labels), Task 8 (embargo strategy), Task 7 (legacy fixtures)
- ✅ §3.6 user YAML migration — Task 3 (config.yaml) + Task 9 (docs)
- ✅ §5.1 零回归 — Task 7 + Task 10 step 1
- ✅ §5.2 embargo 真生效 — Task 8
- ✅ §5.3 配置硬切干净 — Task 1
- ✅ §5.4 接口承诺 — Task 5
- ✅ §5.5 缓存失效自然 — Task 10 step 3 (optional smoke)
- ✅ §5.6 docs 同步 — Task 9

**Placeholders**: none. Every code step has full content.

**Type consistency**: `_embargoed_label_end(current_bar: int) -> int` — same signature used in Tasks 6 and 8. `MLFactorConfig.embargo_days: int | None` — same type in config + tests + helper. `forward_return_panel(..., label_type: str = "return")` — same signature in Task 5 impl + tests.
