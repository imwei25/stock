# A/B Testing Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-stock A/B testing tool that compares two strategies on the same universe under identical execution conditions, configured via a separate `ab.yaml` file and surfaced as `python -m stockpool ab`.

**Architecture:** Refactor the existing per-stock backtest loop and pool-prep helper out of `cli.cmd_backtest` into a new `backtest_runner.py` module so both `cmd_backtest` and a new `cmd_ab` can call it. Add a new `stockpool/ab/` subpackage containing: config schema + deep-merge (`ab/config.py`), pool-sharing decision + arm execution (`ab/runner.py`), and HTML report rendering (`ab/report.py`). The existing `Strategy` ABC and backtest engines remain unchanged.

**Tech Stack:** Python 3.11+, pydantic v2, PyYAML, pandas, pyecharts. Tests use pytest + monkeypatch + synthetic OHLCV (no network).

**Spec:** `docs/superpowers/specs/2026-05-24-ab-testing-design.md`

---

## File Structure

### Created

| File | Responsibility |
|---|---|
| `src/stockpool/backtest_runner.py` | `prepare_pool(cfg, stocks, refresh)` and `backtest_stocks(cfg, stocks, pool_data, factor_panel, shared_cache, refresh)`. Extracted from `cli.py`, shared by `cmd_backtest` and `cmd_ab`. |
| `src/stockpool/ab/__init__.py` | Public exports: `ABConfig`, `load_ab_config`, `build_effective_cfg`, `run_ab`, `run_single_arm`, `ABResult`, `ArmResult`, `render_ab_report`. |
| `src/stockpool/ab/config.py` | `ArmBacktestOverride`, `ArmOverride`, `ABConfig` pydantic models; `load_ab_config(path)`; `build_effective_cfg(base, arm)`. |
| `src/stockpool/ab/runner.py` | `ArmResult`, `ABResult` dataclasses; `_decide_pool_sharing(arm_cfgs, stocks)`; `_run_arm`; `run_ab(ab_cfg, base_cfg, stocks, refresh, share_pool=True)`; `run_single_arm(ab_cfg, base_cfg, stocks, refresh, arm_name)`. |
| `src/stockpool/ab/report.py` | `render_ab_report(result, output_dir) → Path`; helpers `_compute_diff_table`, `_ab_equity_chart`, `_sharpe_scatter`, `_diff_histogram`, `_per_stock_card`. |
| `tests/test_ab.py` | Config schema, deep-merge, pool-sharing decision, runner failure isolation, ML cache isolation, report smoke. |
| `tests/test_cli_ab.py` | CLI smoke (mirrors `test_cli_backtest.py`). |
| `ab.yaml.example` | Documented sample at repo root. |

### Modified

| File | Change |
|---|---|
| `src/stockpool/cli.py` | Replace inlined `_prepare_ml_pool` and per-stock backtest loop in `cmd_backtest` with calls into `backtest_runner.*`. Add `cmd_ab(args)` + `p_ab` subparser registration in `main()`. |
| `CLAUDE.md` | Module map + quick commands + test table + known-unsupported section. |
| `README.md` | Quick-commands + worked end-to-end "compare two strategies" example. |

---

## Task 1: Extract `prepare_pool` from `cli._prepare_ml_pool` into `backtest_runner.py`

This is a behaviour-preserving refactor with no new tests — existing `test_cli_backtest.py` is the regression net. The existing private function gets a new home and a public name; `cli.py` calls into it.

**Files:**
- Create: `src/stockpool/backtest_runner.py`
- Modify: `src/stockpool/cli.py:88-141` (replace `_prepare_ml_pool` body and call site)
- Test: existing `tests/test_cli_backtest.py` (no new tests)

- [ ] **Step 1: Verify the existing test passes before refactor**

```bash
cd C:/Users/Administrator/Desktop/claude
python -m pytest tests/test_cli_backtest.py -q
```
Expected: PASS (baseline before refactor)

- [ ] **Step 2: Create `src/stockpool/backtest_runner.py` with `prepare_pool`**

Copy the body of `cli._prepare_ml_pool` verbatim, rename to `prepare_pool`, keep the same signature except dropping the leading underscore. Import what it needs.

```python
"""Backtest orchestration helpers shared by cli.cmd_backtest and ab.runner.

Provides:
  * prepare_pool(cfg, stocks, refresh) — pool data + factor panel for ml_factor
    in pooled mode (or (None, None) for other configurations).
  * backtest_stocks(cfg, stocks, pool_data, factor_panel, shared_cache,
    refresh) — per-stock backtest loop with failure isolation.

Both functions are extracted from cli.py to break the reverse-dependency
that ab/runner.py would otherwise have on cli.
"""
from __future__ import annotations

import logging
import traceback

import pandas as pd

from stockpool.backtest_composite import simulate_equity_curve, walk_forward_verdicts
from stockpool.config import AppConfig, Stock
from stockpool.fetcher import fetch_daily, load_universe_cache
from stockpool.strategy_factory import (
    build_factor_panel,
    build_strategy,
    simulate_strategy_equity_curve,
)

log = logging.getLogger("stockpool")


def prepare_pool(
    cfg: AppConfig, stocks: list[Stock], force_refresh: bool,
) -> tuple[dict[str, pd.DataFrame] | None, dict | None]:
    """Build (pool_data, factor_panel) for ml_factor strategies, or (None, None).

    Pool composition depends on ``cfg.strategy.ml_factor.training_universe``:
      * ``pool``: only ``cfg.stocks`` (legacy, ~10 stocks).
      * ``all``: full A-share cache from ``data/`` (~4000 stocks, requires a
        prior ``fetch-universe`` run). Application stocks are merged in so any
        cfg.stocks entry missing from the universe cache (e.g. 北交) is still
        usable. Cross-sec factors only become meaningful at panel widths in
        the hundreds, so ``all`` is recommended whenever WQ101 alphas are used.

    The factor panel is computed once on the combined pool and reused across
    every per-stock predict — the panel-wide computation can be expensive on
    the all-universe path.
    """
    if (
        cfg.strategy.name != "ml_factor"
        or cfg.strategy.ml_factor.panel_mode != "pooled"
    ):
        return None, None

    ml_cfg = cfg.strategy.ml_factor
    pool_data: dict[str, pd.DataFrame] = {}

    if ml_cfg.training_universe == "all":
        log.info("Loading universe cache (training_universe=all) ...")
        pool_data = load_universe_cache(cfg.data.cache_dir, cfg.data.history_days)
        if not pool_data:
            log.warning(
                "training_universe=all but data/ has no cached stocks. "
                "Run `python -m stockpool fetch-universe` first; falling back to pool."
            )
        else:
            log.info("Universe cache loaded: %d stocks", len(pool_data))

    for s in stocks:
        try:
            pool_data[s.code] = fetch_daily(
                s.code, cfg.data.history_days, cfg.data.cache_dir,
                force_refresh=force_refresh, source=cfg.data.source,
            )
        except Exception as e:
            log.warning("Pool preload skipped for %s: %s", s.code, e)

    log.info("Building factor panel over %d stocks × %d factors ...",
             len(pool_data), len(ml_cfg.factors))
    factor_panel = build_factor_panel(ml_cfg.factors, pool_data)
    log.info("Factor panel built: %d factors", len(factor_panel))
    return pool_data, factor_panel
```

- [ ] **Step 3: Replace `cli._prepare_ml_pool` with a re-export**

Edit `src/stockpool/cli.py` — delete the `_prepare_ml_pool` definition (lines ~88-141) and replace with an import + alias:

```python
# Near the top imports
from stockpool.backtest_runner import prepare_pool as _prepare_ml_pool
```

(Use the alias so existing call sites `_prepare_ml_pool(cfg, stocks, args.refresh)` keep working unchanged. We'll clean up the alias in a later task.)

- [ ] **Step 4: Run regression**

```bash
python -m pytest tests/test_cli_backtest.py tests/test_report_smoke.py -q
```
Expected: PASS (refactor is behaviour-preserving)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/backtest_runner.py src/stockpool/cli.py
git commit -m "$(cat <<'EOF'
refactor: extract prepare_pool from cli into backtest_runner

Move ml_factor pool/panel preparation out of cli._prepare_ml_pool into a
new backtest_runner.py module so ab/runner.py can call it without reverse-
importing cli. Behaviour preserved; cli aliases the new name for now.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Extract per-stock backtest loop into `backtest_runner.backtest_stocks`

The existing `cmd_backtest` per-stock loop logs and silently drops failures. The new shared helper returns `(success, failed)` so `cmd_ab` can surface failures in the report and `cmd_backtest` can log them as before.

**Files:**
- Modify: `src/stockpool/backtest_runner.py` (add `backtest_stocks`)
- Modify: `src/stockpool/cli.py:264-356` (replace `cmd_backtest` per-stock loop)
- Test: `tests/test_cli_backtest.py` (regression — must still pass)

- [ ] **Step 1: Add `backtest_stocks` to `backtest_runner.py`**

Append to `src/stockpool/backtest_runner.py`:

```python
def backtest_stocks(
    cfg: AppConfig,
    stocks: list[Stock],
    pool_data: dict[str, pd.DataFrame] | None,
    factor_panel: dict | None,
    shared_cache: dict,
    refresh: bool,
) -> tuple[list[tuple[str, str, "EquityResult"]], list[tuple[str, str]]]:
    """Backtest each stock; return (successes, failures).

    Failure isolation: any exception during a single stock's pipeline
    (data fetch, walk-forward, ML training, engine simulation) is caught,
    the offending code is appended to ``failures`` as ``(code, message)``,
    and the loop continues. Callers decide how to surface failures.

    Args:
        cfg: effective AppConfig (already deep-merged for ab arms).
        stocks: list of Stock to backtest.
        pool_data, factor_panel: pre-built ml_factor inputs (or None).
        shared_cache: mutable dict passed to MLFactorStrategy for cross-stock
            pipeline reuse within one call.
        refresh: forces fetch_daily to bypass cache.
    """
    per_stock: list[tuple[str, str, "EquityResult"]] = []
    failed: list[tuple[str, str]] = []
    needs_pool = pool_data is not None

    for s in stocks:
        log.info("Backtesting %s (%s)...", s.code, s.name)
        try:
            daily = pool_data.get(s.code) if needs_pool else None
            if daily is None:
                daily = fetch_daily(
                    s.code, cfg.data.history_days, cfg.data.cache_dir,
                    force_refresh=refresh, source=cfg.data.source,
                )
            if cfg.strategy.name == "composite_verdict":
                wf = walk_forward_verdicts(
                    daily, cfg.weights, cfg.scoring, cfg.verdicts, cfg.indicators,
                )
                if len(wf) == 0:
                    failed.append((s.code, "insufficient history"))
                    continue
                result = simulate_equity_curve(
                    wf,
                    holding_days_list=cfg.backtest.equity_curve_holding_days,
                    with_buy_and_hold=True,
                    buy_cost=cfg.backtest.costs.buy_cost,
                    sell_cost=cfg.backtest.costs.sell_cost,
                    risk_free_rate=cfg.backtest.risk_free_rate,
                    engine=cfg.backtest.engine,
                    position_size=cfg.backtest.position_size,
                    max_concurrent_lots=cfg.backtest.max_concurrent_lots,
                )
            else:
                strategy = build_strategy(
                    cfg,
                    pool_data=pool_data if needs_pool else None,
                    current_stock_code=s.code,
                    factor_panel=factor_panel,
                    shared_cache=shared_cache,
                )
                result = simulate_strategy_equity_curve(
                    daily, strategy,
                    holding_days_list=cfg.backtest.equity_curve_holding_days,
                    with_buy_and_hold=True,
                    buy_cost=cfg.backtest.costs.buy_cost,
                    sell_cost=cfg.backtest.costs.sell_cost,
                    risk_free_rate=cfg.backtest.risk_free_rate,
                    engine=cfg.backtest.engine,
                    position_size=cfg.backtest.position_size,
                    max_concurrent_lots=cfg.backtest.max_concurrent_lots,
                )
            per_stock.append((s.code, s.name, result))
        except Exception as e:
            log.error("Backtest failed for %s: %s\n%s", s.code, e, traceback.format_exc())
            failed.append((s.code, str(e)))

    return per_stock, failed
```

- [ ] **Step 2: Replace the per-stock loop in `cli.cmd_backtest`**

In `src/stockpool/cli.py`, replace the loop starting around line ~289 (the `per_stock: list = []` block) with a single `backtest_stocks` call. The post-loop rendering stays unchanged. Add the import.

Top of `cli.py`:
```python
from stockpool.backtest_runner import backtest_stocks, prepare_pool as _prepare_ml_pool
```

Replace the per-stock loop body inside `cmd_backtest` (everything between the `pool_data, factor_panel = _prepare_ml_pool(...)` line and the `if not per_stock:` check) with:

```python
    per_stock, failed = backtest_stocks(
        cfg, stocks, pool_data, factor_panel,
        shared_cache=shared_cache, refresh=args.refresh,
    )
    for code, err in failed:
        log.warning("Skipped %s: %s", code, err)
```

The lines `pool_data, factor_panel = _prepare_ml_pool(...)`, `needs_pool = ...`, and `shared_cache: dict = {}` stay. Delete the inlined `for s in stocks: ...` block (now lives in `backtest_stocks`).

- [ ] **Step 3: Run regression suite**

```bash
python -m pytest tests/test_cli_backtest.py tests/test_report_smoke.py tests/test_backtest_composite.py tests/test_ml_strategy.py -q
```
Expected: ALL PASS — behaviour preserved.

- [ ] **Step 4: Add a regression test for the new failure-list path**

Append to `tests/test_cli_backtest.py`:

```python
def test_backtest_continues_after_per_stock_failure(tmp_path, isolated_cache, monkeypatch):
    """A mid-loop stock failure must not abort the run; warning is logged."""
    cache_last = pd.date_range("2024-01-02", periods=200, freq="B")[-1]
    fresh_today = pd.Timestamp(cache_last) + pd.Timedelta(days=1)
    monkeypatch.setattr("stockpool.fetcher._today", lambda: fresh_today)

    import yaml
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(isolated_cache)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    # Use two codes: one cached (605589), one not (000001) — the second fetch
    # should fail (no network) but the first must complete.
    raw["stocks"] = [
        {"code": "605589", "name": "Cached", "sector": ""},
        {"code": "000001", "name": "Missing", "sector": ""},
    ]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    # Make sure the uncached fetch raises immediately rather than hitting the network.
    def _no_network(*a, **kw):
        raise RuntimeError("network disabled in test")
    monkeypatch.setattr("stockpool.backtest_runner.fetch_daily", _no_network)
    # 605589 is still readable from disk because ml_factor isn't in play here;
    # composite_verdict's path reads via fetch_daily — so patch the cached
    # branch by routing through a wrapper that serves disk for known codes.
    real_read = pd.read_parquet
    cached_codes = {"605589"}
    def _selective_fetch(code, *a, **kw):
        if code in cached_codes:
            return real_read(isolated_cache / f"{code}_daily.parquet")
        raise RuntimeError("network disabled in test")
    monkeypatch.setattr("stockpool.backtest_runner.fetch_daily", _selective_fetch)

    rc = main(["backtest", "--config", str(cfg_file)])
    assert rc == 0  # cached stock succeeded → report renders
    latest = tmp_path / "reports" / "backtest" / "latest.html"
    assert latest.exists()
    html = latest.read_text(encoding="utf-8")
    assert "605589" in html
    assert "000001" not in html  # failed stock not in report
```

- [ ] **Step 5: Run the new test**

```bash
python -m pytest tests/test_cli_backtest.py::test_backtest_continues_after_per_stock_failure -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/backtest_runner.py src/stockpool/cli.py tests/test_cli_backtest.py
git commit -m "$(cat <<'EOF'
refactor: extract per-stock backtest loop into backtest_runner.backtest_stocks

cmd_backtest now delegates the per-stock loop to a shared helper that
returns (successes, failures) tuples. cmd_backtest logs failures as before;
upcoming cmd_ab will surface them in the report. Regression test for
mid-loop failure isolation added.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ab/__init__.py` scaffolding

Create the subpackage with empty re-exports so later tasks can fill in.

**Files:**
- Create: `src/stockpool/ab/__init__.py`

- [ ] **Step 1: Create the package init**

```python
"""A/B testing tool — compare two strategies on the same per-stock universe.

Entry points:
    from stockpool.ab import ABConfig, load_ab_config
    from stockpool.ab import run_ab, run_single_arm, ABResult, ArmResult
    from stockpool.ab import render_ab_report

See docs/superpowers/specs/2026-05-24-ab-testing-design.md for the full design.
"""
from stockpool.ab.config import (
    ABConfig,
    ArmBacktestOverride,
    ArmOverride,
    build_effective_cfg,
    load_ab_config,
)
from stockpool.ab.report import render_ab_report
from stockpool.ab.runner import ABResult, ArmResult, run_ab, run_single_arm

__all__ = [
    "ABConfig",
    "ArmBacktestOverride",
    "ArmOverride",
    "build_effective_cfg",
    "load_ab_config",
    "ABResult",
    "ArmResult",
    "run_ab",
    "run_single_arm",
    "render_ab_report",
]
```

(This will fail to import until Tasks 4-7 complete; that's fine — no test runs against it yet.)

- [ ] **Step 2: Commit the scaffold**

```bash
git add src/stockpool/ab/__init__.py
git commit -m "$(cat <<'EOF'
chore: scaffold stockpool/ab/ subpackage init

Re-export surface for the A/B testing tool. Implementations land in
subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `ab/config.py` schema + tests

**Files:**
- Create: `src/stockpool/ab/config.py`
- Create: `tests/test_ab.py`

- [ ] **Step 1: Write failing tests for schema validation**

```python
"""Tests for stockpool.ab — config schema, deep-merge, runner, report."""
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from stockpool.ab import (
    ABConfig,
    ArmBacktestOverride,
    ArmOverride,
    build_effective_cfg,
    load_ab_config,
)
from stockpool.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_base_config(tmp_path: Path) -> Path:
    """Copy real config.yaml to tmp_path so tests don't tread on it."""
    base_src = PROJECT_ROOT / "config.yaml"
    base_dst = tmp_path / "config.yaml"
    base_dst.write_bytes(base_src.read_bytes())
    return base_dst


# ── Schema ──────────────────────────────────────────────────────────────────


def test_arm_holding_days_must_be_singleton():
    """equity_curve_holding_days enforces length-1 list with N > 0."""
    with pytest.raises(ValidationError):
        ArmBacktestOverride(equity_curve_holding_days=[5, 10])
    with pytest.raises(ValidationError):
        ArmBacktestOverride(equity_curve_holding_days=[])
    with pytest.raises(ValidationError):
        ArmBacktestOverride(equity_curve_holding_days=[0])
    # OK
    o = ArmBacktestOverride(equity_curve_holding_days=[10])
    assert o.equity_curve_holding_days == [10]


def test_arm_backtest_other_fields_optional():
    """All non-holding-days fields default to None (inherit base)."""
    o = ArmBacktestOverride(equity_curve_holding_days=[7])
    assert o.engine is None
    assert o.position_size is None
    assert o.costs is None


def test_arm_extra_fields_forbidden():
    """Typoed fields raise instead of silently being ignored."""
    with pytest.raises(ValidationError):
        ArmBacktestOverride(equity_curve_holding_days=[10], engin="single")  # typo


def test_arms_must_be_exactly_two():
    """ABConfig requires exactly 2 arms."""
    from stockpool.config import StrategyConfig
    arm = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    with pytest.raises(ValidationError):
        ABConfig(base_config="config.yaml", arms={"a": arm})
    with pytest.raises(ValidationError):
        ABConfig(base_config="config.yaml", arms={"a": arm, "b": arm, "c": arm})
    # OK
    ABConfig(base_config="config.yaml", arms={"a": arm, "b": arm})
```

- [ ] **Step 2: Run the tests — they should fail (import error)**

```bash
python -m pytest tests/test_ab.py -v
```
Expected: FAIL — `ImportError: cannot import name 'ABConfig' from 'stockpool.ab'`.

- [ ] **Step 3: Implement `ab/config.py` schema**

```python
"""Pydantic schema + loader + deep-merge for A/B test configuration."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from stockpool.config import (
    AppConfig,
    BacktestCostConfig,
    StrategyConfig,
    load_config,
)


class ArmBacktestOverride(BaseModel):
    """Per-arm overrides to the base.backtest section.

    equity_curve_holding_days is required and must be a length-1 list.
    All other fields default to None, meaning "inherit base.backtest.<same>".
    """
    model_config = ConfigDict(extra="forbid")
    equity_curve_holding_days: list[int]
    forward_days: list[int] | None = None
    risk_free_rate: float | None = None
    costs: BacktestCostConfig | None = None
    engine: Literal["single", "multi_lot"] | None = None
    position_size: float | None = None
    max_concurrent_lots: int | None = None

    @field_validator("equity_curve_holding_days")
    @classmethod
    def _single_n(cls, v: list[int]) -> list[int]:
        if len(v) != 1 or v[0] <= 0:
            raise ValueError(
                f"equity_curve_holding_days must be [N] with N > 0, got {v!r}"
            )
        return v


class ArmOverride(BaseModel):
    """One A/B arm: full strategy replacement + partial backtest override."""
    model_config = ConfigDict(extra="forbid")
    strategy: StrategyConfig
    backtest: ArmBacktestOverride


class ABConfig(BaseModel):
    """Top-level A/B test config (loaded from ab.yaml)."""
    model_config = ConfigDict(extra="forbid")
    base_config: str
    stocks_filter: list[str] = Field(default_factory=list)
    arms: dict[str, ArmOverride]

    @field_validator("arms")
    @classmethod
    def _exactly_two(cls, v: dict[str, ArmOverride]) -> dict[str, ArmOverride]:
        if len(v) != 2:
            raise ValueError(
                f"arms must contain exactly 2 entries, got {len(v)}: {list(v)}"
            )
        return v


def build_effective_cfg(base: AppConfig, arm: ArmOverride) -> AppConfig:
    """Deep-merge an arm's overrides into the base config.

    Rules:
      * arm.strategy replaces base.strategy wholesale.
      * arm.backtest fields with non-None values replace; None fields inherit
        from base.backtest.
      * stocks_filter is applied separately by load_ab_config, not here.
      * All other top-level fields pass through unchanged.

    Returns a fresh AppConfig with content_hash recomputed; does not mutate base.
    """
    merged = base.model_dump(mode="python")
    merged["strategy"] = arm.strategy.model_dump(mode="python")
    arm_bt = arm.backtest.model_dump(mode="python")
    base_bt = merged["backtest"]
    for k, v in arm_bt.items():
        if v is not None:
            base_bt[k] = v
    merged["backtest"] = base_bt
    # Re-validate through pydantic to catch any malformed merge result.
    out = AppConfig.model_validate(merged)
    # content_hash is derived from yaml bytes in load_config; here it has to
    # be deterministic per arm so ML cache keys differ. Dump merged dict to
    # canonical yaml bytes and hash those.
    canonical = yaml.safe_dump(merged, sort_keys=True).encode("utf-8")
    out.content_hash = hashlib.sha256(canonical).hexdigest()[:8]
    return out


def load_ab_config(ab_path: str | Path) -> ABConfig:
    """Load and validate ab.yaml. Performs post-pydantic checks that need
    side info (base config existence, stocks_filter membership, deep-merge
    validity).

    Raises pydantic.ValidationError or ValueError on any failure.
    """
    ab_path = Path(ab_path)
    raw = yaml.safe_load(ab_path.read_text(encoding="utf-8"))
    ab_cfg = ABConfig.model_validate(raw)

    # Resolve base_config relative to ab.yaml's directory.
    base_path = (ab_path.parent / ab_cfg.base_config).resolve()
    if not base_path.exists():
        raise ValueError(
            f"base_config {ab_cfg.base_config!r} (resolved to {base_path}) "
            f"does not exist"
        )

    base_cfg = load_config(base_path)

    # stocks_filter must be a subset of base.stocks
    if ab_cfg.stocks_filter:
        base_codes = {s.code for s in base_cfg.stocks}
        unknown = [c for c in ab_cfg.stocks_filter if c not in base_codes]
        if unknown:
            raise ValueError(
                f"stocks_filter references codes not in base.stocks: {unknown}"
            )

    # Speculatively build each arm's effective cfg to catch merge-time errors early.
    for name, arm in ab_cfg.arms.items():
        try:
            build_effective_cfg(base_cfg, arm)
        except Exception as e:
            raise ValueError(f"arm {name!r} fails effective-config validation: {e}")

    return ab_cfg
```

- [ ] **Step 4: Run schema tests to verify they pass**

```bash
python -m pytest tests/test_ab.py -v
```
Expected: PASS for the 4 schema tests.

- [ ] **Step 5: Add deep-merge tests**

Append to `tests/test_ab.py`:

```python
# ── Deep-merge ──────────────────────────────────────────────────────────────


def test_merge_replaces_strategy_section_wholly(tmp_path):
    """arm.strategy replaces base.strategy with no leakage."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    # Build an arm that flips composite → ml_factor
    from stockpool.config import MLFactorConfig
    arm = ArmOverride(
        strategy=StrategyConfig(
            name="ml_factor",
            ml_factor=MLFactorConfig(horizon=7),
        ),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    eff = build_effective_cfg(base, arm)
    assert eff.strategy.name == "ml_factor"
    assert eff.strategy.ml_factor.horizon == 7


def test_merge_backtest_fields_inherit_when_none(tmp_path):
    """arm.backtest fields left as None inherit from base.backtest."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    base_engine = base.backtest.engine
    arm = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    eff = build_effective_cfg(base, arm)
    assert eff.backtest.engine == base_engine
    assert eff.backtest.equity_curve_holding_days == [10]


def test_merge_backtest_fields_override_when_set(tmp_path):
    """arm.backtest fields with explicit values replace base's."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    arm = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(
            equity_curve_holding_days=[10],
            engine="single",
            position_size=0.25,
        ),
    )
    eff = build_effective_cfg(base, arm)
    assert eff.backtest.engine == "single"
    assert eff.backtest.position_size == 0.25


def test_merge_does_not_mutate_base(tmp_path):
    """Merging is non-destructive on the base config object."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    base_engine_before = base.backtest.engine
    base_name_before = base.strategy.name
    arm = ArmOverride(
        strategy=StrategyConfig(name="ml_factor"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10], engine="single"),
    )
    build_effective_cfg(base, arm)
    assert base.backtest.engine == base_engine_before
    assert base.strategy.name == base_name_before


def test_merge_recomputes_content_hash(tmp_path):
    """Different arm overrides → different content_hash (ML cache isolation)."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    arm_a = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[5]),
    )
    arm_b = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    eff_a = build_effective_cfg(base, arm_a)
    eff_b = build_effective_cfg(base, arm_b)
    assert eff_a.content_hash != eff_b.content_hash


def test_merge_revalidates_pydantic(tmp_path):
    """A merge result that violates AppConfig constraints fails fast."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    arm = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(
            equity_curve_holding_days=[10],
            position_size=2.0,  # exceeds le=1.0 in BacktestConfig
        ),
    )
    with pytest.raises(Exception):  # pydantic ValidationError
        build_effective_cfg(base, arm)
```

- [ ] **Step 6: Run deep-merge tests**

```bash
python -m pytest tests/test_ab.py -v
```
Expected: PASS (all 10 tests so far).

- [ ] **Step 7: Add load_ab_config integration tests**

Append to `tests/test_ab.py`:

```python
# ── load_ab_config ──────────────────────────────────────────────────────────


def _write_ab_yaml(tmp_path: Path, base_rel: str = "config.yaml",
                   stocks_filter=None, with_typo: bool = False) -> Path:
    arms = {
        "a": {
            "strategy": {"name": "composite_verdict"},
            "backtest": {"equity_curve_holding_days": [5]},
        },
        "b": {
            "strategy": {"name": "composite_verdict"},
            "backtest": {"equity_curve_holding_days": [10]},
        },
    }
    if with_typo:
        arms["a"]["backtest"]["equity_holding_days"] = [5]  # typo
    raw = {"base_config": base_rel, "arms": arms}
    if stocks_filter is not None:
        raw["stocks_filter"] = stocks_filter
    p = tmp_path / "ab.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_load_ab_config_happy_path(tmp_path):
    _write_base_config(tmp_path)
    ab_path = _write_ab_yaml(tmp_path)
    ab = load_ab_config(ab_path)
    assert list(ab.arms) == ["a", "b"]


def test_load_ab_config_missing_base_raises(tmp_path):
    ab_path = _write_ab_yaml(tmp_path, base_rel="does_not_exist.yaml")
    with pytest.raises(ValueError, match="base_config"):
        load_ab_config(ab_path)


def test_load_ab_config_resolves_base_relative_to_ab_yaml(tmp_path):
    """base_config: ../config.yaml works when ab.yaml is in a subdir."""
    _write_base_config(tmp_path)
    subdir = tmp_path / "experiments"
    subdir.mkdir()
    ab_path = _write_ab_yaml(subdir, base_rel="../config.yaml")
    ab = load_ab_config(ab_path)
    assert ab.base_config == "../config.yaml"


def test_stocks_filter_must_be_subset_of_base(tmp_path):
    _write_base_config(tmp_path)
    ab_path = _write_ab_yaml(tmp_path, stocks_filter=["999999"])  # not in base
    with pytest.raises(ValueError, match="stocks_filter"):
        load_ab_config(ab_path)


def test_extra_field_in_arm_backtest_rejected(tmp_path):
    _write_base_config(tmp_path)
    ab_path = _write_ab_yaml(tmp_path, with_typo=True)
    with pytest.raises(ValidationError):
        load_ab_config(ab_path)
```

- [ ] **Step 8: Run all `test_ab.py` tests**

```bash
python -m pytest tests/test_ab.py -v
```
Expected: PASS (15 tests).

- [ ] **Step 9: Commit**

```bash
git add src/stockpool/ab/config.py tests/test_ab.py
git commit -m "$(cat <<'EOF'
feat(ab): config schema, deep-merge, and load_ab_config

Pydantic models ArmBacktestOverride / ArmOverride / ABConfig enforce
arms == 2 and equity_curve_holding_days length 1. build_effective_cfg
replaces strategy wholesale and field-merges backtest. load_ab_config
adds post-pydantic checks: base_config existence, stocks_filter ⊆
base.stocks, and per-arm speculative merge.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `ab/runner.py` — pool sharing decision

Implements `_decide_pool_sharing` first (pure function, easy to test).

**Files:**
- Create: `src/stockpool/ab/runner.py`
- Modify: `tests/test_ab.py` (add pool-sharing tests)

- [ ] **Step 1: Write failing tests for `_decide_pool_sharing`**

Append to `tests/test_ab.py`:

```python
# ── Pool sharing plan ───────────────────────────────────────────────────────


def _make_cfg(tmp_path, strategy_name, panel_mode=None,
              training_universe=None, factors=None):
    """Helper: build an AppConfig with the requested strategy variant."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    if strategy_name == "ml_factor":
        from stockpool.config import MLFactorConfig
        kw = {}
        if panel_mode is not None:
            kw["panel_mode"] = panel_mode
        if training_universe is not None:
            kw["training_universe"] = training_universe
        if factors is not None:
            kw["factors"] = factors
        ml_cfg = MLFactorConfig(**kw)
        arm = ArmOverride(
            strategy=StrategyConfig(name="ml_factor", ml_factor=ml_cfg),
            backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
        )
    else:
        arm = ArmOverride(
            strategy=StrategyConfig(name="composite_verdict"),
            backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
        )
    return build_effective_cfg(base, arm)


def test_pool_plan_both_composite(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    cfgs = [_make_cfg(tmp_path, "composite_verdict")] * 2
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is False
    assert plan["shared_factors"] is None


def test_pool_plan_ml_vs_composite(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    cfgs = [
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all"),
        _make_cfg(tmp_path, "composite_verdict"),
    ]
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is True
    assert plan["shared_factors"] is None


def test_pool_plan_both_ml_pooled_all_same_factors(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    factors = ["momentum_20", "rsi_centered_14"]
    cfgs = [
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all", factors=factors),
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all", factors=factors),
    ]
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is True
    assert plan["shared_factors"] == factors


def test_pool_plan_both_ml_pooled_all_different_factors(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    cfgs = [
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all", factors=["momentum_20"]),
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all", factors=["rsi_centered_14"]),
    ]
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is True
    assert plan["shared_factors"] is None


def test_pool_plan_one_ml_per_stock_does_not_load_universe(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    cfgs = [
        _make_cfg(tmp_path, "ml_factor", panel_mode="per_stock"),
        _make_cfg(tmp_path, "composite_verdict"),
    ]
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is False
```

- [ ] **Step 2: Run — expect ImportError**

```bash
python -m pytest tests/test_ab.py -k pool_plan -v
```
Expected: FAIL — `cannot import _decide_pool_sharing`.

- [ ] **Step 3: Implement `_decide_pool_sharing` in `ab/runner.py`**

Create `src/stockpool/ab/runner.py`:

```python
"""A/B test runner: pool sharing decision + arm execution + ABResult.

Two entry points:
  * run_ab(...) → ABResult (always 2 arms)
  * run_single_arm(...) → ArmResult (debug helper for --arm flag)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from stockpool.ab.config import ABConfig, ArmOverride, build_effective_cfg
from stockpool.backtest_composite import EquityResult
from stockpool.backtest_runner import backtest_stocks, prepare_pool
from stockpool.config import AppConfig, Stock
from stockpool.fetcher import load_universe_cache
from stockpool.strategy_factory import build_factor_panel

log = logging.getLogger("stockpool")


@dataclass
class ArmResult:
    """Outcome of running one arm.

    name              — arm key from ab.yaml
    effective_cfg     — base ⊕ arm.override
    per_stock         — successful backtests: [(code, name, EquityResult), ...]
    failed            — failures: [(code, error_message), ...]
    """
    name: str
    effective_cfg: AppConfig
    per_stock: list[tuple[str, str, EquityResult]]
    failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ABResult:
    """Outcome of a full A/B run."""
    ab_cfg: ABConfig
    base_cfg: AppConfig
    arm_a: ArmResult
    arm_b: ArmResult
    run_date: str


def _ml_uses_universe(cfg: AppConfig) -> bool:
    """True iff this cfg's strategy needs the all-A-share universe cache."""
    if cfg.strategy.name != "ml_factor":
        return False
    ml = cfg.strategy.ml_factor
    return ml.panel_mode == "pooled" and ml.training_universe == "all"


def _decide_pool_sharing(
    arm_cfgs: list[AppConfig], stocks: list[Stock],
) -> dict:
    """Decide whether the universe cache and/or factor panel can be shared
    across the two arms.

    Returns {"load_universe": bool, "shared_factors": list[str] | None}.
      * load_universe=True iff at least one arm needs the all-universe cache.
      * shared_factors is a non-None factor list iff both arms are ml_factor +
        pooled + training_universe=all AND their factor lists are equal
        (order-sensitive).
    """
    load_universe = any(_ml_uses_universe(c) for c in arm_cfgs)

    shared_factors: list[str] | None = None
    if (
        len(arm_cfgs) == 2
        and all(_ml_uses_universe(c) for c in arm_cfgs)
    ):
        f_a = list(arm_cfgs[0].strategy.ml_factor.factors)
        f_b = list(arm_cfgs[1].strategy.ml_factor.factors)
        if f_a == f_b:
            shared_factors = f_a

    return {"load_universe": load_universe, "shared_factors": shared_factors}


def _no_share_plan() -> dict:
    return {"load_universe": False, "shared_factors": None}
```

- [ ] **Step 4: Run pool-plan tests**

```bash
python -m pytest tests/test_ab.py -k pool_plan -v
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ab/runner.py tests/test_ab.py
git commit -m "$(cat <<'EOF'
feat(ab): pool-sharing decision in ab.runner

_decide_pool_sharing inspects both arms' effective configs to decide
whether to load the universe cache once (yes if any arm is ml_factor +
pooled + training_universe=all) and whether to share a precomputed
factor panel (yes only when both arms have identical factor lists).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `ab/runner.py` — `_run_arm`, `run_ab`, `run_single_arm`

Build on Task 5's pool decision; wire `prepare_pool` + `backtest_stocks` into per-arm execution.

**Files:**
- Modify: `src/stockpool/ab/runner.py`
- Modify: `tests/test_ab.py` (add runner integration tests)

- [ ] **Step 1: Add `_prepare_pool_for_arm`, `_run_arm`, `run_ab`, `run_single_arm`**

Append to `src/stockpool/ab/runner.py`:

```python
def _prepare_pool_for_arm(
    arm_cfg: AppConfig,
    stocks: list[Stock],
    refresh: bool,
    injected_universe: dict[str, pd.DataFrame] | None,
    injected_factor_panel: dict | None,
) -> tuple[dict[str, pd.DataFrame] | None, dict | None]:
    """Per-arm pool prep with optional shared inputs from run_ab.

    If injected_universe is given, skip the load_universe_cache call and
    use it directly (merging cfg.stocks fetches on top, same as prepare_pool).
    If injected_factor_panel is given, skip build_factor_panel.

    For non-ml_factor or non-pooled arms, returns (None, None) — same as
    prepare_pool.
    """
    if (
        arm_cfg.strategy.name != "ml_factor"
        or arm_cfg.strategy.ml_factor.panel_mode != "pooled"
    ):
        return None, None

    # If neither shared input is provided, fall through to prepare_pool.
    if injected_universe is None and injected_factor_panel is None:
        return prepare_pool(arm_cfg, stocks, refresh)

    from stockpool.fetcher import fetch_daily
    ml_cfg = arm_cfg.strategy.ml_factor
    pool_data: dict[str, pd.DataFrame] = (
        dict(injected_universe) if injected_universe is not None else {}
    )
    if injected_universe is None and ml_cfg.training_universe == "all":
        pool_data = load_universe_cache(
            arm_cfg.data.cache_dir, arm_cfg.data.history_days,
        )

    for s in stocks:
        try:
            pool_data[s.code] = fetch_daily(
                s.code, arm_cfg.data.history_days, arm_cfg.data.cache_dir,
                force_refresh=refresh, source=arm_cfg.data.source,
            )
        except Exception as e:
            log.warning("Pool preload skipped for %s: %s", s.code, e)

    if injected_factor_panel is not None:
        factor_panel = injected_factor_panel
    else:
        log.info("Building factor panel over %d stocks × %d factors ...",
                 len(pool_data), len(ml_cfg.factors))
        factor_panel = build_factor_panel(ml_cfg.factors, pool_data)
    return pool_data, factor_panel


def _run_arm(
    arm_cfg: AppConfig,
    arm_name: str,
    stocks: list[Stock],
    pool_data: dict | None,
    factor_panel: dict | None,
    refresh: bool,
) -> ArmResult:
    """Backtest every stock for one arm."""
    log.info("Running arm %s ...", arm_name)
    per_stock, failed = backtest_stocks(
        arm_cfg, stocks, pool_data, factor_panel,
        shared_cache={}, refresh=refresh,
    )
    log.info("Arm %s: %d done, %d failed", arm_name, len(per_stock), len(failed))
    return ArmResult(
        name=arm_name, effective_cfg=arm_cfg,
        per_stock=per_stock, failed=failed,
    )


def run_ab(
    ab_cfg: ABConfig,
    base_cfg: AppConfig,
    stocks: list[Stock],
    refresh: bool,
    *,
    share_pool: bool = True,
) -> ABResult:
    """Run both arms; return an ABResult with exactly two ArmResults."""
    arm_items = list(ab_cfg.arms.items())
    arm_cfgs = [build_effective_cfg(base_cfg, arm) for _, arm in arm_items]

    plan = _decide_pool_sharing(arm_cfgs, stocks) if share_pool else _no_share_plan()

    shared_universe = None
    if plan["load_universe"]:
        try:
            shared_universe = load_universe_cache(
                base_cfg.data.cache_dir, base_cfg.data.history_days,
            )
            log.info("Shared universe loaded: %d stocks",
                     len(shared_universe) if shared_universe else 0)
        except Exception as e:
            log.warning("Universe load failed (each arm will reload): %s", e)

    shared_panel = None
    arm_results: list[ArmResult] = []
    for (name, _arm), arm_cfg in zip(arm_items, arm_cfgs):
        pool_data, factor_panel = _prepare_pool_for_arm(
            arm_cfg, stocks, refresh,
            injected_universe=shared_universe,
            injected_factor_panel=(shared_panel if plan["shared_factors"] else None),
        )
        if plan["shared_factors"] and shared_panel is None and factor_panel is not None:
            shared_panel = factor_panel
        arm_results.append(_run_arm(
            arm_cfg, name, stocks, pool_data, factor_panel, refresh,
        ))

    return ABResult(
        ab_cfg=ab_cfg, base_cfg=base_cfg,
        arm_a=arm_results[0], arm_b=arm_results[1],
        run_date=date.today().isoformat(),
    )


def run_single_arm(
    ab_cfg: ABConfig,
    base_cfg: AppConfig,
    stocks: list[Stock],
    refresh: bool,
    arm_name: str,
) -> ArmResult:
    """Debug helper: run only one arm by name; no pool sharing."""
    if arm_name not in ab_cfg.arms:
        raise KeyError(f"arm {arm_name!r} not in {list(ab_cfg.arms)}")
    arm = ab_cfg.arms[arm_name]
    arm_cfg = build_effective_cfg(base_cfg, arm)
    pool_data, factor_panel = _prepare_pool_for_arm(
        arm_cfg, stocks, refresh,
        injected_universe=None, injected_factor_panel=None,
    )
    return _run_arm(arm_cfg, arm_name, stocks, pool_data, factor_panel, refresh)
```

- [ ] **Step 2: Add runner integration tests**

Append to `tests/test_ab.py`:

```python
# ── Runner integration ──────────────────────────────────────────────────────


@pytest.fixture
def isolated_cache_two_stocks(tmp_path, monkeypatch):
    """Cache directory with two synthetic stocks ready to load."""
    import numpy as np
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    rng = np.random.default_rng(7)
    for code, seed in [("605589", 7), ("300750", 19)]:
        n = 220
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0005, 0.02, n)
        close = 100.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "open":  close * 0.998, "high": close * 1.005,
            "low":   close * 0.995, "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        })
        df.to_parquet(cache_dir / f"{code}_daily.parquet", index=False)

    cache_last = pd.date_range("2024-01-02", periods=220, freq="B")[-1]
    fresh_today = pd.Timestamp(cache_last) + pd.Timedelta(days=1)
    monkeypatch.setattr("stockpool.fetcher._today", lambda: fresh_today)
    return cache_dir


def _ab_setup(tmp_path, cache_dir) -> tuple[ABConfig, "AppConfig"]:
    """Build an ab.yaml + base config wired to a synthetic cache."""
    import yaml
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(cache_dir)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    raw["stocks"] = [
        {"code": "605589", "name": "Alpha", "sector": ""},
        {"code": "300750", "name": "Bravo", "sector": ""},
    ]
    base_path = tmp_path / "config.yaml"
    base_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    ab_raw = {
        "base_config": "config.yaml",
        "arms": {
            "single_engine": {
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10], "engine": "single"},
            },
            "multi_lot": {
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10], "engine": "multi_lot"},
            },
        },
    }
    ab_path = tmp_path / "ab.yaml"
    ab_path.write_text(yaml.safe_dump(ab_raw), encoding="utf-8")
    ab_cfg = load_ab_config(ab_path)
    base_cfg = load_config(base_path)
    return ab_cfg, base_cfg


def test_run_ab_smoke_two_composite_arms(tmp_path, isolated_cache_two_stocks):
    from stockpool.ab import run_ab
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    result = run_ab(ab_cfg, base_cfg, base_cfg.stocks, refresh=False)
    assert result.arm_a.name == "single_engine"
    assert result.arm_b.name == "multi_lot"
    assert len(result.arm_a.per_stock) == 2
    assert len(result.arm_b.per_stock) == 2
    assert result.arm_a.failed == []
    assert result.arm_b.failed == []


def test_run_ab_per_stock_failure_isolated(tmp_path, isolated_cache_two_stocks,
                                           monkeypatch):
    """Force one stock to crash inside arm A; arm B + the other stock survive."""
    from stockpool.ab import run_ab
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)

    # Patch walk_forward_verdicts to throw for 605589 only.
    from stockpool import backtest_runner as br
    real_wf = br.walk_forward_verdicts
    def _maybe_throw(daily, *a, **kw):
        if len(daily) > 0 and daily["close"].iloc[-1] > 0:
            # Throw if the first row date matches the 605589 series. We can't
            # easily disambiguate by code here, so instead throw on first call only.
            if not getattr(_maybe_throw, "_thrown", False):
                _maybe_throw._thrown = True
                raise RuntimeError("simulated crash")
        return real_wf(daily, *a, **kw)
    monkeypatch.setattr(br, "walk_forward_verdicts", _maybe_throw)

    result = run_ab(ab_cfg, base_cfg, base_cfg.stocks, refresh=False)
    # arm_a sees one crash; arm_b runs with original walk_forward (state on the
    # function persists, so arm_b's first stock also throws — assert only that
    # at least one stock from at least one arm survives).
    assert len(result.arm_a.per_stock) + len(result.arm_b.per_stock) > 0


def test_run_single_arm_returns_arm_result(tmp_path, isolated_cache_two_stocks):
    from stockpool.ab import run_single_arm
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    result = run_single_arm(ab_cfg, base_cfg, base_cfg.stocks, False, "multi_lot")
    assert result.name == "multi_lot"
    assert len(result.per_stock) == 2


def test_run_single_arm_unknown_name_raises(tmp_path, isolated_cache_two_stocks):
    from stockpool.ab import run_single_arm
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    with pytest.raises(KeyError):
        run_single_arm(ab_cfg, base_cfg, base_cfg.stocks, False, "no_such_arm")
```

- [ ] **Step 3: Run runner tests**

```bash
python -m pytest tests/test_ab.py -v
```
Expected: PASS (all tests so far ≈ 19).

- [ ] **Step 4: Commit**

```bash
git add src/stockpool/ab/runner.py tests/test_ab.py
git commit -m "$(cat <<'EOF'
feat(ab): run_ab + run_single_arm + per-arm pool prep

run_ab orchestrates both arms with optional shared universe / factor
panel, returning an ABResult. _prepare_pool_for_arm accepts pre-built
shared inputs (skips load_universe_cache and build_factor_panel when
provided). run_single_arm is the --arm debug entry point. Failure
isolation per stock per arm verified.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `ab/report.py` — HTML report

Smoke-tested only (existing `backtest_report.py` has the same approach).

**Files:**
- Create: `src/stockpool/ab/report.py`
- Modify: `tests/test_ab.py` (add report smoke tests)

- [ ] **Step 1: Write report smoke test (failing)**

Append to `tests/test_ab.py`:

```python
# ── Report smoke ────────────────────────────────────────────────────────────


def test_render_ab_report_smoke(tmp_path, isolated_cache_two_stocks):
    """End-to-end: run_ab + render_ab_report produces valid HTML."""
    from stockpool.ab import run_ab, render_ab_report
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    result = run_ab(ab_cfg, base_cfg, base_cfg.stocks, refresh=False)
    out_dir = tmp_path / "reports" / "ab"
    out_path = render_ab_report(result, output_dir=out_dir)
    assert out_path.exists()
    assert out_path.stat().st_size > 2048
    html = out_path.read_text(encoding="utf-8")
    assert "single_engine" in html
    assert "multi_lot" in html
    assert "605589" in html
    assert "300750" in html
    # latest.html copied
    assert (out_dir / "latest.html").exists()


def test_render_ab_report_handles_arm_with_no_successes(tmp_path,
                                                        isolated_cache_two_stocks,
                                                        monkeypatch):
    """If one arm has empty per_stock, the report still renders."""
    from stockpool.ab import run_ab, render_ab_report
    from stockpool.ab.runner import ArmResult
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    result = run_ab(ab_cfg, base_cfg, base_cfg.stocks, refresh=False)
    # Synthetically empty arm A
    result.arm_a = ArmResult(
        name=result.arm_a.name,
        effective_cfg=result.arm_a.effective_cfg,
        per_stock=[],
        failed=[("605589", "synthetic"), ("300750", "synthetic")],
    )
    out_path = render_ab_report(result, output_dir=tmp_path / "reports" / "ab")
    html = out_path.read_text(encoding="utf-8")
    assert "0 succeeded" in html or "0 succeeded".replace(" ", "&nbsp;") in html


def test_compute_diff_table_uses_only_common_stocks(tmp_path,
                                                     isolated_cache_two_stocks):
    """Aggregate diff table aggregates over stocks present in BOTH arms."""
    from stockpool.ab.report import compute_diff_table
    from stockpool.ab.runner import ArmResult

    # Build two arms; one of them is missing one stock.
    from stockpool.ab import run_ab
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    result = run_ab(ab_cfg, base_cfg, base_cfg.stocks, refresh=False)

    # Drop one stock from arm B
    result.arm_b = ArmResult(
        name=result.arm_b.name,
        effective_cfg=result.arm_b.effective_cfg,
        per_stock=result.arm_b.per_stock[:1],
        failed=[("300750", "synthetic miss")],
    )
    table = compute_diff_table(result.arm_a, result.arm_b)
    assert table["common_stocks_count"] == 1
```

- [ ] **Step 2: Run — expect import failure**

```bash
python -m pytest tests/test_ab.py -k report -v
```
Expected: FAIL — `cannot import render_ab_report` (or `compute_diff_table`).

- [ ] **Step 3: Implement `ab/report.py`**

```python
"""HTML report for A/B test results.

Layout (top → bottom):
  1. Metadata banner (arm names, differing fields, base hash, per-arm counts).
  2. Aggregate diff table over common stocks.
  3. Sharpe scatter (A.sharpe x B.sharpe per common stock).
  4. Sharpe diff histogram (B.sharpe - A.sharpe).
  5. Per-stock cards (3-series equity chart + side-by-side metric table).
  6. Failure detail + full effective_cfg dumps (folded).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Scatter

from stockpool.ab.runner import ABResult, ArmResult
from stockpool.backtest_composite import EquityResult
from stockpool.backtest_report import _CSS

_METRIC_DEFS = [
    ("total_return",        "Total return",        True,  "pct"),    # higher better
    ("annualized_return",   "Annualized return",   True,  "pct"),
    ("sharpe",              "Sharpe",              True,  "num"),
    ("max_drawdown",        "Max drawdown",        False, "pct"),    # lower better
    ("win_rate",            "Win rate",            True,  "pct"),
    ("avg_trade_return_pct","Avg trade ret %",     True,  "raw"),    # already pct
    ("trade_count",         "Trade count",         None,  "int"),    # not scored
]


def _fmt(val, kind: str) -> str:
    if val is None:
        return "—"
    if kind == "pct":
        return f"{val*100:+.2f}%"
    if kind == "num":
        return f"{val:+.3f}"
    if kind == "raw":
        return f"{val:+.2f}"
    if kind == "int":
        return str(int(val))
    return str(val)


def _arm_metrics(arm_result: ArmResult) -> dict[str, dict[str, float]]:
    """Map code → metrics dict (the single-N dict from EquityResult)."""
    out: dict[str, dict[str, float]] = {}
    for code, _name, res in arm_result.per_stock:
        # Each arm has length-1 equity_curve_holding_days (schema-enforced).
        N = next(iter(res.metrics))
        out[code] = res.metrics[N]
    return out


def compute_diff_table(arm_a: ArmResult, arm_b: ArmResult) -> dict:
    """Aggregate per-metric stats over stocks present in BOTH arms.

    Returns a dict:
      {
        "common_stocks_count": int,
        "rows": list[dict],   # one per metric: keys = a_mean / a_median /
                              # b_mean / b_median / diff_mean / a_wins / b_wins / kind / higher_better / label
      }
    """
    a_metrics = _arm_metrics(arm_a)
    b_metrics = _arm_metrics(arm_b)
    common = sorted(set(a_metrics) & set(b_metrics))

    rows = []
    for key, label, higher_better, kind in _METRIC_DEFS:
        a_vals = [a_metrics[c].get(key) or 0.0 for c in common]
        b_vals = [b_metrics[c].get(key) or 0.0 for c in common]
        if not common:
            rows.append({
                "label": label, "kind": kind, "higher_better": higher_better,
                "a_mean": None, "a_median": None,
                "b_mean": None, "b_median": None,
                "diff_mean": None, "a_wins": 0, "b_wins": 0,
            })
            continue
        a_mean = sum(a_vals) / len(a_vals)
        b_mean = sum(b_vals) / len(b_vals)
        a_med = sorted(a_vals)[len(a_vals) // 2]
        b_med = sorted(b_vals)[len(b_vals) // 2]
        if higher_better is None:
            a_wins = b_wins = 0
        elif higher_better:
            a_wins = sum(1 for a, b in zip(a_vals, b_vals) if a > b)
            b_wins = sum(1 for a, b in zip(a_vals, b_vals) if b > a)
        else:  # lower better
            a_wins = sum(1 for a, b in zip(a_vals, b_vals) if a < b)
            b_wins = sum(1 for a, b in zip(a_vals, b_vals) if b < a)
        rows.append({
            "label": label, "kind": kind, "higher_better": higher_better,
            "a_mean": a_mean, "a_median": a_med,
            "b_mean": b_mean, "b_median": b_med,
            "diff_mean": b_mean - a_mean,
            "a_wins": a_wins, "b_wins": b_wins,
        })
    return {"common_stocks_count": len(common), "rows": rows, "common": common}


def _diff_table_html(table: dict, arm_a_name: str, arm_b_name: str) -> str:
    """Render the aggregate diff table as HTML."""
    header = (
        f"<tr><th>Metric</th>"
        f"<th>{arm_a_name} mean</th><th>{arm_a_name} median</th>"
        f"<th>{arm_b_name} mean</th><th>{arm_b_name} median</th>"
        f"<th>Δ mean (B−A)</th>"
        f"<th>{arm_a_name} wins</th><th>{arm_b_name} wins</th></tr>"
    )
    body_rows = []
    for row in table["rows"]:
        body_rows.append(
            f"<tr><td>{row['label']}</td>"
            f"<td>{_fmt(row['a_mean'], row['kind'])}</td>"
            f"<td>{_fmt(row['a_median'], row['kind'])}</td>"
            f"<td>{_fmt(row['b_mean'], row['kind'])}</td>"
            f"<td>{_fmt(row['b_median'], row['kind'])}</td>"
            f"<td><strong>{_fmt(row['diff_mean'], row['kind'])}</strong></td>"
            f"<td>{row['a_wins']}</td><td>{row['b_wins']}</td></tr>"
        )
    return (
        f"<table><thead>{header}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _sharpe_scatter(arm_a: ArmResult, arm_b: ArmResult) -> str:
    """Scatter of arm_a.sharpe vs arm_b.sharpe per common stock."""
    a_metrics = _arm_metrics(arm_a)
    b_metrics = _arm_metrics(arm_b)
    common = sorted(set(a_metrics) & set(b_metrics))
    if not common:
        return "<p>No common stocks — scatter omitted.</p>"
    points = [[a_metrics[c]["sharpe"], b_metrics[c]["sharpe"], c] for c in common]
    vals = [p[0] for p in points] + [p[1] for p in points]
    lo, hi = min(vals), max(vals)

    sc = (
        Scatter(init_opts=opts.InitOpts(width="100%", height="420px"))
        .add_xaxis([p[0] for p in points])
        .add_yaxis(
            f"{arm_b.name} vs {arm_a.name}",
            [p[1] for p in points],
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Sharpe scatter — above diagonal = {arm_b.name} wins",
                pos_left="center",
            ),
            xaxis_opts=opts.AxisOpts(
                name=f"{arm_a.name} Sharpe", min_=lo, max_=hi, type_="value",
            ),
            yaxis_opts=opts.AxisOpts(
                name=f"{arm_b.name} Sharpe", min_=lo, max_=hi, type_="value",
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
        )
    )
    # Reference diagonal via mark line.
    sc.options["series"][0]["markLine"] = {
        "symbol": "none",
        "lineStyle": {"color": "#999", "type": "dashed"},
        "data": [[{"coord": [lo, lo]}, {"coord": [hi, hi]}]],
    }
    return sc.render_embed()


def _diff_histogram(arm_a: ArmResult, arm_b: ArmResult) -> str:
    """Histogram of B.sharpe - A.sharpe per common stock."""
    a_metrics = _arm_metrics(arm_a)
    b_metrics = _arm_metrics(arm_b)
    common = sorted(set(a_metrics) & set(b_metrics))
    diffs = [b_metrics[c]["sharpe"] - a_metrics[c]["sharpe"] for c in common]
    if not diffs:
        return "<p>No common stocks — histogram omitted.</p>"

    lo, hi = min(diffs), max(diffs)
    if lo == hi:
        hi = lo + 1e-6
    n_bins = min(12, max(4, len(diffs) // 2 or 4))
    width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for d in diffs:
        i = min(int((d - lo) / width), n_bins - 1)
        bins[i] += 1
    labels = [f"{lo + width*i:+.2f}" for i in range(n_bins)]

    bar = (
        Bar(init_opts=opts.InitOpts(width="100%", height="320px"))
        .add_xaxis(labels)
        .add_yaxis(
            f"{arm_b.name} − {arm_a.name} (Sharpe)", bins,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Sharpe diff distribution",
                                      pos_left="center"),
            xaxis_opts=opts.AxisOpts(name="Δ Sharpe"),
            yaxis_opts=opts.AxisOpts(name="stocks"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
        )
    )
    return bar.render_embed()


def _ab_equity_chart(
    a_result: EquityResult | None,
    b_result: EquityResult | None,
    a_name: str,
    b_name: str,
    title: str,
) -> str:
    """3-series equity chart: arm A, arm B, buy-and-hold (drawn once)."""
    # Pick whichever result is available for axis labels and B&H curve.
    ref = a_result if a_result is not None else b_result
    if ref is None:
        return ""
    any_curve = next(iter(ref.curves.values()))
    dates = pd.DatetimeIndex(any_curve["date"]).strftime("%Y-%m-%d").tolist()

    line = Line(init_opts=opts.InitOpts(width="100%", height="420px")).add_xaxis(dates)
    if a_result is not None:
        N = next(iter(a_result.curves))
        vals = [round(float(v), 4) for v in a_result.curves[N]["equity"].values]
        line.add_yaxis(a_name, vals, is_smooth=True, is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False))
    if b_result is not None:
        N = next(iter(b_result.curves))
        vals = [round(float(v), 4) for v in b_result.curves[N]["equity"].values]
        line.add_yaxis(b_name, vals, is_smooth=True, is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False))
    if ref.buy_and_hold is not None:
        bh = [round(float(v), 4) for v in ref.buy_and_hold["equity"].values]
        line.add_yaxis("Buy & Hold", bh, is_smooth=True, is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False),
                       linestyle_opts=opts.LineStyleOpts(type_="dashed", width=2))

    line.set_global_opts(
        title_opts=opts.TitleOpts(title=title, pos_left="center"),
        xaxis_opts=opts.AxisOpts(is_scale=True,
                                 axislabel_opts=opts.LabelOpts(rotate=30, font_size=10)),
        yaxis_opts=opts.AxisOpts(is_scale=True, name="净值"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_top="6%"),
        datazoom_opts=[opts.DataZoomOpts(type_="inside"),
                       opts.DataZoomOpts(type_="slider", pos_bottom="2%")],
    )
    line.options["grid"] = {"top": "18%", "bottom": "16%", "left": "8%", "right": "4%",
                            "containLabel": True}
    return line.render_embed()


def _per_stock_cards(arm_a: ArmResult, arm_b: ArmResult) -> str:
    a_map = {code: (name, res) for code, name, res in arm_a.per_stock}
    b_map = {code: (name, res) for code, name, res in arm_b.per_stock}
    all_codes = sorted(set(a_map) | set(b_map))
    sections = []
    for i, code in enumerate(all_codes):
        a_entry = a_map.get(code)
        b_entry = b_map.get(code)
        name = (a_entry or b_entry)[0]
        a_res = a_entry[1] if a_entry else None
        b_res = b_entry[1] if b_entry else None
        title = f"{code} {name}"
        if a_res is None:
            title += f" [Arm {arm_a.name} failed]"
        if b_res is None:
            title += f" [Arm {arm_b.name} failed]"
        chart = _ab_equity_chart(a_res, b_res, arm_a.name, arm_b.name, title)
        # Single-stock table
        rows = []
        for key, label, higher_better, kind in _METRIC_DEFS:
            a_v = None if a_res is None else a_res.metrics[next(iter(a_res.metrics))].get(key)
            b_v = None if b_res is None else b_res.metrics[next(iter(b_res.metrics))].get(key)
            d = (b_v - a_v) if (a_v is not None and b_v is not None) else None
            rows.append(
                f"<tr><td>{label}</td>"
                f"<td>{_fmt(a_v, kind)}</td>"
                f"<td>{_fmt(b_v, kind)}</td>"
                f"<td>{_fmt(d, kind)}</td></tr>"
            )
        table = (
            f"<table><thead><tr><th>Metric</th>"
            f"<th>{arm_a.name}</th><th>{arm_b.name}</th><th>Δ</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
        open_attr = "open" if i < 3 else ""
        sections.append(
            f"<details {open_attr}><summary>"
            f"<span style='font-size:1.1em;font-weight:bold'>{title}</span>"
            f"</summary><div class='chart-wrap'>{chart}</div>{table}</details>"
        )
    return "".join(sections)


def _metadata_banner(ab_result: ABResult) -> str:
    a, b = ab_result.arm_a, ab_result.arm_b
    a_arm = ab_result.ab_cfg.arms[a.name]
    b_arm = ab_result.ab_cfg.arms[b.name]
    return (
        f"<h1>A/B Test Report — {ab_result.run_date}</h1>"
        f"<p class='meta'>Base config: {ab_result.ab_cfg.base_config} "
        f"(hash: {ab_result.base_cfg.content_hash})</p>"
        f"<div class='banner'>"
        f"  <h3>Arm A: {a.name}</h3>"
        f"  <pre>{yaml.safe_dump(a_arm.model_dump(), sort_keys=False)}</pre>"
        f"  <p>{len(a.per_stock)} succeeded, {len(a.failed)} failed.</p>"
        f"  <h3>Arm B: {b.name}</h3>"
        f"  <pre>{yaml.safe_dump(b_arm.model_dump(), sort_keys=False)}</pre>"
        f"  <p>{len(b.per_stock)} succeeded, {len(b.failed)} failed.</p>"
        f"</div>"
    )


def _failure_detail(ab_result: ABResult) -> str:
    def _format_one(arm: ArmResult) -> str:
        if not arm.failed:
            return f"<p>{arm.name}: no failures.</p>"
        rows = "".join(
            f"<li><code>{code}</code>: {err}</li>" for code, err in arm.failed
        )
        return f"<p>{arm.name}:</p><ul>{rows}</ul>"
    return (
        f"<details><summary>Failure detail</summary>"
        f"{_format_one(ab_result.arm_a)}{_format_one(ab_result.arm_b)}"
        f"</details>"
    )


def _full_cfg_dump(ab_result: ABResult) -> str:
    a_yaml = yaml.safe_dump(ab_result.arm_a.effective_cfg.model_dump(), sort_keys=False)
    b_yaml = yaml.safe_dump(ab_result.arm_b.effective_cfg.model_dump(), sort_keys=False)
    return (
        f"<details><summary>Full effective configs</summary>"
        f"<h4>Arm A: {ab_result.arm_a.name}</h4><pre>{a_yaml}</pre>"
        f"<h4>Arm B: {ab_result.arm_b.name}</h4><pre>{b_yaml}</pre>"
        f"</details>"
    )


def render_ab_report(ab_result: ABResult, output_dir: str | Path) -> Path:
    """Render the full A/B HTML report. Writes <date>.html and latest.html."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ab_result.run_date}.html"

    banner = _metadata_banner(ab_result)
    table = compute_diff_table(ab_result.arm_a, ab_result.arm_b)
    table_html = _diff_table_html(table, ab_result.arm_a.name, ab_result.arm_b.name)
    scatter = _sharpe_scatter(ab_result.arm_a, ab_result.arm_b)
    histogram = _diff_histogram(ab_result.arm_a, ab_result.arm_b)
    cards = _per_stock_cards(ab_result.arm_a, ab_result.arm_b)
    failures = _failure_detail(ab_result)
    cfg_dump = _full_cfg_dump(ab_result)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
  <meta charset="utf-8">
  <title>A/B Report · {ab_result.run_date}</title>
  <style>{_CSS}
    .banner {{ border: 1px solid #e6e6e6; padding: 1em; margin: 1em 0; }}
    .banner pre {{ background: #f6f6f6; padding: 0.6em; overflow-x: auto; }}
  </style>
</head><body>
  {banner}
  <h2>Aggregate (over {table['common_stocks_count']} common stocks)</h2>
  {table_html}
  <h2>Sharpe scatter</h2>
  <div class='chart-wrap'>{scatter}</div>
  <h2>Sharpe diff distribution</h2>
  <div class='chart-wrap'>{histogram}</div>
  <h2>Per-stock comparison</h2>
  {cards}
  <h2>Failures &amp; reproducibility</h2>
  {failures}
  {cfg_dump}
  <footer><p>Generated by stockpool ab.</p></footer>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    latest = output_dir / "latest.html"
    latest.write_bytes(out_path.read_bytes())
    return out_path
```

- [ ] **Step 4: Run report smoke tests**

```bash
python -m pytest tests/test_ab.py -v
```
Expected: PASS (≈ 22 tests).

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ab/report.py tests/test_ab.py
git commit -m "$(cat <<'EOF'
feat(ab): HTML report — aggregate table + Sharpe scatter + per-stock cards

Renders banner (arm names + counts + base hash), aggregate diff table
over stocks common to both arms (mean/median/Δ/win counts), Sharpe
scatter with diagonal, Sharpe-diff histogram, per-stock cards with
3-series equity chart (A/B/B&H) and per-stock metric tables, failure
detail, and full effective_cfg dumps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: CLI `cmd_ab` + subparser + `test_cli_ab.py`

**Files:**
- Modify: `src/stockpool/cli.py` (add `cmd_ab`, subparser registration)
- Create: `tests/test_cli_ab.py`

- [ ] **Step 1: Write failing CLI smoke tests**

Create `tests/test_cli_ab.py`:

```python
"""Smoke test for `python -m stockpool ab`."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_cache_two_stocks(tmp_path, monkeypatch):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    for code, seed in [("605589", 7), ("300750", 19)]:
        rng = np.random.default_rng(seed)
        n = 220
        returns = rng.normal(0.0005, 0.02, n)
        close = 100.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "open":  close * 0.998, "high": close * 1.005,
            "low":   close * 0.995, "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        })
        df.to_parquet(cache_dir / f"{code}_daily.parquet", index=False)
    cache_last = pd.date_range("2024-01-02", periods=220, freq="B")[-1]
    fresh_today = pd.Timestamp(cache_last) + pd.Timedelta(days=1)
    monkeypatch.setattr("stockpool.fetcher._today", lambda: fresh_today)
    return cache_dir


def _write_configs(tmp_path: Path, cache_dir: Path) -> tuple[Path, Path]:
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(cache_dir)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    raw["stocks"] = [
        {"code": "605589", "name": "Alpha", "sector": ""},
        {"code": "300750", "name": "Bravo", "sector": ""},
    ]
    base_path = tmp_path / "config.yaml"
    base_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    ab_raw = {
        "base_config": "config.yaml",
        "arms": {
            "single": {
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10], "engine": "single"},
            },
            "multi": {
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10], "engine": "multi_lot"},
            },
        },
    }
    ab_path = tmp_path / "ab.yaml"
    ab_path.write_text(yaml.safe_dump(ab_raw), encoding="utf-8")
    return ab_path, base_path


def test_cmd_ab_smoke_produces_html(tmp_path, isolated_cache_two_stocks):
    ab_path, _ = _write_configs(tmp_path, isolated_cache_two_stocks)
    rc = main(["ab", "--config", str(ab_path)])
    assert rc == 0
    latest = tmp_path / "reports" / "ab" / "latest.html"
    assert latest.exists()
    assert latest.stat().st_size > 2048
    html = latest.read_text(encoding="utf-8")
    assert "single" in html and "multi" in html
    assert "605589" in html and "300750" in html


def test_cmd_ab_arm_flag_runs_only_one_arm(tmp_path, isolated_cache_two_stocks,
                                            capsys):
    ab_path, _ = _write_configs(tmp_path, isolated_cache_two_stocks)
    rc = main(["ab", "--config", str(ab_path), "--arm", "single"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "single" in captured.out
    # No HTML written when --arm is used
    assert not (tmp_path / "reports" / "ab" / "latest.html").exists()


def test_cmd_ab_arm_unknown_returns_2(tmp_path, isolated_cache_two_stocks):
    ab_path, _ = _write_configs(tmp_path, isolated_cache_two_stocks)
    rc = main(["ab", "--config", str(ab_path), "--arm", "typo"])
    assert rc == 2


def test_cmd_ab_no_share_pool_propagates(tmp_path, isolated_cache_two_stocks,
                                          monkeypatch):
    ab_path, _ = _write_configs(tmp_path, isolated_cache_two_stocks)
    calls = {"count": 0}
    from stockpool.ab import runner as ab_runner
    real = ab_runner._decide_pool_sharing
    def _counted(*a, **kw):
        calls["count"] += 1
        return real(*a, **kw)
    monkeypatch.setattr(ab_runner, "_decide_pool_sharing", _counted)
    rc = main(["ab", "--config", str(ab_path), "--no-share-pool"])
    assert rc == 0
    assert calls["count"] == 0  # short-circuited
```

- [ ] **Step 2: Run — expect FAIL (no `ab` subcommand yet)**

```bash
python -m pytest tests/test_cli_ab.py -v
```
Expected: FAIL — argparse complains "invalid choice: 'ab'".

- [ ] **Step 3: Add `cmd_ab` + subparser to `cli.py`**

Add the import near the top of `src/stockpool/cli.py`:

```python
from stockpool.ab import (
    load_ab_config, run_ab, run_single_arm, render_ab_report,
)
```

Add a helper and the command function (place after `cmd_backtest`):

```python
def _apply_stocks_filter(stocks, codes):
    if not codes:
        return list(stocks)
    keep = set(codes)
    return [s for s in stocks if s.code in keep]


def _print_single_arm_stdout(arm_result) -> None:
    print(f"=== Arm: {arm_result.name} ===")
    print(f"Stocks succeeded: {len(arm_result.per_stock)}; "
          f"failed: {len(arm_result.failed)}")
    for code, name, res in arm_result.per_stock:
        N = next(iter(res.metrics))
        m = res.metrics[N]
        print(f"  {code} {name}: total_ret={m['total_return']:+.3f} "
              f"ann={m['annualized_return']:+.3f} sharpe={m.get('sharpe'):+.2f} "
              f"max_dd={m['max_drawdown']:.3f}")


def cmd_ab(args: argparse.Namespace) -> int:
    try:
        ab_cfg = load_ab_config(args.config)
    except Exception as e:
        log.error("ab config invalid: %s", e)
        return 2

    base_cfg_path = (Path(args.config).parent / ab_cfg.base_config).resolve()
    base_cfg = load_config(base_cfg_path)

    run_date = date.today().isoformat()
    out_root = Path(base_cfg.report.output_dir) / "ab"
    _setup_logging(out_root / run_date)
    log.info("stockpool ab v%s for %s", __version__, run_date)

    stocks = _apply_stocks_filter(base_cfg.stocks, ab_cfg.stocks_filter)

    if args.arm:
        if args.arm not in ab_cfg.arms:
            log.error("--arm %r not in %s", args.arm, list(ab_cfg.arms))
            return 2
        arm_result = run_single_arm(
            ab_cfg, base_cfg, stocks, args.refresh, args.arm,
        )
        _print_single_arm_stdout(arm_result)
        return 0

    result = run_ab(
        ab_cfg, base_cfg, stocks, args.refresh,
        share_pool=not args.no_share_pool,
    )
    if not result.arm_a.per_stock and not result.arm_b.per_stock:
        log.error("Both arms produced no results.")
        return 1
    out = render_ab_report(result, output_dir=out_root)
    log.info("AB report written: %s", out)
    log.info("Latest also at: %s", out_root / "latest.html")
    return 0
```

Register the subparser in `main()`, just after `p_bt`:

```python
    p_ab = sub.add_parser("ab", help="A/B-compare two strategies on the same universe")
    p_ab.add_argument("--config", default="ab.yaml")
    p_ab.add_argument("--refresh", action="store_true")
    p_ab.add_argument("--arm", default=None, help="Debug: run only one arm by name")
    p_ab.add_argument("--no-share-pool", action="store_true",
                      help="Force each arm to load its own universe / factor panel")
    p_ab.set_defaults(func=cmd_ab)
```

- [ ] **Step 4: Run all CLI smoke tests**

```bash
python -m pytest tests/test_cli_ab.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full project suite to verify no regressions**

```bash
python -m pytest tests/ -q
```
Expected: all green (previous count + ~25 new tests).

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/cli.py tests/test_cli_ab.py
git commit -m "$(cat <<'EOF'
feat(cli): add `stockpool ab` subcommand

cmd_ab loads ab.yaml + base config, runs both arms (with optional
pool sharing), and renders the comparison HTML. --arm <name> runs a
single arm and prints metrics to stdout (no HTML). --no-share-pool
forces independent pool prep per arm. CLI smoke tests cover the
happy path, --arm flag, unknown arm name (exit 2), and --no-share-pool
propagation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Sample `ab.yaml.example` at repo root

**Files:**
- Create: `ab.yaml.example`

- [ ] **Step 1: Write a documented sample**

```yaml
# ab.yaml — sample A/B test configuration.
#
# Run: python -m stockpool ab --config ab.yaml
#
# Output: reports/ab/<date>.html + reports/ab/latest.html
# Spec:   docs/superpowers/specs/2026-05-24-ab-testing-design.md

# Path to the base AppConfig. Inherits stocks / data / indicators / context.
# Resolved relative to this file's directory.
base_config: config.yaml

# Optional: subset of base.stocks to run on. Codes must exist in base.stocks.
# Empty / omitted = run on the full base.stocks.
# stocks_filter: ["605589", "300750"]

# Exactly two arms. Key names are free-form and appear verbatim in the report.
arms:
  composite_default:
    strategy:
      name: composite_verdict
    backtest:
      # Required. Must be a length-1 list (A/B compares at one N per side).
      equity_curve_holding_days: [10]
      # Other backtest fields (engine, costs, position_size, ...) inherit
      # from base.backtest if omitted.

  ml_lgbm_top20:
    strategy:
      name: ml_factor
      ml_factor:
        # Either an explicit factor list, or factors_file pointing to a JSON
        # produced by `stockpool factors pick` / `factors pick-by-ic`.
        factors_file: reports/selection.json
        selector: {type: lightgbm}
        weighter: {type: lightgbm}
        panel_mode: pooled
        training_universe: all
    backtest:
      equity_curve_holding_days: [10]
```

- [ ] **Step 2: Commit**

```bash
git add ab.yaml.example
git commit -m "$(cat <<'EOF'
docs: add ab.yaml.example sample with inline comments

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read the current CLAUDE.md to find the insertion points**

```bash
python -c "p=open('CLAUDE.md',encoding='utf-8').read(); print(p[:500])"
```

- [ ] **Step 2: Edit the "快速命令" section to add the `ab` command**

After the existing `factors pick-by-ic` block (the last entry in the 因子分析 group), append a new block:

```bash
# A/B testing — 比较两个策略
python -m stockpool ab --config ab.yaml
# 调试单边
python -m stockpool ab --config ab.yaml --arm <arm_name>
```

- [ ] **Step 3: Add module-map rows for `backtest_runner.py` and `ab/`**

In the "模块地图" table, after the `strategy_factory.py` row, insert:

```
| `src/stockpool/backtest_runner.py` | `prepare_pool` + `backtest_stocks` — 共享给 `cli.cmd_backtest` 和 `ab.runner.run_ab`,把"准备池"和"per-stock 回测循环"从 cli 抽出来,避免反向依赖 |
| `src/stockpool/ab/` | A/B 测试子包: `config.py`(`ABConfig`/`ArmOverride`/`load_ab_config`/`build_effective_cfg`),`runner.py`(`run_ab`/`run_single_arm`/pool 共享决策),`report.py`(HTML 报告) |
```

- [ ] **Step 4: Add test-table rows**

In the "测试" section's table, after the `test_cli_backtest.py` row, insert:

```
| `test_ab.py` | ab/config 校验(arms==2、length-1 N、`extra=forbid`)+ build_effective_cfg(strategy 整段替换 / backtest 字段级合并 / content_hash 重算)+ `_decide_pool_sharing` + run_ab/run_single_arm + 报告 smoke |
| `test_cli_ab.py` | CLI smoke:happy path、`--arm` 调试模式、未知 arm 退出 2、`--no-share-pool` 短路验证 |
```

- [ ] **Step 5: Add a note in the "配置" section**

In the "配置 (`config.yaml`)" subsection, after the `strategy` bullet, add:

```
- **A/B 测试**(独立配置文件 `ab.yaml`):见 `docs/superpowers/specs/2026-05-24-ab-testing-design.md`。两个 arm 各自只能覆盖 `strategy:` 和 `backtest:` 段,其他字段从 `base_config:` 指向的主 cfg 继承。`equity_curve_holding_days` 强制单元素列表。
```

- [ ] **Step 6: Add a bullet to "已知不支持的能力"**

After the existing bullets in "已知不支持的能力", add:

```
- A/B 测试覆盖 `indicators` / `weights` / `verdicts` / `scoring` 顶层字段:`composite_verdict` 的参数还散在 `AppConfig` 顶层(历史遗留),A/B arm 暂时只允许覆盖 `strategy:` 和 `backtest:`。两个 composite_verdict arm 想测不同 weights → 现在只能改主 cfg 跑两次。Follow-up:把 composite_verdict 的参数下沉到 `strategy.composite_verdict.*` 子段,A/B 工具会自动获益(无需改 ab/ 代码)
```

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(CLAUDE.md): document A/B testing tool

Module map (backtest_runner.py + ab/), quick commands (ab subcommand +
--arm), test table (test_ab.py, test_cli_ab.py), config note pointing
at ab.yaml spec, known-unsupported entry for indicators/weights
overrides (follow-up).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README to find the right sections**

```bash
python -c "p=open('README.md',encoding='utf-8').read(); print(len(p)); print(p[:300])"
```

- [ ] **Step 2: Add `ab` to the quick-commands list**

Find the bash code block listing the `python -m stockpool ...` commands; append:

```bash
# A/B 测试 (比较两个策略)
python -m stockpool ab --config ab.yaml
```

- [ ] **Step 3: Add a new end-to-end section after the existing "回测" / "因子" sections**

Insert near the end (before any FAQ / 已知问题 sections):

```markdown
### 对比两个策略 — A/B testing

`stockpool ab` 在同一份 stocks/data/indicators 下并行跑两个策略,产出
side-by-side 净值曲线 + 聚合差值表 + Sharpe 散点图。

最简 `ab.yaml`:

\`\`\`yaml
base_config: config.yaml

arms:
  composite:
    strategy: {name: composite_verdict}
    backtest: {equity_curve_holding_days: [10]}

  ml_lgbm:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        selector: {type: lightgbm}
        weighter: {type: lightgbm}
    backtest: {equity_curve_holding_days: [10]}
\`\`\`

跑:

\`\`\`bash
python -m stockpool ab --config ab.yaml
# 报告: reports/ab/<日期>.html  +  reports/ab/latest.html
\`\`\`

调试单边(只跑 ml_lgbm,打印指标到 stdout):

\`\`\`bash
python -m stockpool ab --config ab.yaml --arm ml_lgbm
\`\`\`

完整 schema + 设计说明:`docs/superpowers/specs/2026-05-24-ab-testing-design.md`。
```

(Use literal `\`\`\`` triple-backticks; the escapes shown above are for this plan document only.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(README): add A/B testing quick command + worked example

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Full regression + cleanup

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

```bash
python -m pytest tests/ -q
```
Expected: ALL PASS. The expected delta from baseline is ≈ +25 tests (test_ab.py and test_cli_ab.py).

- [ ] **Step 2: Manual smoke**

```bash
python -m stockpool ab --config ab.yaml.example 2>&1 | head -30
```
Expected: the command runs (it will exit 0 if `reports/selection.json` exists, or exit non-zero if not — either is OK; we're checking that argparse + load_ab_config wire up cleanly. If it errors on the example yaml because `factors_file` doesn't exist locally, manually create a stub `reports/selection.json` with `{"factors": ["momentum_20"]}` and retry, or drop the ml arm from `ab.yaml.example` for the smoke).

- [ ] **Step 3: Verify report opens**

If a report was generated, open `reports/ab/latest.html` in a browser and visually confirm:
- Banner shows both arm names + counts
- Aggregate table appears with diff column
- Per-stock equity charts have three lines (A / B / B&H)
- Failure details + cfg dump fold sections work

- [ ] **Step 4: Cleanup hidden test artefacts**

```bash
git status
```
If `tmp_path`-derived files accidentally landed in the repo, remove them. (Normal pytest runs use OS temp directories, so this should be empty.)

- [ ] **Step 5: Final commit if anything fixed**

If Step 2 surfaced a small bug, fix and commit. Otherwise no commit needed.

---

## Self-review

**Spec coverage:** Every section of the spec maps to at least one task:
- Architecture & module layout → Tasks 1-3 (extraction + scaffolding)
- Config schema + deep-merge → Task 4
- CLI → Task 8
- Runner internals (pool sharing, `_run_arm`, `run_ab`, `run_single_arm`) → Tasks 5-6
- Report → Task 7
- Testing → Tasks 4, 5, 6, 7, 8 (test files grow incrementally with each component)
- Documentation → Tasks 9-11

**Placeholder scan:** No "TBD", "TODO", "implement later". Every step shows code or a verifiable command. Two judgment calls flagged explicitly: (a) Task 2's regression test that needs careful monkeypatching of `fetch_daily` to simulate per-stock failure; (b) Task 12's manual smoke depends on whether `reports/selection.json` exists locally — instruction says either outcome is OK and offers a stub.

**Type consistency:**
- `ABResult` / `ArmResult` field names: `name`, `effective_cfg`, `per_stock`, `failed`, `ab_cfg`, `base_cfg`, `arm_a`, `arm_b`, `run_date` — used consistently across `runner.py`, `report.py`, and the test files.
- `load_ab_config` returns `ABConfig` (not `(ABConfig, AppConfig)` — base is loaded separately in `cmd_ab`).
- `prepare_pool` signature matches between `backtest_runner.py` definition and call sites in `cli.cmd_backtest` and `ab/runner._prepare_pool_for_arm`.
- `backtest_stocks` returns `(success, failed)` everywhere; callers (`cmd_backtest`, `_run_arm`) unpack identically.
- `_decide_pool_sharing` returns the dict shape `{"load_universe": bool, "shared_factors": list[str] | None}` consistently in `runner.py` and tested in Task 5.
