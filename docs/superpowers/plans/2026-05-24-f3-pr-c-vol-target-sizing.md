# F3 PR-C — Vol-target Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `BacktestConfig.position_size=0.1` with a `SizingConfig` sub-block (`fixed | vol_target`, default `vol_target`), so each lot's size scales inversely to that stock's recent realised volatility — making per-stock risk contribution more uniform and (per A/B gate) cutting portfolio max DD ≥ 20%.

**Architecture:** New `backtesting/sizing.py` exposes `LotSizer` Protocol + `FixedLotSizer` / `VolTargetLotSizer` + `build_lot_sizer` factory (config-agnostic, duck-typed). `MultiLotBacktestEngine` accepts `lot_sizer=` callable; `Trade` gains a `lot_size` attribution field. Five top-level call sites (`cli`, `backtest_runner`, `backtest_composite`, `strategy_factory`, `ab/config`) get rewired. Old `BacktestConfig.position_size` kept as deprecated-with-migration alias.

**Tech Stack:** Python 3.11+, Pydantic v2, pandas, numpy. No new dependencies.

**Source spec:** `docs/superpowers/specs/2026-05-24-f3-pr-c-vol-target-sizing-design.md`

**Worktree note:** This branch (`feat/composite-backtest`) has WIP uncommitted files (`ab.yaml`, `docs/ab_runs/`, `scripts/parse_ab_report.py`, modified `config.yaml`). Consider creating a git worktree before execution to isolate this PR's diffs (`superpowers:using-git-worktrees`). If executing in-place, leave the WIP untouched.

---

## File Structure

**New files:**
- `src/stockpool/backtesting/sizing.py` — LotSizer Protocol + Fixed/VolTarget implementations + `build_lot_sizer` factory
- `tests/test_sizing.py` — sizing unit tests
- `ab_sizing.yaml` — A/B config for PR-C validation

**Modified files:**
- `src/stockpool/backtesting/framework.py` — `MultiLotBacktestEngine.__init__` accepts `lot_sizer`; `_OpenLot` / `Trade` add `lot_size`; `_simulate_multi_lot` consumes the sizer
- `src/stockpool/backtesting/__init__.py` — export `LotSizer`, `FixedLotSizer`, `VolTargetLotSizer`, `build_lot_sizer`
- `src/stockpool/config.py` — `FixedSizingConfig` / `VolTargetSizingConfig` / `SizingConfig` classes; `BacktestConfig.sizing` field + `position_size` deprecated + `_migrate_position_size` validator
- `src/stockpool/cli.py` — `cmd_backtest` wires `lot_sizer=build_lot_sizer(cfg.backtest.sizing)`; engine_label string
- `src/stockpool/backtest_runner.py` — `backtest_stocks` passes lot_sizer through
- `src/stockpool/backtest_composite.py` — `simulate_equity_curve` signature: `lot_sizer` kwarg + deprecated `position_size`
- `src/stockpool/strategy_factory.py` — `simulate_strategy_equity_curve` signature: same as above
- `src/stockpool/ab/config.py` — `ArmBacktestOverride.sizing: SizingConfig | None = None`
- `tests/test_multi_lot_engine.py` — new tests for `lot_sizer=` injection + `Trade.lot_size`
- `tests/test_backtest_composite.py` — fixture update: explicit `lot_sizer` or `sizing.type=fixed`
- `tests/test_timer_reset.py` — fixture update: pass `position_size=0.1` keyword still works (engine handles both)
- `tests/test_ab.py` — new case for `ArmBacktestOverride.sizing` merge
- `tests/test_config.py` — new cases for sizing schema + migration + conflicts
- `config.yaml` — replace inline `position_size: 0.1` with `sizing:` block
- `CLAUDE.md` — module map + config docs + test table
- `README.md` — config example
- `docs/strategy_improvement_2026.md` — §6 P1 PR-C row: 🚧 → ✅ + verdict

---

## Task 1: Sizing module (standalone)

**Files:**
- Create: `src/stockpool/backtesting/sizing.py`
- Create: `tests/test_sizing.py`
- Modify: `src/stockpool/backtesting/__init__.py`

### Step 1.1: Write failing tests for sizing module

- [ ] Create `tests/test_sizing.py` with this content:

```python
"""Tests for backtesting.sizing — LotSizer implementations and factory."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from stockpool.backtesting.sizing import (
    FixedLotSizer,
    VolTargetLotSizer,
    build_lot_sizer,
)


# ============================================================================
# FixedLotSizer
# ============================================================================

def test_fixed_lot_sizer_returns_constant():
    sizer = FixedLotSizer(0.15)
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    opens = closes.copy()
    assert sizer(0, opens, closes) == 0.15
    assert sizer(4, opens, closes) == 0.15


def test_fixed_lot_sizer_validates_size():
    with pytest.raises(ValueError):
        FixedLotSizer(0.0)
    with pytest.raises(ValueError):
        FixedLotSizer(1.5)
    with pytest.raises(ValueError):
        FixedLotSizer(-0.1)


# ============================================================================
# VolTargetLotSizer — basic math
# ============================================================================

def _make_vol_sizer(**overrides) -> VolTargetLotSizer:
    defaults = dict(
        baseline_size=0.1,
        reference_vol_annual=0.30,
        vol_window=20,
        min_size=0.03,
        max_size=0.20,
        fallback="fixed",
    )
    defaults.update(overrides)
    return VolTargetLotSizer(**defaults)


def test_vol_target_low_vol_raises_size():
    """Stock with vol < reference → size > baseline (capped by max_size)."""
    sizer = _make_vol_sizer()
    # 21 closes producing 20 returns of constant +0.005 (low vol = ~8% annual)
    # constant returns → std=0 → falls back. Use small noise instead.
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.005, size=20)
    # build close path from returns
    closes = np.empty(21)
    closes[0] = 100.0
    for i in range(20):
        closes[i + 1] = closes[i] * (1 + rets[i])
    # bar_idx=21 means: window = closes[0:21], returns from those 21 closes
    # daily vol ≈ 0.005, annualised ≈ 0.005 * sqrt(252) ≈ 0.079
    # raw = 0.1 * 0.30 / 0.079 ≈ 0.38 → clipped to max_size=0.20
    size = sizer(21, closes.copy(), closes)
    assert size == pytest.approx(0.20, abs=1e-9), (
        f"low vol should be clipped to max_size, got {size}"
    )


def test_vol_target_high_vol_lowers_size():
    """Stock with vol > reference → size < baseline."""
    sizer = _make_vol_sizer()
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.05, size=20)  # daily ~5%, annual ~80%
    closes = np.empty(21)
    closes[0] = 100.0
    for i in range(20):
        closes[i + 1] = closes[i] * (1 + rets[i])
    # raw = 0.1 * 0.30 / 0.80 ≈ 0.038 — between min and max
    size = sizer(21, closes.copy(), closes)
    assert 0.03 <= size < 0.10, f"high vol should reduce size below 0.10, got {size}"


def test_vol_target_clip_to_min_size():
    """Very-high-vol stock → clipped to min_size."""
    sizer = _make_vol_sizer()
    rng = np.random.default_rng(2)
    rets = rng.normal(0.0, 0.15, size=20)  # extreme daily vol
    closes = np.empty(21)
    closes[0] = 100.0
    for i in range(20):
        closes[i + 1] = closes[i] * (1 + rets[i])
    size = sizer(21, closes.copy(), closes)
    assert size == pytest.approx(0.03, abs=1e-9), (
        f"extreme vol should clip to min_size, got {size}"
    )


# ============================================================================
# VolTargetLotSizer — fallback behavior
# ============================================================================

def test_vol_target_cold_start_uses_fixed_fallback():
    """bar_idx < vol_window+1 → fallback to baseline_size."""
    sizer = _make_vol_sizer()
    closes = np.array([100.0] * 30)
    # bar_idx=20 means we want closes[20-20-1:20] = closes[-1:20] → invalid
    # actually bar_idx < vol_window+1 = 21 triggers fallback
    for bi in [0, 1, 10, 20]:
        assert sizer(bi, closes, closes) == pytest.approx(0.1)


def test_vol_target_skip_fallback_returns_zero():
    sizer = _make_vol_sizer(fallback="skip")
    closes = np.array([100.0] * 30)
    assert sizer(5, closes, closes) == 0.0


def test_vol_target_nan_in_window_uses_fallback():
    sizer = _make_vol_sizer()
    closes = np.array([100.0] * 30)
    closes[10] = np.nan
    # window for bar_idx=25 is closes[4:25] → contains NaN → fallback
    assert sizer(25, closes.copy(), closes) == pytest.approx(0.1)


def test_vol_target_zero_close_uses_fallback():
    sizer = _make_vol_sizer()
    closes = np.array([100.0] * 30)
    closes[10] = 0.0
    assert sizer(25, closes.copy(), closes) == pytest.approx(0.1)


def test_vol_target_zero_vol_uses_fallback():
    """Constant prices → std=0 → fallback."""
    sizer = _make_vol_sizer()
    closes = np.array([100.0] * 30)
    assert sizer(25, closes, closes) == pytest.approx(0.1)


def test_vol_target_skip_fallback_at_cold_start():
    sizer = _make_vol_sizer(fallback="skip")
    closes = np.array([100.0] * 30)
    assert sizer(5, closes, closes) == 0.0


# ============================================================================
# VolTargetLotSizer — validation
# ============================================================================

def test_vol_target_rejects_invalid_fallback():
    with pytest.raises(ValueError):
        VolTargetLotSizer(
            baseline_size=0.1, reference_vol_annual=0.3, vol_window=20,
            min_size=0.03, max_size=0.20, fallback="bogus",
        )


# ============================================================================
# build_lot_sizer factory
# ============================================================================

def test_build_lot_sizer_fixed():
    cfg = SimpleNamespace(
        type="fixed",
        fixed=SimpleNamespace(size=0.12),
        vol_target=SimpleNamespace(),
    )
    s = build_lot_sizer(cfg)
    assert isinstance(s, FixedLotSizer)
    assert s.size == 0.12


def test_build_lot_sizer_vol_target():
    cfg = SimpleNamespace(
        type="vol_target",
        fixed=SimpleNamespace(size=0.1),
        vol_target=SimpleNamespace(
            reference_vol_annual=0.25,
            vol_window=15,
            min_size=0.02,
            max_size=0.30,
            fallback_to="skip",
        ),
    )
    s = build_lot_sizer(cfg)
    assert isinstance(s, VolTargetLotSizer)
    assert s.baseline_size == 0.1
    assert s.reference_vol_annual == 0.25
    assert s.vol_window == 15
    assert s.min_size == 0.02
    assert s.max_size == 0.30
    assert s.fallback == "skip"


def test_build_lot_sizer_unknown_type():
    cfg = SimpleNamespace(
        type="bogus",
        fixed=SimpleNamespace(size=0.1),
        vol_target=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="Unknown sizing.type"):
        build_lot_sizer(cfg)
```

### Step 1.2: Run tests to verify they fail

- [ ] Run: `python -m pytest tests/test_sizing.py -v`
- [ ] Expected: ImportError / collection failure (`sizing` module does not exist yet)

### Step 1.3: Create the sizing module

- [ ] Create `src/stockpool/backtesting/sizing.py` with this content:

```python
"""Lot sizing strategies for MultiLotBacktestEngine.

A LotSizer is a callable that, given the engine's current bar index and the
recent OHLC arrays, returns the fraction of starting capital to commit on the
next entry. Engine remains config-agnostic: callers build a LotSizer from
config (via build_lot_sizer below) and inject it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from stockpool.config import SizingConfig


class LotSizer(Protocol):
    """Callable that returns a lot size in (0, 1] (or 0 = skip)."""

    def __call__(
        self, bar_idx: int, opens: np.ndarray, closes: np.ndarray
    ) -> float:
        """Return the lot size for a buy executed at bar_idx.

        Convention: bar_idx is the EXECUTION bar (where the fill happens at
        opens[bar_idx]). The sizer may only inspect closes[:bar_idx] — it must
        not peek at the execution bar's close, preserving the engine's T+1
        look-ahead safety contract.
        """


class FixedLotSizer:
    """Returns a constant size regardless of state."""

    def __init__(self, size: float):
        if not (0 < size <= 1.0):
            raise ValueError(f"size must be in (0, 1], got {size}")
        self.size = size

    def __call__(
        self, bar_idx: int, opens: np.ndarray, closes: np.ndarray
    ) -> float:
        return self.size


class VolTargetLotSizer:
    """Scales size inversely to recent realised volatility (β formula).

    size = baseline_size * (reference_vol_annual / recent_vol_annual)
    size = clip(size, min_size, max_size)

    Recent vol estimated as the simple rolling std of daily simple returns on
    the most recent vol_window+1 closes ending at bar_idx-1, annualised with
    sqrt(252).

    Fallback (cold-start, NaN, or zero vol):
      - "fixed": return baseline_size (clipped to [min_size, max_size])
      - "skip":  return 0.0 (engine should skip the buy)
    """

    _ANNUALISATION = 252.0

    def __init__(
        self,
        baseline_size: float,
        reference_vol_annual: float,
        vol_window: int,
        min_size: float,
        max_size: float,
        fallback: str = "fixed",
    ):
        if fallback not in ("fixed", "skip"):
            raise ValueError(
                f"fallback must be 'fixed' or 'skip', got {fallback!r}"
            )
        self.baseline_size = baseline_size
        self.reference_vol_annual = reference_vol_annual
        self.vol_window = vol_window
        self.min_size = min_size
        self.max_size = max_size
        self.fallback = fallback

    def _fallback_size(self) -> float:
        if self.fallback == "skip":
            return 0.0
        return float(np.clip(self.baseline_size, self.min_size, self.max_size))

    def __call__(
        self, bar_idx: int, opens: np.ndarray, closes: np.ndarray
    ) -> float:
        # Need vol_window+1 closes (yielding vol_window returns) ending at bar_idx-1.
        if bar_idx < self.vol_window + 1:
            return self._fallback_size()
        window_closes = closes[bar_idx - self.vol_window - 1 : bar_idx]
        if np.any(~np.isfinite(window_closes)) or np.any(window_closes <= 0):
            return self._fallback_size()
        rets = np.diff(window_closes) / window_closes[:-1]
        if rets.size == 0:
            return self._fallback_size()
        recent_vol_daily = float(np.std(rets, ddof=1))
        if not np.isfinite(recent_vol_daily) or recent_vol_daily <= 0:
            return self._fallback_size()
        recent_vol_annual = recent_vol_daily * np.sqrt(self._ANNUALISATION)
        raw = self.baseline_size * (self.reference_vol_annual / recent_vol_annual)
        return float(np.clip(raw, self.min_size, self.max_size))


def build_lot_sizer(cfg: "SizingConfig") -> LotSizer:
    """Build a LotSizer from a SizingConfig (or duck-typed equivalent).

    Pure factory; no I/O, no side effects. Runtime accepts any object with the
    expected attribute shape — see VolTargetLotSizer args for the contract.
    Lives in backtesting.sizing to keep the dependency direction
    strategy_factory → sizing → config (config types are referenced via
    TYPE_CHECKING only).
    """
    if cfg.type == "fixed":
        return FixedLotSizer(cfg.fixed.size)
    if cfg.type == "vol_target":
        vt = cfg.vol_target
        return VolTargetLotSizer(
            baseline_size=cfg.fixed.size,
            reference_vol_annual=vt.reference_vol_annual,
            vol_window=vt.vol_window,
            min_size=vt.min_size,
            max_size=vt.max_size,
            fallback=vt.fallback_to,
        )
    raise ValueError(f"Unknown sizing.type: {cfg.type!r}")
```

### Step 1.4: Run tests to verify they pass

- [ ] Run: `python -m pytest tests/test_sizing.py -v`
- [ ] Expected: all tests pass

### Step 1.5: Export sizing API from package __init__

- [ ] Modify `src/stockpool/backtesting/__init__.py`. After the `from stockpool.backtesting.strategies import ...` block, append:

```python
from stockpool.backtesting.sizing import (
    FixedLotSizer,
    LotSizer,
    VolTargetLotSizer,
    build_lot_sizer,
)
```

And add these names to the `__all__` list (after the existing entries):

```python
    "FixedLotSizer",
    "LotSizer",
    "VolTargetLotSizer",
    "build_lot_sizer",
```

### Step 1.6: Run full sizing test + import smoke

- [ ] Run: `python -m pytest tests/test_sizing.py -v && python -c "from stockpool.backtesting import FixedLotSizer, VolTargetLotSizer, build_lot_sizer; print('ok')"`
- [ ] Expected: all sizing tests pass; smoke prints "ok"

### Step 1.7: Commit

- [ ] Run:

```bash
git add src/stockpool/backtesting/sizing.py src/stockpool/backtesting/__init__.py tests/test_sizing.py
git commit -m "$(cat <<'EOF'
feat(sizing): add LotSizer Protocol + Fixed/VolTarget sizers + build_lot_sizer factory

Standalone module (no engine touch yet). VolTargetLotSizer implements
the β formula: size = baseline * (reference_vol / recent_vol), clipped
to [min_size, max_size], with fixed/skip fallback for cold-start and
degenerate vol cases. Factory is duck-typed at runtime via
TYPE_CHECKING reference to SizingConfig — sizing module has no runtime
dependency on Pydantic config.

Spec: docs/superpowers/specs/2026-05-24-f3-pr-c-vol-target-sizing-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Engine integration

**Files:**
- Modify: `src/stockpool/backtesting/framework.py`
- Modify: `tests/test_multi_lot_engine.py`

### Step 2.1: Write failing tests for engine integration

- [ ] Add the following to `tests/test_multi_lot_engine.py` (append at end of file):

```python
# ============================================================================
# lot_sizer injection (PR-C)
# ============================================================================

from stockpool.backtesting.sizing import FixedLotSizer, VolTargetLotSizer


def test_engine_accepts_lot_sizer_kwarg():
    """Constructing with lot_sizer= works and overrides default sizing."""
    engine = MultiLotBacktestEngine(
        VerdictExecution(),
        lot_sizer=FixedLotSizer(0.25),
    )
    assert engine.lot_sizer.size == 0.25


def test_engine_default_when_neither_provided():
    """No lot_sizer, no position_size → default FixedLotSizer(0.1)."""
    engine = MultiLotBacktestEngine(VerdictExecution())
    assert isinstance(engine.lot_sizer, FixedLotSizer)
    assert engine.lot_sizer.size == 0.1


def test_engine_rejects_both_position_size_and_lot_sizer():
    with pytest.raises(ValueError, match="Pass either"):
        MultiLotBacktestEngine(
            VerdictExecution(),
            position_size=0.1,
            lot_sizer=FixedLotSizer(0.2),
        )


def test_engine_position_size_keyword_still_works():
    """Backwards compatibility: position_size= alone wraps in FixedLotSizer."""
    engine = MultiLotBacktestEngine(VerdictExecution(), position_size=0.15)
    assert isinstance(engine.lot_sizer, FixedLotSizer)
    assert engine.lot_sizer.size == 0.15


def test_trade_lot_size_recorded():
    """Trade.lot_size is populated from the active sizer at entry."""
    sigs = _signals(
        ["buy", "hold", "hold", "hold", "hold"],
        [100, 100, 100, 100, 100],
    )
    engine = MultiLotBacktestEngine(
        VerdictExecution(), lot_sizer=FixedLotSizer(0.07),
    )
    r = engine.run_on_signals(sigs, max_holding_days=3)
    assert len(r.trades) == 1
    assert r.trades[0].lot_size == pytest.approx(0.07)


def test_vol_target_dynamic_sizing_records_per_trade_size():
    """With vol_target, each trade's lot_size reflects vol at that bar."""
    # 30 bars of mild noise + 1 buy signal late enough for vol calc to kick in.
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0, 0.01, size=30)
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * (1 + r))
    sigs = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=31, freq="B"),
        "close": closes,
        "signal": ["hold"] * 25 + ["buy"] + ["hold"] * 5,
    })
    sizer = VolTargetLotSizer(
        baseline_size=0.1, reference_vol_annual=0.30,
        vol_window=20, min_size=0.03, max_size=0.20, fallback="fixed",
    )
    engine = MultiLotBacktestEngine(VerdictExecution(), lot_sizer=sizer)
    r = engine.run_on_signals(sigs, max_holding_days=3)
    assert len(r.trades) == 1
    # daily vol ~1%, annual ~16%; raw size = 0.1 * 0.30 / 0.16 ≈ 0.19 → near max
    assert 0.05 < r.trades[0].lot_size <= 0.20


def test_skip_fallback_zero_size_skips_buy():
    """When sizer returns 0 (skip fallback during cold-start), no lot opens."""
    sigs = _signals(
        ["buy", "hold", "hold"],
        [100, 100, 100],
    )
    sizer = VolTargetLotSizer(
        baseline_size=0.1, reference_vol_annual=0.30,
        vol_window=20, min_size=0.03, max_size=0.20, fallback="skip",
    )
    engine = MultiLotBacktestEngine(VerdictExecution(), lot_sizer=sizer)
    r = engine.run_on_signals(sigs, max_holding_days=3)
    assert r.metrics["trade_count"] == 0
    assert (r.curve["position"] == 0).all()


def test_size_exceeds_cash_skips_buy():
    """Sizer returns size > available cash → buy skipped (no partial fill)."""
    sigs = _signals(
        ["buy", "buy", "buy", "hold"],
        [100, 100, 100, 100],
    )
    # Each lot = 0.5 → first two consume all cash, third must skip.
    engine = MultiLotBacktestEngine(
        VerdictExecution(), lot_sizer=FixedLotSizer(0.5),
    )
    r = engine.run_on_signals(sigs, max_holding_days=10)
    # Only 2 lots opened (cash=1.0 → 0.5 → 0.0 < 0.5)
    assert r.curve["position"].iloc[3] == 2
```

Also add this import at the top of the file (right after the existing `import pytest`):

```python
import numpy as np
```

### Step 2.2: Run tests to verify they fail

- [ ] Run: `python -m pytest tests/test_multi_lot_engine.py -v`
- [ ] Expected: new tests fail with `TypeError: unexpected keyword 'lot_sizer'` or `AttributeError: 'Trade' object has no attribute 'lot_size'`

### Step 2.3: Update `Trade` and `_OpenLot` to carry `lot_size`

- [ ] In `src/stockpool/backtesting/framework.py`, modify the `Trade` dataclass:

Replace:

```python
@dataclass(frozen=True)
class Trade:
    """One closed long position."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    ret: float            # net of buy_cost and sell_cost
    days_held: int
```

With:

```python
@dataclass(frozen=True)
class Trade:
    """One closed long position."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    ret: float            # net of buy_cost and sell_cost
    days_held: int
    lot_size: float = 0.1
```

And modify `_OpenLot`:

Replace:

```python
@dataclass
class _OpenLot:
    """Internal: one open lot in the multi-lot engine."""
    entry_idx: int
    entry_price: float
    committed_cash: float    # cash actually invested, AFTER buy_cost
    current_value: float     # mark-to-market value of this lot
    days_held: int = 0
```

With:

```python
@dataclass
class _OpenLot:
    """Internal: one open lot in the multi-lot engine."""
    entry_idx: int
    entry_price: float
    committed_cash: float    # cash actually invested, AFTER buy_cost
    current_value: float     # mark-to-market value of this lot
    days_held: int = 0
    lot_size: float = 0.1
```

### Step 2.4: Replace `MultiLotBacktestEngine.__init__` signature

- [ ] In `src/stockpool/backtesting/framework.py`, find the `MultiLotBacktestEngine.__init__` method (around line 385) and replace the entire `__init__`:

Replace:

```python
    def __init__(
        self,
        strategy: Strategy,
        position_size: float,
        costs: TradeCosts = TradeCosts(),
        risk_free_rate: float = 0.02,
        max_concurrent_lots: int | None = None,
    ):
        if not (0 < position_size <= 1.0):
            raise ValueError(
                f"position_size must be in (0, 1], got {position_size}"
            )
        self.strategy = strategy
        self.position_size = position_size
        self.costs = costs
        self.risk_free_rate = risk_free_rate
        self.max_concurrent_lots = max_concurrent_lots
```

With:

```python
    def __init__(
        self,
        strategy: Strategy,
        position_size: float | None = None,
        lot_sizer: "LotSizer | None" = None,
        costs: TradeCosts = TradeCosts(),
        risk_free_rate: float = 0.02,
        max_concurrent_lots: int | None = None,
    ):
        if lot_sizer is not None and position_size is not None:
            raise ValueError(
                "Pass either `lot_sizer` or `position_size`, not both. "
                "`position_size` is deprecated; prefer "
                "`lot_sizer=FixedLotSizer(size)`."
            )
        if lot_sizer is None:
            # Bare engine call (legacy) — wrap fixed size.
            size = position_size if position_size is not None else 0.1
            from stockpool.backtesting.sizing import FixedLotSizer
            lot_sizer = FixedLotSizer(size)
        self.strategy = strategy
        self.lot_sizer = lot_sizer
        self.costs = costs
        self.risk_free_rate = risk_free_rate
        self.max_concurrent_lots = max_concurrent_lots
```

Then add a forward-reference type-only import at the top of `framework.py` (right after the existing `from typing import Any, Sequence` line):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockpool.backtesting.sizing import LotSizer
```

(If `TYPE_CHECKING` is already imported, just add the conditional block.)

### Step 2.5: Update `MultiLotBacktestEngine.run_on_signals` to pass `lot_sizer`

- [ ] In `framework.py`, replace:

```python
    def run_on_signals(
        self, signals: pd.DataFrame, max_holding_days: int,
    ) -> BacktestResult:
        return _simulate_multi_lot(
            signals,
            strategy=self.strategy,
            position_size=self.position_size,
            max_concurrent_lots=self.max_concurrent_lots,
            max_holding_days=max_holding_days,
            costs=self.costs,
            risk_free_rate=self.risk_free_rate,
        )
```

With:

```python
    def run_on_signals(
        self, signals: pd.DataFrame, max_holding_days: int,
    ) -> BacktestResult:
        return _simulate_multi_lot(
            signals,
            strategy=self.strategy,
            lot_sizer=self.lot_sizer,
            max_concurrent_lots=self.max_concurrent_lots,
            max_holding_days=max_holding_days,
            costs=self.costs,
            risk_free_rate=self.risk_free_rate,
        )
```

### Step 2.6: Update `_simulate_multi_lot` signature and lot opening

- [ ] In `framework.py`, replace the `_simulate_multi_lot` signature header:

Replace:

```python
def _simulate_multi_lot(
    signals: pd.DataFrame,
    *,
    strategy: Strategy,
    position_size: float,
    max_concurrent_lots: int | None,
    max_holding_days: int,
    costs: TradeCosts,
    risk_free_rate: float,
) -> BacktestResult:
```

With:

```python
def _simulate_multi_lot(
    signals: pd.DataFrame,
    *,
    strategy: Strategy,
    lot_sizer: "LotSizer",
    max_concurrent_lots: int | None,
    max_holding_days: int,
    costs: TradeCosts,
    risk_free_rate: float,
) -> BacktestResult:
```

- [ ] Then find the lot-opening block in `_simulate_multi_lot` (step "4. Maybe open a new lot"):

Replace:

```python
        # 4. Maybe open a new lot — fills at open[t]; first-day exposure is
        #    open[t] → close[t].
        bctx = BarContext(
            bar_idx=t - 1, date=prev_date,
            close=prev_close, signal=prev_signal,
        )
        capacity_ok = (
            max_concurrent_lots is None
            or len(open_lots) < max_concurrent_lots
        )
        if strategy.should_enter(bctx) and cash >= position_size and capacity_ok:
            cash -= position_size
            committed = position_size * (1 - costs.buy_cost)
            open_lots.append(_OpenLot(
                entry_idx=t,
                entry_price=open_t,
                committed_cash=committed,
                current_value=committed * (close_t / open_t),
                days_held=0,
            ))
```

With:

```python
        # 4. Maybe open a new lot — fills at open[t]; first-day exposure is
        #    open[t] → close[t]. Lot size now comes from the sizer (which sees
        #    closes up to bar t-1, preserving look-ahead safety).
        bctx = BarContext(
            bar_idx=t - 1, date=prev_date,
            close=prev_close, signal=prev_signal,
        )
        capacity_ok = (
            max_concurrent_lots is None
            or len(open_lots) < max_concurrent_lots
        )
        if strategy.should_enter(bctx) and capacity_ok:
            size = lot_sizer(t, opens, closes)
            if size > 0 and cash >= size:
                cash -= size
                committed = size * (1 - costs.buy_cost)
                open_lots.append(_OpenLot(
                    entry_idx=t,
                    entry_price=open_t,
                    committed_cash=committed,
                    current_value=committed * (close_t / open_t),
                    days_held=0,
                    lot_size=size,
                ))
```

### Step 2.7: Propagate `lot_size` from `_OpenLot` to `Trade`

- [ ] In `framework.py`, find the `Trade(...)` construction inside `_simulate_multi_lot` (the per-lot exit block, around step 2):

Replace:

```python
                trades.append(Trade(
                    entry_idx=lot.entry_idx,
                    exit_idx=t,
                    entry_price=lot.entry_price,
                    exit_price=open_t,
                    ret=float(exit_value / lot.committed_cash - 1),
                    days_held=lot.days_held,
                ))
```

With:

```python
                trades.append(Trade(
                    entry_idx=lot.entry_idx,
                    exit_idx=t,
                    entry_price=lot.entry_price,
                    exit_price=open_t,
                    ret=float(exit_value / lot.committed_cash - 1),
                    days_held=lot.days_held,
                    lot_size=lot.lot_size,
                ))
```

### Step 2.8: Run engine tests to verify they pass

- [ ] Run: `python -m pytest tests/test_multi_lot_engine.py -v`
- [ ] Expected: all tests pass (including the existing ones — the `position_size=` kwarg backward-compat path keeps them working)

### Step 2.9: Run the full backtesting test suite to confirm zero regression

- [ ] Run: `python -m pytest tests/test_backtesting_framework.py tests/test_multi_lot_engine.py tests/test_timer_reset.py tests/test_backtest_composite.py -v`
- [ ] Expected: all tests pass

### Step 2.10: Commit

- [ ] Run:

```bash
git add src/stockpool/backtesting/framework.py tests/test_multi_lot_engine.py
git commit -m "$(cat <<'EOF'
feat(backtesting): MultiLotBacktestEngine accepts LotSizer; Trade.lot_size attribution

Engine __init__ now takes lot_sizer= (preferred) or position_size=
(deprecated, wrapped in FixedLotSizer internally). Old call sites work
unchanged. _simulate_multi_lot threads the sizer into the lot-open step
and skips entries when sizer returns 0 or size > cash.

Trade and _OpenLot gain a lot_size field (default 0.1 for fixture
compat) so per-trade attribution survives into BacktestResult — A/B
reports can derive "what was saved by vol-adjusting which stocks."

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Config schema + migration

**Files:**
- Modify: `src/stockpool/config.py`
- Modify: `tests/test_config.py`

### Step 3.1: Write failing tests for config schema

- [ ] Add the following to `tests/test_config.py` (append at end):

```python
# ============================================================================
# F3 PR-C — SizingConfig + position_size deprecation
# ============================================================================

import warnings
from stockpool.config import (
    BacktestConfig, SizingConfig, FixedSizingConfig, VolTargetSizingConfig,
)


def test_sizing_config_defaults():
    """Default sizing.type is vol_target with all-default sub-fields."""
    s = SizingConfig()
    assert s.type == "vol_target"
    assert s.fixed.size == 0.1
    assert s.vol_target.reference_vol_annual == 0.30
    assert s.vol_target.vol_window == 20
    assert s.vol_target.min_size == 0.03
    assert s.vol_target.max_size == 0.20
    assert s.vol_target.fallback_to == "fixed"


def test_sizing_config_rejects_extra_fields():
    """SizingConfig has extra='forbid'."""
    with pytest.raises(ValidationError):
        SizingConfig(type="fixed", unknown_field=1)


def test_vol_target_rejects_min_gt_max():
    with pytest.raises(ValidationError):
        VolTargetSizingConfig(min_size=0.5, max_size=0.1)


def test_vol_target_rejects_invalid_fallback():
    with pytest.raises(ValidationError):
        VolTargetSizingConfig(fallback_to="bogus")


def test_backtest_config_default_sizing_is_vol_target():
    bt = BacktestConfig(forward_days=[5], equity_curve_holding_days=[5])
    assert bt.sizing.type == "vol_target"
    assert bt.position_size is None


def test_position_size_alone_migrates_with_deprecation_warning():
    """Setting position_size= triggers DeprecationWarning + auto-migration."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bt = BacktestConfig(
            forward_days=[5], equity_curve_holding_days=[5],
            position_size=0.15,
        )
        depr = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(depr) == 1
        assert "position_size" in str(depr[0].message)
    assert bt.sizing.type == "fixed"
    assert bt.sizing.fixed.size == 0.15
    assert bt.position_size is None  # cleared after migration


def test_position_size_alone_at_default_value_still_migrates():
    """Even position_size=0.1 (= default) triggers migration."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bt = BacktestConfig(
            forward_days=[5], equity_curve_holding_days=[5],
            position_size=0.1,
        )
    assert bt.sizing.type == "fixed"
    assert bt.sizing.fixed.size == 0.1


def test_position_size_plus_explicit_sizing_type_raises():
    """Explicit non-default sizing.type alongside position_size → conflict."""
    with pytest.raises(ValidationError, match="position_size"):
        BacktestConfig(
            forward_days=[5], equity_curve_holding_days=[5],
            position_size=0.1,
            sizing=SizingConfig(type="vol_target"),  # type defaults to vol_target
            # But we want to assert the FORCE-conflict path:
            # actually with type=vol_target (= default), it's not detected.
            # Use type=fixed with a non-default size instead.
        )


def test_position_size_plus_explicit_sizing_fixed_size_raises():
    """Explicit non-default sizing.fixed.size alongside position_size → conflict."""
    with pytest.raises(ValidationError, match="position_size"):
        BacktestConfig(
            forward_days=[5], equity_curve_holding_days=[5],
            position_size=0.1,
            sizing=SizingConfig(
                type="fixed",
                fixed=FixedSizingConfig(size=0.2),
            ),
        )


def test_yaml_with_sizing_block_loads(tmp_path):
    """End-to-end: YAML with sizing: block loads cleanly."""
    raw = _minimal_yaml()
    raw["backtest"]["sizing"] = {
        "type": "vol_target",
        "vol_target": {
            "reference_vol_annual": 0.25,
            "vol_window": 30,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.backtest.sizing.type == "vol_target"
    assert cfg.backtest.sizing.vol_target.reference_vol_annual == 0.25
    assert cfg.backtest.sizing.vol_target.vol_window == 30


def test_yaml_with_legacy_position_size_loads_with_warning(tmp_path):
    """End-to-end: legacy position_size YAML still works, emits warning."""
    raw = _minimal_yaml()
    raw["backtest"]["position_size"] = 0.07
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(cfg_file)
        depr = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(depr) == 1
    assert cfg.backtest.sizing.type == "fixed"
    assert cfg.backtest.sizing.fixed.size == 0.07
```

(Note: the test `test_position_size_plus_explicit_sizing_type_raises` above uses the conflict path via `fixed.size=0.2`; the separate `test_position_size_plus_explicit_sizing_fixed_size_raises` is the canonical case.)

### Step 3.2: Run tests to verify they fail

- [ ] Run: `python -m pytest tests/test_config.py -v -k "sizing or position_size"`
- [ ] Expected: tests fail (no SizingConfig class)

### Step 3.3: Add SizingConfig classes to config.py

- [ ] In `src/stockpool/config.py`, add this `import warnings` near the top of the imports section (after `from typing import Literal`):

```python
import warnings
```

Then **insert** the following three classes right before the existing `class BacktestConfig(BaseModel):` definition (around line 115):

```python
class FixedSizingConfig(BaseModel):
    """Constant lot size — every buy commits the same fraction of capital."""
    model_config = ConfigDict(extra="forbid")
    size: float = Field(default=0.1, gt=0.0, le=1.0)


class VolTargetSizingConfig(BaseModel):
    """Vol-target sizing — scale each lot inversely to recent stock vol.

    Formula (β, relative-to-baseline):
        size = fixed.size * (reference_vol_annual / recent_vol_annual)
        size = clip(size, min_size, max_size)

    ``fixed.size`` doubles as the baseline anchor: at recent_vol = reference_vol,
    the lot equals fixed.size. Vol estimator: simple rolling std over
    ``vol_window`` bars of daily simple returns, annualised with sqrt(252).
    """
    model_config = ConfigDict(extra="forbid")
    reference_vol_annual: float = Field(default=0.30, gt=0.0)
    vol_window: int = Field(default=20, gt=1)
    min_size: float = Field(default=0.03, gt=0.0, le=1.0)
    max_size: float = Field(default=0.20, gt=0.0, le=1.0)
    fallback_to: Literal["fixed", "skip"] = "fixed"

    @model_validator(mode="after")
    def _check_min_le_max(self) -> "VolTargetSizingConfig":
        if self.min_size > self.max_size:
            raise ValueError(
                f"min_size ({self.min_size}) must be <= max_size ({self.max_size})"
            )
        return self


class SizingConfig(BaseModel):
    """Per-lot sizing strategy. Default flipped to vol_target in F3 PR-C."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["fixed", "vol_target"] = "vol_target"
    fixed: FixedSizingConfig = Field(default_factory=FixedSizingConfig)
    vol_target: VolTargetSizingConfig = Field(default_factory=VolTargetSizingConfig)
```

### Step 3.4: Update BacktestConfig with `sizing` field + migration validator

- [ ] In `src/stockpool/config.py`, modify `BacktestConfig` (around line 115):

Replace:

```python
class BacktestConfig(BaseModel):
    forward_days: list[int]
    equity_curve_holding_days: list[int] = Field(default_factory=lambda: [5, 10, 20])
    risk_free_rate: float = 0.02
    costs: BacktestCostConfig = Field(default_factory=BacktestCostConfig)
    engine: Literal["single", "multi_lot"] = "multi_lot"
    position_size: float = Field(default=0.1, gt=0.0, le=1.0)
    max_concurrent_lots: int | None = Field(default=None, gt=0)

    @field_validator("equity_curve_holding_days")
    @classmethod
    def _validate_holding_days(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("equity_curve_holding_days must be a non-empty list")
        if any(n <= 0 for n in v):
            raise ValueError("equity_curve_holding_days entries must be positive integers")
        return v
```

With:

```python
class BacktestConfig(BaseModel):
    forward_days: list[int]
    equity_curve_holding_days: list[int] = Field(default_factory=lambda: [5, 10, 20])
    risk_free_rate: float = 0.02
    costs: BacktestCostConfig = Field(default_factory=BacktestCostConfig)
    engine: Literal["single", "multi_lot"] = "multi_lot"
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    # Deprecated alias for sizing.fixed.size. None = use sizing.
    # If set alongside a non-default sizing block, raises ValueError.
    # If set alone, auto-migrates to sizing.type=fixed + emits DeprecationWarning.
    position_size: float | None = Field(default=None, gt=0.0, le=1.0)
    max_concurrent_lots: int | None = Field(default=None, gt=0)

    @field_validator("equity_curve_holding_days")
    @classmethod
    def _validate_holding_days(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("equity_curve_holding_days must be a non-empty list")
        if any(n <= 0 for n in v):
            raise ValueError("equity_curve_holding_days entries must be positive integers")
        return v

    @model_validator(mode="after")
    def _migrate_position_size(self) -> "BacktestConfig":
        if self.position_size is None:
            return self
        # Heuristic: detect "user wrote sizing explicitly" by checking whether
        # any sizing field differs from defaults. Edge case (sizing block
        # written but values exactly match defaults) collapses silently —
        # documented in the spec (§2.1) as a tolerable false negative.
        sizing_explicit = (
            self.sizing.type != "vol_target"
            or self.sizing.fixed.size != 0.1
        )
        if sizing_explicit:
            raise ValueError(
                "Cannot set both backtest.position_size (deprecated) and "
                "backtest.sizing. Migrate position_size into sizing.fixed.size."
            )
        warnings.warn(
            "backtest.position_size is deprecated; use "
            "backtest.sizing.fixed.size (with sizing.type=fixed) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.sizing = SizingConfig(
            type="fixed",
            fixed=FixedSizingConfig(size=self.position_size),
        )
        self.position_size = None
        return self
```

### Step 3.5: Run config tests to verify they pass

- [ ] Run: `python -m pytest tests/test_config.py -v`
- [ ] Expected: all tests pass

### Step 3.6: Confirm the repo's own config.yaml still loads

- [ ] Run: `python -m pytest tests/test_config.py::test_default_config_yaml_loads -v`
- [ ] Expected: PASS — repo's `config.yaml` still uses `position_size: 0.1` → migration path triggers + DeprecationWarning (warning is fine in this test; it doesn't assert on absence)

### Step 3.7: Commit

- [ ] Run:

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
feat(config): SizingConfig sub-block + position_size deprecation+migration

Default backtest.sizing.type is vol_target. Legacy
backtest.position_size still works alone (auto-migrates to
sizing.type=fixed + emits DeprecationWarning) but conflicts with an
explicit non-default sizing block (ValidationError, fail-loud).

VolTargetSizingConfig validates min_size <= max_size.

Spec compatibility matrix: §8 of the design doc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Top-level wiring + fixture updates

**Files:**
- Modify: `src/stockpool/cli.py`
- Modify: `src/stockpool/backtest_runner.py`
- Modify: `src/stockpool/backtest_composite.py`
- Modify: `src/stockpool/strategy_factory.py`
- Modify: `src/stockpool/ab/config.py`
- Modify: `config.yaml`
- Modify: `tests/test_backtest_composite.py` (one call site)
- Modify: `tests/test_ab.py` (add sizing-merge case)

### Step 4.1: Update `cli.cmd_backtest` to use build_lot_sizer

- [ ] In `src/stockpool/cli.py`, find the multi_lot engine_label block (around line 252-255). Replace:

```python
    if cfg.backtest.engine == "multi_lot":
        engine_label = (
            f"multi_lot · 每次买入 {cfg.backtest.position_size:.0%} 起始资本独立一单"
        )
    else:
        engine_label = "single · 同时只持一只票,信号反转换仓"
```

With:

```python
    if cfg.backtest.engine == "multi_lot":
        sizing = cfg.backtest.sizing
        if sizing.type == "fixed":
            sizing_desc = f"fixed {sizing.fixed.size:.0%}"
        else:
            vt = sizing.vol_target
            sizing_desc = (
                f"vol_target ref={vt.reference_vol_annual:.0%} "
                f"window={vt.vol_window} clip=[{vt.min_size:.0%},{vt.max_size:.0%}]"
            )
        engine_label = f"multi_lot · {sizing_desc}"
    else:
        engine_label = "single · 同时只持一只票,信号反转换仓"
```

### Step 4.2: Update `backtest_runner.backtest_stocks` to pass lot_sizer

- [ ] In `src/stockpool/backtest_runner.py`, add this import at the top (after the existing imports):

```python
from stockpool.backtesting.sizing import build_lot_sizer
```

- [ ] Find the `simulate_equity_curve(...)` call (around lines 125-135) and the `simulate_strategy_equity_curve(...)` call (around lines 144-154). Replace both `position_size=cfg.backtest.position_size,` lines with:

```python
                    lot_sizer=build_lot_sizer(cfg.backtest.sizing),
```

(The two replacements are identical — both inside `backtest_stocks`.)

### Step 4.3: Update `backtest_composite.simulate_equity_curve` signature

- [ ] In `src/stockpool/backtest_composite.py`, find the `simulate_equity_curve` function. Add the `lot_sizer` kwarg, deprecate `position_size`, and update the engine construction.

First, add this import at the top:

```python
from stockpool.backtesting.sizing import FixedLotSizer, LotSizer
```

Then update the signature header:

Replace:

```python
def simulate_equity_curve(
    wf_df: pd.DataFrame,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    risk_free_rate: float = 0.02,
    engine: str = "single",
    position_size: float = 0.1,
    max_concurrent_lots: int | None = None,
) -> EquityResult:
```

With:

```python
def simulate_equity_curve(
    wf_df: pd.DataFrame,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    risk_free_rate: float = 0.02,
    engine: str = "single",
    position_size: float | None = None,
    lot_sizer: LotSizer | None = None,
    max_concurrent_lots: int | None = None,
) -> EquityResult:
```

Then find the `multi_lot` branch (where `MultiLotBacktestEngine` is constructed) and replace:

```python
    elif engine == "multi_lot":
        bt = MultiLotBacktestEngine(
            VerdictExecution(),
            position_size=position_size,
            costs=costs,
            risk_free_rate=risk_free_rate,
            max_concurrent_lots=max_concurrent_lots,
        )
```

With:

```python
    elif engine == "multi_lot":
        if lot_sizer is None:
            size = position_size if position_size is not None else 0.1
            lot_sizer = FixedLotSizer(size)
        elif position_size is not None:
            raise ValueError(
                "Pass either lot_sizer or position_size, not both"
            )
        bt = MultiLotBacktestEngine(
            VerdictExecution(),
            lot_sizer=lot_sizer,
            costs=costs,
            risk_free_rate=risk_free_rate,
            max_concurrent_lots=max_concurrent_lots,
        )
```

### Step 4.4: Update `strategy_factory.simulate_strategy_equity_curve` signature

- [ ] In `src/stockpool/strategy_factory.py`, apply the parallel changes to `simulate_strategy_equity_curve`.

Add import:

```python
from stockpool.backtesting.sizing import FixedLotSizer, LotSizer
```

Replace the signature:

```python
def simulate_strategy_equity_curve(
    daily_df: pd.DataFrame,
    strategy,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    risk_free_rate: float = 0.02,
    engine: str = "single",
    position_size: float = 0.1,
    max_concurrent_lots: int | None = None,
) -> EquityResult:
```

With:

```python
def simulate_strategy_equity_curve(
    daily_df: pd.DataFrame,
    strategy,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    risk_free_rate: float = 0.02,
    engine: str = "single",
    position_size: float | None = None,
    lot_sizer: LotSizer | None = None,
    max_concurrent_lots: int | None = None,
) -> EquityResult:
```

And replace the multi_lot branch:

```python
    elif engine == "multi_lot":
        bt = MultiLotBacktestEngine(
            strategy, position_size=position_size, costs=costs,
            risk_free_rate=risk_free_rate, max_concurrent_lots=max_concurrent_lots,
        )
```

With:

```python
    elif engine == "multi_lot":
        if lot_sizer is None:
            size = position_size if position_size is not None else 0.1
            lot_sizer = FixedLotSizer(size)
        elif position_size is not None:
            raise ValueError(
                "Pass either lot_sizer or position_size, not both"
            )
        bt = MultiLotBacktestEngine(
            strategy, lot_sizer=lot_sizer, costs=costs,
            risk_free_rate=risk_free_rate, max_concurrent_lots=max_concurrent_lots,
        )
```

### Step 4.5: Update `ab/config.py` ArmBacktestOverride with sizing field

- [ ] In `src/stockpool/ab/config.py`, modify the imports at the top to include `SizingConfig`:

Replace:

```python
from stockpool.config import (
    AppConfig,
    BacktestCostConfig,
    StrategyConfig,
    load_config,
)
```

With:

```python
from stockpool.config import (
    AppConfig,
    BacktestCostConfig,
    SizingConfig,
    StrategyConfig,
    load_config,
)
```

Then in `ArmBacktestOverride`, add the `sizing` field after `costs` (and keep `position_size` for backward compat):

Replace:

```python
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
```

With:

```python
class ArmBacktestOverride(BaseModel):
    """Per-arm overrides to the base.backtest section.

    equity_curve_holding_days is required and must be a length-1 list.
    All other fields default to None, meaning "inherit base.backtest.<same>".

    Note: sizing is whole-block-replace (matches strategy override semantics).
    position_size is the deprecated alias kept for arm-level back-compat —
    the merged AppConfig's _migrate_position_size handles the deprecation
    warning and conflict checks on the merged result.
    """
    model_config = ConfigDict(extra="forbid")
    equity_curve_holding_days: list[int]
    forward_days: list[int] | None = None
    risk_free_rate: float | None = None
    costs: BacktestCostConfig | None = None
    engine: Literal["single", "multi_lot"] | None = None
    sizing: SizingConfig | None = None
    position_size: float | None = None
    max_concurrent_lots: int | None = None
```

### Step 4.6: Add A/B sizing-merge test

- [ ] In `tests/test_ab.py`, append:

```python
# ============================================================================
# F3 PR-C — sizing merge through ArmBacktestOverride
# ============================================================================

from stockpool.config import (
    SizingConfig, FixedSizingConfig, VolTargetSizingConfig,
)


def test_arm_sizing_replaces_base_sizing():
    """When an arm provides sizing, it replaces base.backtest.sizing wholesale."""
    base = _make_base_cfg()
    arm = ArmOverride(
        strategy=base.strategy,
        backtest=ArmBacktestOverride(
            equity_curve_holding_days=[10],
            sizing=SizingConfig(
                type="vol_target",
                vol_target=VolTargetSizingConfig(reference_vol_annual=0.25),
            ),
        ),
    )
    eff = build_effective_cfg(base, arm)
    assert eff.backtest.sizing.type == "vol_target"
    assert eff.backtest.sizing.vol_target.reference_vol_annual == 0.25


def test_arm_without_sizing_inherits_base_sizing():
    """When arm.sizing is None, base.backtest.sizing carries through."""
    base = _make_base_cfg()
    # Pin base sizing to a recognisable value.
    base.backtest.sizing = SizingConfig(
        type="fixed",
        fixed=FixedSizingConfig(size=0.07),
    )
    arm = ArmOverride(
        strategy=base.strategy,
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    eff = build_effective_cfg(base, arm)
    assert eff.backtest.sizing.type == "fixed"
    assert eff.backtest.sizing.fixed.size == 0.07


def test_two_arms_with_different_sizing_stay_isolated():
    """Different arms produce different effective_cfgs with different content_hash."""
    base = _make_base_cfg()
    arm_fixed = ArmOverride(
        strategy=base.strategy,
        backtest=ArmBacktestOverride(
            equity_curve_holding_days=[10],
            sizing=SizingConfig(type="fixed", fixed=FixedSizingConfig(size=0.1)),
        ),
    )
    arm_vol = ArmOverride(
        strategy=base.strategy,
        backtest=ArmBacktestOverride(
            equity_curve_holding_days=[10],
            sizing=SizingConfig(type="vol_target"),
        ),
    )
    eff_fixed = build_effective_cfg(base, arm_fixed)
    eff_vol = build_effective_cfg(base, arm_vol)
    assert eff_fixed.backtest.sizing.type == "fixed"
    assert eff_vol.backtest.sizing.type == "vol_target"
    assert eff_fixed.content_hash != eff_vol.content_hash
```

(If `_make_base_cfg` does not exist in test_ab.py, search for the existing pattern of building a base cfg — it might be inline. Use the pattern matching the file's existing tests.)

### Step 4.7: Update the in-repo `config.yaml` to use the new sizing block

- [ ] In `config.yaml`, find the `backtest:` block (around lines 73-87). Replace the line `position_size: 0.1            # multi_lot 模式下每单占用的起始资本比例 (0, 1]` and its surrounding comment block:

Replace:

```yaml
  engine: multi_lot
  position_size: 0.1            # multi_lot 模式下每单占用的起始资本比例 (0, 1]
  max_concurrent_lots: null     # null = 由现金自然封顶
```

With:

```yaml
  engine: multi_lot
  # sizing 默认 vol_target (波动大的票仓位小, 波动小的票仓位大),
  # 想回到老的"每只票都 10% 仓"行为, 把 type 切回 fixed 即可。
  sizing:
    type: vol_target            # fixed | vol_target
    fixed:
      size: 0.1                 # fixed 模式下每单资本占比, 也是 vol_target 公式的锚点
    vol_target:
      reference_vol_annual: 0.30   # "标准 A 股个股年化 vol", recent_vol = ref 时仓位 = fixed.size
      vol_window: 20               # 滚动 std 用的 bar 数
      min_size: 0.03               # 最小仓位下限
      max_size: 0.20               # 最大仓位上限
      fallback_to: fixed           # 冷启动 / vol 算不出时: fixed 用 baseline, skip 不开仓
  max_concurrent_lots: null     # null = 由现金自然封顶
```

### Step 4.8: Update the failing test_backtest_composite.py call site

- [ ] In `tests/test_backtest_composite.py`, find the call to `simulate_equity_curve` with `position_size=0.1` (around line 330):

The line `engine="multi_lot", position_size=0.1,` continues to work because `position_size` is kept as a kwarg in `simulate_equity_curve`. **No change needed.** Verify with the next step.

### Step 4.9: Run the full test suite + integration smoke

- [ ] Run: `python -m pytest tests/ -q`
- [ ] Expected: all 338+ tests pass. Expect DeprecationWarnings from the repo's `config.yaml` loading (which still uses `position_size: 0.1`); after Step 4.7 above, the repo `config.yaml` uses the new sizing block, so warnings should stop showing during `test_default_config_yaml_loads`.

If `test_default_config_yaml_loads` fails because of a sizing block parsing error, debug the YAML formatting before proceeding.

### Step 4.10: Smoke-test CLI commands

- [ ] Run: `python -m stockpool backtest --config config.yaml --stocks 605589`
- [ ] Expected: command runs to completion (or fails only due to data fetch issues unrelated to sizing); look at the log line "Backtest engine: multi_lot · vol_target ref=30% window=20 clip=[3%,20%]" — confirms wiring works

### Step 4.11: Commit

- [ ] Run:

```bash
git add src/stockpool/cli.py src/stockpool/backtest_runner.py src/stockpool/backtest_composite.py src/stockpool/strategy_factory.py src/stockpool/ab/config.py config.yaml tests/test_ab.py
git commit -m "$(cat <<'EOF'
feat(backtesting): wire build_lot_sizer through cli/runner/composite/strategy_factory/ab

All five call sites that previously passed cfg.backtest.position_size
now build a LotSizer via backtesting.sizing.build_lot_sizer and pass it
through. The two simulate_*_equity_curve helpers keep position_size= as
a deprecated kwarg for fixture compat.

ArmBacktestOverride gains sizing: SizingConfig | None; whole-block
replace semantics on merge, matching arm.strategy. Two-arm A/B with
different sizing configs produces correctly isolated content_hashes.

config.yaml refreshed to use the new sizing block (no behavior change
relative to the previous position_size: 0.1 once the migration path
ran, but cleaner and no DeprecationWarning at startup).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: A/B validation run + verdict writeback

**Files:**
- Create: `ab_sizing.yaml`
- Create: `docs/ab_runs/2026-05-24-pr-c-sizing.html` (output of A/B run, archived)
- Modify: `docs/strategy_improvement_2026.md`

### Step 5.1: Create the A/B config

- [ ] Create `ab_sizing.yaml` in repo root with this content:

```yaml
# A/B validation for F3 PR-C — vol-target sizing.
#
# Spec:  docs/superpowers/specs/2026-05-24-f3-pr-c-vol-target-sizing-design.md
# Run:   python -m stockpool ab --config ab_sizing.yaml
# Gate:  Δmax_dd / fixed.max_dd ≤ -0.20   AND   Δsharpe ≥ -0.05
#
# Note: vol-target's main payoff is DD reduction, NOT Sharpe lift.
# Sharpe persisting (±0.05) + DD shrinking ≥ 20% counts as success.

base_config: config.yaml

arms:
  fixed_baseline:
    strategy:
      name: ml_factor
      ml_factor:
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      sizing:
        type: fixed
        fixed: {size: 0.10}
      equity_curve_holding_days: [10]

  vol_target:
    strategy:
      name: ml_factor
      ml_factor:
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      sizing:
        type: vol_target
        fixed: {size: 0.10}
        vol_target:
          reference_vol_annual: 0.30
          vol_window: 20
          min_size: 0.03
          max_size: 0.20
          fallback_to: fixed
      equity_curve_holding_days: [10]
```

### Step 5.2: Run the A/B

- [ ] Run: `python -m stockpool ab --config ab_sizing.yaml`
- [ ] Expected: completes in 1-3 minutes; writes `reports/ab/<today>.html` and `reports/ab/latest.html`. Watch the log for any failed-stock messages.

If the A/B fails to run (e.g. data fetch issues), debug before proceeding. Do not fabricate verdicts.

### Step 5.3: Inspect the A/B report

- [ ] Open `reports/ab/latest.html` (or `reports/ab/<today>.html`) in a browser
- [ ] Read off the per-stock aggregate table and record:
  - `mean_sharpe` for each arm
  - `mean_max_drawdown` for each arm
  - `mean_annualized_return` for each arm
- [ ] Compute:
  - `Δsharpe = vol_target.mean_sharpe - fixed.mean_sharpe`
  - `Δmax_dd_ratio = (vol_target.mean_max_drawdown - fixed.mean_max_drawdown) / abs(fixed.mean_max_drawdown)`
  - `Δreturn_pp = vol_target.mean_annualized_return - fixed.mean_annualized_return` (in percentage points)

### Step 5.4: Determine verdict

- [ ] Apply the gate:
  - **✅ Pass gate** if `Δmax_dd_ratio ≤ -0.20` AND `Δsharpe ≥ -0.05`
  - **🎯 Hit success** if `Δmax_dd_ratio ≤ -0.30` AND `Δreturn_pp ≥ -2.0` AND `Δsharpe ≥ 0`
  - **⚠️ Tied** if neither gate clearly hit nor missed
  - **❌ Regression** if gate fails

### Step 5.5: Archive the A/B report

- [ ] Run:

```bash
cp reports/ab/latest.html docs/ab_runs/2026-05-24-pr-c-sizing.html
```

### Step 5.6: Write verdict into strategy_improvement_2026.md

- [ ] Open `docs/strategy_improvement_2026.md`. In §6 "🚧 待做" section, find the PR-C row in the "P1: F3" table.
- [ ] Move that row OUT of the 待做 table and INTO the "✅ 已完成(经 A/B 验证)" table with this format:

```
| **F3 PR-C** — Sizing 子段化 + vol-target | spec `2026-05-24-f3-pr-c-vol-target-sizing-design.md` | **<verdict_emoji>** — Δsharpe=<value>, Δmax_dd=<value>%, Δreturn=<value>%. <one-line conclusion> |
```

Choose `<verdict_emoji>` from {✅ success, 🎯 hit success, ⚠️ tied, ❌ regression} per Step 5.4.

- [ ] Update the "当前 sweet spot 默认" YAML snippet at the end of the ✅ table — if vol_target wins, add a `sizing: {type: vol_target}` line; if it ties/loses, keep `sizing: {type: fixed, fixed: {size: 0.10}}` as the documented default.
- [ ] Remove PR-C from the P1 待做 sub-table; PR-D and PR-E remain.

### Step 5.7: If verdict is ❌ regression, roll back the default

- [ ] **Only if verdict is ❌ regression:** revert the default in `src/stockpool/config.py`:

  Change `SizingConfig.type` default from `"vol_target"` to `"fixed"`.

- [ ] Update `config.yaml` to set `sizing.type: fixed` explicitly.
- [ ] Re-run `python -m pytest tests/test_config.py -v` to confirm the migration path tests still pass (the validator detects "default sizing" via `type != "vol_target"`; changing default means the heuristic must invert — review `_migrate_position_size` and adjust the `sizing_explicit` check accordingly).
- [ ] Document the rollback as a `fix(config):` follow-up commit before proceeding to Task 6.

### Step 5.8: Commit A/B materials + verdict

- [ ] Run:

```bash
git add ab_sizing.yaml docs/ab_runs/2026-05-24-pr-c-sizing.html docs/strategy_improvement_2026.md
git commit -m "$(cat <<'EOF'
chore(ab): vol-target sizing A/B validation + verdict writeback

Archives the A/B report for F3 PR-C and records the verdict in
docs/strategy_improvement_2026.md §6. Verdict: <fill in from Step 5.4>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `<fill in from Step 5.4>` with the actual verdict line you derived.)

---

## Task 6: Documentation sync (CLAUDE.md + README.md)

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

### Step 6.1: Update CLAUDE.md — module map

- [ ] In `CLAUDE.md`, find the section "## 模块地图" (around the table of modules). Find the `src/stockpool/backtesting/` entries (currently includes `backtesting/` referenced as a directory). Add this row to the table (after `backtesting/`):

```
| `src/stockpool/backtesting/sizing.py` | **LotSizer Protocol** + `FixedLotSizer` / `VolTargetLotSizer` + `build_lot_sizer(SizingConfig)` 工厂 (F3 PR-C);独立模块,无 config 运行时依赖 (TYPE_CHECKING 引用) |
```

### Step 6.2: Update CLAUDE.md — config section

- [ ] In `CLAUDE.md`, find the section "## 配置 (`config.yaml`)" and the line that mentions `backtest`. Replace the `backtest` field description bullet with:

```
- `backtest` — `forward_days` / `equity_curve_holding_days` / `risk_free_rate` / `costs` / **`engine`** / **`sizing`**(`type: fixed | vol_target`, 默认 `vol_target`;`fixed.size` 是 vol_target 公式的 baseline 锚点) / **~~`position_size`~~**(deprecated alias of `sizing.fixed.size`,自动迁移 + DeprecationWarning) / **`max_concurrent_lots`**
```

### Step 6.3: Update CLAUDE.md — sizing detail (引擎约定 后面追加)

- [ ] In `CLAUDE.md`, find the "## 引擎约定(重要)" section. After the existing bullets, add a new subsection:

```
## Sizing(F3 PR-C 起)

`MultiLotBacktestEngine` 不再硬编码 `position_size`,改由 `LotSizer` 注入:

- `FixedLotSizer(size)` — 老行为,每单恒定 `size` 比例
- `VolTargetLotSizer(baseline, ref_vol, window, min, max, fallback)` — 按个股最近 `window` bars 的滚动 std 反比调仓:`size = baseline × (ref_vol / recent_vol)`,clip 到 `[min, max]`
  - 冷启动(< window+1 bar)/ NaN / vol=0 → 走 `fallback`: `"fixed"` 退回 baseline,`"skip"` 返 0(本次不开仓)
  - 公式锚点 `baseline = cfg.backtest.sizing.fixed.size`:fixed 和 vol_target 之间切换时,锚点不变,差异纯来自 vol-adjust
- 工厂 `build_lot_sizer(cfg.backtest.sizing)` 是顶层 wiring(cli / backtest_runner / backtest_composite / strategy_factory / ab/config 全部走它)
- `Trade.lot_size` 记录每笔成交的实际仓位,A/B 报告可用其做归因
```

### Step 6.4: Update CLAUDE.md — test table

- [ ] In the test table section ("## 测试"), add a row:

```
| `test_sizing.py` | FixedLotSizer / VolTargetLotSizer 数学 + fallback + build_lot_sizer 工厂 |
```

And update the `test_multi_lot_engine.py` row to mention the new coverage:

```
| `test_multi_lot_engine.py` | 多仓位 lot 独立计时、现金约束、reset hook;`lot_sizer` 注入 + `Trade.lot_size` 透传 + skip-fallback 不开仓 |
```

### Step 6.5: Update CLAUDE.md — A/B testing section

- [ ] In the A/B testing section (mentions `ArmOverride` and `ArmBacktestOverride`), add a note that `sizing` is now a supported per-arm override:

Find the bullet about A/B and add (or extend) to:

```
- A/B testing 用 `ab.yaml`(独立配置文件):见 `docs/superpowers/specs/2026-05-24-ab-testing-design.md`。每个 arm 可覆盖 `strategy:`(整段替换)和 `backtest:`(字段级合并),`backtest.sizing` 是新支持的覆盖字段(整段替换),用于比较 fixed vs vol_target sizing。其他顶层字段继承 base。
```

### Step 6.6: Update README.md

- [ ] In `README.md`, find the quickstart `config.yaml` example or backtest section. Add a note about the new sizing block. (Exact location depends on README structure — locate the section that talks about `position_size` or `backtest` config.)

If README does not mention `position_size` specifically, add a brief note under "回测" or "配置":

```markdown
### 仓位 sizing

`backtest.sizing` 子段控制每笔买入的仓位大小:

- **`sizing.type: fixed`** — 每只票同样大小(`sizing.fixed.size`,默认 10%)
- **`sizing.type: vol_target`** (默认) — 按个股近期波动反比调仓,目标是让每只票贡献的风险大致相等。主效应是降低组合 max DD

老的 `backtest.position_size: 0.1` 仍能工作(自动迁移 + DeprecationWarning),但新代码请直接写 `sizing` 子段。
```

### Step 6.7: Final test suite run

- [ ] Run: `python -m pytest tests/ -q`
- [ ] Expected: all tests pass, no DeprecationWarnings from `test_default_config_yaml_loads` (since config.yaml was updated in Task 4.7)

### Step 6.8: Commit

- [ ] Run:

```bash
git add CLAUDE.md README.md
git commit -m "$(cat <<'EOF'
docs: F3 PR-C — CLAUDE.md + README.md refresh for sizing sub-block

Adds module map row for backtesting/sizing.py, documents the new
backtest.sizing config layout and deprecated position_size alias,
records test_sizing.py + extended test_multi_lot_engine.py coverage
in the test table, and notes that ArmBacktestOverride.sizing is a
supported per-arm A/B override.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

**Spec coverage check:**

- §2.1 SizingConfig schema → Task 3 ✓
- §2.1 BacktestConfig.position_size deprecation + _migrate_position_size → Task 3 ✓
- §2.2 backtesting/sizing.py module (Protocol + Fixed + VolTarget) → Task 1 ✓
- §2.3 MultiLotBacktestEngine signature change → Task 2 ✓
- §2.3 _simulate_multi_lot signature change + lot opening logic → Task 2 ✓
- §2.4 Trade.lot_size + _OpenLot.lot_size → Task 2 ✓
- §2.5 build_lot_sizer factory → Task 1 ✓
- §2.5 five top-level wiring sites (cli, backtest_runner, backtest_composite, strategy_factory, ab/config) → Task 4 ✓
- §2.6 ArmBacktestOverride.sizing field → Task 4 ✓
- §3 test_sizing.py / test_multi_lot_engine.py / test_config.py / test_ab.py → Tasks 1-4 ✓
- §3 fixture update note (推荐 b: explicit sizing.type=fixed) → Task 4.7 covers config.yaml; existing tests pass position_size= kwarg through engine, which still works ✓
- §4 doc sync (CLAUDE.md, README.md, strategy_improvement_2026.md §6) → Tasks 5 + 6 ✓
- §6 A/B validation (ab_sizing.yaml, gates, verdict writeback) → Task 5 ✓

No gaps.

**Type consistency check:**
- `LotSizer` protocol signature matches all callers ✓
- `FixedLotSizer.size`, `VolTargetLotSizer.baseline_size/reference_vol_annual/...` match factory access patterns ✓
- `SizingConfig.type/fixed/vol_target`, `VolTargetSizingConfig.fallback_to` match factory dispatch ✓
- `Trade.lot_size` default 0.1 matches existing fixture pattern ✓

**Risks during execution:**
- A/B in Task 5 produces a real verdict; if ❌ regression, Task 5.7 prescribes the rollback (default flip to fixed). Plan covers this branch.
- Task 4.7 changes config.yaml — if user has uncommitted local changes to config.yaml (per WIP status), reconcile with their `git status` before staging.

---

## Execution Choice

Plan complete and saved to `docs/superpowers/plans/2026-05-24-f3-pr-c-vol-target-sizing.md`.
