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


# ============================================================================
# VolTargetLotSizer — validation
# ============================================================================

def test_vol_target_rejects_invalid_fallback():
    with pytest.raises(ValueError):
        VolTargetLotSizer(
            baseline_size=0.1, reference_vol_annual=0.3, vol_window=20,
            min_size=0.03, max_size=0.20, fallback="bogus",
        )


def test_vol_target_rejects_min_greater_than_max():
    with pytest.raises(ValueError, match="min_size"):
        VolTargetLotSizer(
            baseline_size=0.1, reference_vol_annual=0.3, vol_window=20,
            min_size=0.30, max_size=0.10, fallback="fixed",
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
