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
        if min_size > max_size:
            raise ValueError(
                f"min_size ({min_size}) must be <= max_size ({max_size})"
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
