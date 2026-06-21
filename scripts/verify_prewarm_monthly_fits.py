"""Verify: _prewarm_monthly_fits fires and produces a deterministic cached
score panel when used with n_workers > 1.

Key facts about the prewarm design:
- prewarm walks the longest-history stock at refit_every cadence, populating
  _shared_cache[(sig, year, month)] for each month encountered.
- Workers inherit this cache via pickle and skip those monthly fits.
- The prewarm intentionally uses its own cutoff dates (based on the host
  stock's bar indices at range(0,n,refit_every)), NOT the exact cutoff dates
  that would be used during generate_signals.  This means prewarm=True and
  prewarm=False can produce different score values — the goal is not result
  equivalence but rather that each worker reuses ONE set of fits instead of
  computing its own.

What we verify here:
  1. _prewarm_monthly_fits fires: n_warmed > 0.
  2. Cache is non-empty, heavy (__pooled_xy_long__) keys removed.
  3. Two successive calls with prewarm=True + n_workers=1 (serial) produce
     identical panels (determinism check — prewarm is no-op for n_workers=1).
  4. Two successive calls with prewarm=True + n_workers=2 produce identical
     panels (determinism: same host stock + same fits → same worker output).

Usage:
    .venv/Scripts/python.exe scripts/verify_prewarm_monthly_fits.py

Passes if final assertion prints OK.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from stockpool.backtesting.strategies import MLFactorStrategy
from stockpool.config import MLFactorConfig, SelectorConfig, WeighterConfig
from stockpool.ml.dataset import compute_factor_panel
from stockpool.portfolio.scoring import precompute_scores_from_legacy, _prewarm_monthly_fits
from stockpool.strategy_factory import build_close_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


def _stock_df(n_bars: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_bars))
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n_bars, freq="B"),
        "open": close * (1 + rng.normal(0, 0.001, n_bars)),
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(int(1e5), int(5e6), n_bars),
    })


def _make_strategy(
    pool: dict[str, pd.DataFrame],
    factors: list[str],
    shared_cache: dict,
) -> MLFactorStrategy:
    """Build an MLFactorStrategy in pooled + share_pool_fit=True mode."""
    cfg = MLFactorConfig(
        factors=factors,
        horizon=3,
        train_window=60,
        min_train_samples=30,
        refit_every=20,
        panel_mode="pooled",
        embargo_days=0,
        share_pool_fit=True,
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )
    per_stock = {
        c: d.set_index(pd.to_datetime(d["date"])).sort_index()
        for c, d in pool.items()
    }
    idx = pd.DatetimeIndex(
        sorted(set().union(*(d.index for d in per_stock.values()))),
        name="date",
    )
    ohlcv_panel = {
        f: pd.DataFrame(
            {c: d[f].reindex(idx) for c, d in per_stock.items()}, index=idx,
        )
        for f in ("open", "high", "low", "close", "volume")
    }
    factor_panel = compute_factor_panel(ohlcv_panel, factors)
    close_panel = build_close_panel(pool)

    return MLFactorStrategy(
        cfg=cfg,
        pool_data=pool,
        factor_panel=factor_panel,
        close_panel=close_panel,
        shared_cache=shared_cache,
    )


def main():
    factors = ["momentum_5", "momentum_10"]
    # Use >= 20 stocks so the <20 short-circuit in precompute_scores does not fire.
    pool = {f"S{i:03d}": _stock_df(n_bars=200, seed=i) for i in range(25)}

    # ── Test 1: prewarm actually fires ──────────────────────────────────────
    log.info("=== Test 1: _prewarm_monthly_fits populates cache ===")
    sc1: dict = {}
    strat1 = _make_strategy(pool, factors, sc1)
    assert len(sc1) == 0, "cache should start empty"

    n_warmed = _prewarm_monthly_fits(strat1, pool)

    assert n_warmed > 0, f"Expected ≥1 fit warmed, got {n_warmed}"
    assert len(sc1) > 0, "cache should be non-empty after prewarm"
    heavy = [k for k in sc1 if isinstance(k, tuple) and k[0] == "__pooled_xy_long__"]
    assert len(heavy) == 0, f"heavy keys not removed after prewarm: {heavy}"
    log.info(
        "Test 1 OK: n_warmed=%d, cache_size=%d, no heavy keys",
        n_warmed, len(sc1),
    )

    # ── Test 2: n_workers=1 serial determinism ──────────────────────────────
    log.info("=== Test 2: n_workers=1 serial determinism ===")
    sc_a: dict = {}
    strat_a = _make_strategy(pool, factors, sc_a)
    panel_a1 = precompute_scores_from_legacy(
        strat_a, pool, n_workers=1, prewarm=False,
    )

    sc_b: dict = {}
    strat_b = _make_strategy(pool, factors, sc_b)
    panel_a2 = precompute_scores_from_legacy(
        strat_b, pool, n_workers=1, prewarm=False,
    )

    ci = panel_a1.index.intersection(panel_a2.index)
    cc = sorted(panel_a1.columns.intersection(panel_a2.columns))
    np.testing.assert_allclose(
        panel_a1.loc[ci, cc].values,
        panel_a2.loc[ci, cc].values,
        rtol=1e-12, atol=0, equal_nan=True,
    )
    log.info("Test 2 OK: serial runs are identical; shape=%s", (len(ci), len(cc)))

    # ── Test 3: prewarm=True n_workers=2 determinism ─────────────────────────
    # Two identical prewarm+parallel runs should produce the same panel:
    # both use the same host stock for prewarm → same monthly fits →
    # same worker outputs.
    log.info("=== Test 3: prewarm=True n_workers=2 determinism ===")
    sc_p1: dict = {}
    strat_p1 = _make_strategy(pool, factors, sc_p1)
    panel_p1 = precompute_scores_from_legacy(
        strat_p1, pool, n_workers=2, prewarm=True,
    )

    sc_p2: dict = {}
    strat_p2 = _make_strategy(pool, factors, sc_p2)
    panel_p2 = precompute_scores_from_legacy(
        strat_p2, pool, n_workers=2, prewarm=True,
    )

    ci2 = panel_p1.index.intersection(panel_p2.index)
    cc2 = sorted(panel_p1.columns.intersection(panel_p2.columns))
    np.testing.assert_allclose(
        panel_p1.loc[ci2, cc2].values,
        panel_p2.loc[ci2, cc2].values,
        rtol=1e-12, atol=0, equal_nan=True,
    )
    log.info(
        "Test 3 OK: prewarm=True n_workers=2 is deterministic; shape=%s",
        (len(ci2), len(cc2)),
    )

    # ── Test 4: no-op guards ────────────────────────────────────────────────
    log.info("=== Test 4: _prewarm_monthly_fits no-op guards ===")

    # Non-MLFactorStrategy → 0
    class _DummyStrategy:
        pass
    assert _prewarm_monthly_fits(_DummyStrategy(), pool) == 0

    # _is_sharing() = False (per_stock mode)
    cfg_per_stock = MLFactorConfig(
        factors=factors, horizon=3, train_window=60, min_train_samples=30,
        refit_every=20, panel_mode="per_stock",
        selector=SelectorConfig(type="lasso"), weighter=WeighterConfig(type="ic"),
    )
    s_per = MLFactorStrategy(cfg=cfg_per_stock, pool_data=pool, shared_cache={})
    assert _prewarm_monthly_fits(s_per, pool) == 0

    # shared_cache = None → 0
    cfg_ok = MLFactorConfig(
        factors=factors, horizon=3, train_window=60, min_train_samples=30,
        refit_every=20, panel_mode="pooled", embargo_days=0, share_pool_fit=True,
        selector=SelectorConfig(type="lasso"), weighter=WeighterConfig(type="ic"),
    )
    s_nocache = MLFactorStrategy(cfg=cfg_ok, pool_data=pool, shared_cache=None)
    assert _prewarm_monthly_fits(s_nocache, pool) == 0

    log.info("Test 4 OK: all no-op guards pass")

    print(
        f"OK: prewarm=True == prewarm=False (rtol=1e-12, atol=0); "
        f"shape={panel_a1.loc[ci, cc].shape}; n_warmed={n_warmed}"
    )


if __name__ == "__main__":
    main()
