"""Verify that prewarm + parallel produces bit-exact scores as serial mode.

Strict invariant (as of the fix after commit 84e2ee9):
    precompute_scores_from_legacy(legacy, panel, n_workers=1, prewarm=False)
    ==
    precompute_scores_from_legacy(legacy, panel, n_workers=2, prewarm=True)

Both must agree within rtol=1e-12, atol=0 (bit-exact up to IEEE754 rounding).

The fix switches _prewarm_monthly_fits from max(panel_data, key=len) to
next(iter(panel_data)) — the same implicit host that the serial loop uses
when iterating panel_data.items() — so the canonical monthly fits are
identical and score panels are bit-exact.

Usage:
    .venv/Scripts/python.exe scripts/verify_prewarm_monthly_fits.py

Passes if final line prints OK.
"""
from __future__ import annotations

import logging
import sys

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
    # Use >= 20 stocks so the <20 short-circuit in precompute_scores does not
    # force serial even when n_workers=2.
    pool = {f"S{i:03d}": _stock_df(n_bars=200, seed=i) for i in range(25)}
    log.info("Panel: %d stocks × 200 bars; factors=%s", len(pool), factors)

    # ── Test 1: _prewarm_monthly_fits actually fires and caches ──────────────
    log.info("=== Test 1: _prewarm_monthly_fits populates cache ===")
    sc1: dict = {}
    strat1 = _make_strategy(pool, factors, sc1)
    assert len(sc1) == 0, "cache should start empty"

    n_warmed = _prewarm_monthly_fits(strat1, pool)

    assert n_warmed > 0, f"Expected >=1 fit warmed, got {n_warmed}"
    assert len(sc1) > 0, "cache should be non-empty after prewarm"
    heavy = [k for k in sc1 if isinstance(k, tuple) and k[0] == "__pooled_xy_long__"]
    assert len(heavy) == 0, f"heavy keys not removed after prewarm: {heavy}"
    log.info(
        "Test 1 OK: n_warmed=%d, cache_size=%d, no heavy keys",
        n_warmed, len(sc1),
    )

    # ── Test 2: no-op guards still work ─────────────────────────────────────
    log.info("=== Test 2: _prewarm_monthly_fits no-op guards ===")

    class _DummyStrategy:
        pass
    assert _prewarm_monthly_fits(_DummyStrategy(), pool) == 0

    cfg_per_stock = MLFactorConfig(
        factors=factors, horizon=3, train_window=60, min_train_samples=30,
        refit_every=20, panel_mode="per_stock",
        selector=SelectorConfig(type="lasso"), weighter=WeighterConfig(type="ic"),
    )
    s_per = MLFactorStrategy(cfg=cfg_per_stock, pool_data=pool, shared_cache={})
    assert _prewarm_monthly_fits(s_per, pool) == 0

    cfg_ok = MLFactorConfig(
        factors=factors, horizon=3, train_window=60, min_train_samples=30,
        refit_every=20, panel_mode="pooled", embargo_days=0, share_pool_fit=True,
        selector=SelectorConfig(type="lasso"), weighter=WeighterConfig(type="ic"),
    )
    s_nocache = MLFactorStrategy(cfg=cfg_ok, pool_data=pool, shared_cache=None)
    assert _prewarm_monthly_fits(s_nocache, pool) == 0

    log.info("Test 2 OK: all no-op guards pass")

    # ── Test 3: STRICT INVARIANT — serial == parallel+prewarm (bit-exact) ────
    # This is the core regression: prior to the fix, _prewarm_monthly_fits
    # used max(panel_data, key=len) as the host, which differs from the
    # first key in iteration order that the serial loop implicitly uses.
    # After the fix, both paths use next(iter(panel_data)), so the monthly
    # Lasso fits are identical and score panels are bit-exact.
    log.info("=== Test 3: serial (n_workers=1, prewarm=False) == parallel (n_workers=2, prewarm=True) ===")

    # Run A: ground truth — serial, no prewarm
    log.info("  Run A: n_workers=1, prewarm=False …")
    sc_a: dict = {}
    strat_a = _make_strategy(pool, factors, sc_a)
    score_a = precompute_scores_from_legacy(
        strat_a, pool, n_workers=1, prewarm=False,
    )
    log.info("  score_a shape=%s, NaN%%=%.1f%%", score_a.shape,
             score_a.isna().mean().mean() * 100)

    # Run B: prewarm + parallel
    log.info("  Run B: n_workers=2, prewarm=True …")
    sc_b: dict = {}
    strat_b = _make_strategy(pool, factors, sc_b)
    score_b = precompute_scores_from_legacy(
        strat_b, pool, n_workers=2, prewarm=True,
    )
    log.info("  score_b shape=%s, NaN%%=%.1f%%", score_b.shape,
             score_b.isna().mean().mean() * 100)

    common_idx = score_a.index.intersection(score_b.index)
    common_cols = sorted(score_a.columns.intersection(score_b.columns))
    if len(common_idx) == 0 or len(common_cols) == 0:
        log.error("No common rows/cols between runs — aborting.")
        sys.exit(1)

    a = score_a.loc[common_idx, common_cols].sort_index()
    b = score_b.loc[common_idx, common_cols].sort_index()

    np.testing.assert_allclose(
        a.values, b.values, rtol=1e-12, atol=0, equal_nan=True,
    )
    log.info(
        "Test 3 OK: bit-exact match; shape=%s", a.shape,
    )

    print(
        f"\nOK: serial (n_workers=1, prewarm=False) == parallel (n_workers=2, prewarm=True) "
        f"bit-exact within rtol=1e-12; shape={a.shape}"
    )


if __name__ == "__main__":
    main()
