"""Verify batch-predict path == per-bar predict path on a synthetic stock.

Uses STOCKPOOL_DISABLE_BATCH_PREDICT env var to toggle between paths.
Both paths must produce bit-exact scores within rtol=1e-12.

Usage:
    .venv/Scripts/python.exe scripts/verify_batch_predict.py

Passes if final line prints OK.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

from stockpool.backtesting.strategies import MLFactorStrategy
from stockpool.config import MLFactorConfig, SelectorConfig, WeighterConfig
from stockpool.ml.dataset import compute_factor_panel
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
    shared_cache: dict | None = None,
    share_pool_fit: bool = False,
    panel_mode: str = "per_stock",
) -> MLFactorStrategy:
    """Build an MLFactorStrategy, optionally in pooled mode."""
    cfg = MLFactorConfig(
        factors=factors,
        horizon=5,
        train_window=60,
        min_train_samples=30,
        refit_every=25,
        panel_mode=panel_mode,
        embargo_days=0,
        share_pool_fit=share_pool_fit,
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )

    if panel_mode == "pooled":
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
            shared_cache=shared_cache if shared_cache is not None else {},
        )
    else:
        return MLFactorStrategy(cfg=cfg, pool_data=pool, shared_cache=None)


def _run_generate_signals(
    strategy: MLFactorStrategy,
    daily_df: pd.DataFrame,
    stock_code: str,
    use_batch: bool,
) -> pd.DataFrame:
    """Run generate_signals with the specified batch-predict mode."""
    if use_batch:
        os.environ.pop("STOCKPOOL_DISABLE_BATCH_PREDICT", None)
    else:
        os.environ["STOCKPOOL_DISABLE_BATCH_PREDICT"] = "1"
    strat = strategy.with_stock(stock_code)
    return strat.generate_signals(daily_df)


def _compare_signals(sig_a: pd.DataFrame, sig_b: pd.DataFrame, label: str) -> None:
    """Assert score and signal columns are bit-exact between two runs."""
    np.testing.assert_allclose(
        sig_a["score"].values,
        sig_b["score"].values,
        rtol=1e-12,
        atol=0,
        equal_nan=True,
        err_msg=f"{label}: score mismatch",
    )
    signal_match = (sig_a["signal"] == sig_b["signal"]).all()
    if not signal_match:
        mismatches = sig_a.index[sig_a["signal"] != sig_b["signal"]]
        raise AssertionError(
            f"{label}: signal mismatch at bars {list(mismatches)}\n"
            f"  batch:  {sig_b.loc[mismatches, 'signal'].values}\n"
            f"  per-bar:{sig_a.loc[mismatches, 'signal'].values}"
        )
    log.info("%s OK: score bit-exact, signals match; n=%d", label, len(sig_a))


def main() -> None:
    factors = ["momentum_5", "momentum_10", "momentum_20"]

    # ── Test 1: per_stock mode (single stock, no pool sharing) ───────────────
    log.info("=== Test 1: per_stock mode ===")
    n_bars = 300
    pool_single = {"A000001": _stock_df(n_bars=n_bars, seed=42)}
    strat_single = _make_strategy(pool_single, factors, panel_mode="per_stock")
    daily_df = pool_single["A000001"]

    sig_perbar = _run_generate_signals(strat_single, daily_df, "A000001", use_batch=False)
    sig_batch = _run_generate_signals(strat_single, daily_df, "A000001", use_batch=True)
    _compare_signals(sig_perbar, sig_batch, "Test 1 (per_stock)")

    # ── Test 2: pooled mode, multiple stocks ──────────────────────────────────
    log.info("=== Test 2: pooled mode, multiple stocks ===")
    n_stocks = 8
    pool_multi = {f"S{i:03d}": _stock_df(n_bars=250, seed=i + 100) for i in range(n_stocks)}
    strat_pooled = _make_strategy(
        pool_multi, factors, panel_mode="pooled", share_pool_fit=False,
    )
    for code, daily in pool_multi.items():
        sig_perbar = _run_generate_signals(strat_pooled, daily, code, use_batch=False)
        sig_batch = _run_generate_signals(strat_pooled, daily, code, use_batch=True)
        _compare_signals(sig_perbar, sig_batch, f"Test 2 stock {code}")

    # ── Test 3: pooled + share_pool_fit (cache-sync elif branch) ─────────────
    log.info("=== Test 3: pooled + share_pool_fit (cache-sync elif branch) ===")
    n_stocks = 6
    pool_shared = {f"T{i:03d}": _stock_df(n_bars=250, seed=i + 200) for i in range(n_stocks)}
    shared_cache: dict = {}
    strat_shared = _make_strategy(
        pool_shared, factors, shared_cache=shared_cache,
        panel_mode="pooled", share_pool_fit=True,
    )
    # First stock populates the shared_cache; subsequent stocks hit the elif
    # (cache-sync) branch — exactly the branch batch-predict must handle.
    for code, daily in pool_shared.items():
        sig_perbar = _run_generate_signals(strat_shared, daily, code, use_batch=False)
        sig_batch = _run_generate_signals(strat_shared, daily, code, use_batch=True)
        _compare_signals(sig_perbar, sig_batch, f"Test 3 stock {code}")

    # ── Test 4: short history (edge — first fit fires late, all-NaN warmup) ──
    log.info("=== Test 4: short history / all-NaN warmup rows ===")
    pool_short = {"Z001": _stock_df(n_bars=80, seed=99)}
    strat_short = _make_strategy(pool_short, factors, panel_mode="per_stock")
    daily_short = pool_short["Z001"]
    sig_perbar = _run_generate_signals(strat_short, daily_short, "Z001", use_batch=False)
    sig_batch = _run_generate_signals(strat_short, daily_short, "Z001", use_batch=True)
    _compare_signals(sig_perbar, sig_batch, "Test 4 (short history)")

    # ── Test 5: batch boundary (refit_every divides n exactly) ───────────────
    log.info("=== Test 5: batch boundary alignment ===")
    # refit_every=25, n=125 → exactly 5 refit intervals; boundary bars must
    # not fall in the wrong batch slot.
    pool_boundary = {"B001": _stock_df(n_bars=125, seed=77)}
    strat_boundary = _make_strategy(pool_boundary, factors, panel_mode="per_stock")
    daily_boundary = pool_boundary["B001"]
    sig_perbar = _run_generate_signals(strat_boundary, daily_boundary, "B001", use_batch=False)
    sig_batch = _run_generate_signals(strat_boundary, daily_boundary, "B001", use_batch=True)
    _compare_signals(sig_perbar, sig_batch, "Test 5 (boundary alignment)")

    # Summary
    nan_ratio_perbar = sig_perbar["score"].isna().mean()
    nan_ratio_batch = sig_batch["score"].isna().mean()
    log.info(
        "NaN ratio perbar=%.1f%%, batch=%.1f%%",
        nan_ratio_perbar * 100, nan_ratio_batch * 100,
    )

    print(
        f"\nOK: batch == per-bar bit-exact within rtol=1e-12; "
        f"all 5 test scenarios passed."
    )


if __name__ == "__main__":
    main()
