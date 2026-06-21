"""Wall-clock benchmark: StaggeredRunner serial vs parallel.

Uses a realistic-size synthetic panel + precomputed score panel to focus
the timing on the N-offset engine.run loop (the part that PR-T1.3
parallelises). Skips ml_factor training + score_panel build — those are
upstream of what we're benchmarking.

Usage:
    .venv/Scripts/python.exe scripts/bench_staggered_parallel.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stockpool.backtesting.framework import TradeCosts
from stockpool.config import PortfolioRunConfig
from stockpool.portfolio.engine import PortfolioEngine
from stockpool.portfolio.ensemble import StaggeredRunner
from stockpool.portfolio.strategy import PrecomputedScoreStrategy


def _make_panel(n_codes: int, n_bars: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_bars, freq="B")
    panel = {}
    for i in range(n_codes):
        code = f"S{i:04d}"
        prices = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_bars))
        panel[code] = pd.DataFrame({
            "date": dates,
            "open": prices * (1 + rng.normal(0, 0.001, n_bars)),
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": rng.integers(1e5, 5e6, n_bars),
        })
    return panel, dates


def run_once(parallel: bool, n_offsets: int, panel, scores) -> tuple[float, np.ndarray]:
    codes = list(panel.keys())
    strategy = PrecomputedScoreStrategy(scores, name="bench")
    portfolio_cfg = PortfolioRunConfig(top_k=20, rebalance_n_days=5)
    costs = TradeCosts(buy_cost=0.001, sell_cost=0.001)
    components = (strategy, portfolio_cfg, costs, 0.02, None, {})

    def _factory():
        return PortfolioEngine(
            strategy=strategy, portfolio_cfg=portfolio_cfg, costs=costs,
            risk_free_rate=0.02, eligibility=None, sector_map={},
        )

    runner = StaggeredRunner(_factory, components=components, risk_free_rate=0.02)
    t0 = time.perf_counter()
    ens = runner.run(panel, n_offsets=n_offsets, parallel=parallel)
    elapsed = time.perf_counter() - t0
    return elapsed, ens.ensemble_curve["equity"].values


def main():
    # Realistic-ish size: ~100 stocks × ~500 bars (~2 years daily).
    # Big enough that each engine.run takes seconds (not ms) so we
    # actually see parallelism win.
    import sys
    if len(sys.argv) >= 3:
        N_CODES, N_BARS = int(sys.argv[1]), int(sys.argv[2])
    else:
        N_CODES, N_BARS = 200, 500
    N_OFFSETS = 5  # rebalance_n_days = 5 → max useful offsets
    print(f"Panel: {N_CODES} codes × {N_BARS} bars, n_offsets={N_OFFSETS}")
    print()

    panel, dates = _make_panel(N_CODES, N_BARS)
    rng_scores = np.random.default_rng(0)
    scores = pd.DataFrame(
        rng_scores.standard_normal((len(dates), N_CODES)),
        index=dates, columns=panel.keys(),
    )

    # Warmup: tiny pre-run so JIT / import overhead doesn't get charged to serial.
    print("Warmup (single engine, ignored) ...", flush=True)
    _ = run_once(parallel=False, n_offsets=1, panel=panel, scores=scores)
    print()

    print("Serial run ...", flush=True)
    t_serial, curve_serial = run_once(False, N_OFFSETS, panel, scores)
    print(f"  wall: {t_serial:.2f} s")
    print()

    print("Parallel run (ProcessPoolExecutor) ...", flush=True)
    t_par, curve_par = run_once(True, N_OFFSETS, panel, scores)
    print(f"  wall: {t_par:.2f} s")
    print()

    speedup = t_serial / t_par if t_par > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}× (serial / parallel)")

    # Equivalence sanity check.
    max_abs_diff = float(np.max(np.abs(curve_serial - curve_par)))
    max_rel_diff = float(np.max(np.abs(curve_serial - curve_par) / np.abs(curve_serial)))
    print(f"Equivalence: max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e}")
    if max_rel_diff > 1e-12:
        print("WARN: exceeds rtol=1e-12 — investigate FP determinism")


if __name__ == "__main__":
    main()
