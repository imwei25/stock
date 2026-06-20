"""One-shot wall-clock profiler for `python -m stockpool backtest`.

Monkey-patches the hot ml_factor functions with time.perf_counter() wrappers,
runs cmd_backtest on a given config, and prints a table of per-step totals +
call counts at the end. No code changes to the package itself — delete this
script when profiling is done.

Usage:
    python scripts/profile_backtest.py --config config.yaml [--stocks 605589]

Pass the same args you'd pass to `python -m stockpool backtest`.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict


_TIMINGS: dict[str, list[float]] = defaultdict(list)


def _wrap(label: str, fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _TIMINGS[label].append(time.perf_counter() - t0)
    wrapper.__wrapped__ = fn
    wrapper.__name__ = getattr(fn, "__name__", label)
    return wrapper


def install_patches(limit_universe: int | None = None):
    """Monkey-patch the hot path. Order matters: import THEN replace attr."""
    from stockpool import backtest_runner
    from stockpool import strategy_factory
    from stockpool.backtesting import strategies as bt_strategies
    from stockpool.ml import dataset as ml_dataset
    from stockpool.ml import pipeline as ml_pipeline
    from stockpool import fetcher

    # Top-level entry points.
    orig_prepare = backtest_runner.prepare_pool

    def patched_prepare(cfg, stocks, force_refresh, refresh_factor_panel=False):
        t0 = time.perf_counter()
        try:
            pool, fp, cp = orig_prepare(cfg, stocks, force_refresh, refresh_factor_panel)
            if limit_universe is not None and pool is not None and len(pool) > limit_universe:
                # Keep app stocks first; truncate the rest. Re-build factor/close
                # panel against the truncated pool so widths match.
                app_codes = {s.code for s in stocks}
                kept = {c: pool[c] for c in pool if c in app_codes}
                for c in pool:
                    if c not in kept and len(kept) < limit_universe:
                        kept[c] = pool[c]
                pool = kept
                print(f"[profile] universe truncated to {len(pool)} stocks", flush=True)
                # Rebuild panels to match the truncated pool (drop unused columns).
                fp2: dict = {}
                codes = list(pool.keys())
                for nm, wide in fp.items():
                    keep_cols = [c for c in wide.columns if c in pool]
                    fp2[nm] = wide[keep_cols]
                cp2 = cp[[c for c in cp.columns if c in pool]]
                return pool, fp2, cp2
            return pool, fp, cp
        finally:
            _TIMINGS["prepare_pool"].append(time.perf_counter() - t0)

    backtest_runner.prepare_pool = patched_prepare
    backtest_runner.backtest_stocks = _wrap("backtest_stocks", backtest_runner.backtest_stocks)

    # Pool + panel loads.
    fetcher.load_universe_cache = _wrap("load_universe_cache", fetcher.load_universe_cache)
    strategy_factory.load_or_build_factor_panel = _wrap(
        "load_or_build_factor_panel", strategy_factory.load_or_build_factor_panel
    )
    strategy_factory.build_factor_panel = _wrap(
        "build_factor_panel(MISS)", strategy_factory.build_factor_panel
    )

    # The big suspects: per-refit training-set materialization.
    orig_build_xy = bt_strategies.MLFactorStrategy._build_pooled_xy_from_panel

    def patched_build_xy(self, daily_df, current_bar):
        t0 = time.perf_counter()
        try:
            return orig_build_xy(self, daily_df, current_bar)
        finally:
            _TIMINGS["_build_pooled_xy_from_panel"].append(time.perf_counter() - t0)

    bt_strategies.MLFactorStrategy._build_pooled_xy_from_panel = patched_build_xy

    # The two sub-steps inside _build_pooled_xy_from_panel.
    ml_dataset.forward_return_panel = _wrap("forward_return_panel", ml_dataset.forward_return_panel)
    ml_dataset.stack_panel_to_xy = _wrap("stack_panel_to_xy", ml_dataset.stack_panel_to_xy)
    # strategies.py imported these by name — patch its module too.
    bt_strategies.forward_return_panel = ml_dataset.forward_return_panel
    bt_strategies.stack_panel_to_xy = ml_dataset.stack_panel_to_xy

    # Fit / predict.
    orig_fit = ml_pipeline.TwoStepPipeline.fit

    def patched_fit(self, X, y):
        t0 = time.perf_counter()
        try:
            return orig_fit(self, X, y)
        finally:
            _TIMINGS["TwoStepPipeline.fit"].append(time.perf_counter() - t0)

    ml_pipeline.TwoStepPipeline.fit = patched_fit

    orig_predict = ml_pipeline.TwoStepPipeline.predict

    def patched_predict(self, X):
        t0 = time.perf_counter()
        try:
            return orig_predict(self, X)
        finally:
            _TIMINGS["TwoStepPipeline.predict"].append(time.perf_counter() - t0)

    ml_pipeline.TwoStepPipeline.predict = patched_predict

    # Per-stock generate_signals total.
    orig_gen = bt_strategies.MLFactorStrategy.generate_signals

    def patched_gen(self, daily_df):
        t0 = time.perf_counter()
        try:
            return orig_gen(self, daily_df)
        finally:
            _TIMINGS["MLFactorStrategy.generate_signals"].append(time.perf_counter() - t0)

    bt_strategies.MLFactorStrategy.generate_signals = patched_gen

    # _try_fit total (covers shared_cache lookup overhead too).
    orig_try_fit = bt_strategies.MLFactorStrategy._try_fit

    def patched_try_fit(self, daily_df, X_full, y_full, current_bar):
        t0 = time.perf_counter()
        try:
            return orig_try_fit(self, daily_df, X_full, y_full, current_bar)
        finally:
            _TIMINGS["MLFactorStrategy._try_fit"].append(time.perf_counter() - t0)

    bt_strategies.MLFactorStrategy._try_fit = patched_try_fit


def print_report(total_wall: float):
    print()
    print("=" * 78)
    print(f"PROFILING SUMMARY (total wall: {total_wall:.1f}s)")
    print("=" * 78)
    rows = []
    for label, times in _TIMINGS.items():
        total = sum(times)
        rows.append((label, total, len(times), total / max(1, len(times))))
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'step':<42} {'total(s)':>10} {'%wall':>7} {'calls':>8} {'avg(s)':>10}")
    print("-" * 78)
    for label, total, n, avg in rows:
        pct = 100.0 * total / total_wall if total_wall > 0 else 0.0
        print(f"{label:<42} {total:>10.2f} {pct:>6.1f}% {n:>8d} {avg:>10.4f}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh-factor-panel", action="store_true")
    parser.add_argument("--limit-universe", type=int, default=None,
                        help="Cap pool_data to first N stocks (app stocks kept).")
    args = parser.parse_args()

    install_patches(limit_universe=args.limit_universe)

    from stockpool.cli import cmd_backtest

    bt_args = argparse.Namespace(
        config=args.config,
        stocks=args.stocks,
        refresh=args.refresh,
        refresh_factor_panel=args.refresh_factor_panel,
    )

    t0 = time.perf_counter()
    rc = cmd_backtest(bt_args)
    total = time.perf_counter() - t0

    print_report(total)
    sys.exit(rc)


if __name__ == "__main__":
    main()
