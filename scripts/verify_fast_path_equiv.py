"""End-to-end equivalence check: fast path (shared_cache pre-stack) vs
legacy path (per-call re-stack) must produce identical backtest results.

Runs cmd_backtest twice on the SAME small config:
  1. legacy: monkey-patch ``_ensure_pooled_xy_long`` to always return None.
  2. fast: restore original (current) behavior.

Compares per-stock equity curves and metrics for near-equality.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

_CAPTURED: dict[str, list] = {"per_stock": []}


def install_capture():
    from stockpool import backtest_runner as br

    orig = br.backtest_stocks

    def wrapper(*args, **kwargs):
        per, failed = orig(*args, **kwargs)
        _CAPTURED["per_stock"] = per
        return per, failed

    br.backtest_stocks = wrapper


def install_universe_limiter(limit: int):
    from stockpool import backtest_runner

    orig_prepare = backtest_runner.prepare_pool

    def patched(cfg, stocks, force_refresh, refresh_factor_panel=False):
        pool, fp, cp = orig_prepare(cfg, stocks, force_refresh, refresh_factor_panel)
        if pool is None or len(pool) <= limit:
            return pool, fp, cp
        app_codes = {s.code for s in stocks}
        kept = {c: pool[c] for c in pool if c in app_codes}
        for c in pool:
            if c not in kept and len(kept) < limit:
                kept[c] = pool[c]
        fp2 = {nm: w[[c for c in w.columns if c in kept]] for nm, w in fp.items()}
        cp2 = cp[[c for c in cp.columns if c in kept]]
        return kept, fp2, cp2

    backtest_runner.prepare_pool = patched


def run_once(args, legacy: bool) -> list[tuple]:
    """Run cmd_backtest once. If legacy=True, patch the fast path off."""
    from stockpool.backtesting import strategies

    orig_method = strategies.MLFactorStrategy._ensure_pooled_xy_long
    if legacy:
        def always_none(self):
            return None
        strategies.MLFactorStrategy._ensure_pooled_xy_long = always_none

    try:
        _CAPTURED["per_stock"] = []
        import argparse as _ap
        bt_args = _ap.Namespace(
            config=args.config, stocks=args.stocks,
            refresh=False, refresh_factor_panel=False,
        )
        from stockpool.cli import cmd_backtest
        rc = cmd_backtest(bt_args)
        if rc != 0:
            raise SystemExit(f"backtest returned non-zero: {rc}")
        return list(_CAPTURED["per_stock"])
    finally:
        strategies.MLFactorStrategy._ensure_pooled_xy_long = orig_method


def compare_curves(fast_df: pd.DataFrame, legacy_df: pd.DataFrame) -> dict:
    out = {}
    common = [c for c in fast_df.columns if c in legacy_df.columns]
    for c in common:
        if not pd.api.types.is_numeric_dtype(fast_df[c]):
            continue
        a = fast_df[c].to_numpy(dtype=float)
        b = legacy_df[c].to_numpy(dtype=float)
        if len(a) != len(b):
            out[c] = float("inf")
            continue
        a_isnan = np.isnan(a); b_isnan = np.isnan(b)
        if not np.array_equal(a_isnan, b_isnan):
            out[c] = float("inf")
            continue
        m = ~a_isnan
        if m.any():
            out[c] = float(np.max(np.abs(a[m] - b[m])))
        else:
            out[c] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stocks", default="605589")
    parser.add_argument("--limit-universe", type=int, default=100)
    args = parser.parse_args()

    install_universe_limiter(args.limit_universe)
    install_capture()

    # Run legacy FIRST so the shared_cache pre-stack from a possible prior
    # run doesn't accidentally short-circuit anything (cmd_backtest builds a
    # fresh shared_cache per call, but be explicit).
    print("[1/2] running LEGACY path (forced fallback)...", flush=True)
    legacy_per_stock = run_once(args, legacy=True)

    print("[2/2] running FAST path (current code)...", flush=True)
    fast_per_stock = run_once(args, legacy=False)

    print()
    print("=" * 72)
    print(f"COMPARISON — fast: {len(fast_per_stock)} stocks, legacy: {len(legacy_per_stock)}")
    print("=" * 72)

    fast_map = {code: (name, result) for code, name, result in fast_per_stock}
    legacy_map = {code: (name, result) for code, name, result in legacy_per_stock}

    all_codes = sorted(set(fast_map) | set(legacy_map))
    overall_ok = True
    for code in all_codes:
        if code not in fast_map or code not in legacy_map:
            print(f"  {code}: MISSING (fast={code in fast_map}, legacy={code in legacy_map})")
            overall_ok = False
            continue
        _, r_fast = fast_map[code]
        _, r_legacy = legacy_map[code]
        ok = True
        worst = 0.0
        for N in sorted(r_fast.curves.keys()):
            if N not in r_legacy.curves:
                print(f"  {code} N={N}: curve missing in legacy")
                ok = False
                continue
            diffs = compare_curves(r_fast.curves[N], r_legacy.curves[N])
            mx = max(diffs.values()) if diffs else 0.0
            worst = max(worst, mx)
            if mx > 1e-9:
                print(f"  {code} N={N}: max abs diff = {mx:.3e}  per-col: {diffs}")
                ok = False
        for N in sorted(r_fast.metrics.keys()):
            mf = r_fast.metrics[N]
            ml = r_legacy.metrics.get(N, {})
            for k, vf in mf.items():
                if not isinstance(vf, (int, float)):
                    continue
                vl = ml.get(k, float("nan"))
                if pd.isna(vf) and pd.isna(vl):
                    continue
                d = abs(float(vf) - float(vl))
                if d > 1e-6:
                    print(f"  {code} N={N} metric {k}: fast={vf:.6f} legacy={vl:.6f} (Δ={d:.2e})")
                    ok = False
        print(f"  {code}: {'OK' if ok else 'DIFF'} (max curve Δ = {worst:.2e})")
        if not ok:
            overall_ok = False

    print("=" * 72)
    print("RESULT:", "ALL EQUIVALENT" if overall_ok else "DIFFERENCES FOUND")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
