"""Minimal example: verify the portfolio-AB perf optimizations are bit-exact.

Run:
    python scripts/verify_portfolio_ab_perf.py [N_STOCKS]

Compares, on the local ``data/`` cache (no network), the OPTIMIZED code paths
against verbatim copies of the pre-optimization "slow" logic, and prints timing.

Two optimizations covered (see docs/superpowers/plans/2026-06-24-portfolio-ab-perf-decisions.md):
  1. CompositeVerdictStrategy weekly score: O(T^2) per-bar resample+add_all ->
     one-shot precompute. Must be bit-identical.
  2. EligibilityFilter: per-bar re-parse/re-aggregate -> one-shot precompute +
     searchsorted. Must be bit-identical.

Exit code 0 iff both are bit-exact (0 mismatches).
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from stockpool.backtesting.composite_weekly import weekly_scores_by_bar
from stockpool.backtesting.strategies import DAILY_WARMUP, WEEKLY_WARMUP
from stockpool.config import load_config
from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.portfolio.eligibility import EligibilityFilter, _is_st
from stockpool.signals import detect_signals, score_triggers


def _slow_weekly(daily, ind, weights, start, warmup):
    out = {}
    for i in range(start, len(daily)):
        wk = resample_to_weekly(daily.iloc[:i + 1])
        out[i] = (score_triggers(detect_signals(add_all(wk, ind), weights))
                  if len(wk) >= warmup else 0)
    return out


def _slow_eligible(cfg, name_map, date_t, panel):
    date_t = pd.Timestamp(date_t)
    out = set()
    for code, daily in panel.items():
        if cfg.exclude_st and _is_st(name_map.get(code, "")):
            continue
        if "date" not in daily.columns or "close" not in daily.columns:
            continue
        df = daily[pd.to_datetime(daily["date"]) <= date_t]
        if len(df) < cfg.min_history_bars:
            continue
        if cfg.min_avg_amount_20d > 0:
            if "volume" not in df.columns:
                continue
            recent = df.tail(20)
            if len(recent) == 0:
                continue
            avg = float((recent["close"].astype(float) * recent["volume"].astype(float) * 100.0).mean())
            if pd.isna(avg) or avg < cfg.min_avg_amount_20d:
                continue
        out.add(code)
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cfg = load_config("portfolio_ab_simple_base.yaml")
    files = sorted(glob.glob("data/*_daily.parquet"))[:n]
    if not files:
        print("No data/*_daily.parquet found — run `stockpool fetch-universe` first.")
        return 2
    panel = {os.path.basename(f).split("_")[0]: pd.read_parquet(f) for f in files}
    print(f"Loaded {len(panel)} stocks from data/.")

    # ---- 1. Composite weekly scores ----
    print("\n[1] CompositeVerdictStrategy weekly score (fast vs slow)")
    start = DAILY_WARMUP - 1
    mism = 0
    t_fast = t_slow = 0.0
    for code, daily in panel.items():
        t0 = time.perf_counter()
        fast = weekly_scores_by_bar(daily, cfg.indicators, cfg.weights, start, WEEKLY_WARMUP)
        t_fast += time.perf_counter() - t0
        t0 = time.perf_counter()
        slow = _slow_weekly(daily, cfg.indicators, cfg.weights, start, WEEKLY_WARMUP)
        t_slow += time.perf_counter() - t0
        mism += sum(1 for i in slow if slow[i] != fast.get(i))
    print(f"    mismatches: {mism}")
    print(f"    slow: {t_slow:.2f}s   fast: {t_fast:.2f}s   speedup: {t_slow / max(t_fast, 1e-9):.1f}x")

    # ---- 2. Eligibility ----
    print("\n[2] EligibilityFilter (fast vs slow)")
    ecfg = cfg.portfolio_backtest.eligibility
    name_map = {c: "" for c in panel}
    all_dates = sorted(set(pd.concat([d["date"] for d in panel.values()])))
    test_dates = all_dates[60::10]
    f = EligibilityFilter(ecfg, name_map=name_map)
    emism = 0
    t0 = time.perf_counter()
    fast_sets = {dt: f.eligible(dt, panel) for dt in test_dates}
    t_efast = time.perf_counter() - t0
    t0 = time.perf_counter()
    for dt in test_dates:
        if _slow_eligible(ecfg, name_map, dt, panel) != fast_sets[dt]:
            emism += 1
    t_eslow = time.perf_counter() - t0
    print(f"    dates checked: {len(test_dates)}   mismatches: {emism}")
    print(f"    slow: {t_eslow:.2f}s   fast: {t_efast:.2f}s   speedup: {t_eslow / max(t_efast, 1e-9):.1f}x")
    print("    (eligibility win scales with universe x rebalances; at ~12 stocks it isn't the bottleneck)")

    ok = (mism == 0 and emism == 0)
    print(f"\nRESULT: {'BIT-EXACT (0 mismatches)' if ok else 'MISMATCHES FOUND'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
