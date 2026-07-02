"""Compute as-of top-N liquidity membership over the full cached universe.

Answers two questions for the survivorship-corrected re-verification wave:
  1. How many codes were EVER in the PIT top-N (union size = scoring cost)?
  2. How many member-days do the backfilled delisted codes contribute?

Writes ``data/asof_top<N>_union.parquet`` with per-code member-day counts and
an ``is_delisted`` flag, to be used as the scored-code set.

Usage:
  .venv/Scripts/python.exe docs/improvement_loop/analysis/asof_membership_stats.py \\
      [--top-n 1000] [--min-member-days 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--min-member-days", type=int, default=20)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--history-days", type=int, default=3750)
    ap.add_argument("--warmup-days", type=int, default=200)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from stockpool.fetcher import load_universe_cache

    cache_dir = Path(args.cache_dir)
    pool = load_universe_cache(cache_dir, args.history_days,
                               warmup_days=args.warmup_days)
    print(f"[asof] loaded {len(pool)} codes")

    amt = {}
    for c, df in pool.items():
        if "volume" in df.columns and "close" in df.columns:
            s = df.set_index("date")
            amt[c] = (s["close"] * s["volume"] * 100.0).astype(float)
    wide = pd.DataFrame(amt).sort_index()
    avg20 = wide.rolling(20, min_periods=1).mean()
    rank = avg20.rank(axis=1, ascending=False, method="first")
    member = rank <= args.top_n
    del wide, avg20, rank

    days = member.sum(axis=0)
    ever = days[days > 0].sort_values(ascending=False)
    kept = days[days >= args.min_member_days]

    delisted_path = cache_dir / "universe_delisted.parquet"
    delisted = (set(pd.read_parquet(delisted_path)["code"].astype(str))
                if delisted_path.exists() else set())
    uni_path = cache_dir / "universe.parquet"
    listed = (set(pd.read_parquet(uni_path)["code"].astype(str).str.zfill(6))
              if uni_path.exists() else set())

    static = None
    static_path = cache_dir / "top1000_liquid.parquet"
    if static_path.exists():
        static = set(pd.read_parquet(static_path)["code"].astype(str).str.zfill(6))

    print(f"[asof] union ever-member: {len(ever)} codes")
    print(f"[asof] union with >= {args.min_member_days} member-days: {len(kept)}")
    n_delisted = sum(1 for c in kept.index if c in delisted)
    print(f"[asof]   of which delisted: {n_delisted}")
    if static is not None:
        missed = [c for c in kept.index if c not in static]
        print(f"[asof]   not in static top1000_liquid: {len(missed)} "
              f"(these were liquid at some point but invisible to the "
              f"static pool)")
    out = pd.DataFrame({
        "code": kept.index,
        "member_days": kept.values,
    })
    out["is_delisted"] = out["code"].isin(delisted)
    out["in_current_universe"] = out["code"].isin(listed)
    out_path = cache_dir / f"asof_top{args.top_n}_union.parquet"
    out.to_parquet(out_path)
    print(f"[asof] wrote {out_path} ({len(out)} codes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
