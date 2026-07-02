"""Time-holdout factor-pool reselection — the true-OOS test for the P0-3 win.

The production ``selection.json`` (15yr-IR pool) was picked with ``factors
analyze`` over the FULL 15-yr window and then "validated" via Layer-B IC on
that same window: the walk-forward model is per-step OOS, but the *candidate
pool itself* saw the whole evaluation period — classic selection bias. The
disjoint second pool (pool2) fixed the universe axis, not the time axis.

This script re-runs the exact P0-3 pipeline with the panel truncated to
``--cutoff`` (default 2020-12-31):

  analyze(386 factors, top-1000 pool, data ≤ cutoff, label_basis=open,
          horizon=3) → pick_top_factors(score_by=ir, min_ir=0.15,
          max_corr=0.6, max_degenerate_ratio=0.3, top_n=30)

and writes ``reports/selection_holdout<year>.json``. Comparing this pool vs
the production pool via layer_b_direct.py --date-start <cutoff+1> answers:
"how much of the +18.5% ΔIC survives when the pool cannot see the eval years?"

Usage:
  .venv/Scripts/python.exe docs/improvement_loop/analysis/holdout_reselect.py \\
      --pool data/top1000_liquid.parquet --cutoff 2020-12-31
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/top1000_liquid.parquet")
    ap.add_argument("--cutoff", default="2020-12-31")
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--history-days", type=int, default=3750)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--min-ir", type=float, default=0.15)
    ap.add_argument("--max-corr", type=float, default=0.6)
    ap.add_argument("--max-degenerate-ratio", type=float, default=0.3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from stockpool.factors import list_factors
    from stockpool.factors.context import set_sector_map
    from stockpool.factors_analysis import analyze_factors, pick_top_factors
    from stockpool.industry_map import load_or_build_industry_map
    from stockpool.panel import build_panel_from_cache

    cutoff = pd.Timestamp(args.cutoff)
    cache_dir = Path(args.cache_dir)
    codes = [str(c).zfill(6) for c in pd.read_parquet(args.pool)["code"]]
    codes = [c for c in codes if (cache_dir / f"{c}_daily.parquet").exists()]
    print(f"[holdout] pool codes with cache: {len(codes)}")

    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    set_sector_map(sector_map)

    print(f"[holdout] building panel (history_days={args.history_days})…")
    panel = build_panel_from_cache(codes, args.history_days, cache_dir)
    # Truncate EVERY field to <= cutoff — the pool selection must not see a
    # single bar after it.
    panel = {k: v.loc[v.index <= cutoff] for k, v in panel.items()}
    t_bars = len(panel["close"])
    print(f"[holdout] panel truncated to <= {cutoff.date()}: {t_bars} bars, "
          f"{panel['close'].shape[1]} codes")

    names = list_factors()
    print(f"[holdout] analyzing {len(names)} factors "
          f"(horizon={args.horizon}, label_basis=open)…")
    result = analyze_factors(
        panel, names, horizon=args.horizon, label_basis="open",
    )

    picked = pick_top_factors(
        result,
        top_n=args.top_n,
        max_correlation=args.max_corr,
        min_ir=args.min_ir,
        score_by="ir",
        max_degenerate_ratio=args.max_degenerate_ratio,
    )
    print(f"[holdout] picked {len(picked)}:")
    for name in picked:
        print(f"  {name}")

    out = Path(args.output or f"reports/selection_holdout{cutoff.year}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"factors": picked}, indent=2), encoding="utf-8")
    print(f"[holdout] wrote {out}")

    # Also archive the analyze result for later inspection.
    res_path = out.with_suffix(".analyze.json")
    try:
        res_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
        print(f"[holdout] archived analyze result to {res_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[holdout] analyze archive skipped ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
