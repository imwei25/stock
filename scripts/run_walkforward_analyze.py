"""Walk-forward wrapper around analyze_factors: splits the date range
in half, runs once per half, saves to <output>/h1/<date>.json and h2/."""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from stockpool.config import load_config
from stockpool.factors_analysis import analyze_factors
from stockpool.factors_analysis_report import render_factor_analysis_report
from stockpool.factors.context import set_sector_map
from stockpool.industry_map import load_or_build_industry_map
from stockpool.panel import build_panel_from_cache


def _slice_panel(panel: dict, start, end) -> dict:
    return {k: v.loc[start:end] for k, v in panel.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--factors-file", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--horizon", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(cfg.data.cache_dir)
    universe_file = cache_dir / "universe.parquet"
    all_codes = pd.read_parquet(universe_file)["code"].tolist()
    codes = [c for c in all_codes if (cache_dir / f"{c}_daily.parquet").exists()]

    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    set_sector_map(sector_map or {})

    panel = build_panel_from_cache(codes, cfg.data.history_days, cache_dir)
    factor_names = list(json.loads(
        args.factors_file.read_text(encoding="utf-8"))["factors"])

    dates = panel["close"].index
    mid = dates[len(dates) // 2]
    halves = [
        ("h1", dates.min(), mid),
        ("h2", mid + pd.Timedelta(days=1), dates.max()),
    ]
    stamp = date.today().isoformat()
    for tag, lo, hi in halves:
        sub = _slice_panel(panel, lo, hi)
        result = analyze_factors(
            panel=sub, factor_names=factor_names,
            horizon=args.horizon,
        )
        out_dir = args.output_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        result.to_json(out_dir / f"{stamp}.json")
        render_factor_analysis_report(result, out_dir / f"{stamp}.html")
        print(f"wrote {tag} to {out_dir}/{stamp}.{{json,html}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
