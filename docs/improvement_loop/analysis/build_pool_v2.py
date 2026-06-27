"""M1: build a SECOND, disjoint evaluation pool (ab_pool_v2.parquet).

Rationale (audit H3): the original ab_pool is top-2-mcap + top-2-liq per industry
-> survivorship/large-cap tilt that couples train and eval. A confirmed win must
also hold on a pool that is NOT large-cap stratified.

Design: liquidity-DECILE stratified random sample, EXCLUDING the original ab_pool
codes (disjoint), industry-spread, fully offline (universe.parquet + cached daily
bars only; no baostock/akshare). ~240 names to match the original pool size.

Run: .venv/Scripts/python.exe docs/improvement_loop/analysis/build_pool_v2.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from stockpool.config import load_config
    from stockpool.fetcher import load_universe_cache
    from stockpool.industry_map import load_or_build_industry_map
    from stockpool.ab_pool import load_ab_pool

    cfg = load_config("config.yaml")
    cache_dir = Path(cfg.data.cache_dir)
    rng = np.random.default_rng(20260627)

    uni = pd.read_parquet(cache_dir / "universe.parquet")
    name_map = dict(zip(uni["code"], uni.get("name", uni["code"])))
    pool_data = load_universe_cache(cache_dir, cfg.data.history_days,
                                    warmup_days=cfg.data.warmup_days)
    sector_map = load_or_build_industry_map(cache_dir, source="auto")

    # exclude original pool (disjoint second pool)
    try:
        orig = set(str(c).zfill(6) for c in load_ab_pool(cfg.ab_pool.cache_path)["code"])
    except Exception:
        orig = set()

    MIN_BARS = 250
    MIN_AMT = 5.0e7  # same liquidity floor as recommend_pool (vol*close*100)
    rows = []
    for code, df in pool_data.items():
        if code in orig or len(df) < MIN_BARS:
            continue
        name = str(name_map.get(code, code))
        if "st" in name.lower() or "ST" in name or "*" in name:
            continue
        tail = df.tail(20)
        amt = float((tail["close"] * tail["volume"] * 100).mean())
        if not np.isfinite(amt) or amt < MIN_AMT:
            continue
        rows.append((code, name, sector_map.get(code, "未知"), amt))

    df = pd.DataFrame(rows, columns=["code", "name", "industry", "avg_amount_20d"])
    if df.empty:
        print("ERROR: no eligible stocks")
        return 1

    # liquidity-decile stratified random sample -> ~240, spread across deciles
    df["liq_decile"] = pd.qcut(df["avg_amount_20d"].rank(method="first"), 10, labels=False)
    per_decile = 24
    picks = []
    for d, g in df.groupby("liq_decile"):
        take = min(per_decile, len(g))
        picks.append(g.sample(take, random_state=int(rng.integers(0, 1_000_000))))
    out = pd.concat(picks).drop(columns="liq_decile").reset_index(drop=True)
    out["source_tag"] = "liq_decile_v2"
    out["build_date"] = "2026-06-27"

    out_path = cache_dir / "ab_pool_v2.parquet"
    out.to_parquet(out_path, index=False)
    overlap = len(set(out["code"]) & orig)
    print(f"ab_pool_v2: {len(out)} codes, {out['industry'].nunique()} industries, "
          f"overlap with original={overlap}, saved {out_path}")
    print("industry spread (top 8):", out["industry"].value_counts().head(8).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
