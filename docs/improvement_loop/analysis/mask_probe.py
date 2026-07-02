"""P1-2 probe: does masking limit-up/halt bars from FACTOR INPUTS help the
score IC (survey B hypothesis) or hurt it (project prior: limit-up is signal)?

Faithful isolation: both arms use the SAME strategy config + the SAME (unmasked)
close_panel for labels; the ONLY difference is the factor-input panel —
  arm A (unmasked): factors computed on raw OHLCV (production behavior)
  arm B (masked):   factors computed on OHLCV with |ret|>=limit or volume<=0
                    bars NaN'd, so the NaN-safe ops skip those cells.
Then compute per-day cross-sectional rank-IC on top-1000 and paired block
bootstrap on ΔIC. Does NOT touch production code — masks a copy of pool_data.

Usage:
  .venv/Scripts/python.exe docs/improvement_loop/analysis/mask_probe.py \
      --pool data/top1000_liquid.parquet --workers 2 \
      --regime-boundaries 2015-06-15,2016-02-01,2018-06-15,2020-03-01,2022-01-01,2024-04-12
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

# reuse the significance helpers from layer_b_direct
sys.path.insert(0, str(Path(__file__).parent))
from layer_b_direct import _per_day_ic, _block_bootstrap_mean_ci  # noqa: E402


def _limit_threshold(code: str) -> float:
    c = str(code).zfill(6)
    if c[:2] in ("30", "68"):   # ChiNext / STAR
        return 0.198
    if c[:2] == "8" or c[:1] == "4":  # BSE fallback
        return 0.298
    return 0.098                 # main board


def _mask_pool_data(pool_data: dict) -> dict:
    """Return a copy where limit-up/down/halt bars have OHLCV set to NaN."""
    out = {}
    cols = ["open", "high", "low", "close", "volume"]
    for code, df in pool_data.items():
        d = df.copy()
        thr = _limit_threshold(code)
        ret = d["close"] / d["close"].shift(1) - 1.0
        bad = (ret.abs() >= thr) | (d["volume"] <= 0)
        for c in cols:
            if c in d.columns:
                d.loc[bad, c] = np.nan
        out[code] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pool", default="data/top1000_liquid.parquet")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--regime-boundaries", type=str, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from stockpool.config import load_config
    from stockpool.fetcher import load_universe_cache
    from stockpool.industry_map import load_or_build_industry_map
    from stockpool.factors.context import set_sector_map
    from stockpool.portfolio.scoring import precompute_scores_from_legacy
    from stockpool.strategy_factory import (
        build_factor_panel, build_close_panel, build_strategy,
    )

    cfg = load_config(args.config)
    cache_dir = Path(cfg.data.cache_dir)
    mf = cfg.strategy.ml_factor
    factors = list(mf.factors) if mf.factors else None
    if mf.factors_file:
        import json
        factors = json.load(open(mf.factors_file, encoding="utf-8"))["factors"]

    print(f"[setup] loading universe ({cfg.data.history_days} hist)…", flush=True)
    pool_data = load_universe_cache(cache_dir, cfg.data.history_days,
                                    warmup_days=cfg.data.warmup_days)
    print(f"[setup] training pool: {len(pool_data)} codes", flush=True)
    pool_df = pd.read_parquet(args.pool)
    port_codes = [str(c).zfill(6) for c in pool_df["code"]]
    port_pool = {c: pool_data[c] for c in port_codes if c in pool_data}
    print(f"[setup] portfolio universe: {len(port_pool)} codes", flush=True)

    set_sector_map(load_or_build_industry_map(cache_dir, source="auto"))

    # Shared UNMASKED close_panel (labels identical for both arms).
    close_panel = build_close_panel(pool_data)
    pp = mf.preprocess

    def score_arm(name, fdata):
        print(f"[{name}] building factor panel…", flush=True)
        fp = build_factor_panel(factors, fdata, preprocess_cfg=pp, cache_dir=str(cache_dir))
        print(f"[{name}] building strategy…", flush=True)
        legacy = build_strategy(cfg, pool_data=pool_data, factor_panel=fp,
                                close_panel=close_panel, shared_cache={})
        print(f"[{name}] precomputing scores ({len(port_pool)} stocks)…", flush=True)
        sp = precompute_scores_from_legacy(legacy, port_pool, n_workers=args.workers)
        print(f"[{name}] done shape={sp.shape}", flush=True)
        return sp

    sp_A = score_arm("unmasked", pool_data)
    print("[mask] building masked pool_data (NaN limit/halt bars)…", flush=True)
    masked = _mask_pool_data(pool_data)
    sp_B = score_arm("masked", masked)

    # Layer-B IC vs unmasked forward returns.
    close = pd.DataFrame({c: pool_data[c].set_index("date")["close"]
                          for c in port_codes if c in pool_data})
    fwd = close.shift(-args.horizon) / close - 1.0
    icA = _per_day_ic(sp_A, fwd); icB = _per_day_ic(sp_B, fwd)
    common = icA.index.intersection(icB.index)
    icA, icB = icA.loc[common], icB.loc[common]
    delta = (icB - icA).to_numpy()
    T = delta.size
    mean_d, std_d = float(delta.mean()), float(delta.std(ddof=1))
    from math import erf, sqrt
    t_stat = mean_d / (std_d / np.sqrt(T)) if std_d > 0 else float("nan")
    lo, hi, block = _block_bootstrap_mean_ci(delta, n_boot=args.n_boot)

    print("\n" + "=" * 70)
    print("MASK PROBE (Layer-B): masked (B) vs unmasked (A) factor inputs")
    print(f"T={T}  block={block}")
    print(f"IC mean   A(unmasked)={icA.mean():+.5f}  B(masked)={icB.mean():+.5f}  ΔIC={mean_d:+.5f}")
    print(f"IR proxy  A={icA.mean()/icA.std(ddof=1):+.3f}  B={icB.mean()/icB.std(ddof=1):+.3f}")
    print(f"paired t={t_stat:+.3f}   95% CI ΔIC=[{lo:+.5f}, {hi:+.5f}]  excludes 0: {(lo>0) or (hi<0)}")
    if args.regime_boundaries:
        bnds = sorted(pd.Timestamp(x) for x in args.regime_boundaries.split(","))
        edges = [common[0]] + bnds + [common[-1] + pd.Timedelta(days=1)]
        signs = []
        print("regime sub-periods:")
        for j in range(len(edges) - 1):
            m = (common >= edges[j]) & (common < edges[j + 1])
            if m.sum() < 5:
                continue
            sd = delta[m]; sm = sd.mean(); signs.append(np.sign(sm))
            print(f"  {edges[j].date()}~{(edges[j+1]-pd.Timedelta(days=1)).date()} n={int(m.sum()):>4} ΔIC={sm:+.5f} [{'+' if sm>0 else '-'}]")
        print(f"  consistent: {len(set(signs))==1}  ({sum(1 for s in signs if s>0)}+ {sum(1 for s in signs if s<0)}-)")
    excl = (lo > 0) or (hi < 0)
    print(f"VERDICT: {'CONFIRMED' if excl and mean_d>0 else 'NOT CONFIRMED'} "
          f"(mask {'HELPS' if mean_d>0 else 'HURTS'} factor IC, mean ΔIC={mean_d:+.5f})")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
