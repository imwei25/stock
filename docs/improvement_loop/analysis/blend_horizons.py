"""Direction-1 (2026-07-18 audit wave): multi-horizon score blend Layer B.

Hypothesis: h3 and h10 score panels carry partially independent
cross-sectional information (kill-check: per-day rank corr ~0.84, not ~1);
a per-day z-scored convex blend 0.5*z(s3)+0.5*z(s10) beats h3 alone on
daily rank-IC at fwd=3, and is at least at parity at fwd=10.

Pre-registered: w=0.5 is THE hypothesis. w in {0.3, 0.7} reported as
sensitivity only (never selected post-hoc).

Uses the corrected Layer B oracle (open-basis fwd returns, entry
limit-up tradability mask, as-of top-1000 PIT membership) and the h3/h10
score panels cached by RV-h10 (v3 keys) on rv_eval + pool2.

Run:
  .venv/Scripts/python.exe docs/improvement_loop/analysis/blend_horizons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import layer_b_direct as lb  # _per_day_ic, _block_bootstrap_mean_ci

ASOF_TOP_N = 1000
BLEND_W = 0.5            # pre-registered
SENS_WS = (0.3, 0.7)     # sensitivity report only
POOLS = {
    "rv_eval": ROOT / "data/rv_eval_pool.parquet",
    "pool2": ROOT / "data/pool2_midliq.parquet",
}


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def _fwd_ret_open_basis(opn: pd.DataFrame, horizon: int) -> pd.DataFrame:
    entry = opn.shift(-1)
    exit_ = opn.shift(-(horizon + 1))
    return exit_ / entry - 1.0


def _report(tag: str, ic_a: pd.Series, ic_b: pd.Series) -> None:
    common = ic_a.index.intersection(ic_b.index)
    a, b = ic_a.loc[common], ic_b.loc[common]
    delta = (b - a).dropna()
    if delta.empty:
        print(f"  [{tag}] no common obs")
        return
    x = delta.to_numpy()
    lo, hi, _block = lb._block_bootstrap_mean_ci(x)
    mean = float(x.mean())
    t = mean / (x.std(ddof=1) / np.sqrt(len(x))) if x.std(ddof=1) > 0 else np.nan
    halves = [x[: len(x) // 2].mean(), x[len(x) // 2:].mean()]
    third = len(x) // 3
    thirds = [x[:third].mean(), x[third:2 * third].mean(), x[2 * third:].mean()]
    sign_ok = (all(v > 0 for v in halves) and all(v > 0 for v in thirds)) or \
              (all(v < 0 for v in halves) and all(v < 0 for v in thirds))
    print(f"  [{tag}] IC A={a.mean():+.5f} B={b.mean():+.5f} "
          f"ΔIC={mean:+.5f} t={t:+.2f} CI95=[{lo:+.5f},{hi:+.5f}] "
          f"excl0={'Y' if lo > 0 or hi < 0 else 'N'} "
          f"halves={['%+.4f' % v for v in halves]} "
          f"thirds={['%+.4f' % v for v in thirds]} sign_consistent={sign_ok}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from stockpool.config import load_config
    from stockpool.fetcher import load_universe_cache
    from stockpool.portfolio.scoring import score_cache_key
    from stockpool.portfolio_ab.config import (
        load_portfolio_ab_config, build_effective_cfg,
    )
    from stockpool.backtesting.limits import _REL_TOL, infer_limit_pct

    cfg_path = ROOT / "docs/improvement_loop/configs/RV_h10d.yaml"
    ab_cfg = load_portfolio_ab_config(cfg_path)
    base_cfg = load_config(cfg_path.parent / ab_cfg.base_config)
    score_dir = ROOT / base_cfg.portfolio_backtest.score_cache_dir

    print("[setup] loading universe cache…", flush=True)
    pool_data = load_universe_cache(
        Path(base_cfg.data.cache_dir), base_cfg.data.history_days,
        warmup_days=base_cfg.data.warmup_days,
    )
    print(f"[setup] {len(pool_data)} codes loaded", flush=True)

    # As-of top-N membership over the FULL universe (same as layer_b_direct).
    amt = {}
    for c, df in pool_data.items():
        if "volume" in df.columns and "close" in df.columns:
            s = df.set_index("date")
            amt[c] = (s["close"] * s["volume"] * 100.0).astype(float)
    avg20 = pd.DataFrame(amt).sort_index().rolling(20, min_periods=1).mean()
    member_full = avg20.rank(axis=1, ascending=False, method="first") <= ASOF_TOP_N
    del amt, avg20

    for pool_name, pool_path in POOLS.items():
        print(f"\n===== pool: {pool_name} =====", flush=True)
        pool_df = pd.read_parquet(pool_path)
        codes = [str(c).zfill(6) for c in pool_df["code"] if str(c).zfill(6) in pool_data]
        ppd_keys = codes

        panels = {}
        for arm_name, arm in ab_cfg.arms.items():
            eff = build_effective_cfg(base_cfg, arm)
            key = score_cache_key(eff, ppd_keys)
            p = score_dir / f"{key}.parquet"
            if not p.exists():
                print(f"  [{arm_name}] MISSING cache {key} — skip pool")
                panels = None
                break
            panels[arm_name] = pd.read_parquet(p)
        if not panels:
            continue
        s3, s10 = panels["h3"], panels["h10"]

        idx = s3.index.intersection(s10.index)
        cols = s3.columns.intersection(s10.columns)
        z3 = _zscore_rows(s3.loc[idx, cols])
        z10 = _zscore_rows(s10.loc[idx, cols])

        opn = pd.DataFrame({
            c: pool_data[c].set_index("date")["open"].astype(float)
            if "open" in pool_data[c].columns
            else pool_data[c].set_index("date")["close"].astype(float)
            for c in cols
        }).sort_index()
        close = pd.DataFrame({
            c: pool_data[c].set_index("date")["close"].astype(float)
            for c in cols
        }).sort_index()

        limit_pct = pd.Series({c: infer_limit_pct(c) for c in cols})
        limit_px = close.mul((1.0 + limit_pct) * (1.0 - _REL_TOL), axis=1)
        blocked = opn.shift(-1) >= limit_px
        member = member_full.reindex(index=opn.index, columns=cols).fillna(False)

        for horizon in (3, 10):
            fwd = _fwd_ret_open_basis(opn, horizon).mask(blocked).where(member)
            ic3 = lb._per_day_ic(z3, fwd)
            blend = BLEND_W * z3 + (1 - BLEND_W) * z10
            icb = lb._per_day_ic(blend, fwd)
            print(f"\n fwd={horizon} — blend(w={BLEND_W}) vs h3 (full period):")
            _report("full", ic3, icb)
            lo_ts = pd.Timestamp("2021-01-01")
            _report("2021+", ic3[ic3.index >= lo_ts], icb[icb.index >= lo_ts])
            for w in SENS_WS:
                bl = w * z3 + (1 - w) * z10
                _report(f"sens w={w}", ic3, lb._per_day_ic(bl, fwd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
