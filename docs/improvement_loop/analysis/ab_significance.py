"""Methodology hardening for the improvement loop (audit M2/M3/M4).

Given a portfolio-AB config, run both arms (scores are cached, so top_k/rebalance/
cap arms are fast), then judge the B-vs-A difference with noise-aware statistics
instead of a bare ΔSharpe point estimate:

  M2  paired stationary-block bootstrap 95% CI for ΔSharpe (annualized).
      Pairs (rA_t, rB_t) are resampled in blocks to preserve autocorrelation and
      the A/B pairing -> the CI reflects how much of ΔSharpe is sampling noise.
  M3  sub-period sign check: split the common dates into halves and thirds and
      report ΔSharpe in each -> guards against a one-sub-period artifact.
  M4  arm-validity guard: trade_count and traded-code coverage per arm; flags a
      silently-degenerate arm (the kind that produced A1's 0-trade first run).

A win is "confirmed" iff: point ΔSharpe >= +0.10 AND the bootstrap 95% CI excludes
0 AND the sign holds in every sub-period AND both arms pass the validity guard.

Usage:
  .venv/Scripts/python.exe docs/improvement_loop/analysis/ab_significance.py \
      --config docs/improvement_loop/configs/G1.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANN = 252.0  # trading days/yr; matches metrics.compute_metrics daily basis


def _ann_sharpe(daily_ret: np.ndarray, rf_annual: float = 0.0) -> float:
    if daily_ret.size < 2:
        return float("nan")
    excess = daily_ret - rf_annual / ANN
    sd = excess.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ANN))


def _curve_to_daily_ret(curve: pd.DataFrame) -> pd.Series:
    s = curve.set_index("date")["equity"].astype(float).sort_index()
    return s.pct_change().dropna()


def _paired_block_bootstrap(rA: np.ndarray, rB: np.ndarray, *, n_boot=5000,
                            block=None, rf_annual=0.0, seed=42):
    """Circular-block bootstrap of ΔSharpe = Sharpe(B) - Sharpe(A), paired."""
    T = rA.size
    if block is None:
        block = max(5, int(round(T ** 0.5)))  # ~ sqrt(T)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    deltas = np.empty(n_boot)
    idx_base = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = (starts[:, None] + idx_base[None, :]).ravel()[:T] % T
        dA = _ann_sharpe(rA[idx], rf_annual)
        dB = _ann_sharpe(rB[idx], rf_annual)
        deltas[b] = dB - dA
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    p_le0 = float(np.mean(deltas <= 0))  # one-sided mass at/below 0
    return float(lo), float(hi), p_le0, block


def _subperiod_deltas(rA: pd.Series, rB: pd.Series, rf_annual=0.0):
    out = {}
    for label, k in [("halves", 2), ("thirds", 3)]:
        idx = rA.index
        bounds = np.linspace(0, len(idx), k + 1, dtype=int)
        segs = []
        for j in range(k):
            sl = slice(bounds[j], bounds[j + 1])
            a = _ann_sharpe(rA.iloc[sl].to_numpy(), rf_annual)
            b = _ann_sharpe(rB.iloc[sl].to_numpy(), rf_annual)
            segs.append(b - a)
        out[label] = segs
    return out


def _arm_validity(arm, intended_n: int) -> dict:
    trades = list(arm.trades)
    codes = {t.code for t in trades if getattr(t, "code", None)}
    return {
        "trade_count": len(trades),
        "traded_codes": len(codes),
        "coverage": (len(codes) / intended_n) if intended_n else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--refresh-scores", action="store_true")
    ap.add_argument("--pool", default=None,
                    help="override portfolio universe with a pool parquet "
                         "(e.g. data/ab_pool_v2.parquet for M1 second-pool check)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from stockpool.config import load_config
    from stockpool.fetcher import load_universe_cache
    from stockpool.industry_map import load_or_build_industry_map
    from stockpool.factors.context import set_sector_map
    from stockpool.portfolio_ab.config import load_portfolio_ab_config
    from stockpool.portfolio_ab.runner import run_portfolio_ab

    ab_cfg = load_portfolio_ab_config(args.config)
    base_path = (Path(args.config).parent / ab_cfg.base_config).resolve()
    base_cfg = load_config(base_path)
    cache_dir = Path(base_cfg.data.cache_dir)

    universe_df = pd.read_parquet(cache_dir / "universe.parquet")
    name_map = dict(zip(universe_df["code"], universe_df.get("name", universe_df["code"])))
    pool_data = load_universe_cache(cache_dir, base_cfg.data.history_days,
                                    warmup_days=base_cfg.data.warmup_days)
    for s in base_cfg.stocks:
        name_map.setdefault(s.code, s.name)
    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    set_sector_map(sector_map)

    if args.pool:
        pool_df = pd.read_parquet(args.pool)
        portfolio_codes = [str(c).zfill(6) for c in pool_df["code"]]
        print(f"[pool override] {args.pool}: {len(portfolio_codes)} codes")
    else:
        portfolio_codes = base_cfg.portfolio_backtest.universe_codes
        if not portfolio_codes and ab_cfg.use_ab_pool:
            from stockpool.ab_pool import load_ab_pool
            pool_df = load_ab_pool(base_cfg.ab_pool.cache_path)
            portfolio_codes = [str(c).zfill(6) for c in pool_df["code"]]
    portfolio_pool_data = ({c: pool_data[c] for c in portfolio_codes if c in pool_data}
                           if portfolio_codes else None)
    intended_n = len(portfolio_pool_data) if portfolio_pool_data else len(pool_data)

    res = run_portfolio_ab(ab_cfg, base_cfg, pool_data=pool_data,
                           sector_map=sector_map, name_map=name_map,
                           refresh_scores=args.refresh_scores,
                           portfolio_pool_data=portfolio_pool_data)

    arms = list(res.arms.items())  # insertion order: A then B
    (nameA, armA), (nameB, armB) = arms[0], arms[1]
    rf = base_cfg.backtest.risk_free_rate

    rA = _curve_to_daily_ret(armA.primary_curve)
    rB = _curve_to_daily_ret(armB.primary_curve)
    common = rA.index.intersection(rB.index)
    rA, rB = rA.loc[common], rB.loc[common]
    aA, aB = rA.to_numpy(), rB.to_numpy()

    shA, shB = _ann_sharpe(aA, rf), _ann_sharpe(aB, rf)
    dpt = shB - shA
    lo, hi, p_le0, block = _paired_block_bootstrap(aA, aB, n_boot=args.n_boot, rf_annual=rf)
    subs = _subperiod_deltas(rA, rB, rf)
    vA = _arm_validity(armA, intended_n)
    vB = _arm_validity(armB, intended_n)

    valid = (vA["trade_count"] > 0 and vB["trade_count"] > 0
             and vA["coverage"] >= 0.5 and vB["coverage"] >= 0.5)
    sign = np.sign(dpt)
    subs_hold = all(np.sign(x) == sign for seg in subs.values() for x in seg if np.isfinite(x)) and sign != 0
    ci_excl0 = (lo > 0) or (hi < 0)
    confirmed = (abs(dpt) >= 0.10) and ci_excl0 and subs_hold and valid and (dpt > 0)

    print(f"\n{'='*70}\nAB significance: {nameB} (B) vs {nameA} (A)  |  config={args.config}")
    print(f"common bars: {len(common)}  block: {block}  n_boot: {args.n_boot}")
    print(f"{'-'*70}")
    print(f"Sharpe A={shA:.3f}  B={shB:.3f}  ΔSharpe(B-A) point = {dpt:+.3f}")
    print(f"M2 paired-bootstrap 95% CI for ΔSharpe: [{lo:+.3f}, {hi:+.3f}]  "
          f"P(Δ<=0)={p_le0:.3f}  -> CI excludes 0: {ci_excl0}")
    print(f"M3 sub-period ΔSharpe:")
    for label, segs in subs.items():
        print(f"     {label}: " + "  ".join(f"{x:+.3f}" for x in segs))
    print(f"     sign holds in all sub-periods: {subs_hold}")
    print(f"M4 validity: A trades={vA['trade_count']} cov={vA['coverage']:.2f} | "
          f"B trades={vB['trade_count']} cov={vB['coverage']:.2f}  -> ok: {valid}")
    print(f"{'-'*70}")
    print(f"VERDICT: {'CONFIRMED WIN' if confirmed else 'NOT CONFIRMED'} "
          f"(|Δ|>=0.10 & CI excl 0 & sub-period sign & validity & Δ>0)")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
