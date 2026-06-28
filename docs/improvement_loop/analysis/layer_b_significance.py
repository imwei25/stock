"""Layer B (daily cross-sectional IC) significance: paired comparison of two
weighter score panels against the same forward-return labels.

Why this layer:
  Layer D (portfolio Sharpe) bootstrap on ~763 daily P&L observations has
  effective N ≈ √T ≈ 27 (autocorr-aware block bootstrap), so CIs around
  ΔSharpe stay at the ±0.3 scale. Layer B uses one IC observation per
  cross-section per day → effective N ≈ T = 763, lifting power by ~√(763/27)
  ≈ 5.3×. A weighter improvement that's invisible at Layer D's noise floor
  can still show up at Layer B if the cross-sectional ranking itself is
  meaningfully different.

Pipeline:
  1. Load two precomputed score panels (T × N) keyed by the AB config.
  2. Build the forward-return label panel (close[t+h]/close[t] − 1).
  3. For every overlap date t, IC^A_t = SpearmanRho(score_A[t,:], fwdret[t,:]);
     same for IC^B_t.
  4. Pair (IC^A_t, IC^B_t) across all valid t, run paired t-test on
     ΔIC = IC^B − IC^A. Report mean, std, t, p, and a stationary-block
     bootstrap 95% CI on mean(ΔIC).

Usage:
  .venv/Scripts/python.exe docs/improvement_loop/analysis/layer_b_significance.py \\
      --config docs/improvement_loop/configs/D3b_sharpe_full.yaml --horizon 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _per_day_ic(score: pd.DataFrame, fwd: pd.DataFrame, min_n: int = 10) -> pd.Series:
    """Per-date Spearman IC between score and forward returns.

    NaN-safe: each date uses the row's joint non-NaN columns; days with fewer
    than ``min_n`` valid stocks are dropped.
    """
    aligned_idx = score.index.intersection(fwd.index)
    aligned_cols = score.columns.intersection(fwd.columns)
    score = score.loc[aligned_idx, aligned_cols]
    fwd = fwd.loc[aligned_idx, aligned_cols]

    out = {}
    for t in aligned_idx:
        s = score.loc[t]
        f = fwd.loc[t]
        mask = s.notna() & f.notna()
        if mask.sum() < min_n:
            continue
        s_r = s[mask].rank()
        f_r = f[mask].rank()
        if s_r.std(ddof=0) < 1e-12 or f_r.std(ddof=0) < 1e-12:
            continue
        ic = float(((s_r - s_r.mean()) * (f_r - f_r.mean())).mean() / (s_r.std(ddof=0) * f_r.std(ddof=0)))
        out[t] = ic
    return pd.Series(out, name="ic").sort_index()


def _block_bootstrap_mean_ci(x: np.ndarray, *, n_boot=5000, seed=42):
    """Stationary block bootstrap 95% CI on mean(x), block size ≈ √T."""
    T = x.size
    block = max(5, int(round(T ** 0.5)))
    n_blocks = int(np.ceil(T / block))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    idx_base = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = (starts[:, None] + idx_base[None, :]).ravel()[:T] % T
        means[b] = x[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), int(block)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--horizon", type=int, required=True,
                    help="forward-return horizon in bars (matches strategy.ml_factor.horizon)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--min-stocks-per-day", type=int, default=10)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from stockpool.config import load_config
    from stockpool.fetcher import load_universe_cache
    from stockpool.portfolio.scoring import score_cache_key
    from stockpool.portfolio_ab.config import load_portfolio_ab_config, build_effective_cfg

    cfg_path = Path(args.config)
    ab_cfg = load_portfolio_ab_config(cfg_path)
    base_cfg = load_config(cfg_path.parent / ab_cfg.base_config)
    cache_dir = Path(base_cfg.data.cache_dir)

    print(f"[layer-B] loading universe ({base_cfg.data.history_days} hist + "
          f"{base_cfg.data.warmup_days} warmup)…")
    pool_data = load_universe_cache(
        cache_dir, base_cfg.data.history_days,
        warmup_days=base_cfg.data.warmup_days,
    )
    universe_codes = sorted(pool_data.keys())

    # Two arms → two score panels.
    arms = list(ab_cfg.arms.items())
    if len(arms) != 2:
        raise SystemExit(f"need exactly 2 arms, got {len(arms)}")
    panels: dict[str, pd.DataFrame] = {}
    for arm_name, arm in arms:
        eff = build_effective_cfg(base_cfg, arm)
        key = score_cache_key(eff, universe_codes)
        path = cache_dir / "portfolio_scores" / f"{key}.parquet"
        if not path.exists():
            raise SystemExit(f"score panel missing: {path}\n"
                             f"  arm: {arm_name}\n"
                             f"  run portfolio-ab first to materialise it.")
        panels[arm_name] = pd.read_parquet(path)
        print(f"[layer-B] {arm_name}: {path.name} {panels[arm_name].shape}")

    # Build forward-return label panel from close.
    print(f"[layer-B] building forward-return panel (horizon={args.horizon})…")
    close = pd.DataFrame({c: pool_data[c].set_index("date")["close"] for c in universe_codes})
    fwd_ret = close.shift(-args.horizon) / close - 1.0

    # Layer B: per-day IC for each arm.
    ic_series = {}
    for name, sp in panels.items():
        ic = _per_day_ic(sp, fwd_ret, min_n=args.min_stocks_per_day)
        ic_series[name] = ic
        print(f"[layer-B] {name}: IC observations = {len(ic)}, mean = {ic.mean():+.5f}, "
              f"std = {ic.std(ddof=1):.5f}")

    nameA, nameB = arms[0][0], arms[1][0]
    icA, icB = ic_series[nameA], ic_series[nameB]
    common = icA.index.intersection(icB.index)
    icA, icB = icA.loc[common], icB.loc[common]
    delta = (icB - icA).to_numpy()
    T = delta.size

    # Paired t-test on ΔIC = IC^B − IC^A.
    mean_d = float(delta.mean())
    std_d = float(delta.std(ddof=1))
    # Standard t-stat ignores autocorrelation in IC series; block bootstrap
    # below gives the autocorr-aware CI which is what to actually trust.
    t_stat = mean_d / (std_d / np.sqrt(T)) if std_d > 0 else float("nan")
    # Two-sided p approximated via large-N normal (T ≈ 760 → t ≈ z).
    from math import erf, sqrt
    p_two = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))

    lo, hi, block = _block_bootstrap_mean_ci(delta, n_boot=args.n_boot)

    print(f"\n{'='*70}")
    print(f"Layer-B significance: {nameB} (B) vs {nameA} (A)  |  config={args.config}")
    print(f"common IC observations: T = {T}    bootstrap block size: {block}    n_boot: {args.n_boot}")
    print(f"{'-'*70}")
    print(f"IC mean   A = {icA.mean():+.5f}   B = {icB.mean():+.5f}   ΔIC mean = {mean_d:+.5f}")
    print(f"IC std    A = {icA.std(ddof=1):.5f}   B = {icB.std(ddof=1):.5f}   ΔIC std  = {std_d:.5f}")
    print(f"IR proxy  A = {icA.mean()/icA.std(ddof=1):+.3f}  "
          f"B = {icB.mean()/icB.std(ddof=1):+.3f}  "
          f"ΔIR_per_sqrtT = {mean_d/std_d*np.sqrt(T) if std_d>0 else float('nan'):+.3f}")
    print(f"paired t  = {t_stat:+.3f}  (approx two-sided p = {p_two:.4f})  [iid assumption]")
    print(f"block-bootstrap 95% CI for mean(ΔIC): [{lo:+.5f}, {hi:+.5f}]  -> excludes 0: "
          f"{(lo>0) or (hi<0)}")

    # Sub-period sign for parallelism with ab_significance.py.
    for label, k in [("halves", 2), ("thirds", 3)]:
        bounds = np.linspace(0, T, k + 1, dtype=int)
        sub_means = [float(delta[bounds[j]:bounds[j+1]].mean()) for j in range(k)]
        signs = [np.sign(x) for x in sub_means]
        print(f"sub {label}: " + "  ".join(f"{x:+.5f}" for x in sub_means)
              + f"   sign consistent: {len(set(signs)) == 1 and signs[0] != 0}")

    # Verdict mirrors ab_significance.py: practical-significant ΔIC + CI excludes 0.
    # ΔIC magnitude is small in absolute terms; standard Barra threshold is
    # |ΔIC| ≥ 0.005 for "economically meaningful". We don't enforce that here
    # — just report.
    ci_excl = (lo > 0) or (hi < 0)
    print(f"{'-'*70}")
    print(f"VERDICT: {'CONFIRMED at Layer B' if ci_excl and mean_d > 0 else 'NOT CONFIRMED at Layer B'}")
    print(f"        (mean ΔIC = {mean_d:+.5f}, 95% CI {'excludes' if ci_excl else 'includes'} 0)")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
