"""Direct Layer-B (daily IC) significance — bypasses the portfolio_ab runner
which deadlocks post-precompute at 15-yr scale.

Pipeline:
  1. Load the factor panel (cached) for each arm of an AB config.
  2. Build each arm's MLFactorStrategy via build_strategy.
  3. Call precompute_scores_from_legacy directly (this part works fine).
  4. Save score panels to disk + compute daily cross-sectional IC.
  5. Paired bootstrap CI on ΔIC; report sub-periods + regime buckets.

Usage:
  .venv/Scripts/python.exe docs/improvement_loop/analysis/layer_b_direct.py \\
      --config docs/improvement_loop/configs/D3b_sharpe_full.yaml \\
      --pool data/top1000_liquid.parquet \\
      --workers 6 \\
      --regime-boundaries 2015-06-15,2016-02-01,2018-06-15,2020-03-01,2022-01-01,2024-04-12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _per_day_ic(score: pd.DataFrame, fwd: pd.DataFrame, min_n: int = 10) -> pd.Series:
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
        out[t] = float(
            ((s_r - s_r.mean()) * (f_r - f_r.mean())).mean()
            / (s_r.std(ddof=0) * f_r.std(ddof=0))
        )
    return pd.Series(out, name="ic").sort_index()


def _block_bootstrap_mean_ci(x: np.ndarray, *, n_boot=5000, seed=42):
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
    ap.add_argument("--pool", default=None, help="parquet of codes to score (else full training pool)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--regime-boundaries", type=str, default=None)
    ap.add_argument("--subperiods", type=str, default="5,8")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from stockpool.config import load_config
    from stockpool.fetcher import load_universe_cache
    from stockpool.industry_map import load_or_build_industry_map
    from stockpool.factors.context import set_sector_map
    from stockpool.portfolio.scoring import (
        precompute_scores_from_legacy, score_cache_key,
    )
    from stockpool.portfolio_ab.config import (
        load_portfolio_ab_config, build_effective_cfg,
    )
    from stockpool.strategy_factory import build_strategy, load_or_build_factor_panel

    cfg_path = Path(args.config)
    ab_cfg = load_portfolio_ab_config(cfg_path)
    base_cfg = load_config(cfg_path.parent / ab_cfg.base_config)
    cache_dir = Path(base_cfg.data.cache_dir)

    print(f"[setup] loading universe ({base_cfg.data.history_days} hist + "
          f"{base_cfg.data.warmup_days} warmup)…")
    pool_data = load_universe_cache(
        cache_dir, base_cfg.data.history_days,
        warmup_days=base_cfg.data.warmup_days,
    )
    print(f"[setup] training pool: {len(pool_data)} codes")

    if args.pool:
        pool_df = pd.read_parquet(args.pool)
        portfolio_codes = [str(c).zfill(6) for c in pool_df["code"]]
        portfolio_pool_data = {c: pool_data[c] for c in portfolio_codes if c in pool_data}
        print(f"[setup] portfolio universe (--pool): {len(portfolio_pool_data)} codes")
    else:
        portfolio_pool_data = pool_data
        portfolio_codes = sorted(pool_data.keys())

    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    set_sector_map(sector_map)

    arms = list(ab_cfg.arms.items())
    if len(arms) != 2:
        raise SystemExit(f"need exactly 2 arms, got {len(arms)}")
    panels: dict[str, pd.DataFrame] = {}
    score_dir = Path(base_cfg.portfolio_backtest.score_cache_dir)
    score_dir.mkdir(parents=True, exist_ok=True)

    for arm_name, arm in arms:
        eff = build_effective_cfg(base_cfg, arm)
        cache_key = score_cache_key(eff, portfolio_pool_data.keys())
        score_path = score_dir / f"{cache_key}.parquet"
        if score_path.exists():
            print(f"[{arm_name}] cache hit: {score_path.name}")
            panels[arm_name] = pd.read_parquet(score_path)
            continue

        print(f"[{arm_name}] building factor panel…")
        factor_panel = close_panel = None
        if (eff.strategy.name == "ml_factor"
                and eff.strategy.ml_factor.panel_mode == "pooled"):
            factor_panel, close_panel = load_or_build_factor_panel(
                eff.strategy.ml_factor.factors, pool_data,
                eff.data.cache_dir,
                preprocess_cfg=eff.strategy.ml_factor.preprocess,
            )

        print(f"[{arm_name}] building strategy…")
        legacy = build_strategy(
            eff, pool_data=pool_data,
            factor_panel=factor_panel, close_panel=close_panel,
            shared_cache={},
        )
        print(f"[{arm_name}] precomputing scores ({len(portfolio_pool_data)} stocks, "
              f"workers={args.workers})…")
        sp = precompute_scores_from_legacy(
            legacy, portfolio_pool_data, n_workers=args.workers,
        )
        print(f"[{arm_name}] saving score panel to {score_path.name} (shape={sp.shape})…")
        sp.to_parquet(score_path)
        panels[arm_name] = sp
        print(f"[{arm_name}] done.")

    print(f"\n[layer-B] building forward-return panel (horizon={args.horizon})…")
    close = pd.DataFrame({c: pool_data[c].set_index("date")["close"]
                          for c in portfolio_codes if c in pool_data})
    fwd_ret = close.shift(-args.horizon) / close - 1.0

    nameA, nameB = arms[0][0], arms[1][0]
    icA = _per_day_ic(panels[nameA], fwd_ret)
    icB = _per_day_ic(panels[nameB], fwd_ret)
    print(f"[{nameA}] IC obs={len(icA)} mean={icA.mean():+.5f} std={icA.std(ddof=1):.5f}")
    print(f"[{nameB}] IC obs={len(icB)} mean={icB.mean():+.5f} std={icB.std(ddof=1):.5f}")

    common = icA.index.intersection(icB.index)
    icA, icB = icA.loc[common], icB.loc[common]
    delta = (icB - icA).to_numpy()
    T = delta.size
    mean_d, std_d = float(delta.mean()), float(delta.std(ddof=1))
    from math import erf, sqrt
    t_stat = mean_d / (std_d / np.sqrt(T)) if std_d > 0 else float("nan")
    p_two = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
    lo, hi, block = _block_bootstrap_mean_ci(delta, n_boot=args.n_boot)

    print(f"\n{'='*70}")
    print(f"Layer-B significance: {nameB} (B) vs {nameA} (A)")
    print(f"common IC observations: T = {T}    bootstrap block: {block}    n_boot: {args.n_boot}")
    print(f"{'-'*70}")
    print(f"IC mean   A = {icA.mean():+.5f}   B = {icB.mean():+.5f}   ΔIC mean = {mean_d:+.5f}")
    print(f"IR proxy  A = {icA.mean()/icA.std(ddof=1):+.3f}  "
          f"B = {icB.mean()/icB.std(ddof=1):+.3f}")
    print(f"paired t  = {t_stat:+.3f}  (iid p ≈ {p_two:.4f})")
    print(f"block-bootstrap 95% CI for mean(ΔIC): [{lo:+.5f}, {hi:+.5f}]  → excludes 0: "
          f"{(lo>0) or (hi<0)}")

    # Sub-periods
    common_idx = icA.index
    print(f"\nsub-periods:")
    extras = [int(x) for x in args.subperiods.split(",")] if args.subperiods else []
    for k in [2, 3, *extras]:
        bounds = np.linspace(0, T, k + 1, dtype=int)
        rows = []
        for j in range(k):
            sl = slice(bounds[j], bounds[j + 1])
            if sl.stop <= sl.start:
                continue
            sub_delta = delta[sl]
            sub_mean = sub_delta.mean()
            sub_t = sub_mean / (sub_delta.std(ddof=1) / np.sqrt(len(sub_delta))) if len(sub_delta) > 1 else float("nan")
            rows.append((common_idx[sl.start].date(), common_idx[sl.stop-1].date(),
                         len(sub_delta), sub_mean, sub_t))
        signs = [np.sign(r[3]) for r in rows]
        consistent = len(set(signs)) == 1 and signs[0] != 0
        label = {2: "halves", 3: "thirds"}.get(k, f"{k}_buckets")
        print(f"  [{label}]")
        for s, e, n, m, t in rows:
            tag = "+" if m > 0 else "-" if m < 0 else "0"
            print(f"    {s}~{e}  n={n:>4}  ΔIC={m:+.5f}  t={t:+.2f}  [{tag}]")
        print(f"    → consistent: {consistent}  "
              f"({sum(1 for s in signs if s>0)}+ {sum(1 for s in signs if s<0)}- of {len(signs)})")

    if args.regime_boundaries:
        boundaries = sorted(pd.Timestamp(x.strip()) for x in args.regime_boundaries.split(","))
        edges = [common_idx[0]] + boundaries + [common_idx[-1] + pd.Timedelta(days=1)]
        rows = []
        for j in range(len(edges) - 1):
            lo_d, hi_d = edges[j], edges[j + 1]
            mask = (common_idx >= lo_d) & (common_idx < hi_d)
            if mask.sum() < 5:
                continue
            sub_delta = delta[mask]
            sub_mean = sub_delta.mean()
            sub_t = sub_mean / (sub_delta.std(ddof=1) / np.sqrt(len(sub_delta))) if len(sub_delta) > 1 else float("nan")
            rows.append((lo_d.date(), (hi_d - pd.Timedelta(days=1)).date(),
                         int(mask.sum()), sub_mean, sub_t))
        signs = [np.sign(r[3]) for r in rows]
        consistent = len(set(signs)) == 1 and signs and signs[0] != 0
        print(f"  [regime-defined]")
        for s, e, n, m, t in rows:
            tag = "+" if m > 0 else "-" if m < 0 else "0"
            print(f"    {s}~{e}  n={n:>4}  ΔIC={m:+.5f}  t={t:+.2f}  [{tag}]")
        print(f"    → consistent: {consistent}  "
              f"({sum(1 for s in signs if s>0)}+ {sum(1 for s in signs if s<0)}- of {len(signs)})")

    print(f"{'-'*70}")
    ci_excl = (lo > 0) or (hi < 0)
    print(f"VERDICT: {'CONFIRMED Layer B' if ci_excl and mean_d > 0 else 'NOT CONFIRMED Layer B'}")
    print(f"  (mean ΔIC = {mean_d:+.5f}, 95% CI {'excludes' if ci_excl else 'includes'} 0)")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
