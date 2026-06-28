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
    sr = float(excess.mean() / sd * np.sqrt(ANN))
    # Degenerate segment guard: a near-constant return window yields an
    # absurd |Sharpe| (tiny denominator). No real strategy exceeds ~10;
    # treat such values as unreliable (NaN) so they don't poison sign checks.
    return sr if abs(sr) <= 10.0 else float("nan")


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


def _subperiod_deltas(
    rA: pd.Series, rB: pd.Series, rf_annual=0.0,
    extra_partitions: int | list[int] | None = None,
    regime_boundaries: list[pd.Timestamp] | None = None,
):
    """Per-subperiod ΔSharpe (B − A).

    Always returns halves + thirds (back-compat). Optionally also returns:
      * uniform-calendar partitions of size N (or list of Ns) via ``extra_partitions``;
      * event-defined regime buckets split at ``regime_boundaries`` (timestamps).

    Each bucket reports its (Sharpe_A, Sharpe_B, ΔSharpe, n_bars, label).
    """
    out: dict[str, list] = {}
    idx = rA.index

    def _calc_uniform(k: int) -> list[tuple]:
        bounds = np.linspace(0, len(idx), k + 1, dtype=int)
        rows = []
        for j in range(k):
            sl = slice(bounds[j], bounds[j + 1])
            if sl.stop <= sl.start:
                continue
            a = _ann_sharpe(rA.iloc[sl].to_numpy(), rf_annual)
            b = _ann_sharpe(rB.iloc[sl].to_numpy(), rf_annual)
            label = f"{idx[sl.start].date()}~{idx[sl.stop-1].date()}"
            rows.append((label, sl.stop - sl.start, a, b, b - a))
        return rows

    out["halves"] = [r[4] for r in _calc_uniform(2)]
    out["thirds"] = [r[4] for r in _calc_uniform(3)]
    out["_halves_detail"] = _calc_uniform(2)
    out["_thirds_detail"] = _calc_uniform(3)

    extras = []
    if isinstance(extra_partitions, int):
        extras = [extra_partitions]
    elif isinstance(extra_partitions, list):
        extras = extra_partitions
    for k in extras:
        out[f"{k}_buckets"] = [r[4] for r in _calc_uniform(k)]
        out[f"_{k}_buckets_detail"] = _calc_uniform(k)

    if regime_boundaries:
        rows = []
        boundaries = sorted(pd.Timestamp(b) for b in regime_boundaries)
        edges = [idx[0]] + boundaries + [idx[-1] + pd.Timedelta(days=1)]
        for j in range(len(edges) - 1):
            lo, hi = edges[j], edges[j + 1]
            mask = (idx >= lo) & (idx < hi)
            if mask.sum() < 5:
                continue
            a = _ann_sharpe(rA[mask].to_numpy(), rf_annual)
            b = _ann_sharpe(rB[mask].to_numpy(), rf_annual)
            label = f"{lo.date()}~{(hi - pd.Timedelta(days=1)).date()}"
            rows.append((label, int(mask.sum()), a, b, b - a))
        out["regime"] = [r[4] for r in rows]
        out["_regime_detail"] = rows
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
    ap.add_argument("--full-market", action="store_true",
                    help="evaluate the portfolio on the FULL training universe "
                         "(no sub-pool restriction). Use with --workers 1 first.")
    ap.add_argument("--workers", type=int, default=None,
                    help="n_workers for score precompute (1 = serial in-process, "
                         "safest against the full-universe parallel segfault).")
    ap.add_argument("--subperiods", type=str, default=None,
                    help="extra uniform-calendar subperiod counts, comma-separated "
                         "(e.g. '5,8'). Halves+thirds always reported.")
    ap.add_argument("--regime-boundaries", type=str, default=None,
                    help="event-defined regime split dates, comma-separated "
                         "YYYY-MM-DD (e.g. '2015-06-15,2016-02-01,2018-06-15,"
                         "2020-03-01,2022-01-01,2024-04-01')")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from stockpool.config import load_config
    from stockpool.fetcher import load_universe_cache
    from stockpool.industry_map import load_or_build_industry_map
    from stockpool.factors.context import set_sector_map
    from stockpool.portfolio_ab.config import load_portfolio_ab_config
    from stockpool.portfolio_ab.runner import run_portfolio_ab

    ab_cfg = load_portfolio_ab_config(args.config)
    if args.full_market:
        # full market = no sub-pool: disable ab_pool injection in per-arm cfgs
        try:
            ab_cfg = ab_cfg.model_copy(update={"use_ab_pool": False})
        except Exception:
            ab_cfg.use_ab_pool = False
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

    if args.full_market:
        portfolio_codes = None
        print(f"[full-market] portfolio universe = full training pool ({len(pool_data)} codes)")
    elif args.pool:
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
                           portfolio_pool_data=portfolio_pool_data,
                           n_workers=args.workers)

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
    extra_parts = (
        [int(x) for x in args.subperiods.split(",")]
        if args.subperiods else None
    )
    regime_bdys = (
        [pd.Timestamp(x.strip()) for x in args.regime_boundaries.split(",")]
        if args.regime_boundaries else None
    )
    subs = _subperiod_deltas(
        rA, rB, rf,
        extra_partitions=extra_parts,
        regime_boundaries=regime_bdys,
    )
    vA = _arm_validity(armA, intended_n)
    vB = _arm_validity(armB, intended_n)

    # Validity: both arms must trade, and their coverage must be comparable
    # (catches a silently-degenerate arm). Do NOT require an absolute coverage
    # floor — a top-K portfolio on the full ~4599 universe only ever touches a
    # small fraction, so an absolute floor (e.g. 0.5) is only meaningful on a
    # small sub-pool. Parity (within ~3x) is the universe-agnostic check.
    cov_min, cov_max = min(vA["coverage"], vB["coverage"]), max(vA["coverage"], vB["coverage"])
    valid = (vA["trade_count"] > 0 and vB["trade_count"] > 0
             and cov_min > 0 and (cov_min / cov_max) >= 0.33)
    sign = np.sign(dpt)
    # halves + thirds for the back-compat "subs_hold" verdict (extras are
    # reported but don't gate the verdict — they're for diagnostic colour).
    legacy_segs = subs.get("halves", []) + subs.get("thirds", [])
    subs_hold = all(np.sign(x) == sign for x in legacy_segs if np.isfinite(x)) and sign != 0
    ci_excl0 = (lo > 0) or (hi < 0)
    confirmed = (abs(dpt) >= 0.10) and ci_excl0 and subs_hold and valid and (dpt > 0)

    print(f"\n{'='*70}\nAB significance: {nameB} (B) vs {nameA} (A)  |  config={args.config}")
    print(f"common bars: {len(common)}  block: {block}  n_boot: {args.n_boot}")
    print(f"{'-'*70}")
    print(f"Sharpe A={shA:.3f}  B={shB:.3f}  ΔSharpe(B-A) point = {dpt:+.3f}")
    print(f"M2 paired-bootstrap 95% CI for ΔSharpe: [{lo:+.3f}, {hi:+.3f}]  "
          f"P(Δ<=0)={p_le0:.3f}  -> CI excludes 0: {ci_excl0}")
    print(f"M3 sub-period ΔSharpe:")
    # Detail rows for each partition: label, n_bars, SharpeA, SharpeB, ΔSharpe.
    for key in list(subs.keys()):
        if not key.startswith("_") or not key.endswith("_detail"):
            continue
        partition_name = key[1:-len("_detail")]
        rows = subs[key]
        if not rows:
            continue
        print(f"     [{partition_name}]")
        signs = [np.sign(r[4]) for r in rows if np.isfinite(r[4])]
        consistent = len(set(signs)) == 1 and signs and signs[0] != 0
        for lbl, n, a, b, d in rows:
            tag = "+" if d > 0 else "-" if d < 0 else "0"
            print(f"       {lbl}  n={n:>4}  A={a:+.3f}  B={b:+.3f}  Δ={d:+.3f}  [{tag}]")
        print(f"       → sign consistent: {consistent}  ({sum(1 for s in signs if s>0)}+ "
              f"{sum(1 for s in signs if s<0)}- of {len(signs)})")
    print(f"     legacy halves+thirds sign holds: {subs_hold}")
    print(f"M4 validity: A trades={vA['trade_count']} cov={vA['coverage']:.2f} | "
          f"B trades={vB['trade_count']} cov={vB['coverage']:.2f}  -> ok: {valid}")
    print(f"{'-'*70}")
    print(f"VERDICT: {'CONFIRMED WIN' if confirmed else 'NOT CONFIRMED'} "
          f"(|Δ|>=0.10 & CI excl 0 & sub-period sign & validity & Δ>0)")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
