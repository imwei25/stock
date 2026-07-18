"""Direction-1 Layer D: portfolio-Sharpe AB of h3 vs 0.5*z(h3)+0.5*z(h10).

Layer B CONFIRMED cross-pool (see blend_horizons.py output, WORKLOG
AUDIT-2026-07-18). This driver feeds the blended panel straight into
PortfolioEngine via PrecomputedScoreStrategy (no src changes needed) and
applies the M2/M3/M4 significance battery from ab_significance.py.

Engine config = config_rv_engine.yaml (hold_survivors + limit_guard +
top_n_liquidity 1000), corrected engine (last_mark fix).

Run:
  .venv/Scripts/python.exe docs/improvement_loop/analysis/blend_layer_d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_significance as sig  # _ann_sharpe, _curve_to_daily_ret, _paired_block_bootstrap, _subperiod_deltas

BLEND_W = 0.5
POOLS = {
    "rv_eval": ROOT / "data/rv_eval_pool.parquet",
    "pool2": ROOT / "data/pool2_midliq.parquet",
}


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from stockpool.config import load_config
    from stockpool.fetcher import load_universe_cache
    from stockpool.portfolio.scoring import score_cache_key
    from stockpool.portfolio_ab.config import (
        load_portfolio_ab_config, build_effective_cfg,
    )
    from stockpool.portfolio.strategy import PrecomputedScoreStrategy
    from stockpool.portfolio.eligibility import EligibilityFilter
    from stockpool.portfolio.engine import PortfolioEngine
    from stockpool.backtesting.framework import TradeCosts
    from stockpool.industry_map import load_or_build_industry_map

    cfg_path = ROOT / "docs/improvement_loop/configs/RV_h10d.yaml"
    ab_cfg = load_portfolio_ab_config(cfg_path)
    base_cfg = load_config(cfg_path.parent / ab_cfg.base_config)
    score_dir = ROOT / base_cfg.portfolio_backtest.score_cache_dir

    print("[setup] loading universe cache…", flush=True)
    pool_data = load_universe_cache(
        Path(base_cfg.data.cache_dir), base_cfg.data.history_days,
        warmup_days=base_cfg.data.warmup_days,
    )
    uni = pd.read_parquet(ROOT / "data/universe.parquet")
    name_map = {str(c).zfill(6): n for c, n in zip(uni["code"], uni["name"])}
    sector_map = load_or_build_industry_map(
        Path(base_cfg.data.cache_dir), source="auto")

    for pool_name, pool_path in POOLS.items():
        print(f"\n===== pool: {pool_name} =====", flush=True)
        pool_df = pd.read_parquet(pool_path)
        codes = [str(c).zfill(6) for c in pool_df["code"]
                 if str(c).zfill(6) in pool_data]
        ppd = {c: pool_data[c] for c in codes}

        panels = {}
        for arm_name, arm in ab_cfg.arms.items():
            eff = build_effective_cfg(base_cfg, arm)
            key = score_cache_key(eff, ppd.keys())
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
        blend = BLEND_W * z3 + (1 - BLEND_W) * _zscore_rows(s10.loc[idx, cols])

        eff = build_effective_cfg(base_cfg, ab_cfg.arms["h3"])
        costs = TradeCosts(buy_cost=eff.backtest.costs.buy_cost,
                           sell_cost=eff.backtest.costs.sell_cost)
        curves = {}
        for label, panel in [("h3", s3), ("blend", blend)]:
            elig = EligibilityFilter(
                eff.portfolio_backtest.eligibility, name_map=name_map)
            eng = PortfolioEngine(
                strategy=PrecomputedScoreStrategy(panel, name=label),
                portfolio_cfg=eff.portfolio_backtest.portfolio,
                costs=costs,
                risk_free_rate=eff.backtest.risk_free_rate,
                eligibility=elig,
                sector_map=sector_map,
            )
            res = eng.run(ppd, start_offset=0)
            curves[label] = res.curve
            print(f"  [{label}] sharpe={res.metrics.get('sharpe'):.4f} "
                  f"total_ret={res.metrics.get('total_return'):.4f} "
                  f"maxDD={res.metrics.get('max_drawdown'):.4f} "
                  f"trades={len(res.trades)}", flush=True)

        rA = sig._curve_to_daily_ret(curves["h3"])
        rB = sig._curve_to_daily_ret(curves["blend"])
        common = rA.index.intersection(rB.index)
        a, b = rA.loc[common].to_numpy(), rB.loc[common].to_numpy()
        shA, shB = sig._ann_sharpe(a), sig._ann_sharpe(b)
        point = shB - shA
        lo, hi, p_le0, _block = sig._paired_block_bootstrap(a, b)
        print(f"  ΔSharpe(blend-h3) = {point:+.4f}  "
              f"CI95=[{lo:+.4f},{hi:+.4f}] P(Δ<=0)={p_le0:.3f} "
              f"excl0={'Y' if lo > 0 or hi < 0 else 'N'}")
        dates = pd.DatetimeIndex(common)
        for scheme in (2, 3):
            n = len(a)
            step = n // scheme
            deltas = []
            for i in range(scheme):
                s, e = i * step, (n if i == scheme - 1 else (i + 1) * step)
                deltas.append(sig._ann_sharpe(b[s:e]) - sig._ann_sharpe(a[s:e]))
            signs = ["+" if d > 0 else "-" for d in deltas]
            print(f"  subperiods x{scheme}: "
                  f"{['%+.3f' % d for d in deltas]} signs={signs} "
                  f"consistent={len(set(signs)) == 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
