"""Minimal example 校验:Staggered serial vs parallel 在 rtol=1e-12 容差内一致.

Usage:
    .venv/Scripts/python.exe scripts/verify_staggered_parallel.py

跑完看到 'OK: serial == parallel (rtol=1e-12, atol=0)' 即通过.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.backtesting.framework import TradeCosts
from stockpool.config import PortfolioRunConfig
from stockpool.portfolio.engine import PortfolioEngine
from stockpool.portfolio.ensemble import StaggeredRunner
from stockpool.portfolio.strategy import PrecomputedScoreStrategy


def _make_panel(n_codes=8, n_bars=60, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_bars, freq="B")
    panel = {}
    for i in range(n_codes):
        code = f"S{i:03d}"
        prices = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_bars))
        panel[code] = pd.DataFrame({
            "date": dates,
            "open": prices * (1 + rng.normal(0, 0.001, n_bars)),
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": rng.integers(1e5, 5e6, n_bars),
        })
    return panel, dates


def main():
    panel, dates = _make_panel()
    codes = list(panel.keys())
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        rng.standard_normal((len(dates), len(codes))),
        index=dates, columns=codes,
    )
    strategy = PrecomputedScoreStrategy(scores, name="test")
    portfolio_cfg = PortfolioRunConfig(top_k=3, rebalance_n_days=5)
    costs = TradeCosts(buy_cost=0.001, sell_cost=0.001)
    components = (strategy, portfolio_cfg, costs, 0.02, None, {})

    def _factory():
        return PortfolioEngine(
            strategy=strategy, portfolio_cfg=portfolio_cfg, costs=costs,
            risk_free_rate=0.02, eligibility=None, sector_map={},
        )

    runner = StaggeredRunner(_factory, components=components, risk_free_rate=0.02)
    ens_serial = runner.run(panel, n_offsets=3, parallel=False)
    ens_par = runner.run(panel, n_offsets=3, parallel=True)

    np.testing.assert_allclose(
        ens_serial.ensemble_curve["equity"].values,
        ens_par.ensemble_curve["equity"].values,
        rtol=1e-12, atol=0,
    )
    print(
        f"OK: serial == parallel (rtol=1e-12, atol=0); "
        f"n_bars={len(dates)}, n_codes={len(codes)}"
    )


if __name__ == "__main__":
    main()
