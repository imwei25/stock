"""Smoke test for ensemble HTML rendering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.backtesting.framework import TradeCosts
from stockpool.config import PortfolioRunConfig
from stockpool.portfolio.engine import PortfolioEngine
from stockpool.portfolio.ensemble import StaggeredRunner
from stockpool.portfolio.report import render_ensemble_report
from stockpool.portfolio.strategy import PrecomputedScoreStrategy


def _build_ensemble(n_offsets=3):
    dates = pd.bdate_range("2024-01-02", periods=20)
    codes = ["A", "B", "C", "D"]
    panel = {c: pd.DataFrame({
        "date": dates,
        "open": np.linspace(10, 12, 20),
        "close": np.linspace(10, 12, 20),
    }) for c in codes}
    sp = pd.DataFrame(
        np.tile([4.0, 3.0, 2.0, 1.0], (20, 1)), index=dates, columns=codes,
    )
    strat = PrecomputedScoreStrategy(sp)

    def factory():
        return PortfolioEngine(
            strategy=strat,
            portfolio_cfg=PortfolioRunConfig(
                top_k=2, rebalance_n_days=3, max_per_industry=None,
            ),
            costs=TradeCosts(0.0, 0.0),
        )
    return panel, StaggeredRunner(factory).run(panel, n_offsets=n_offsets)


def test_ensemble_report_renders(tmp_path):
    panel, ens = _build_ensemble(n_offsets=3)
    out = render_ensemble_report(
        ens, panel_data=panel,
        run_date="2026-05-27", output_dir=tmp_path,
        config_hash="testhash",
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    html = out.read_text(encoding="utf-8")
    assert "ensemble" in html.lower()
    assert "k=3" in html or "k=3" in html.replace(" ", "")
    assert "Per-offset metrics" in html
    assert "Ensemble net asset value" in html
    # latest.html should be created alongside
    assert (tmp_path / "latest.html").exists()


def test_ensemble_report_empty_result(tmp_path):
    from stockpool.portfolio.ensemble import EnsembleResult
    empty = EnsembleResult(
        individual_results=[],
        ensemble_curve=pd.DataFrame({"date": [], "equity": []}),
        envelope=pd.DataFrame(columns=["date", "min", "p25", "median", "p75", "max"]),
        aggregated_metrics={},
        strategy_name="empty",
    )
    out = render_ensemble_report(
        empty, panel_data={}, run_date="2026-05-27", output_dir=tmp_path,
    )
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Empty" in html
