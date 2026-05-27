"""Smoke tests for portfolio_ab.report rendering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.portfolio.result import PortfolioBacktestResult, PortfolioTrade
from stockpool.portfolio_ab.report import render_portfolio_ab_report
from stockpool.portfolio_ab.runner import ABResult, ArmResult


def _mk_arm(name, last_equity=1.05, trades=None):
    dates = pd.bdate_range("2024-01-02", periods=10)
    curve = pd.DataFrame({
        "date": dates,
        "equity": np.linspace(1.0, last_equity, 10),
        "num_positions": [2] * 10,
        "cash_ratio": [0.0] * 10,
    })
    metrics = {
        "total_return": last_equity - 1.0,
        "annualized_return": 0.1,
        "sharpe": 1.2,
        "max_drawdown": 0.05,
        "trade_count": len(trades or []),
        "win_rate": 0.6,
    }
    res = PortfolioBacktestResult(
        curve=curve,
        trades=list(trades or []),
        rebalance_log=pd.DataFrame(columns=["date", "target_codes", "num_target"]),
        metrics=metrics,
        strategy_name="stub",
    )
    return ArmResult(name=name, effective_cfg=None, single=res)


def _trade(code, ret, weight=0.5):
    return PortfolioTrade(
        code=code,
        entry_date=pd.Timestamp("2024-01-05"),
        exit_date=pd.Timestamp("2024-01-10"),
        entry_price=10.0, exit_price=10.5,
        weight_at_entry=weight,
        ret=ret,
        days_held=3,
        exit_reason="rebalance_drop",
    )


def test_render_happy(tmp_path):
    arm_a = _mk_arm("arm_a", 1.10, trades=[_trade("000001", 0.05), _trade("000002", -0.02)])
    arm_b = _mk_arm("arm_b", 0.95, trades=[_trade("000001", 0.03), _trade("000003", -0.04)])
    result = ABResult(arms={"arm_a": arm_a, "arm_b": arm_b})
    out = render_portfolio_ab_report(result, run_date="2026-05-27", output_dir=tmp_path)
    assert out.exists()
    assert out.stat().st_size > 1024
    html = out.read_text(encoding="utf-8")
    assert "Portfolio A/B" in html
    assert "arm_a" in html and "arm_b" in html
    assert "Aggregated metrics" in html
    assert "Per-stock contribution" in html
    assert "Traded-code set analysis" in html
    # Set analysis: 000001 in both, 000002 only A, 000003 only B
    assert (tmp_path / "latest.html").exists()


def test_render_with_failed_arm(tmp_path):
    arm_a = _mk_arm("good", trades=[_trade("000001", 0.05)])
    arm_b = ArmResult(name="failed", effective_cfg=None, failed=True, error="boom")
    result = ABResult(arms={"good": arm_a, "failed": arm_b})
    out = render_portfolio_ab_report(result, run_date="2026-05-27", output_dir=tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "FAILED" in html
    assert "boom" in html


def test_render_invalid_arms_count(tmp_path):
    result = ABResult(arms={})
    out = render_portfolio_ab_report(result, run_date="2026-05-27", output_dir=tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "Invalid AB result" in html
