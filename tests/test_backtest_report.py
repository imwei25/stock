"""Smoke test for backtest_report.render_backtest_report."""
import pandas as pd

from stockpool.backtest_composite import EquityResult
from stockpool.backtest_report import render_backtest_report


def _result_for(closes: list[float]) -> EquityResult:
    dates = pd.date_range("2026-01-02", periods=len(closes), freq="B")
    curve = pd.DataFrame({
        "date": dates,
        "equity": [c / closes[0] for c in closes],
        "position": [1] * len(closes),
    })
    bh = pd.DataFrame({"date": dates, "equity": [c / closes[0] for c in closes]})
    return EquityResult(
        curves={5: curve, 10: curve, 20: curve},
        metrics={
            N: {
                "total_return": 0.1, "annualized_return": 0.05,
                "max_drawdown": 0.02, "trade_count": 3,
                "win_rate": 0.67, "avg_trade_return_pct": 1.2,
            } for N in (5, 10, 20)
        },
        buy_and_hold=bh,
        buy_and_hold_metrics={
            "total_return": 0.1, "annualized_return": 0.05,
            "max_drawdown": 0.02, "trade_count": 1,
            "win_rate": None, "avg_trade_return_pct": None,
        },
    )


def test_render_backtest_report_smoke(tmp_path):
    closes = [100, 102, 105, 103, 108, 110]
    per_stock = [
        ("605589", "圣泉集团", _result_for(closes)),
        ("603986", "兆易创新", _result_for(closes)),
    ]
    out = render_backtest_report(
        per_stock, run_date="2026-05-17", output_dir=tmp_path
    )
    html = out.read_text(encoding="utf-8")
    assert out.exists()
    assert out.stat().st_size > 1024
    assert "N=5" in html and "N=10" in html and "N=20" in html
    assert "Buy &amp; Hold" in html or "Buy & Hold" in html
    assert "605589" in html and "603986" in html
    assert (tmp_path / "latest.html").exists()
