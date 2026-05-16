"""Smoke tests — verify HTML generates and contains expected markers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpool.config import BOLLConfig, IndicatorsConfig, KDJConfig, MACDConfig
from stockpool.indicators import add_all
from stockpool.report import StockAnalysis, build_stock_chart, render_report
from stockpool.signals import Trigger


@pytest.fixture
def indicators_cfg() -> IndicatorsConfig:
    return IndicatorsConfig(
        ma_periods=[5, 10, 20, 60],
        macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3),
        rsi_periods=[6, 12, 24],
        boll=BOLLConfig(n=20, k=2),
        volume_ratio_window=5,
        breakout_window=20,
    )


def _make_long_history(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 10 + np.cumsum(rng.normal(0.02, 0.3, n))
    return pd.DataFrame({
        "date": pd.date_range("2025-08-01", periods=n, freq="B"),
        "open": close + rng.normal(0, 0.1, n),
        "high": close + np.abs(rng.normal(0.2, 0.1, n)),
        "low":  close - np.abs(rng.normal(0.2, 0.1, n)),
        "close": close,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    })


def test_build_stock_chart_returns_html(indicators_cfg):
    raw = _make_long_history(120)
    enriched = add_all(raw, indicators_cfg)

    grid = build_stock_chart("605589", "圣泉集团", enriched, klines_to_show=120)
    html = grid.render_embed()

    assert "605589" in html
    # pyecharts JSON-escapes non-ASCII; just verify echarts wiring is present
    assert "echarts" in html.lower()
    assert "candlestick" in html.lower()


def _make_analysis(code: str, name: str, score: float, verdict: str,
                   indicators_cfg) -> StockAnalysis:
    raw = _make_long_history(120)
    enriched = add_all(raw, indicators_cfg)
    return StockAnalysis(
        code=code, name=name,
        daily_score=int(round(score)),
        weekly_score=int(round(score / 3)),
        final_score=score, verdict=verdict,
        triggers_daily=[Trigger("macd_cross_above_zero", +1, 2, "MACD 零轴上方金叉")],
        triggers_weekly=[],
        hit_rates={
            "macd_cross_above_zero": {
                "count": 5, "direction": +1,
                "forward_5": {"mean_return_pct": 2.1, "win_rate": 0.6, "sample_size": 5},
                "forward_10": {"mean_return_pct": 3.4, "win_rate": 0.6, "sample_size": 5},
                "forward_20": {"mean_return_pct": 5.0, "win_rate": 0.6, "sample_size": 5},
            }
        },
        daily_with_indicators=enriched,
    )


def test_render_report_writes_html(tmp_path, indicators_cfg):
    analyses = [
        _make_analysis("605589", "圣泉集团", 6.1, "strong_buy", indicators_cfg),
        _make_analysis("603986", "兆易创新", -4.0, "sell", indicators_cfg),
        _make_analysis("000528", "柳工",     0.5, "neutral", indicators_cfg),
    ]
    out = render_report(
        analyses, run_date="2026-05-17",
        config_path=Path("config.yaml"), config_hash="abc12345",
        output_dir=tmp_path, keep_history=True, klines_to_show=120,
    )

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    for keyword in ["养龙股池", "2026-05-17", "圣泉集团", "兆易创新", "柳工",
                    "强烈买入", "卖出观察", "abc12345", "免责声明"]:
        assert keyword in text, f"missing {keyword!r} in report"

    assert (tmp_path / "latest.html").exists()
