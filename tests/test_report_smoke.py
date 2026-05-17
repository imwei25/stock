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


from unittest.mock import patch

from stockpool.cli import main


def test_cli_run_smoke(tmp_path, monkeypatch):
    """End-to-end: mock fetcher + trading-day check, confirm CLI produces report."""
    out_dir = (tmp_path / "out").as_posix()
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
stocks:
  - {{code: "605589", name: "圣泉集团"}}
data: {{history_days: 120, cache_dir: "data", force_refresh: false}}
indicators:
  ma_periods: [5, 10, 20, 60]
  macd: {{fast: 12, slow: 26, signal: 9}}
  kdj: {{n: 9, m1: 3, m2: 3}}
  rsi_periods: [6, 12, 24]
  boll: {{n: 20, k: 2}}
  volume_ratio_window: 5
  breakout_window: 20
weights:
  ma_cross_strong: 2
  ma_alignment: 1
  macd_cross_above_zero: 2
  macd_cross_below_zero: 1
  macd_histogram_expand: 1
  kdj_oversold_cross: 2
  kdj_overbought_cross: 2
  kdj_normal_cross: 1
  rsi_oversold: 1
  rsi_overbought: 1
  boll_band_touch: 2
  boll_mid_cross: 1
  volume_surge_bullish: 1
  volume_surge_bearish: 1
  breakout_new_high: 2
  breakout_new_low: 2
scoring: {{daily_weight: 0.7, weekly_weight: 0.3, resonance_bonus: 2,
          resonance_daily_threshold: 3, resonance_weekly_threshold: 1}}
verdicts: {{strong_buy: 6, buy: 3, sell: -3, strong_sell: -6}}
backtest: {{forward_days: [5, 10, 20]}}
report: {{output_dir: "{out_dir}", keep_history: true, klines_to_show: 120}}
""", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(7)
    close = 10 + np.cumsum(rng.normal(0.02, 0.3, 200))
    fake = pd.DataFrame({
        "日期": pd.date_range("2025-08-01", periods=200, freq="B").strftime("%Y-%m-%d"),
        "开盘": close - 0.1, "收盘": close, "最高": close + 0.2, "最低": close - 0.2,
        "成交量": rng.integers(500_000, 2_000_000, 200), "成交额": [0] * 200,
        "振幅": [0] * 200, "涨跌幅": [0] * 200, "涨跌额": [0] * 200, "换手率": [0] * 200,
    })

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake), \
         patch("stockpool.cli.ak.tool_trade_date_hist_sina",
               return_value=pd.DataFrame({"trade_date": [pd.Timestamp.today().date()]})):
        exit_code = main(["run", "--config", str(config_yaml)])

    assert exit_code == 0
    reports = list((tmp_path / "out").rglob("index.html"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "605589" in text
    assert "圣泉集团" in text
    assert (tmp_path / "out" / "latest.html").exists()


def test_report_includes_verdict_bucket_section():
    from stockpool.report import _stock_section_html, StockAnalysis
    a = StockAnalysis(
        code="000001", name="测试",
        daily_score=2, weekly_score=1, final_score=3.0, verdict="buy",
        verdict_hit_rates={
            "buy": {
                "count": 5,
                "forward_5":  {"mean_return_pct": 1.2, "win_rate": 0.6, "sample_size": 5},
                "forward_10": {"mean_return_pct": 0.0, "win_rate": 0.0, "sample_size": 0},
                "forward_20": {"mean_return_pct": 0.0, "win_rate": 0.0, "sample_size": 0},
            },
        },
    )
    html = _stock_section_html(a, klines_to_show=60)
    assert "综合评级历史回测" in html
    assert "🟢 买入" in html
    assert "+1.20%" in html
