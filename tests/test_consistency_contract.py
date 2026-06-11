"""D3 / P1-7:predict/backtest 一致性契约 + 端到端黄金值回归锚点。

这是项目作为"实盘信号工具"的核心契约:日报(predict_latest)与回测
(generate_signals 末行)对同一天必须给出同一信号。
"""
import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.framework import BacktestEngine, TradeCosts
from stockpool.backtesting.strategies import CompositeVerdictStrategy, MLFactorStrategy
from stockpool.config import (
    BOLLConfig, IndicatorsConfig, KDJConfig, MACDConfig, MLFactorConfig,
    ScoringConfig, SelectorConfig, VerdictsConfig, WeighterConfig, WeightsConfig,
)


def _synthetic_daily(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n),
        "open": close * 0.998, "high": close * 1.005,
        "low": close * 0.995, "close": close,
        "volume": rng.integers(5e5, 5e6, n).astype(float),
    })


def _composite_cfgs():
    weights = WeightsConfig(
        ma_cross_strong=2, ma_alignment=1, macd_cross_above_zero=2,
        macd_cross_below_zero=2, macd_histogram_expand=1,
        kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1, boll_band_touch=1, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1,
        breakout_new_high=2, breakout_new_low=2,
    )
    scoring = ScoringConfig(daily_weight=0.7, weekly_weight=0.3,
                            resonance_bonus=2, resonance_daily_threshold=3,
                            resonance_weekly_threshold=1)
    verdicts = VerdictsConfig(strong_buy=6, buy=3, sell=-3, strong_sell=-6)
    indicators = IndicatorsConfig(
        ma_periods=[5, 10, 20, 60], macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3), rsi_periods=[6, 12, 24],
        boll=BOLLConfig(n=20, k=2.0), volume_ratio_window=5, breakout_window=20,
    )
    return weights, scoring, verdicts, indicators


def test_composite_predict_matches_signals_last_row():
    w, s, v, ind = _composite_cfgs()
    strat = CompositeVerdictStrategy(weights=w, scoring=s, verdicts_cfg=v,
                                     indicators_cfg=ind)
    daily = _synthetic_daily()
    latest = strat.predict_latest(daily)
    sig = strat.generate_signals(daily)
    assert latest["signal"] == sig["signal"].iloc[-1], (
        f"日报 {latest['signal']} != 回测末行 {sig['signal'].iloc[-1]}"
    )
    assert latest["final_score"] == pytest.approx(
        float(sig["final_score"].iloc[-1]), abs=1e-9,
    )


def test_ml_predict_matches_signals_last_row(tmp_path):
    """refit_every=1 + 无月度缓存差异下,两路径对最后一根 bar 必须等价。
    (月度缓存导致的 fit 时点差是已知豁免;基础数学必须一致。)"""
    cfg = MLFactorConfig(
        factors=["momentum_10", "rsi_centered_14"],
        horizon=3, train_window=120, min_train_samples=40,
        refit_every=1, panel_mode="per_stock", embargo_days=0,
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
        share_pool_fit=False,
    )
    daily = _synthetic_daily(n=200, seed=3)
    strat = MLFactorStrategy(cfg=cfg)  # 无 cache_dir → 不落盘,纯数学路径
    latest = strat.predict_latest(daily)
    sig = strat.generate_signals(daily)
    assert latest["signal"] == sig["signal"].iloc[-1]
    assert latest["score"] == pytest.approx(float(sig["score"].iloc[-1]), rel=1e-9)


def test_ml_predict_reports_fit_date(tmp_path):
    cfg = MLFactorConfig(
        factors=["momentum_10"], horizon=3, train_window=120,
        min_train_samples=40, refit_every=20, panel_mode="per_stock",
        embargo_days=0, share_pool_fit=False,
    )
    daily = _synthetic_daily(n=200, seed=5)
    strat = MLFactorStrategy(cfg=cfg, cache_dir=tmp_path)
    latest = strat.predict_latest(daily)
    assert latest.get("model_fit_date") is not None, "日报必须标注模型训练时点"


# ── 黄金值回归锚点 ────────────────────────────────────────────────────────────

class _GoldenStrategy:
    """确定性策略:bar2 buy,bar6 sell。"""
    name = "golden"

    def generate_signals(self, daily_df):
        sig = ["neutral"] * len(daily_df)
        sig[2], sig[6] = "buy", "sell"
        return pd.DataFrame({
            "date": daily_df["date"], "open": daily_df["open"],
            "close": daily_df["close"], "signal": sig,
        })

    def should_enter(self, ctx):
        return ctx.signal == "buy"

    def should_exit(self, ctx):
        return ctx.signal == "sell"

    def should_reset_timer(self, ctx):
        return False


def test_golden_backtest_exact_values():
    """端到端黄金值:固定 OHLCV → 完整回测 → 断言精确 equity/metrics。
    引擎任何记账语义变化(费率、口径、T+1)都会在这里报警。"""
    closes = [10.0, 10.2, 10.1, 10.5, 10.4, 10.8, 10.6, 10.9, 11.0, 11.2]
    opens = [10.0, 10.1, 10.15, 10.3, 10.45, 10.6, 10.7, 10.75, 10.95, 11.1]
    df = pd.DataFrame({
        "date": pd.bdate_range("2025-03-03", periods=10),
        "open": opens, "high": [c + 0.1 for c in closes],
        "low": [c - 0.1 for c in closes], "close": closes,
        "volume": [1e6] * 10,
    })
    eng = BacktestEngine(strategy=_GoldenStrategy(),
                         costs=TradeCosts(buy_cost=0.001, sell_cost=0.002))
    res = eng.run(df, max_holding_days=30)

    # 手算:bar2 收盘 buy 信号 → bar3 open=10.3 成交,扣 0.1%:
    #   equity[3] = 0.999 × (10.5/10.3) = 1.018403...
    #   持有到 bar6 收盘 sell 信号 → bar7 open=10.75 卖出,扣 0.2%:
    #   equity[7] = 0.999 × (10.75/10.3) × 0.998
    eq = res.curve["equity"]
    assert eq.iloc[2] == pytest.approx(1.0)
    assert eq.iloc[3] == pytest.approx(0.999 * 10.5 / 10.3, rel=1e-12)
    expected_exit = 0.999 * (10.75 / 10.3) * 0.998
    assert eq.iloc[7] == pytest.approx(expected_exit, rel=1e-12)
    assert eq.iloc[-1] == pytest.approx(expected_exit, rel=1e-12)

    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.entry_price == pytest.approx(10.3)
    assert tr.exit_price == pytest.approx(10.75)
    # Trade.ret 含双边成本(P2-8 口径):exit_equity / pre_buy_equity − 1
    assert tr.ret == pytest.approx(expected_exit / 1.0 - 1, rel=1e-12)
    assert res.metrics["total_return"] == pytest.approx(expected_exit - 1, rel=1e-12)
    assert res.metrics["trade_count"] == 1
    assert res.metrics["win_rate"] == 1.0
    # 10 根 bar < 60 → 年化置 None(P3-15 口径)
    assert res.metrics["annualized_return"] is None
