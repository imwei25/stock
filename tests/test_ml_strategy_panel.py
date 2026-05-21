"""验证 strategy_factory 在 pooled 模式下注入 factor_panel,使得 WQ101 cross-sec
因子在 predict 阶段也走真实横截面值(而不是 1-stock 退化)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.backtesting.strategies import MLFactorStrategy
from stockpool.ml.dataset import compute_factor_panel


def _stock_df(close: list[float], code: str = "X", seed: int = 0) -> pd.DataFrame:
    n = len(close)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "open": [c * 0.998 for c in close],
        "high": [c * 1.005 for c in close],
        "low": [c * 0.995 for c in close],
        "close": close,
        "volume": rng.uniform(5e5, 2e6, n),
    })


def _panel_from_pool(pool: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    per_stock = {c: d.set_index(pd.to_datetime(d["date"])).sort_index()
                 for c, d in pool.items()}
    dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(dates, name="date")
    return {
        f: pd.DataFrame({c: d[f].reindex(idx) for c, d in per_stock.items()}, index=idx)
        for f in ("open", "high", "low", "close", "volume")
    }


def test_factor_panel_propagates_via_with_stock():
    pool = {
        "A": _stock_df(list(np.linspace(100, 150, 80))),
        "B": _stock_df(list(np.linspace(80, 60, 80))),
    }
    panel = _panel_from_pool(pool)
    factor_panel = compute_factor_panel(panel, ["alpha_003", "momentum_5"])

    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig(
        factors=["alpha_003", "momentum_5"], horizon=3,
        train_window=30, min_train_samples=20, refit_every=10,
        panel_mode="pooled",
    )
    strat = MLFactorStrategy(cfg=cfg, pool_data=pool, factor_panel=factor_panel)
    sa = strat.with_stock("A")
    sb = strat.with_stock("B")
    assert sa._factor_panel is not None
    assert "alpha_003" in sa._factor_panel
    # with_stock 应当延续同一份 panel
    assert sa._factor_panel is sb._factor_panel or sa._factor_panel == sb._factor_panel


def test_xfull_from_panel_differs_from_singleton_for_cross_sec():
    """关键正确性:cross-sec 因子在 panel 模式下应当不是常数。"""
    # 让价格在时间上交叉(否则单股 rank 是常数,corr 全 NaN)
    rng = np.random.default_rng(0)
    pool = {
        "A": _stock_df(list(100 + np.cumsum(rng.standard_normal(80))), seed=1),
        "B": _stock_df(list(100 + np.cumsum(rng.standard_normal(80))), seed=2),
        "C": _stock_df(list(100 + np.cumsum(rng.standard_normal(80))), seed=3),
    }
    panel = _panel_from_pool(pool)
    factor_panel = compute_factor_panel(panel, ["alpha_003"])  # cross-sec

    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig(
        factors=["alpha_003"], horizon=3, train_window=30,
        min_train_samples=20, refit_every=10, panel_mode="pooled",
    )

    # 注入 panel
    strat_a = MLFactorStrategy(cfg=cfg, pool_data=pool, factor_panel=factor_panel,
                                current_stock_code="A")
    X_panel = strat_a._build_x_full(pool["A"])
    # 不注入 panel,走 1-stock 退化
    strat_a2 = MLFactorStrategy(cfg=cfg, pool_data=pool, current_stock_code="A")
    X_singleton = strat_a2._build_x_full(pool["A"])

    # cross-sec 因子在 panel 模式下,B/C 存在时 rank 在 (0, 1) 区间,
    # 而 1-stock 退化时 rank 永远等于 1.0 (或常数)
    # 过滤 inf 后比较 finite 值集合 —— panel 注入应当与 1-stock 退化版本不同
    pv = X_panel["alpha_003"].replace([np.inf, -np.inf], np.nan).dropna()
    sv = X_singleton["alpha_003"].replace([np.inf, -np.inf], np.nan).dropna()
    # 退化版本: 单股内 rank 是常数,corr(const, _, 10) 无定义 → 应该全 NaN
    assert len(sv) == 0, f"1-stock 退化应该全 NaN,实际有 {len(sv)} 个有效值"
    # panel 注入: 应该产出真实的横截面 rank,有非空值
    assert len(pv) >= 5, f"panel 注入应有 >=5 个 finite 值,实际 {len(pv)}"


def test_build_strategy_injects_panel_in_pooled_mode():
    from stockpool.config import (
        AppConfig, DataConfig, IndicatorsConfig, WeightsConfig, ScoringConfig,
        VerdictsConfig, BacktestConfig, ReportConfig, StrategyConfig,
        MLFactorConfig, MACDConfig, KDJConfig, BOLLConfig, Stock,
    )
    from stockpool.strategy_factory import build_strategy

    pool = {
        "A": _stock_df(list(np.linspace(100, 150, 80))),
        "B": _stock_df(list(np.linspace(80, 100, 80))),
    }
    cfg = AppConfig(
        stocks=[Stock(code="A", name="a"), Stock(code="B", name="b")],
        data=DataConfig(history_days=80, cache_dir="data"),
        indicators=IndicatorsConfig(
            ma_periods=[5, 10], macd=MACDConfig(fast=12, slow=26, signal=9),
            kdj=KDJConfig(n=9, m1=3, m2=3), rsi_periods=[14],
            boll=BOLLConfig(n=20, k=2.0),
            volume_ratio_window=5, breakout_window=20,
        ),
        weights=WeightsConfig(**{
            "ma_cross_strong": 1, "ma_alignment": 1,
            "macd_cross_above_zero": 1, "macd_cross_below_zero": -1,
            "macd_histogram_expand": 1,
            "kdj_oversold_cross": 1, "kdj_overbought_cross": -1, "kdj_normal_cross": 0,
            "rsi_oversold": 1, "rsi_overbought": -1,
            "boll_band_touch": 1, "boll_mid_cross": 0,
            "volume_surge_bullish": 1, "volume_surge_bearish": -1,
            "breakout_new_high": 1, "breakout_new_low": -1,
        }),
        scoring=ScoringConfig(daily_weight=0.6, weekly_weight=0.4, resonance_bonus=2,
                              resonance_daily_threshold=3, resonance_weekly_threshold=3),
        verdicts=VerdictsConfig(strong_buy=8, buy=4, sell=-4, strong_sell=-8),
        backtest=BacktestConfig(forward_days=[3, 5]),
        report=ReportConfig(output_dir="reports", keep_history=False, klines_to_show=60),
        strategy=StrategyConfig(name="ml_factor", ml_factor=MLFactorConfig(
            factors=["alpha_003"], horizon=3, train_window=30,
            min_train_samples=20, panel_mode="pooled",
        )),
    )
    strat = build_strategy(cfg, pool_data=pool, current_stock_code="A")
    assert isinstance(strat, MLFactorStrategy)
    assert strat._factor_panel is not None
    assert "alpha_003" in strat._factor_panel
