"""C5:multi_lot 信号边沿开仓(P2-13)+ LotSizer 只见历史切片(P3-17)。"""
import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.framework import (
    BarContext, MultiLotBacktestEngine, PositionContext, Strategy, TradeCosts,
)


class _AlwaysBuy(Strategy):
    """每根 bar 都给 buy(模拟 ml_factor 分位区间内连续 buy)。"""
    @property
    def name(self):
        return "always_buy"

    def generate_signals(self, daily_df):
        return pd.DataFrame({
            "date": daily_df["date"], "open": daily_df["open"],
            "close": daily_df["close"], "signal": ["buy"] * len(daily_df),
        })

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal == "buy"

    def should_exit(self, ctx: PositionContext) -> bool:
        return False


def _df(n=10):
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-06", periods=n),
        "open": [10.0] * n, "high": [10.1] * n, "low": [9.9] * n,
        "close": [10.0] * n, "volume": [1e6] * n,
    })


def test_edge_mode_opens_single_lot_on_persistent_buy():
    """edge 模式:连续 buy 只在信号边沿开一个 lot(不再隐式金字塔加仓)。"""
    eng = MultiLotBacktestEngine(
        strategy=_AlwaysBuy(), position_size=0.1,
        costs=TradeCosts(0, 0), entry_mode="edge",
    )
    res = eng.run(_df(), max_holding_days=100)
    assert res.curve["position"].max() == 1, (
        f"edge 模式连续 buy 应只开 1 个 lot,实际峰值 {res.curve['position'].max()}"
    )


def test_every_bar_mode_pyramids():
    """every_bar(legacy):每根 bar 开新 lot 直到现金耗尽。"""
    eng = MultiLotBacktestEngine(
        strategy=_AlwaysBuy(), position_size=0.1,
        costs=TradeCosts(0, 0), entry_mode="every_bar",
    )
    res = eng.run(_df(), max_holding_days=100)
    assert res.curve["position"].max() > 3


def test_default_entry_mode_in_config_is_edge():
    from stockpool.config import BacktestConfig
    cfg = BacktestConfig(forward_days=[5])
    assert cfg.entry_mode == "edge"


def test_lot_sizer_receives_only_history():
    """P3-17:sizer 不能看到执行 bar 及之后的 close(物理切片防 look-ahead)。"""
    seen = []

    class _SpySizer:
        def __call__(self, bar_idx, opens, closes):
            seen.append((bar_idx, len(opens), len(closes)))
            return 0.1

    eng = MultiLotBacktestEngine(
        strategy=_AlwaysBuy(), lot_sizer=_SpySizer(),
        costs=TradeCosts(0, 0), entry_mode="every_bar",
    )
    eng.run(_df(), max_holding_days=100)
    assert seen
    for bar_idx, n_opens, n_closes in seen:
        assert n_closes == bar_idx, (
            f"closes 应只含执行 bar 之前的历史(len={bar_idx}),实际 {n_closes}"
        )
        assert n_opens == bar_idx + 1, "opens 可含执行 bar 自身(成交价)"
