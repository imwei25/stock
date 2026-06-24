"""C2 / P1-3:涨跌停拒单 —— 一字涨停开盘买不进,一字跌停开盘卖不出(顺延)。"""
import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.framework import (
    BacktestEngine, MultiLotBacktestEngine, Strategy, BarContext, PositionContext,
    TradeCosts,
)


class _BuyOnce(Strategy):
    """bar0 给 buy 信号(应在 bar1 open 成交),之后 neutral。"""
    def __init__(self, buy_bars=(0,), sell_bars=()):
        self._buy = set(buy_bars)
        self._sell = set(sell_bars)

    @property
    def name(self):
        return "buy_once"

    def generate_signals(self, daily_df):
        sig = ["neutral"] * len(daily_df)
        for b in self._buy:
            sig[b] = "buy"
        for b in self._sell:
            sig[b] = "sell"
        return pd.DataFrame({
            "date": daily_df["date"], "open": daily_df["open"],
            "close": daily_df["close"], "signal": sig,
        })

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal == "buy"

    def should_exit(self, ctx: PositionContext) -> bool:
        return ctx.signal == "sell"


def _df(opens, closes):
    n = len(opens)
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-06", periods=n),
        "open": opens, "high": [max(o, c) for o, c in zip(opens, closes)],
        "low": [min(o, c) for o, c in zip(opens, closes)],
        "close": closes, "volume": [1e6] * n,
    })


def test_limit_up_open_rejects_entry():
    """bar1 开盘一字涨停(+10%)→ 买入被拒;bar0 信号只持续一根 → 永不进场。"""
    df = _df(opens=[10.0, 11.0, 11.1, 11.2],   # bar1 open = prev_close×1.10
             closes=[10.0, 11.05, 11.1, 11.2])
    eng = BacktestEngine(strategy=_BuyOnce(), costs=TradeCosts(0, 0),
                         limit_pct=0.10)
    res = eng.run(df, max_holding_days=10)
    assert len(res.trades) == 0
    assert (res.curve["position"] == 0).all(), "涨停开盘应拒单"
    assert res.curve["equity"].iloc[-1] == pytest.approx(1.0)


def test_limit_up_entry_retries_next_bar():
    """信号持续两根:bar1 涨停拒单,bar2 开盘正常 → bar2 成交。"""
    df = _df(opens=[10.0, 11.0, 11.0, 11.1],
             closes=[10.0, 11.0, 11.05, 11.1])
    eng = BacktestEngine(strategy=_BuyOnce(buy_bars=(0, 1)),
                         costs=TradeCosts(0, 0), limit_pct=0.10)
    res = eng.run(df, max_holding_days=10)
    assert res.curve["position"].iloc[1] == 0
    assert res.curve["position"].iloc[2] == 1, "次日开盘未涨停应成交"


def test_limit_down_open_defers_exit():
    """持仓中 bar2 开盘一字跌停 → 卖不出,顺延到 bar3 成交。"""
    df = _df(opens=[10.0, 10.0, 9.0, 8.9],     # bar2 open = 10×0.90 跌停
             closes=[10.0, 10.0, 9.0, 8.9])
    eng = BacktestEngine(strategy=_BuyOnce(buy_bars=(0,), sell_bars=(1, 2)),
                         costs=TradeCosts(0, 0), limit_pct=0.10)
    res = eng.run(df, max_holding_days=10)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.exit_idx == 3, f"跌停日卖不出,应顺延到 bar3,实际 exit_idx={tr.exit_idx}"
    # 跌停日继续吃下跌(equity 反映 bar2 持仓)
    assert res.curve["position"].iloc[2] == 1


def test_no_limit_pct_means_no_rejection():
    """limit_pct=None(默认)保持旧行为 —— 既有测试/夹具不受影响。"""
    df = _df(opens=[10.0, 11.0, 11.1, 11.2],
             closes=[10.0, 11.05, 11.1, 11.2])
    eng = BacktestEngine(strategy=_BuyOnce(), costs=TradeCosts(0, 0))
    res = eng.run(df, max_holding_days=10)
    assert res.curve["position"].iloc[1] == 1


def test_multi_lot_limit_up_rejects_entry():
    df = _df(opens=[10.0, 11.0, 11.0, 11.1],
             closes=[10.0, 11.0, 11.05, 11.1])
    eng = MultiLotBacktestEngine(strategy=_BuyOnce(buy_bars=(0, 1)),
                                 costs=TradeCosts(0, 0), limit_pct=0.10)
    res = eng.run(df, max_holding_days=10)
    assert res.curve["position"].iloc[1] == 0, "涨停开盘应拒单"
    assert res.curve["position"].iloc[2] == 1


def test_multi_lot_limit_down_defers_exit():
    df = _df(opens=[10.0, 10.0, 9.0, 8.9],
             closes=[10.0, 10.0, 9.0, 8.9])
    eng = MultiLotBacktestEngine(strategy=_BuyOnce(buy_bars=(0,), sell_bars=(1, 2)),
                                 costs=TradeCosts(0, 0), limit_pct=0.10)
    res = eng.run(df, max_holding_days=10)
    assert len(res.trades) == 1
    assert res.trades[0].exit_idx == 3


def test_infer_limit_pct():
    from stockpool.backtesting.limits import infer_limit_pct
    assert infer_limit_pct("300123") == pytest.approx(0.20)
    assert infer_limit_pct("688001") == pytest.approx(0.20)
    assert infer_limit_pct("600000") == pytest.approx(0.10)
    assert infer_limit_pct("600000", st_codes={"600000"}) == pytest.approx(0.05)
    assert infer_limit_pct("300123", st_codes={"300123"}) == pytest.approx(0.20), (
        "创业板 ST 仍 20%,板块优先"
    )
