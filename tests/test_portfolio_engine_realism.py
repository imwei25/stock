"""C3+C4:组合引擎执行真实性。

- P1-4 差量调仓(存活仓不动)+ turnover 指标
- P1-5 last_valid_close 计值 + 退市核销
- P2-9 top-K 不窥视 open[t+1](买不进的腿现金闲置,不顺位替补)
- P2-10 weight_at_entry 分母固定
- P1-3 组合级涨跌停拒单
- P2-3 min_commission 地板费
"""
import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.framework import TradeCosts
from stockpool.config import PortfolioRunConfig
from stockpool.portfolio.engine import PortfolioEngine


class _FixedScores:
    """每个 rebalance bar 给固定分数表(按日期切换)。"""
    name = "fixed_scores"

    def __init__(self, scores_by_call: list[dict]):
        self._calls = list(scores_by_call)
        self._i = 0

    def predict_scores(self, date_t, panel_data):
        s = self._calls[min(self._i, len(self._calls) - 1)]
        self._i += 1
        return dict(s)


def _df(closes, opens=None, start="2025-01-06"):
    n = len(closes)
    opens = opens if opens is not None else closes
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": opens, "high": closes, "low": closes,
        "close": closes, "volume": [1e6] * n,
    })


def _cfg(**kw):
    base = dict(top_k=2, rebalance_n_days=2, max_per_industry=None,
                initial_cash=1_000_000.0)
    base.update(kw)
    return PortfolioRunConfig(**base)


def test_diff_rebalance_keeps_survivors():
    """目标 {A,B} → {A,C}:A 不动(无虚构换手),B 卖出,C 买入。"""
    n = 8
    panel = {c: _df([10.0] * n) for c in ("A", "B", "C")}
    strat = _FixedScores([
        {"A": 3.0, "B": 2.0, "C": 1.0},   # bar0 决策 → bar1 买 A,B
        {"A": 3.0, "C": 2.0, "B": 1.0},   # bar2 决策 → bar3 卖 B 买 C
        {"A": 3.0, "C": 2.0, "B": 1.0},
        {"A": 3.0, "C": 2.0, "B": 1.0},
    ])
    eng = PortfolioEngine(strat, _cfg(), costs=TradeCosts(0.001, 0.001))
    res = eng.run(panel)

    # A 在中途不应出现任何平仓记录(存活仓不动)
    mid_trades = [t for t in res.trades if t.code == "A"
                  and t.exit_reason != "end_of_backtest"]
    assert mid_trades == [], f"存活仓 A 不应被虚构换手: {mid_trades}"
    b_exits = [t for t in res.trades if t.code == "B"]
    assert len(b_exits) == 1 and b_exits[0].exit_reason == "rebalance_drop"


def test_turnover_metrics_present():
    n = 8
    panel = {c: _df([10.0] * n) for c in ("A", "B", "C")}
    strat = _FixedScores([
        {"A": 3.0, "B": 2.0, "C": 1.0},
        {"A": 3.0, "C": 2.0, "B": 1.0},
        {"C": 3.0, "B": 2.0, "A": 1.0},
        {"C": 3.0, "B": 2.0, "A": 1.0},
    ])
    eng = PortfolioEngine(strat, _cfg(), costs=TradeCosts(0.001, 0.001))
    res = eng.run(panel)
    assert "turnover" in res.rebalance_log.columns
    assert "annualized_turnover" in res.metrics
    assert res.metrics["annualized_turnover"] is not None
    # 第一次建仓后的 rebalance 应有非零 turnover(B→C 切换)
    assert res.rebalance_log["turnover"].iloc[1] > 0


def test_weight_at_entry_uniform_for_equal_buys():
    """P2-10:同批等额买入的 weight_at_entry 必须相等(旧 bug:递减 cash 当分母)。"""
    n = 6
    panel = {c: _df([10.0] * n) for c in ("A", "B")}
    strat = _FixedScores([{"A": 2.0, "B": 1.0}] * 3)
    eng = PortfolioEngine(strat, _cfg(top_k=2), costs=TradeCosts(0, 0))
    res = eng.run(panel)
    entries = {t.code: t.weight_at_entry for t in res.trades}
    assert entries["A"] == pytest.approx(entries["B"]), (
        f"等额买入权重应相等: {entries}"
    )
    assert entries["A"] == pytest.approx(0.5, abs=0.05)


def test_suspended_position_marks_at_last_valid_close():
    """P1-5:停牌(close NaN)期间按最后有效价计值,不回落 entry_price。"""
    n = 8
    closes_a = [10.0, 10.0, 15.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    panel = {
        "A": _df(closes_a),
        "B": _df([10.0] * n),
    }
    strat = _FixedScores([{"A": 2.0, "B": 1.0}] * 4)
    eng = PortfolioEngine(strat, _cfg(top_k=1, rebalance_n_days=10),
                          costs=TradeCosts(0, 0))
    res = eng.run(panel)
    # bar1 以 10 买入 A;bar2 涨到 15;bar3 起停牌 → equity 应保持 15 的 mark
    eq = res.curve["equity"]
    assert eq.iloc[4] == pytest.approx(eq.iloc[2]), (
        "停牌期间应按最后有效 close(15)计值,而非入场价 10"
    )
    assert eq.iloc[2] == pytest.approx(1_000_000.0 * 1.5)


def test_delisted_position_written_off():
    """P1-5:连续 delist_after_bars 根无报价 → 强制核销 exit_reason=delisted。"""
    n = 12
    closes_a = [10.0, 10.0, 8.0] + [np.nan] * (n - 3)
    panel = {"A": _df(closes_a), "B": _df([10.0] * n)}
    strat = _FixedScores([{"A": 2.0, "B": 1.0}] * 6)
    eng = PortfolioEngine(
        strat, _cfg(top_k=1, rebalance_n_days=20, delist_after_bars=4),
        costs=TradeCosts(0, 0),
    )
    res = eng.run(panel)
    delisted = [t for t in res.trades if t.exit_reason == "delisted"]
    assert len(delisted) == 1 and delisted[0].code == "A"
    assert delisted[0].exit_price == pytest.approx(8.0), "按最后有效价核销"


def test_top_k_does_not_peek_next_open():
    """P2-9:候选按 t 时点信息选;t+1 停牌的腿现金闲置,不顺位替补。"""
    n = 6
    opens_a = [10.0, np.nan, 10.0, 10.0, 10.0, 10.0]  # bar1 无开盘价(停牌)
    panel = {
        "A": _df([10.0] * n, opens=opens_a),
        "B": _df([10.0] * n),
        "C": _df([10.0] * n),
    }
    # top_k=2:A(最高分)与 B 入选;A bar1 买不进 → 现金闲置,C 不得替补
    strat = _FixedScores([{"A": 3.0, "B": 2.0, "C": 1.0}] * 3)
    eng = PortfolioEngine(strat, _cfg(top_k=2, rebalance_n_days=10),
                          costs=TradeCosts(0, 0))
    res = eng.run(panel)
    held_codes = {t.code for t in res.trades}
    assert "C" not in held_codes, "买不进的腿应现金闲置,不能顺位替补 C"
    assert res.curve["cash_ratio"].iloc[1] == pytest.approx(0.5, abs=0.01)


def test_portfolio_limit_up_blocks_buy():
    """P1-3:执行 bar 开盘一字涨停 → 该腿买不进。"""
    n = 6
    opens_a = [10.0, 11.0, 11.0, 11.0, 11.0, 11.0]  # bar1 open = +10% 涨停
    closes_a = [10.0, 11.0, 11.0, 11.0, 11.0, 11.0]
    panel = {"600001": _df(closes_a, opens=opens_a), "600002": _df([10.0] * n)}
    strat = _FixedScores([{"600001": 2.0, "600002": 1.0}] * 3)
    eng = PortfolioEngine(strat, _cfg(top_k=2, rebalance_n_days=10),
                          costs=TradeCosts(0, 0))
    res = eng.run(panel)
    held = {t.code for t in res.trades}
    assert "600001" not in held, "涨停开盘应拒买"


def test_portfolio_limit_down_defers_sell():
    """P1-3:rebalance 要卖的票开盘跌停 → 卖不出,继续持有。"""
    n = 8
    closes_a = [10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    opens_a = [10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0]  # bar3 open −10% 跌停
    panel = {"600001": _df(closes_a, opens=opens_a), "600002": _df([10.0] * n)}
    strat = _FixedScores([
        {"600001": 2.0, "600002": 1.0},   # bar0 → bar1 买 600001
        {"600002": 2.0, "600001": 0.0},   # bar2 → bar3 卖 600001(跌停卖不出)
        {"600002": 2.0, "600001": 0.0},
        {"600002": 2.0, "600001": 0.0},
    ])
    eng = PortfolioEngine(strat, _cfg(top_k=1, rebalance_n_days=2),
                          costs=TradeCosts(0, 0))
    res = eng.run(panel)
    exits = [t for t in res.trades if t.code == "600001"
             and t.exit_reason == "rebalance_drop"]
    assert exits, "之后应成功卖出"
    exit_date = exits[0].exit_date
    bar3 = pd.bdate_range("2025-01-06", periods=n)[3]
    assert exit_date > bar3, f"跌停日不可卖出,实际 {exit_date}"


def test_min_commission_floor():
    """P2-3:min_commission > 0 时,每笔订单费用有地板。"""
    n = 4
    panel = {"A": _df([10.0] * n)}
    strat = _FixedScores([{"A": 1.0}] * 2)
    eng = PortfolioEngine(
        strat, _cfg(top_k=1, rebalance_n_days=10, initial_cash=10_000.0,
                    min_commission=5.0),
        costs=TradeCosts(0.0001, 0.0001),  # 比例费仅 1 元 → 地板 5 元生效
    )
    res = eng.run(panel)
    # 买入一笔:equity 应比无地板时低 ~(5−1) 元
    assert res.curve["equity"].iloc[1] <= 10_000.0 - 5.0 + 1e-6
