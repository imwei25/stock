"""P0-5:组合 score panel 必须逐 code 用 with_stock 绑定(训练/预测特征一致)。"""
import numpy as np
import pandas as pd
import pytest

from stockpool.portfolio.scoring import precompute_scores_from_legacy


def _daily(code_seed: int, n=30):
    rng = np.random.default_rng(code_seed)
    close = 10 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-06", periods=n),
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6,
    })


class _BindingStrategy:
    """记录 with_stock 调用,分数编码绑定的 code,用于断言绑定真的发生。"""

    def __init__(self, bound: str | None = None, calls: list | None = None):
        self.bound = bound
        self.calls = calls if calls is not None else []

    def with_stock(self, code: str) -> "_BindingStrategy":
        self.calls.append(code)
        return _BindingStrategy(bound=code, calls=self.calls)

    def generate_signals(self, daily: pd.DataFrame) -> pd.DataFrame:
        score = float(int(self.bound)) if self.bound else -1.0
        return pd.DataFrame({
            "date": daily["date"],
            "final_score": score,
        })


class _PlainStrategy:
    """无 with_stock 的策略(composite)→ 直接复用。"""

    def generate_signals(self, daily: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"date": daily["date"], "final_score": 7.0})


def test_scoring_binds_with_stock_per_code():
    base = _BindingStrategy()
    pool = {"000001": _daily(1), "000002": _daily(2)}
    panel = precompute_scores_from_legacy(base, pool)
    assert sorted(base.calls) == ["000001", "000002"], "每个 code 都必须 with_stock 绑定"
    # 分数来自绑定后的实例(否则全是 -1)
    assert panel["000001"].iloc[-1] == pytest.approx(1.0)
    assert panel["000002"].iloc[-1] == pytest.approx(2.0)


def test_scoring_plain_strategy_reused():
    pool = {"000001": _daily(1)}
    panel = precompute_scores_from_legacy(_PlainStrategy(), pool)
    assert panel["000001"].iloc[-1] == pytest.approx(7.0)


def test_ml_strategy_with_stock_uses_panel_slice():
    """端到端:MLFactorStrategy 经 with_stock 绑定后,_build_x_full 必须
    从 factor_panel 切片(而非单股退化重算)。"""
    from stockpool.backtesting.strategies import MLFactorStrategy
    from stockpool.config import MLFactorConfig

    dates = pd.bdate_range("2025-01-06", periods=10)
    fp = {"f1": pd.DataFrame(
        {"000001": np.arange(10.0), "000002": np.arange(10.0) * 2},
        index=dates,
    )}
    cfg = MLFactorConfig(factors=["momentum_5"], panel_mode="pooled")
    base = MLFactorStrategy(cfg=cfg, factor_panel=fp)
    bound = base.with_stock("000002")
    daily = _daily(2, n=10)
    daily["date"] = dates
    X = bound._build_x_full(daily)
    # 来自 panel 切片:f1 列且值是 000002 那列
    assert "f1" in X.columns
    assert X["f1"].iloc[-1] == pytest.approx(18.0)
