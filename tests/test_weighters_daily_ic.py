"""P2-1:ICWeighter/IRWeighter 改逐日截面 IC(与选因子统计量一致,且与行序解耦)。

旧实现把 (stock,date) 长表整体算一个池化 Spearman——时序变异与截面变异混杂,
高离散度交易日主导;IRWeighter 按行号分块,行序为 stock-major 时算出的是
"跨股稳定性"而非时间稳定性。
"""
import numpy as np
import pandas as pd
import pytest

from stockpool.ml.weighters import ICWeighter, IRWeighter


def _panel_xy(n_days=40, n_stocks=12, seed=7):
    """构造两因子长表:
    - cs_factor:每日截面上与 y 完全同序(日内 rank 相关 = 1),但日间均值漂移大;
    - date_factor:日内常数、日间趋势 —— 截面信息为零,池化相关却很高。
    y = 日效应(大)+ 截面信号(小)。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    stocks = [f"S{i:02d}" for i in range(n_stocks)]
    rows = []
    day_effect = np.cumsum(rng.normal(0, 0.05, n_days))  # 日间大幅漂移
    for d_i, d in enumerate(dates):
        cs = rng.normal(0, 1, n_stocks)  # 当日截面信号
        for s_i, s in enumerate(stocks):
            rows.append({
                "stock": stocks[s_i], "date": d,
                "cs_factor": cs[s_i] + rng.normal(0, 5),  # 加日内噪声?不:保持同序
            })
    # 重新构造:cs_factor 与 y 的截面部分严格同值(日内同序),日间加漂移
    idx = pd.MultiIndex.from_product([stocks, dates], names=["stock", "date"])
    cs_signal = rng.normal(0, 0.01, (n_days, n_stocks))
    X = pd.DataFrame(index=idx, columns=["cs_factor", "date_factor"], dtype=float)
    y = pd.Series(index=idx, dtype=float)
    for d_i, d in enumerate(dates):
        for s_i, s in enumerate(stocks):
            X.loc[(s, d), "cs_factor"] = cs_signal[d_i, s_i] - 10 * day_effect[d_i]
            X.loc[(s, d), "date_factor"] = day_effect[d_i]
            y.loc[(s, d)] = cs_signal[d_i, s_i] + day_effect[d_i]
    return X, y


def test_ic_weighter_uses_daily_cross_sectional_ic():
    X, y = _panel_xy()
    w = ICWeighter(use_rank=True)
    w.fit(X, y)
    weights = w.weights()
    # cs_factor 每日截面与 y 完全同序 → 日均 IC ≈ 1;date_factor 日内常数 → IC = 0
    assert weights["cs_factor"] > 0.9, f"cs_factor 应拿走几乎全部权重,实际 {weights.to_dict()}"
    assert abs(weights["date_factor"]) < 0.1


def test_ic_weighter_row_order_invariant():
    X, y = _panel_xy()
    w1 = ICWeighter()
    w1.fit(X, y)
    # 打乱为 date-major 顺序
    X2 = X.swaplevel().sort_index()
    y2 = y.swaplevel().sort_index()
    w2 = ICWeighter()
    w2.fit(X2, y2)
    pd.testing.assert_series_equal(
        w1.weights().sort_index(), w2.weights().sort_index(), atol=1e-12, rtol=0,
    )


def test_ir_weighter_chunks_by_time_not_rows():
    """IR 的分块必须按时间切;stock-major 与 date-major 行序结果一致。"""
    X, y = _panel_xy(n_days=60)
    w1 = IRWeighter(n_chunks=5)
    w1.fit(X, y)  # stock-major(字典序)
    X2 = X.swaplevel().sort_index()  # date-major
    y2 = y.swaplevel().sort_index()
    w2 = IRWeighter(n_chunks=5)
    w2.fit(X2, y2)
    pd.testing.assert_series_equal(
        w1.weights().sort_index(), w2.weights().sort_index(), atol=1e-12, rtol=0,
    )
    # 且 cs_factor 的 IR 应显著为正
    assert w1.ir["cs_factor"] > 1.0


def test_plain_index_falls_back_to_pooled_corr():
    """per_stock 模式:X index 是普通 date 索引(单股无截面)→ 池化时序相关。"""
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.bdate_range("2025-01-02", periods=n, name="date")
    f = pd.Series(rng.normal(0, 1, n), index=idx)
    X = pd.DataFrame({"f": f})
    y = pd.Series(f.to_numpy() * 0.5 + rng.normal(0, 0.1, n), index=idx)
    w = ICWeighter()
    w.fit(X, y)
    assert w.weights()["f"] > 0.5  # 单因子 L1 归一后应为 ±1 量级
    w2 = IRWeighter(n_chunks=4)
    w2.fit(X, y)
    assert w2.weights()["f"] > 0.5
