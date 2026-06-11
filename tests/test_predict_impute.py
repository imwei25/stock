"""B7:predict 缺失值填 fit-time 均值(P3-9)+ mcap 缺失 NaN(P3-10)+ Lasso y 标准化(P3-12)。"""
import numpy as np
import pandas as pd
import pytest

from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LassoSelector
from stockpool.ml.weighters import ICWeighter


def _xy(n=200, seed=0, y_scale=1.0):
    rng = np.random.default_rng(seed)
    # f1 均值远离 0(模拟未 zscore 的 rank 类因子)
    f1 = rng.normal(5.0, 1.0, n)
    f2 = rng.normal(-3.0, 2.0, n)
    y = (0.6 * (f1 - 5.0) + 0.3 * (f2 + 3.0) + rng.normal(0, 0.5, n)) * y_scale
    X = pd.DataFrame({"f1": f1, "f2": f2})
    return X, pd.Series(y)


def test_fill_missing_uses_fit_means():
    X, y = _xy()
    pipe = TwoStepPipeline(selector=LassoSelector(alpha=0.001), weighter=ICWeighter())
    pipe.fit(X, y)

    row = pd.DataFrame({"f1": [np.nan], "f2": [1.0]})
    filled = pipe.fill_missing(row)
    assert filled["f1"].iloc[0] == pytest.approx(X["f1"].mean(), rel=1e-6), (
        "缺失值应填 fit-time 均值(标准化后=0,真正中性),而不是 0"
    )
    # 填均值 → 标准化后该因子贡献 ≈ 0 → 预测不被缺失因子方向性污染
    pred_filled = float(pipe.predict(filled).iloc[0])
    full_neutral = pd.DataFrame({"f1": [X["f1"].mean()], "f2": [1.0]})
    assert pred_filled == pytest.approx(float(pipe.predict(full_neutral).iloc[0]))


def test_mcap_neutralize_missing_mcap_is_nan():
    from stockpool.ml.preprocess import market_cap_neutralize_panel
    idx = pd.bdate_range("2025-01-06", periods=3)
    f = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [2.0, 2.0, 2.0],
                      "C": [3.0, 3.0, 3.0]}, index=idx)
    m = pd.DataFrame({"A": [10.0] * 3, "B": [11.0] * 3,
                      "C": [np.nan] * 3}, index=idx)
    out = market_cap_neutralize_panel(f, m)
    assert out["C"].isna().all(), "mcap 缺失的格子应为 NaN(残差与原始值不可混截面)"
    assert out["A"].notna().all()


def test_lasso_selection_invariant_to_y_scale():
    """P3-12:y 缩放 100×(高/低波动期)不应改变选中的因子集合。"""
    X, y_small = _xy(y_scale=0.01, seed=3)
    _, y_big = _xy(y_scale=1.0, seed=3)

    s1 = LassoSelector(alpha=0.02)
    s1.fit(X, y_small)
    s2 = LassoSelector(alpha=0.02)
    s2.fit(X, y_big)
    assert s1.selected_factors() == s2.selected_factors(), (
        f"选择不应随 y 量纲漂移: small={s1.selected_factors()} big={s2.selected_factors()}"
    )
    assert len(s1.selected_factors()) > 0


def test_pipeline_warns_on_empty_selection(caplog):
    import logging
    X, y = _xy(n=100, seed=7)
    pipe = TwoStepPipeline(selector=LassoSelector(alpha=1e9), weighter=ICWeighter())
    with caplog.at_level(logging.WARNING):
        info = pipe.fit(X, y)
    assert info.fallback_used
    assert any("回退" in r.message for r in caplog.records)
