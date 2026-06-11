"""Tests for stockpool.factors.ops NaN-safety after mask-first refactor."""
import numpy as np
import pandas as pd
import pytest


def test_ts_mean_full_valid_input_unchanged():
    from stockpool.factors.ops import ts_mean
    x = pd.DataFrame({"A": np.arange(30, dtype=float)})
    out = ts_mean(x, 10)
    assert out["A"].iloc[:5].isna().all()
    assert out["A"].iloc[9] == pytest.approx(4.5)
    assert out["A"].iloc[29] == pytest.approx(24.5)


def test_ts_mean_with_nan_in_window_uses_valid():
    from stockpool.factors.ops import ts_mean
    vals = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    x = pd.DataFrame({"A": vals})
    out = ts_mean(x, 10)
    assert out["A"].iloc[9] == pytest.approx(52.0 / 9.0)


def test_ts_std_with_nan_in_window():
    from stockpool.factors.ops import ts_std
    vals = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    x = pd.DataFrame({"A": vals})
    out = ts_std(x, 10)
    assert not np.isnan(out["A"].iloc[9])


def test_ts_sum_with_nan_in_window():
    from stockpool.factors.ops import ts_sum
    vals = [1.0, 2.0, np.nan, 4.0, 5.0]
    x = pd.DataFrame({"A": vals})
    out = ts_sum(x, 5)
    # P3-5 重标定:mean(1,2,4,5) × 5 = 15(部分和按 d/count 放大,保持量纲)
    assert out["A"].iloc[4] == pytest.approx(15.0)


def test_ts_mean_too_few_valid_returns_nan():
    from stockpool.factors.ops import ts_mean
    vals = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0]
    x = pd.DataFrame({"A": vals})
    out = ts_mean(x, 10)
    assert np.isnan(out["A"].iloc[9])


def test_decay_linear_full_valid_unchanged():
    from stockpool.factors.ops import decay_linear
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = decay_linear(x, 5)
    # 加权: (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15 = 55/15
    assert out["A"].iloc[4] == pytest.approx(55.0 / 15.0)


def test_decay_linear_with_nan_renormalizes():
    from stockpool.factors.ops import decay_linear
    # 窗口 [1, nan, 3, 4, 5], 权重 [1, 2, 3, 4, 5]
    # Valid vals [1, 3, 4, 5], weights [1, 3, 4, 5]
    # 加权和 / 权重和 = (1+9+16+25)/(1+3+4+5) = 51/13
    vals = [1.0, np.nan, 3.0, 4.0, 5.0]
    x = pd.DataFrame({"A": vals})
    out = decay_linear(x, 5)
    assert out["A"].iloc[4] == pytest.approx(51.0 / 13.0)


def test_decay_linear_all_nan_returns_nan():
    from stockpool.factors.ops import decay_linear
    x = pd.DataFrame({"A": [np.nan] * 5})
    out = decay_linear(x, 5)
    assert np.isnan(out["A"].iloc[4])


def test_ts_product_full_valid_unchanged():
    from stockpool.factors.ops import ts_product
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    out = ts_product(x, 4)
    assert out["A"].iloc[3] == pytest.approx(24.0)


def test_ts_product_nan_window_is_nan():
    from stockpool.factors.ops import ts_product
    x = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0]})
    out = ts_product(x, 4)
    # P3-5:product 要求满窗 —— 部分积无法线性重标定,缺值即 NaN
    assert np.isnan(out["A"].iloc[3])

    full = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    assert ts_product(full, 4)["A"].iloc[3] == pytest.approx(24.0)
