"""训练矩阵的非有限值防线。

2026-06 事故:alpha_045(rolling(2).corr)在平盘日产出 ±inf,而
``stack_panel_to_xy`` / ``align_xy`` 的样本过滤只检查 ``isnan``,inf 直进
训练矩阵 → Lasso 标准化整列 NaN → 残差污染 → 全部系数 NaN → 模型死亡。
本文件锁定:任何含 ±inf 的样本行与 NaN 同等待遇 —— 整行剔除。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.ml.dataset import align_xy, stack_panel_to_xy


def _wide(dates, codes, value):
    return pd.DataFrame(value, index=dates, columns=codes)


def test_stack_panel_to_xy_drops_inf_factor_rows():
    dates = pd.date_range("2025-01-01", periods=4, freq="B")
    codes = ["A", "B"]
    f1 = _wide(dates, codes, 1.0)
    f1.loc[dates[1], "A"] = np.inf
    f2 = _wide(dates, codes, 2.0)
    fwd = _wide(dates, codes, 0.01)

    X, y = stack_panel_to_xy({"f1": f1, "f2": f2}, fwd, dropna=True)

    assert np.isfinite(X.to_numpy()).all()
    assert ("A", dates[1]) not in X.index
    # 其它样本不受牵连
    assert ("B", dates[1]) in X.index
    assert len(X) == 7


def test_stack_panel_to_xy_drops_inf_label_rows():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    codes = ["A", "B"]
    fp = {"f1": _wide(dates, codes, 1.0)}
    fwd = _wide(dates, codes, 0.01)
    fwd.loc[dates[2], "B"] = -np.inf

    X, y = stack_panel_to_xy(fp, fwd, dropna=True)

    assert np.isfinite(y.to_numpy()).all()
    assert ("B", dates[2]) not in y.index


def test_align_xy_drops_inf_rows():
    idx = pd.RangeIndex(4)
    X = pd.DataFrame({"f1": [1.0, np.inf, 3.0, 4.0], "f2": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    y = pd.Series([0.01, 0.02, -np.inf, 0.04], index=idx)

    Xa, ya = align_xy(X, y)

    assert list(Xa.index) == [0, 3]
    assert np.isfinite(Xa.to_numpy()).all()
    assert np.isfinite(ya.to_numpy()).all()
