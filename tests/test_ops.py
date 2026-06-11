"""WQ ops 算子库测试。

主要验证:
  * 时间序列算子的语义和窗口
  * 横截面 rank 沿 axis=1
  * indneutralize 分组 demean
  * NaN 与零分母安全
  * look-ahead 安全(第 i 行不依赖 i+1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.factors import ops


def _panel(T: int = 20, N: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=T, freq="B")
    codes = [f"c{i}" for i in range(N)]
    close = pd.DataFrame(
        100 + np.cumsum(rng.standard_normal((T, N)) * 0.5, axis=0),
        index=dates, columns=codes,
    )
    return close


# ─────────────────────────────────────────────────────────────────────────────
# 时间序列
# ─────────────────────────────────────────────────────────────────────────────

def test_delay_and_delta():
    x = _panel(5, 2)
    assert ops.delay(x, 1).iloc[0].isna().all()
    assert ops.delay(x, 1).iloc[1].equals(x.iloc[0].rename(x.index[1]))
    d = ops.delta(x, 2)
    assert np.allclose(d.iloc[2].values, (x.iloc[2] - x.iloc[0]).values)


def test_ts_sum_and_mean():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    s = ops.ts_sum(x, 3)
    assert s["a"].iloc[2] == 6.0  # 1+2+3
    m = ops.ts_mean(x, 3)
    assert m["a"].iloc[2] == 2.0
    # P3-5: 部分窗口按 d/count 重标定(= mean × d),保持与满窗同量纲,
    # 否则历史短的列在截面 rank 里被结构性压低
    assert s["a"].iloc[0] == pytest.approx(1.0 * 3)      # mean(1)×3
    assert s["a"].iloc[1] == pytest.approx(1.5 * 3)      # mean(1,2)×3


def test_ts_min_max_argmax_argmin():
    x = pd.DataFrame({"a": [3.0, 1.0, 4.0, 1.0, 5.0]})
    assert ops.ts_min(x, 3)["a"].iloc[4] == 1.0
    assert ops.ts_max(x, 3)["a"].iloc[4] == 5.0
    # argmax 在过去 3 期(idx 2,3,4)中最大值是 idx4 → 距今 0
    assert ops.ts_argmax(x, 3)["a"].iloc[4] == 0.0
    # argmin idx3 是 1.0(最小,平局取第一个)→ 距今 1
    assert ops.ts_argmin(x, 3)["a"].iloc[4] == 1.0


def test_ts_rank():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    # 当前值在过去 d 个里排第 d/d (最大) = 1.0
    r = ops.ts_rank(x, 3)["a"]
    assert r.iloc[2] == 1.0
    assert r.iloc[3] == 1.0
    # 中间值
    x2 = pd.DataFrame({"a": [3.0, 1.0, 2.0]})
    assert ops.ts_rank(x2, 3)["a"].iloc[2] == pytest.approx(2 / 3)


def test_decay_linear_weights_normalize():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = ops.decay_linear(x, 3)["a"].iloc[2]
    # 权重 (1,2,3)/6 → (1*1 + 2*2 + 3*3)/6 = 14/6
    assert out == pytest.approx(14.0 / 6.0)


def test_correlation_two_series():
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    b = pd.DataFrame({"x": [2.0, 4.0, 6.0, 8.0, 10.0]})
    c = ops.correlation(a, b, 4)
    assert c["x"].iloc[4] == pytest.approx(1.0)


def test_lookahead_safety_via_truncation():
    """同样的 panel,截断到前 k 行算因子,值与全长版本的前 k 行相同。"""
    x = _panel(30, 3, seed=7)
    full = ops.ts_mean(x, 5)
    trunc = ops.ts_mean(x.iloc[:15], 5)
    pd.testing.assert_frame_equal(full.iloc[:15], trunc)


# ─────────────────────────────────────────────────────────────────────────────
# 横截面
# ─────────────────────────────────────────────────────────────────────────────

def test_rank_axis_1():
    x = pd.DataFrame({"a": [3.0, 1.0], "b": [1.0, 3.0], "c": [2.0, 2.0]})
    r = ops.rank(x)
    # 第一行: a=3 -> rank 3, b=1 -> 1, c=2 -> 2 → pct: 1.0, 0.333, 0.667
    assert r["a"].iloc[0] == pytest.approx(1.0)
    assert r["b"].iloc[0] == pytest.approx(1.0 / 3.0)


def test_scale_l1_normalize():
    x = pd.DataFrame({"a": [1.0, -2.0], "b": [3.0, 4.0]})
    s = ops.scale(x)
    # 第一行: |1|+|3|=4 → a=1/4, b=3/4
    assert s["a"].iloc[0] == pytest.approx(0.25)
    assert s["b"].iloc[0] == pytest.approx(0.75)


def test_signedpower():
    x = pd.DataFrame({"a": [-2.0, 3.0]})
    out = ops.signedpower(x, 2)
    assert out["a"].iloc[0] == -4.0  # -1 * 4
    assert out["a"].iloc[1] == 9.0


def test_indneutralize_demeans_within_group():
    x = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [10.0]})
    # a,b 同组,c 单独
    out = ops.indneutralize(x, {"a": "g1", "b": "g1", "c": "g2"})
    # g1 mean=1.5 → a=-0.5, b=0.5; g2 mean=10 → c=0
    assert out["a"].iloc[0] == -0.5
    assert out["b"].iloc[0] == 0.5
    assert out["c"].iloc[0] == 0.0


def test_indneutralize_unmapped_codes_become_nan():
    """P3-7:无行业映射的 code 输出 NaN(旧实现独立成组 → 恒 0,
    以"完美中性"的假值参与后续 rank,静默且危险)。"""
    x = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out = ops.indneutralize(x, {})
    assert out.isna().all().all()

    # 部分映射:有映射的正常 demean,无映射的 NaN
    x2 = pd.DataFrame({"a": [1.0], "b": [3.0], "c": [9.0]})
    out2 = ops.indneutralize(x2, {"a": "g1", "b": "g1"})
    assert out2["a"].iloc[0] == -1.0
    assert out2["b"].iloc[0] == 1.0
    assert pd.isna(out2["c"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_div_zero_to_nan():
    a = pd.DataFrame({"x": [1.0, 2.0]})
    b = pd.DataFrame({"x": [2.0, 0.0]})
    out = ops.safe_div(a, b)
    assert out["x"].iloc[0] == 0.5
    assert pd.isna(out["x"].iloc[1])


def test_vwap_proxy():
    panel = {
        "high": pd.DataFrame({"a": [10.0]}),
        "low": pd.DataFrame({"a": [4.0]}),
        "close": pd.DataFrame({"a": [7.0]}),
    }
    assert ops.vwap(panel)["a"].iloc[0] == 7.0  # (10+4+7)/3
