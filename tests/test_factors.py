"""Factor library + registry tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.factors import Factor, list_factors, make_factor


def _ohlcv(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "open": [c * 0.998 for c in closes],
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1_000_000.0] * n,
    })


# === registry ===

def test_list_factors_has_builtins():
    names = list_factors()
    for expected in ("momentum", "macd_hist", "rsi_centered", "ma_distance",
                     "ma_slope", "vol_ratio", "boll_position", "kdj_j",
                     "hl_range", "macd_dif_norm"):
        assert expected in names, f"missing factor: {expected}"


def test_make_factor_exact_match_uses_defaults():
    f = make_factor("macd_hist")
    assert f.name == "macd_hist"


def test_make_factor_with_suffix_args():
    f = make_factor("momentum_20")
    assert f.name == "momentum_20"
    assert f.n == 20


def test_make_factor_with_multi_suffix():
    f = make_factor("ma_slope_20_5")
    assert f.name == "ma_slope_20_5"
    assert f.n == 20 and f.k == 5


def test_make_factor_unknown_raises():
    with pytest.raises(KeyError):
        make_factor("not_a_real_factor")


# === look-ahead safety ===

def test_factor_value_at_t_depends_only_on_past():
    """Computing the factor on df[:k+1] must match df[:N] truncated to k."""
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
              110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
              120, 121, 122, 123, 124, 125]
    df = _ohlcv(closes)

    # Pick representative parametric and zero-arg factors.
    factor_names = [
        "momentum_5", "rsi_centered_6", "ma_distance_10",
        "macd_hist", "boll_position_20", "kdj_j", "vol_ratio_5",
    ]
    for name in factor_names:
        f = make_factor(name)
        full = f.compute(df).reset_index(drop=True)
        for k in (10, 15, 20, len(df) - 1):
            partial = f.compute(df.iloc[: k + 1]).reset_index(drop=True)
            for i in range(k + 1):
                a, b = full.iloc[i], partial.iloc[i]
                if pd.isna(a) and pd.isna(b):
                    continue
                assert a == pytest.approx(b, abs=1e-10), (
                    f"{name}: row {i} differs at truncation k={k}: "
                    f"full={a}, partial={b}"
                )


# === per-factor correctness on hand-checked values ===

def test_momentum_formula():
    closes = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    df = _ohlcv(closes)
    f = make_factor("momentum_3")
    s = f.compute(df).reset_index(drop=True)
    # row 3: 106/100 - 1 = 0.06
    assert s.iloc[3] == pytest.approx(0.06)
    # row 5: 110/104 - 1 ≈ 0.0577
    assert s.iloc[5] == pytest.approx(110 / 104 - 1)
    # first 3 rows NaN
    assert pd.isna(s.iloc[0]) and pd.isna(s.iloc[2])


def test_ma_distance_zero_when_flat():
    df = _ohlcv([100.0] * 30)
    f = make_factor("ma_distance_10")
    s = f.compute(df).reset_index(drop=True)
    # After warmup, distance to flat MA is 0.
    assert s.iloc[-1] == pytest.approx(0.0)


def test_rsi_centered_sign():
    # Noisy uptrend → RSI > 50 → centered > 0. The noise must be loud enough
    # to produce at least a few down days: a purely monotonic series has
    # zero losses and the indicator collapses RSI to 50 via its NaN fill.
    rng = np.random.default_rng(0)
    drift = np.linspace(0, 30, 30)
    closes = list(100 + drift + rng.normal(0, 3.0, 30))
    df = _ohlcv(closes)
    f = make_factor("rsi_centered_6")
    s = f.compute(df).reset_index(drop=True)
    assert s.iloc[-1] > 0


def test_vol_ratio_centered_at_zero():
    df = _ohlcv([100.0] * 30)
    df["volume"] = 1_000_000.0
    f = make_factor("vol_ratio_5")
    s = f.compute(df).reset_index(drop=True)
    # Constant volume → ratio = 1 → factor = 0
    assert s.iloc[-1] == pytest.approx(0.0)


def test_boll_position_inside_bands():
    # Random walk; check that |position| stays modest for most bars.
    rng = np.random.default_rng(0)
    closes = list(100 + np.cumsum(rng.normal(0, 0.5, 100)))
    df = _ohlcv(closes)
    f = make_factor("boll_position_20")
    s = f.compute(df).dropna()
    # 95% of values should fall within roughly [-2, 2] (= 2σ from mid).
    inside = (s.abs() < 2.5).mean()
    assert inside > 0.85


# === Factor ABC contract ===

def test_factor_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        Factor()  # type: ignore[abstract]


def test_factor_does_not_mutate_input():
    df = _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    snapshot = df.copy()
    f = make_factor("momentum_3")
    _ = f.compute(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_factor_count_in_expected_range():
    """所有新因子族落地后,总数应在合理范围内。
    防止漏注册或意外重名。

    Task 14 实测: 注册了 165 个基础因子名(builtin/wq101/custom + 11 个新族)。
    list_specs() 返回每个 @register 一条,不展开 variant 后缀。
    range 取实测值 ± 缓冲,作为漏注册保护。

    2026-06-24: 上限放宽到 320 以容纳两批新增 ——
      * GTJA191 A 股因子族(验证子集,25 个);
      * 可选生成的 WQ101 本土化变体(``wq101_variants.py``,top-30×3=90 个,
        文件存在才注册,故计数会在 ~192(无变体)与 ~282(有变体)间浮动)。"""
    from stockpool.factors import list_specs
    n = len(list_specs())
    assert 140 <= n <= 400, f"factor count={n} outside expected [140, 400]"


def test_new_type_fundamental_registered():
    """新引入的 type 标签 'fundamental' 应出现在 all_types 里。"""
    from stockpool.factors import all_types
    assert "fundamental" in all_types()
