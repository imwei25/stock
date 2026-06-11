import numpy as np
import pandas as pd
import pytest

from stockpool.indicators import (
    add_all,
    add_boll,
    add_breakout_markers,
    add_kdj,
    add_ma,
    add_macd,
    add_rsi,
    add_volume_ratio,
)


# ===== MA =====

def test_ma_basic(synthetic_daily):
    df = add_ma(synthetic_daily, periods=[5, 10, 20])
    expected_ma5_at_4 = synthetic_daily["close"].iloc[:5].mean()
    assert df["ma5"].iloc[4] == pytest.approx(expected_ma5_at_4)
    assert df["ma5"].iloc[:4].isna().all()
    assert "ma10" in df.columns
    assert "ma20" in df.columns


def test_ma_preserves_original_columns(synthetic_daily):
    df = add_ma(synthetic_daily, periods=[5])
    for col in ["date", "open", "high", "low", "close", "volume"]:
        assert col in df.columns


# ===== MACD =====

def test_macd_columns_present(synthetic_daily):
    df = add_macd(synthetic_daily, fast=12, slow=26, signal=9)
    assert {"macd_dif", "macd_dea", "macd_hist"}.issubset(df.columns)


def test_macd_values_match_textbook_formula(synthetic_daily):
    df = add_macd(synthetic_daily, fast=12, slow=26, signal=9)

    ema_fast = synthetic_daily["close"].ewm(span=12, adjust=False).mean()
    ema_slow = synthetic_daily["close"].ewm(span=26, adjust=False).mean()
    expected_dif = ema_fast - ema_slow
    expected_dea = expected_dif.ewm(span=9, adjust=False).mean()

    assert df["macd_dif"].iloc[-1] == pytest.approx(expected_dif.iloc[-1])
    assert df["macd_dea"].iloc[-1] == pytest.approx(expected_dea.iloc[-1])
    assert df["macd_hist"].iloc[-1] == pytest.approx(
        2 * (expected_dif.iloc[-1] - expected_dea.iloc[-1])
    )


# ===== KDJ =====

def test_kdj_columns_and_range(synthetic_daily):
    df = add_kdj(synthetic_daily, n=9, m1=3, m2=3)
    assert {"kdj_k", "kdj_d", "kdj_j"}.issubset(df.columns)
    valid = df.dropna(subset=["kdj_k", "kdj_d"])
    assert valid["kdj_k"].between(-50, 150).all()
    assert valid["kdj_d"].between(-50, 150).all()


def test_kdj_trending_up_pushes_k_high(synthetic_daily):
    df = add_kdj(synthetic_daily, n=9, m1=3, m2=3)
    assert df["kdj_k"].iloc[-1] > 70


# ===== RSI =====

def test_rsi_columns(synthetic_daily):
    df = add_rsi(synthetic_daily, periods=[6, 12, 24])
    assert {"rsi6", "rsi12", "rsi24"}.issubset(df.columns)


def test_rsi_monotonic_up_data_above_50(synthetic_daily):
    df = add_rsi(synthetic_daily, periods=[6])
    assert df["rsi6"].iloc[-1] > 50


def test_rsi_all_down_below_50():
    dates = pd.date_range("2026-01-02", periods=15, freq="B")
    close = np.linspace(20, 10, 15)
    df = pd.DataFrame({
        "date": dates, "open": close, "high": close + 0.1,
        "low": close - 0.1, "close": close, "volume": [1_000_000] * 15,
    })
    out = add_rsi(df, periods=[6])
    assert out["rsi6"].iloc[-1] < 50


# ===== BOLL =====

def test_boll_three_lines(synthetic_daily):
    df = add_boll(synthetic_daily, n=20, k=2)
    assert {"boll_up", "boll_mid", "boll_low"}.issubset(df.columns)


def test_boll_mid_equals_ma_n(synthetic_daily):
    df = add_boll(synthetic_daily, n=20, k=2)
    expected_mid = synthetic_daily["close"].rolling(20).mean()
    assert df["boll_mid"].iloc[-1] == pytest.approx(expected_mid.iloc[-1])


def test_boll_up_above_mid(synthetic_daily):
    df = add_boll(synthetic_daily, n=20, k=2)
    valid = df.dropna(subset=["boll_up", "boll_mid"])
    assert (valid["boll_up"] >= valid["boll_mid"]).all()
    assert (valid["boll_low"] <= valid["boll_mid"]).all()


# ===== Volume + Breakout + add_all =====

def test_volume_ratio(synthetic_daily):
    df = add_volume_ratio(synthetic_daily, window=5)
    assert "vol_ratio5" in df.columns
    assert df["vol_ratio5"].dropna().iloc[-1] == pytest.approx(1.0)


def test_breakout_markers(synthetic_daily):
    df = add_breakout_markers(synthetic_daily, window=20)
    assert df["is_breakout_high"].iloc[-1] == True
    assert df["is_breakout_low"].iloc[-1] == False


def test_add_all_runs_everything(synthetic_daily):
    from stockpool.config import IndicatorsConfig, MACDConfig, KDJConfig, BOLLConfig
    cfg = IndicatorsConfig(
        ma_periods=[5, 10, 20, 60],  # P1-8 校验要求含 5/20/60
        macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3),
        rsi_periods=[6, 12],
        boll=BOLLConfig(n=20, k=2),
        volume_ratio_window=5,
        breakout_window=20,
    )
    df = add_all(synthetic_daily, cfg)
    expected = {"ma5", "ma10", "ma20", "macd_dif", "macd_dea", "macd_hist",
                "kdj_k", "kdj_d", "kdj_j", "rsi6", "rsi12",
                "boll_up", "boll_mid", "boll_low", "vol_ratio5",
                "is_breakout_high", "is_breakout_low"}
    assert expected.issubset(df.columns)
