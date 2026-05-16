import numpy as np
import pandas as pd
import pytest

from stockpool.backtest import compute_hit_rates
from stockpool.config import WeightsConfig


@pytest.fixture
def weights() -> WeightsConfig:
    return WeightsConfig(
        ma_cross_strong=2, ma_alignment=1, macd_cross_above_zero=2, macd_cross_below_zero=1,
        macd_histogram_expand=1, kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1, boll_band_touch=2, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1, breakout_new_high=2, breakout_new_low=2,
    )


def _make_history_with_planted_breakouts() -> pd.DataFrame:
    """30 days: day 20 is a planted 20-day new high (prior max 100, close 110);
       then days 21-29 each +1%."""
    n = 30
    close = np.full(n, 100.0)
    close[20] = 110.0
    for i in range(21, n):
        close[i] = close[i - 1] * 1.01

    df = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=n, freq="B"),
        "open": close - 0.1, "high": close + 0.1, "low": close - 0.2,
        "close": close, "volume": [1e6] * n,
        "ma5": close, "ma10": close, "ma20": close, "ma60": close,
        "macd_dif": np.zeros(n), "macd_dea": np.zeros(n), "macd_hist": np.zeros(n),
        "kdj_k": np.full(n, 50.0), "kdj_d": np.full(n, 50.0), "kdj_j": np.full(n, 50.0),
        "rsi6": np.full(n, 50.0), "rsi12": np.full(n, 50.0), "rsi24": np.full(n, 50.0),
        "boll_up": close + 1, "boll_mid": close, "boll_low": close - 1,
        "vol_ratio5": np.ones(n),
        "is_breakout_high": [False] * 20 + [True] + [False] * 9,
        "is_breakout_low": [False] * n,
    })
    return df


def test_hit_rate_breakout_high_5d(weights):
    df = _make_history_with_planted_breakouts()
    stats = compute_hit_rates(df, weights, forward_days=[5, 10, 20])

    assert "breakout_new_high" in stats
    s = stats["breakout_new_high"]
    assert s["count"] == 1
    expected_5d = (110 * 1.01 ** 5 / 110 - 1) * 100
    assert s["forward_5"]["mean_return_pct"] == pytest.approx(expected_5d, rel=1e-3)
    assert s["forward_5"]["win_rate"] == 1.0
    assert s["direction"] == +1


def test_no_signals_returns_empty(weights):
    n = 30
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=n, freq="B"),
        "open": close, "high": close, "low": close,
        "close": close, "volume": [1e6] * n,
        "ma5": close, "ma10": close, "ma20": close, "ma60": close,
        "macd_dif": np.zeros(n), "macd_dea": np.zeros(n), "macd_hist": np.zeros(n),
        "kdj_k": np.full(n, 50.0), "kdj_d": np.full(n, 50.0), "kdj_j": np.full(n, 50.0),
        "rsi6": np.full(n, 50.0), "rsi12": np.full(n, 50.0), "rsi24": np.full(n, 50.0),
        "boll_up": close + 1, "boll_mid": close, "boll_low": close - 1,
        "vol_ratio5": np.ones(n),
        "is_breakout_high": [False] * n, "is_breakout_low": [False] * n,
    })
    stats = compute_hit_rates(df, weights, forward_days=[5])
    assert stats == {}
