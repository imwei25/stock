import numpy as np
import pandas as pd
import pytest

from stockpool.config import ScoringConfig, VerdictsConfig, WeightsConfig
from stockpool.signals import (
    Trigger,
    combine_daily_weekly,
    detect_signals,
    score_triggers,
    verdict_of,
)


@pytest.fixture
def default_weights() -> WeightsConfig:
    return WeightsConfig(
        ma_cross_strong=2, ma_alignment=1,
        macd_cross_above_zero=2, macd_cross_below_zero=1, macd_histogram_expand=1,
        kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1,
        boll_band_touch=2, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1,
        breakout_new_high=2, breakout_new_low=2,
    )


def _make_df_with_macd_golden_cross_above_zero() -> pd.DataFrame:
    """Construct last two rows: DIF crosses from <DEA to >DEA, both above zero."""
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=3, freq="B"),
        "open": [10, 10, 10], "high": [10, 10, 10],
        "low": [10, 10, 10], "close": [10, 10, 10], "volume": [1e6] * 3,
        "ma5": [9, 9.5, 10], "ma10": [8.5, 9, 9.5],
        "ma20": [8, 8.5, 9], "ma60": [7, 7.5, 8],
        "macd_dif": [0.3, 0.5, 0.8], "macd_dea": [0.4, 0.6, 0.7], "macd_hist": [-0.2, -0.2, 0.2],
        "kdj_k": [50, 55, 60], "kdj_d": [50, 53, 56], "kdj_j": [50, 59, 68],
        "rsi6": [50, 55, 60], "rsi12": [50, 53, 56], "rsi24": [50, 52, 54],
        "boll_up": [11, 11, 11], "boll_mid": [10, 10, 10], "boll_low": [9, 9, 9],
        "vol_ratio5": [1.0, 1.0, 1.0],
        "is_breakout_high": [False, False, False],
        "is_breakout_low": [False, False, False],
    })


def test_macd_golden_cross_above_zero_detected(default_weights):
    df = _make_df_with_macd_golden_cross_above_zero()
    triggers = detect_signals(df, default_weights)
    sigs = [t.signal_type for t in triggers]
    assert "macd_cross_above_zero" in sigs
    assert "ma_alignment_bull" in sigs


def test_oversold_kdj_with_cross(default_weights):
    """J<20 + K crosses up D → strong signal."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=3, freq="B"),
        "open": [10] * 3, "high": [10] * 3, "low": [10] * 3, "close": [10] * 3, "volume": [1e6] * 3,
        "ma5": [10] * 3, "ma10": [10] * 3, "ma20": [10] * 3, "ma60": [10] * 3,
        "macd_dif": [0] * 3, "macd_dea": [0] * 3, "macd_hist": [0] * 3,
        "kdj_k": [10, 12, 18], "kdj_d": [15, 14, 13], "kdj_j": [5, 8, 18],
        "rsi6": [25] * 3, "rsi12": [40] * 3, "rsi24": [50] * 3,
        "boll_up": [11] * 3, "boll_mid": [10] * 3, "boll_low": [9] * 3,
        "vol_ratio5": [1.0] * 3,
        "is_breakout_high": [False] * 3, "is_breakout_low": [False] * 3,
    })
    triggers = detect_signals(df, default_weights)
    sigs = [t.signal_type for t in triggers]
    assert "kdj_oversold_cross" in sigs


def test_volume_surge_with_red_candle_is_bearish(default_weights):
    """High volume + bearish candle → bearish signal."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=2, freq="B"),
        "open":   [10, 11],
        "high":   [10, 11],
        "low":    [9, 9],
        "close":  [10, 9.5],
        "volume": [1e6, 2e6],
        "ma5": [10, 10], "ma10": [10, 10], "ma20": [10, 10], "ma60": [10, 10],
        "macd_dif": [0] * 2, "macd_dea": [0] * 2, "macd_hist": [0] * 2,
        "kdj_k": [50] * 2, "kdj_d": [50] * 2, "kdj_j": [50] * 2,
        "rsi6": [50] * 2, "rsi12": [50] * 2, "rsi24": [50] * 2,
        "boll_up": [12] * 2, "boll_mid": [10] * 2, "boll_low": [8] * 2,
        "vol_ratio5": [1.0, 2.0],
        "is_breakout_high": [False] * 2, "is_breakout_low": [False] * 2,
    })
    triggers = detect_signals(df, default_weights)
    sigs = [t.signal_type for t in triggers]
    assert "volume_surge_bearish" in sigs
    bearish_trigger = [t for t in triggers if t.signal_type == "volume_surge_bearish"][0]
    assert bearish_trigger.direction == -1


# ===== scoring =====

def _make_scoring() -> ScoringConfig:
    return ScoringConfig(
        daily_weight=0.7, weekly_weight=0.3,
        resonance_bonus=2, resonance_daily_threshold=3, resonance_weekly_threshold=1,
    )


def _make_verdicts() -> VerdictsConfig:
    return VerdictsConfig(strong_buy=6, buy=3, sell=-3, strong_sell=-6)


def test_score_triggers_sum_with_cap():
    triggers = [
        Trigger("a", +1, 2, ""), Trigger("b", +1, 2, ""),
        Trigger("c", +1, 2, ""), Trigger("d", +1, 2, ""),
        Trigger("e", +1, 2, ""), Trigger("f", +1, 2, ""),
    ]
    assert score_triggers(triggers) == 10


def test_score_triggers_mixed_signs():
    triggers = [Trigger("a", +1, 3, ""), Trigger("b", -1, 1, "")]
    assert score_triggers(triggers) == 2


def test_combine_no_resonance():
    cfg = _make_scoring()
    assert combine_daily_weekly(4, 0, cfg) == pytest.approx(2.8)


def test_combine_with_bullish_resonance():
    cfg = _make_scoring()
    assert combine_daily_weekly(5, 2, cfg) == pytest.approx(6.1)


def test_combine_with_bearish_resonance():
    cfg = _make_scoring()
    assert combine_daily_weekly(-4, -2, cfg) == pytest.approx(-5.4)


def test_combine_caps_at_10():
    cfg = _make_scoring()
    assert combine_daily_weekly(10, 10, cfg) == 10


def test_verdict_thresholds():
    v = _make_verdicts()
    assert verdict_of(7, v) == "strong_buy"
    assert verdict_of(5, v) == "buy"
    assert verdict_of(0, v) == "neutral"
    assert verdict_of(-4, v) == "sell"
    assert verdict_of(-7, v) == "strong_sell"
