"""Lock in the bit-exact equivalence of the fast weekly-score path.

``composite_weekly.weekly_scores_by_bar`` replaces the old O(T^2) per-bar
``resample_to_weekly`` + ``add_all`` loop. This test asserts it produces
*identical* weekly scores to that slow reference across several synthetic
histories (including the partial-week / week-boundary edge cases), plus the
``< WEEKLY_WARMUP`` short-history degenerate case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.composite_weekly import weekly_scores_by_bar
from stockpool.backtesting.strategies import DAILY_WARMUP, WEEKLY_WARMUP
from stockpool.config import (
    BOLLConfig, IndicatorsConfig, KDJConfig, MACDConfig, WeightsConfig,
)
from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.signals import detect_signals, score_triggers


def _synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-02", periods=n, freq="B"),
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
    })


def _slow_weekly_scores(daily_df, indicators_cfg, weights, start, weekly_warmup):
    out = {}
    for i in range(start, len(daily_df)):
        weekly = resample_to_weekly(daily_df.iloc[:i + 1])
        if len(weekly) >= weekly_warmup:
            ew = add_all(weekly, indicators_cfg)
            out[i] = score_triggers(detect_signals(ew, weights))
        else:
            out[i] = 0
    return out


@pytest.fixture
def indicators_cfg():
    return IndicatorsConfig(
        ma_periods=[5, 10, 20, 60],
        macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3),
        rsi_periods=[6, 12, 24],
        boll=BOLLConfig(n=20, k=2.0),
        volume_ratio_window=5,
        breakout_window=20,
    )


@pytest.fixture
def weights():
    return WeightsConfig(
        ma_cross_strong=2, ma_alignment=1,
        macd_cross_above_zero=2, macd_cross_below_zero=1, macd_histogram_expand=1,
        kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1,
        boll_band_touch=2, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1,
        breakout_new_high=2, breakout_new_low=2,
    )


@pytest.mark.parametrize("seed", [0, 7, 42, 123])
def test_fast_matches_slow_weekly_scores(seed, indicators_cfg, weights):
    daily = _synthetic(n=400, seed=seed)
    start = DAILY_WARMUP - 1
    fast = weekly_scores_by_bar(daily, indicators_cfg, weights, start, WEEKLY_WARMUP)
    slow = _slow_weekly_scores(daily, indicators_cfg, weights, start, WEEKLY_WARMUP)
    assert fast.keys() == slow.keys()
    mismatches = {i: (slow[i], fast[i]) for i in slow if slow[i] != fast[i]}
    assert not mismatches, f"weekly score mismatches (i -> (slow, fast)): {mismatches}"


def test_short_history_all_zero(indicators_cfg, weights):
    """< WEEKLY_WARMUP weekly bars -> every weekly score is 0 (matches slow)."""
    daily = _synthetic(n=60, seed=1)  # ~12 weeks < 30
    start = DAILY_WARMUP - 1
    fast = weekly_scores_by_bar(daily, indicators_cfg, weights, start, WEEKLY_WARMUP)
    assert fast, "expected some bars"
    assert all(v == 0 for v in fast.values())


def test_generate_signals_matches_live_pipeline(indicators_cfg, weights):
    """End-to-end: CompositeVerdictStrategy.generate_signals must still equal the
    bar-by-bar live pipeline (the fast weekly path is wired into it)."""
    from stockpool.backtesting.strategies import CompositeVerdictStrategy
    from stockpool.config import ScoringConfig, VerdictsConfig
    from stockpool.signals import combine_daily_weekly, verdict_of

    scoring = ScoringConfig(
        daily_weight=0.7, weekly_weight=0.3,
        resonance_bonus=2, resonance_daily_threshold=3, resonance_weekly_threshold=1,
    )
    verdicts_cfg = VerdictsConfig(strong_buy=6, buy=3, sell=-3, strong_sell=-6)
    daily = _synthetic(n=400, seed=11)
    strat = CompositeVerdictStrategy(weights, scoring, verdicts_cfg, indicators_cfg)
    wf = strat.generate_signals(daily)

    start = DAILY_WARMUP - 1
    slow_weekly = _slow_weekly_scores(daily, indicators_cfg, weights, start, WEEKLY_WARMUP)
    for row_pos, i in enumerate(range(start, len(daily))):
        enriched_d = add_all(daily.iloc[:i + 1], indicators_cfg)
        d_score = score_triggers(detect_signals(enriched_d, weights))
        final = combine_daily_weekly(d_score, slow_weekly[i], scoring)
        expected = verdict_of(final, verdicts_cfg)
        assert wf.iloc[row_pos]["signal"] == expected
        assert wf.iloc[row_pos]["weekly_score"] == slow_weekly[i]
