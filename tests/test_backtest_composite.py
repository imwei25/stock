"""Walk-forward composite verdict tests — guard against look-ahead bias."""
import numpy as np
import pandas as pd
import pytest

from stockpool.backtest_composite import walk_forward_verdicts
from stockpool.config import (
    BOLLConfig, IndicatorsConfig, KDJConfig, MACDConfig,
    ScoringConfig, VerdictsConfig, WeightsConfig,
)
from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.signals import (
    combine_daily_weekly, detect_signals, score_triggers, verdict_of,
)


@pytest.fixture
def weights() -> WeightsConfig:
    return WeightsConfig(
        ma_cross_strong=2, ma_alignment=1,
        macd_cross_above_zero=2, macd_cross_below_zero=1, macd_histogram_expand=1,
        kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1,
        boll_band_touch=2, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1,
        breakout_new_high=2, breakout_new_low=2,
    )


@pytest.fixture
def scoring() -> ScoringConfig:
    return ScoringConfig(
        daily_weight=0.7, weekly_weight=0.3,
        resonance_bonus=2, resonance_daily_threshold=3, resonance_weekly_threshold=1,
    )


@pytest.fixture
def verdicts_cfg() -> VerdictsConfig:
    return VerdictsConfig(strong_buy=6, buy=3, sell=-3, strong_sell=-6)


@pytest.fixture
def indicators_cfg() -> IndicatorsConfig:
    return IndicatorsConfig(
        ma_periods=[5, 10, 20, 60],
        macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3),
        rsi_periods=[6, 12, 24],
        boll=BOLLConfig(n=20, k=2.0),
        volume_ratio_window=5,
        breakout_window=20,
    )


def _synthetic_history(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """n trading days of pseudo-realistic OHLCV with embedded volatility."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
    })


def _live_verdict_at(daily, i, weights, scoring, verdicts_cfg, indicators_cfg):
    """Reproduce _analyze_one's verdict on daily.iloc[:i+1]."""
    sub_daily = daily.iloc[:i + 1].copy()
    enriched_d = add_all(sub_daily, indicators_cfg)
    daily_triggers = detect_signals(enriched_d, weights)
    daily_score = score_triggers(daily_triggers)

    weekly = resample_to_weekly(sub_daily)
    if len(weekly) >= 30:
        enriched_w = add_all(weekly, indicators_cfg)
        weekly_score = score_triggers(detect_signals(enriched_w, weights))
    else:
        weekly_score = 0

    final = combine_daily_weekly(daily_score, weekly_score, scoring)
    return verdict_of(final, verdicts_cfg), daily_score, weekly_score


def test_walk_forward_matches_live_at_final_bar(weights, scoring, verdicts_cfg, indicators_cfg):
    """The verdict at the last bar must equal the live pipeline's verdict on the full data."""
    daily = _synthetic_history(n=300)
    wf = walk_forward_verdicts(daily, weights, scoring, verdicts_cfg, indicators_cfg)

    assert len(wf) > 0, "walk-forward returned no rows"
    last = wf.iloc[-1]
    expected_verdict, expected_d, expected_w = _live_verdict_at(
        daily, len(daily) - 1, weights, scoring, verdicts_cfg, indicators_cfg
    )
    assert last["verdict"] == expected_verdict
    assert last["daily_score"] == expected_d
    assert last["weekly_score"] == expected_w


def test_walk_forward_matches_live_at_middle_bars(weights, scoring, verdicts_cfg, indicators_cfg):
    """For 20 random middle bars, walk-forward output must equal live pipeline."""
    daily = _synthetic_history(n=300, seed=7)
    wf = walk_forward_verdicts(daily, weights, scoring, verdicts_cfg, indicators_cfg)

    rng = np.random.default_rng(123)
    middle_indices = rng.choice(range(20, len(wf) - 20), size=20, replace=False)

    for k in middle_indices:
        daily_idx = 29 + int(k)
        expected_verdict, expected_d, expected_w = _live_verdict_at(
            daily, daily_idx, weights, scoring, verdicts_cfg, indicators_cfg
        )
        row = wf.iloc[int(k)]
        assert row["verdict"] == expected_verdict, f"verdict mismatch at k={k}, daily_idx={daily_idx}"
        assert row["daily_score"] == expected_d
        assert row["weekly_score"] == expected_w


def test_walk_forward_handles_short_history(weights, scoring, verdicts_cfg, indicators_cfg):
    """Less than 30 daily bars returns an empty DataFrame."""
    daily = _synthetic_history(n=20)
    wf = walk_forward_verdicts(daily, weights, scoring, verdicts_cfg, indicators_cfg)
    assert len(wf) == 0
    assert list(wf.columns) == [
        "date", "close", "daily_score", "weekly_score", "final_score", "verdict",
    ]


def test_walk_forward_weekly_score_zero_when_insufficient_weekly_bars(
    weights, scoring, verdicts_cfg, indicators_cfg
):
    """When weekly bars < 30, weekly_score must be 0 (matches _analyze_one)."""
    daily = _synthetic_history(n=50)  # ~10 weeks → too few
    wf = walk_forward_verdicts(daily, weights, scoring, verdicts_cfg, indicators_cfg)
    assert len(wf) > 0
    assert (wf["weekly_score"] == 0).all()


from stockpool.backtest_composite import verdict_bucket_stats


def _wf_from_verdicts(verdicts: list[str], closes: list[float]) -> pd.DataFrame:
    """Build a synthetic walk-forward DataFrame from manually-set verdicts."""
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=len(verdicts), freq="B"),
        "close": closes,
        "daily_score": [0] * len(verdicts),
        "weekly_score": [0] * len(verdicts),
        "final_score": [0.0] * len(verdicts),
        "verdict": verdicts,
    })


def test_verdict_bucket_stats_counts():
    wf = _wf_from_verdicts(
        ["buy", "buy", "neutral", "sell", "buy", "neutral", "strong_buy", "strong_sell", "neutral", "neutral"],
        [100, 102, 103, 104, 100, 105, 110, 108, 105, 106],
    )
    stats = verdict_bucket_stats(wf, forward_days=[2])

    assert stats["buy"]["count"] == 3
    assert stats["neutral"]["count"] == 4
    assert stats["sell"]["count"] == 1
    assert stats["strong_buy"]["count"] == 1
    assert stats["strong_sell"]["count"] == 1


def test_verdict_bucket_stats_forward_return_and_win_rate():
    """buy at idx 0 (close 100), idx 1 (close 102), idx 4 (close 100).
    Forward 2 returns: idx 0 → close[2]=103 → +3.0%; idx 1 → close[3]=104 → +1.96%;
    idx 4 → close[6]=110 → +10.0%.
    All positive → win_rate 1.0 (buy wins on positive return).
    Mean ≈ (3.0 + 1.96 + 10.0) / 3 ≈ 4.99%
    """
    wf = _wf_from_verdicts(
        ["buy", "buy", "neutral", "sell", "buy", "neutral", "strong_buy", "strong_sell", "neutral", "neutral"],
        [100, 102, 103, 104, 100, 105, 110, 108, 105, 106],
    )
    stats = verdict_bucket_stats(wf, forward_days=[2])
    buy = stats["buy"]["forward_2"]
    assert buy["sample_size"] == 3
    assert buy["mean_return_pct"] == pytest.approx((3.0 + (104/102 - 1) * 100 + 10.0) / 3, rel=1e-4)
    assert buy["win_rate"] == 1.0


def test_verdict_bucket_stats_sell_win_rate_direction():
    """sell at idx 0 (close 100), close[2]=95 → -5% → win for sell (negative is good)."""
    wf = _wf_from_verdicts(
        ["sell", "neutral", "neutral", "neutral"],
        [100, 99, 95, 96],
    )
    stats = verdict_bucket_stats(wf, forward_days=[2])
    assert stats["sell"]["forward_2"]["win_rate"] == 1.0
    assert stats["sell"]["forward_2"]["mean_return_pct"] == pytest.approx(-5.0, rel=1e-4)


def test_verdict_bucket_stats_omits_out_of_range_forward():
    """Last 2 rows can't have forward_2; sample_size reflects that."""
    wf = _wf_from_verdicts(["buy", "buy", "buy"], [100, 101, 102])
    stats = verdict_bucket_stats(wf, forward_days=[2])
    # Only idx 0 has close[2]=102 available; idx 1 and 2 are out of range.
    assert stats["buy"]["forward_2"]["sample_size"] == 1
