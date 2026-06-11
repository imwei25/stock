"""D1 / P2-24/25, P1-8:配置防呆收口。"""
import pytest
from pydantic import ValidationError

from stockpool.config import (
    BOLLConfig, IndicatorsConfig, KDJConfig, MACDConfig,
    ScoringConfig, VerdictsConfig,
)


def _indicators(**overrides):
    base = dict(
        ma_periods=[5, 20, 60],
        macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3),
        rsi_periods=[6, 12, 24],
        boll=BOLLConfig(n=20, k=2.0),
        volume_ratio_window=5,
        breakout_window=20,
    )
    base.update(overrides)
    return IndicatorsConfig(**base)


def test_verdicts_ordering_enforced():
    with pytest.raises(ValidationError, match="strong_sell < sell < 0 < buy < strong_buy"):
        VerdictsConfig(strong_buy=3, buy=6, sell=-3, strong_sell=-6)
    VerdictsConfig(strong_buy=6, buy=3, sell=-3, strong_sell=-6)  # 合法


def test_scoring_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="daily_weight \\+ weekly_weight"):
        ScoringConfig(daily_weight=0.6, weekly_weight=0.5, resonance_bonus=2,
                      resonance_daily_threshold=3, resonance_weekly_threshold=1)
    ScoringConfig(daily_weight=0.7, weekly_weight=0.3, resonance_bonus=2,
                  resonance_daily_threshold=3, resonance_weekly_threshold=1)


def test_indicators_require_signal_periods():
    """P1-8:detect_signals 依赖 ma5/ma20/ma60、rsi6、vol_ratio5;缺了
    旧实现会静默回退默认值,信号无声消失。现在 fail loud。"""
    with pytest.raises(ValidationError, match="ma_periods"):
        _indicators(ma_periods=[10, 30])
    with pytest.raises(ValidationError, match="rsi_periods"):
        _indicators(rsi_periods=[14])
    with pytest.raises(ValidationError, match="volume_ratio_window"):
        _indicators(volume_ratio_window=10)
    # 额外周期允许(只增不减)
    _indicators(ma_periods=[5, 10, 20, 60], rsi_periods=[6, 14])


def test_top_level_typo_rejected():
    """P2-25:顶层 Config 设 extra=forbid,yaml 写错键名不再静默走默认值。"""
    with pytest.raises(ValidationError):
        VerdictsConfig(strong_buy=6, buy=3, sell=-3, strong_sell=-6,
                       strong_byu=9)  # typo
    with pytest.raises(ValidationError):
        _indicators(breakout_windw=20)  # typo
