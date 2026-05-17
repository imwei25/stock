"""Walk-forward composite-score backtest (A: bucket stats, B: equity curve).

Distinct from backtest.py, which computes per-signal hit rates. This module
reconstructs the full composite verdict (daily + weekly + resonance) for every
historical day, without future-data leakage.
"""
from __future__ import annotations

import pandas as pd

from stockpool.config import (
    IndicatorsConfig, ScoringConfig, VerdictsConfig, WeightsConfig,
)
from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.signals import (
    combine_daily_weekly, detect_signals, score_triggers, verdict_of,
)


_DAILY_WARMUP = 30  # match cli.py::_analyze_one's threshold


def walk_forward_verdicts(
    daily_df: pd.DataFrame,
    weights: WeightsConfig,
    scoring_cfg: ScoringConfig,
    verdicts_cfg: VerdictsConfig,
    indicators_cfg: IndicatorsConfig,
) -> pd.DataFrame:
    """For each daily bar (after warmup) compute the composite verdict that
    the live pipeline would have produced on data available at that bar.

    Returns a DataFrame with columns:
        date, close, daily_score, weekly_score, final_score, verdict
    """
    if len(daily_df) < _DAILY_WARMUP:
        return pd.DataFrame(columns=[
            "date", "close", "daily_score", "weekly_score", "final_score", "verdict",
        ])

    enriched_daily = add_all(daily_df, indicators_cfg)

    rows: list[dict] = []

    for i in range(_DAILY_WARMUP - 1, len(daily_df)):
        daily_window = enriched_daily.iloc[:i + 1]
        daily_triggers = detect_signals(daily_window, weights)
        daily_score = score_triggers(daily_triggers)

        weekly = resample_to_weekly(daily_df.iloc[:i + 1])
        if len(weekly) >= 30:
            enriched_w = add_all(weekly, indicators_cfg)
            weekly_score = score_triggers(detect_signals(enriched_w, weights))
        else:
            weekly_score = 0

        final_score = combine_daily_weekly(daily_score, weekly_score, scoring_cfg)
        verdict = verdict_of(final_score, verdicts_cfg)

        rows.append({
            "date": daily_df["date"].iloc[i],
            "close": float(daily_df["close"].iloc[i]),
            "daily_score": int(daily_score),
            "weekly_score": int(weekly_score),
            "final_score": float(final_score),
            "verdict": verdict,
        })

    return pd.DataFrame(rows)
