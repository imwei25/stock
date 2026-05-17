"""Walk-forward composite-score backtest (A: bucket stats, B: equity curve).

Distinct from backtest.py, which computes per-signal hit rates. This module
reconstructs the full composite verdict (daily + weekly + resonance) for every
historical day, without future-data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def _iso_week_key(ts) -> tuple[int, int]:
    iso = pd.Timestamp(ts).isocalendar()
    return (int(iso.year), int(iso.week))


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
    cached_week_key: tuple[int, int] | None = None
    cached_weekly_score: int = 0

    for i in range(_DAILY_WARMUP - 1, len(daily_df)):
        # Daily score: slice up to bar i (indicators precomputed left-to-right,
        # so bar i depends only on bars ≤ i — no future leakage).
        # detect_signals reads only iloc[-2], iloc[-1], iloc[-3:] from this slice.
        daily_window = enriched_daily.iloc[:i + 1]
        daily_triggers = detect_signals(daily_window, weights)
        daily_score = score_triggers(daily_triggers)

        # Weekly score: cache by ISO week to avoid redundant resampling,
        # but always recompute on a new week (week boundary forces re-aggregation)
        week_key = _iso_week_key(daily_df["date"].iloc[i])
        if cached_week_key == week_key:
            weekly_score = cached_weekly_score
        else:
            # Always slice daily_df.iloc[:i+1] so the most-recent partial week
            # is not contaminated by future days
            weekly = resample_to_weekly(daily_df.iloc[:i + 1])
            if len(weekly) >= 30:
                enriched_w = add_all(weekly, indicators_cfg)
                weekly_score = score_triggers(detect_signals(enriched_w, weights))
            else:
                weekly_score = 0
            cached_week_key = week_key
            cached_weekly_score = weekly_score

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
