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


_VERDICT_LABELS = ("strong_buy", "buy", "neutral", "sell", "strong_sell")
_BULL_VERDICTS = {"strong_buy", "buy"}
_BEAR_VERDICTS = {"strong_sell", "sell"}


def verdict_bucket_stats(
    wf_df: pd.DataFrame, forward_days: list[int]
) -> dict[str, dict]:
    """For each verdict bucket, aggregate forward-N return stats.

    Returns:
        {
          "strong_buy": {"count": N, "forward_5": {"mean_return_pct", "win_rate", "sample_size"}, ...},
          "buy":        {...},
          "neutral":    {...},
          "sell":       {...},
          "strong_sell":{...},
        }
    Every bucket key is always present, even with count=0.
    """
    closes = wf_df["close"].values
    verdicts = wf_df["verdict"].values

    buckets: dict[str, dict] = {
        label: {"count": 0, "_returns": {n: [] for n in forward_days}}
        for label in _VERDICT_LABELS
    }

    for i in range(len(wf_df)):
        v = verdicts[i]
        if v not in buckets:
            continue
        buckets[v]["count"] += 1
        for n in forward_days:
            j = i + n
            if j >= len(closes):
                continue
            ret_pct = (closes[j] / closes[i] - 1) * 100
            buckets[v]["_returns"][n].append(ret_pct)

    result: dict[str, dict] = {}
    for label, b in buckets.items():
        entry: dict = {"count": b["count"]}
        for n in forward_days:
            rets = b["_returns"][n]
            if rets:
                mean_ret = sum(rets) / len(rets)
                if label in _BULL_VERDICTS:
                    wins = sum(1 for r in rets if r > 0)
                elif label in _BEAR_VERDICTS:
                    wins = sum(1 for r in rets if r < 0)
                else:
                    wins = sum(1 for r in rets if r > 0)
                win_rate = wins / len(rets)
                entry[f"forward_{n}"] = {
                    "mean_return_pct": mean_ret,
                    "win_rate": win_rate,
                    "sample_size": len(rets),
                }
            else:
                entry[f"forward_{n}"] = {
                    "mean_return_pct": 0.0,
                    "win_rate": 0.0,
                    "sample_size": 0,
                }
        result[label] = entry
    return result
