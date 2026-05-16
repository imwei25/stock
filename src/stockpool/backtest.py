"""Historical signal hit-rate stats.

For each bar in history, run detect_signals; for each trigger, look forward
N days and record (close_{i+N} / close_i - 1). Aggregate per (signal_type, N).
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from stockpool.config import WeightsConfig
from stockpool.signals import detect_signals


def compute_hit_rates(
    df: pd.DataFrame,
    weights: WeightsConfig,
    forward_days: list[int],
) -> dict[str, dict]:
    """
    Returns per-signal stats:
      {
        "macd_cross_above_zero": {
          "count": 9, "direction": +1,
          "forward_5":  {"mean_return_pct": 2.1, "win_rate": 0.67, "sample_size": 9},
          ...
        },
        ...
      }
    """
    if len(df) < 2:
        return {}

    buckets: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "direction": 0,
        "returns": defaultdict(list),
    })

    closes = df["close"].values

    for i in range(1, len(df)):
        window = df.iloc[max(0, i - 1): i + 1]
        triggers = detect_signals(window, weights)

        for t in triggers:
            b = buckets[t.signal_type]
            b["count"] += 1
            b["direction"] = t.direction
            for n in forward_days:
                j = i + n
                if j < len(df):
                    ret_pct = (closes[j] / closes[i] - 1) * 100
                    b["returns"][n].append(ret_pct)

    result: dict[str, dict] = {}
    for sig, b in buckets.items():
        entry = {"count": b["count"], "direction": b["direction"]}
        for n in forward_days:
            rs = b["returns"][n]
            if rs:
                mean_ret = sum(rs) / len(rs)
                if b["direction"] == +1:
                    wins = sum(1 for r in rs if r > 0)
                else:
                    wins = sum(1 for r in rs if r < 0)
                win_rate = wins / len(rs)
                sample = len(rs)
            else:
                mean_ret = 0.0
                win_rate = 0.0
                sample = 0
            entry[f"forward_{n}"] = {
                "mean_return_pct": mean_ret,
                "win_rate": win_rate,
                "sample_size": sample,
            }
        result[sig] = entry
    return result
