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

    # P2-11: 桶 key = (signal_type, direction)。ma_cross_strong / macd_cross_* /
    # kdj_normal_cross / boll_* / macd_histogram_expand 都是 ±1 双向复用同一
    # signal_type,旧实现混进同一桶且 direction 被最后一次触发覆盖 —— 日报
    # "单信号命中率"在这些信号上是统计噪声。
    buckets: dict[tuple, dict] = defaultdict(lambda: {
        "count": 0,
        "direction": 0,
        "returns": defaultdict(list),
    })

    closes = df["close"].values

    for i in range(1, len(df)):
        # P2-11: 窗口 ≥4 行 —— macd_histogram_expand 需要 len(df)>=4,
        # 旧的 2 行窗口让它永远进不了命中率表,但实时评级又给它计分,
        # 同一报告两套口径。与实时路径一致地传足够历史(detect 仍只看尾部)。
        window = df.iloc[max(0, i - 3): i + 1]
        triggers = detect_signals(window, weights)

        for t in triggers:
            b = buckets[(t.signal_type, t.direction)]
            b["count"] += 1
            b["direction"] = t.direction
            for n in forward_days:
                j = i + n
                if j < len(df):
                    ret_pct = (closes[j] / closes[i] - 1) * 100
                    b["returns"][n].append(ret_pct)

    result: dict[str, dict] = {}
    for (sig_type, direction), b in buckets.items():
        # 展示 key:双向信号区分多空(如 "macd_cross_above_zero[+]" / "[-]")
        sig = f"{sig_type}[{'+' if direction >= 0 else '-'}]"
        entry = {"count": b["count"], "direction": b["direction"],
                 "signal_type": sig_type}
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
