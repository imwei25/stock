"""Signal detection + composite scoring.

The full rubric lives in spec § 5. Each detection function reads the last 1-2
rows of a DataFrame with indicator columns, and emits Triggers.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpool.config import ScoringConfig, VerdictsConfig, WeightsConfig


@dataclass
class Trigger:
    signal_type: str          # e.g. "macd_cross_above_zero"
    direction: int            # +1 bullish, -1 bearish
    weight: int               # from WeightsConfig
    description: str          # human-readable Chinese


def _golden_cross(prev_fast: float, prev_slow: float,
                  curr_fast: float, curr_slow: float) -> bool:
    return prev_fast <= prev_slow and curr_fast > curr_slow


def _dead_cross(prev_fast: float, prev_slow: float,
                curr_fast: float, curr_slow: float) -> bool:
    return prev_fast >= prev_slow and curr_fast < curr_slow


def detect_signals(df: pd.DataFrame, weights: WeightsConfig) -> list[Trigger]:
    """Scan last K bar, return all triggered signals."""
    if len(df) < 2:
        return []

    triggers: list[Trigger] = []
    prev, curr = df.iloc[-2], df.iloc[-1]

    # === MA golden/dead cross (5 over 20) ===
    if "ma5" in df.columns and "ma20" in df.columns:
        if _golden_cross(prev["ma5"], prev["ma20"], curr["ma5"], curr["ma20"]):
            triggers.append(Trigger("ma_cross_strong", +1, weights.ma_cross_strong,
                                    "MA5 上穿 MA20(金叉)"))
        elif _dead_cross(prev["ma5"], prev["ma20"], curr["ma5"], curr["ma20"]):
            triggers.append(Trigger("ma_cross_strong", -1, weights.ma_cross_strong,
                                    "MA5 下穿 MA20(死叉)"))

    # === MA bullish/bearish alignment ===
    ma_cols = [c for c in ["ma5", "ma10", "ma20", "ma60"] if c in df.columns]
    if len(ma_cols) >= 3:
        vals = [curr[c] for c in ma_cols]
        if not any(pd.isna(v) for v in vals):
            if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
                triggers.append(Trigger("ma_alignment_bull", +1, weights.ma_alignment,
                                        "MA 多头排列(短>长)"))
            elif all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
                triggers.append(Trigger("ma_alignment_bear", -1, weights.ma_alignment,
                                        "MA 空头排列(短<长)"))

    # === MACD ===
    if "macd_dif" in df.columns:
        cross_up = _golden_cross(prev["macd_dif"], prev["macd_dea"],
                                 curr["macd_dif"], curr["macd_dea"])
        cross_down = _dead_cross(prev["macd_dif"], prev["macd_dea"],
                                 curr["macd_dif"], curr["macd_dea"])
        above_zero = curr["macd_dif"] > 0

        if cross_up:
            if above_zero:
                triggers.append(Trigger("macd_cross_above_zero", +1,
                                        weights.macd_cross_above_zero,
                                        "MACD 零轴上方金叉(强多)"))
            else:
                triggers.append(Trigger("macd_cross_below_zero", +1,
                                        weights.macd_cross_below_zero,
                                        "MACD 零轴下方金叉(弱多)"))
        elif cross_down:
            if above_zero:
                triggers.append(Trigger("macd_cross_below_zero", -1,
                                        weights.macd_cross_below_zero,
                                        "MACD 零轴上方死叉(弱空)"))
            else:
                triggers.append(Trigger("macd_cross_above_zero", -1,
                                        weights.macd_cross_above_zero,
                                        "MACD 零轴下方死叉(强空)"))

        if len(df) >= 4:
            last3 = df["macd_hist"].iloc[-3:].tolist()
            if all(last3[i] > 0 for i in range(3)) and last3[2] > last3[1] > last3[0]:
                triggers.append(Trigger("macd_histogram_expand", +1,
                                        weights.macd_histogram_expand,
                                        "MACD 红柱连续 3 日放大"))
            elif all(last3[i] < 0 for i in range(3)) and last3[2] < last3[1] < last3[0]:
                triggers.append(Trigger("macd_histogram_expand", -1,
                                        weights.macd_histogram_expand,
                                        "MACD 绿柱连续 3 日放大"))

    # === KDJ ===
    if "kdj_k" in df.columns and "kdj_d" in df.columns:
        cross_up = _golden_cross(prev["kdj_k"], prev["kdj_d"],
                                 curr["kdj_k"], curr["kdj_d"])
        cross_down = _dead_cross(prev["kdj_k"], prev["kdj_d"],
                                 curr["kdj_k"], curr["kdj_d"])
        j_val = curr.get("kdj_j", 50)

        if cross_up:
            if j_val < 20:
                triggers.append(Trigger("kdj_oversold_cross", +1,
                                        weights.kdj_oversold_cross,
                                        f"KDJ 超卖金叉(J={j_val:.1f})"))
            else:
                triggers.append(Trigger("kdj_normal_cross", +1,
                                        weights.kdj_normal_cross,
                                        "KDJ 普通金叉"))
        elif cross_down:
            if j_val > 80:
                triggers.append(Trigger("kdj_overbought_cross", -1,
                                        weights.kdj_overbought_cross,
                                        f"KDJ 超买死叉(J={j_val:.1f})"))
            else:
                triggers.append(Trigger("kdj_normal_cross", -1,
                                        weights.kdj_normal_cross,
                                        "KDJ 普通死叉"))

    # === RSI ===
    if "rsi6" in df.columns:
        rsi6 = curr["rsi6"]
        if pd.notna(rsi6):
            if rsi6 < 20:
                triggers.append(Trigger("rsi_oversold", +1, weights.rsi_oversold,
                                        f"RSI6 超卖({rsi6:.1f})"))
            elif rsi6 > 80:
                triggers.append(Trigger("rsi_overbought", -1, weights.rsi_overbought,
                                        f"RSI6 超买({rsi6:.1f})"))

    # === BOLL ===
    if "boll_up" in df.columns:
        if prev["close"] <= prev["boll_low"] and curr["close"] > curr["boll_low"]:
            triggers.append(Trigger("boll_band_touch", +1, weights.boll_band_touch,
                                    "收盘上穿 BOLL 下轨(反弹)"))
        elif prev["close"] >= prev["boll_up"] and curr["close"] < curr["boll_up"]:
            triggers.append(Trigger("boll_band_touch", -1, weights.boll_band_touch,
                                    "收盘跌破 BOLL 上轨(回落)"))
        elif _golden_cross(prev["close"], prev["boll_mid"], curr["close"], curr["boll_mid"]):
            triggers.append(Trigger("boll_mid_cross", +1, weights.boll_mid_cross,
                                    "收盘上穿 BOLL 中轨"))
        elif _dead_cross(prev["close"], prev["boll_mid"], curr["close"], curr["boll_mid"]):
            triggers.append(Trigger("boll_mid_cross", -1, weights.boll_mid_cross,
                                    "收盘下穿 BOLL 中轨"))

    # === Volume ===
    vol_ratio = curr.get("vol_ratio5", 1.0)
    if vol_ratio is not None and pd.notna(vol_ratio) and vol_ratio > 1.5:
        is_bullish_candle = curr["close"] > curr["open"]
        is_bearish_candle = curr["close"] < curr["open"]
        if is_bullish_candle:
            triggers.append(Trigger("volume_surge_bullish", +1,
                                    weights.volume_surge_bullish,
                                    f"放量阳线(量比 {vol_ratio:.2f})"))
        elif is_bearish_candle:
            triggers.append(Trigger("volume_surge_bearish", -1,
                                    weights.volume_surge_bearish,
                                    f"放量阴线(量比 {vol_ratio:.2f})"))

    # === Breakout ===
    if bool(curr.get("is_breakout_high", False)) and not bool(prev.get("is_breakout_high", False)):
        triggers.append(Trigger("breakout_new_high", +1, weights.breakout_new_high,
                                "收盘创 20 日新高"))
    if bool(curr.get("is_breakout_low", False)) and not bool(prev.get("is_breakout_low", False)):
        triggers.append(Trigger("breakout_new_low", -1, weights.breakout_new_low,
                                "收盘创 20 日新低"))

    return triggers


def score_triggers(triggers: list[Trigger]) -> int:
    """Sum of (direction × weight), capped to [-10, +10]."""
    raw = sum(t.direction * t.weight for t in triggers)
    return max(-10, min(10, raw))


def combine_daily_weekly(daily_score: int, weekly_score: int,
                         cfg: ScoringConfig) -> float:
    """final = w_d × daily + w_w × weekly; ± resonance bonus; cap to [-10, +10]."""
    base = cfg.daily_weight * daily_score + cfg.weekly_weight * weekly_score
    if daily_score >= cfg.resonance_daily_threshold and weekly_score >= cfg.resonance_weekly_threshold:
        base += cfg.resonance_bonus
    elif daily_score <= -cfg.resonance_daily_threshold and weekly_score <= -cfg.resonance_weekly_threshold:
        base -= cfg.resonance_bonus
    return max(-10, min(10, base))


def verdict_of(final_score: float, cfg: VerdictsConfig) -> str:
    """Return one of: strong_buy, buy, neutral, sell, strong_sell."""
    if final_score >= cfg.strong_buy:
        return "strong_buy"
    if final_score >= cfg.buy:
        return "buy"
    if final_score <= cfg.strong_sell:
        return "strong_sell"
    if final_score <= cfg.sell:
        return "sell"
    return "neutral"
