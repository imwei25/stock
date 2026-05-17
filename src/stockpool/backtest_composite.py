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


@dataclass
class EquityResult:
    """Output of simulate_equity_curve.

    curves[N] -> DataFrame with columns: date, equity, position
    metrics[N] -> dict with total_return, annualized_return, max_drawdown,
                  trade_count, win_rate, avg_trade_return_pct
    buy_and_hold -> DataFrame[date, equity] or None
    buy_and_hold_metrics -> dict (win_rate / avg_trade_return_pct are None)
    """
    curves: dict[int, pd.DataFrame]
    metrics: dict[int, dict]
    buy_and_hold: pd.DataFrame | None = None
    buy_and_hold_metrics: dict | None = None


_TRADING_DAYS_PER_YEAR = 252


def _compute_metrics(equity_series, trades: list[dict]) -> dict:
    eq = equity_series.values
    total_return = float(eq[-1] / eq[0] - 1) if len(eq) > 0 else 0.0
    n_days = len(eq)
    if n_days > 1 and total_return > -1:
        ann = (1 + total_return) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1
    else:
        ann = 0.0

    running_peak = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > running_peak:
            running_peak = v
        dd = (running_peak - v) / running_peak if running_peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    if trades:
        wins = sum(1 for t in trades if t["ret"] > 0)
        win_rate = wins / len(trades)
        avg_trade = sum(t["ret"] for t in trades) / len(trades) * 100
    else:
        win_rate = 0.0
        avg_trade = 0.0

    return {
        "total_return": total_return,
        "annualized_return": float(ann),
        "max_drawdown": float(max_dd),
        "trade_count": len(trades),
        "win_rate": float(win_rate),
        "avg_trade_return_pct": float(avg_trade),
    }


def _simulate_one(wf_df: pd.DataFrame, N: int) -> tuple[pd.DataFrame, dict]:
    closes = wf_df["close"].values
    verdicts = wf_df["verdict"].values
    n = len(wf_df)

    position = [0] * n
    equity = [1.0] * n
    days_held = 0
    entry_idx: int | None = None
    trades: list[dict] = []

    for t in range(1, n):
        prev_v = verdicts[t - 1]

        if position[t - 1] == 0:
            if prev_v in ("buy", "strong_buy"):
                position[t] = 1
                entry_idx = t - 1
                days_held = 0
            else:
                position[t] = 0
        else:
            held_now = days_held + 1
            if held_now >= N or prev_v in ("sell", "strong_sell"):
                position[t] = 0
                exit_idx = t - 1
                ret = closes[exit_idx] / closes[entry_idx] - 1
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": exit_idx, "ret": float(ret),
                })
                entry_idx = None
                days_held = 0
            else:
                position[t] = 1
                days_held = held_now

        daily_ret = closes[t] / closes[t - 1] - 1
        equity[t] = equity[t - 1] * (1 + position[t] * daily_ret)

    curve = pd.DataFrame({
        "date": wf_df["date"].values,
        "equity": equity,
        "position": position,
    })
    return curve, _compute_metrics(curve["equity"], trades)


def _simulate_buy_and_hold(wf_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    closes = wf_df["close"].values
    equity = closes / closes[0]
    curve = pd.DataFrame({"date": wf_df["date"].values, "equity": equity})
    total_return = float(equity[-1] - 1)
    n_days = len(equity)
    ann = (1 + total_return) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1 if n_days > 1 else 0.0

    running_peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > running_peak:
            running_peak = v
        dd = (running_peak - v) / running_peak if running_peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    metrics = {
        "total_return": total_return,
        "annualized_return": float(ann),
        "max_drawdown": float(max_dd),
        "trade_count": 1,
        "win_rate": None,
        "avg_trade_return_pct": None,
    }
    return curve, metrics


def simulate_equity_curve(
    wf_df: pd.DataFrame,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
) -> EquityResult:
    """Simulate the B1 strategy for each N in holding_days_list.

    For each N, equity starts at 1.0 on the first walk-forward bar. Decisions
    are made at end-of-day t-1 based on verdict[t-1], realized at close[t-1].
    Long-only; no fees; no T+1; no slippage.
    """
    curves: dict[int, pd.DataFrame] = {}
    metrics: dict[int, dict] = {}
    for N in holding_days_list:
        curve, m = _simulate_one(wf_df, N)
        curves[N] = curve
        metrics[N] = m

    bh_curve = None
    bh_metrics = None
    if with_buy_and_hold and len(wf_df) > 0:
        bh_curve, bh_metrics = _simulate_buy_and_hold(wf_df)

    return EquityResult(
        curves=curves, metrics=metrics,
        buy_and_hold=bh_curve, buy_and_hold_metrics=bh_metrics,
    )
