"""Pure metric calculations — no strategy state, no engine state.

All functions here are deterministic and side-effect-free; they operate on
already-realised equity curves and closed trades.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Below these sample sizes the corresponding metric is undefined (None):
# annualising a short window explodes (e.g. +5% over 10 days → +242%/yr),
# and a Sharpe from a handful of bars is statistical noise.
MIN_DAYS_FOR_ANNUALIZED = 60
MIN_DAYS_FOR_SHARPE = 20


def compute_metrics(
    equity_series: pd.Series | Iterable[float],
    trades: list,
    risk_free_rate: float = 0.02,
    active_from_idx: int | None = None,
) -> dict:
    """Standard metrics for an equity curve plus a list of closed trades.

    Args:
        equity_series: per-bar equity values (1.0-normalised on bar 0).
        trades: items each exposing a numeric ``ret`` (object attr or dict key);
                trade returns are assumed to be net of fees and slippage.
        risk_free_rate: annualised, used for the Sharpe daily-RF subtraction.
        active_from_idx: when not None, slice the equity curve from this index
                before computing — used for "active span" metrics that exclude
                a cold-start flat head (e.g. ML strategies that emit neutral
                until enough training samples accrue). The anchor bar is
                ``equity[active_from_idx]``; trade stats are unaffected.

    Returns:
        dict with keys:
            total_return          — eq[-1]/eq[0] - 1
            annualized_return     — geometric annualisation over the equity
                                    span; ``None`` when the span has fewer than
                                    ``MIN_DAYS_FOR_ANNUALIZED`` (60) bars
            max_drawdown          — largest peak-to-trough drawdown (positive)
            sharpe                — annualised Sharpe of bar-to-bar returns;
                                    ``None`` when the span has fewer than
                                    ``MIN_DAYS_FOR_SHARPE`` (20) bars
            trade_count           — len(trades)
            win_rate              — share of trades with ret > 0; ``None``
                                    when there are no closed trades
            avg_trade_return_pct  — mean(ret) * 100; ``None`` when there are
                                    no closed trades
    """
    eq = np.asarray(list(equity_series), dtype=float) if not isinstance(equity_series, pd.Series) \
        else equity_series.to_numpy(dtype=float)

    if active_from_idx is not None:
        eq = eq[max(int(active_from_idx), 0):]

    if len(eq) == 0:
        m = _empty_metrics()
        m["trade_count"] = len(trades)
        return m

    total_return = float(eq[-1] / eq[0] - 1)
    n_days = len(eq)
    ann: float | None
    if n_days < MIN_DAYS_FOR_ANNUALIZED:
        ann = None
    elif total_return > -1:
        ann = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)
    else:
        ann = 0.0

    running_peak = float(eq[0])
    max_dd = 0.0
    for v in eq:
        if v > running_peak:
            running_peak = float(v)
        dd = (running_peak - float(v)) / running_peak if running_peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    sharpe: float | None
    if n_days < MIN_DAYS_FOR_SHARPE:
        sharpe = None
    else:
        daily_rets = eq[1:] / eq[:-1] - 1
        std = float(np.std(daily_rets, ddof=1))
        if std > 0:
            daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
            sharpe = float(
                (np.mean(daily_rets) - daily_rf) / std * TRADING_DAYS_PER_YEAR ** 0.5
            )
        else:
            sharpe = 0.0

    win_rate: float | None
    avg_trade: float | None
    if trades:
        rets = [_trade_ret(t) for t in trades]
        wins = sum(1 for r in rets if r > 0)
        win_rate = float(wins / len(rets))
        avg_trade = float(sum(rets) / len(rets) * 100)
    else:
        win_rate = None
        avg_trade = None

    return {
        "total_return": total_return,
        "annualized_return": ann,
        "max_drawdown": float(max_dd),
        "sharpe": sharpe,
        "trade_count": len(trades),
        "win_rate": win_rate,
        "avg_trade_return_pct": avg_trade,
    }


def _empty_metrics() -> dict:
    return {
        "total_return": 0.0,
        "annualized_return": None,
        "max_drawdown": 0.0,
        "sharpe": None,
        "trade_count": 0,
        "win_rate": None,
        "avg_trade_return_pct": None,
    }


def _trade_ret(t) -> float:
    if isinstance(t, dict):
        return float(t["ret"])
    return float(t.ret)
