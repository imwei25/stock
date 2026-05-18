"""Pure metric calculations — no strategy state, no engine state.

All functions here are deterministic and side-effect-free; they operate on
already-realised equity curves and closed trades.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    equity_series: pd.Series | Iterable[float],
    trades: list,
    risk_free_rate: float = 0.02,
) -> dict:
    """Standard metrics for an equity curve plus a list of closed trades.

    Args:
        equity_series: per-bar equity values (1.0-normalised on bar 0).
        trades: items each exposing a numeric ``ret`` (object attr or dict key);
                trade returns are assumed to be net of fees and slippage.
        risk_free_rate: annualised, used for the Sharpe daily-RF subtraction.

    Returns:
        dict with keys:
            total_return          — eq[-1]/eq[0] - 1
            annualized_return     — geometric annualisation over the equity span
            max_drawdown          — largest peak-to-trough drawdown (positive)
            sharpe                — annualised Sharpe of bar-to-bar returns
            trade_count           — len(trades)
            win_rate              — share of trades with ret > 0
            avg_trade_return_pct  — mean(ret) * 100
    """
    eq = np.asarray(list(equity_series), dtype=float) if not isinstance(equity_series, pd.Series) \
        else equity_series.to_numpy(dtype=float)

    if len(eq) == 0:
        return _empty_metrics()

    total_return = float(eq[-1] / eq[0] - 1)
    n_days = len(eq)
    if n_days > 1 and total_return > -1:
        ann = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
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

    if n_days > 2:
        daily_rets = eq[1:] / eq[:-1] - 1
        std = float(np.std(daily_rets, ddof=1))
        if std > 0:
            daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
            sharpe = float(
                (np.mean(daily_rets) - daily_rf) / std * TRADING_DAYS_PER_YEAR ** 0.5
            )
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    if trades:
        rets = [_trade_ret(t) for t in trades]
        wins = sum(1 for r in rets if r > 0)
        win_rate = wins / len(rets)
        avg_trade = sum(rets) / len(rets) * 100
    else:
        win_rate = 0.0
        avg_trade = 0.0

    return {
        "total_return": total_return,
        "annualized_return": float(ann),
        "max_drawdown": float(max_dd),
        "sharpe": sharpe,
        "trade_count": len(trades),
        "win_rate": float(win_rate),
        "avg_trade_return_pct": float(avg_trade),
    }


def _empty_metrics() -> dict:
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "trade_count": 0,
        "win_rate": 0.0,
        "avg_trade_return_pct": 0.0,
    }


def _trade_ret(t) -> float:
    if isinstance(t, dict):
        return float(t["ret"])
    return float(t.ret)
