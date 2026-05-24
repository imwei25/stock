"""Composite-strategy walk-forward backtest — adapter over ``backtesting``.

Historically this module owned both the signal generator and the equity
simulator. Both have moved into ``stockpool.backtesting``; this module is
now a thin compatibility layer that:

  * exposes the original public API (``walk_forward_verdicts``,
    ``verdict_bucket_stats``, ``simulate_equity_curve``, ``EquityResult``);
  * preserves the original DataFrame column names (``verdict`` rather than
    the framework's neutral ``signal``).

New code should depend directly on ``stockpool.backtesting`` — see
``docs/backtesting_framework.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpool.backtesting import (
    BacktestEngine,
    CompositeVerdictStrategy,
    MultiLotBacktestEngine,
    TradeCosts,
    VerdictExecution,
    buy_and_hold_baseline,
)
from stockpool.backtesting.sizing import FixedLotSizer, LotSizer
from stockpool.config import (
    IndicatorsConfig, ScoringConfig, VerdictsConfig, WeightsConfig,
)


_WF_COLUMNS = ["date", "open", "close", "daily_score", "weekly_score", "final_score", "verdict"]


def walk_forward_verdicts(
    daily_df: pd.DataFrame,
    weights: WeightsConfig,
    scoring_cfg: ScoringConfig,
    verdicts_cfg: VerdictsConfig,
    indicators_cfg: IndicatorsConfig,
) -> pd.DataFrame:
    """Reconstruct the composite verdict for every bar (after warmup).

    Thin wrapper over ``CompositeVerdictStrategy.generate_signals``; renames
    the framework's neutral ``signal`` column back to ``verdict`` to keep the
    original schema.

    Returns a DataFrame with columns:
        date, close, daily_score, weekly_score, final_score, verdict
    """
    strategy = CompositeVerdictStrategy(weights, scoring_cfg, verdicts_cfg, indicators_cfg)
    signals = strategy.generate_signals(daily_df)
    if len(signals) == 0:
        return pd.DataFrame(columns=_WF_COLUMNS)
    out = signals.rename(columns={"signal": "verdict"})
    return out[_WF_COLUMNS].reset_index(drop=True)


_VERDICT_LABELS = ("strong_buy", "buy", "neutral", "sell", "strong_sell")
_BULL_VERDICTS = {"strong_buy", "buy"}
_BEAR_VERDICTS = {"strong_sell", "sell"}


def verdict_bucket_stats(
    wf_df: pd.DataFrame, forward_days: list[int]
) -> dict[str, dict]:
    """Forward-N return statistics aggregated by verdict bucket.

    Pure analysis: looks only at the (verdict, close) columns of the
    walk-forward output. Not strategy-agnostic — buy/sell win-rate direction
    is baked in (buy wins on positive return, sell wins on negative).

    Returns a dict keyed by verdict label (all five labels always present):
        {
          "strong_buy": {"count": N, "forward_5": {...}, "forward_10": {...}},
          "buy":        {...},
          "neutral":    {...},
          "sell":       {...},
          "strong_sell":{...},
        }

    Each ``forward_n`` sub-dict has ``mean_return_pct``, ``win_rate``,
    ``sample_size``.
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
    """Aggregated output of ``simulate_equity_curve``.

    curves[N]                — DataFrame with columns ``date``, ``equity``, ``position``.
    metrics[N]               — total_return, annualized_return, max_drawdown,
                               sharpe, trade_count, win_rate, avg_trade_return_pct.
    buy_and_hold             — DataFrame with columns ``date``, ``equity`` (or None).
    buy_and_hold_metrics     — same metrics dict shape; win_rate /
                               avg_trade_return_pct are ``None`` for B&H.
    """
    curves: dict[int, pd.DataFrame]
    metrics: dict[int, dict]
    buy_and_hold: pd.DataFrame | None = None
    buy_and_hold_metrics: dict | None = None


def simulate_equity_curve(
    wf_df: pd.DataFrame,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    risk_free_rate: float = 0.02,
    engine: str = "single",
    position_size: float | None = None,
    lot_sizer: LotSizer | None = None,
    max_concurrent_lots: int | None = None,
) -> EquityResult:
    """Simulate the composite strategy for each holding-day cap in the list.

    Delegates to either ``BacktestEngine`` (single-position) or
    ``MultiLotBacktestEngine`` (each buy opens an independent lot) under the
    hood. Default is ``engine="single"`` to preserve legacy behaviour for
    direct callers; the CLI overrides this via ``BacktestConfig.engine``.

    Args:
        engine: ``"single"`` (default, satisfies legacy callers/tests) or
                ``"multi_lot"``.
        lot_sizer: only used when ``engine="multi_lot"`` — a ``LotSizer``
                   callable (e.g. ``FixedLotSizer(0.1)`` or ``VolTargetLotSizer(...)``)
                   that determines lot size per buy. Preferred over the
                   deprecated ``position_size``.
        position_size: deprecated alias of ``lot_sizer=FixedLotSizer(position_size)``;
                       kept for backwards compat (existing tests pass it as a
                       kwarg). Mutually exclusive with ``lot_sizer`` — passing
                       both raises ValueError. If both are None, defaults to
                       ``FixedLotSizer(0.1)``.
        max_concurrent_lots: only used when ``engine="multi_lot"`` — cap on
                             simultaneous open lots; None = uncapped by count
                             (cash still self-caps).
    """
    signals = wf_df.rename(columns={"verdict": "signal"})
    costs = TradeCosts(buy_cost=buy_cost, sell_cost=sell_cost)

    if engine == "single":
        bt = BacktestEngine(
            VerdictExecution(),
            costs=costs,
            risk_free_rate=risk_free_rate,
        )
    elif engine == "multi_lot":
        if lot_sizer is None:
            size = position_size if position_size is not None else 0.1
            lot_sizer = FixedLotSizer(size)
        elif position_size is not None:
            raise ValueError(
                "Pass either lot_sizer or position_size, not both"
            )
        bt = MultiLotBacktestEngine(
            VerdictExecution(),
            lot_sizer=lot_sizer,
            costs=costs,
            risk_free_rate=risk_free_rate,
            max_concurrent_lots=max_concurrent_lots,
        )
    else:
        raise ValueError(f"engine must be 'single' or 'multi_lot', got {engine!r}")

    curves: dict[int, pd.DataFrame] = {}
    metrics: dict[int, dict] = {}
    for N in holding_days_list:
        result = bt.run_on_signals(signals, max_holding_days=N)
        curves[N] = result.curve
        metrics[N] = result.metrics

    bh_curve = None
    bh_metrics = None
    if with_buy_and_hold and len(wf_df) > 0:
        bh = buy_and_hold_baseline(wf_df, risk_free_rate=risk_free_rate)
        bh_curve = bh.curve[["date", "equity"]].reset_index(drop=True)
        bh_metrics = bh.metrics

    return EquityResult(
        curves=curves,
        metrics=metrics,
        buy_and_hold=bh_curve,
        buy_and_hold_metrics=bh_metrics,
    )
