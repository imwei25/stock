"""Portfolio-level backtest engine.

Per-bar mechanics (T+1 compliant, mirrors ``MultiLotBacktestEngine``):

  * Bar ``t`` ``close`` is the decision price; ``open[t+1]`` is the
    execution price.
  * Mark-to-market uses ``close[t]``.
  * Rebalance bars: at indices ``{start_offset, start_offset+n, ...}``
    (bar-index based, not calendar-based — immune to holidays).
  * On a rebalance bar ``t`` (with ``t+1 < T``): compute scores at ``date_t``,
    pick top-K, queue a target set; execute next bar at ``open[t+1]`` by
    full-rebalance equal-weight.
  * Codes missing today's bar are skipped (un-investable). Existing positions
    in such codes are *held* (no forced sale) — that mirrors how a real
    portfolio would behave on a no-quote day.

PR-1 simplifications (see spec §10.2 retrenchments):
  * No eligibility filter (universe = scores.keys())
  * No industry cap (top-K is pure score rank)
  * No staggered ensemble (engine accepts ``start_offset`` but no aggregator)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.backtesting.framework import Trade, TradeCosts
from stockpool.backtesting.metrics import compute_metrics
from stockpool.config import PortfolioRunConfig
from stockpool.portfolio.eligibility import EligibilityFilter
from stockpool.portfolio.result import PortfolioBacktestResult, PortfolioTrade
from stockpool.portfolio.strategy import PortfolioStrategy

log = logging.getLogger("stockpool")


@dataclass
class _Position:
    """Internal: one open position."""
    code: str
    entry_idx: int
    entry_date: pd.Timestamp
    entry_price: float
    shares: float            # share count (continuous: dollars / price)
    weight_at_entry: float   # fraction of equity allocated at entry


class PortfolioEngine:
    """Top-K equal-weight portfolio engine with periodic rebalance."""

    def __init__(
        self,
        strategy: PortfolioStrategy,
        portfolio_cfg: PortfolioRunConfig,
        costs: TradeCosts = TradeCosts(),
        risk_free_rate: float = 0.02,
        eligibility: EligibilityFilter | None = None,
        sector_map: Mapping[str, str] | None = None,
    ):
        """Construct an engine.

        Args:
            eligibility: optional ``EligibilityFilter``. When ``None`` (PR-1
                behavior) every code in ``scores`` is eligible.
            sector_map: ``{code: sector}``. Used for ``max_per_industry`` cap
                in ``portfolio_cfg``. ``None`` or empty dict disables the cap
                (irrespective of ``max_per_industry``).
        """
        self.strategy = strategy
        self.cfg = portfolio_cfg
        self.costs = costs
        self.risk_free_rate = risk_free_rate
        self.eligibility = eligibility
        self.sector_map = dict(sector_map or {})

    # ---- public ----

    def run(
        self,
        panel_data: Mapping[str, pd.DataFrame],
        start_offset: int = 0,
    ) -> PortfolioBacktestResult:
        opens, closes = _build_wide_pivots(panel_data)
        if opens.empty:
            return _empty_result(self.strategy.name, self.risk_free_rate)

        dates = opens.index
        n_bars = len(dates)
        rebalance_bars = set(range(start_offset, n_bars, self.cfg.rebalance_n_days))

        cash = float(self.cfg.initial_cash)
        positions: dict[str, _Position] = {}
        closed: list[PortfolioTrade] = []
        equity = np.zeros(n_bars)
        num_pos = np.zeros(n_bars, dtype=int)
        cash_ratio = np.zeros(n_bars)
        rebalance_records: list[dict] = []
        pending_target: set[str] | None = None

        for t in range(n_bars):
            date_t = dates[t]
            opens_t = opens.iloc[t]
            closes_t = closes.iloc[t]

            # 1. Execute pending trade decided on bar t-1.
            if pending_target is not None:
                cash, closed_now = _rebalance_to_target(
                    positions=positions, cash=cash,
                    target=pending_target,
                    opens_t=opens_t, t=t, date_t=date_t,
                    costs=self.costs,
                )
                closed.extend(closed_now)
                pending_target = None

            # 2. Mark-to-market at close[t]. Positions lacking close[t] keep
            #    their previous mark (defensive: shouldn't normally happen
            #    inside the date range).
            held_value = 0.0
            for pos in positions.values():
                close_t = closes_t.get(pos.code)
                if pd.notna(close_t):
                    held_value += pos.shares * float(close_t)
                else:
                    # Fall back to entry-price valuation if today's close is
                    # missing (rare; defensive only).
                    held_value += pos.shares * pos.entry_price
            total_equity = cash + held_value
            equity[t] = total_equity
            num_pos[t] = len(positions)
            cash_ratio[t] = cash / total_equity if total_equity > 0 else 1.0

            # 3. Decide next-bar target on rebalance bars.
            can_execute_next_bar = (t + 1) < n_bars
            if t in rebalance_bars and can_execute_next_bar:
                scores = self.strategy.predict_scores(date_t, dict(_to_panel_dict(panel_data)))
                if self.eligibility is not None:
                    eligible_codes = self.eligibility.eligible(date_t, panel_data)
                    scores = {c: s for c, s in scores.items() if c in eligible_codes}
                if scores:
                    pending_target = _select_top_k(
                        scores=scores,
                        k=self.cfg.top_k,
                        opens_next=opens.iloc[t + 1],
                        sector_map=self.sector_map,
                        max_per_industry=self.cfg.max_per_industry,
                    )
                    rebalance_records.append({
                        "date": date_t,
                        "target_codes": sorted(pending_target),
                        "num_target": len(pending_target),
                    })

        # End-of-backtest close-out: realize remaining positions so trade-level
        # metrics (win rate, avg trade ret) include them. Equity[-1] is *not*
        # overwritten — the curve already reflects mark-to-market at close[-1];
        # forcing a post-liquidation cash figure would shave off sell_cost on
        # the last bar in a way the live portfolio wouldn't observe.
        if positions:
            last_idx = n_bars - 1
            last_date = dates[last_idx]
            closes_last = closes.iloc[last_idx]
            for code, pos in list(positions.items()):
                price = closes_last.get(code)
                if pd.isna(price):
                    price = pos.entry_price
                proceeds = pos.shares * float(price) * (1 - self.costs.sell_cost)
                notional = pos.shares * pos.entry_price
                ret = (proceeds / notional) - 1 if notional > 0 else 0.0
                closed.append(PortfolioTrade(
                    code=code,
                    entry_date=pos.entry_date,
                    exit_date=last_date,
                    entry_price=pos.entry_price,
                    exit_price=float(price),
                    weight_at_entry=pos.weight_at_entry,
                    ret=float(ret),
                    days_held=last_idx - pos.entry_idx,
                    exit_reason="end_of_backtest",
                ))

        curve = pd.DataFrame({
            "date": dates,
            "equity": equity,
            "num_positions": num_pos,
            "cash_ratio": cash_ratio,
        })
        rebalance_log = (
            pd.DataFrame(rebalance_records)
            if rebalance_records
            else pd.DataFrame(columns=["date", "target_codes", "num_target"])
        )
        # Adapt PortfolioTrade → Trade for compute_metrics.
        adapter_trades = [
            Trade(
                entry_idx=0,                # not used by compute_metrics
                exit_idx=0,
                entry_price=t.entry_price,
                exit_price=t.exit_price if t.exit_price is not None else t.entry_price,
                ret=t.ret,
                days_held=t.days_held,
            )
            for t in closed
        ]
        metrics = compute_metrics(
            pd.Series(equity), adapter_trades, risk_free_rate=self.risk_free_rate,
        )
        return PortfolioBacktestResult(
            curve=curve,
            trades=closed,
            rebalance_log=rebalance_log,
            metrics=metrics,
            strategy_name=self.strategy.name,
        )


# ---- helpers ----


def _to_panel_dict(panel_data):
    # Cheap pass-through (kept as a function in case we later want to slice
    # ≤ date_t per call — currently PrecomputedScoreStrategy doesn't need it).
    return panel_data


def _build_wide_pivots(
    panel_data: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack per-stock OHLCV into wide (T × N) pivots of open / close.

    Falls back to ``open = close`` when a stock's frame lacks an ``open`` column
    (mirrors per-stock engine's fallback in ``_opens_with_fallback``).
    """
    opens_by_code: dict[str, pd.Series] = {}
    closes_by_code: dict[str, pd.Series] = {}
    for code, df in panel_data.items():
        if "date" not in df.columns or "close" not in df.columns:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date").sort_index()
        closes_by_code[code] = d["close"].astype(float)
        if "open" in d.columns:
            opens_by_code[code] = d["open"].astype(float)
        else:
            # Fall back: open[t] = close[t-1], open[0] = close[0].
            o = d["close"].shift(1)
            o.iloc[0] = d["close"].iloc[0]
            opens_by_code[code] = o.astype(float)
    if not closes_by_code:
        return pd.DataFrame(), pd.DataFrame()
    opens_df = pd.DataFrame(opens_by_code).sort_index()
    closes_df = pd.DataFrame(closes_by_code).sort_index()
    # Align columns (defensive).
    cols = sorted(set(opens_df.columns) & set(closes_df.columns))
    return opens_df[cols], closes_df[cols]


def _select_top_k(
    scores: dict[str, float],
    k: int,
    opens_next: pd.Series,
    sector_map: Mapping[str, str] | None = None,
    max_per_industry: int | None = None,
) -> set[str]:
    """Take top-K by score, skipping codes without executable open[t+1].

    If ``max_per_industry`` and ``sector_map`` are both provided, applies a
    greedy per-industry cap during the top-K walk. ``"Unknown"`` semantics
    (matching Pool B): if *every* candidate has no sector mapping, the cap
    is skipped (everyone would be in the same bucket otherwise); else
    unmapped codes go into a single ``"Unknown"`` bucket that counts
    normally against the cap.
    """
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    apply_cap = (
        max_per_industry is not None
        and sector_map is not None
        and any(sector_map.get(c) for c, _ in ranked)
    )
    out: set[str] = set()
    industry_count: dict[str, int] = {}
    for code, _ in ranked:
        if len(out) >= k:
            break
        px = opens_next.get(code)
        if pd.isna(px):
            continue
        if apply_cap:
            ind = sector_map.get(code) or "Unknown"
            if industry_count.get(ind, 0) >= max_per_industry:
                continue
            industry_count[ind] = industry_count.get(ind, 0) + 1
        out.add(code)
    return out


def _rebalance_to_target(
    positions: dict[str, _Position],
    cash: float,
    target: set[str],
    opens_t: pd.Series,
    t: int,
    date_t: pd.Timestamp,
    costs: TradeCosts,
) -> tuple[float, list[PortfolioTrade]]:
    """Full equal-weight rebalance: liquidate, redistribute.

    PR-1 simplification per spec — clean semantics, no turnover_cap. Sells
    everything (or holds if next-bar open is NaN), pools cash, then buys
    target codes equal-weight with whatever cash remains.

    Survivors in (current ∩ target) are sold and re-bought (T+1 fictitious
    churn) so the math stays uniform; PR-2/3 can add a "hold survivors"
    optimization. The cost is real (extra round-trip fees) but small at
    typical buy/sell ratios (~0.1%).
    """
    closed: list[PortfolioTrade] = []
    # 1. Liquidate every position with a valid open price today.
    for code in list(positions.keys()):
        pos = positions[code]
        px = opens_t.get(code)
        if pd.isna(px):
            # No quote today — keep the position; metric impact is small.
            continue
        proceeds = pos.shares * float(px) * (1 - costs.sell_cost)
        cash += proceeds
        notional = pos.shares * pos.entry_price
        ret = (proceeds / notional) - 1 if notional > 0 else 0.0
        closed.append(PortfolioTrade(
            code=code,
            entry_date=pos.entry_date,
            exit_date=date_t,
            entry_price=pos.entry_price,
            exit_price=float(px),
            weight_at_entry=pos.weight_at_entry,
            ret=float(ret),
            days_held=t - pos.entry_idx,
            exit_reason="rebalance_drop",
        ))
        del positions[code]

    # 2. Of the target, keep only codes with a valid open price.
    buyable = [c for c in target if pd.notna(opens_t.get(c))]
    if not buyable:
        return cash, closed
    per_lot = cash / len(buyable)
    for code in buyable:
        px = float(opens_t.get(code))
        committed = per_lot * (1 - costs.buy_cost)
        shares = committed / px if px > 0 else 0.0
        if shares <= 0:
            continue
        positions[code] = _Position(
            code=code, entry_idx=t, entry_date=date_t,
            entry_price=px, shares=shares,
            weight_at_entry=per_lot / max(cash, 1e-12),
        )
        cash -= per_lot
    return cash, closed


def _empty_result(name: str, risk_free_rate: float) -> PortfolioBacktestResult:
    empty_curve = pd.DataFrame({
        "date": [], "equity": [], "num_positions": [], "cash_ratio": [],
    })
    return PortfolioBacktestResult(
        curve=empty_curve,
        trades=[],
        rebalance_log=pd.DataFrame(columns=["date", "target_codes", "num_target"]),
        metrics=compute_metrics(pd.Series([], dtype=float), [], risk_free_rate),
        strategy_name=name,
    )
