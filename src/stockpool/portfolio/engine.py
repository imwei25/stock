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
from stockpool.backtesting.limits import (
    infer_limit_pct,
    open_hits_limit_down,
    open_hits_limit_up,
)
from stockpool.backtesting.metrics import compute_metrics
from stockpool.config import PortfolioRunConfig
from stockpool.portfolio.eligibility import EligibilityFilter
from stockpool.portfolio.result import PortfolioBacktestResult, PortfolioTrade
from stockpool.portfolio.strategy import PortfolioStrategy
from stockpool.portfolio.weighting import compute_target_weights

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
        pending_target: dict[str, float] | None = None

        for t in range(n_bars):
            date_t = dates[t]
            opens_t = opens.iloc[t]
            closes_t = closes.iloc[t]

            # 1. Execute pending trade decided on bar t-1.
            if pending_target is not None:
                cash, closed_now = _rebalance_to_target(
                    positions=positions, cash=cash,
                    target_weights=pending_target,
                    opens_t=opens_t, t=t, date_t=date_t,
                    costs=self.costs,
                    hold_survivors=getattr(self.cfg, "hold_survivors", False),
                    limit_guard=getattr(self.cfg, "limit_guard", False),
                    prev_closes=closes.iloc[t - 1] if t > 0 else None,
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
                    target = _select_top_k(
                        scores=scores,
                        k=self.cfg.top_k,
                        opens_next=opens.iloc[t + 1],
                        sector_map=self.sector_map,
                        max_per_industry=self.cfg.max_per_industry,
                        # limit_guard: skip candidates whose open[t+1] is a
                        # 一字涨停 vs close[t] — un-buyable at the auction;
                        # the next-ranked name substitutes (same anticipation
                        # class as the existing NaN-open skip).
                        prev_closes=(closes_t
                                     if getattr(self.cfg, "limit_guard", False)
                                     else None),
                    )
                    pending_target = self._target_weights(target, scores, closes, t)
                    rebalance_records.append({
                        "date": date_t,
                        "target_codes": sorted(target),
                        "num_target": len(target),
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

    # ---- weighting ----

    def _target_weights(
        self,
        target: set[str],
        scores: Mapping[str, float],
        closes: pd.DataFrame,
        t: int,
    ) -> dict[str, float]:
        """Map a selected target set to ``{code: weight}`` (sums to 1).

        ``weighting="equal"`` returns the 1/K vector (legacy, bit-exact).
        ``weighting="mvo"`` builds a trailing return window from ``closes`` up
        to and including bar ``t`` and delegates to ``compute_target_weights``
        (Ledoit-Wolf cov + box-constrained QP), which itself falls back to
        equal weight on any degenerate window.
        """
        if not target:
            return {}
        method = getattr(self.cfg, "weighting", "equal")
        if method == "equal":
            w = 1.0 / len(target)
            return {c: w for c in target}
        # mvo: trailing daily returns over the lookback window (inclusive of t).
        lookback = self.cfg.mvo_lookback
        lo = max(0, t - lookback)
        cols = [c for c in target if c in closes.columns]
        window = closes.iloc[lo:t + 1][cols].pct_change().iloc[1:] if cols else pd.DataFrame()
        return compute_target_weights(
            sorted(target), scores, window,
            method="mvo",
            risk_aversion=self.cfg.mvo_risk_aversion,
            w_max=self.cfg.mvo_w_max,
            min_obs=self.cfg.mvo_min_obs,
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
    prev_closes: pd.Series | None = None,
) -> set[str]:
    """Take top-K by score, skipping codes without executable open[t+1].

    If ``max_per_industry`` and ``sector_map`` are both provided, applies a
    greedy per-industry cap during the top-K walk. ``"Unknown"`` semantics
    (matching Pool B): if *every* candidate has no sector mapping, the cap
    is skipped (everyone would be in the same bucket otherwise); else
    unmapped codes go into a single ``"Unknown"`` bucket that counts
    normally against the cap.

    If ``prev_closes`` is given (limit_guard), candidates whose ``open[t+1]``
    hits the limit-up price vs ``close[t]`` are skipped — a 一字涨停 auction
    open cannot be bought, so the next-ranked name substitutes.
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
        if prev_closes is not None:
            pc = prev_closes.get(code)
            if pd.notna(pc) and open_hits_limit_up(
                float(px), float(pc), infer_limit_pct(code)
            ):
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
    target_weights: Mapping[str, float],
    opens_t: pd.Series,
    t: int,
    date_t: pd.Timestamp,
    costs: TradeCosts,
    *,
    hold_survivors: bool = False,
    limit_guard: bool = False,
    prev_closes: pd.Series | None = None,
) -> tuple[float, list[PortfolioTrade]]:
    """Rebalance to a target weight vector.

    ``target_weights`` maps ``{code: weight}`` (weights over the selected
    target, summing to ~1). Codes without an executable ``open[t]`` are dropped
    and the remaining weights renormalized, so the engine stays fully invested.
    With an equal-weight vector this is bit-exact with the legacy 1/K behavior.

    Legacy mode (``hold_survivors=False``): sells everything (or holds if the
    open is NaN), pools cash, then buys target codes by weight — survivors in
    (current ∩ target) are sold and re-bought, paying a fictitious round-trip.

    ``hold_survivors=True``: survivors are held untouched (no churn, no cost);
    only dropped names are sold and only newcomers bought, funded by the
    pooled proceeds (weights drift between rebalances — accepted, this mirrors
    a real membership-turnover portfolio).

    ``limit_guard=True`` (needs ``prev_closes`` = close[t-1]): positions whose
    open hits limit-down are unsellable and held until the next rebalance;
    buys whose open hits limit-up are rejected at fill and their allocation
    stays in cash (selection normally already substituted such names).
    """
    def _limit_down_blocked(code: str, px: float) -> bool:
        if not limit_guard or prev_closes is None:
            return False
        pc = prev_closes.get(code)
        return pd.notna(pc) and open_hits_limit_down(
            px, float(pc), infer_limit_pct(code))

    def _limit_up_blocked(code: str, px: float) -> bool:
        if not limit_guard or prev_closes is None:
            return False
        pc = prev_closes.get(code)
        return pd.notna(pc) and open_hits_limit_up(
            px, float(pc), infer_limit_pct(code))

    closed: list[PortfolioTrade] = []
    # 1. Liquidate positions leaving the portfolio (all of them in legacy
    #    mode; only non-target names with hold_survivors).
    for code in list(positions.keys()):
        pos = positions[code]
        if hold_survivors and code in target_weights:
            continue  # survivor — held through the rebalance, zero churn
        px = opens_t.get(code)
        if pd.isna(px):
            # No quote today — keep the position; metric impact is small.
            continue
        if _limit_down_blocked(code, float(px)):
            continue  # 一字跌停 — cannot sell, hold until next rebalance
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

    # 2. Of the target, keep only *new* codes (hold_survivors) with a valid
    #    open price; renormalize weights over the buyable subset (equal-weight
    #    => 1/len(buyable), bit-exact with legacy per_lot = cash/len(buyable)).
    candidates = (
        [c for c in target_weights if c not in positions]
        if hold_survivors else list(target_weights)
    )
    buyable = [c for c in candidates if pd.notna(opens_t.get(c))]
    buyable = [c for c in buyable
               if not _limit_up_blocked(c, float(opens_t.get(c)))]
    if not buyable:
        return cash, closed
    wsum = sum(max(0.0, float(target_weights[c])) for c in buyable)
    if wsum <= 1e-12:
        w_re = {c: 1.0 / len(buyable) for c in buyable}
    else:
        w_re = {c: max(0.0, float(target_weights[c])) / wsum for c in buyable}
    base_cash = cash  # post-liquidation pool to allocate
    # weight_at_entry denominator = execution-time total equity (survivors
    # marked at today's open where available). NOT the loop-mutated `cash`
    # (pre-2026-07 bug: later buys divided by an already-depleted pool).
    surv_val = 0.0
    for pos in positions.values():
        mark = opens_t.get(pos.code)
        surv_val += pos.shares * (
            float(mark) if pd.notna(mark) else pos.entry_price)
    equity_exec = max(base_cash + surv_val, 1e-12)
    for code in buyable:
        px = float(opens_t.get(code))
        alloc = base_cash * w_re[code]
        committed = alloc * (1 - costs.buy_cost)
        shares = committed / px if px > 0 else 0.0
        if shares <= 0:
            continue
        positions[code] = _Position(
            code=code, entry_idx=t, entry_date=date_t,
            entry_price=px, shares=shares,
            weight_at_entry=alloc / equity_exec,
        )
        cash -= alloc
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
