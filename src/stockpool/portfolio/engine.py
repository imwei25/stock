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
    infer_limit_pct, open_hits_limit_down, open_hits_limit_up,
)
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
        # P1-3: 涨跌停拒单的 ST 集合(±5% 阈值);None = 仅按代码前缀推断。
        self.st_codes: set | None = None

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
        # P1-5: 逐 code 最后有效 close(估值/核销基准)与连续无报价计数。
        last_close: dict[str, float] = {}
        stale_bars: dict[str, int] = {}
        limit_pcts = {
            c: infer_limit_pct(str(c), self.st_codes) for c in closes.columns
        }

        for t in range(n_bars):
            date_t = dates[t]
            opens_t = opens.iloc[t]
            closes_t = closes.iloc[t]
            prev_closes_t = closes.iloc[t - 1] if t > 0 else closes.iloc[0]

            # 1. Execute pending trade decided on bar t-1.
            if pending_target is not None:
                cash, closed_now, turnover_val = _rebalance_to_target(
                    positions=positions, cash=cash,
                    target=pending_target,
                    opens_t=opens_t, prev_closes_t=prev_closes_t,
                    t=t, date_t=date_t,
                    costs=self.costs,
                    limit_pcts=limit_pcts,
                    min_commission=getattr(self.cfg, "min_commission", 0.0),
                    last_close=last_close,
                )
                closed.extend(closed_now)
                if rebalance_records:
                    base_eq = equity[t - 1] if t > 0 else float(self.cfg.initial_cash)
                    rebalance_records[-1]["turnover"] = turnover_val
                    rebalance_records[-1]["turnover_ratio"] = (
                        turnover_val / base_eq if base_eq > 0 else 0.0
                    )
                pending_target = None

            # 2. Mark-to-market at close[t]。无报价持仓按最后有效 close 计值
            #    (P1-5:旧实现回落 entry_price,停牌一天市值被打回入场价,
            #    制造虚假波动);连续 delist_after_bars 根无报价 → 强制核销。
            held_value = 0.0
            delist_after = getattr(self.cfg, "delist_after_bars", 60)
            for code, pos in list(positions.items()):
                close_t = closes_t.get(code)
                if pd.notna(close_t):
                    last_close[code] = float(close_t)
                    stale_bars[code] = 0
                    held_value += pos.shares * float(close_t)
                else:
                    stale_bars[code] = stale_bars.get(code, 0) + 1
                    mark = last_close.get(code, pos.entry_price)
                    if stale_bars[code] >= delist_after:
                        # 退市/无限期停牌:按最后有效价核销(现实退市整理期
                        # 往往更惨,这里偏乐观,见 config 注释)。
                        proceeds = pos.shares * mark * (1 - self.costs.sell_cost)
                        cash += proceeds
                        denom = pos.shares * pos.entry_price
                        closed.append(PortfolioTrade(
                            code=code, entry_date=pos.entry_date,
                            exit_date=date_t, entry_price=pos.entry_price,
                            exit_price=mark,
                            weight_at_entry=pos.weight_at_entry,
                            ret=float(proceeds / denom - 1) if denom > 0 else 0.0,
                            days_held=t - pos.entry_idx,
                            exit_reason="delisted",
                        ))
                        del positions[code]
                        continue
                    held_value += pos.shares * mark
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
                        closes_t=closes_t,
                        sector_map=self.sector_map,
                        max_per_industry=self.cfg.max_per_industry,
                    )
                    rebalance_records.append({
                        "date": date_t,
                        "target_codes": sorted(pending_target),
                        "num_target": len(pending_target),
                        "turnover": 0.0,  # 执行 bar 回填
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
                    price = last_close.get(code, pos.entry_price)  # P1-5
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
            else pd.DataFrame(columns=["date", "target_codes", "num_target", "turnover"])
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
        # P1-4: 换手率指标(此前根本没有统计,虚构换手的量级无从察觉)。
        ratios = [
            r.get("turnover_ratio", 0.0) for r in rebalance_records
            if r.get("turnover_ratio") is not None
        ]
        years = n_bars / 252.0 if n_bars > 0 else 0.0
        metrics["avg_turnover_per_rebalance"] = (
            float(np.mean(ratios)) if ratios else None
        )
        metrics["annualized_turnover"] = (
            float(np.sum(ratios) / years) if ratios and years > 0 else None
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
    closes_t: pd.Series,
    sector_map: Mapping[str, str] | None = None,
    max_per_industry: int | None = None,
) -> set[str]:
    """Take top-K by score using **information available at bar t** (P2-9).

    旧实现用 ``opens.iloc[t+1]`` 过滤候选 —— 决策时刻(t 收盘后)窥视了
    "明日是否停牌",且买不进时自动顺位补下一名(免费预知 + 乐观执行)。
    现在只按 ``close[t]`` 是否有报价过滤(今天在交易);t+1 真停牌的腿在
    执行层现金闲置,下次 rebalance 再补 —— 与真实下单一致。

    If ``max_per_industry`` and ``sector_map`` are both provided, applies a
    greedy per-industry cap during the top-K walk. ``"Unknown"`` semantics
    (matching Pool B): if *every* candidate has no sector mapping, the cap
    is skipped; else unmapped codes go into a single ``"Unknown"`` bucket.
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
        px = closes_t.get(code)
        if pd.isna(px):
            continue
        if apply_cap:
            ind = sector_map.get(code) or "Unknown"
            if industry_count.get(ind, 0) >= max_per_industry:
                continue
            industry_count[ind] = industry_count.get(ind, 0) + 1
        out.add(code)
    return out


def _order_cost(notional: float, rate: float, min_commission: float) -> float:
    """单笔订单费用:比例费率,带可选地板(P2-3 最低佣金近似)。"""
    return max(notional * rate, min_commission) if notional > 0 else 0.0


def _rebalance_to_target(
    positions: dict[str, _Position],
    cash: float,
    target: set[str],
    opens_t: pd.Series,
    prev_closes_t: pd.Series,
    t: int,
    date_t: pd.Timestamp,
    costs: TradeCosts,
    limit_pcts: Mapping[str, float] | None = None,
    min_commission: float = 0.0,
    last_close: Mapping[str, float] | None = None,
) -> tuple[float, list[PortfolioTrade], float]:
    """**差量调仓**(P1-4):只卖出局者、只买新进者,存活仓不动。

    旧实现"全清仓再买回":存活持仓每轮 rebalance 被虚构双边换手,默认
    成本下 ~0.21%/轮 × 年 ~50 轮 ≈ 10%/年的虚构拖累,且 PortfolioTrade
    全是 5 天碎片(days_held/胜率失真)。差量调仓后存活仓权重自然漂移
    (不再强制等权),与"买入后持有到被换出"的真实操作一致。

    执行约束(P1-3):卖出腿开盘一字跌停 → 卖不出,保留持仓下轮再试;
    买入腿开盘一字涨停 → 买不进,该腿现金闲置(不顺位替补)。

    Returns:
        (cash, closed_trades, turnover_value) — turnover_value =
        (卖出额 + 买入额) / 2,调用方据此算换手率。
    """
    closed: list[PortfolioTrade] = []
    sells_value = 0.0
    buys_value = 0.0
    limit_pcts = limit_pcts or {}
    last_close = last_close or {}

    # 1. 卖出:持仓中不在 target 的(出局者)。
    for code in list(positions.keys()):
        if code in target:
            continue  # 存活仓不动(差量语义)
        pos = positions[code]
        px = opens_t.get(code)
        if pd.isna(px):
            # 无报价(停牌)— 保留持仓,下轮再试。
            continue
        px = float(px)
        prev_c = prev_closes_t.get(code)
        lp = limit_pcts.get(code, 0.10)
        if pd.notna(prev_c) and open_hits_limit_down(px, float(prev_c), lp):
            continue  # 跌停卖不出(P1-3)
        gross = pos.shares * px
        fee = _order_cost(gross, costs.sell_cost, min_commission)
        proceeds = gross - fee
        cash += proceeds
        sells_value += gross
        notional = pos.shares * pos.entry_price
        ret = (proceeds / notional) - 1 if notional > 0 else 0.0
        closed.append(PortfolioTrade(
            code=code,
            entry_date=pos.entry_date,
            exit_date=date_t,
            entry_price=pos.entry_price,
            exit_price=px,
            weight_at_entry=pos.weight_at_entry,
            ret=float(ret),
            days_held=t - pos.entry_idx,
            exit_reason="rebalance_drop",
        ))
        del positions[code]

    # 2. 买入:target 中尚未持有的(新进者)。等分当前现金;买不进的腿
    #    (无报价/涨停)现金闲置 —— 分母用"计划买入腿数",不向可买腿
    #    再集中(那等价于顺位替补)。
    new_codes = [c for c in target if c not in positions]
    if not new_codes:
        return cash, closed, (sells_value + buys_value) / 2.0

    # P2-10: weight_at_entry 分母 = 执行时点组合总值(现金 + 存活仓按
    # 开盘价/最后有效价 mark),循环外固定 —— 旧实现用递减中的 cash,
    # 最后一只恒记 ~100%。
    survivors_value = 0.0
    for code, pos in positions.items():
        px = opens_t.get(code)
        mark = float(px) if pd.notna(px) else last_close.get(code, pos.entry_price)
        survivors_value += pos.shares * mark
    total_equity_exec = max(cash + survivors_value, 1e-12)

    per_lot = cash / len(new_codes)
    for code in new_codes:
        px = opens_t.get(code)
        if pd.isna(px):
            continue  # t+1 停牌,现金闲置
        px = float(px)
        prev_c = prev_closes_t.get(code)
        lp = limit_pcts.get(code, 0.10)
        if pd.notna(prev_c) and open_hits_limit_up(px, float(prev_c), lp):
            continue  # 涨停买不进(P1-3),现金闲置
        fee = _order_cost(per_lot, costs.buy_cost, min_commission)
        committed = per_lot - fee
        shares = committed / px if px > 0 else 0.0
        if shares <= 0:
            continue
        positions[code] = _Position(
            code=code, entry_idx=t, entry_date=date_t,
            entry_price=px, shares=shares,
            weight_at_entry=per_lot / total_equity_exec,
        )
        cash -= per_lot
        buys_value += committed
    return cash, closed, (sells_value + buys_value) / 2.0


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
