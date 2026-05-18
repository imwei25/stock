"""Built-in strategies for the backtesting framework.

Adding a new strategy:
    1. Subclass ``Strategy`` (from ``stockpool.backtesting.framework``).
    2. Implement ``name``, ``generate_signals``, ``should_enter``, ``should_exit``.
    3. Pass an instance to ``BacktestEngine``.
"""
from __future__ import annotations

import pandas as pd

from stockpool.backtesting.framework import (
    BarContext, PositionContext, Strategy,
)
from stockpool.config import (
    IndicatorsConfig, ScoringConfig, VerdictsConfig, WeightsConfig,
)
from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.signals import (
    combine_daily_weekly, detect_signals, score_triggers, verdict_of,
)


DAILY_WARMUP = 30   # mirrors cli.py::_analyze_one
WEEKLY_WARMUP = 30


class CompositeVerdictStrategy(Strategy):
    """The project's original composite-score strategy.

    The per-bar signal is the verdict (strong_buy / buy / neutral / sell /
    strong_sell) that the live pipeline would have produced on that bar's
    history. Look-ahead-safe by construction: each bar reconstructs the verdict
    from ``daily_df.iloc[:i+1]`` only.

    Entry: signal in ``buy_verdicts`` (defaults to ``buy``, ``strong_buy``).
    Exit: signal in ``sell_verdicts`` (defaults to ``sell``, ``strong_sell``).
    The engine separately bounds the hold by ``max_holding_days``.
    """

    def __init__(
        self,
        weights: WeightsConfig,
        scoring: ScoringConfig,
        verdicts_cfg: VerdictsConfig,
        indicators_cfg: IndicatorsConfig,
        buy_verdicts: tuple[str, ...] = ("buy", "strong_buy"),
        sell_verdicts: tuple[str, ...] = ("sell", "strong_sell"),
    ):
        self.weights = weights
        self.scoring = scoring
        self.verdicts_cfg = verdicts_cfg
        self.indicators_cfg = indicators_cfg
        self.buy_verdicts = set(buy_verdicts)
        self.sell_verdicts = set(sell_verdicts)

    @property
    def name(self) -> str:
        return "composite_verdict"

    def generate_signals(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        empty_cols = ["date", "close", "signal", "daily_score", "weekly_score", "final_score"]
        if len(daily_df) < DAILY_WARMUP:
            return pd.DataFrame(columns=empty_cols)

        enriched_daily = add_all(daily_df, self.indicators_cfg)
        rows: list[dict] = []

        for i in range(DAILY_WARMUP - 1, len(daily_df)):
            daily_window = enriched_daily.iloc[:i + 1]
            daily_score = score_triggers(detect_signals(daily_window, self.weights))

            weekly = resample_to_weekly(daily_df.iloc[:i + 1])
            if len(weekly) >= WEEKLY_WARMUP:
                enriched_w = add_all(weekly, self.indicators_cfg)
                weekly_score = score_triggers(detect_signals(enriched_w, self.weights))
            else:
                weekly_score = 0

            final = combine_daily_weekly(daily_score, weekly_score, self.scoring)
            verdict = verdict_of(final, self.verdicts_cfg)

            rows.append({
                "date": daily_df["date"].iloc[i],
                "close": float(daily_df["close"].iloc[i]),
                "signal": verdict,
                "daily_score": int(daily_score),
                "weekly_score": int(weekly_score),
                "final_score": float(final),
            })

        return pd.DataFrame(rows, columns=empty_cols)

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal in self.buy_verdicts

    def should_exit(self, ctx: PositionContext) -> bool:
        return ctx.signal in self.sell_verdicts


class VerdictExecution(Strategy):
    """Execution-only adapter for pre-generated verdict frames.

    Use this when you already have a signal frame (e.g. cached from a previous
    run, or produced by an external pipeline) and only need the engine's
    decision rules. ``generate_signals`` raises — feed signals via
    ``BacktestEngine.run_on_signals`` instead of ``run``.
    """

    def __init__(
        self,
        buy_verdicts: tuple[str, ...] = ("buy", "strong_buy"),
        sell_verdicts: tuple[str, ...] = ("sell", "strong_sell"),
        name: str = "verdict_execution",
    ):
        self._name = name
        self.buy_verdicts = set(buy_verdicts)
        self.sell_verdicts = set(sell_verdicts)

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__} is execution-only — call "
            "BacktestEngine.run_on_signals(signals, ...) with a pre-generated frame."
        )

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal in self.buy_verdicts

    def should_exit(self, ctx: PositionContext) -> bool:
        return ctx.signal in self.sell_verdicts


class SMACrossStrategy(Strategy):
    """Reference example: classic SMA golden/dead cross.

    Demonstrates that the framework generalises beyond the composite pipeline.
    The signal column is ``"buy"`` / ``"sell"`` / ``"hold"``.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
            )
        self.fast = fast_period
        self.slow = slow_period

    @property
    def name(self) -> str:
        return f"sma_cross_{self.fast}_{self.slow}"

    def generate_signals(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        cols = ["date", "close", "signal", "sma_fast", "sma_slow"]
        if len(daily_df) < self.slow:
            return pd.DataFrame(columns=cols)

        df = daily_df.copy()
        df["sma_fast"] = df["close"].rolling(self.fast).mean()
        df["sma_slow"] = df["close"].rolling(self.slow).mean()
        df = df.dropna(subset=["sma_fast", "sma_slow"]).reset_index(drop=True)

        signals: list[str] = []
        prev_fast = df["sma_fast"].shift(1)
        prev_slow = df["sma_slow"].shift(1)
        for i in range(len(df)):
            pf, ps = prev_fast.iloc[i], prev_slow.iloc[i]
            cf, cs = df["sma_fast"].iloc[i], df["sma_slow"].iloc[i]
            if pd.isna(pf) or pd.isna(ps):
                signals.append("hold")
            elif pf <= ps and cf > cs:
                signals.append("buy")
            elif pf >= ps and cf < cs:
                signals.append("sell")
            else:
                signals.append("hold")

        df["signal"] = signals
        return df[cols].reset_index(drop=True)

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal == "buy"

    def should_exit(self, ctx: PositionContext) -> bool:
        return ctx.signal == "sell"
