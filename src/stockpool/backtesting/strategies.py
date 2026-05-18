"""Built-in strategies for the backtesting framework.

Adding a new strategy:
    1. Subclass ``Strategy`` (from ``stockpool.backtesting.framework``).
    2. Implement ``name``, ``generate_signals``, ``should_enter``, ``should_exit``.
    3. Pass an instance to ``BacktestEngine``.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.backtesting.framework import (
    BarContext, PositionContext, Strategy,
)
from stockpool.config import (
    IndicatorsConfig, MLFactorConfig, ScoringConfig,
    VerdictsConfig, WeightsConfig,
)
from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.ml.dataset import (
    align_xy, build_factor_matrix, build_panel, forward_return,
)
from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import LassoSelector
from stockpool.ml.weighters import (
    EqualWeighter, FactorWeighter, ICWeighter, IRWeighter,
)
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
    Timer reset (while long): signal in ``refresh_verdicts``
        (defaults to ``strong_buy``) — restarts the N-day hold from this bar.
        Pass ``refresh_verdicts=()`` to opt out and keep the old
        "ignore signals while long" behavior.
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
        refresh_verdicts: tuple[str, ...] = ("strong_buy",),
    ):
        self.weights = weights
        self.scoring = scoring
        self.verdicts_cfg = verdicts_cfg
        self.indicators_cfg = indicators_cfg
        self.buy_verdicts = set(buy_verdicts)
        self.sell_verdicts = set(sell_verdicts)
        self.refresh_verdicts = set(refresh_verdicts)

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

    def should_reset_timer(self, ctx: PositionContext) -> bool:
        return ctx.signal in self.refresh_verdicts


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
        refresh_verdicts: tuple[str, ...] = ("strong_buy",),
        name: str = "verdict_execution",
    ):
        self._name = name
        self.buy_verdicts = set(buy_verdicts)
        self.sell_verdicts = set(sell_verdicts)
        self.refresh_verdicts = set(refresh_verdicts)

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

    def should_reset_timer(self, ctx: PositionContext) -> bool:
        return ctx.signal in self.refresh_verdicts


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


def _build_weighter(cfg) -> FactorWeighter:
    """Translate WeighterConfig → concrete FactorWeighter."""
    if cfg.type == "ic":
        return ICWeighter(use_rank=cfg.use_rank, min_abs_ic=cfg.min_abs_ic)
    if cfg.type == "ir":
        return IRWeighter(
            n_chunks=cfg.n_chunks, use_rank=cfg.use_rank,
            min_abs_ir=cfg.min_abs_ir,
        )
    if cfg.type == "equal":
        return EqualWeighter()
    raise ValueError(f"unknown weighter type: {cfg.type!r}")


class MLFactorStrategy(Strategy):
    """Walk-forward two-step ML factor strategy.

    Per bar ``t``:

      1. **Refit** the ``TwoStepPipeline`` (Lasso → IC/IR/equal) on the most
         recent ``train_window`` samples where the forward return is observable
         (i.e. trained on bars whose date ≤ date[t] - horizon).
      2. **Predict** the score for the current bar's factor row.
      3. **Map** the score to a verdict via the *training-set* quantiles (so
         the discretisation adapts to the fit-time distribution).

    Refit cadence: every ``refit_every`` bars. Between refits the most recent
    pipeline + quantiles are reused, predict-only.

    Panel mode:

      * ``per_stock``: training window comes from this stock's own history.
      * ``pooled``: training panel is built from ``pool_data`` (a mapping of
        ``{code: daily_df}``) plus this stock's history, truncated to bars
        whose date < the current decision date.

    Look-ahead safety: factors at bar ``i`` use only ``daily_df.iloc[:i+1]``;
    forward-return labels exclude bars whose future close lies past the
    truncation point; pool truncation uses strict ``date < current_date`` so
    other stocks contribute only past data.
    """

    def __init__(
        self,
        cfg: MLFactorConfig,
        pool_data: Mapping[str, pd.DataFrame] | None = None,
        current_stock_code: str | None = None,
    ):
        self.cfg = cfg
        self.pool_data: dict[str, pd.DataFrame] = dict(pool_data or {})
        self._current_stock_code = current_stock_code
        self.buy_verdicts = set(cfg.buy_verdicts)
        self.sell_verdicts = set(cfg.sell_verdicts)
        self.refresh_verdicts = set(cfg.refresh_verdicts)

    @property
    def name(self) -> str:
        return f"ml_factor_{self.cfg.weighter.type}_{self.cfg.panel_mode}"

    def with_stock(self, code: str) -> "MLFactorStrategy":
        """Return a copy bound to a specific stock (used in pooled mode)."""
        return MLFactorStrategy(
            cfg=self.cfg, pool_data=self.pool_data, current_stock_code=code,
        )

    def generate_signals(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        cols = ["date", "close", "signal", "score"]
        n = len(daily_df)
        if n == 0:
            return pd.DataFrame(columns=cols)

        # Factor matrix and labels are computed once on the full history.
        # The walk-forward training slices use `.iloc` cuts that never look
        # past the current bar, so this is still look-ahead-safe.
        X_full = build_factor_matrix(daily_df, cfg.factors)
        y_full = forward_return(daily_df, cfg.horizon)

        pipeline: TwoStepPipeline | None = None
        quantiles: dict[str, float] | None = None
        last_fit_bar = -10**9

        rows: list[dict] = []
        for i in range(n):
            date_i = daily_df["date"].iloc[i]
            close_i = float(daily_df["close"].iloc[i])

            if (
                (pipeline is None or (i - last_fit_bar) >= cfg.refit_every)
                and i - cfg.horizon >= cfg.min_train_samples
            ):
                fitted = self._try_fit(daily_df, X_full, y_full, i)
                if fitted is not None:
                    pipeline, quantiles = fitted
                    last_fit_bar = i

            signal = "neutral"
            score_value: float = float("nan")
            if pipeline is not None and quantiles is not None:
                xi_row = X_full.iloc[[i]]
                if bool(xi_row.notna().all(axis=1).iloc[0]):
                    pred = float(pipeline.predict(xi_row).iloc[0])
                    score_value = pred
                    signal = _classify_by_quantile(pred, quantiles)

            rows.append({
                "date": date_i, "close": close_i,
                "signal": signal, "score": score_value,
            })

        return pd.DataFrame(rows, columns=cols)

    def _try_fit(
        self,
        daily_df: pd.DataFrame,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        current_bar: int,
    ) -> tuple[TwoStepPipeline, dict[str, float]] | None:
        cfg = self.cfg
        # Labels are NaN for the last `horizon` rows of any window; we
        # exclude those from training to avoid using unobserved futures.
        label_end = current_bar - cfg.horizon
        if label_end <= 0:
            return None

        if cfg.panel_mode == "per_stock":
            train_start = max(0, label_end - cfg.train_window)
            X_train_raw = X_full.iloc[train_start:label_end]
            y_train_raw = y_full.iloc[train_start:label_end]
            X_train, y_train = align_xy(X_train_raw, y_train_raw)
        else:
            # In pooled mode `train_window` is the per-stock recency window:
            # each pool stock contributes at most its last `train_window`
            # post-warmup rows, then we concatenate. Total training rows
            # ≈ train_window × (# stocks with usable history at current_date).
            current_date = daily_df["date"].iloc[current_bar]
            pool = self._build_truncated_pool(daily_df, current_date, current_bar)
            X_pool, y_pool = build_panel(pool, cfg.factors, cfg.horizon)
            if len(X_pool) > 0 and cfg.train_window > 0:
                X_pool = X_pool.groupby(
                    level="stock", group_keys=False, sort=False,
                ).tail(cfg.train_window)
                y_pool = y_pool.loc[X_pool.index]
            X_train, y_train = X_pool, y_pool

        if len(X_train) < cfg.min_train_samples:
            return None

        pipeline = TwoStepPipeline(
            selector=LassoSelector(
                alpha=cfg.selector.alpha,
                max_iter=cfg.selector.max_iter,
                tol=cfg.selector.tol,
            ),
            weighter=_build_weighter(cfg.weighter),
        )
        pipeline.fit(X_train, y_train)
        train_preds = pipeline.predict(X_train)
        q = {
            "strong_buy":  float(train_preds.quantile(cfg.thresholds.strong_buy)),
            "buy":         float(train_preds.quantile(cfg.thresholds.buy)),
            "sell":        float(train_preds.quantile(cfg.thresholds.sell)),
            "strong_sell": float(train_preds.quantile(cfg.thresholds.strong_sell)),
        }
        return pipeline, q

    def _build_truncated_pool(
        self, daily_df: pd.DataFrame, current_date, current_bar: int,
    ) -> dict[str, pd.DataFrame]:
        """All pool stocks truncated to ``date < current_date``; the host
        stock's truncation is via row index (``iloc[:current_bar]``)."""
        out: dict[str, pd.DataFrame] = {}
        for code, df in self.pool_data.items():
            if code == self._current_stock_code:
                continue
            mask = df["date"] < current_date
            sub = df.loc[mask].reset_index(drop=True)
            if len(sub) > 0:
                out[code] = sub
        host_key = self._current_stock_code or "_self_"
        out[host_key] = daily_df.iloc[:current_bar].reset_index(drop=True)
        return out

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal in self.buy_verdicts

    def should_exit(self, ctx: PositionContext) -> bool:
        return ctx.signal in self.sell_verdicts

    def should_reset_timer(self, ctx: PositionContext) -> bool:
        return ctx.signal in self.refresh_verdicts


def _classify_by_quantile(pred: float, q: dict[str, float]) -> str:
    """Map a predicted score to one of strong_buy / buy / neutral / sell / strong_sell.

    Order of checks matters: ``strong_*`` thresholds are stricter and must be
    tested before ``buy`` / ``sell``.
    """
    if pred >= q["strong_buy"]:
        return "strong_buy"
    if pred >= q["buy"]:
        return "buy"
    if pred <= q["strong_sell"]:
        return "strong_sell"
    if pred <= q["sell"]:
        return "sell"
    return "neutral"
