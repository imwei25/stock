"""Built-in strategies for the backtesting framework.

Adding a new strategy:
    1. Subclass ``Strategy`` (from ``stockpool.backtesting.framework``).
    2. Implement ``name``, ``generate_signals``, ``should_enter``, ``should_exit``.
    3. Pass an instance to ``BacktestEngine``.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
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
    forward_return_panel, slice_stock_factor_matrix, stack_panel_to_xy,
)
from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import FactorSelector, LassoSelector, LightGBMSelector
from stockpool.ml.weighters import (
    EqualWeighter, FactorWeighter, ICWeighter, IRWeighter, LightGBMWeighter,
)
from stockpool.signals import (
    Trigger, combine_daily_weekly, detect_signals, score_triggers, verdict_of,
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
        empty_cols = ["date", "open", "close", "signal", "daily_score", "weekly_score", "final_score"]
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
                "open": float(daily_df["open"].iloc[i]),
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

    def predict_latest(self, daily_df: pd.DataFrame) -> dict:
        """Single-bar verdict for the daily report (skips the walk-forward)."""
        if len(daily_df) < DAILY_WARMUP:
            return {
                "signal": "neutral",
                "daily_score": 0, "weekly_score": 0, "final_score": 0.0,
            }
        enriched = add_all(daily_df, self.indicators_cfg)
        d_score = score_triggers(detect_signals(enriched, self.weights))
        weekly = resample_to_weekly(daily_df)
        if len(weekly) >= WEEKLY_WARMUP:
            w_score = score_triggers(
                detect_signals(add_all(weekly, self.indicators_cfg), self.weights)
            )
        else:
            w_score = 0
        final = combine_daily_weekly(d_score, w_score, self.scoring)
        verdict = verdict_of(final, self.verdicts_cfg)
        return {
            "signal": verdict,
            "daily_score": int(d_score),
            "weekly_score": int(w_score),
            "final_score": float(final),
        }


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
        cols = ["date", "open", "close", "signal", "sma_fast", "sma_slow"]
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
    """Translate WeighterConfig → concrete FactorWeighter (PR-B2 subnested)."""
    if cfg.type == "ic":
        return ICWeighter(use_rank=cfg.ic.use_rank, min_abs_ic=cfg.ic.min_abs_ic)
    if cfg.type == "ir":
        return IRWeighter(
            n_chunks=cfg.ir.n_chunks,
            use_rank=cfg.ir.use_rank,
            min_abs_ir=cfg.ir.min_abs_ir,
        )
    if cfg.type == "equal":
        return EqualWeighter()
    if cfg.type == "lightgbm":
        c = cfg.lightgbm
        return LightGBMWeighter(
            num_leaves=c.num_leaves,
            min_data_in_leaf=c.min_data_in_leaf,
            learning_rate=c.learning_rate,
            num_iterations=c.num_iterations,
            max_depth=c.max_depth,
            random_state=c.random_state,
            verbose=c.verbose,
        )
    raise ValueError(f"unknown weighter type: {cfg.type!r}")


def _build_selector(cfg) -> FactorSelector:
    """Translate SelectorConfig → concrete FactorSelector."""
    if cfg.type == "lasso":
        return LassoSelector(
            alpha=cfg.lasso.alpha,
            max_iter=cfg.lasso.max_iter,
            tol=cfg.lasso.tol,
        )
    if cfg.type == "lightgbm":
        c = cfg.lightgbm
        return LightGBMSelector(
            num_leaves=c.num_leaves,
            min_data_in_leaf=c.min_data_in_leaf,
            learning_rate=c.learning_rate,
            num_iterations=c.num_iterations,
            max_depth=c.max_depth,
            random_state=c.random_state,
            top_k_factors=c.top_k_factors,
            min_importance_ratio=c.min_importance_ratio,
            verbose=c.verbose,
        )
    raise ValueError(f"unknown selector type: {cfg.type!r}")


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
        factor_panel: Mapping[str, pd.DataFrame] | None = None,
        close_panel: pd.DataFrame | None = None,
        cache_dir: str | Path | None = None,
        shared_cache: dict | None = None,
    ):
        self.cfg = cfg
        self.pool_data: dict[str, pd.DataFrame] = dict(pool_data or {})
        self._current_stock_code = current_stock_code
        # 可选: 跨股票预算好的因子面板 (name -> T×N wide frame)。
        # 提供时,WQ101 cross-sec 因子在 predict 阶段也走真实横截面值;
        # 不提供时,fall back 到 build_factor_matrix 单股退化 (cross-sec → 常数)。
        self._factor_panel: dict[str, pd.DataFrame] | None = (
            dict(factor_panel) if factor_panel is not None else None
        )
        # 可选: 预算好的 close 宽表 (T×N)。提供时,pooled mode 的 _try_fit
        # 直接切 factor_panel + close_panel 拼训练集,跳过每个 refit_bar 的
        # build_panel(pool, factors, horizon) 全量因子重算 (PR-1 速度优化)。
        self._close_panel: pd.DataFrame | None = (
            close_panel.copy() if close_panel is not None else None
        )
        self.buy_verdicts = set(cfg.buy_verdicts)
        self.sell_verdicts = set(cfg.sell_verdicts)
        self.refresh_verdicts = set(cfg.refresh_verdicts)
        # 缓存目录: 启用月度训练复用 (daily-report path)。None 表示不缓存。
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        # CLI 跨股票共享的进程内缓存(由调用方传入空 dict 并复用)。
        # 当前用于 pooled 模式下复用同一 refit-bar 的 (pipeline, quantiles)。
        self._shared_cache: dict | None = shared_cache

    @property
    def name(self) -> str:
        return f"ml_factor_{self.cfg.weighter.type}_{self.cfg.panel_mode}"

    def with_stock(self, code: str) -> "MLFactorStrategy":
        """Return a copy bound to a specific stock (used in pooled mode)."""
        return MLFactorStrategy(
            cfg=self.cfg, pool_data=self.pool_data, current_stock_code=code,
            factor_panel=self._factor_panel, close_panel=self._close_panel,
            cache_dir=self._cache_dir, shared_cache=self._shared_cache,
        )

    def _strategy_signature(self) -> str:
        """8-char hash of MLFactorConfig — used to invalidate stale caches
        when factors/horizon/selector/weighter/etc. change."""
        blob = repr(self.cfg.model_dump()).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:8]

    def _is_sharing(self) -> bool:
        """是否走跨股共享 fit:pooled + 配置开关 + 拿到了 pool_data。"""
        return (
            self.cfg.panel_mode == "pooled"
            and getattr(self.cfg, "share_pool_fit", False)
            and bool(self.pool_data)
        )

    @property
    def _label_lag(self) -> int:
        """标签向前看的 bar 数。close 基准 = horizon;open 基准的标签
        open[t+1+h]/open[t+1] 多看 1 根 bar(进场移到 t+1),= horizon + 1。
        embargo / 截断 / warmup 数学都必须用它而不是裸 horizon。"""
        extra = 1 if getattr(self.cfg, "label_basis", "close") == "open" else 0
        return self.cfg.horizon + extra

    def _open_panel_for_labels(self) -> pd.DataFrame | None:
        """label_basis=open 时给 fast path 提供与 close_panel 对齐的 open 宽表。

        复用 ``_build_ohlcv_from_pool_data``(已带 shared_cache);pool_data
        缺失时返回 None,调用方回退 close 基准并 warning 一次。"""
        if getattr(self.cfg, "label_basis", "close") != "open":
            return None
        ohlcv = self._build_ohlcv_from_pool_data()
        if ohlcv is None:
            if not getattr(self, "_warned_open_fallback", False):
                import logging
                logging.getLogger(__name__).warning(
                    "label_basis=open 但缺少 pool_data/close_panel,无法构造 "
                    "open 面板;训练标签回退 close 基准。"
                )
                self._warned_open_fallback = True
            return None
        return ohlcv["open"]

    def _build_ohlcv_from_pool_data(self) -> dict | None:
        """Build a T×N OHLCV panel from ``self.pool_data`` for mask computation.

        Reuses ``self._close_panel.index`` as the authoritative date axis
        (so the resulting mask aligns 1:1 with the cached factor/close panels).
        Returns ``None`` if pool_data or close_panel is missing.
        Cached in ``self._shared_cache`` under a strategy-sig key.
        """
        if not self.pool_data or self._close_panel is None:
            return None
        cache_key = ("__ohlcv_for_mask__", self._strategy_signature())
        if (
            self._shared_cache is not None
            and cache_key in self._shared_cache
        ):
            return self._shared_cache[cache_key]
        per_stock: dict[str, pd.DataFrame] = {}
        for code, df in self.pool_data.items():
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"])
            per_stock[code] = d.set_index("date").sort_index()
        idx = self._close_panel.index
        panel: dict[str, pd.DataFrame] = {}
        for field in ("open", "high", "low", "close", "volume"):
            panel[field] = pd.DataFrame(
                {c: d[field].reindex(idx) for c, d in per_stock.items()},
                index=idx,
            )
        if self._shared_cache is not None:
            self._shared_cache[cache_key] = panel
        return panel

    def _get_ipo_dates(self) -> dict | None:
        """Load IPO dates for ``_listing_mask`` (avoids first_valid_index
        heuristic that mis-flags mature stocks with short cache history).

        Only loaded when mask is enabled + cache_dir is set. Cached in
        ``shared_cache`` (process-local) so successive ``_try_fit`` calls
        on different stocks reuse the same dict. Returns ``None`` if
        unavailable — ``_listing_mask`` falls back to its heuristic.
        """
        if not self.cfg.mask.enabled or self._cache_dir is None:
            return None
        if self._shared_cache is not None and "__ipo_dates__" in self._shared_cache:
            return self._shared_cache["__ipo_dates__"]
        return self._load_ipo_dates_uncached()

    def _load_ipo_dates_uncached(self) -> dict | None:
        try:
            from stockpool.ipo_dates import load_or_build_ipo_dates
            dates = load_or_build_ipo_dates(self._cache_dir)
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load IPO dates (%s); listing_mask will fall back "
                "to first_valid_index heuristic", e,
            )
            dates = None
        if self._shared_cache is not None:
            self._shared_cache["__ipo_dates__"] = dates
        return dates

    def _get_st_codes(self) -> set | None:
        """当前 ST 代码集合(干净名单),供涨跌停 mask 用 ±5% 阈值(P2-23)。

        与 ``_get_ipo_dates`` 同样的加载/缓存策略;不可用时返回 None
        (mask 退化为按板块阈值,不致命)。
        """
        if not self.cfg.mask.enabled or self._cache_dir is None:
            return None
        if self._shared_cache is not None and "__st_codes__" in self._shared_cache:
            return self._shared_cache["__st_codes__"]
        try:
            from stockpool.ipo_dates import load_st_codes
            st = load_st_codes(self._cache_dir)
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load ST codes (%s); limit mask uses board thresholds only", e,
            )
            st = None
        if self._shared_cache is not None:
            self._shared_cache["__st_codes__"] = st
        return st

    def _cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        sig = self._strategy_signature()
        if self._is_sharing():
            return self._cache_dir / "ml_models" / f"{sig}_shared.pkl"
        if self._current_stock_code is None:
            return None
        return self._cache_dir / "ml_models" / f"{sig}_{self._current_stock_code}.pkl"

    def _load_cached_pipeline(self):
        p = self._cache_path()
        if p is None or not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cached_pipeline(self, pipeline, quantiles, fit_date) -> None:
        p = self._cache_path()
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(
                {"pipeline": pipeline, "quantiles": quantiles,
                 "fit_date": pd.Timestamp(fit_date)},
                f,
            )

    def predict_latest(self, daily_df: pd.DataFrame) -> dict:
        """Today's verdict with monthly model refit.

        Loads a pickled ``(pipeline, quantiles, fit_date)`` from
        ``<cache_dir>/ml_models/<sig>_<code>.pkl``. Refits only if the cache
        is missing OR the cached ``fit_date`` falls in a different calendar
        month from ``daily_df``'s last bar; otherwise predict-only.
        """
        if len(daily_df) == 0:
            return {"signal": "neutral", "score": float("nan")}
        X_full = self._build_x_full(daily_df)
        y_full = forward_return(daily_df, self.cfg.horizon, basis=self.cfg.label_basis)
        current_bar = len(daily_df) - 1
        today = pd.to_datetime(daily_df["date"].iloc[-1])

        # Try shared in-memory cache first (so all 8 stocks in one CLI run
        # share the same month's fit without even touching disk).
        shared_key = self._shared_key(today)
        hit = None
        if shared_key is not None and self._shared_cache is not None:
            hit = self._shared_cache.get(shared_key)

        if hit is not None:
            pipeline, quantiles = hit
        else:
            cached = self._load_cached_pipeline()
            same_month = (
                cached is not None
                and (cached["fit_date"].year, cached["fit_date"].month)
                == (today.year, today.month)
            )
            if same_month:
                pipeline = cached["pipeline"]
                quantiles = cached["quantiles"]
                if shared_key is not None and self._shared_cache is not None:
                    self._shared_cache[shared_key] = (pipeline, quantiles)
            else:
                fitted = self._try_fit(daily_df, X_full, y_full, current_bar)
                if fitted is None:
                    return {"signal": "neutral", "score": float("nan")}
                pipeline, quantiles = fitted
                self._save_cached_pipeline(pipeline, quantiles, today)

        xi_row = X_full.iloc[[-1]]
        # Partial NaN tolerance: impute missing factor values with 0 so a
        # single long-warmup factor (e.g. alpha_037 with 200d rolling corr)
        # can't gate the entire predict path. 0 is the neutral value for
        # IC-weighted dot products. Skip only when every factor is NaN —
        # there is nothing to predict from.
        if not bool(xi_row.notna().any(axis=1).iloc[0]):
            return {"signal": "neutral", "score": float("nan")}
        xi_row = xi_row.fillna(0.0)
        pred = float(pipeline.predict(xi_row).iloc[0])
        signal = _classify_by_quantile(pred, quantiles)
        triggers = _ml_factor_triggers(pipeline, xi_row, top_n=8)
        return {
            "signal": signal, "score": pred, "final_score": pred,
            "triggers_daily": triggers, "triggers_weekly": [],
        }

    def _build_x_full(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """从 factor_panel(若有)切出本股 X;否则单股退化算。"""
        if self._factor_panel is not None and self._current_stock_code is not None:
            wide = slice_stock_factor_matrix(
                self._factor_panel, self._current_stock_code,
            )
            dates = pd.DatetimeIndex(pd.to_datetime(daily_df["date"]).values)
            X = wide.reindex(dates)
            X.index = pd.Index(daily_df["date"].reset_index(drop=True), name="date")
            return X
        return build_factor_matrix(daily_df, self.cfg.factors)

    def generate_signals(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        # ``final_score`` mirrors ``score`` and is consumed by the portfolio
        # scoring layer (``precompute_scores_from_legacy`` expects this column,
        # default ``score_field="final_score"``); keeping the duplicate column
        # avoids a portfolio-specific code path. Same convention as
        # ``predict_latest`` which also exposes both.
        cols = ["date", "open", "close", "signal", "score", "final_score"]
        n = len(daily_df)
        if n == 0:
            return pd.DataFrame(columns=cols)

        # Factor matrix and labels are computed once on the full history.
        # The walk-forward training slices use `.iloc` cuts that never look
        # past the current bar, so this is still look-ahead-safe.
        X_full = self._build_x_full(daily_df)
        y_full = forward_return(daily_df, cfg.horizon, basis=cfg.label_basis)

        pipeline: TwoStepPipeline | None = None
        quantiles: dict[str, float] | None = None
        last_fit_bar = -10**9

        rows: list[dict] = []
        for i in range(n):
            date_i = daily_df["date"].iloc[i]
            open_i = float(daily_df["open"].iloc[i])
            close_i = float(daily_df["close"].iloc[i])

            # Refit triggers: (a) no model yet, (b) bar-cadence reached, or
            # (c) sharing mode and this month not in shared_cache yet.
            shared_key_i = self._shared_key(date_i)
            need_month_fit = (
                shared_key_i is not None
                and self._shared_cache is not None
                and shared_key_i not in self._shared_cache
            )
            if (
                (pipeline is None or (i - last_fit_bar) >= cfg.refit_every or need_month_fit)
                and i - self._label_lag >= cfg.min_train_samples
            ):
                fitted = self._try_fit(daily_df, X_full, y_full, i)
                if fitted is not None:
                    pipeline, quantiles = fitted
                    last_fit_bar = i

            signal = "neutral"
            score_value: float = float("nan")
            if pipeline is not None and quantiles is not None:
                xi_row = X_full.iloc[[i]]
                # Partial NaN tolerance — see predict_latest for rationale.
                if bool(xi_row.notna().any(axis=1).iloc[0]):
                    xi_row = xi_row.fillna(0.0)
                    pred = float(pipeline.predict(xi_row).iloc[0])
                    score_value = pred
                    signal = _classify_by_quantile(pred, quantiles)

            rows.append({
                "date": date_i, "open": open_i, "close": close_i,
                "signal": signal, "score": score_value,
                "final_score": score_value,
            })

        return pd.DataFrame(rows, columns=cols)

    def _shared_key(self, current_date) -> tuple | None:
        """Month-granularity key for the cross-stock fit cache."""
        if not self._is_sharing() or self._shared_cache is None:
            return None
        ts = pd.Timestamp(current_date)
        return (self._strategy_signature(), int(ts.year), int(ts.month))

    def _embargoed_label_end(self, current_bar: int) -> int:
        """Return the bar index where training labels must stop, accounting for
        ``cfg.embargo_days``.

        Without embargo, labels are valid up to ``current_bar - label_lag``
        (close 基准 lag=horizon;open 基准标签 open[t+1+h]/open[t+1] 多看
        1 根 bar,lag=horizon+1)。With embargo ``E``, push another ``E`` bars
        back so the most recent training label's forward-return window ends
        at least ``E`` bars before the test bar — eliminating overlap when
        E ≥ horizon.

        ``embargo_days = None`` means "auto = horizon" (the default).
        ``embargo_days = 0`` reproduces pre-PR-A behavior.
        """
        cfg = self.cfg
        effective_embargo = (
            cfg.embargo_days if cfg.embargo_days is not None else cfg.horizon
        )
        return current_bar - self._label_lag - effective_embargo

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
        label_end = self._embargoed_label_end(current_bar)
        if label_end <= 0:
            return None

        # Shared in-memory cache: one fit per (sig, year, month) shared by
        # all stocks in the CLI run. Skip both pool build and Lasso refit
        # if the current bar's month already has a hit.
        current_date = daily_df["date"].iloc[current_bar]
        shared_key = self._shared_key(current_date)
        if shared_key is not None:
            hit = self._shared_cache.get(shared_key)  # type: ignore[union-attr]
            if hit is not None:
                return hit

        if cfg.panel_mode == "per_stock":
            train_start = max(0, label_end - cfg.train_window)
            X_train_raw = X_full.iloc[train_start:label_end]
            y_train_raw = y_full.iloc[train_start:label_end]
            X_train, y_train = align_xy(X_train_raw, y_train_raw)
        elif self._factor_panel is not None and self._close_panel is not None:
            # Pre-check: if the pre-stacked long panel doesn't have enough
            # rows up to the label cutoff date, the eventual X_train will be
            # too small no matter what. Skip the slice + groupby + tail work.
            # This short-circuits ~95% of early-bar attempts in long backtests.
            pre = self._ensure_pooled_xy_long()
            if pre is not None:
                label_iloc = max(0, label_end - 1)
                label_cutoff_ts = pd.Timestamp(daily_df["date"].iloc[label_iloc])
                dates_level = pre[0].index.get_level_values("date")
                n_up_to = int(dates_level.searchsorted(label_cutoff_ts, side="right"))
                if n_up_to < cfg.min_train_samples:
                    return None
            # Fast path (PR-1): slice precomputed factor + close panels by
            # cutoff_date instead of rebuilding factors on the truncated pool
            # at every refit_bar.
            X_pool, y_pool = self._build_pooled_xy_from_panel(
                daily_df, current_bar,
            )
            X_train, y_train = X_pool, y_pool
        else:
            # Legacy path: rebuild factors from raw OHLCV at every refit.
            # Kept for per_stock-mode tests and CLI paths that don't pre-build
            # the close panel.
            pool = self._build_truncated_pool(daily_df, current_date, current_bar)
            X_pool, y_pool = build_panel(
                pool, cfg.factors, cfg.horizon,
                mask_config=cfg.mask,
                ipo_dates=self._get_ipo_dates(),
                label_basis=cfg.label_basis,
                st_codes=self._get_st_codes(),
            )
            if len(X_pool) > 0 and cfg.train_window > 0:
                X_pool = X_pool.groupby(
                    level="stock", group_keys=False, sort=False,
                ).tail(cfg.train_window)
                y_pool = y_pool.loc[X_pool.index]
            X_train, y_train = X_pool, y_pool

        if len(X_train) < cfg.min_train_samples:
            return None

        pipeline = TwoStepPipeline(
            selector=_build_selector(cfg.selector),
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
        result = (pipeline, q)
        if shared_key is not None:
            self._shared_cache[shared_key] = result  # type: ignore[index]
        return result

    def _ensure_pooled_xy_long(self) -> tuple[pd.DataFrame, pd.Series] | None:
        """Build (or fetch from shared_cache) the FULL pooled long-format
        ``(X, y)`` once.

        Returns ``(X, y)`` indexed by ``MultiIndex(date, stock)`` (date is the
        outer level, sorted) — letting per-refit slicing become an O(log N)
        ``.loc[:cutoff_ts]`` instead of re-stacking 20 wide panels each time.

        Cached in ``self._shared_cache`` under a key derived from the strategy
        signature so all stocks in one backtest share the same prebuilt panel.
        Returns ``None`` when shared_cache or panels are unavailable — the
        caller falls back to the legacy per-call stacking path.
        """
        if (
            self._shared_cache is None
            or self._factor_panel is None
            or self._close_panel is None
        ):
            return None
        sig = self._strategy_signature()
        key = ("__pooled_xy_long__", sig)
        hit = self._shared_cache.get(key)
        if hit is not None:
            return hit
        # If mask enabled, build OHLCV from pool_data and compute label-side
        # bidirectional mask. Same logic as build_panel's slow path so
        # quantile fits match.
        mask: pd.DataFrame | None = None
        if (
            self.cfg.mask.enabled
            and self.pool_data
        ):
            ohlcv = self._build_ohlcv_from_pool_data()
            if ohlcv:
                from stockpool.panel import compute_tradability_mask
                mask = compute_tradability_mask(
                    ohlcv, self.cfg.mask, ipo_dates=self._get_ipo_dates(),
                    st_codes=self._get_st_codes(),
                )
        fwd = forward_return_panel(
            self._close_panel, self.cfg.horizon, mask=mask,
            open_=self._open_panel_for_labels(),
        )
        X, y = stack_panel_to_xy(self._factor_panel, fwd, dropna=True)
        # Original layout is (stock, date); swap so date is outer + sort so
        # ``.loc[:cutoff_ts]`` becomes a fast range cut on the leading level.
        X = X.swaplevel("stock", "date").sort_index()
        y = y.swaplevel("stock", "date")
        y = y.loc[X.index]
        self._shared_cache[key] = (X, y)
        return X, y

    def _build_pooled_xy_from_panel(
        self, daily_df: pd.DataFrame, current_bar: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fast pooled training-set builder (PR-1 + PR-3).

        PR-1 sliced ``self._factor_panel`` / ``self._close_panel`` per refit
        and re-stacked. PR-3 hoists that stack into ``shared_cache``: the full
        long-format ``(X, y)`` is built once and every refit just takes
        ``.loc[:cutoff_ts]`` + optional host-row filter + per-stock
        ``tail(train_window)``.

        Host handling (P3-11 统一): 非 sharing 模式下 host 的**截断历史**也
        留在训练集 —— 与 ``_build_truncated_pool``(legacy 路径)一致;此前
        fast path 把 host 整段剔除,同配置训练集会随缓存可用性而不同。
        host 截断历史的标签都在 cutoff 之前,无泄漏。

        Falls back to per-call stacking when ``shared_cache`` is unavailable
        (e.g. unit tests that construct a strategy without a shared cache).
        """
        assert self._factor_panel is not None and self._close_panel is not None
        cfg = self.cfg
        label_end = self._embargoed_label_end(current_bar)
        host_slice_end = max(0, label_end + self._label_lag)
        cutoff_ts = pd.Timestamp(
            daily_df["date"].iloc[host_slice_end - 1]
            if host_slice_end > 0 else daily_df["date"].iloc[0]
        )

        pre = self._ensure_pooled_xy_long()
        if pre is not None:
            # Fast path uses labels from the FULL forward_return panel — so
            # rows in (label_end - 1, label_end + horizon - 1] would carry
            # forward returns computed from close PAST cutoff_ts (look-ahead).
            # Slice by label_end - 1's date instead (= cutoff_ts - horizon in
            # date terms): same observable-label window as the legacy path.
            label_iloc = max(0, label_end - 1)
            label_cutoff_ts = pd.Timestamp(daily_df["date"].iloc[label_iloc])
            X_long, y_long = pre
            X = X_long.loc[:label_cutoff_ts]
            y = y_long.loc[X.index]
            if len(X) > 0 and cfg.train_window > 0:
                X = X.groupby(
                    level="stock", group_keys=False, sort=False,
                ).tail(cfg.train_window)
                y = y.loc[X.index]
            # Match legacy contract: MultiIndex levels (stock, date). swaplevel
            # is metadata-only — no data move — so the cost is negligible even
            # on 1M-row training frames.
            X = X.swaplevel("date", "stock")
            y = y.swaplevel("date", "stock")
            return X, y

        # Legacy fallback: per-call slice + stack (used when shared_cache
        # isn't provided, e.g. some single-stock unit tests). Host 截断历史
        # 同样保留(P3-11 统一,与 fast path / _build_truncated_pool 一致)。
        sliced_fp: dict[str, pd.DataFrame] = {}
        for name, wide in self._factor_panel.items():
            sliced_fp[name] = wide.loc[wide.index <= cutoff_ts]
        close_sub = self._close_panel.loc[self._close_panel.index <= cutoff_ts]

        open_full = self._open_panel_for_labels()
        open_sub = None
        if open_full is not None:
            open_sub = open_full.loc[open_full.index <= cutoff_ts]

        fwd = forward_return_panel(close_sub, cfg.horizon, open_=open_sub)
        X, y = stack_panel_to_xy(sliced_fp, fwd, dropna=True)
        if len(X) > 0 and cfg.train_window > 0:
            X = X.groupby(
                level="stock", group_keys=False, sort=False,
            ).tail(cfg.train_window)
            y = y.loc[X.index]
        return X, y

    def _build_truncated_pool(
        self, daily_df: pd.DataFrame, current_date, current_bar: int,
    ) -> dict[str, pd.DataFrame]:
        """All pool stocks truncated to ``date < current_date``; the host
        stock's truncation is via row index (``iloc[:current_bar]``).

        When ``share_pool_fit`` 开启,host 不再被排除——训练集对所有 host 一致,
        换得跨股 fit 复用。host 自己贡献 ~1/N 权重,IC 加权下偏差可忽略。
        """
        sharing = self._is_sharing()

        # Embargo: truncate pool stocks to data older than the host's
        # label_end date so labels can't reach into the embargo gap.
        label_end = self._embargoed_label_end(current_bar)
        # label_end may be <= 0 if there isn't enough history; caller (_try_fit)
        # already guards by returning None in that case, but be defensive.
        # host_slice_end is where we cut the host's data: it must include
        # `label_lag` extra bars beyond label_end so the labels at bars
        # [label_end - label_lag, label_end) have observable forward returns
        # (close 基准 lag=horizon;open 基准 lag=horizon+1)。
        host_slice_end = max(0, label_end + self._label_lag)
        cutoff_date = (
            daily_df["date"].iloc[host_slice_end - 1]
            if host_slice_end > 0 else
            daily_df["date"].iloc[0]
        )

        out: dict[str, pd.DataFrame] = {}
        for code, df in self.pool_data.items():
            if not sharing and code == self._current_stock_code:
                continue
            mask = df["date"] <= cutoff_date
            sub = df.loc[mask].reset_index(drop=True)
            if len(sub) > 0:
                out[code] = sub
        if not sharing:
            host_key = self._current_stock_code or "_self_"
            out[host_key] = daily_df.iloc[:host_slice_end].reset_index(drop=True)
        return out

    def should_enter(self, ctx: BarContext) -> bool:
        return ctx.signal in self.buy_verdicts

    def should_exit(self, ctx: PositionContext) -> bool:
        return ctx.signal in self.sell_verdicts

    def should_reset_timer(self, ctx: PositionContext) -> bool:
        return ctx.signal in self.refresh_verdicts


def _ml_factor_triggers(pipeline, xi_row: pd.DataFrame, top_n: int = 8) -> list[Trigger]:
    """Top-|contribution| factors at the latest bar, packaged as Trigger.

    Each Trigger represents one selected factor's signed contribution
    ``z_i * w_i`` to today's predicted score. ``weight`` is ``contribution × 100``
    rounded to int so the existing report renderer's ``(±N)`` badge stays
    consistent with the composite path.
    """
    try:
        contrib_row = pipeline.contributions(xi_row).iloc[0]
    except Exception:
        return []
    ranked = contrib_row.reindex(
        contrib_row.abs().sort_values(ascending=False).index
    )
    out: list[Trigger] = []
    for name, contrib in ranked.head(top_n).items():
        if not pd.notna(contrib) or contrib == 0.0:
            continue
        weight_int = int(round(float(contrib) * 100))
        if weight_int == 0:
            continue
        out.append(Trigger(
            signal_type=str(name),
            direction=1 if contrib > 0 else -1,
            weight=weight_int,
            description=f"因子 {name} 贡献 {float(contrib):+.3f}",
        ))
    return out


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
