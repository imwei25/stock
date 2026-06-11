"""Factor weighters: re-weight selected factors by predictive power.

A ``FactorWeighter`` consumes the (already filtered) factor matrix and target,
learns one weight per factor, and produces a per-bar prediction via a simple
weighted sum on standardised inputs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from stockpool.ml.dataset import standardize_apply, standardize_fit


class FactorWeighter(ABC):
    """Assign a weight to each factor and produce a composite score."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def weights(self) -> pd.Series: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-bar per-factor contribution to ``predict(X)``.

        Linear weighters return ``standardised(X) * weights`` (row sums equal
        ``predict(X)`` by construction). Non-linear weighters (e.g. LightGBM)
        return their model-specific decomposition, e.g. SHAP values.
        """


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Returns 0 if either side is constant."""
    if len(a) < 2:
        return 0.0
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa = ra.std(ddof=0)
    sb = rb.std(ddof=0)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    sa = a.std(ddof=0)
    sb = b.std(ddof=0)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


# 单日截面至少要有这么多只股票,日 IC 才有统计意义;不足的日子跳过。
_MIN_OBS_PER_DAY = 5


def _date_level_values(index: pd.Index) -> np.ndarray | None:
    """从训练样本 index 提取 date level;无法提取(per_stock 普通索引)返回 None。"""
    if isinstance(index, pd.MultiIndex) and "date" in (index.names or []):
        return index.get_level_values("date").to_numpy()
    return None


def _daily_ic_matrix(
    Xs: np.ndarray,
    y_np: np.ndarray,
    dates: np.ndarray,
    corr_fn,
) -> tuple[np.ndarray, np.ndarray] | None:
    """逐日截面 IC 矩阵 (D × F) 及对应日期数组 (D,)。

    与选因子工具 (factors_analysis.compute_daily_ic) 同一统计量:每个交易日
    在横截面上算一次 corr,再做时间维聚合 —— 而不是把 (stock,date) 长表
    池化成一个相关系数(那会让时序变异与截面变异混杂、高离散度日主导)。
    截面 < _MIN_OBS_PER_DAY 的日子跳过;没有任何有效日时返回 None,
    调用方回退池化相关(per_stock 单股场景)。
    """
    order = np.argsort(dates, kind="stable")
    dates_sorted = dates[order]
    uniq, starts = np.unique(dates_sorted, return_index=True)
    ends = np.append(starts[1:], len(dates_sorted))

    day_ics: list[np.ndarray] = []
    day_keys: list = []
    F = Xs.shape[1]
    for u, lo, hi in zip(uniq, starts, ends):
        rows = order[lo:hi]
        if len(rows) < _MIN_OBS_PER_DAY:
            continue
        yb = y_np[rows]
        day_ics.append(np.array([corr_fn(Xs[rows, j], yb) for j in range(F)]))
        day_keys.append(u)
    if not day_ics:
        return None
    return np.vstack(day_ics), np.asarray(day_keys)


class _StandardisingMixin:
    """Shared standardisation helpers — fit on training, replay on predict."""

    def _fit_standardiser(self, X: pd.DataFrame) -> np.ndarray:
        self._feature_names = list(X.columns)
        Xn = X.to_numpy(dtype=float, copy=True)
        self._x_mean, self._x_std = standardize_fit(Xn)
        return standardize_apply(Xn, self._x_mean, self._x_std)

    def _apply_standardiser(self, X: pd.DataFrame) -> np.ndarray:
        if self._x_mean is None or self._x_std is None or self._feature_names is None:
            raise RuntimeError("Weighter not fitted yet")
        # Reorder columns to fit-time order; missing cols filled with 0 (mean).
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(
                f"predict() missing columns from fit: {missing}"
            )
        Xn = X[self._feature_names].to_numpy(dtype=float, copy=True)
        return standardize_apply(Xn, self._x_mean, self._x_std)


class _LinearWeighterContributionsMixin:
    """Shared ``contributions()`` impl for linear-combination weighters
    (IC / IR / Equal). Returns ``standardised(X) * weights`` per cell.

    Depends on the class to provide:
      * ``self._weights`` — pd.Series of per-factor weights
      * ``self._feature_names`` — list[str] of fit-time feature names
      * ``self._apply_standardiser(X)`` — z-score apply (from _StandardisingMixin)
    """

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._weights is None or self._weights.empty:
            return pd.DataFrame(index=X.index)
        Xs = self._apply_standardiser(X)
        w = self._weights.to_numpy()
        return pd.DataFrame(
            Xs * w, index=X.index, columns=self._feature_names,
        )


class EqualWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Equal weight (1/k) on standardised factors. Useful as a baseline."""

    def __init__(self):
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            return
        k = len(self._feature_names)
        self._weights = pd.Series(
            [1.0 / k] * k, index=self._feature_names, name="weight",
        )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class ICWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Weight each factor by its IC with the target.

    By default uses Spearman rank IC (more robust to outliers in returns).
    Set ``use_rank=False`` for Pearson IC. Weights are L1-normalised
    (``Σ|w| = 1``) so prediction magnitudes are comparable across refits.

    Sign of the weight follows the sign of the IC (a negative-IC factor is
    inverted), so the resulting score is monotone-increasing in expected return.
    """

    def __init__(self, use_rank: bool = True, min_abs_ic: float = 0.0):
        self.use_rank = use_rank
        self.min_abs_ic = min_abs_ic
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None
        self._ic: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            self._ic = pd.Series(dtype=float)
            return

        y_np = y.to_numpy(dtype=float)
        corr_fn = _spearman_corr if self.use_rank else _pearson_corr
        # P2-1: pooled (stock,date) 训练集 → 逐日截面 IC 的时间均值,与
        # factors_analysis 的选因子统计量一致;无 date level(per_stock
        # 单股)→ 池化时序相关(单股本就没有截面)。
        dates = _date_level_values(X.index)
        daily = (
            _daily_ic_matrix(Xs, y_np, dates, corr_fn)
            if dates is not None else None
        )
        if daily is not None:
            ic_values = daily[0].mean(axis=0)
        else:
            ic_values = np.array(
                [corr_fn(Xs[:, j], y_np) for j in range(Xs.shape[1])]
            )
        self._ic = pd.Series(ic_values, index=self._feature_names, name="ic")

        masked = np.where(np.abs(ic_values) >= self.min_abs_ic, ic_values, 0.0)
        total = np.sum(np.abs(masked))
        if total < 1e-12:
            # Fall back to equal weight if no factor crosses the IC threshold.
            k = len(self._feature_names)
            self._weights = pd.Series(
                [1.0 / k] * k, index=self._feature_names, name="weight",
            )
        else:
            self._weights = pd.Series(
                masked / total, index=self._feature_names, name="weight",
            )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    @property
    def ic(self) -> pd.Series:
        """The fit-time IC of each factor (signed)."""
        assert self._ic is not None
        return self._ic.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class IRWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Weight each factor by its information ratio over rolling sub-windows.

    IR_i = mean(IC_i) / std(IC_i) computed over equal-sized chunks of the
    training window. Falls back to ICWeighter behaviour when only one chunk
    fits.

    Lower the noise sensitivity of plain IC weighting at the cost of needing
    a longer training window (recommend ≥ 5 * n_chunks bars).
    """

    def __init__(
        self, n_chunks: int = 6, use_rank: bool = True, min_abs_ir: float = 0.0,
    ):
        if n_chunks <= 0:
            raise ValueError(f"n_chunks must be > 0, got {n_chunks}")
        self.n_chunks = n_chunks
        self.use_rank = use_rank
        self.min_abs_ir = min_abs_ir
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None
        self._ir: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            self._ir = pd.Series(dtype=float)
            return

        corr_fn = _spearman_corr if self.use_rank else _pearson_corr
        y_np = y.to_numpy(dtype=float)
        F = Xs.shape[1]

        # P2-1: pooled 训练集 → 逐日截面 IC 序列,再按**自然时间**切块聚合
        # IR = mean(日IC)/std(日IC)。旧实现按行号切块,行序是 stock-major 时
        # 每块是"一组股票的全历史",算出的是跨股稳定性而非时间稳定性,且
        # 同配置因构建路径不同而语义漂移 —— 现按 date level 聚合,与行序解耦。
        dates = _date_level_values(X.index)
        daily = (
            _daily_ic_matrix(Xs, y_np, dates, corr_fn)
            if dates is not None else None
        )
        ir_values = np.zeros(F)
        if daily is not None:
            ic_mat, _day_keys = daily  # 已按日期升序
            d = ic_mat.shape[0]
            n_chunks = min(self.n_chunks, max(1, d // 2))
            bounds = np.linspace(0, d, n_chunks + 1, dtype=int)
            for j in range(F):
                chunk_ics = [
                    ic_mat[bounds[c]:bounds[c + 1], j].mean()
                    for c in range(n_chunks)
                    if bounds[c + 1] - bounds[c] >= 1
                ]
                if len(chunk_ics) < 2:
                    ir_values[j] = ic_mat[:, j].mean()
                    continue
                arr = np.array(chunk_ics)
                std = arr.std(ddof=0)
                ir_values[j] = arr.mean() if std < 1e-12 else arr.mean() / std
        else:
            # per_stock 单股(普通 date 索引):行序即时间序,按行号切块仍然
            # 是时间块 —— 保留 legacy 行为。
            n = Xs.shape[0]
            n_chunks = min(self.n_chunks, max(1, n // 5))
            bounds = np.linspace(0, n, n_chunks + 1, dtype=int)
            for j in range(F):
                chunk_ics = []
                for c in range(n_chunks):
                    lo, hi = bounds[c], bounds[c + 1]
                    if hi - lo < 2:
                        continue
                    chunk_ics.append(corr_fn(Xs[lo:hi, j], y_np[lo:hi]))
                if len(chunk_ics) < 2:
                    ir_values[j] = corr_fn(Xs[:, j], y_np)
                    continue
                arr = np.array(chunk_ics)
                std = arr.std(ddof=0)
                ir_values[j] = arr.mean() if std < 1e-12 else arr.mean() / std

        self._ir = pd.Series(ir_values, index=self._feature_names, name="ir")
        masked = np.where(np.abs(ir_values) >= self.min_abs_ir, ir_values, 0.0)
        total = np.sum(np.abs(masked))
        if total < 1e-12:
            k = len(self._feature_names)
            self._weights = pd.Series(
                [1.0 / k] * k, index=self._feature_names, name="weight",
            )
        else:
            self._weights = pd.Series(
                masked / total, index=self._feature_names, name="weight",
            )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    @property
    def ir(self) -> pd.Series:
        """Per-factor information ratio (mean(IC) / std(IC) across chunks)."""
        assert self._ir is not None
        return self._ir.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class LightGBMWeighter(FactorWeighter):
    """Tree-based weighter using LightGBM.

    ``fit(X, y)`` trains a regression LGB and caches mean|SHAP| as ``_weights``
    (computed once on training data, returned by ``weights()``).
    ``predict(X)`` runs ``booster.predict(X.values)``.
    ``contributions(X)`` runs ``booster.predict(X.values, pred_contrib=True)``
    and returns per-feature SHAP values (drops the trailing base-value column).

    Unlike linear weighters, this class does NOT inherit
    ``_StandardisingMixin`` — LightGBM is scale-invariant. Look-ahead safety
    rests on the same ABC contract: predict only consumes X, never y.

    Lazy import: ``import lightgbm`` happens inside ``fit()`` so the module
    can be imported without lightgbm installed.
    """

    def __init__(
        self,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        learning_rate: float = 0.05,
        num_iterations: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
        verbose: int = -1,
    ):
        if num_leaves <= 1:
            raise ValueError(f"num_leaves must be > 1, got {num_leaves}")
        if min_data_in_leaf <= 0:
            raise ValueError(f"min_data_in_leaf must be > 0, got {min_data_in_leaf}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be > 0, got {num_iterations}")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {max_depth}")

        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose

        self._booster = None
        self._feature_names: list[str] | None = None
        self._weights: pd.Series | None = None  # cached mean|SHAP|

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb  # lazy import

        if X.empty or len(y) == 0:
            self._feature_names = list(X.columns)
            self._weights = pd.Series(dtype=float)
            self._booster = None
            return

        self._feature_names = list(X.columns)
        dataset = lgb.Dataset(
            X.values, label=y.values, feature_name=self._feature_names,
        )
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "seed": self.random_state,
            "verbose": self.verbose,
        }
        self._booster = lgb.train(
            params, dataset, num_boost_round=self.num_iterations,
        )

        # Cache mean|SHAP| as weights (per Q1+Q5 design decisions).
        # pred_contrib returns shape (n, n_features + 1); last col is base value.
        contribs = self._booster.predict(X.values, pred_contrib=True)
        feature_contribs = contribs[:, :-1]
        mean_abs = np.abs(feature_contribs).mean(axis=0)
        self._weights = pd.Series(
            mean_abs, index=self._feature_names, name="lgb_mean_abs_shap",
        )

    def weights(self) -> pd.Series:
        if self._weights is None:
            raise RuntimeError("Weighter not fitted yet")
        return self._weights.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._booster is None:
            return pd.Series(0.0, index=X.index)
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"predict() missing columns: {missing}")
        Xn = X[self._feature_names].values
        preds = self._booster.predict(Xn)
        return pd.Series(preds, index=X.index, name="score")

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._booster is None:
            return pd.DataFrame(index=X.index)
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"contributions() missing columns: {missing}")
        Xn = X[self._feature_names].values
        contribs = self._booster.predict(Xn, pred_contrib=True)
        return pd.DataFrame(
            contribs[:, :-1], index=X.index, columns=self._feature_names,
        )
