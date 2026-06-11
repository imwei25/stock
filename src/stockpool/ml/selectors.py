"""Factor selectors.

A ``FactorSelector`` decides which factors survive based on training data.
Currently the only implementation is ``LassoSelector`` (L1-regularised
coordinate descent, no sklearn dependency).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from stockpool.ml.dataset import (
    standardize_apply,
    standardize_fit,
)


class FactorSelector(ABC):
    """Drop irrelevant factors based on (X, y) training data."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def selected_factors(self) -> list[str]: ...


def _soft_threshold(x: float, lam: float) -> float:
    """Soft-thresholding operator used in coordinate-descent Lasso."""
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def _coordinate_descent_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Plain coordinate-descent solver for
        argmin_w 1/(2n) ||y - X w||^2 + alpha ||w||_1

    Assumes X has been standardised (zero mean, unit std) and y is centered.
    """
    n, p = X.shape
    w = np.zeros(p)
    # Pre-compute column norms (Xj.T @ Xj / n). With standardised X, == 1.
    col_norm_sq = (X * X).sum(axis=0) / n
    # Avoid division by zero on degenerate columns.
    col_norm_sq = np.where(col_norm_sq < 1e-12, 1.0, col_norm_sq)

    residual = y - X @ w
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            # Add back j-th contribution.
            residual = residual + X[:, j] * w[j]
            rho_j = (X[:, j] @ residual) / n
            new_wj = _soft_threshold(rho_j, alpha) / col_norm_sq[j]
            max_delta = max(max_delta, abs(new_wj - w[j]))
            w[j] = new_wj
            residual = residual - X[:, j] * w[j]
        if max_delta < tol:
            break
    return w


class LassoSelector(FactorSelector):
    """L1-regularised linear regression for factor selection.

    Standardises X (zero mean, unit std) and centers y before solving:

        argmin_w 1/(2n) ||y_c - X_s w||^2 + alpha ||w||_1

    where ``X_s`` is standardised and ``y_c`` is mean-centered. Factors with
    a coefficient of (approximately) zero are dropped.

    Attributes after ``fit``:
      * ``coef_``: pd.Series of raw coefficients indexed by factor name (in
        the standardised space — magnitudes are comparable across factors).
      * ``selected_``: list of factor names with non-zero coefficients.
    """

    def __init__(
        self,
        alpha: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-6,
        coef_threshold: float = 1e-8,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_threshold = coef_threshold

        self.coef_: pd.Series | None = None
        self.selected_: list[str] = []
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: float | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if X.empty:
            self.coef_ = pd.Series(dtype=float)
            self.selected_ = []
            self._feature_names = list(X.columns)
            return

        names = list(X.columns)
        Xn = X.to_numpy(dtype=float, copy=True)
        yn = y.to_numpy(dtype=float, copy=True)

        x_mean, x_std = standardize_fit(Xn)
        Xs = standardize_apply(Xn, x_mean, x_std)
        y_mean = float(yn.mean())
        y_centered = yn - y_mean
        # P3-12: y 标准化到单位方差再解 Lasso。否则 alpha 是绝对量,与
        # ρ_j·σ_y 同量级 —— 选中因子数随市场波动状态(σ_y)漂移:高波动期
        # 全选、低波动期全杀。标准化后 alpha 作用在相关系数尺度上,跨期
        # 语义稳定。选择(非零模式)是我们要的输出;coef_ 仍按原 y 尺度
        # 回乘,便于诊断展示。
        y_std = float(y_centered.std())
        y_unit = y_centered / y_std if y_std > 1e-12 else y_centered

        w = _coordinate_descent_lasso(
            Xs, y_unit, alpha=self.alpha,
            max_iter=self.max_iter, tol=self.tol,
        )
        if y_std > 1e-12:
            w = w * y_std  # 回到原 y 尺度,coef_ 语义不变

        self._x_mean = x_mean
        self._x_std = x_std
        self._y_mean = y_mean
        self._feature_names = names
        self.coef_ = pd.Series(w, index=names, name="lasso_coef")
        # 选择阈值在"单位 y 方差"空间判定(与 alpha 同尺度),避免回乘
        # y_std 后小 σ_y 时全部跌破绝对阈值。
        w_unit = w / y_std if y_std > 1e-12 else w
        self.selected_ = [
            n for n, c in zip(names, w_unit) if abs(c) > self.coef_threshold
        ]

    def selected_factors(self) -> list[str]:
        return list(self.selected_)


class LightGBMSelector(FactorSelector):
    """Tree-based selector using LightGBM gain importance.

    ``fit()`` trains a regression LightGBM on (X, y);
    ``selected_factors()`` returns columns whose normalized gain importance
    is in the top-K AND >= ``max_importance * min_importance_ratio``.

    Attributes after ``fit``:
      * ``coef_``: pd.Series of normalized gain importance (sums to 1.0 in
        non-degenerate fits; sums to 0 when y is constant or empty).
      * ``selected_``: list of factor names that passed the top-K + ratio gate.

    Lazy import: ``import lightgbm`` happens inside ``fit`` so this module
    can be imported without lightgbm installed (only fitting requires it).
    """

    def __init__(
        self,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        learning_rate: float = 0.05,
        num_iterations: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
        top_k_factors: int = 20,
        min_importance_ratio: float = 0.01,
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
        if top_k_factors <= 0:
            raise ValueError(f"top_k_factors must be > 0, got {top_k_factors}")
        if not (0 <= min_importance_ratio <= 1):
            raise ValueError(
                f"min_importance_ratio must be in [0, 1], got {min_importance_ratio}"
            )

        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.random_state = random_state
        self.top_k_factors = top_k_factors
        self.min_importance_ratio = min_importance_ratio
        self.verbose = verbose

        self.coef_: pd.Series | None = None
        self.selected_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb  # lazy import — ImportError surfaces only at fit

        if X.empty or len(y) == 0:
            self.coef_ = pd.Series(dtype=float)
            self.selected_ = []
            return

        feature_names = list(X.columns)
        dataset = lgb.Dataset(
            X.values, label=y.values, feature_name=feature_names,
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
        booster = lgb.train(params, dataset, num_boost_round=self.num_iterations)
        gain = booster.feature_importance(importance_type="gain").astype(float)

        total = float(gain.sum())
        if total < 1e-12:
            # Constant y or no learnable signal → no selection.
            self.coef_ = pd.Series(0.0, index=feature_names, name="lgb_importance")
            self.selected_ = []
            return

        norm = gain / total
        self.coef_ = pd.Series(norm, index=feature_names, name="lgb_importance")

        max_val = float(self.coef_.max())
        threshold = max_val * self.min_importance_ratio
        ranked = self.coef_.sort_values(ascending=False)
        eligible = ranked[ranked >= threshold].head(self.top_k_factors)
        self.selected_ = list(eligible.index)

    def selected_factors(self) -> list[str]:
        return list(self.selected_)
