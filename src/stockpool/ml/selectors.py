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

        w = _coordinate_descent_lasso(
            Xs, y_centered, alpha=self.alpha,
            max_iter=self.max_iter, tol=self.tol,
        )

        self._x_mean = x_mean
        self._x_std = x_std
        self._y_mean = y_mean
        self._feature_names = names
        self.coef_ = pd.Series(w, index=names, name="lasso_coef")
        self.selected_ = [
            n for n, c in zip(names, w) if abs(c) > self.coef_threshold
        ]

    def selected_factors(self) -> list[str]:
        return list(self.selected_)
