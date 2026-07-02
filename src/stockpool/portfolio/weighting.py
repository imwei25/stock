"""Portfolio weighting schemes for ``PortfolioEngine`` (PR-P1-1).

Two schemes:

* ``equal`` — 1/K over the selected target (legacy behavior).
* ``mvo``  — mean-variance optimization with a Ledoit-Wolf shrunk covariance,
  long-only, box-constrained (``0 <= w_i <= w_max``), fully invested
  (``sum w = 1``).

Design note: **no new dependencies**. Ledoit-Wolf shrinkage is the closed-form
2004 "honey, I shrunk the covariance" estimator implemented in pure numpy; the
small (K~=20 asset) box-constrained QP is solved with ``scipy.optimize`` (SLSQP),
which is already a project dependency. This mirrors the project precedent of a
pure-numpy ``RidgeWeighter`` rather than pulling in sklearn / cvxpy.

The optimizer maximizes ``mu^T w - (gamma/2) w^T Sigma w`` where ``mu`` is the
cross-sectional score vector (expected-return proxy, standardized) and ``Sigma``
is the LW-shrunk daily-return covariance over a trailing window. On any
degenerate input (too few return observations, singular system, solver failure,
infeasible box) it falls back to equal weight — the engine must always get a
valid simplex weight vector.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage covariance toward a scaled identity.

    Args:
        returns: ``(T, N)`` array of (not necessarily demeaned) returns. Rows =
            observations, columns = assets.

    Returns:
        ``(N, N)`` shrunk covariance estimate, symmetric PSD.

    Falls back to the sample covariance when ``T < 2``. The shrinkage intensity
    is clipped to ``[0, 1]``. Norms use the trace inner product normalized by
    ``N`` (``<A, B> = trace(A B^T) / N``), per Ledoit-Wolf.
    """
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2:
        raise ValueError("returns must be 2-D (T, N)")
    T, N = X.shape
    if T < 2 or N == 0:
        # Not enough to estimate covariance — diagonal of variances (or zeros).
        var = np.var(X, axis=0, ddof=0) if T >= 1 else np.zeros(N)
        return np.diag(var)

    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T  # MLE sample covariance (N, N)

    mu = np.trace(S) / N
    # d^2 = || S - mu I ||^2  (normalized Frobenius)
    d2 = np.sum((S - mu * np.eye(N)) ** 2) / N

    # b̄^2 = (1/T^2) sum_t || x_t x_t^T - S ||^2  (normalized Frobenius)
    b_bar2 = 0.0
    for t in range(T):
        xt = Xc[t][:, None]
        diff = xt @ xt.T - S
        b_bar2 += np.sum(diff ** 2) / N
    b_bar2 /= T ** 2

    b2 = min(b_bar2, d2)        # b^2 = min(b̄^2, d^2)
    a2 = d2 - b2               # a^2 = d^2 - b^2
    shrink = (b2 / d2) if d2 > 0 else 0.0
    shrink = float(np.clip(shrink, 0.0, 1.0))

    target = mu * np.eye(N)
    cov = shrink * target + (1.0 - shrink) * S
    # Numerical symmetry guard.
    return 0.5 * (cov + cov.T)


def solve_mvo(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    risk_aversion: float = 10.0,
    w_max: float = 0.15,
) -> np.ndarray:
    """Long-only, box-constrained mean-variance weights.

    Maximizes ``mu^T w - (risk_aversion / 2) * w^T cov w`` subject to
    ``sum(w) == 1`` and ``0 <= w_i <= w_max``.

    Returns equal weights on any degenerate / infeasible / solver-failure case
    (caller always receives a valid simplex vector summing to 1).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    n = mu.size
    eq = np.full(n, 1.0 / n) if n else np.array([])
    if n == 0:
        return eq
    if n == 1:
        return np.array([1.0])
    # Box must be able to reach sum == 1: n * w_max >= 1, else infeasible.
    if w_max * n < 1.0 - 1e-9:
        return eq
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(cov))):
        return eq

    from scipy.optimize import minimize

    # Standardize mu so risk_aversion has a stable scale across rebalances.
    sd = mu.std()
    mu_z = (mu - mu.mean()) / sd if sd > 1e-12 else np.zeros(n)

    def neg_obj(w: np.ndarray) -> float:
        return -(mu_z @ w) + 0.5 * risk_aversion * (w @ cov @ w)

    def neg_grad(w: np.ndarray) -> np.ndarray:
        return -mu_z + risk_aversion * (cov @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones(n)},)
    bounds = [(0.0, w_max)] * n
    try:
        res = minimize(
            neg_obj, eq, jac=neg_grad, bounds=bounds,
            constraints=constraints, method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-9},
        )
    except Exception:  # noqa: BLE001 — never let the optimizer crash a backtest
        return eq
    if not res.success or not np.all(np.isfinite(res.x)):
        return eq
    w = np.clip(res.x, 0.0, w_max)
    s = w.sum()
    if s <= 1e-12:
        return eq
    return w / s


def compute_target_weights(
    codes: Sequence[str],
    scores: Mapping[str, float],
    return_window: pd.DataFrame,
    *,
    method: str = "equal",
    risk_aversion: float = 10.0,
    w_max: float = 0.15,
    min_obs: int = 20,
) -> dict[str, float]:
    """Map a selected target set to ``{code: weight}`` summing to 1.

    Args:
        codes: selected target codes (order-independent).
        scores: ``{code: score}`` used as the expected-return proxy ``mu``.
        return_window: trailing wide returns frame (index=date, columns=code).
            Only the intersection with ``codes`` is used; covariance is built
            from columns with no NaN over the window.
        method: ``"equal"`` or ``"mvo"``.
        min_obs: minimum return rows required for ``mvo``; below this, equal.

    ``mvo`` falls back to equal weight whenever the covariance cannot be built
    (too few observations, missing columns) so the engine always gets a valid
    weight vector. Codes missing from ``return_window`` keep equal weight.
    """
    ordered = list(codes)
    n = len(ordered)
    if n == 0:
        return {}
    equal = {c: 1.0 / n for c in ordered}
    if method == "equal" or n == 1:
        return equal
    if method != "mvo":
        raise ValueError(f"unknown weighting method: {method!r}")

    # Build the return matrix for codes present in the window with full coverage.
    present = [c for c in ordered if c in return_window.columns]
    if len(present) < 2:
        return equal
    sub = return_window[present]
    sub = sub.dropna(axis=0, how="any")
    if len(sub) < min_obs:
        return equal
    usable = list(sub.columns)
    if len(usable) < 2:
        return equal

    mu = np.array([float(scores.get(c, 0.0)) for c in usable])
    cov = ledoit_wolf_cov(sub.to_numpy())
    w = solve_mvo(mu, cov, risk_aversion=risk_aversion, w_max=w_max)

    weights = {c: float(wi) for c, wi in zip(usable, w)}
    # Codes selected but absent from the covariance set (no return history):
    # give them the average usable weight, then renormalize the whole vector.
    missing = [c for c in ordered if c not in weights]
    if missing:
        fill = 1.0 / n
        for c in missing:
            weights[c] = fill
    total = sum(weights.values())
    if total <= 1e-12:
        return equal
    return {c: weights[c] / total for c in weights}
