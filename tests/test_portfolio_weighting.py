"""Tests for portfolio/weighting.py — Ledoit-Wolf cov + box-constrained MVO."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.portfolio.weighting import (
    ledoit_wolf_cov,
    solve_mvo,
    compute_target_weights,
)


# ---------------- Ledoit-Wolf covariance ----------------

def _rng(seed=0):
    return np.random.default_rng(seed)


def test_lw_symmetric_and_psd():
    X = _rng(1).normal(size=(200, 8))
    cov = ledoit_wolf_cov(X)
    assert cov.shape == (8, 8)
    assert np.allclose(cov, cov.T)
    eig = np.linalg.eigvalsh(cov)
    assert (eig >= -1e-10).all()  # PSD


def test_lw_shrinks_more_when_few_observations():
    # T just above N -> ill-conditioned sample cov -> strong shrinkage toward
    # diagonal. T >> N -> close to the sample covariance.
    N = 10
    Xfew = _rng(2).normal(size=(12, N))
    Xmany = _rng(3).normal(size=(5000, N))
    cov_few = ledoit_wolf_cov(Xfew)
    cov_many = ledoit_wolf_cov(Xmany)
    S_few = np.cov(Xfew, rowvar=False, bias=True)
    S_many = np.cov(Xmany, rowvar=False, bias=True)
    # Off-diagonal mass is pulled toward 0 more in the few-obs case.
    off_few = np.abs(cov_few - np.diag(np.diag(cov_few))).sum()
    offS_few = np.abs(S_few - np.diag(np.diag(S_few))).sum()
    assert off_few < offS_few  # shrinkage reduced off-diagonal magnitude
    # Many-obs estimate is close to the sample covariance.
    assert np.linalg.norm(cov_many - S_many) < np.linalg.norm(cov_few - S_few)


def test_lw_degenerate_single_row():
    X = np.array([[0.1, 0.2, 0.3]])
    cov = ledoit_wolf_cov(X)  # T=1 -> diagonal of (zero) variances
    assert cov.shape == (3, 3)
    assert np.allclose(cov, 0.0)


# ---------------- MVO solver ----------------

def test_mvo_respects_box_and_simplex():
    mu = np.array([0.03, 0.01, -0.02, 0.005])
    cov = np.diag([0.04, 0.04, 0.04, 0.04])
    w = solve_mvo(mu, cov, risk_aversion=5.0, w_max=0.5)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.5 + 1e-9).all()


def test_mvo_prefers_high_score_when_risk_cheap():
    # Equal variances, uncorrelated -> ranking driven by mu; top score gets most.
    mu = np.array([0.05, 0.0, 0.0, 0.0])
    cov = np.diag([0.04, 0.04, 0.04, 0.04])
    w = solve_mvo(mu, cov, risk_aversion=1.0, w_max=0.9)
    assert np.argmax(w) == 0
    assert w[0] > 0.4


def test_mvo_high_risk_aversion_diversifies():
    mu = np.array([0.05, 0.0, 0.0, 0.0])
    cov = np.diag([0.04, 0.04, 0.04, 0.04])
    w_lo = solve_mvo(mu, cov, risk_aversion=1.0, w_max=0.9)
    w_mid = solve_mvo(mu, cov, risk_aversion=200.0, w_max=0.9)
    w_hi = solve_mvo(mu, cov, risk_aversion=5000.0, w_max=0.9)
    # Monotone: higher risk aversion -> lower concentration (more diversified).
    assert w_hi.max() < w_mid.max() < w_lo.max()
    # In the risk-dominated limit it approaches equal weight (1/N).
    assert abs(w_hi.max() - 0.25) < 0.1


def test_mvo_infeasible_box_falls_back_equal():
    mu = np.array([0.03, 0.01, 0.02])
    cov = np.eye(3) * 0.04
    # 3 assets * w_max 0.2 = 0.6 < 1 -> infeasible -> equal weight.
    w = solve_mvo(mu, cov, w_max=0.2)
    assert np.allclose(w, 1.0 / 3)


def test_mvo_nonfinite_falls_back_equal():
    mu = np.array([0.03, np.nan, 0.02])
    cov = np.eye(3) * 0.04
    w = solve_mvo(mu, cov, w_max=0.9)
    assert np.allclose(w, 1.0 / 3)


def test_mvo_single_asset():
    assert np.allclose(solve_mvo(np.array([0.1]), np.array([[0.04]])), [1.0])


# ---------------- compute_target_weights ----------------

def _ret_window(codes, T=60, seed=7):
    rng = _rng(seed)
    return pd.DataFrame(
        rng.normal(scale=0.02, size=(T, len(codes))),
        columns=list(codes),
        index=pd.date_range("2020-01-01", periods=T, freq="B"),
    )


def test_ctw_equal_is_exact():
    codes = ["a", "b", "c", "d"]
    rw = _ret_window(codes)
    w = compute_target_weights(codes, {c: 1.0 for c in codes}, rw, method="equal")
    assert set(w) == set(codes)
    assert all(abs(v - 0.25) < 1e-12 for v in w.values())


def test_ctw_empty():
    assert compute_target_weights([], {}, pd.DataFrame()) == {}


def test_ctw_mvo_sums_to_one_and_in_box():
    codes = [f"s{i}" for i in range(10)]
    rw = _ret_window(codes, T=120)
    scores = {c: float(i) for i, c in enumerate(codes)}
    w = compute_target_weights(
        codes, scores, rw, method="mvo", risk_aversion=10.0, w_max=0.3, min_obs=20,
    )
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(-1e-9 <= v <= 0.3 + 1e-6 for v in w.values())


def test_ctw_mvo_too_few_obs_falls_back_equal():
    codes = ["a", "b", "c"]
    rw = _ret_window(codes, T=5)  # < min_obs
    w = compute_target_weights(codes, {c: 1.0 for c in codes}, rw,
                               method="mvo", min_obs=20, w_max=0.9)
    assert all(abs(v - 1.0 / 3) < 1e-12 for v in w.values())


def test_ctw_mvo_missing_code_gets_weight_and_renormalizes():
    codes = ["a", "b", "c"]
    rw = _ret_window(["a", "b"], T=120)  # "c" absent from return window
    scores = {"a": 1.0, "b": 0.5, "c": 0.2}
    w = compute_target_weights(codes, scores, rw, method="mvo", w_max=0.9, min_obs=20)
    assert set(w) == set(codes)
    assert "c" in w and w["c"] > 0
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_ctw_unknown_method_raises():
    with pytest.raises(ValueError):
        compute_target_weights(["a", "b"], {}, _ret_window(["a", "b"]), method="foo")
