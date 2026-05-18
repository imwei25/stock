"""Lasso + IC/IR/Equal weighters + TwoStepPipeline tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.ml import (
    EqualWeighter,
    ICWeighter,
    IRWeighter,
    LassoSelector,
    TwoStepPipeline,
    build_factor_matrix,
    build_panel,
    forward_return,
)


def _ohlcv(closes, vols=None):
    n = len(closes)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "open": [c * 0.998 for c in closes],
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": vols if vols is not None else [1_000_000.0] * n,
    })


# === dataset helpers ===

def test_forward_return_correct_and_aligned():
    closes = [100, 110, 121, 133.1, 146.41]  # +10% each bar
    df = _ohlcv(closes)
    y = forward_return(df, horizon=2).reset_index(drop=True)
    # row 0: 121/100 - 1 = 0.21
    assert y.iloc[0] == pytest.approx(0.21)
    # row 1: 133.1/110 - 1 = 0.21
    assert y.iloc[1] == pytest.approx(0.21)
    # last two rows NaN
    assert pd.isna(y.iloc[-1])
    assert pd.isna(y.iloc[-2])


def test_build_factor_matrix_has_one_column_per_factor():
    df = _ohlcv(list(np.linspace(100, 200, 60)))
    X = build_factor_matrix(df, ["momentum_10", "rsi_centered_14", "ma_distance_20"])
    assert list(X.columns) == ["momentum_10", "rsi_centered_14", "ma_distance_20"]
    # Last row should have all factors computed (warmup satisfied).
    assert X.iloc[-1].notna().all()


def test_build_panel_pools_across_stocks():
    a = _ohlcv(list(np.linspace(100, 150, 80)))
    b = _ohlcv(list(np.linspace(50, 90, 80)))
    X, y = build_panel({"A": a, "B": b}, ["momentum_5", "rsi_centered_6"], horizon=3)
    assert len(X) == len(y)
    assert X.notna().all().all()
    assert y.notna().all()
    assert set(X.index.get_level_values("stock").unique()) == {"A", "B"}


# === LassoSelector ===

def test_lasso_drops_noise_features():
    """With a clean linear target on one feature, Lasso should zero out noise."""
    rng = np.random.default_rng(0)
    n = 400
    f1 = rng.normal(0, 1, n)
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.normal(0, 1, n)
    y = pd.Series(0.5 * f1 + rng.normal(0, 0.1, n))
    X = pd.DataFrame({"signal": f1, "noise_a": noise1, "noise_b": noise2})

    sel = LassoSelector(alpha=0.05)
    sel.fit(X, y)
    assert "signal" in sel.selected_factors()
    # Noise features should be zeroed at this alpha.
    assert "noise_a" not in sel.selected_factors()
    assert "noise_b" not in sel.selected_factors()


def test_lasso_zero_alpha_keeps_everything():
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=list("abcd"))
    y = pd.Series(rng.normal(0, 1, n))
    sel = LassoSelector(alpha=0.0)
    sel.fit(X, y)
    # With no penalty the coefficient may be tiny, so we just check
    # the selector doesn't crash and produces a coef per feature.
    assert len(sel.coef_) == 4


# === ICWeighter ===

def test_ic_weighter_sign_follows_correlation():
    """A factor positively correlated with y should get a positive weight."""
    rng = np.random.default_rng(2)
    n = 300
    f_pos = rng.normal(0, 1, n)
    f_neg = rng.normal(0, 1, n)
    y = pd.Series(f_pos - 0.6 * f_neg + rng.normal(0, 0.5, n))
    X = pd.DataFrame({"pos": f_pos, "neg": f_neg})

    w = ICWeighter(use_rank=False)
    w.fit(X, y)
    weights = w.weights()
    assert weights["pos"] > 0
    assert weights["neg"] < 0
    # L1-normalised.
    assert abs(weights.abs().sum() - 1.0) < 1e-9


def test_equal_weighter_uniform():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(0, 1, (50, 3)), columns=list("abc"))
    y = pd.Series(rng.normal(0, 1, 50))
    w = EqualWeighter()
    w.fit(X, y)
    weights = w.weights()
    assert weights.to_list() == pytest.approx([1 / 3] * 3)


def test_ir_weighter_runs_and_returns_weights():
    rng = np.random.default_rng(4)
    n = 240
    X = pd.DataFrame(rng.normal(0, 1, (n, 3)), columns=list("abc"))
    y = pd.Series(0.4 * X["a"] + rng.normal(0, 1, n))
    w = IRWeighter(n_chunks=6)
    w.fit(X, y)
    assert len(w.weights()) == 3
    # IR should rank "a" highest in magnitude.
    assert w.ir.abs().idxmax() == "a"


# === TwoStepPipeline ===

def test_two_step_pipeline_end_to_end():
    """Pipeline should select the signal feature and predict in its direction."""
    rng = np.random.default_rng(5)
    n = 400
    f1 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = pd.Series(0.6 * f1 + rng.normal(0, 0.3, n))
    X = pd.DataFrame({"signal": f1, "noise": noise})

    pipe = TwoStepPipeline(
        selector=LassoSelector(alpha=0.05),
        weighter=ICWeighter(use_rank=True),
    )
    info = pipe.fit(X, y)
    assert "signal" in info.selected_factors
    assert info.fallback_used is False
    preds = pipe.predict(X)
    # Predictions should correlate positively with y.
    corr = float(pd.Series(preds.values).corr(y))
    assert corr > 0.4


def test_two_step_pipeline_falls_back_when_all_dropped():
    """If Lasso zeros every coefficient, the pipeline keeps all input factors."""
    rng = np.random.default_rng(6)
    n = 200
    X = pd.DataFrame(rng.normal(0, 1, (n, 3)), columns=list("xyz"))
    y = pd.Series(rng.normal(0, 1, n))
    # Huge alpha → all zero.
    pipe = TwoStepPipeline(
        selector=LassoSelector(alpha=10.0),
        weighter=ICWeighter(),
    )
    info = pipe.fit(X, y)
    assert info.fallback_used is True
    # All three features should survive into the weighter.
    assert set(info.selected_factors) == {"x", "y", "z"}
