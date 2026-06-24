"""Embargo behavior tests for MLFactorStrategy (F2 PR-A)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting.strategies import MLFactorStrategy
from stockpool.config import MLFactorConfig, SelectorConfig, WeighterConfig


def _synthetic_daily(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_days))
    return pd.DataFrame({
        "date": dates,
        "open":  close * 0.998,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, n_days).astype(float),
    })


def test_embargoed_label_end_default_uses_horizon():
    # 本文件验证 legacy close 基准的 embargo 数学;open 基准的 +1 见 test_label_basis.py
    cfg = MLFactorConfig(horizon=3, label_basis="close")
    strat = MLFactorStrategy(cfg=cfg)
    assert strat._embargoed_label_end(100) == 94


def test_embargoed_label_end_explicit_zero_matches_legacy():
    cfg = MLFactorConfig(horizon=5, embargo_days=0, label_basis="close")
    strat = MLFactorStrategy(cfg=cfg)
    assert strat._embargoed_label_end(100) == 95


def test_embargoed_label_end_explicit_positive_overrides_horizon():
    cfg = MLFactorConfig(horizon=3, embargo_days=10, label_basis="close")
    strat = MLFactorStrategy(cfg=cfg)
    assert strat._embargoed_label_end(100) == 87


def test_embargoed_label_end_can_go_negative_when_history_short():
    cfg = MLFactorConfig(horizon=3, embargo_days=5, label_basis="close")
    strat = MLFactorStrategy(cfg=cfg)
    assert strat._embargoed_label_end(5) == 5 - 3 - 5


def test_refit_with_default_embargo_returns_none_when_insufficient_history():
    """Short history + default embargo + 20-bar factor warmup → _refit refuses."""
    cfg = MLFactorConfig(
        horizon=3, train_window=50, min_train_samples=20,
        refit_every=10, panel_mode="per_stock", label_basis="close",
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )
    strat = MLFactorStrategy(cfg=cfg)
    df = _synthetic_daily(n_days=30)
    from stockpool.ml.dataset import forward_return, build_factor_matrix
    X = build_factor_matrix(df, cfg.factors)
    y = forward_return(df, cfg.horizon)
    # current_bar=29 is the last valid index (0-based) in a 30-row frame
    result = strat._try_fit(df, X, y, current_bar=29)
    assert result is None


def test_refit_with_legacy_no_embargo_runs_to_completion():
    """Long history + embargo_days=0 → fit succeeds and returns quantiles."""
    cfg = MLFactorConfig(
        horizon=3, train_window=120, min_train_samples=60,
        refit_every=20, panel_mode="per_stock",
        embargo_days=0, label_basis="close",
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )
    strat = MLFactorStrategy(cfg=cfg)
    df = _synthetic_daily(n_days=300)
    from stockpool.ml.dataset import forward_return, build_factor_matrix
    X = build_factor_matrix(df, cfg.factors)
    y = forward_return(df, cfg.horizon)
    # current_bar=299 is the last valid index (0-based) in a 300-row frame
    result = strat._try_fit(df, X, y, current_bar=299)
    assert result is not None
    pipeline, quantiles = result
    assert set(quantiles) == {"strong_buy", "buy", "sell", "strong_sell"}


def test_refit_with_default_embargo_long_history_also_runs_to_completion():
    """Sanity: with plenty of history, default auto-embargo still leaves enough samples."""
    cfg = MLFactorConfig(
        horizon=3, train_window=120, min_train_samples=60,
        refit_every=20, panel_mode="per_stock", label_basis="close",
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )
    strat = MLFactorStrategy(cfg=cfg)
    df = _synthetic_daily(n_days=300)
    from stockpool.ml.dataset import forward_return, build_factor_matrix
    X = build_factor_matrix(df, cfg.factors)
    y = forward_return(df, cfg.horizon)
    # current_bar=299 is the last valid index (0-based) in a 300-row frame
    result = strat._try_fit(df, X, y, current_bar=299)
    assert result is not None
    pipeline, quantiles = result
    assert set(quantiles) == {"strong_buy", "buy", "sell", "strong_sell"}


def test_default_embargo_shifts_training_label_end_vs_legacy():
    cfg_default = MLFactorConfig(
        horizon=3, train_window=200, min_train_samples=30,
        refit_every=20, panel_mode="per_stock", label_basis="close",
    )
    cfg_legacy = MLFactorConfig(
        horizon=3, train_window=200, min_train_samples=30,
        refit_every=20, panel_mode="per_stock",
        embargo_days=0, label_basis="close",
    )
    strat_default = MLFactorStrategy(cfg=cfg_default)
    strat_legacy = MLFactorStrategy(cfg=cfg_legacy)
    assert strat_default._embargoed_label_end(100) == 94
    assert strat_legacy._embargoed_label_end(100) == 97
    assert (
        strat_default._embargoed_label_end(100)
        < strat_legacy._embargoed_label_end(100)
    )


def test_embargo_eliminates_label_leak_on_synthetic():
    """The 3 bars (194, 195, 196) — whose labels come from closes adjacent to
    the test bar (200) — must NOT be in the embargoed training set, but must
    be in the legacy training set.
    """
    cfg_legacy = MLFactorConfig(
        horizon=3, train_window=100, min_train_samples=20,
        refit_every=20, panel_mode="per_stock", embargo_days=0, label_basis="close",
    )
    cfg_embargo = MLFactorConfig(
        horizon=3, train_window=100, min_train_samples=20,
        refit_every=20, panel_mode="per_stock", label_basis="close",
    )
    strat_legacy = MLFactorStrategy(cfg=cfg_legacy)
    strat_embargo = MLFactorStrategy(cfg=cfg_embargo)

    assert strat_legacy._embargoed_label_end(200) == 197
    assert strat_embargo._embargoed_label_end(200) == 194
    leak_bars = set(range(194, 197))
    embargo_train_bars = set(range(0, strat_embargo._embargoed_label_end(200)))
    legacy_train_bars = set(range(0, strat_legacy._embargoed_label_end(200)))
    assert leak_bars.issubset(legacy_train_bars), \
        "legacy mode is supposed to include leaky bars"
    assert leak_bars.isdisjoint(embargo_train_bars), \
        "embargo mode must exclude leaky bars"
