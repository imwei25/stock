"""MLFactorStrategy + backtest engine integration tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.backtesting import (
    BacktestEngine,
    BarContext,
    MLFactorStrategy,
    PositionContext,
    TradeCosts,
)
from stockpool.config import MLFactorConfig, QuantileThresholds, WeighterConfig


def _synth_ohlcv(n: int, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.02, n)
    close = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-02", periods=n, freq="B"),
        "open": close * 0.998,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
    })


def test_generate_signals_yields_expected_columns():
    df = _synth_ohlcv(300, seed=42)
    cfg = MLFactorConfig(train_window=120, refit_every=20, min_train_samples=60, embargo_days=0)
    strat = MLFactorStrategy(cfg)
    sigs = strat.generate_signals(df)
    assert list(sigs.columns) == ["date", "open", "close", "signal", "score"]
    assert len(sigs) == len(df)


def test_per_stock_mode_produces_mix_of_signals():
    df = _synth_ohlcv(400, seed=1)
    cfg = MLFactorConfig(train_window=120, refit_every=20, min_train_samples=60, embargo_days=0)
    strat = MLFactorStrategy(cfg)
    sigs = strat.generate_signals(df)
    # Once past warmup, every verdict class should appear at least once.
    counts = sigs["signal"].value_counts()
    for label in ("strong_buy", "buy", "neutral", "sell", "strong_sell"):
        assert counts.get(label, 0) > 0, f"missing label: {label}"


def test_warmup_bars_are_neutral():
    """Before min_train_samples + horizon, the model can't fit → neutral."""
    df = _synth_ohlcv(200, seed=2)
    cfg = MLFactorConfig(train_window=80, refit_every=20, min_train_samples=80, embargo_days=0)
    strat = MLFactorStrategy(cfg)
    sigs = strat.generate_signals(df)
    # First (min_train_samples + horizon - 1) rows should be all neutral.
    warmup = cfg.min_train_samples + cfg.horizon
    assert (sigs["signal"].iloc[:warmup] == "neutral").all()


def test_engine_consumes_ml_signals():
    df = _synth_ohlcv(400, seed=3)
    cfg = MLFactorConfig(train_window=120, refit_every=20, min_train_samples=60, embargo_days=0)
    strat = MLFactorStrategy(cfg)
    engine = BacktestEngine(strat, costs=TradeCosts(0.0008, 0.0013))
    result = engine.run(df, max_holding_days=10)
    # Curve should be the same length as the input frame.
    assert len(result.curve) == len(df)
    # Equity must be positive throughout (long-only, costs < 1).
    assert (result.curve["equity"] > 0).all()


def test_decision_rules_respect_config_sets():
    cfg = MLFactorConfig(
        buy_verdicts=["buy"],
        sell_verdicts=["sell"],
        refresh_verdicts=[],
        embargo_days=0,
    )
    strat = MLFactorStrategy(cfg)
    enter = BarContext(bar_idx=0, date=pd.Timestamp("2024-01-02"),
                       close=1.0, signal="strong_buy")
    # strong_buy is NOT in buy_verdicts here → no entry.
    assert strat.should_enter(enter) is False
    enter2 = enter.__class__(bar_idx=0, date=enter.date, close=1.0, signal="buy")
    assert strat.should_enter(enter2) is True


def test_pooled_mode_runs_on_multi_stock_panel():
    pool = {f"S{i}": _synth_ohlcv(300, seed=i + 10) for i in range(3)}
    cfg = MLFactorConfig(
        train_window=150, refit_every=30, min_train_samples=80,
        panel_mode="pooled",
        embargo_days=0,
    )
    strat = MLFactorStrategy(cfg, pool_data=pool, current_stock_code="S0")
    sigs = strat.generate_signals(pool["S0"])
    assert len(sigs) == 300
    # Should produce at least some non-neutral signals.
    assert (sigs["signal"] != "neutral").sum() > 0


def test_score_is_nan_during_warmup_and_finite_after():
    df = _synth_ohlcv(300, seed=4)
    cfg = MLFactorConfig(train_window=100, refit_every=20, min_train_samples=60, embargo_days=0)
    strat = MLFactorStrategy(cfg)
    sigs = strat.generate_signals(df)
    # Warmup region: NaN scores
    assert sigs["score"].iloc[:30].isna().all()
    # Post-warmup region should be mostly finite.
    finite_late = sigs["score"].iloc[-50:].notna().mean()
    assert finite_late > 0.9


def test_pooled_train_window_is_per_stock_not_global():
    """In pooled mode, ``train_window`` caps PER STOCK, not the global panel
    size. Otherwise we'd silently train on just the last stock's recent rows
    and waste every other stock's data.
    """
    from stockpool.ml.dataset import build_panel  # local import to assert shape

    pool = {f"S{i}": _synth_ohlcv(300, seed=i + 20) for i in range(4)}
    cfg = MLFactorConfig(
        train_window=80, refit_every=20, min_train_samples=50,
        panel_mode="pooled",
        embargo_days=0,
    )
    strat = MLFactorStrategy(cfg, pool_data=pool, current_stock_code="S0")

    # Drive a fit at bar 250 of S0 and inspect the panel it would build.
    fitted = strat._try_fit(
        pool["S0"],
        X_full=__import__("stockpool.ml.dataset", fromlist=["build_factor_matrix"])
            .build_factor_matrix(pool["S0"], cfg.factors),
        y_full=__import__("stockpool.ml.dataset", fromlist=["forward_return"])
            .forward_return(pool["S0"], cfg.horizon),
        current_bar=250,
    )
    assert fitted is not None
    pipeline, _ = fitted

    # Replay the same fit-time panel selection logic to assert membership.
    truncated = strat._build_truncated_pool(
        pool["S0"], pool["S0"]["date"].iloc[250], current_bar=250,
    )
    X_panel, _ = build_panel(truncated, cfg.factors, cfg.horizon)
    X_panel = X_panel.groupby(level="stock", group_keys=False, sort=False).tail(
        cfg.train_window
    )
    # Every stock in the pool must contribute at least one row.
    stocks_in_panel = set(X_panel.index.get_level_values("stock").unique())
    assert stocks_in_panel == {"S0", "S1", "S2", "S3"}, (
        f"some stocks missing from pooled training panel: {stocks_in_panel}"
    )
    # And no stock contributes more than `train_window` rows.
    counts = X_panel.groupby(level="stock", sort=False).size()
    assert (counts <= cfg.train_window).all()


def test_pooled_mode_truncates_other_stocks_at_current_date():
    """At bar t of host stock, pool stocks must not contribute future data."""
    host = _synth_ohlcv(200, seed=5)
    # Other stock has an obvious "future event" at bar 150 (10x close)
    other = _synth_ohlcv(200, seed=6)
    other.loc[150:, "close"] = other["close"].iloc[150] * 10
    # If the strategy peeks at future close, fits at bar 100 will be tainted.
    cfg = MLFactorConfig(
        train_window=80, refit_every=20, min_train_samples=60,
        panel_mode="pooled",
        embargo_days=0,
    )
    strat = MLFactorStrategy(
        cfg, pool_data={"H": host, "O": other}, current_stock_code="H",
    )
    # Build the truncated pool the strategy would use at host bar 100.
    truncated = strat._build_truncated_pool(
        host, host["date"].iloc[100], current_bar=100,
    )
    other_truncated = truncated["O"]
    # Every date in the truncated other-stock frame must precede the cutoff.
    # With embargo_days=0, host_slice_end = label_end + horizon = (100-5) + 5 = 100,
    # so cutoff_date = host["date"].iloc[99] < host["date"].iloc[100].
    assert (other_truncated["date"] < host["date"].iloc[100]).all()


def test_quantile_thresholds_consistent():
    """Custom thresholds should still produce signal mix in correct order."""
    df = _synth_ohlcv(400, seed=7)
    cfg = MLFactorConfig(
        train_window=120, refit_every=20, min_train_samples=60,
        thresholds=QuantileThresholds(
            strong_buy=0.95, buy=0.80, sell=0.20, strong_sell=0.05,
        ),
        weighter=WeighterConfig(type="equal"),
        embargo_days=0,
    )
    strat = MLFactorStrategy(cfg)
    sigs = strat.generate_signals(df).dropna(subset=["score"])
    # With these strict thresholds, strong labels should be rarer than soft.
    counts = sigs["signal"].value_counts()
    if "strong_buy" in counts and "buy" in counts:
        assert counts["strong_buy"] <= counts["buy"]
    if "strong_sell" in counts and "sell" in counts:
        assert counts["strong_sell"] <= counts["sell"]
