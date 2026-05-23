"""Tests for stockpool.factors_analysis core."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpool.factors_analysis import (
    FactorAnalysisResult,
    analyze_factors,
    classify_regimes,
    compute_daily_ic,
    pick_top_factors,
)


def _make_result_fixture() -> FactorAnalysisResult:
    factor_names = ["f1", "f2", "f3"]
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    return FactorAnalysisResult(
        factor_names=factor_names,
        daily_ic={
            "f1": pd.Series([0.1] * 20, index=dates),
            "f2": pd.Series([-0.05] * 20, index=dates),
            "f3": pd.Series([0.0] * 20, index=dates),
        },
        mean_ic=pd.Series({"f1": 0.1, "f2": -0.05, "f3": 0.0}),
        ic_ir=pd.Series({"f1": 2.0, "f2": -1.0, "f3": 0.0}),
        abs_ic_mean=pd.Series({"f1": 0.1, "f2": 0.05, "f3": 0.0}),
        half_life=pd.Series({"f1": 10.0, "f2": 5.0, "f3": float("nan")}),
        ic_correlation=pd.DataFrame(
            [[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=factor_names, columns=factor_names,
        ),
        regime_ic={
            "bull": pd.Series({"f1": 0.15, "f2": -0.05, "f3": 0.0}),
            "bear": pd.Series({"f1": 0.05, "f2": -0.10, "f3": 0.0}),
            "sideways": pd.Series({"f1": 0.10, "f2": 0.0, "f3": 0.0}),
        },
        horizon=3,
        ic_window=20,
        n_stocks=10,
        n_days=20,
        start_date=dates[0],
        end_date=dates[-1],
    )


def test_factor_analysis_result_to_dict_roundtrip(tmp_path):
    res = _make_result_fixture()
    out_path = tmp_path / "result.json"
    res.to_json(out_path)
    loaded = FactorAnalysisResult.from_json(out_path)
    assert loaded.factor_names == res.factor_names
    assert loaded.horizon == 3
    assert loaded.n_stocks == 10
    pd.testing.assert_series_equal(loaded.mean_ic, res.mean_ic)
    pd.testing.assert_frame_equal(loaded.ic_correlation, res.ic_correlation)
    assert set(loaded.regime_ic.keys()) == {"bull", "bear", "sideways"}
    pd.testing.assert_series_equal(loaded.regime_ic["bull"], res.regime_ic["bull"])
    # NaN survives roundtrip (was the C1 bug case)
    assert pd.isna(loaded.half_life["f3"])
    assert loaded.half_life["f1"] == 10.0
    # date roundtrip
    assert loaded.start_date == res.start_date
    assert loaded.end_date == res.end_date
    # other scalar fields
    assert loaded.ic_window == res.ic_window
    assert loaded.n_days == res.n_days
    # daily_ic survives
    pd.testing.assert_series_equal(loaded.daily_ic["f1"], res.daily_ic["f1"], check_freq=False)
    pd.testing.assert_series_equal(loaded.ic_ir, res.ic_ir)
    pd.testing.assert_series_equal(loaded.abs_ic_mean, res.abs_ic_mean)


def test_factor_analysis_result_json_is_rfc_compliant(tmp_path):
    """to_json must emit valid RFC 8259 JSON (no bare NaN / Infinity tokens)."""
    res = _make_result_fixture()
    out_path = tmp_path / "result.json"
    res.to_json(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "NaN" not in text, "NaN must serialize to null per RFC 8259"
    assert "Infinity" not in text
    # And the file should parse with the strict json module
    import json
    parsed = json.loads(text)  # would fail with strict parser if NaN present;
                               # CPython's json is lenient but we still want None
    assert parsed["half_life"]["f3"] is None


def _synth_panel(n_days: int = 60, n_stocks: int = 8, seed: int = 0):
    """Build a deterministic OHLCV panel for unit tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    codes = [f"s{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, (n_days, n_stocks)), axis=0),
        index=dates, columns=codes,
    )
    panel = {
        "open":  close * 0.998,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": pd.DataFrame(
            rng.integers(1_000_000, 5_000_000, (n_days, n_stocks)).astype(float),
            index=dates, columns=codes,
        ),
    }
    return panel


def test_compute_daily_ic_perfect_negative_correlation():
    panel = _synth_panel(n_days=30, n_stocks=10, seed=1)
    # Forward return is close.pct_change(3).shift(-3).
    fwd = panel["close"].pct_change(3).shift(-3)
    # Factor = -forward_return → daily Spearman IC == -1 on rows where data is complete.
    factor = -fwd
    ic = compute_daily_ic(factor, fwd, method="spearman")
    # Drop rows where either side is all-NaN (head/tail).
    valid = ic.dropna()
    assert len(valid) >= 10
    assert (valid < -0.999).all(), f"expected IC ≈ -1, got {valid.head()}"


def test_compute_daily_ic_zero_for_random_factor():
    panel = _synth_panel(n_days=200, n_stocks=20, seed=2)
    fwd = panel["close"].pct_change(3).shift(-3)
    rng = np.random.default_rng(99)
    factor = pd.DataFrame(
        rng.normal(0, 1, fwd.shape), index=fwd.index, columns=fwd.columns,
    )
    ic = compute_daily_ic(factor, fwd, method="spearman").dropna()
    # Mean IC should be small (|μ| < 0.1) over 200 days.
    assert abs(ic.mean()) < 0.1, f"expected mean IC ≈ 0, got {ic.mean()}"


def test_compute_daily_ic_skips_constant_rows():
    """A day where the factor is constant across stocks must yield NaN, not 0/error."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    codes = ["a", "b", "c"]
    factor = pd.DataFrame(
        [[1.0, 1.0, 1.0],  # constant — IC should be NaN
         [1.0, 2.0, 3.0],
         [3.0, 2.0, 1.0],
         [1.0, 2.0, 3.0],
         [1.0, 2.0, 3.0]],
        index=dates, columns=codes,
    )
    fwd = pd.DataFrame(
        [[0.01, 0.02, 0.03]] * 5, index=dates, columns=codes,
    )
    ic = compute_daily_ic(factor, fwd, method="spearman")
    assert pd.isna(ic.iloc[0])
    assert ic.iloc[1] == pytest.approx(1.0)
    assert ic.iloc[2] == pytest.approx(-1.0)


def test_classify_regimes_pure_uptrend_is_bull():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    # Linear uptrend → close always above SMA, SMA rising.
    close = pd.Series(np.linspace(100, 200, 120), index=dates)
    regimes = classify_regimes(close, sma_window=60)
    # First sma_window-ish days are warmup (NaN); after that should be "bull".
    tail = regimes.iloc[80:]
    assert (tail == "bull").all(), f"got {tail.value_counts()}"


def test_classify_regimes_pure_downtrend_is_bear():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    close = pd.Series(np.linspace(200, 100, 120), index=dates)
    regimes = classify_regimes(close, sma_window=60)
    tail = regimes.iloc[80:]
    assert (tail == "bear").all()


def test_classify_regimes_choppy_is_sideways():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    # Oscillation around 100 → SMA flat, close crosses repeatedly.
    close = pd.Series(100.0 + 5.0 * np.sin(np.linspace(0, 12 * np.pi, 120)), index=dates)
    regimes = classify_regimes(close, sma_window=60)
    tail = regimes.iloc[80:]
    sideways_share = (tail == "sideways").sum() / len(tail)
    assert sideways_share >= 0.2, f"sideways share {sideways_share} too low"


def test_classify_regimes_warmup_is_nan():
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    close = pd.Series(np.linspace(100, 200, 120), index=dates)
    regimes = classify_regimes(close, sma_window=60)
    warmup = 60 + 5 - 1   # sma_window + slope_lookback - 1, defaults
    assert regimes.iloc[:warmup].isna().all()
    assert regimes.iloc[warmup:warmup + 5].notna().all()


from stockpool.factors_analysis import _half_life_from_acf


def test_half_life_ar1_known_decay():
    """AR(1) with ρ=0.5 → half-life = log(0.5)/log(0.5) = 1.0"""
    n = 500
    rng = np.random.default_rng(7)
    rho = 0.5
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, 1)
    series = pd.Series(x)
    hl = _half_life_from_acf(series)
    assert 0.7 < hl < 1.5, f"expected half-life ≈ 1.0, got {hl}"


def test_half_life_ar1_slow_decay():
    """AR(1) with ρ=0.9 → half-life = log(0.5)/log(0.9) ≈ 6.58"""
    n = 2000
    rng = np.random.default_rng(11)
    rho = 0.9
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, 1)
    series = pd.Series(x)
    hl = _half_life_from_acf(series)
    assert 5.0 < hl < 9.0, f"expected half-life ≈ 6.58, got {hl}"


def test_half_life_white_noise_is_nan_or_zero():
    """White noise has ρ ≈ 0, so half-life is either NaN (rho ≤ 0) or tiny."""
    rng = np.random.default_rng(13)
    series = pd.Series(rng.normal(0, 1, 500))
    hl = _half_life_from_acf(series)
    assert pd.isna(hl) or hl < 0.5, f"got {hl}"


def test_half_life_handles_nan_input():
    series = pd.Series([np.nan] * 10)
    hl = _half_life_from_acf(series)
    assert pd.isna(hl)
