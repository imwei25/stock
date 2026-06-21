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


def test_analyze_factors_end_to_end_synthetic():
    """One strong factor + one anti-factor + one noise factor on a synthetic panel."""
    panel = _synth_panel(n_days=120, n_stocks=15, seed=42)
    factor_names = ["momentum_20", "rsi_centered_14", "vol_ratio_5"]
    result = analyze_factors(
        panel=panel,
        factor_names=factor_names,
        horizon=3,
        ic_window=60,
    )
    assert result.factor_names == factor_names
    assert set(result.daily_ic.keys()) == set(factor_names)
    assert len(result.mean_ic) == 3
    assert len(result.ic_ir) == 3
    assert result.ic_correlation.shape == (3, 3)
    assert result.horizon == 3
    assert result.n_stocks == 15
    # No regime data when no index series given.
    assert result.regime_ic == {}


def test_analyze_factors_with_regime_index():
    panel = _synth_panel(n_days=200, n_stocks=10, seed=5)
    # Build a 200-day uptrending "index" so all post-warmup days are "bull".
    idx_close = pd.Series(
        np.linspace(100, 300, 200), index=panel["close"].index, name="sh000001",
    )
    result = analyze_factors(
        panel=panel,
        factor_names=["momentum_20", "rsi_centered_14"],
        horizon=3,
        ic_window=60,
        regime_index_close=idx_close,
    )
    # Should have at least the "bull" regime (and possibly only bull).
    assert "bull" in result.regime_ic
    assert len(result.regime_ic["bull"]) == 2


def test_analyze_factors_rejects_unknown_factor():
    panel = _synth_panel(n_days=60, n_stocks=5, seed=0)
    with pytest.raises(KeyError):
        analyze_factors(
            panel=panel, factor_names=["this_factor_does_not_exist"], horizon=3,
        )


def test_analyze_factors_ic_correlation_diagonal_is_one():
    panel = _synth_panel(n_days=120, n_stocks=10, seed=8)
    result = analyze_factors(
        panel=panel, factor_names=["momentum_20", "rsi_centered_14"],
        horizon=3, ic_window=60,
    )
    diag = np.diag(result.ic_correlation.values)
    assert np.allclose(diag, 1.0, atol=1e-9)


def _build_pick_fixture(factor_names, ir_values, corr_pairs=None):
    """Build a minimal FactorAnalysisResult for pick_top_factors tests."""
    n = len(factor_names)
    corr = pd.DataFrame(
        np.eye(n), index=factor_names, columns=factor_names,
    )
    for (a, b, v) in (corr_pairs or []):
        corr.loc[a, b] = v
        corr.loc[b, a] = v
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    return FactorAnalysisResult(
        factor_names=list(factor_names),
        daily_ic={n: pd.Series([0.0] * 10, index=dates) for n in factor_names},
        mean_ic=pd.Series(dict(zip(factor_names, ir_values))),
        ic_ir=pd.Series(dict(zip(factor_names, ir_values))),
        abs_ic_mean=pd.Series(dict(zip(factor_names, [abs(v) for v in ir_values]))),
        half_life=pd.Series(dict(zip(factor_names, [10.0] * n))),
        ic_correlation=corr,
        regime_ic={},
        horizon=3, ic_window=60, n_stocks=5, n_days=10,
        start_date=dates[0], end_date=dates[-1],
    )


def test_pick_top_factors_drops_correlated():
    res = _build_pick_fixture(
        factor_names=["a", "b", "c", "d"],
        ir_values=[0.5, 0.45, 0.4, 0.3],
        corr_pairs=[("a", "b", 0.8)],  # a and b too correlated
    )
    picked = pick_top_factors(res, top_n=3, max_correlation=0.6, min_ir=0.0)
    assert picked == ["a", "c", "d"]   # b is dropped, c/d are independent


def test_pick_top_factors_respects_min_ir():
    res = _build_pick_fixture(
        factor_names=["a", "b", "c", "d"],
        ir_values=[0.5, 0.4, 0.03, 0.01],
    )
    picked = pick_top_factors(res, top_n=4, max_correlation=0.6, min_ir=0.05)
    assert picked == ["a", "b"]


def test_pick_top_factors_uses_absolute_score():
    """Negative IR is still informative — sort by |ir|, not raw ir."""
    res = _build_pick_fixture(
        factor_names=["a", "b", "c"],
        ir_values=[0.1, -0.5, 0.3],
    )
    picked = pick_top_factors(res, top_n=2, max_correlation=0.99, min_ir=0.0)
    assert picked == ["b", "c"]


def test_pick_top_factors_returns_empty_when_all_below_threshold():
    res = _build_pick_fixture(
        factor_names=["a", "b"],
        ir_values=[0.01, 0.02],
    )
    picked = pick_top_factors(res, top_n=5, max_correlation=0.6, min_ir=0.1)
    assert picked == []


def test_analyze_factors_with_industry_neutral_factor():
    """When the factor list contains an industry_neutral factor, callers must
    inject sector_map first via factors.context.set_sector_map."""
    import numpy as np
    import pandas as pd
    from stockpool.factors.context import set_sector_map
    from stockpool.factors_analysis import analyze_factors

    n_bars = 60
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=n_bars)
    codes = ["A", "B", "C", "D"]
    closes = pd.DataFrame(
        {c: 100 + np.cumsum(rng.normal(0, 1, n_bars)) for c in codes},
        index=dates,
    )
    panel = {
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": pd.DataFrame(1000.0, index=dates, columns=codes),
    }
    set_sector_map({"A": "X", "B": "X", "C": "Y", "D": "Y"})

    result = analyze_factors(
        panel=panel,
        factor_names=["industry_relative_strength_20", "momentum_5"],
        horizon=3,
        ic_window=20,
    )
    # Both factors should produce *some* finite daily IC values
    assert "industry_relative_strength_20" in result.daily_ic
    assert result.daily_ic["industry_relative_strength_20"].notna().any()
    set_sector_map({})  # cleanup


def test_analyze_factors_applies_winsorize(monkeypatch):
    """winsorize=(lo,hi) clips factor cross-section before IC."""
    import numpy as np
    import pandas as pd
    from stockpool.factors_analysis import analyze_factors
    from stockpool.factors.registry import _REGISTRY, FactorSpec
    from stockpool.factors.base import Factor

    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    codes = [f"S{i:03d}" for i in range(20)]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        np.cumprod(1 + rng.normal(0, 0.01, (40, 20)), axis=0),
        index=dates, columns=codes,
    )
    panel = {"open": close, "high": close, "low": close, "close": close,
             "volume": pd.DataFrame(1.0, index=dates, columns=codes)}

    class _SpikeFactor(Factor):
        sources = ("test",); types = ("cross_sectional",)
        description = "factor with one giant outlier per day"
        @property
        def name(self): return "spike_test"
        def compute(self, panel):
            base = panel["close"].rank(axis=1)
            base.iloc[:, 0] = 1e6  # huge outlier in column 0 every day
            return base

    monkeypatch.setitem(_REGISTRY, "spike_test", FactorSpec(
        base_name="spike_test", cls=_SpikeFactor,
        sources=("test",), types=("cross_sectional",), description="",
    ))

    # NOTE: use method="pearson" here because Spearman is rank-invariant: clipping
    # a single outlier still leaves its column as the row-max so rank IC is
    # unchanged. Pearson IC responds to the actual clipped values, which is what
    # we need to prove the winsorize plumbing actually runs.
    r_winsorized = analyze_factors(panel, ["spike_test"], horizon=2,
                                   winsorize=(0.05, 0.95), method="pearson")
    r_raw = analyze_factors(panel, ["spike_test"], horizon=2,
                            winsorize=None, method="pearson")
    # With winsorize on, the outlier column is clipped → IC shape differs from raw.
    assert r_raw.daily_ic["spike_test"].std() > 0
    assert r_winsorized.daily_ic["spike_test"].std() > 0
    assert not r_winsorized.daily_ic["spike_test"].equals(
        r_raw.daily_ic["spike_test"]
    ), "winsorize=(0.05,0.95) must change daily IC vs winsorize=None"


def test_analyze_factors_winsorize_default_is_lenient():
    """Default winsorize=(0.01, 0.99) should NOT change healthy-factor IC by much."""
    import numpy as np
    import pandas as pd
    from stockpool.factors_analysis import analyze_factors

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = [f"S{i:03d}" for i in range(50)]
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        np.cumprod(1 + rng.normal(0, 0.01, (80, 50)), axis=0),
        index=dates, columns=codes,
    )
    panel = {"open": close, "high": close, "low": close, "close": close,
             "volume": pd.DataFrame(1.0, index=dates, columns=codes)}

    r_default = analyze_factors(panel, ["momentum_20"], horizon=3)
    r_none = analyze_factors(panel, ["momentum_20"], horizon=3, winsorize=None)
    diff = (r_default.abs_ic_mean["momentum_20"]
            - r_none.abs_ic_mean["momentum_20"])
    assert abs(diff) < 0.02, f"winsorize=(0.01,0.99) shifted abs_ic by {diff:.4f}"


def test_analyze_factors_winsorize_none_skips_winsorize_panel(monkeypatch):
    """winsorize=None must not invoke winsorize_panel."""
    import numpy as np
    import pandas as pd
    from stockpool.factors_analysis import analyze_factors
    from stockpool.ml import preprocess as _preprocess

    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    codes = [f"S{i:03d}" for i in range(20)]
    rng = np.random.default_rng(2)
    close = pd.DataFrame(
        np.cumprod(1 + rng.normal(0, 0.01, (30, 20)), axis=0),
        index=dates, columns=codes,
    )
    panel = {"open": close, "high": close, "low": close, "close": close,
             "volume": pd.DataFrame(1.0, index=dates, columns=codes)}

    calls = {"n": 0}
    real_winsorize = _preprocess.winsorize_panel
    def _spy(df, lo, hi):
        calls["n"] += 1
        return real_winsorize(df, lo, hi)
    monkeypatch.setattr(_preprocess, "winsorize_panel", _spy)

    analyze_factors(panel, ["momentum_20"], horizon=2, winsorize=None)
    assert calls["n"] == 0, "winsorize=None must skip winsorize_panel entirely"

    analyze_factors(panel, ["momentum_20"], horizon=2, winsorize=(0.01, 0.99))
    assert calls["n"] >= 1, "winsorize=(lo,hi) must call winsorize_panel"
