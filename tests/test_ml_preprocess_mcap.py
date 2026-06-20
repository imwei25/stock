"""Unit tests for market-cap neutralization preprocessing."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


def _panel(n_days, n_stocks, seed):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(
        rng.standard_normal((n_days, n_stocks)), index=dates, columns=codes,
    )


def test_mcap_neutralize_removes_log_mcap_loading():
    """Y = 2 * log_mcap + noise → residuals should have ~zero correlation with log_mcap."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=4, freq="B")
    codes = [f"S{i:03d}" for i in range(50)]
    log_mcap = pd.DataFrame(
        rng.standard_normal((4, 50)) * 0.5 + 10.0,
        index=dates, columns=codes,
    )
    noise = pd.DataFrame(
        rng.standard_normal((4, 50)) * 0.1, index=dates, columns=codes,
    )
    y = 2.0 * log_mcap + noise

    resid = mcap_neutralize_panel(y, log_mcap)

    # Per-day OLS residual should be ~noise (corr with log_mcap ~0)
    for d in dates:
        r = resid.loc[d]
        m = log_mcap.loc[d]
        corr = np.corrcoef(r.values, m.values)[0, 1]
        assert abs(corr) < 0.05, f"residual still correlated with log_mcap on {d}: corr={corr}"


def test_mcap_neutralize_preserves_shape_and_nan_cells():
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(3, 30, seed=1)
    log_mcap = _panel(3, 30, seed=2)
    df.iloc[0, 5] = np.nan
    log_mcap.iloc[1, 7] = np.nan
    out = mcap_neutralize_panel(df, log_mcap)
    assert out.shape == df.shape
    assert np.isnan(out.iloc[0, 5])  # original NaN in y stays NaN


def test_mcap_neutralize_falls_back_when_too_few_codes(caplog):
    """A day with < 10 valid codes returns original row + emits warning count."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(2, 8, seed=3)  # only 8 codes — below hard minimum 10
    log_mcap = _panel(2, 8, seed=4)
    with caplog.at_level(logging.WARNING, logger="stockpool.ml.preprocess"):
        out = mcap_neutralize_panel(df, log_mcap)
    # All days should fall back → df returned unchanged
    pd.testing.assert_frame_equal(out, df)
    assert any("fallback on 2 / 2 days" in rec.message for rec in caplog.records)


def test_mcap_neutralize_handles_all_nan_log_mcap_day():
    """A day where log_mcap is fully NaN falls back to original df row."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(3, 30, seed=5)
    log_mcap = _panel(3, 30, seed=6)
    log_mcap.iloc[1] = np.nan
    out = mcap_neutralize_panel(df, log_mcap)
    pd.testing.assert_series_equal(out.iloc[1], df.iloc[1])


def test_mcap_neutralize_days_are_independent():
    """Mutating log_mcap on one day must not change residuals on other days,
    AND must change residuals on the mutated day (so the test isn't trivially satisfied
    by a function that ignores log_mcap entirely)."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(3, 30, seed=7)
    log_mcap = _panel(3, 30, seed=8)
    out1 = mcap_neutralize_panel(df, log_mcap)
    log_mcap_mod = log_mcap.copy()
    log_mcap_mod.iloc[1] = log_mcap_mod.iloc[1] * 100
    out2 = mcap_neutralize_panel(df, log_mcap_mod)
    # Independence: rows 0 and 2 unchanged
    pd.testing.assert_series_equal(out1.iloc[0], out2.iloc[0])
    pd.testing.assert_series_equal(out1.iloc[2], out2.iloc[2])
    # Effectiveness: mutated row 1 is different
    assert not out1.iloc[1].equals(out2.iloc[1]), (
        "mutating log_mcap on day 1 produced identical residuals — "
        "function may be ignoring log_mcap input"
    )


def test_industry_neutralize_legacy_behavior_unchanged_when_log_mcap_none():
    """log_mcap=None must produce bit-for-bit identical output to the pre-PR code path."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _panel(3, 20, seed=10)
    sector_map = {c: f"IND{i % 4}" for i, c in enumerate(df.columns)}
    legacy = industry_neutralize_panel(df, sector_map)  # no log_mcap kwarg
    explicit_none = industry_neutralize_panel(df, sector_map, log_mcap=None)
    pd.testing.assert_frame_equal(legacy, explicit_none)


def test_industry_neutralize_joint_ols_residual_orthogonal_to_inputs():
    """Y = 3 * log_mcap + 1.5 * industry_effect + noise →
    residuals ~ noise, ~uncorrelated with log_mcap and industry membership."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    rng = np.random.default_rng(11)
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    codes = [f"S{i:03d}" for i in range(60)]
    sector_map = {c: f"IND{i % 4}" for i, c in enumerate(codes)}
    industry_offset = pd.Series(
        {c: float({"IND0": -1.0, "IND1": 0.0, "IND2": 1.0, "IND3": 2.0}[sector_map[c]])
         for c in codes}
    )
    log_mcap = pd.DataFrame(
        rng.standard_normal((3, 60)) * 0.5 + 10.0, index=dates, columns=codes,
    )
    noise = pd.DataFrame(
        rng.standard_normal((3, 60)) * 0.1, index=dates, columns=codes,
    )
    y = 3.0 * log_mcap + industry_offset.values[None, :] * 1.5 + noise

    resid = industry_neutralize_panel(y, sector_map, log_mcap=log_mcap)

    # Residual should be uncorrelated with log_mcap per day
    for d in dates:
        r = resid.loc[d]
        m = log_mcap.loc[d]
        corr = np.corrcoef(r.values, m.values)[0, 1]
        assert abs(corr) < 0.1, f"residual ~ log_mcap on {d}: corr={corr}"

    # Per-industry mean of residual should be ~0 (industry demeaned)
    for d in dates:
        for ind in {"IND0", "IND1", "IND2", "IND3"}:
            members = [c for c in codes if sector_map[c] == ind]
            assert abs(resid.loc[d, members].mean()) < 0.1


def test_industry_neutralize_single_member_industry_keeps_original_value():
    """A single-member industry must NOT be silently demeaned to 0;
    its code is excluded from the regression and the original y is kept."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    rng = np.random.default_rng(12)
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    codes = [f"S{i:03d}" for i in range(30)]
    sector_map = {c: "BIG" for c in codes[:-1]}
    sector_map[codes[-1]] = "LONELY"
    log_mcap = pd.DataFrame(
        rng.standard_normal((2, 30)) * 0.5 + 10.0, index=dates, columns=codes,
    )
    df = pd.DataFrame(
        rng.standard_normal((2, 30)), index=dates, columns=codes,
    )
    out = industry_neutralize_panel(df, sector_map, log_mcap=log_mcap)
    # The lonely code should keep its original value (NOT silently zeroed)
    pd.testing.assert_series_equal(out[codes[-1]], df[codes[-1]])


def test_apply_pipeline_skips_mcap_when_log_mcap_none(caplog):
    """mcap_neutralize=True without log_mcap_panel → log warning, skip mcap step."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _panel(3, 30, seed=20)
    cfg = PreprocessConfig(
        winsorize=None, zscore=True,
        industry_neutralize=False, mcap_neutralize=True,
    )
    with caplog.at_level(logging.WARNING, logger="stockpool.ml.preprocess"):
        out = apply_preprocess_pipeline({"f1": df}, cfg, log_mcap_panel=None)
    assert "mcap_neutralize=True" in " ".join(rec.message for rec in caplog.records)
    # zscore still applied (rows mean ~0)
    assert abs(out["f1"].iloc[0].mean()) < 1e-9


def test_apply_pipeline_runs_mcap_neutralize_when_enabled():
    """mcap_neutralize=True with valid log_mcap → factor residualised."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    rng = np.random.default_rng(21)
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    codes = [f"S{i:03d}" for i in range(50)]
    log_mcap = pd.DataFrame(
        rng.standard_normal((3, 50)) * 0.5 + 10.0, index=dates, columns=codes,
    )
    df = 2.0 * log_mcap + pd.DataFrame(
        rng.standard_normal((3, 50)) * 0.1, index=dates, columns=codes,
    )
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
        min_pool_size=50,
    )
    out = apply_preprocess_pipeline(
        {"f1": df}, cfg, log_mcap_panel=log_mcap, n_codes=50,
    )
    for d in dates:
        corr = np.corrcoef(out["f1"].loc[d].values, log_mcap.loc[d].values)[0, 1]
        assert abs(corr) < 0.05


def test_apply_pipeline_size_guard_short_circuits_mcap():
    """n_codes < min_pool_size → mcap step also skipped along with others."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _panel(3, 5, seed=22)
    log_mcap = _panel(3, 5, seed=23)
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
        min_pool_size=200,
    )
    out = apply_preprocess_pipeline(
        {"f1": df}, cfg, log_mcap_panel=log_mcap, n_codes=5,
    )
    pd.testing.assert_frame_equal(out["f1"], df)


def test_apply_pipeline_fundamental_factor_skip_industry_but_runs_mcap():
    """ROE (fundamental, no contains_mcap tag) skips industry, runs mcap."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    rng = np.random.default_rng(24)
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    codes = [f"S{i:03d}" for i in range(60)]
    log_mcap = pd.DataFrame(
        rng.standard_normal((2, 60)) * 0.5 + 10.0, index=dates, columns=codes,
    )
    roe_df = 3.0 * log_mcap + pd.DataFrame(
        rng.standard_normal((2, 60)) * 0.1, index=dates, columns=codes,
    )
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=True, mcap_neutralize=True,
        min_pool_size=60,
    )
    sector_map = {c: f"IND{i % 4}" for i, c in enumerate(codes)}
    factor_types = {"roe": ("fundamental", "cross_sectional")}
    out = apply_preprocess_pipeline(
        {"roe": roe_df}, cfg,
        sector_map=sector_map, factor_types=factor_types,
        log_mcap_panel=log_mcap, n_codes=60,
    )
    # mcap should still have been removed
    for d in dates:
        corr = np.corrcoef(out["roe"].loc[d].values, log_mcap.loc[d].values)[0, 1]
        assert abs(corr) < 0.1


def test_apply_pipeline_pe_with_contains_mcap_skips_both():
    """PE (contains_mcap) skips both industry AND mcap neutralization."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _panel(2, 50, seed=25)
    log_mcap = _panel(2, 50, seed=26)
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=True, mcap_neutralize=True,
        min_pool_size=50,
    )
    sector_map = {c: f"IND{i % 4}" for i, c in enumerate(df.columns)}
    factor_types = {"pe": ("fundamental", "cross_sectional", "contains_mcap")}
    out = apply_preprocess_pipeline(
        {"pe": df}, cfg,
        sector_map=sector_map, factor_types=factor_types,
        log_mcap_panel=log_mcap, n_codes=50,
    )
    pd.testing.assert_frame_equal(out["pe"], df)
