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
    """Mutating log_mcap on one day must not change residuals on other days."""
    from stockpool.ml.preprocess import mcap_neutralize_panel
    df = _panel(3, 30, seed=7)
    log_mcap = _panel(3, 30, seed=8)
    out1 = mcap_neutralize_panel(df, log_mcap)
    log_mcap_mod = log_mcap.copy()
    log_mcap_mod.iloc[1] = log_mcap_mod.iloc[1] * 100
    out2 = mcap_neutralize_panel(df, log_mcap_mod)
    pd.testing.assert_series_equal(out1.iloc[0], out2.iloc[0])
    pd.testing.assert_series_equal(out1.iloc[2], out2.iloc[2])
