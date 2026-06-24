"""Smoke tests for Beta / IVOL (market = cross-sectional equal-weight mean)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.beta as _beta  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    codes = ["A", "B", "C", "D"]
    rng = np.random.default_rng(11)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((120, 4)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    return {"close": close, "high": close + 1, "low": close - 1,
            "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=codes)}


def test_identical_columns_beta_one_ivol_zero():
    """所有列完全相同 → 市场=每只票 → beta≈1, ivol≈0。"""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(1)
    series = 100.0 + rng.standard_normal(120).cumsum()
    close = pd.DataFrame({c: series for c in ["A", "B", "C"]}, index=dates)
    panel = {"close": close, "high": close + 1, "low": close - 1,
             "open": close, "volume": pd.DataFrame(1e6, index=dates, columns=close.columns)}

    beta = make_factor("beta_60").compute(panel).iloc[65:]
    ivol = make_factor("ivol_60").compute(panel).iloc[65:]
    assert np.allclose(beta.to_numpy(), 1.0, atol=1e-6)
    assert np.allclose(ivol.to_numpy(), 0.0, atol=1e-8)


def test_beta_finite(panel):
    out = make_factor("beta_60").compute(panel)
    valid = out.iloc[65:]
    assert np.isfinite(valid.to_numpy()).all()


def test_ivol_nonnegative(panel):
    out = make_factor("ivol_60").compute(panel)
    valid = out.iloc[65:]
    assert (valid >= 0).all().all()


def test_warmup_nan(panel):
    out = make_factor("beta_60").compute(panel)
    # 前 60 行内不足窗口 → NaN
    assert out.iloc[:59].isna().all().all()


def test_no_look_ahead(panel):
    f = make_factor("ivol_60")
    full = f.compute(panel)
    trunc = {k: v.iloc[:90] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(full.iloc[:90], short, check_exact=False, rtol=1e-9)


def test_specs_registered():
    for name in ("beta", "ivol"):
        assert get_spec(name) is not None
