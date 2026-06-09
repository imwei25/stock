"""Unit tests for per-day symmetric (Löwdin) orthogonalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _panel(n_days=4, n_stocks=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    return rng.standard_normal((n_days, n_stocks)), dates, codes


def _correlated_pair(n_days=4, n_stocks=300, seed=0, rho=0.8):
    """Two factor panels f2 strongly correlated with f1 (per day)."""
    base, dates, codes = _panel(n_days, n_stocks, seed)
    noise, _, _ = _panel(n_days, n_stocks, seed + 100)
    f1 = pd.DataFrame(base, index=dates, columns=codes)
    f2 = pd.DataFrame(rho * base + np.sqrt(1 - rho**2) * noise,
                      index=dates, columns=codes)
    return {"f1": f1, "f2": f2}


def test_output_is_orthogonal_per_day():
    """After transform, the per-day cross-sectional Gram matrix is ~diagonal."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.85)
    out = symmetric_orthogonalize_panel(fp)
    for d in fp["f1"].index:
        a = out["f1"].loc[d].to_numpy()
        b = out["f2"].loc[d].to_numpy()
        ca = (a - a.mean()) / a.std(ddof=0)
        cb = (b - b.mean()) / b.std(ddof=0)
        corr = float((ca * cb).mean())
        assert abs(corr) < 1e-6, f"day {d} corr={corr}"


def test_order_independent():
    """Permuting input factor order yields the same per-column result (Löwdin)."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.7)
    out_ab = symmetric_orthogonalize_panel({"f1": fp["f1"], "f2": fp["f2"]})
    out_ba = symmetric_orthogonalize_panel({"f2": fp["f2"], "f1": fp["f1"]})
    pd.testing.assert_frame_equal(out_ab["f1"], out_ba["f1"])
    pd.testing.assert_frame_equal(out_ab["f2"], out_ba["f2"])


def test_close_to_original():
    """Each orthogonalized factor stays sign-aligned & correlated with original."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.6)
    out = symmetric_orthogonalize_panel(fp)
    for name in ("f1", "f2"):
        d = fp[name].index[0]
        orig = fp[name].loc[d].to_numpy()
        new = out[name].loc[d].to_numpy()
        co = (orig - orig.mean()) / orig.std(ddof=0)
        cn = (new - new.mean()) / new.std(ddof=0)
        assert float((co * cn).mean()) > 0.5


def test_degenerate_day_passthrough():
    """A day with fewer valid stocks than factors is returned unchanged."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(n_days=3, n_stocks=300, rho=0.7)
    for name in fp:
        fp[name].iloc[1, 1:] = np.nan
    out = symmetric_orthogonalize_panel(fp)
    for name in fp:
        pd.testing.assert_series_equal(out[name].iloc[1], fp[name].iloc[1])


def test_nan_cells_stay_nan():
    """Stocks with any NaN factor stay NaN in the output."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(n_days=3, n_stocks=300, rho=0.7)
    fp["f1"].iloc[0, 5] = np.nan
    out = symmetric_orthogonalize_panel(fp)
    assert np.isnan(out["f1"].iloc[0, 5])
    assert np.isnan(out["f2"].iloc[0, 5])


def test_fundamental_factor_skipped():
    """A fundamental-tagged factor passes through byte-for-byte."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.7)
    pe = fp["f1"].copy() * 3.0 + 1.0
    fp["pe"] = pe
    factor_types = {"f1": ("momentum",), "f2": ("reversal",), "pe": ("fundamental",)}
    out = symmetric_orthogonalize_panel(fp, factor_types=factor_types)
    pd.testing.assert_frame_equal(out["pe"], pe)
    d = fp["f1"].index[0]
    a = out["f1"].loc[d].to_numpy(); b = out["f2"].loc[d].to_numpy()
    ca = (a - a.mean()) / a.std(ddof=0); cb = (b - b.mean()) / b.std(ddof=0)
    assert abs(float((ca * cb).mean())) < 1e-6


def test_single_non_fundamental_factor_no_crash():
    """K_nf == 1 → orthogonalization reduces to a per-day z-score, no crash."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.7)
    fp = {"f1": fp["f1"]}
    out = symmetric_orthogonalize_panel(fp)
    assert out["f1"].shape == fp["f1"].shape
    assert not out["f1"].isna().all().all()


def test_input_not_mutated():
    """Original input frames are unchanged after the call."""
    from stockpool.ml.preprocess import symmetric_orthogonalize_panel
    fp = _correlated_pair(rho=0.7)
    snap = {k: v.copy() for k, v in fp.items()}
    _ = symmetric_orthogonalize_panel(fp)
    for k in fp:
        pd.testing.assert_frame_equal(fp[k], snap[k])
