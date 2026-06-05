"""Smoke tests for composite factors (built from existing ops)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.composite as _comp  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B", "C"]
    rng = np.random.default_rng(23)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 3)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_rank_signed_mom_10_runs(panel):
    f = make_factor("rank_signed_mom_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_decay_corr_pv_20_runs(panel):
    f = make_factor("decay_corr_pv_20")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_mom_vol_interact_10_runs(panel):
    f = make_factor("mom_vol_interact_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_no_look_ahead(panel):
    f = make_factor("rank_signed_mom_10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("rank_signed_mom", "decay_corr_pv",
                 "scale_decay_mom", "mom_vol_interact"):
        spec = get_spec(name)
        assert spec is not None
