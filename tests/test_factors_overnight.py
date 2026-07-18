"""Tests for the overnight/intraday return decomposition family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.overnight as _ov  # noqa: F401
from stockpool.factors import get_spec, make_factor


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, (80, 2)), axis=0),
        index=dates, columns=codes,
    )
    # Opens gap ±1% off the previous close.
    open_ = close.shift(1) * (1 + rng.normal(0, 0.01, (80, 2)))
    open_.iloc[0] = close.iloc[0]
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close, "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "volume": volume}


def test_registered_with_metadata():
    for base in ("overnight_mom", "intraday_mom", "oi_spread", "overnight_vol"):
        spec = get_spec(base)
        assert "time_series" in spec.types
        assert spec.description


def test_overnight_plus_intraday_equals_close_to_close(panel):
    """Identity: Σ[ln(o/c_prev) + ln(c/o)] over N = ln(c_t / c_{t-N})."""
    n = 10
    on = make_factor(f"overnight_mom_{n}").compute(panel)
    intra = make_factor(f"intraday_mom_{n}").compute(panel)
    total = on + intra
    expected = np.log(panel["close"] / panel["close"].shift(n))
    pd.testing.assert_frame_equal(total, expected, check_exact=False, rtol=1e-9)


def test_oi_spread_is_difference(panel):
    n = 10
    spread = make_factor(f"oi_spread_{n}").compute(panel)
    on = make_factor(f"overnight_mom_{n}").compute(panel)
    intra = make_factor(f"intraday_mom_{n}").compute(panel)
    pd.testing.assert_frame_equal(spread, on - intra, check_exact=False, rtol=1e-9)


def test_overnight_vol_matches_formula(panel):
    n = 20
    out = make_factor(f"overnight_vol_{n}").compute(panel)
    lo = np.log(panel["open"].where(panel["open"] > 0)
                / panel["close"].where(panel["close"] > 0).shift(1))
    expected = lo.rolling(n, min_periods=n).std(ddof=0)
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-9)


def test_warmup_rows_are_nan(panel):
    out = make_factor("overnight_mom_10").compute(panel)
    # Row 0 has no prev close; rows 0..9 lack a full 10-day window.
    assert out.iloc[:10].isna().all().all()
    assert out.iloc[11:].notna().all().all()


def test_nonpositive_prices_yield_nan(panel):
    panel["open"].iloc[30, 0] = 0.0
    out = make_factor("overnight_mom_5").compute(panel)
    # The poisoned overnight return contaminates every window containing it.
    assert out.iloc[30:35, 0].isna().all()
    # Other stock unaffected.
    assert out.iloc[30:35, 1].notna().all()


def test_no_look_ahead(panel):
    """Truncating the panel must not change earlier rows."""
    for name in ("overnight_mom_10", "intraday_mom_10",
                 "oi_spread_10", "overnight_vol_20"):
        full = make_factor(name).compute(panel)
        cut = {k: v.iloc[:60] for k, v in panel.items()}
        trunc = make_factor(name).compute(cut)
        pd.testing.assert_frame_equal(full.iloc[:60], trunc)
