"""Tests for the GTJA191 A-share alpha family (validated subset) + ops.sma."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.factors import list_factors, make_factor
from stockpool.factors import ops


# Derived from the registry so adding GTJA factors doesn't require editing the
# list. A floor guards against accidental mass de-registration.
EXPECTED = sorted(f for f in list_factors() if f.startswith("gtja_"))
# Long-warmup factors: their longest window needs >~150-250 days, so the tail of
# a 300-day synthetic panel can be thin — excluded from the coverage assertion.
LONG_WINDOW = {"gtja_025", "gtja_026", "gtja_033", "gtja_045"}
SHORT_WINDOW = [n for n in EXPECTED if n not in LONG_WINDOW]


def test_gtja_family_minimum_count():
    assert len(EXPECTED) >= 50, f"expected >=50 gtja factors, got {len(EXPECTED)}"


def _synth_panel(n_days=300, n_stocks=60, seed=0):
    # 60 stocks (not a handful): cross-sectional RANK needs enough resolution
    # that rank-then-CORR factors (gtja_001/016/032) aren't degenerate purely
    # from a coarse synthetic universe. On the real ~4597 universe this is moot.
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    codes = [f"s{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0003, 0.02, (n_days, n_stocks)), axis=0),
        index=dates, columns=codes,
    )
    high = close * (1 + rng.uniform(0.0, 0.03, (n_days, n_stocks)))
    low = close * (1 - rng.uniform(0.0, 0.03, (n_days, n_stocks)))
    open_ = low + (high - low) * rng.uniform(0, 1, (n_days, n_stocks))
    vol = pd.DataFrame(
        rng.integers(1_000_000, 9_000_000, (n_days, n_stocks)).astype(float),
        index=dates, columns=codes,
    )
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


def test_sma_matches_recursive_definition():
    """ops.sma(X, n, m) == recursive Y[t] = (X[t]*m + Y[t-1]*(n-m))/n."""
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = ops.sma(x, 2, 1)["a"].tolist()  # alpha = 1/2
    # Y0=1; Y1=.5*2+.5*1=1.5; Y2=.5*3+.5*1.5=2.25; Y3=.5*4+.5*2.25=3.125; Y4=4.0625
    assert out == pytest.approx([1.0, 1.5, 2.25, 3.125, 4.0625])

    # n=3, m=1 → alpha=1/3, against an explicit recursion.
    xs = [10.0, 20.0, 0.0, 40.0]
    expected, prev = [], None
    for v in xs:
        prev = v if prev is None else (v * 1 + prev * (3 - 1)) / 3
        expected.append(prev)
    got = ops.sma(pd.DataFrame({"a": xs}), 3, 1)["a"].tolist()
    assert got == pytest.approx(expected)


def test_count_counts_true_in_window():
    """ops.count(cond, d) = rolling count of True over trailing d (strict)."""
    c = pd.DataFrame({"a": [1.0, 2.0, 1.5, 3.0, 2.0]})
    cond = c > c.shift(1)  # pandas: cmp with NaN -> False, so [F,T,F,T,F] = [0,1,0,1,0]
    out = ops.count(cond, 3)["a"].tolist()
    # rolling(3, min_periods=3).sum(): idx0,1 -> NaN; idx2=[0,1,0]=1; idx3=[1,0,1]=2; idx4=[0,1,0]=1
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == 1.0 and out[3] == 2.0 and out[4] == 1.0


def test_sma_rejects_bad_params():
    x = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        ops.sma(x, 2, 3)   # m > n
    with pytest.raises(ValueError):
        ops.sma(x, 5, 0)   # m == 0


def test_gtja_family_registered():
    registered = set(list_factors())
    for name in EXPECTED:
        assert name in registered, f"{name} not registered"


def test_gtja_compute_shape_and_no_crash():
    panel = _synth_panel()
    shape = panel["close"].shape
    for name in EXPECTED:
        out = make_factor(name).compute(panel)
        assert isinstance(out, pd.DataFrame)
        assert out.shape == shape, f"{name}: shape {out.shape} != {shape}"
        # no ±inf should leak (safe_div / clean ops)
        assert not np.isinf(out.to_numpy(dtype=float)).any(), f"{name} produced inf"


def test_gtja_short_window_factors_have_coverage():
    """Short-window alphas must produce real values on the post-warmup tail."""
    panel = _synth_panel()
    for name in SHORT_WINDOW:
        out = make_factor(name).compute(panel)
        tail = out.iloc[-30:]
        frac = tail.notna().to_numpy().mean()
        assert frac > 0.5, f"{name}: only {frac:.0%} non-NaN in last 30 days"
