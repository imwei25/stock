"""Tests for A-share custom factors."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _reset_sector_map():
    from stockpool.factors.context import set_sector_map
    set_sector_map({})
    yield
    set_sector_map({})


def _make_panel(prices_dict, vols_dict=None):
    """Build minimal OHLCV panel from {code: [close_series]}."""
    codes = list(prices_dict.keys())
    n_bars = len(next(iter(prices_dict.values())))
    dates = pd.date_range("2024-01-01", periods=n_bars)
    close = pd.DataFrame(prices_dict, index=dates)
    volume = (
        pd.DataFrame(vols_dict, index=dates)
        if vols_dict is not None
        else pd.DataFrame(1.0, index=dates, columns=codes)
    )
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": volume,
    }


def test_industry_relative_strength_basic():
    """Factor = own n-day return minus sector median n-day return."""
    from stockpool.factors.context import set_sector_map
    from stockpool.factors.registry import make_factor

    # 2 sectors (X, Y), 3 stocks each. 20-bar prices designed so that
    # at t=20, returns are known.
    n = 20
    base = np.linspace(10.0, 20.0, n + 1)  # +100% over n bars for everyone
    # Stock A (sector X): final price 22 → return = (22/10) - 1 = 1.2
    # Stock B (sector X): final price 20 → return = 1.0
    # Stock C (sector Y): final price 30 → return = 2.0
    # Stock D (sector Y): final price 15 → return = 0.5
    prices = {
        "A": np.r_[base[:-1], [22.0]],
        "B": np.r_[base[:-1], [20.0]],
        "C": np.r_[base[:-1], [30.0]],
        "D": np.r_[base[:-1], [15.0]],
    }
    panel = _make_panel(prices)
    set_sector_map({"A": "X", "B": "X", "C": "Y", "D": "Y"})

    factor = make_factor(f"industry_relative_strength_{n}")
    out = factor.compute(panel)

    last = out.iloc[-1]
    # X median = (1.2 + 1.0) / 2 = 1.1
    # Y median = (2.0 + 0.5) / 2 = 1.25
    assert last["A"] == pytest.approx(1.2 - 1.1)
    assert last["B"] == pytest.approx(1.0 - 1.1)
    assert last["C"] == pytest.approx(2.0 - 1.25)
    assert last["D"] == pytest.approx(0.5 - 1.25)


def test_industry_relative_strength_no_sector_map():
    """Empty sector_map → entire output is NaN."""
    from stockpool.factors.registry import make_factor

    prices = {"A": np.linspace(10, 12, 25), "B": np.linspace(10, 11, 25)}
    panel = _make_panel(prices)

    factor = make_factor("industry_relative_strength_20")
    out = factor.compute(panel)
    assert out.isna().all().all()


def test_industry_relative_strength_singleton_sector():
    """Sector with only 1 stock → that column NaN at last bar."""
    from stockpool.factors.context import set_sector_map
    from stockpool.factors.registry import make_factor

    n = 20
    prices = {
        "A": np.r_[np.linspace(10, 19, n), [22.0]],
        "B": np.r_[np.linspace(10, 19, n), [20.0]],
        "C": np.r_[np.linspace(10, 19, n), [30.0]],  # solo in sector Y
    }
    panel = _make_panel(prices)
    set_sector_map({"A": "X", "B": "X", "C": "Y"})

    factor = make_factor(f"industry_relative_strength_{n}")
    out = factor.compute(panel).iloc[-1]
    assert not np.isnan(out["A"])
    assert not np.isnan(out["B"])
    assert np.isnan(out["C"])


def test_industry_relative_strength_unmapped_stock():
    """Stock not in sector_map → that column NaN at last bar; others unaffected."""
    from stockpool.factors.context import set_sector_map
    from stockpool.factors.registry import make_factor

    n = 20
    prices = {
        "A": np.r_[np.linspace(10, 19, n), [22.0]],
        "B": np.r_[np.linspace(10, 19, n), [20.0]],
        "MISSING": np.r_[np.linspace(10, 19, n), [99.0]],
    }
    panel = _make_panel(prices)
    set_sector_map({"A": "X", "B": "X"})  # MISSING omitted

    factor = make_factor(f"industry_relative_strength_{n}")
    out = factor.compute(panel).iloc[-1]
    assert np.isnan(out["MISSING"])
    assert not np.isnan(out["A"])
    assert not np.isnan(out["B"])


def test_industry_relative_strength_look_ahead():
    """Truncating panel must not change earlier rows."""
    from stockpool.factors.context import set_sector_map
    from stockpool.factors.registry import make_factor

    n = 20
    rng = np.random.RandomState(42)
    prices = {
        "A": np.cumsum(rng.normal(0, 0.5, 50)) + 100,
        "B": np.cumsum(rng.normal(0, 0.5, 50)) + 100,
        "C": np.cumsum(rng.normal(0, 0.5, 50)) + 100,
    }
    panel_full = _make_panel(prices)
    panel_trunc = {k: v.iloc[:-5] for k, v in panel_full.items()}
    set_sector_map({"A": "X", "B": "X", "C": "Y"})

    factor = make_factor(f"industry_relative_strength_{n}")
    full_out = factor.compute(panel_full).iloc[:-5]
    trunc_out = factor.compute(panel_trunc)

    pd.testing.assert_frame_equal(full_out, trunc_out)


def test_industry_relative_strength_registered():
    from stockpool.factors.registry import get_spec
    spec = get_spec("industry_relative_strength")
    assert spec.sources == ("custom",)
    assert "industry_neutral" in spec.types
    assert "momentum" in spec.types
