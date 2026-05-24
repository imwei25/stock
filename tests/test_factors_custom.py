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


# ── LimitUpCountFactor ──────────────────────────────────────────────────────

def test_limit_up_count_basic():
    """Count of bars with pct_change > 0.099 in rolling N window."""
    from stockpool.factors.registry import make_factor

    # Daily returns design:
    #   bar 0:  NaN (no prev)
    #   bar 1:  +5%  → 0
    #   bar 2:  +10% → 1 (0.10 > 0.099)
    #   bar 3:  +9.9% → 0 (0.099 not strictly > 0.099)
    #   bar 4:  +11% → 1
    #   bar 5:  +9.95% → 1
    # With n=3:
    #   bars 0,1,2 → NaN (warmup, need 3 non-NaN observations)
    #   bar 3: window [bar1, bar2, bar3] is_limit = [0, 1, 0] → sum = 1
    #   bar 4: window [bar2, bar3, bar4] is_limit = [1, 0, 1] → sum = 2
    #   bar 5: window [bar3, bar4, bar5] is_limit = [0, 1, 1] → sum = 2
    rets = [np.nan, 0.05, 0.10, 0.099, 0.11, 0.0995]
    close = [100.0]
    for r in rets[1:]:
        close.append(close[-1] * (1 + r))
    prices = {"A": np.array(close)}
    panel = _make_panel(prices)

    factor = make_factor("limit_up_count_3")
    out = factor.compute(panel)["A"].tolist()
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    assert out[3] == pytest.approx(1.0)
    assert out[4] == pytest.approx(2.0)
    assert out[5] == pytest.approx(2.0)


def test_limit_up_count_warmup_nan():
    """First n bars must be NaN due to rolling min_periods=n."""
    from stockpool.factors.registry import make_factor

    prices = {"A": np.linspace(100, 110, 25)}  # smooth, no limit-ups
    panel = _make_panel(prices)
    factor = make_factor("limit_up_count_20")
    out = factor.compute(panel)["A"]
    # bars [0, 19] are warmup (need 20 non-NaN; bar 0 has NaN pct_change)
    assert out.iloc[:20].isna().all()
    # later bars are valid
    assert out.iloc[20:].notna().all()


def test_limit_up_count_look_ahead():
    from stockpool.factors.registry import make_factor

    rng = np.random.RandomState(7)
    prices = {"A": np.cumsum(rng.normal(0, 1, 50)) + 100}
    panel_full = _make_panel(prices)
    panel_trunc = {k: v.iloc[:-5] for k, v in panel_full.items()}

    factor = make_factor("limit_up_count_20")
    full_out = factor.compute(panel_full).iloc[:-5]
    trunc_out = factor.compute(panel_trunc)
    pd.testing.assert_frame_equal(full_out, trunc_out)


def test_limit_up_count_registered():
    from stockpool.factors.registry import get_spec
    spec = get_spec("limit_up_count")
    assert spec.sources == ("custom",)
    assert "momentum" in spec.types


# ── TurnoverZScoreFactor ────────────────────────────────────────────────────

def test_turnover_zscore_basic():
    """log(volume) z-scored over rolling N=3 window."""
    from stockpool.factors.registry import make_factor

    # Volumes: [1, 1, 1, 100, 1, 1]
    # log:      [0, 0, 0, log(100), 0, 0]
    # At bar 3 (window 1..3 = [0, 0, log(100)]): mean=log(100)/3, std≠0
    vols = [1.0, 1.0, 1.0, 100.0, 1.0, 1.0]
    prices = {"A": np.full(len(vols), 100.0)}
    panel = _make_panel(prices, vols_dict={"A": np.array(vols)})

    factor = make_factor("turnover_zscore_3")
    out = factor.compute(panel)["A"]
    # bars 0..1 NaN (warmup), bar 2 OK (window [0,0,0] → std=0 → NaN via replace)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])  # std=0 → NaN
    # bar 3: log(100) is a positive outlier, z-score > 0
    assert out.iloc[3] > 0


def test_turnover_zscore_zero_volume():
    """volume=0 (suspension) → log undefined → NaN; non-zero rows unaffected."""
    from stockpool.factors.registry import make_factor

    vols = [1.0, 2.0, 0.0, 4.0, 5.0]  # bar 2 suspended
    prices = {"A": np.full(len(vols), 100.0)}
    panel = _make_panel(prices, vols_dict={"A": np.array(vols)})

    factor = make_factor("turnover_zscore_3")
    out = factor.compute(panel)["A"]
    # bar 2 has volume=0 → log → NaN → window calc fails for bars containing it
    assert np.isnan(out.iloc[2])


def test_turnover_zscore_warmup_nan():
    """rolling(60, min_periods=60) → first 59 rows NaN (first valid at index 59)."""
    from stockpool.factors.registry import make_factor

    rng = np.random.RandomState(11)
    vols = np.abs(rng.normal(1000, 100, 80))
    prices = {"A": np.full(len(vols), 100.0)}
    panel = _make_panel(prices, vols_dict={"A": vols})

    factor = make_factor("turnover_zscore_60")
    out = factor.compute(panel)["A"]
    # min_periods=60 → window covering rows 0..59 (60 obs) yields first valid at idx 59
    assert out.iloc[:59].isna().all()
    # at least some later bars should be finite
    assert out.iloc[59:].notna().any()


def test_turnover_zscore_look_ahead():
    from stockpool.factors.registry import make_factor

    rng = np.random.RandomState(13)
    vols = np.abs(rng.normal(1000, 100, 80))
    prices = {"A": np.full(len(vols), 100.0)}
    panel_full = _make_panel(prices, vols_dict={"A": vols})
    panel_trunc = {k: v.iloc[:-5] for k, v in panel_full.items()}

    factor = make_factor("turnover_zscore_60")
    full_out = factor.compute(panel_full).iloc[:-5]
    trunc_out = factor.compute(panel_trunc)
    pd.testing.assert_frame_equal(full_out, trunc_out)


def test_turnover_zscore_registered():
    from stockpool.factors.registry import get_spec
    spec = get_spec("turnover_zscore")
    assert spec.sources == ("custom",)
    assert "volume" in spec.types
