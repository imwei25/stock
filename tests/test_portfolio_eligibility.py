"""Tests for portfolio.eligibility.EligibilityFilter."""
from __future__ import annotations

import pandas as pd
import pytest

from stockpool.config import PortfolioEligibilityConfig
from stockpool.portfolio.eligibility import EligibilityFilter, _is_st


def _mk_daily(n_bars: int, close: float, volume: float):
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n_bars, freq="B"),
        "close": [close] * n_bars,
        "volume": [volume] * n_bars,
    })


def test_min_history_bars_filters_short():
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=0, exclude_st=False, min_history_bars=60,
    )
    panel = {
        "A": _mk_daily(100, 10.0, 100_000),
        "B": _mk_daily(30, 10.0, 100_000),    # too short
    }
    f = EligibilityFilter(cfg)
    out = f.eligible(pd.Timestamp("2024-12-31"), panel)
    assert "A" in out
    assert "B" not in out


def test_liquidity_boundary():
    """Avg amount = close * volume * 100. Just-above passes, just-below fails."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=5e7, exclude_st=False, min_history_bars=1,
    )
    # 10 * 50_001 * 100 = 50_001_000 > 5e7 ✓
    # 10 * 49_999 * 100 = 49_999_000 < 5e7 ✗
    panel = {
        "PASS": _mk_daily(30, 10.0, 50_001),
        "FAIL": _mk_daily(30, 10.0, 49_999),
    }
    out = EligibilityFilter(cfg).eligible(pd.Timestamp("2024-12-31"), panel)
    assert out == {"PASS"}


def test_st_excluded():
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=0, exclude_st=True, min_history_bars=1,
    )
    panel = {"A": _mk_daily(10, 10.0, 100_000), "B": _mk_daily(10, 10.0, 100_000)}
    name_map = {"A": "正常股", "B": "*ST 雷股"}
    out = EligibilityFilter(cfg, name_map=name_map).eligible(
        pd.Timestamp("2024-12-31"), panel,
    )
    assert out == {"A"}


def test_st_disabled_keeps_st_stocks():
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=0, exclude_st=False, min_history_bars=1,
    )
    panel = {"A": _mk_daily(10, 10.0, 100_000)}
    name_map = {"A": "*ST 雷股"}
    out = EligibilityFilter(cfg, name_map=name_map).eligible(
        pd.Timestamp("2024-12-31"), panel,
    )
    assert out == {"A"}


def test_unknown_name_passes_st_check():
    """Codes missing from name_map are not assumed ST."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=0, exclude_st=True, min_history_bars=1,
    )
    panel = {"A": _mk_daily(10, 10.0, 100_000)}
    out = EligibilityFilter(cfg, name_map={}).eligible(
        pd.Timestamp("2024-12-31"), panel,
    )
    assert out == {"A"}


def test_date_truncation():
    """eligible at an early date sees only bars <= date_t (so liquidity changes)."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=5e7, exclude_st=False, min_history_bars=20,
    )
    # First 30 bars have low volume, then volume jumps.
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "close": [10.0] * 60,
        "volume": [10_000] * 30 + [100_000] * 30,   # 30M vs 100M
    })
    panel = {"A": df}
    f = EligibilityFilter(cfg)
    # At date 30 (low-vol period): avg = 10 * 10_000 * 100 = 10M < 50M → fail
    assert f.eligible(dates[29], panel) == set()
    # At date 59 (after jump): last 20 bars all 100k → avg = 100M ≥ 50M → pass
    assert f.eligible(dates[59], panel) == {"A"}


def test_top_n_liquidity_keeps_most_liquid():
    """top_n_liquidity=2 keeps the 2 highest as-of 20d-amount codes."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=0, exclude_st=False, min_history_bars=1,
        top_n_liquidity=2,
    )
    panel = {
        "HI": _mk_daily(30, 10.0, 300_000),
        "MID": _mk_daily(30, 10.0, 200_000),
        "LO": _mk_daily(30, 10.0, 100_000),
    }
    out = EligibilityFilter(cfg).eligible(pd.Timestamp("2024-12-31"), panel)
    assert out == {"HI", "MID"}


def test_top_n_liquidity_is_point_in_time():
    """Membership follows *as-of* liquidity: a stock that is liquid early and
    dries up later must be in the early membership and out of the late one —
    the whole point vs a static end-of-period pool file."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=0, exclude_st=False, min_history_bars=1,
        top_n_liquidity=1,
    )
    n = 200
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    fade = pd.DataFrame({  # liquid first half, illiquid second half
        "date": dates, "close": [10.0] * n,
        "volume": [500_000] * (n // 2) + [1_000] * (n // 2),
    })
    rise = pd.DataFrame({  # mirror image
        "date": dates, "close": [10.0] * n,
        "volume": [10_000] * (n // 2) + [400_000] * (n // 2),
    })
    f = EligibilityFilter(cfg)
    panel = {"FADE": fade, "RISE": rise}
    early = f.eligible(dates[n // 2 - 1], panel)
    late = f.eligible(dates[-1], panel)
    assert early == {"FADE"}
    assert late == {"RISE"}


def test_top_n_liquidity_none_is_legacy():
    """Default None applies no top-N cut (all threshold-passers kept)."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=0, exclude_st=False, min_history_bars=1,
    )
    panel = {
        "A": _mk_daily(30, 10.0, 300_000),
        "B": _mk_daily(30, 10.0, 1),
    }
    out = EligibilityFilter(cfg).eligible(pd.Timestamp("2024-12-31"), panel)
    assert out == {"A", "B"}


def test_top_n_liquidity_composes_with_threshold():
    """Threshold filters first; top-N ranks the survivors only."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=5e7, exclude_st=False, min_history_bars=1,
        top_n_liquidity=2,
    )
    panel = {
        "HI": _mk_daily(30, 10.0, 300_000),      # 3e8 → pass
        "MID": _mk_daily(30, 10.0, 60_000),      # 6e7 → pass
        "BIGBUTBELOW": _mk_daily(30, 10.0, 40_000),  # 4e7 → threshold-fail
    }
    out = EligibilityFilter(cfg).eligible(pd.Timestamp("2024-12-31"), panel)
    assert out == {"HI", "MID"}


def test_missing_volume_column_excluded():
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=5e7, exclude_st=False, min_history_bars=1,
    )
    df = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=10, freq="B"),
                       "close": [10.0] * 10})
    out = EligibilityFilter(cfg).eligible(pd.Timestamp("2024-12-31"), {"A": df})
    assert out == set()


def test_zero_threshold_skips_liquidity_check():
    """min_avg_amount_20d=0 → no volume check, codes without volume still pass."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=0, exclude_st=False, min_history_bars=1,
    )
    df = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=10, freq="B"),
                       "close": [10.0] * 10})
    out = EligibilityFilter(cfg).eligible(pd.Timestamp("2024-12-31"), {"A": df})
    assert out == {"A"}


@pytest.mark.parametrize("name,expected", [
    ("ST 雷股", True),
    ("*ST 雷股", True),
    ("st soft", True),
    ("贵州茅台", False),
    ("", False),
])
def test_is_st(name, expected):
    assert _is_st(name) is expected
