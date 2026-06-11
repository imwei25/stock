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
    """Avg amount = close * volume(volume 单位 = 股, P1-6)。
    Just-above passes, just-below fails."""
    cfg = PortfolioEligibilityConfig(
        min_avg_amount_20d=5e7, exclude_st=False, min_history_bars=1,
    )
    # 10 * 5_000_100 = 50_001_000 > 5e7 ✓
    # 10 * 4_999_900 = 49_999_000 < 5e7 ✗
    panel = {
        "PASS": _mk_daily(30, 10.0, 5_000_100),
        "FAIL": _mk_daily(30, 10.0, 4_999_900),
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
    # First 30 bars have low volume, then volume jumps.(volume 单位 = 股)
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "close": [10.0] * 60,
        "volume": [1_000_000] * 30 + [10_000_000] * 30,   # 10M vs 100M 元
    })
    panel = {"A": df}
    f = EligibilityFilter(cfg)
    # At date 30 (low-vol period): avg = 10 * 1_000_000 = 10M < 50M → fail
    assert f.eligible(dates[29], panel) == set()
    # At date 59 (after jump): last 20 bars all 10M 股 → avg = 100M ≥ 50M → pass
    assert f.eligible(dates[59], panel) == {"A"}


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
