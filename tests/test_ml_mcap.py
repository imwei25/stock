"""Unit tests for log(market_cap) panel construction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _close_panel():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    return pd.DataFrame(
        {"600000": [10.0, 11.0, 12.0], "000001": [20.0, 22.0, 24.0]},
        index=dates,
    )


def test_build_log_mcap_panel_uses_close_times_total_share(monkeypatch, tmp_path):
    """mcap = close × totalShare → log(mcap), PIT-aligned by pubDate.

    totalShare lives in baostock's *profit* table (despite the name, "balance"
    in baostock is solvency ratios, not the balance sheet). See ml/mcap.py
    docstring for details.
    """
    from stockpool.ml.mcap import build_log_mcap_panel

    # Fake profit table: 6e8 shares for 600000 announced 2024-12-15;
    #                    1e9 shares for 000001 announced 2024-12-20.
    fake_profit = pd.DataFrame({
        "code": ["600000", "000001"],
        "pubDate": pd.to_datetime(["2024-12-15", "2024-12-20"]),
        "statDate": pd.to_datetime(["2024-09-30", "2024-09-30"]),
        "totalShare": [6e8, 1e9],
    })

    def fake_loader(table, cache_dir=None):
        assert table == "profit"
        return fake_profit

    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        fake_loader,
    )

    close = _close_panel()
    panel = {"close": close}
    log_mcap = build_log_mcap_panel(panel, cache_dir=str(tmp_path))

    # Expected: mcap[date, code] = close × totalShare (ffill from pubDate)
    expected_mcap = close.copy()
    expected_mcap["600000"] = close["600000"] * 6e8
    expected_mcap["000001"] = close["000001"] * 1e9
    expected_log = np.log(expected_mcap)

    pd.testing.assert_frame_equal(log_mcap, expected_log, check_dtype=False)


def test_build_log_mcap_panel_returns_nan_when_shares_missing(monkeypatch, tmp_path):
    """No totalShare row for a code → NaN log_mcap (so per-day OLS dropna handles it)."""
    from stockpool.ml.mcap import build_log_mcap_panel

    fake_balance = pd.DataFrame({
        "code": ["600000"],  # 000001 missing
        "pubDate": pd.to_datetime(["2024-12-15"]),
        "statDate": pd.to_datetime(["2024-09-30"]),
        "totalShare": [6e8],
    })
    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: fake_balance,
    )
    close = _close_panel()
    log_mcap = build_log_mcap_panel({"close": close}, cache_dir=str(tmp_path))
    assert log_mcap["000001"].isna().all()
    assert log_mcap["600000"].notna().all()


def test_build_log_mcap_panel_handles_empty_balance(monkeypatch, tmp_path):
    """Empty balance table → all-NaN log_mcap panel of correct shape."""
    from stockpool.ml.mcap import build_log_mcap_panel

    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: pd.DataFrame(),
    )
    close = _close_panel()
    log_mcap = build_log_mcap_panel({"close": close}, cache_dir=str(tmp_path))
    assert log_mcap.shape == close.shape
    assert log_mcap.isna().all().all()
