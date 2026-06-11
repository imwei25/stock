"""P2-22:market_cap_neutralize 的 mcap 面板改用 PIT 逐季股本。

旧实现把最新 totalShare 快照静态广播到全历史(轻度前视,增发/回购错位);
profit 表本就有逐季 PIT 的 totalShare,按 pubDate ffill 即得。覆盖不到的
格子(早于基本面覆盖 / 缺数据的票)回退静态快照近似。
"""
import numpy as np
import pandas as pd
import pytest


def _write_profit(cache_dir, rows):
    df = pd.DataFrame(rows, columns=["code", "pubDate", "statDate", "totalShare"])
    df["pubDate"] = pd.to_datetime(df["pubDate"])
    df["statDate"] = pd.to_datetime(df["statDate"])
    # 其余 profit 字段缺失没关系——只取 totalShare
    (cache_dir / "fundamentals_profit.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / "fundamentals_profit.parquet", index=False)


def _write_snapshot(cache_dir, shares_by_code):
    snap = pd.DataFrame({
        "code": list(shares_by_code), "totalShare": list(shares_by_code.values()),
    })
    snap.to_parquet(cache_dir / "mcap_shares.parquet", index=False)


def _pool(dates, close=10.0):
    df = pd.DataFrame({
        "date": dates,
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6,
    })
    return {"600000": df}


def test_mcap_uses_pit_shares_with_static_fallback(tmp_path):
    from stockpool.strategy_factory import build_log_mcap_panel

    dates = pd.bdate_range("2025-01-06", periods=10)
    _write_profit(tmp_path, [
        ("600000", dates[2], "2024-12-31", 1e8),  # day2 披露:1 亿股
        ("600000", dates[6], "2025-03-31", 2e8),  # day6 披露:2 亿股(增发)
    ])
    _write_snapshot(tmp_path, {"600000": 3e8})  # 静态快照(最新口径)

    panel = build_log_mcap_panel(_pool(dates), tmp_path)
    assert panel is not None

    # 披露前(day0-1):PIT 无值 → 回退静态快照 3e8
    assert panel.iloc[0, 0] == pytest.approx(np.log(10.0 * 3e8))
    # day2-5:PIT 1 亿股
    assert panel.iloc[3, 0] == pytest.approx(np.log(10.0 * 1e8))
    # day6 起:PIT 2 亿股(增发被按公告日反映,而非静态广播)
    assert panel.iloc[7, 0] == pytest.approx(np.log(10.0 * 2e8))


def test_mcap_falls_back_to_snapshot_when_profit_missing(tmp_path):
    from stockpool.strategy_factory import build_log_mcap_panel

    dates = pd.bdate_range("2025-01-06", periods=5)
    _write_snapshot(tmp_path, {"600000": 3e8})

    panel = build_log_mcap_panel(_pool(dates), tmp_path)
    assert panel is not None
    assert panel.iloc[-1, 0] == pytest.approx(np.log(10.0 * 3e8))


def test_mcap_none_when_both_sources_missing(tmp_path):
    from stockpool.strategy_factory import build_log_mcap_panel

    dates = pd.bdate_range("2025-01-06", periods=5)
    panel = build_log_mcap_panel(_pool(dates), tmp_path)
    assert panel is None
