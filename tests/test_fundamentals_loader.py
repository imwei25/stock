"""Tests for stockpool.fundamentals_loader — baostock 5-table PIT cache."""
from __future__ import annotations

import os
import time

import pandas as pd
import pytest


def _mock_long_df():
    """3 codes × 4 quarters mock fundamentals DataFrame."""
    rows = []
    for code in ["000001", "600000", "300001"]:
        for q_idx, (year, q) in enumerate([(2023, 4), (2024, 1), (2024, 2), (2024, 3)]):
            rows.append({
                "code": code,
                "pubDate": pd.Timestamp(f"{year}-{q*3:02d}-28") + pd.Timedelta(days=q_idx),
                "statDate": pd.Timestamp(f"{year}-{q*3:02d}-30"),
                "roeAvg": 0.12 + 0.01 * q_idx,
                "netProfit": 1e9 * (1 + 0.05 * q_idx),
            })
    return pd.DataFrame(rows)


def test_load_or_build_fundamentals_cache_hit(tmp_path):
    """Fresh cache parquet → 直接读盘,不调 baostock。"""
    from stockpool.fundamentals_loader import load_or_build_fundamentals

    df = _mock_long_df()
    cache = tmp_path / "fundamentals_profit.parquet"
    df.to_parquet(cache, index=False)

    result = load_or_build_fundamentals("profit", cache_dir=tmp_path)
    assert len(result) == 12
    assert set(result["code"]) == {"000001", "600000", "300001"}
    assert "pubDate" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["pubDate"])


def test_load_or_build_fundamentals_stale_triggers_refresh(monkeypatch, tmp_path):
    """Mtime 老于 max_age_days → 触发 _fetch_table。"""
    from stockpool import fundamentals_loader as fl

    cache = tmp_path / "fundamentals_profit.parquet"
    _mock_long_df().head(3).to_parquet(cache, index=False)
    old = time.time() - 60 * 86400
    os.utime(cache, (old, old))

    called = {"n": 0}
    def fake_fetch(table, codes):
        called["n"] += 1
        return _mock_long_df()
    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    result = fl.load_or_build_fundamentals("profit", cache_dir=tmp_path, max_age_days=30)
    assert called["n"] == 1
    assert len(result) == 12


def test_load_or_build_fundamentals_force_refresh(monkeypatch, tmp_path):
    """force_refresh=True → 即便缓存新鲜也重拉。"""
    from stockpool import fundamentals_loader as fl

    cache = tmp_path / "fundamentals_profit.parquet"
    _mock_long_df().head(3).to_parquet(cache, index=False)

    called = {"n": 0}
    def fake_fetch(table, codes):
        called["n"] += 1
        return _mock_long_df()
    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    fl.load_or_build_fundamentals("profit", cache_dir=tmp_path, force_refresh=True)
    assert called["n"] == 1


def test_load_or_build_fundamentals_fetch_fail_falls_back_to_stale(monkeypatch, tmp_path):
    """baostock 抛错 + 有 stale 缓存 → 用 stale 缓存。"""
    from stockpool import fundamentals_loader as fl

    cache = tmp_path / "fundamentals_profit.parquet"
    _mock_long_df().to_parquet(cache, index=False)
    old = time.time() - 60 * 86400
    os.utime(cache, (old, old))

    def fake_fetch(table, codes):
        raise RuntimeError("network down")
    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    result = fl.load_or_build_fundamentals("profit", cache_dir=tmp_path, max_age_days=30)
    assert len(result) == 12  # 从 stale 缓存读


def test_load_or_build_fundamentals_unknown_table_raises(tmp_path):
    """非法 table 名 → ValueError。"""
    from stockpool.fundamentals_loader import load_or_build_fundamentals

    with pytest.raises(ValueError, match="table"):
        load_or_build_fundamentals("does_not_exist", cache_dir=tmp_path)
