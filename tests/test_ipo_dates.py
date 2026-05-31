"""Tests for stockpool.ipo_dates — IPO date loader for listing_mask."""
from __future__ import annotations

import time

import pandas as pd
import pytest


def test_load_or_build_ipo_dates_cache_hit(tmp_path):
    """Fresh cache parquet → 直接读盘,不调 baostock。"""
    from stockpool.ipo_dates import load_or_build_ipo_dates

    df = pd.DataFrame({
        "code": ["600000", "300001"],
        "ipo_date": pd.to_datetime(["1999-11-10", "2009-10-30"]),
    })
    cache = tmp_path / "ipo_dates.parquet"
    df.to_parquet(cache, index=False)

    result = load_or_build_ipo_dates(tmp_path)
    assert result["600000"] == pd.Timestamp("1999-11-10")
    assert result["300001"] == pd.Timestamp("2009-10-30")


def test_load_or_build_ipo_dates_stale_triggers_refresh(monkeypatch, tmp_path):
    """Mtime 老于 max_age_days → 触发 _fetch_from_baostock。"""
    from stockpool import ipo_dates

    cache = tmp_path / "ipo_dates.parquet"
    df_old = pd.DataFrame({
        "code": ["600000"], "ipo_date": pd.to_datetime(["1999-11-10"]),
    })
    df_old.to_parquet(cache, index=False)

    # 强制 mtime = 60 天前
    old_mtime = time.time() - 60 * 86400
    cache.touch()
    import os
    os.utime(cache, (old_mtime, old_mtime))

    called = {"count": 0}

    def fake_fetch():
        called["count"] += 1
        return pd.DataFrame({
            "code": ["600000", "601398"],
            "ipo_date": pd.to_datetime(["1999-11-10", "2006-10-27"]),
        })

    monkeypatch.setattr(ipo_dates, "_fetch_from_baostock", fake_fetch)

    result = ipo_dates.load_or_build_ipo_dates(tmp_path, max_age_days=30)
    assert called["count"] == 1
    assert "601398" in result
    assert result["601398"] == pd.Timestamp("2006-10-27")


def test_load_or_build_ipo_dates_force_refresh(monkeypatch, tmp_path):
    from stockpool import ipo_dates

    df_old = pd.DataFrame({
        "code": ["600000"], "ipo_date": pd.to_datetime(["1999-11-10"]),
    })
    (tmp_path / "ipo_dates.parquet").write_bytes(b"")  # placeholder
    df_old.to_parquet(tmp_path / "ipo_dates.parquet", index=False)

    called = {"count": 0}

    def fake_fetch():
        called["count"] += 1
        return pd.DataFrame({
            "code": ["999999"], "ipo_date": pd.to_datetime(["2024-01-01"]),
        })

    monkeypatch.setattr(ipo_dates, "_fetch_from_baostock", fake_fetch)

    result = ipo_dates.load_or_build_ipo_dates(tmp_path, force_refresh=True)
    assert called["count"] == 1
    assert result == {"999999": pd.Timestamp("2024-01-01")}


def test_load_or_build_ipo_dates_fetch_failure_uses_stale_cache(monkeypatch, tmp_path):
    """Fetch 失败但有旧缓存 → 返回旧缓存(不直接 0 返回)。"""
    from stockpool import ipo_dates

    cache = tmp_path / "ipo_dates.parquet"
    df_old = pd.DataFrame({
        "code": ["600000"], "ipo_date": pd.to_datetime(["1999-11-10"]),
    })
    df_old.to_parquet(cache, index=False)
    import os
    old_mtime = time.time() - 60 * 86400
    os.utime(cache, (old_mtime, old_mtime))

    def failing_fetch():
        raise RuntimeError("baostock offline")

    monkeypatch.setattr(ipo_dates, "_fetch_from_baostock", failing_fetch)

    result = ipo_dates.load_or_build_ipo_dates(tmp_path, max_age_days=30)
    # Failed fetch + stale cache → returns stale
    assert "600000" in result


def test_load_or_build_ipo_dates_fetch_failure_no_cache(monkeypatch, tmp_path):
    """无缓存 + fetch 失败 → 空字典(不 raise)。"""
    from stockpool import ipo_dates

    def failing_fetch():
        raise RuntimeError("baostock offline")

    monkeypatch.setattr(ipo_dates, "_fetch_from_baostock", failing_fetch)

    result = ipo_dates.load_or_build_ipo_dates(tmp_path)
    assert result == {}


def test_df_to_dict_strips_invalid_dates(tmp_path):
    """空白/NaN ipo_date 行被丢弃。"""
    from stockpool.ipo_dates import _df_to_dict

    df = pd.DataFrame({
        "code": ["600000", "888888"],
        "ipo_date": [pd.Timestamp("1999-11-10"), pd.NaT],
    })
    result = _df_to_dict(df)
    assert "600000" in result
    assert "888888" not in result
