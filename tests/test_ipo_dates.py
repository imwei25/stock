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
    monkeypatch.setattr(ipo_dates, "_fetch_from_akshare", failing_fetch)

    result = ipo_dates.load_or_build_ipo_dates(tmp_path, max_age_days=30)
    # Failed fetch + stale cache → returns stale
    assert "600000" in result


def test_load_or_build_ipo_dates_fetch_failure_no_cache(monkeypatch, tmp_path):
    """无缓存 + fetch 失败 → 空字典(不 raise)。"""
    from stockpool import ipo_dates

    def failing_fetch():
        raise RuntimeError("baostock offline")

    monkeypatch.setattr(ipo_dates, "_fetch_from_baostock", failing_fetch)
    monkeypatch.setattr(ipo_dates, "_fetch_from_akshare", failing_fetch)

    result = ipo_dates.load_or_build_ipo_dates(tmp_path)
    assert result == {}


def test_auto_chain_falls_back_to_akshare(monkeypatch, tmp_path):
    """baostock 失败(黑名单)→ auto 链自动落到 akshare。"""
    from stockpool import ipo_dates

    def failing_baostock():
        raise RuntimeError("login failed: 黑名单用户")

    def fake_akshare():
        return pd.DataFrame({
            "code": ["600000", "000001"],
            "ipo_date": pd.to_datetime(["1999-11-10", "1991-04-03"]),
        })

    monkeypatch.setattr(ipo_dates, "_fetch_from_baostock", failing_baostock)
    monkeypatch.setattr(ipo_dates, "_fetch_from_akshare", fake_akshare)

    result = ipo_dates.load_or_build_ipo_dates(tmp_path)
    assert result["600000"] == pd.Timestamp("1999-11-10")
    assert result["000001"] == pd.Timestamp("1991-04-03")
    # akshare 结果落盘,二次调用走缓存
    assert (tmp_path / "ipo_dates.parquet").exists()


def test_akshare_source_only(monkeypatch, tmp_path):
    """source='akshare' 不碰 baostock。"""
    from stockpool import ipo_dates

    def must_not_call():
        raise AssertionError("baostock should not be called")

    def fake_akshare():
        return pd.DataFrame({
            "code": ["830799"], "ipo_date": pd.to_datetime(["2015-01-01"]),
        })

    monkeypatch.setattr(ipo_dates, "_fetch_from_baostock", must_not_call)
    monkeypatch.setattr(ipo_dates, "_fetch_from_akshare", fake_akshare)

    result = ipo_dates.load_or_build_ipo_dates(tmp_path, source="akshare")
    assert result == {"830799": pd.Timestamp("2015-01-01")}


def test_fetch_from_akshare_merges_exchanges(monkeypatch):
    """三张交易所名录拼接 + 列名兼容 + 单交易所失败只丢那一张。"""
    from stockpool import ipo_dates

    class FakeAk:
        @staticmethod
        def stock_info_sh_name_code():
            return pd.DataFrame({
                "证券代码": ["600000"], "证券简称": ["浦发银行"],
                "上市日期": ["1999-11-10"],
            })

        @staticmethod
        def stock_info_sz_name_code():
            return pd.DataFrame({
                "A股代码": ["000001"], "A股简称": ["平安银行"],
                "A股上市日期": ["1991-04-03"],
            })

        @staticmethod
        def stock_info_bj_name_code():
            raise RuntimeError("bj endpoint down")

    import sys as _sys
    monkeypatch.setitem(_sys.modules, "akshare", FakeAk)
    out = ipo_dates._fetch_from_akshare()
    assert set(out["code"]) == {"600000", "000001"}
    assert out.set_index("code").loc["000001", "ipo_date"] == pd.Timestamp("1991-04-03")


def test_unknown_source_raises():
    from stockpool import ipo_dates
    with pytest.raises(ValueError, match="unknown ipo_dates source"):
        ipo_dates._fetch("tushare")


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
