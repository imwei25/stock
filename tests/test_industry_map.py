"""Tests for stockpool.industry_map.

akshare is mocked everywhere — no network calls.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from stockpool import industry_map


def _stub_boards() -> pd.DataFrame:
    return pd.DataFrame({"板块名称": ["半导体", "化工", "电力"]})


def _stub_cons(symbol: str) -> pd.DataFrame:
    table = {
        "半导体": ["603986", "688008", "300475"],
        "化工":   ["605589", "603026"],
        "电力":   ["000922", "603986"],   # 603986 在两个板块,first wins
    }
    return pd.DataFrame({"代码": table[symbol]})


@pytest.fixture
def mock_ak(monkeypatch):
    """Patch the akshare functions used by industry_map."""
    import akshare as ak
    monkeypatch.setattr(ak, "stock_board_industry_name_em", _stub_boards)
    monkeypatch.setattr(ak, "stock_board_industry_cons_em",
                        lambda symbol: _stub_cons(symbol))


def test_fresh_build_writes_parquet(tmp_path: Path, mock_ak):
    mapping = industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="akshare",
    )
    cache_file = tmp_path / industry_map._CACHE_FILENAME
    assert cache_file.exists()

    # 603986 在半导体里先出现 → 应保留半导体
    assert mapping["603986"] == "半导体"
    assert mapping["605589"] == "化工"
    assert mapping["000922"] == "电力"
    # 6 unique codes (603986 appears in two boards, dedup'd)
    assert len(mapping) == 6


def test_cache_hit_skips_akshare(tmp_path: Path, mock_ak, monkeypatch):
    # 1) First call builds cache
    industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="akshare",
    )

    # 2) Replace akshare with a stub that explodes if called
    def _boom(*a, **k):
        raise AssertionError("akshare should not be called on cache hit")
    import akshare as ak
    monkeypatch.setattr(ak, "stock_board_industry_name_em", _boom)
    monkeypatch.setattr(ak, "stock_board_industry_cons_em", _boom)

    # 3) Second call must use the cache
    mapping = industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="akshare",
    )
    assert mapping["603986"] == "半导体"


def test_stale_cache_rebuilds(tmp_path: Path, mock_ak):
    industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="akshare",
    )
    cache_file = tmp_path / industry_map._CACHE_FILENAME
    # Backdate mtime 100 days
    old_mtime = time.time() - 100 * 86400
    import os
    os.utime(cache_file, (old_mtime, old_mtime))

    # Trigger rebuild with shorter window
    industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="akshare",
    )
    # mtime should now be fresh
    fresh_mtime = cache_file.stat().st_mtime
    assert (time.time() - fresh_mtime) < 5


def test_force_refresh_ignores_fresh_cache(tmp_path: Path, mock_ak):
    industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="akshare",
    )
    cache_file = tmp_path / industry_map._CACHE_FILENAME
    first_mtime = cache_file.stat().st_mtime
    time.sleep(0.05)  # ensure mtime resolution

    industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, force_refresh=True, source="akshare",
    )
    assert cache_file.stat().st_mtime > first_mtime


def test_akshare_failure_returns_empty_dict(tmp_path: Path, monkeypatch):
    """If akshare blows up entirely, return {} and let caller handle 'unknown'."""
    import akshare as ak
    monkeypatch.setattr(
        ak, "stock_board_industry_name_em",
        lambda: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    mapping = industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="akshare",
    )
    assert mapping == {}


def test_industry_of_returns_unknown_when_unmapped():
    assert industry_map.industry_of("999999", {"600000": "银行"}) == "未知"
    assert industry_map.industry_of("600000", {"600000": "银行"}) == "银行"


def test_individual_board_failure_is_isolated(tmp_path: Path, monkeypatch):
    """One bad akshare board should not nuke the whole build."""
    import akshare as ak

    def _bad_cons(symbol: str) -> pd.DataFrame:
        if symbol == "化工":
            raise RuntimeError("east-money 500")
        return _stub_cons(symbol)

    monkeypatch.setattr(ak, "stock_board_industry_name_em", _stub_boards)
    monkeypatch.setattr(ak, "stock_board_industry_cons_em", _bad_cons)

    mapping = industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="akshare",
    )
    # 半导体 + 电力 still mapped; 化工 skipped
    assert "603986" in mapping
    assert "000922" in mapping
    assert "605589" not in mapping


# === baostock source ===

class _StubBSResult:
    def __init__(self, rows: list[list[str]], fields: list[str]):
        self._rows = list(rows)
        self.fields = fields
        self.error_code = "0"
        self.error_msg = "ok"
        self._i = 0

    def next(self) -> bool:
        if self._i < len(self._rows):
            return True
        return False

    def get_row_data(self) -> list[str]:
        row = self._rows[self._i]
        self._i += 1
        return row


@pytest.fixture
def mock_baostock(monkeypatch):
    """Patch baostock's login / query_stock_industry / logout."""
    import baostock as bs

    class _Lg:
        error_code = "0"
        error_msg = "ok"

    rows = [
        ["2026-05-18", "sh.600000", "浦发银行", "J66货币金融服务", "证监会"],
        ["2026-05-18", "sh.603986", "兆易创新", "C39计算机", "证监会"],
        ["2026-05-18", "sz.000528", "柳工",     "C35专用设备", "证监会"],
        ["2026-05-18", "sh.605589", "圣泉集团", "",            "证监会"],  # empty → drop
    ]
    fields = ["updateDate", "code", "code_name", "industry", "industryClassification"]
    monkeypatch.setattr(bs, "login", lambda: _Lg())
    monkeypatch.setattr(bs, "logout", lambda: None)
    monkeypatch.setattr(bs, "query_stock_industry",
                        lambda: _StubBSResult(rows, fields))


def test_baostock_source_builds_and_strips_prefix(tmp_path: Path, mock_baostock):
    mapping = industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="baostock",
    )
    # sh./sz. prefix stripped, codes are 6-digit
    assert mapping["600000"] == "J66货币金融服务"
    assert mapping["603986"] == "C39计算机"
    assert mapping["000528"] == "C35专用设备"
    # empty industry filtered out
    assert "605589" not in mapping


def test_auto_uses_baostock_first(tmp_path: Path, mock_baostock, monkeypatch):
    """auto mode should pick baostock when it returns data, never touch akshare."""
    import akshare as ak

    def _boom_ak(*a, **k):
        raise AssertionError("akshare must not be called when baostock succeeds")
    monkeypatch.setattr(ak, "stock_board_industry_name_em", _boom_ak)

    mapping = industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="auto",
    )
    assert "600000" in mapping


def test_auto_falls_back_to_akshare(tmp_path: Path, mock_ak, monkeypatch):
    """If baostock raises, auto should still try akshare."""
    import baostock as bs

    def _fail_login():
        class L:
            error_code = "10001"
            error_msg = "down"
        return L()
    monkeypatch.setattr(bs, "login", _fail_login)
    monkeypatch.setattr(bs, "logout", lambda: None)

    mapping = industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="auto",
    )
    # akshare stub data still loaded
    assert mapping["603986"] == "半导体"


def test_auto_returns_empty_when_both_fail(tmp_path: Path, monkeypatch):
    import baostock as bs
    import akshare as ak

    monkeypatch.setattr(bs, "login",
                        lambda: type("L", (), {"error_code": "1", "error_msg": "x"})())
    monkeypatch.setattr(bs, "logout", lambda: None)
    monkeypatch.setattr(
        ak, "stock_board_industry_name_em",
        lambda: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    mapping = industry_map.load_or_build_industry_map(
        tmp_path, max_age_days=30, source="auto",
    )
    assert mapping == {}
