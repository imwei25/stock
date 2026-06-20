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
    def fake_fetch(table, codes, partial_path=None):
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
    def fake_fetch(table, codes, partial_path=None):
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

    def fake_fetch(table, codes, partial_path=None):
        raise RuntimeError("network down")
    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    result = fl.load_or_build_fundamentals("profit", cache_dir=tmp_path, max_age_days=30)
    assert len(result) == 12  # 从 stale 缓存读


def test_load_or_build_fundamentals_unknown_table_raises(tmp_path):
    """非法 table 名 → ValueError。"""
    from stockpool.fundamentals_loader import load_or_build_fundamentals

    with pytest.raises(ValueError, match="table"):
        load_or_build_fundamentals("does_not_exist", cache_dir=tmp_path)


def test_write_partial_dumps_dataframe(tmp_path):
    """_write_partial 把 rows 写成 parquet,datetime 列正确化。"""
    from stockpool.fundamentals_loader import _write_partial

    rows = [
        {"code": "600000", "pubDate": "2024-12-15", "roeAvg": "0.12"},
        {"code": "000001", "pubDate": "2024-12-20", "roeAvg": "0.08"},
    ]
    partial = tmp_path / "fundamentals_profit.partial.parquet"
    _write_partial(rows, partial)
    assert partial.exists()

    df = pd.read_parquet(partial)
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["pubDate"])


def test_write_partial_empty_rows_skip(tmp_path):
    """空 rows → 不写文件,无 crash。"""
    from stockpool.fundamentals_loader import _write_partial

    partial = tmp_path / "fundamentals_profit.partial.parquet"
    _write_partial([], partial)
    assert not partial.exists()


def test_fetch_table_resumes_from_partial(monkeypatch, tmp_path):
    """partial 已含部分 code → 本次只补未抓的,不重抓已有的。"""
    from stockpool import fundamentals_loader as fl

    partial = tmp_path / "fundamentals_profit.partial.parquet"
    pd.DataFrame([
        {"code": "600000", "pubDate": pd.Timestamp("2024-12-15"), "roeAvg": 0.12},
    ]).to_parquet(partial, index=False)

    # 替换 _fetch_table 的内部 baostock 调用:用 fake codes & fake baostock
    fetched: list[str] = []

    class FakeRs:
        error_code = "0"
        fields = ["pubDate", "statDate", "roeAvg"]
        _yielded = False
        def __init__(self, code):
            self.code = code
            self._rows = [
                ["2024-12-20", "2024-09-30", "0.10"],
            ]
            self._i = 0
        def next(self):
            if self._i < len(self._rows):
                self._row = self._rows[self._i]
                self._i += 1
                return True
            return False
        def get_row_data(self):
            return self._row

    class FakeBs:
        @staticmethod
        def login():
            class R: error_code = "0"; error_msg = ""
            return R()
        @staticmethod
        def logout():
            pass
        @staticmethod
        def query_profit_data(code, year, quarter):
            fetched.append(code)
            return FakeRs(code)

    import sys
    monkeypatch.setitem(sys.modules, "baostock", FakeBs)

    df = fl._fetch_table(
        "profit", codes=["600000", "000001"], partial_path=partial,
    )
    # 600000 在 partial 中,本次不应再被抓
    assert not any(c.endswith(".600000") for c in fetched), \
        f"600000 应该跳过,但被重抓了: {fetched}"
    # 000001 必须被抓(它不在 partial)
    assert any(c.endswith(".000001") for c in fetched), \
        f"000001 应该被抓,但 fetched={fetched}"
    # 最终 DataFrame 含两个 code
    assert set(df["code"]) == {"600000", "000001"}


def test_fetch_table_corrupt_partial_restarts_from_scratch(monkeypatch, tmp_path):
    """partial parquet 损坏 → warning + 从头开始抓全部 codes。"""
    from stockpool import fundamentals_loader as fl

    partial = tmp_path / "fundamentals_profit.partial.parquet"
    partial.write_bytes(b"not a valid parquet")

    fetched: list[str] = []

    class FakeRs:
        error_code = "0"
        fields = ["pubDate", "statDate", "roeAvg"]
        def __init__(self): self._used = False
        def next(self):
            if not self._used:
                self._used = True
                return True
            return False
        def get_row_data(self):
            return ["2024-12-20", "2024-09-30", "0.10"]

    class FakeBs:
        @staticmethod
        def login():
            class R: error_code = "0"; error_msg = ""
            return R()
        @staticmethod
        def logout(): pass
        @staticmethod
        def query_profit_data(code, year, quarter):
            fetched.append(code)
            return FakeRs()

    import sys
    monkeypatch.setitem(sys.modules, "baostock", FakeBs)

    fl._fetch_table("profit", codes=["600000"], partial_path=partial)
    # 损坏 partial 应被忽略,所有 codes 都被抓
    assert any(c.endswith(".600000") for c in fetched)


def test_load_or_build_fundamentals_clears_partial_after_success(monkeypatch, tmp_path):
    """成功落 final 后,partial parquet 应被删除。"""
    from stockpool import fundamentals_loader as fl

    partial = tmp_path / "fundamentals_profit.partial.parquet"
    pd.DataFrame([
        {"code": "600000", "pubDate": pd.Timestamp("2024-12-15"), "roeAvg": 0.12},
    ]).to_parquet(partial, index=False)
    assert partial.exists()

    def fake_fetch(table, codes, partial_path=None):
        return _mock_long_df()

    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    fl.load_or_build_fundamentals("profit", cache_dir=tmp_path)
    # final 落盘
    assert (tmp_path / "fundamentals_profit.parquet").exists()
    # partial 清理
    assert not partial.exists()
