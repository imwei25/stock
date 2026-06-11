"""baostock_backend 加固回归测试 (P2-19/20/21)。

覆盖:
- P2-19 线程安全:8 线程并发 fetch_stock,"登录 + 查询 + 取行"整段不被重入
- P2-20 停牌行过滤:fetch_stock 请求 tradestatus 并过滤 !="1" 的填充行;
  fetch_index 不请求 tradestatus(指数查询不支持该字段)
- P2-21 增量空结果:start 给定时空结果返回空 DataFrame(列/dtype 正确),
  全量(start=None)空结果仍 raise
- 回归:adjustflag 仍为 "1"(后复权)
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pandas as pd
import pytest

from stockpool.data_sources import baostock_backend

STOCK_FIELDS = "date,open,high,low,close,volume,tradestatus"
INDEX_FIELDS = "date,open,high,low,close,volume"
OUT_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _row(cols: list[str], *, date: str, base: float = 0.0, tradestatus: str = "1") -> list[str]:
    mapping = {
        "date": date,
        "open": str(10.0 + base),
        "high": str(11.0 + base),
        "low": str(9.0 + base),
        "close": str(10.5 + base),
        "volume": "100000",
        "tradestatus": tradestatus,
    }
    return [mapping[c] for c in cols]


class _FakeRS:
    error_code = "0"
    error_msg = ""

    def __init__(self, fields, rows, on_iter=None, on_exhaust=None):
        self.fields = fields
        self._rows = list(rows)
        self._on_iter = on_iter
        self._on_exhaust = on_exhaust

    def next(self):
        if self._on_iter is not None:
            self._on_iter(self)
        if self._rows:
            return True
        if self._on_exhaust is not None:
            self._on_exhaust(self)
        return False

    def get_row_data(self):
        return self._rows.pop(0)


def _make_fake_bs(query_fn):
    return types.SimpleNamespace(
        login=lambda: types.SimpleNamespace(error_code="0", error_msg=""),
        query_history_k_data_plus=query_fn,
    )


def _install(monkeypatch, fake_bs):
    monkeypatch.setitem(sys.modules, "baostock", fake_bs)
    monkeypatch.setattr(baostock_backend, "_logged_in", False)
    # 失败路径不要真睡 2+4 秒
    monkeypatch.setattr(baostock_backend, "_RETRY_DELAYS", [0])


# ---------------------------------------------------------------------------
# P2-19 线程安全
# ---------------------------------------------------------------------------

class _ConcurrencyFakeBS:
    """模拟 baostock 全局单 socket:'query + 取行' 整段被其他线程重入即记 violation。"""

    def __init__(self):
        self.violations = 0
        self._current: _FakeRS | None = None  # 当前占用"socket"的结果集

    def login(self):
        return types.SimpleNamespace(error_code="0", error_msg="")

    def query_history_k_data_plus(self, code, fields, **kwargs):
        if self._current is not None:
            self.violations += 1  # 上一个查询还没取完行就来了新查询
        cols = fields.split(",")
        base = float(code[-1])
        rows = [
            _row(cols, date="2026-01-05", base=base),
            _row(cols, date="2026-01-06", base=base),
        ]

        def on_iter(rs):
            time.sleep(0.001)
            if self._current is not rs:
                self.violations += 1  # 取行期间 socket 被别的查询抢走

        def on_exhaust(rs):
            if self._current is rs:
                self._current = None

        rs = _FakeRS(cols, rows, on_iter=on_iter, on_exhaust=on_exhaust)
        self._current = rs
        time.sleep(0.002)
        return rs


def test_concurrent_fetch_stock_is_serialized(monkeypatch):
    fake = _ConcurrencyFakeBS()
    _install(monkeypatch, fake)

    codes = [f"60000{i}" for i in range(8)]
    results: dict[str, pd.DataFrame] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(len(codes))

    def worker(code: str) -> None:
        try:
            barrier.wait()
            results[code] = baostock_backend.fetch_stock(code, start="2026-01-01")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(c,)) for c in codes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发查询报错: {errors}"
    assert fake.violations == 0, (
        f"临界区被重入 {fake.violations} 次 —— 'query + 取行' 必须整段持锁"
    )
    for code in codes:
        df = results[code]
        assert len(df) == 2
        # 每只票拿到自己的数据(close 编码了 code 尾号)
        assert df["close"].iloc[0] == pytest.approx(10.5 + float(code[-1]))


# ---------------------------------------------------------------------------
# P2-20 停牌行过滤
# ---------------------------------------------------------------------------

def test_fetch_stock_filters_suspended_rows(monkeypatch):
    captured: dict = {}

    def fake_query(code, fields, **kwargs):
        captured["fields"] = fields
        cols = fields.split(",")
        rows = [
            _row(cols, date="2026-01-05", tradestatus="1"),
            _row(cols, date="2026-01-06", tradestatus="0"),  # 停牌填充行
            _row(cols, date="2026-01-07", tradestatus="1"),
        ]
        return _FakeRS(cols, rows)

    _install(monkeypatch, _make_fake_bs(fake_query))

    df = baostock_backend.fetch_stock("605589", start="2026-01-01")
    assert captured["fields"] == STOCK_FIELDS
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2026-01-05", "2026-01-07"]
    assert "tradestatus" not in df.columns
    assert list(df.columns) == OUT_COLUMNS


def test_fetch_index_does_not_request_tradestatus(monkeypatch):
    captured: dict = {}

    def fake_query(code, fields, **kwargs):
        captured["fields"] = fields
        cols = fields.split(",")
        rows = [_row(cols, date="2026-01-05"), _row(cols, date="2026-01-06")]
        return _FakeRS(cols, rows)

    _install(monkeypatch, _make_fake_bs(fake_query))

    df = baostock_backend.fetch_index("sh000001")
    assert captured["fields"] == INDEX_FIELDS  # 指数查询不支持 tradestatus
    assert len(df) == 2


def test_incremental_all_rows_suspended_returns_empty(monkeypatch):
    def fake_query(code, fields, **kwargs):
        cols = fields.split(",")
        rows = [_row(cols, date="2026-01-06", tradestatus="0")]
        return _FakeRS(cols, rows)

    _install(monkeypatch, _make_fake_bs(fake_query))

    df = baostock_backend.fetch_stock("605589", start="2026-06-01")
    assert df.empty
    assert list(df.columns) == OUT_COLUMNS


# ---------------------------------------------------------------------------
# P2-21 增量空结果
# ---------------------------------------------------------------------------

def test_incremental_empty_returns_empty_df(monkeypatch):
    def fake_query(code, fields, **kwargs):
        return _FakeRS(fields.split(","), [])

    _install(monkeypatch, _make_fake_bs(fake_query))

    df = baostock_backend.fetch_stock("605589", start="2026-06-01")
    assert df.empty
    assert list(df.columns) == OUT_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        assert pd.api.types.is_float_dtype(df[c]), f"{c} 应为 float dtype"


def test_full_fetch_empty_still_raises(monkeypatch):
    def fake_query(code, fields, **kwargs):
        return _FakeRS(fields.split(","), [])

    _install(monkeypatch, _make_fake_bs(fake_query))

    with pytest.raises(RuntimeError, match="empty"):
        baostock_backend.fetch_stock("605589")  # start=None → 全量,空=fail loud


def test_full_fetch_index_empty_still_raises(monkeypatch):
    def fake_query(code, fields, **kwargs):
        return _FakeRS(fields.split(","), [])

    _install(monkeypatch, _make_fake_bs(fake_query))

    with pytest.raises(RuntimeError, match="empty"):
        baostock_backend.fetch_index("sh000001")


# ---------------------------------------------------------------------------
# 回归:复权参数
# ---------------------------------------------------------------------------

def test_adjustflag_still_hfq(monkeypatch):
    captured: dict = {}

    def fake_query(code, fields, **kwargs):
        captured.update(kwargs)
        cols = fields.split(",")
        return _FakeRS(cols, [_row(cols, date="2026-01-05")])

    _install(monkeypatch, _make_fake_bs(fake_query))

    df = baostock_backend.fetch_stock("605589", start="2026-01-01")
    assert captured.get("adjustflag") == "1", (
        f"baostock 应使用后复权 adjustflag=1,实际 {captured.get('adjustflag')!r}"
    )
    assert len(df) == 1
