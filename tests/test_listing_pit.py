"""P0-4 轻量 / P2-23 / P3-4:PIT listing(干净名称 + outDate/status + ST 标记)。

- baostock query_stock_basic → stock_basics.parquet(code/name/ipo_date/out_date/status/is_st)
- mootdx list_a_shares 不再按乱码名剔 ST(训练池保留 ST,应用层各自决定)
- 训练标签 mask 对 ST 票用 ±5% 阈值
"""
import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


class _FakeRS:
    error_code = "0"
    error_msg = ""

    def __init__(self, fields, rows):
        self.fields = fields
        self._rows = list(rows)

    def next(self):
        return bool(self._rows)

    def get_row_data(self):
        return self._rows.pop(0)


_BASIC_FIELDS = ["code", "code_name", "ipoDate", "outDate", "type", "status"]
_BASIC_ROWS = [
    ["sh.600000", "浦发银行", "1999-11-10", "", "1", "1"],
    ["sz.000001", "平安银行", "1991-04-03", "", "1", "1"],
    ["sz.000003", "PT金田A", "1991-07-03", "2002-06-14", "1", "0"],  # 已退市
    ["sh.600001", "ST邯郸", "1998-01-22", "", "1", "1"],             # 在市 ST
    ["sh.000001", "上证指数", "1991-07-15", "", "2", "1"],            # 指数,应被剔
]


def _install_fake_bs(monkeypatch):
    fake = types.SimpleNamespace(
        login=lambda: types.SimpleNamespace(error_code="0", error_msg=""),
        logout=lambda: None,
        query_stock_basic=lambda: _FakeRS(_BASIC_FIELDS, [list(r) for r in _BASIC_ROWS]),
    )
    monkeypatch.setitem(sys.modules, "baostock", fake)
    return fake


def test_stock_basics_build_and_flags(tmp_path, monkeypatch):
    _install_fake_bs(monkeypatch)
    from stockpool.ipo_dates import load_or_build_stock_basics

    df = load_or_build_stock_basics(tmp_path)
    assert set(["code", "name", "ipo_date", "out_date", "status", "is_st"]) <= set(df.columns)
    assert "000001" in df["code"].values
    assert (tmp_path / "stock_basics.parquet").exists()
    # 指数(type=2)被剔
    assert "上证指数" not in df["name"].values
    # ST 标记(含 PT/*ST 变体按含 "ST" 判定)
    st = df[df["is_st"]]
    assert "600001" in st["code"].values
    # 退市股保留在表里(out_date 有值)——这正是 PIT 名单的意义
    delisted = df[df["code"] == "000003"].iloc[0]
    assert pd.Timestamp(delisted["out_date"]) == pd.Timestamp("2002-06-14")


def test_stock_basics_cache_hit_no_network(tmp_path, monkeypatch):
    _install_fake_bs(monkeypatch)
    from stockpool.ipo_dates import load_or_build_stock_basics
    load_or_build_stock_basics(tmp_path)
    # 第二次:卸掉 fake,若还碰网络会 import 失败/报错
    monkeypatch.setitem(sys.modules, "baostock", None)
    df = load_or_build_stock_basics(tmp_path)
    assert len(df) >= 3


def test_load_st_codes(tmp_path, monkeypatch):
    _install_fake_bs(monkeypatch)
    from stockpool.ipo_dates import load_or_build_stock_basics, load_st_codes
    load_or_build_stock_basics(tmp_path)  # 建缓存(fetch-universe 的职责)
    st = load_st_codes(tmp_path)
    assert "600001" in st
    assert "000001" not in st


def test_load_st_codes_no_cache_no_network(tmp_path):
    """缓存缺失时返回空集且不碰网络(高频路径安全)。"""
    from stockpool.ipo_dates import load_st_codes
    assert load_st_codes(tmp_path) == set()


def test_list_a_shares_keeps_st(monkeypatch):
    """训练池不再按当前名称整段剔除 ST(那是前视);ST 由干净名单标记,
    应用层(Pool B / 推荐)自行剔除。"""
    from stockpool.data_sources import mootdx_backend

    fake_stocks = pd.DataFrame({
        "code": ["000001", "000004"],
        "name": ["平安银行", "ST国华"],
    })
    with patch.object(mootdx_backend, "_call_with_retry", return_value=fake_stocks):
        out = mootdx_backend.list_a_shares()
    assert "000004" in out["code"].values, "ST 票应保留在 universe 里"


def test_mask_config_has_st_threshold():
    from stockpool.config import MaskConfig
    cfg = MaskConfig()
    assert cfg.limit_up_threshold_st == pytest.approx(0.048)


def test_tradability_mask_st_threshold(tmp_path):
    """主板 ST ±5%:6% 的日内变动对 ST 票应判涨跌停(mask=False),
    对普通主板票不应。"""
    from stockpool.config import MaskConfig
    from stockpool.panel import compute_tradability_mask

    dates = pd.bdate_range("2025-06-02", periods=4)
    closes = pd.DataFrame({
        "600001": [10.0, 10.6, 10.0, 10.5],   # +6% 在 ST 阈值之上
        "600000": [10.0, 10.6, 10.0, 10.5],   # 同样 +6%,普通票 <9.8% 不 mask
    }, index=dates)
    panel = {
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": pd.DataFrame(1e6, index=dates, columns=closes.columns),
    }
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    ipo = {c: pd.Timestamp("2000-01-01") for c in closes.columns}

    mask = compute_tradability_mask(panel, cfg, ipo_dates=ipo, st_codes={"600001"})
    assert not mask.loc[dates[1], "600001"], "ST 票 +6% 应被 mask"
    assert mask.loc[dates[1], "600000"], "普通票 +6% 不应被 mask"


def test_chinext_st_still_uses_20pct():
    """创业板 ST 涨跌幅仍为 20%(注册制规则),板块判定优先于 ST。"""
    from stockpool.config import MaskConfig
    from stockpool.panel import _limit_threshold_for_config

    cfg = MaskConfig()
    assert _limit_threshold_for_config("300001", cfg, st_codes={"300001"}) == pytest.approx(0.198)
    assert _limit_threshold_for_config("600001", cfg, st_codes={"600001"}) == pytest.approx(0.048)
