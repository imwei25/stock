"""基本面因子层修复的行为测试 (P1-1 / P2-16 / P2-17 / P2-18 / P2-26 / P3-13).

覆盖:
  - 字段修复:roa(dupont 杜邦恒等式)、netprofit_yoy(growth.YOYNI,替代 revenue_yoy)、
    pe(close/epsTTM)、pb(pe_ttm × roe_ttm)
  - _ytd_ratio_to_ttm helper(YTD 累计比率 → 单季差分 → TTM)
  - _pit_align:缺字段 fail-loud / 同 pubDate 取最新 statDate / ffill limit / 覆盖不足告警
  - fundamentals_loader:codes 覆盖率增量补拉 / set_force_refresh
  - cli:--refresh-fundamentals 透传

全部用合成数据,不依赖网络。
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.fundamentals as fund
from stockpool import fundamentals_loader as fl
from stockpool.factors import make_factor, get_spec


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def panel():
    """120 个工作日 panel,2 只票,2024-01-01 起。"""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    codes = ["000001", "600000"]
    close = pd.DataFrame(
        100.0 + np.arange(120).reshape(-1, 1).repeat(2, axis=1) * 0.1,
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(1e6, index=dates, columns=codes)
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


@pytest.fixture(autouse=True)
def _reset_force_refresh():
    """每个测试结束后清掉模块级 force flag,避免串扰。"""
    yield
    fl.set_force_refresh(False)


def _mock_loader(monkeypatch, tables: dict):
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: tables.get(table, pd.DataFrame()),
    )


# ---------------------------------------------------------------- P1-1: roa

def test_roa_uses_dupont_identity(monkeypatch, panel):
    """roa = dupontROE / dupontAssetStoEquity(dupont 表无 dupontROA 字段)。"""
    dupont = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"),
         "dupontROE": 0.12, "dupontAssetStoEquity": 2.0},
    ])
    _mock_loader(monkeypatch, {"dupont": dupont})

    out = make_factor("roa").compute(panel)
    # pubDate 之后:roa = 0.12 / 2.0 = 0.06
    assert out["000001"].iloc[0] == pytest.approx(0.06)
    assert out["600000"].isna().all()


def test_roa_nonpositive_multiplier_is_nan(monkeypatch, panel):
    dupont = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"),
         "dupontROE": 0.12, "dupontAssetStoEquity": -1.0},
    ])
    _mock_loader(monkeypatch, {"dupont": dupont})
    out = make_factor("roa").compute(panel)
    assert out["000001"].isna().all()


# ------------------------------------------------- P1-1: netprofit_yoy 改名

def test_netprofit_yoy_registered_uses_yoyni(monkeypatch, panel):
    growth = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "YOYNI": 0.25},
    ])
    _mock_loader(monkeypatch, {"growth": growth})

    spec = get_spec("netprofit_yoy")
    assert spec is not None

    out = make_factor("netprofit_yoy").compute(panel)
    assert out["000001"].iloc[0] == pytest.approx(0.25)


def test_revenue_yoy_no_longer_registered():
    """revenue_yoy 用的 growth.YOYIncome 不存在 → 因子改名 netprofit_yoy,旧名注销。"""
    with pytest.raises(Exception):
        make_factor("revenue_yoy")


# ---------------------------------------------------------------- P1-1: pe

def test_pe_uses_epsttm(monkeypatch, panel):
    profit = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "epsTTM": 10.0},
    ])
    _mock_loader(monkeypatch, {"profit": profit})

    out = make_factor("pe").compute(panel)
    d0 = panel["close"].index[0]
    assert out.loc[d0, "000001"] == pytest.approx(panel["close"].loc[d0, "000001"] / 10.0)


def test_pe_nonpositive_epsttm_is_nan(monkeypatch, panel):
    profit = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "epsTTM": -2.0},
        {"code": "600000", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "epsTTM": 0.0},
    ])
    _mock_loader(monkeypatch, {"profit": profit})
    out = make_factor("pe").compute(panel)
    assert out["000001"].isna().all()
    assert out["600000"].isna().all()


# ------------------------------------------------- _ytd_ratio_to_ttm helper

def _make_ytd_8q():
    """8 个季度 YTD 累计 roeAvg:y1=[.02,.05,.09,.12], y2=[.03,.07,.12,.16]。

    单季:y1=[.02,.03,.04,.03], y2=[.03,.04,.05,.04]。
    """
    rows = []
    stat_dates = ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
                  "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
    ytd = [0.02, 0.05, 0.09, 0.12, 0.03, 0.07, 0.12, 0.16]
    for sd, v in zip(stat_dates, ytd):
        rows.append({
            "code": "000001",
            "statDate": pd.Timestamp(sd),
            "pubDate": pd.Timestamp(sd) + pd.Timedelta(days=30),
            "roeAvg": v,
        })
    return pd.DataFrame(rows)


def test_ytd_ratio_to_ttm_synthetic_8_quarters():
    out = fund._ytd_ratio_to_ttm(_make_ytd_8q(), "roeAvg")
    out = out.set_index("statDate")["roeAvg_ttm"]

    # 前 3 季不足 4 个单季 → NaN
    assert np.isnan(out.loc[pd.Timestamp("2023-09-30")])
    # 2023Q4: .02+.03+.04+.03 = .12(= 年度 YTD 本身)
    assert out.loc[pd.Timestamp("2023-12-31")] == pytest.approx(0.12)
    # 2024Q1: .03+.04+.03+.03 = .13
    assert out.loc[pd.Timestamp("2024-03-31")] == pytest.approx(0.13)
    # 2024Q2: .04+.03+.03+.04 = .14
    assert out.loc[pd.Timestamp("2024-06-30")] == pytest.approx(0.14)
    # 2024Q4: .03+.04+.05+.04 = .16
    assert out.loc[pd.Timestamp("2024-12-31")] == pytest.approx(0.16)


def test_ytd_ratio_to_ttm_missing_quarter_gives_nan():
    """中间缺一季 → 受影响窗口 TTM 为 NaN(不会跨期错配差分)。"""
    df = _make_ytd_8q()
    df = df[df["statDate"] != pd.Timestamp("2024-06-30")].reset_index(drop=True)
    out = fund._ytd_ratio_to_ttm(df, "roeAvg")
    out = out.set_index("statDate")["roeAvg_ttm"]
    # 2024Q3 的单季依赖 Q2 YTD;Q2 缺 → Q3 单季 NaN → 含 Q3 的 TTM 全 NaN
    assert np.isnan(out.loc[pd.Timestamp("2024-09-30")])
    assert np.isnan(out.loc[pd.Timestamp("2024-12-31")])
    # 2023Q4 不受影响
    assert out.loc[pd.Timestamp("2023-12-31")] == pytest.approx(0.12)


# ---------------------------------------------------------------- P1-1: pb

def test_pb_equals_pe_times_roe_ttm(monkeypatch, panel):
    profit = _make_ytd_8q()
    profit["epsTTM"] = 10.0
    _mock_loader(monkeypatch, {"profit": profit})

    out = make_factor("pb").compute(panel)
    # 2024-03-31 报告 pubDate = 2024-04-30;在 2024-05 的首个交易日:
    # roe_ttm = 0.13, pe = close/10 → pb = close/10 * 0.13
    target = pd.Timestamp("2024-05-02")
    close = panel["close"].loc[target, "000001"]
    assert out.loc[target, "000001"] == pytest.approx(close / 10.0 * 0.13)


def test_pb_nonpositive_roe_ttm_is_nan(monkeypatch, panel):
    profit = _make_ytd_8q()
    # 全部 YTD 取负 → roe_ttm < 0
    profit["roeAvg"] = -profit["roeAvg"]
    profit["epsTTM"] = 10.0
    _mock_loader(monkeypatch, {"profit": profit})
    out = make_factor("pb").compute(panel)
    assert out["000001"].isna().all()


# ----------------------------------------------- _pit_align: fail-loud 等

def test_pit_align_missing_field_raises_keyerror(panel):
    raw = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "roeAvg": 0.1},
    ])
    with pytest.raises(KeyError) as ei:
        fund._pit_align(raw, "roaAvg", panel["close"], table="profit")
    msg = str(ei.value)
    assert "profit" in msg and "roaAvg" in msg and "roeAvg" in msg


def test_pit_align_empty_raw_still_returns_nan_panel(panel):
    """整表拉取失败(empty)→ 维持优雅降级,返回全 NaN panel 而非 raise。"""
    out = fund._pit_align(pd.DataFrame(), "roeAvg", panel["close"], table="profit")
    assert out.isna().all().all()
    assert out.shape == panel["close"].shape


def test_pit_align_same_pubdate_keeps_latest_statdate(panel):
    """年报 + 一季报同日披露 → 保留最新报告期(一季报)的值。(P2-18)"""
    raw = pd.DataFrame([
        # 一季报在前(行序),报告期更晚
        {"code": "000001", "statDate": pd.Timestamp("2024-03-31"),
         "pubDate": pd.Timestamp("2024-04-25"), "roeAvg": 0.03},
        # 年报行在后,报告期更早 —— 旧实现 keep="last" 会错误留下它
        {"code": "000001", "statDate": pd.Timestamp("2023-12-31"),
         "pubDate": pd.Timestamp("2024-04-25"), "roeAvg": 0.12},
    ])
    out = fund._pit_align(raw, "roeAvg", panel["close"], table="profit")
    assert out["000001"].iloc[-1] == pytest.approx(0.03)


def test_pit_align_ffill_limit_250_trading_days():
    """停止披露后最多沿用 250 个交易日,之后 NaN。(P3-13)"""
    dates = pd.date_range("2023-01-02", periods=300, freq="B")
    close = pd.DataFrame(100.0, index=dates, columns=["000001"])
    raw = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2022-09-30"),
         "pubDate": pd.Timestamp("2023-01-02"), "roeAvg": 0.1},
    ])
    out = fund._pit_align(raw, "roeAvg", close, table="profit")
    s = out["000001"]
    assert s.iloc[0] == pytest.approx(0.1)        # pubDate 当天(精确命中)
    assert s.iloc[250] == pytest.approx(0.1)      # 之后 250 个填充位仍有值
    assert np.isnan(s.iloc[251])                  # 超限 → 视为停止披露


def test_pit_align_warns_when_panel_predates_coverage(caplog):
    """panel 起始早于多数 code 的最早 pubDate → log.warning。(P2-17)"""
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    close = pd.DataFrame(100.0, index=dates, columns=["000001", "600000"])
    raw = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "roeAvg": 0.1},
        {"code": "600000", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-30"), "roeAvg": 0.2},
    ])
    with caplog.at_level(logging.WARNING, logger="stockpool.factors.fundamentals"):
        fund._pit_align(raw, "roeAvg", close, table="profit")
    assert any("基本面" in r.message or "fundament" in r.message.lower()
               for r in caplog.records)


def test_pit_align_no_warning_when_coverage_ok(panel, caplog):
    raw = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "roeAvg": 0.1},
    ])
    with caplog.at_level(logging.WARNING, logger="stockpool.factors.fundamentals"):
        fund._pit_align(raw, "roeAvg", panel["close"], table="profit")
    assert not caplog.records


# ----------------------------------------- loader: codes 覆盖率 (P2-16)

def _write_cache(tmp_path, table, codes):
    rows = [{"code": c, "pubDate": pd.Timestamp("2024-04-25"),
             "statDate": pd.Timestamp("2024-03-31"), "roeAvg": 0.1}
            for c in codes]
    df = pd.DataFrame(rows)
    path = tmp_path / f"fundamentals_{table}.parquet"
    df.to_parquet(path, index=False)
    return path


def test_coverage_backfill_when_missing_over_30pct(tmp_path, monkeypatch):
    _write_cache(tmp_path, "profit", ["000001", "000002"])
    fetched: list = []

    def fake_fetch(table, codes):
        fetched.append((table, sorted(codes)))
        return pd.DataFrame([
            {"code": c, "pubDate": pd.Timestamp("2024-04-26"),
             "statDate": pd.Timestamp("2024-03-31"), "roeAvg": 0.2}
            for c in codes
        ])

    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    req = ["000001", "000002", "600000", "600001", "600002"]  # 3/5 缺失 = 60%
    out = fl.load_or_build_fundamentals("profit", codes=req, cache_dir=tmp_path)

    assert fetched == [("profit", ["600000", "600001", "600002"])]
    assert set(out["code"]) == set(req)
    # 合并结果写回缓存
    cached = pd.read_parquet(tmp_path / "fundamentals_profit.parquet")
    assert set(cached["code"]) == set(req)


def test_coverage_no_backfill_when_missing_under_30pct(tmp_path, monkeypatch):
    _write_cache(tmp_path, "profit", [f"00000{i}" for i in range(8)])

    def fake_fetch(table, codes):
        raise AssertionError("should not fetch when coverage is sufficient")

    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    req = [f"00000{i}" for i in range(8)] + ["600000", "600001"]  # 2/10 = 20%
    out = fl.load_or_build_fundamentals("profit", codes=req, cache_dir=tmp_path)
    assert set(out["code"]) == {f"00000{i}" for i in range(8)}


def test_coverage_backfill_failure_falls_back_to_cache(tmp_path, monkeypatch):
    _write_cache(tmp_path, "profit", ["000001"])

    def fake_fetch(table, codes):
        raise RuntimeError("network down")

    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)
    out = fl.load_or_build_fundamentals(
        "profit", codes=["000001", "600000"], cache_dir=tmp_path)
    assert set(out["code"]) == {"000001"}


# ------------------------------------- loader: set_force_refresh (P2-26)

def test_set_force_refresh_bypasses_fresh_cache(tmp_path, monkeypatch):
    _write_cache(tmp_path, "profit", ["000001"])  # 刚写入,绝对新鲜
    called: list = []

    def fake_fetch(table, codes):
        called.append(table)
        return pd.DataFrame([
            {"code": "000001", "pubDate": pd.Timestamp("2024-04-26"),
             "statDate": pd.Timestamp("2024-03-31"), "roeAvg": 0.3}])

    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    fl.set_force_refresh(True)
    out = fl.load_or_build_fundamentals("profit", cache_dir=tmp_path)
    assert called == ["profit"]
    assert out["roeAvg"].iloc[0] == pytest.approx(0.3)


def test_no_force_refresh_uses_fresh_cache(tmp_path, monkeypatch):
    _write_cache(tmp_path, "profit", ["000001"])

    def fake_fetch(table, codes):
        raise AssertionError("fresh cache should be used")

    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)
    out = fl.load_or_build_fundamentals("profit", cache_dir=tmp_path)
    assert out["roeAvg"].iloc[0] == pytest.approx(0.1)


# ------------------------------------------- cli 透传 (P2-26)

def test_cli_wires_refresh_fundamentals_flag():
    from stockpool import cli
    ns = argparse.Namespace(refresh_fundamentals=True)
    cli._wire_refresh_fundamentals(ns)
    try:
        assert fl._FORCE_REFRESH is True
    finally:
        fl.set_force_refresh(False)
    cli._wire_refresh_fundamentals(argparse.Namespace(refresh_fundamentals=False))
    assert fl._FORCE_REFRESH is False


def test_cli_handlers_call_wire(monkeypatch):
    """run/backtest/portfolio-backtest 三个 handler 源码里都接了透传。"""
    import inspect
    from stockpool import cli
    for fn in (cli.cmd_run, cli.cmd_backtest, cli.cmd_portfolio_backtest):
        src = inspect.getsource(fn)
        assert "_wire_refresh_fundamentals" in src, fn.__name__
