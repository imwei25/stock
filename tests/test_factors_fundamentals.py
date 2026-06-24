"""Smoke tests for fundamental factors with strict PIT alignment.

关键 case: pubDate vs statDate 的看见时点 — 在 pubDate 之前必须 NaN。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.fundamentals as _fund  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    """80 个工作日 panel,2 只票。"""
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["000001", "600000"]
    close = pd.DataFrame(
        100.0 + np.arange(80).reshape(-1, 1).repeat(2, axis=1) * 0.1,
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(1e6, index=dates, columns=codes)
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


@pytest.fixture
def mock_profit_df():
    """3 季利润数据 mock。"""
    return pd.DataFrame([
        # 2023 Q3: statDate=2023-09-30, pubDate=2023-10-28 (~1 月延迟)
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"),
         "roeAvg": 0.10, "netProfit": 1e9, "npMargin": 0.20,
         "gpMargin": 0.30, "totalRevenue": 5e9, "roaAvg": 0.05},
        # 2023 Q4: statDate=2023-12-31, pubDate=2024-03-15
        {"code": "000001", "statDate": pd.Timestamp("2023-12-31"),
         "pubDate": pd.Timestamp("2024-03-15"),
         "roeAvg": 0.11, "netProfit": 1.1e9, "npMargin": 0.21,
         "gpMargin": 0.31, "totalRevenue": 5.2e9, "roaAvg": 0.05},
        # 2024 Q1: statDate=2024-03-31, pubDate=2024-04-29
        {"code": "000001", "statDate": pd.Timestamp("2024-03-31"),
         "pubDate": pd.Timestamp("2024-04-29"),
         "roeAvg": 0.12, "netProfit": 1.2e9, "npMargin": 0.22,
         "gpMargin": 0.32, "totalRevenue": 5.4e9, "roaAvg": 0.06},
    ])


def test_roe_factor_uses_pubdate_not_statdate(monkeypatch, panel, mock_profit_df):
    """关键 PIT 测试: 2024-04-01 (statDate 之后但 pubDate 之前) ROE 仍是上季的值。"""
    from stockpool import fundamentals_loader as fl
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: mock_profit_df if table == "profit" else pd.DataFrame()
    )

    f = make_factor("roe")
    out = f.compute(panel)

    # 2024-03-31 是周日,前一个工作日 2024-03-29:此时只能看到 2023 Q4 (pubDate=2024-03-15)
    target = pd.Timestamp("2024-03-29")
    if target in out.index:
        assert out.loc[target, "000001"] == pytest.approx(0.11)

    # 2024-04-01 (statDate 已过) 仍然只能看到 2023 Q4
    target2 = pd.Timestamp("2024-04-01")
    if target2 in out.index:
        assert out.loc[target2, "000001"] == pytest.approx(0.11)

    # 2024-04-30 (pubDate 已过) 才能看到 2024 Q1
    target3 = pd.Timestamp("2024-04-30")
    if target3 in out.index:
        assert out.loc[target3, "000001"] == pytest.approx(0.12)


def test_roe_pre_first_pubdate_is_nan(monkeypatch, panel, mock_profit_df):
    """首份财报 pubDate (2023-10-28) 之前所有日为 NaN。"""
    from stockpool import fundamentals_loader as fl
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: mock_profit_df if table == "profit" else pd.DataFrame()
    )

    f = make_factor("roe")
    out = f.compute(panel)
    # 2024-01-01 (panel start) 在 pubDate 2023-10-28 之后,所以有值
    assert out["000001"].iloc[0] == pytest.approx(0.10)


def test_roe_missing_code_in_fundamentals_is_nan(monkeypatch, panel, mock_profit_df):
    """panel 里有但 fundamentals 没有的 code 全列 NaN。"""
    from stockpool import fundamentals_loader as fl
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: mock_profit_df if table == "profit" else pd.DataFrame()
    )

    f = make_factor("roe")
    out = f.compute(panel)
    assert out["600000"].isna().all()  # 600000 不在 mock 里


def test_pe_negative_earnings_returns_nan(monkeypatch, panel):
    """亏损 (net_income_ttm <= 0) → PE = NaN。"""
    from stockpool import fundamentals_loader as fl
    # 构造净利润全为负的 mock(4 季,符合 TTM min_periods=4)
    bad = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp(f"2023-{q*3:02d}-30"),
         "pubDate": pd.Timestamp(f"2023-{q*3:02d}-30") + pd.Timedelta(days=30),
         "netProfit": -1e8 * q}
        for q in (1, 2, 3, 4)
    ])
    bal = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-12-31"),
         "pubDate": pd.Timestamp("2024-01-30"), "totalShare": 1e10}
    ])
    def fake(table, **kw):
        return {"profit": bad, "balance": bal}.get(table, pd.DataFrame())
    monkeypatch.setattr(fl, "load_or_build_fundamentals", fake)

    f = make_factor("pe")
    out = f.compute(panel)
    assert out["000001"].dropna().empty or (out["000001"] <= 0).any() is False


def test_specs_registered():
    for name in ("roe", "roa", "pe", "pb", "gross_margin", "net_margin",
                 "revenue_yoy"):
        spec = get_spec(name)
        assert spec is not None


def test_market_cap_factor_registered_and_computes(monkeypatch, tmp_path):
    """market_cap factor: close × totalShare PIT-aligned by pubDate."""
    from stockpool.factors.registry import make_factor
    import pandas as pd
    import numpy as np

    fake_balance = pd.DataFrame({
        "code": ["600000"],
        "pubDate": pd.to_datetime(["2024-12-15"]),
        "statDate": pd.to_datetime(["2024-09-30"]),
        "totalShare": [6e8],
    })
    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: fake_balance,
    )

    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    close = pd.DataFrame({"600000": [10.0, 11.0, 12.0]}, index=dates)
    panel = {"close": close, "open": close, "high": close, "low": close, "volume": close}

    factor = make_factor("market_cap")
    out = factor.compute(panel)
    expected = close["600000"] * 6e8
    np.testing.assert_allclose(out["600000"].values, expected.values)


def test_log_market_cap_factor_registered_and_computes(monkeypatch):
    """log_market_cap = log(close × totalShare), NaN where mcap ≤ 0 / missing."""
    from stockpool.factors.registry import make_factor
    import pandas as pd
    import numpy as np

    fake_balance = pd.DataFrame({
        "code": ["600000"],
        "pubDate": pd.to_datetime(["2024-12-15"]),
        "statDate": pd.to_datetime(["2024-09-30"]),
        "totalShare": [6e8],
    })
    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: fake_balance,
    )

    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    close = pd.DataFrame({"600000": [10.0, 11.0, 12.0]}, index=dates)
    panel = {"close": close, "open": close, "high": close, "low": close, "volume": close}

    factor = make_factor("log_market_cap")
    out = factor.compute(panel)
    expected = np.log(close["600000"] * 6e8)
    np.testing.assert_allclose(out["600000"].values, expected.values)


def test_pe_factor_has_contains_mcap_tag():
    """PE registration tag tuple must include 'contains_mcap'."""
    from stockpool.factors.registry import list_specs
    pe_spec = next(s for s in list_specs() if s.base_name == "pe")
    assert "contains_mcap" in pe_spec.types


def test_pb_factor_has_contains_mcap_tag():
    from stockpool.factors.registry import list_specs
    pb_spec = next(s for s in list_specs() if s.base_name == "pb")
    assert "contains_mcap" in pb_spec.types


def test_market_cap_factor_has_size_tag():
    from stockpool.factors.registry import list_specs
    spec = next(s for s in list_specs() if s.base_name == "market_cap")
    assert "size" in spec.types


# ── ported from composite-backtest 2026-06-24: extended Barra fundamentals ──


def test_pit_align_sparse_multicode_forward_fills_each_code():
    """回归:两 code 在**不同** pubDate 披露,各自的值必须前向填充到全部后续交易日。

    旧实现用 ``reindex(method='ffill')`` 在稀疏多 code pivot 上会塌成"只有自己
    pubDate 当天有值"(全市场覆盖率 ~1%)。这里 A 报 01-15、B 报 02-15,断言
    在 B 的披露日 A 仍保留自己的值(而非被 B 行的 NaN 覆盖)。
    """
    from stockpool.factors.fundamentals import _pit_align

    raw = pd.DataFrame([
        {"code": "A", "statDate": pd.Timestamp("2023-12-31"),
         "pubDate": pd.Timestamp("2024-01-15"), "roeAvg": 0.10},
        {"code": "B", "statDate": pd.Timestamp("2023-12-31"),
         "pubDate": pd.Timestamp("2024-02-15"), "roeAvg": 0.20},
    ])
    dates = pd.bdate_range("2024-01-01", "2024-04-30")
    close = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    out = _pit_align(raw, "roeAvg", close, table="profit")

    # A 在 02-20(B 披露之后)仍是 0.10,没被 B 的 pubDate 行覆盖成 NaN
    assert out.loc[pd.Timestamp("2024-02-20"), "A"] == pytest.approx(0.10)
    assert out.loc[pd.Timestamp("2024-02-20"), "B"] == pytest.approx(0.20)
    # A 披露日前 NaN;披露后稠密填充(PIT 正确)
    assert pd.isna(out.loc[pd.Timestamp("2024-01-10"), "A"])
    a_after = out.loc[pd.Timestamp("2024-01-15"):, "A"]
    assert a_after.notna().mean() > 0.95  # 几乎全填充,非 ~1%


def test_pit_align_missing_field_fails_loud():
    """表非空但缺请求字段 → raise KeyError(防字段名拼错被静默吞成全 NaN)。"""
    from stockpool.factors.fundamentals import _pit_align

    raw = pd.DataFrame([
        {"code": "A", "statDate": pd.Timestamp("2023-12-31"),
         "pubDate": pd.Timestamp("2024-01-15"), "roeAvg": 0.10},
    ])
    close = pd.DataFrame(1.0, index=pd.bdate_range("2024-01-01", periods=5),
                         columns=["A"])
    with pytest.raises(KeyError):
        _pit_align(raw, "no_such_field", close, table="profit")


def test_ep_uses_epsttm_over_close(monkeypatch, panel):
    """EP = epsTTM/close;亏损(eps<0)时保留负值(不像 PE 返回 NaN)。"""
    from stockpool import fundamentals_loader as fl
    prof = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "epsTTM": 2.0},
        {"code": "600000", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"), "epsTTM": -1.0},  # 亏损
    ])
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: prof if table == "profit" else pd.DataFrame())
    out = make_factor("ep").compute(panel)
    # close ≈ 100;EP(盈利) > 0,EP(亏损) < 0
    assert (out["000001"].dropna() > 0).all()
    assert (out["600000"].dropna() < 0).all()


def test_scalar_fundamentals_pit(monkeypatch, panel):
    """balance / growth / cash_flow 标量因子按 pubDate PIT 对齐。"""
    from stockpool import fundamentals_loader as fl

    def _mk(field, val):
        return pd.DataFrame([
            {"code": "000001", "statDate": pd.Timestamp("2023-12-31"),
             "pubDate": pd.Timestamp("2024-03-15"), field: val},
        ])

    tables = {
        "balance": _mk("liabilityToAsset", 0.45),
        "growth": _mk("YOYAsset", 0.12).assign(YOYEPSBasic=0.08, YOYNI=0.30),
        "cash_flow": _mk("CFOToNP", 1.3),
    }
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: tables.get(table, pd.DataFrame()))

    checks = {
        "debt_to_asset": 0.45, "asset_yoy": 0.12,
        "eps_yoy": 0.08, "cfo_to_np": 1.3, "netprofit_yoy": 0.30,
    }
    for name, expected in checks.items():
        out = make_factor(name).compute(panel)
        # pubDate 2024-03-15 之前 NaN,之后取值
        before = pd.Timestamp("2024-03-14")
        after = pd.Timestamp("2024-03-15")
        if before in out.index:
            assert pd.isna(out.loc[before, "000001"])
        if after in out.index:
            assert out.loc[after, "000001"] == pytest.approx(expected)


def test_new_fundamental_specs_registered():
    for name in ("netprofit_yoy", "ep", "bp", "debt_to_asset",
                 "asset_yoy", "eps_yoy", "cfo_to_np"):
        spec = get_spec(name)
        assert spec is not None
