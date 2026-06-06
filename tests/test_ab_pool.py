"""Tests for stockpool.ab_pool — AB candidate pool build / load / yaml integration."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stockpool.ab_pool import AbPoolConfig, _apply_hard_filters
from stockpool.config import AppConfig, load_config


def test_ab_pool_config_defaults():
    cfg = AbPoolConfig()
    assert cfg.cache_path == "data/ab_pool.parquet"
    assert cfg.industry_source == "auto"
    assert cfg.min_listing_days == 252
    assert cfg.min_avg_amount_20d == 5.0e7
    assert cfg.per_industry_top_mcap == 2
    assert cfg.per_industry_top_liq == 2
    assert cfg.exclude_st is True
    assert cfg.include_unknown_industry is True


def test_ab_pool_config_extra_forbidden():
    with pytest.raises(Exception):  # pydantic ValidationError
        AbPoolConfig(unknown_field=42)


def test_app_config_has_ab_pool_default(tmp_path: Path):
    yaml_text = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert isinstance(cfg.ab_pool, AbPoolConfig)
    assert cfg.ab_pool.cache_path == "data/ab_pool.parquet"


def _make_candidate_df(rows: list[dict]) -> pd.DataFrame:
    """Build the input DataFrame shape that _apply_hard_filters expects.

    Columns: code / name / industry / circ_mv / avg_amount_20d / ipo_date
    """
    return pd.DataFrame(rows)


def test_hard_filters_drops_st():
    cfg = AbPoolConfig()
    today = date.today()
    df = _make_candidate_df([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "000001", "name": "ST平安", "industry": "银行",
         "circ_mv": 1e11, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["600519"]


def test_hard_filters_drops_new_ipo():
    cfg = AbPoolConfig()
    today = date.today()
    df = _make_candidate_df([
        {"code": "688001", "name": "新股", "industry": "电子",
         "circ_mv": 1e10, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=100)},  # < 252 days
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["600519"]


def test_hard_filters_drops_illiquid():
    cfg = AbPoolConfig(min_avg_amount_20d=5e7)
    today = date.today()
    df = _make_candidate_df([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "300999", "name": "小盘", "industry": "电子",
         "circ_mv": 5e8, "avg_amount_20d": 1e7,  # below 5e7 floor
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["600519"]


def test_hard_filters_drops_nan_circ_mv():
    cfg = AbPoolConfig()
    today = date.today()
    import numpy as np
    df = _make_candidate_df([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "999999", "name": "无快照", "industry": "未知",
         "circ_mv": np.nan, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["600519"]


def test_hard_filters_st_variants():
    """ST detection should catch ST / *ST / 退 (delisting marker)."""
    cfg = AbPoolConfig()
    today = date.today()
    df = _make_candidate_df([
        {"code": "1", "name": "正常股", "industry": "银行",
         "circ_mv": 1e11, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "2", "name": "ST某某", "industry": "银行",
         "circ_mv": 1e10, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "3", "name": "*ST某某", "industry": "银行",
         "circ_mv": 1e10, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "4", "name": "某某退", "industry": "银行",
         "circ_mv": 1e10, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["1"]


from stockpool.ab_pool import _stratified_select


def test_stratified_no_overlap():
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        # 银行: 4 stocks, mcap-rank distinct from liq-rank
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 1},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 2},
        {"code": "B3", "name": "B3", "industry": "银行", "circ_mv": 1, "avg_amount_20d": 9},
        {"code": "B4", "name": "B4", "industry": "银行", "circ_mv": 2, "avg_amount_20d": 8},
    ])
    out = _stratified_select(df, cfg)
    # Expect 4 rows: top-2 mcap = {B1, B2}, top-2 liq = {B3, B4}, no overlap
    assert set(out["code"]) == {"B1", "B2", "B3", "B4"}
    assert dict(zip(out["code"], out["source_tag"])) == {
        "B1": "mcap", "B2": "mcap", "B3": "liq", "B4": "liq",
    }


def test_stratified_full_overlap_3_rows():
    """Top-2 mcap fully overlaps top-2 liq → 1 shared + bucket yields 3 rows."""
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        # 银行: top-2 by mcap = {B1, B2}; top-2 by liq = {B1, B3}
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 1},
        {"code": "B3", "name": "B3", "industry": "银行", "circ_mv": 1, "avg_amount_20d": 8},
    ])
    out = _stratified_select(df, cfg)
    assert set(out["code"]) == {"B1", "B2", "B3"}
    tags = dict(zip(out["code"], out["source_tag"]))
    assert tags["B1"] == "mcap+liq"
    assert tags["B2"] == "mcap"
    assert tags["B3"] == "liq"


def test_stratified_multiple_industries():
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 8},
        {"code": "F1", "name": "F1", "industry": "食品", "circ_mv": 5, "avg_amount_20d": 5},
        {"code": "F2", "name": "F2", "industry": "食品", "circ_mv": 4, "avg_amount_20d": 4},
    ])
    out = _stratified_select(df, cfg)
    assert set(out["code"]) == {"B1", "B2", "F1", "F2"}
    assert set(out[out["industry"] == "银行"]["code"]) == {"B1", "B2"}
    assert set(out[out["industry"] == "食品"]["code"]) == {"F1", "F2"}


def test_stratified_small_bucket_partial_fill():
    """Bucket with only 1 stock contributes 1 row (no error, no warning escalation)."""
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        {"code": "X1", "name": "X1", "industry": "稀有", "circ_mv": 1, "avg_amount_20d": 1},
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 8},
    ])
    out = _stratified_select(df, cfg)
    assert "X1" in set(out["code"])


def test_stratified_unknown_industry_included():
    cfg = AbPoolConfig(include_unknown_industry=True)
    df = pd.DataFrame([
        {"code": "U1", "name": "U1", "industry": "未知", "circ_mv": 5, "avg_amount_20d": 5},
        {"code": "U2", "name": "U2", "industry": "未知", "circ_mv": 4, "avg_amount_20d": 4},
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
    ])
    out = _stratified_select(df, cfg)
    assert {"U1", "U2"}.issubset(set(out["code"]))


def test_stratified_unknown_industry_excluded():
    cfg = AbPoolConfig(include_unknown_industry=False)
    df = pd.DataFrame([
        {"code": "U1", "name": "U1", "industry": "未知", "circ_mv": 5, "avg_amount_20d": 5},
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
    ])
    out = _stratified_select(df, cfg)
    assert "U1" not in set(out["code"])
    assert "B1" in set(out["code"])


def test_stratified_output_columns():
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 8},
    ])
    out = _stratified_select(df, cfg)
    assert set(out.columns) >= {"code", "name", "industry", "circ_mv",
                                "avg_amount_20d", "source_tag"}


from stockpool.ab_pool import _fetch_circ_mv_snapshot, _compute_avg_amount_20d


def test_fetch_circ_mv_snapshot_normalizes(monkeypatch):
    """Mock akshare; verify shape: columns code/name/circ_mv (float, yuan)."""
    fake_df = pd.DataFrame({
        "代码": ["600519", "000001"],
        "名称": ["贵州茅台", "平安银行"],
        "流通市值": [2.1e12, 3.2e11],  # akshare already returns yuan
    })
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.return_value = fake_df
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)

    out = _fetch_circ_mv_snapshot()

    assert list(out.columns) == ["code", "name", "circ_mv"]
    assert list(out["code"]) == ["600519", "000001"]
    assert list(out["name"]) == ["贵州茅台", "平安银行"]
    assert out["circ_mv"].dtype.kind == "f"
    assert out["circ_mv"].iloc[0] == pytest.approx(2.1e12)


def test_fetch_circ_mv_snapshot_propagates_error(monkeypatch):
    def raise_err():
        raise RuntimeError("akshare timeout")
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.side_effect = raise_err
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)
    with pytest.raises(RuntimeError, match="akshare"):
        _fetch_circ_mv_snapshot()


def test_compute_avg_amount_20d_basic(tmp_path: Path):
    """Synthesize per-stock parquet, verify avg_amount = mean(vol*close*100) tail-20."""
    cache_dir = tmp_path
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    # vol*close*100 average of last 20 should be 100 * (100 * 10) = 100000
    df = pd.DataFrame({
        "date": dates,
        "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
        "volume": 100.0,
    })
    df.to_parquet(cache_dir / "600519_daily.parquet")

    out = _compute_avg_amount_20d(["600519"], cache_dir)
    assert list(out["code"]) == ["600519"]
    assert out["avg_amount_20d"].iloc[0] == pytest.approx(100.0 * 10.0 * 100)


def test_compute_avg_amount_20d_missing_file_nan(tmp_path: Path):
    """Missing parquet → NaN, not crash."""
    out = _compute_avg_amount_20d(["600519"], tmp_path)
    assert list(out["code"]) == ["600519"]
    import math
    assert math.isnan(out["avg_amount_20d"].iloc[0])


from stockpool.ab_pool import build_ab_pool, load_ab_pool


def _stub_app_cfg(tmp_path: Path) -> "AppConfig":
    """Build a minimal AppConfig with cache_dir = tmp_path / 'data'."""
    yaml_text = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    cfg.data.cache_dir = str(data_dir)
    cfg.ab_pool.cache_path = str(data_dir / "ab_pool.parquet")
    return cfg


def _seed_universe_and_daily(cfg, codes_industries: list[tuple[str, str, str, float]]):
    """Seed universe.parquet + per-stock parquets + industry_map cache.

    Each tuple = (code, name, industry, daily_amount_yuan)
    """
    data_dir = Path(cfg.data.cache_dir)
    universe = pd.DataFrame([
        {"code": c, "name": n, "market": "sh" if c.startswith("6") else "sz"}
        for c, n, _, _ in codes_industries
    ])
    universe.to_parquet(data_dir / "universe.parquet")

    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    for code, _, _, daily_amt in codes_industries:
        # daily_amt = volume * close * 100  →  set volume=daily_amt/(close*100)
        close = 10.0
        volume = daily_amt / (close * 100)
        df = pd.DataFrame({
            "date": dates, "open": close, "high": close, "low": close,
            "close": close, "volume": volume,
        })
        df.to_parquet(data_dir / f"{code}_daily.parquet")

    industry_df = pd.DataFrame([
        {"code": c, "industry": ind}
        for c, _, ind, _ in codes_industries
    ])
    industry_df.to_parquet(data_dir / "stock_industry_map.parquet")


def test_build_basic(tmp_path, monkeypatch):
    cfg = _stub_app_cfg(tmp_path)
    cfg.ab_pool.min_listing_days = 0  # disable IPO filter for synthetic data
    _seed_universe_and_daily(cfg, [
        ("600001", "Bank1", "银行", 1e9),
        ("600002", "Bank2", "银行", 1e9),
        ("600003", "Bank3", "银行", 1e9),
        ("600004", "Bank4", "银行", 1e9),
        ("600005", "Food1", "食品", 1e9),
        ("600006", "Food2", "食品", 1e9),
    ])
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.return_value = pd.DataFrame({
        "代码": ["600001", "600002", "600003", "600004", "600005", "600006"],
        "名称": ["Bank1", "Bank2", "Bank3", "Bank4", "Food1", "Food2"],
        "流通市值": [9e10, 8e10, 7e10, 6e10, 5e10, 4e10],
    })
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)
    monkeypatch.setattr("stockpool.ab_pool._load_industry_map",
                        lambda *_a, **_k: {"600001": "银行", "600002": "银行",
                                            "600003": "银行", "600004": "银行",
                                            "600005": "食品", "600006": "食品"})
    monkeypatch.setattr("stockpool.ab_pool._load_ipo_dates",
                        lambda *_a, **_k: {})

    out_path = build_ab_pool(cfg, refresh=False)

    assert Path(out_path).exists()
    df = load_ab_pool(out_path)
    assert set(df.columns) >= {"code", "name", "industry", "circ_mv",
                                "avg_amount_20d", "source_tag", "build_date"}
    assert set(df["industry"]) == {"银行", "食品"}


def test_build_idempotent_guard(tmp_path, monkeypatch):
    cfg = _stub_app_cfg(tmp_path)
    cache_path = Path(cfg.ab_pool.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_ab_pool(cfg, refresh=False)
    assert cache_path.read_bytes() == b"existing"


def test_build_refresh_overwrites(tmp_path, monkeypatch):
    cfg = _stub_app_cfg(tmp_path)
    cfg.ab_pool.min_listing_days = 0
    _seed_universe_and_daily(cfg, [
        ("600001", "Bank1", "银行", 1e9),
        ("600002", "Bank2", "银行", 1e9),
    ])
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.return_value = pd.DataFrame({
        "代码": ["600001", "600002"], "名称": ["Bank1", "Bank2"],
        "流通市值": [9e10, 8e10],
    })
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)
    monkeypatch.setattr("stockpool.ab_pool._load_industry_map",
                        lambda *_a, **_k: {"600001": "银行", "600002": "银行"})
    monkeypatch.setattr("stockpool.ab_pool._load_ipo_dates",
                        lambda *_a, **_k: {})

    cache_path = Path(cfg.ab_pool.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"old")
    build_ab_pool(cfg, refresh=True)
    # File should be overwritten with a valid parquet
    df = load_ab_pool(cache_path)
    assert "600001" in set(df["code"])


def test_build_universe_missing(tmp_path):
    cfg = _stub_app_cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="universe.parquet"):
        build_ab_pool(cfg, refresh=False)


def test_build_all_buckets_empty(tmp_path, monkeypatch):
    cfg = _stub_app_cfg(tmp_path)
    _seed_universe_and_daily(cfg, [("600001", "Only", "银行", 1e3)])  # below floor
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.return_value = pd.DataFrame({
        "代码": ["600001"], "名称": ["Only"], "流通市值": [1e10],
    })
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)
    monkeypatch.setattr("stockpool.ab_pool._load_industry_map",
                        lambda *_a, **_k: {"600001": "银行"})
    monkeypatch.setattr("stockpool.ab_pool._load_ipo_dates",
                        lambda *_a, **_k: {})
    with pytest.raises(RuntimeError, match="empty"):
        build_ab_pool(cfg, refresh=False)


# ============================================================================
# Task 6: HTML renderer (render_ab_pool_html)
# ============================================================================
from stockpool.ab_pool_report import render_ab_pool_html


def test_render_html_smoke(tmp_path):
    df = pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2.1e12, "avg_amount_20d": 5e9,
         "source_tag": "mcap+liq", "build_date": "2026-06-06"},
        {"code": "000001", "name": "平安银行", "industry": "银行",
         "circ_mv": 3e11, "avg_amount_20d": 8e8,
         "source_tag": "liq", "build_date": "2026-06-06"},
    ])
    out_path = tmp_path / "ab_pool.html"
    render_ab_pool_html(df, out_path)
    html = out_path.read_text(encoding="utf-8")

    # Inline JSON data
    assert "POOL_DATA" in html
    assert "600519" in html
    assert "贵州茅台" in html
    # Three filter inputs
    assert 'id="filter-industry"' in html
    assert 'id="filter-code"' in html
    assert 'id="filter-name"' in html
    # Build date footer
    assert "2026-06-06" in html
    # Table header
    assert "代码" in html and "流通市值" in html


def test_render_html_empty_df(tmp_path):
    """Empty df should still produce a valid HTML page."""
    df = pd.DataFrame(columns=["code", "name", "industry", "circ_mv",
                                "avg_amount_20d", "source_tag", "build_date"])
    out_path = tmp_path / "ab_pool.html"
    render_ab_pool_html(df, out_path)
    html = out_path.read_text(encoding="utf-8")
    assert "POOL_DATA" in html
    assert "[]" in html  # empty JSON array
