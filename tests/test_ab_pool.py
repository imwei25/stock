"""Tests for stockpool.ab_pool — AB candidate pool build / load / yaml integration."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

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
