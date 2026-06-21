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


from stockpool.ab_pool import _fetch_circ_mv_snapshot, _compute_avg_amount_20d


def test_fetch_circ_mv_snapshot_computes_from_baostock(tmp_path):
    """Builds fundamentals_profit.parquet + per-code daily caches under tmp;
    verifies circ_mv = latest_liqaShare × latest_close, and that empty-string
    liqaShare values coerce to NaN."""
    cache_dir = tmp_path
    # Profit table with 3 codes:
    #   600519 has two rows; the later pubDate's liqaShare must win
    #   000001 has one row with valid liqaShare
    #   600000 has liqaShare == '' (coerce to NaN -> circ_mv NaN)
    prof = pd.DataFrame({
        "code": ["600519", "600519", "000001", "600000"],
        "pubDate": pd.to_datetime(
            ["2026-04-25", "2026-01-25", "2026-04-25", "2026-04-25"]
        ),
        "liqaShare": ["1000000", "999999", "2000000", ""],
    })
    prof.to_parquet(cache_dir / "fundamentals_profit.parquet")
    # Daily caches: 600519 close=2100, 000001 close=10, 600000 close=5 (irrelevant)
    for code, close in [("600519", 2100.0), ("000001", 10.0), ("600000", 5.0)]:
        pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=3, freq="B"),
            "close": [close - 0.1, close - 0.05, close],
        }).to_parquet(cache_dir / f"{code}_daily.parquet")

    out = _fetch_circ_mv_snapshot(cache_dir)

    assert list(out.columns) == ["code", "circ_mv"]
    by_code = out.set_index("code")["circ_mv"]
    assert by_code["600519"] == pytest.approx(1_000_000 * 2100.0)  # latest pubDate wins
    assert by_code["000001"] == pytest.approx(2_000_000 * 10.0)
    assert pd.isna(by_code["600000"])  # empty-string liqaShare -> NaN circ_mv


def test_fetch_circ_mv_snapshot_missing_profile_raises(tmp_path):
    """fundamentals_profit.parquet absent → clear error pointing to the fix."""
    with pytest.raises(FileNotFoundError, match="fundamentals_profit"):
        _fetch_circ_mv_snapshot(tmp_path)


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
    """Build a minimal AppConfig with cache_dir = tmp_path / 'data'.

    Also resolves ``factors_file`` to an absolute path so the cfg can be
    dumped + reloaded from a different cwd (subprocess CLI tests).
    """
    yaml_text = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    cfg.data.cache_dir = str(data_dir)
    cfg.ab_pool.cache_path = str(data_dir / "ab_pool.parquet")
    # Make factors_file absolute so model_dump → reload from a different cwd
    # still resolves the selection JSON.
    ff = cfg.strategy.ml_factor.factors_file
    if ff:
        cfg.strategy.ml_factor.factors_file = str(
            (Path(__file__).parent.parent / ff).resolve()
        )
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
    fake_snapshot = pd.DataFrame({
        "code": ["600001", "600002", "600003", "600004", "600005", "600006"],
        "circ_mv": [9e10, 8e10, 7e10, 6e10, 5e10, 4e10],
    })
    monkeypatch.setattr(
        "stockpool.ab_pool._fetch_circ_mv_snapshot",
        lambda *_a, **_k: fake_snapshot,
    )
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
    fake_snapshot = pd.DataFrame({
        "code": ["600001", "600002"], "circ_mv": [9e10, 8e10],
    })
    monkeypatch.setattr(
        "stockpool.ab_pool._fetch_circ_mv_snapshot",
        lambda *_a, **_k: fake_snapshot,
    )
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
    fake_snapshot = pd.DataFrame({
        "code": ["600001"], "circ_mv": [1e10],
    })
    monkeypatch.setattr(
        "stockpool.ab_pool._fetch_circ_mv_snapshot",
        lambda *_a, **_k: fake_snapshot,
    )
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


# ============================================================================
# Task 7: CLI subcommand `ab-pool build` / `ab-pool show`
# ============================================================================
import subprocess
import sys


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run `python -m stockpool` with the given args from a tmp cwd."""
    proj_root = Path(__file__).parent.parent
    env = {"PYTHONPATH": str(proj_root / "src")}
    import os
    env.update(os.environ)
    return subprocess.run(
        [sys.executable, "-m", "stockpool", *args],
        cwd=cwd, capture_output=True, text=True, env=env,
    )


def test_cli_ab_pool_build_missing_universe(tmp_path):
    """Build without universe.parquet → exit 1, helpful message."""
    cfg = _stub_app_cfg(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    # Re-dump cfg with updated cache_dir
    import yaml as _yaml
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    res = _run_cli(["ab-pool", "build", "--config", str(cfg_path)], cwd=tmp_path)
    assert res.returncode == 1
    assert "universe.parquet" in (res.stderr + res.stdout)


def test_cli_ab_pool_build_idempotent_guard(tmp_path):
    cfg = _stub_app_cfg(tmp_path)
    cache_path = Path(cfg.ab_pool.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"old")
    cfg_path = tmp_path / "config.yaml"
    import yaml as _yaml
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    res = _run_cli(["ab-pool", "build", "--config", str(cfg_path)], cwd=tmp_path)
    assert res.returncode == 1
    assert "--refresh" in (res.stderr + res.stdout)


def test_cli_ab_pool_show_missing_parquet(tmp_path):
    cfg = _stub_app_cfg(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    import yaml as _yaml
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    res = _run_cli(["ab-pool", "show", "--config", str(cfg_path)], cwd=tmp_path)
    assert res.returncode == 1
    assert "ab-pool build" in (res.stderr + res.stdout)


def test_cli_ab_pool_show_renders(tmp_path, monkeypatch):
    """End-to-end: write a parquet directly, call show, assert HTML created."""
    cfg = _stub_app_cfg(tmp_path)
    cache_path = Path(cfg.ab_pool.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
                        "circ_mv": 2e12, "avg_amount_20d": 5e9,
                        "source_tag": "mcap+liq", "build_date": "2026-06-06"}])
    df.to_parquet(cache_path)

    cfg_path = tmp_path / "config.yaml"
    import yaml as _yaml
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    # Disable browser auto-open via env var (will be read by cmd_ab_pool_show)
    import os
    env = {**os.environ, "STOCKPOOL_NO_BROWSER": "1"}
    proj_root = Path(__file__).parent.parent
    env["PYTHONPATH"] = str(proj_root / "src")
    res = subprocess.run(
        [sys.executable, "-m", "stockpool", "ab-pool", "show",
         "--config", str(cfg_path)],
        cwd=tmp_path, capture_output=True, text=True, env=env,
    )
    assert res.returncode == 0, res.stderr
    # Output: reports/ab_pool.html relative to cwd
    out_html = tmp_path / "reports" / "ab_pool.html"
    assert out_html.exists()
    assert "贵州茅台" in out_html.read_text(encoding="utf-8")


# ============================================================================
# Task 8: ab.yaml use_ab_pool integration
# ============================================================================
from stockpool.ab.config import ABConfig, ArmOverride, load_ab_config, _resolve_stocks


def _write_ab_yaml(tmp_path: Path, use_ab_pool: bool, stocks_filter: list[str] | None = None):
    """Write a tmp config.yaml + ab.yaml. Uses _stub_app_cfg-style normalization
    so factors_file and ab_pool.cache_path are absolute (no cwd dependency)."""
    import yaml as _yaml
    proj_root = Path(__file__).parent.parent
    base_cfg_text = (proj_root / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_cfg_text, encoding="utf-8")
    # Re-load + re-dump with absolute factors_file + ab_pool.cache_path so
    # load_ab_config → load_config works regardless of cwd.
    cfg = load_config(tmp_path / "config.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    cfg.data.cache_dir = str(data_dir)
    cfg.ab_pool.cache_path = str(data_dir / "ab_pool.parquet")
    ff = cfg.strategy.ml_factor.factors_file
    if ff and not Path(ff).is_absolute():
        cfg.strategy.ml_factor.factors_file = str((proj_root / ff).resolve())
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(tmp_path / "config.yaml", "w", encoding="utf-8"),
                    allow_unicode=True)

    ab_text = f"""
base_config: config.yaml
use_ab_pool: {str(use_ab_pool).lower()}
{("stocks_filter: " + repr(stocks_filter)) if stocks_filter else ""}
arms:
  baseline:
    strategy:
      name: composite_verdict
    backtest:
      equity_curve_holding_days: [10]
  challenger:
    strategy:
      name: composite_verdict
    backtest:
      equity_curve_holding_days: [10]
"""
    ab_path = tmp_path / "ab.yaml"
    ab_path.write_text(ab_text, encoding="utf-8")
    return ab_path


def test_ab_config_use_ab_pool_default_false(tmp_path):
    ab_path = _write_ab_yaml(tmp_path, use_ab_pool=False)
    ab_cfg = load_ab_config(ab_path)
    assert ab_cfg.use_ab_pool is False


def test_ab_config_use_ab_pool_true_field(tmp_path):
    ab_path = _write_ab_yaml(tmp_path, use_ab_pool=True)
    # Seed an ab_pool.parquet so load_ab_config doesn't fail on membership check.
    # _write_ab_yaml normalized cache_path to tmp_path/data/ab_pool.parquet.
    pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 5e9,
         "source_tag": "mcap+liq", "build_date": "2026-06-06"},
    ]).to_parquet(tmp_path / "data" / "ab_pool.parquet")
    ab_cfg = load_ab_config(ab_path)
    assert ab_cfg.use_ab_pool is True


def test_ab_config_use_ab_pool_missing_parquet_raises(tmp_path):
    ab_path = _write_ab_yaml(tmp_path, use_ab_pool=True)
    # No parquet exists at tmp_path/data/ab_pool.parquet
    with pytest.raises(Exception, match="ab_pool"):
        load_ab_config(ab_path)


def test_resolve_stocks_use_ab_pool_replaces(tmp_path):
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    base_cfg.data.cache_dir = str(data_dir)
    base_cfg.ab_pool.cache_path = str(data_dir / "ab_pool.parquet")

    pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 5e9,
         "source_tag": "mcap+liq", "build_date": "2026-06-06"},
        {"code": "000001", "name": "平安银行", "industry": "银行",
         "circ_mv": 3e11, "avg_amount_20d": 8e8,
         "source_tag": "liq", "build_date": "2026-06-06"},
    ]).to_parquet(base_cfg.ab_pool.cache_path)

    ab_cfg = ABConfig(
        base_config="config.yaml", use_ab_pool=True, stocks_filter=[],
        arms={
            "a": ArmOverride.model_validate({
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10]},
            }),
            "b": ArmOverride.model_validate({
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10]},
            }),
        },
    )
    stocks = _resolve_stocks(ab_cfg, base_cfg)
    assert [s.code for s in stocks] == ["600519", "000001"]
    assert [s.sector for s in stocks] == ["食品饮料", "银行"]


def test_resolve_stocks_filter_intersect_with_ab_pool(tmp_path):
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    base_cfg.data.cache_dir = str(data_dir)
    base_cfg.ab_pool.cache_path = str(data_dir / "ab_pool.parquet")
    pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 5e9,
         "source_tag": "mcap+liq", "build_date": "2026-06-06"},
        {"code": "000001", "name": "平安银行", "industry": "银行",
         "circ_mv": 3e11, "avg_amount_20d": 8e8,
         "source_tag": "liq", "build_date": "2026-06-06"},
    ]).to_parquet(base_cfg.ab_pool.cache_path)
    ab_cfg = ABConfig(
        base_config="config.yaml", use_ab_pool=True,
        stocks_filter=["600519"],
        arms={
            "a": ArmOverride.model_validate({
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10]},
            }),
            "b": ArmOverride.model_validate({
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10]},
            }),
        },
    )
    stocks = _resolve_stocks(ab_cfg, base_cfg)
    assert [s.code for s in stocks] == ["600519"]


# ============================================================================
# Task 9: portfolio_ab.yaml use_ab_pool integration
# ============================================================================
from stockpool.portfolio_ab.config import (
    PortfolioABConfig, PortfolioArmOverride, load_portfolio_ab_config,
    build_effective_cfg as portfolio_build_effective_cfg,
)


def _seed_ab_pool(tmp_path: Path, codes: list[str]):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame([
        {"code": c, "name": f"N{c}", "industry": "test",
         "circ_mv": 1e10, "avg_amount_20d": 1e8,
         "source_tag": "mcap", "build_date": "2026-06-06"}
        for c in codes
    ]).to_parquet(data_dir / "ab_pool.parquet")
    return str(data_dir / "ab_pool.parquet")


def test_portfolio_ab_use_ab_pool_default_false():
    cfg = PortfolioABConfig(
        base_config="config.yaml",
        arms={"a": PortfolioArmOverride(), "b": PortfolioArmOverride()},
    )
    assert cfg.use_ab_pool is False


def test_portfolio_ab_injects_universe_codes(tmp_path):
    """use_ab_pool=true → both arms' effective_cfg.portfolio_backtest.universe_codes
    == parquet codes; training_universe unchanged by ab_pool injection."""
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    base_cfg.ab_pool.cache_path = _seed_ab_pool(tmp_path, ["600001", "600002"])

    arm = PortfolioArmOverride(
        strategy={"name": "composite_verdict"},
        portfolio_backtest={"enabled": True},
    )
    eff_with_pool = portfolio_build_effective_cfg(
        base_cfg, arm, use_ab_pool=True,
    )
    eff_without_pool = portfolio_build_effective_cfg(
        base_cfg, arm, use_ab_pool=False,
    )
    assert eff_with_pool.portfolio_backtest.universe_codes == ["600001", "600002"]
    # Training pool field on ml_factor must be untouched by ab_pool injection
    # (the wholesale strategy override decides ml_factor; ab_pool injection
    # only touches portfolio_backtest.universe_codes).
    assert (
        eff_with_pool.strategy.ml_factor.training_universe
        == eff_without_pool.strategy.ml_factor.training_universe
    )


def test_portfolio_ab_per_arm_override_wins(tmp_path):
    """Per-arm explicit universe_codes wins over use_ab_pool."""
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    base_cfg.ab_pool.cache_path = _seed_ab_pool(tmp_path, ["600001", "600002"])

    arm = PortfolioArmOverride(
        strategy={"name": "composite_verdict"},
        portfolio_backtest={"enabled": True, "universe_codes": ["999999"]},
    )
    eff = portfolio_build_effective_cfg(base_cfg, arm, use_ab_pool=True)
    assert eff.portfolio_backtest.universe_codes == ["999999"]


def test_portfolio_ab_use_ab_pool_missing_parquet_raises(tmp_path):
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    base_cfg.ab_pool.cache_path = str(tmp_path / "data" / "ab_pool.parquet")
    arm = PortfolioArmOverride(strategy={"name": "composite_verdict"})
    with pytest.raises(FileNotFoundError, match="ab_pool"):
        portfolio_build_effective_cfg(base_cfg, arm, use_ab_pool=True)
