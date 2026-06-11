"""Smoke tests for `python -m stockpool factors analyze` and `pick-by-ic`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_cache(tmp_path):
    """Seed the cache with 3 synthetic stocks so factor_panel has columns to rank across."""
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    rng = np.random.default_rng(7)
    n = 200
    for code in ("605589", "603986", "000528"):
        returns = rng.normal(0.0005, 0.02, n)
        close = 100.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "open": close * 0.998,
            "high": close * 1.005,
            "low":  close * 0.995,
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        })
        df.to_parquet(cache_dir / f"{code}_daily.parquet", index=False)
    return cache_dir


def _make_config(tmp_path, cache_dir):
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    # factors_file 相对配置目录解析(P1-9)→ selection.json 一并复制
    _sel = PROJECT_ROOT / "reports" / "selection.json"
    if _sel.exists():
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
        (tmp_path / "reports" / "selection.json").write_bytes(_sel.read_bytes())
    raw["data"]["cache_dir"] = str(cache_dir)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    # Restrict stocks to only the 3 seeded in isolated_cache so build_panel_from_cache
    # doesn't look for parquet files that don't exist in the temp directory.
    raw["stocks"] = [
        {"code": "605589", "name": "圣泉集团", "sector": "化工"},
        {"code": "603986", "name": "兆易创新", "sector": "半导体"},
        {"code": "000528", "name": "柳工",      "sector": "工程机械"},
    ]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_file


def test_factors_analyze_cli_writes_outputs(tmp_path, isolated_cache):
    cfg_file = _make_config(tmp_path, isolated_cache)
    out_dir = tmp_path / "factor_analysis"
    rc = main([
        "factors", "analyze",
        "--config", str(cfg_file),
        "--universe", "pool",
        "--factors", "momentum_20", "rsi_centered_14", "vol_ratio_5",
        "--horizon", "3",
        "--output", str(out_dir),
    ])
    assert rc == 0
    html_files = list(out_dir.glob("*.html"))
    json_files = list(out_dir.glob("*.json"))
    assert len(html_files) == 2  # one dated, one "latest.html"
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["factor_names"] == ["momentum_20", "rsi_centered_14", "vol_ratio_5"]
    assert payload["n_stocks"] == 3
    assert payload["horizon"] == 3


def test_factors_pick_by_ic_writes_selection(tmp_path, isolated_cache):
    cfg_file = _make_config(tmp_path, isolated_cache)
    analyze_dir = tmp_path / "factor_analysis"
    rc = main([
        "factors", "analyze",
        "--config", str(cfg_file),
        "--universe", "pool",
        "--factors", "momentum_20", "rsi_centered_14", "vol_ratio_5",
        "--horizon", "3",
        "--output", str(analyze_dir),
    ])
    assert rc == 0
    json_files = list(analyze_dir.glob("[0-9]*.json"))
    assert len(json_files) == 1
    input_json = json_files[0]

    selection_path = tmp_path / "selection.json"
    rc = main([
        "factors", "pick-by-ic",
        "--input", str(input_json),
        "--output", str(selection_path),
        "--top-n", "2",
        "--max-corr", "0.99",
        "--min-ir", "0.0",
    ])
    assert rc == 0
    assert selection_path.exists()
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    assert "factors" in payload
    assert isinstance(payload["factors"], list)
    assert 0 < len(payload["factors"]) <= 2
    for n in payload["factors"]:
        assert n in {"momentum_20", "rsi_centered_14", "vol_ratio_5"}
