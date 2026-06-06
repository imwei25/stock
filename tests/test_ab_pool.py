"""Tests for stockpool.ab_pool — AB candidate pool build / load / yaml integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from stockpool.ab_pool import AbPoolConfig
from stockpool.config import AppConfig, load_config


def test_ab_pool_config_defaults():
    cfg = AbPoolConfig()
    assert cfg.cache_path == Path("data/ab_pool.parquet")
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
    assert cfg.ab_pool.cache_path == Path("data/ab_pool.parquet")
