"""Tests for portfolio_ab.config: schema + build_effective_cfg."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from stockpool.config import load_config
from stockpool.portfolio_ab.config import (
    PortfolioABConfig,
    PortfolioArmOverride,
    build_effective_cfg,
    load_portfolio_ab_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _base_cfg():
    return load_config(PROJECT_ROOT / "config.yaml")


def _write_ab(tmp_path: Path, arms: dict) -> Path:
    base_path = tmp_path / "base.yaml"
    base_path.write_bytes((PROJECT_ROOT / "config.yaml").read_bytes())
    # factors_file 相对配置目录解析(P1-9)→ selection.json 一并复制
    sel_src = PROJECT_ROOT / "reports" / "selection.json"
    if sel_src.exists():
        (tmp_path / "reports").mkdir(exist_ok=True)
        (tmp_path / "reports" / "selection.json").write_bytes(sel_src.read_bytes())
    ab_path = tmp_path / "ab.yaml"
    ab_path.write_text(
        yaml.safe_dump({"base_config": "base.yaml", "arms": arms}),
        encoding="utf-8",
    )
    return ab_path


def test_loader_rejects_one_arm(tmp_path):
    ab_path = _write_ab(tmp_path, arms={"a": {"strategy": {"name": "composite_verdict"}}})
    with pytest.raises(ValidationError):
        load_portfolio_ab_config(ab_path)


def test_loader_rejects_three_arms(tmp_path):
    ab_path = _write_ab(tmp_path, arms={
        "a": {"strategy": {"name": "composite_verdict"}},
        "b": {"strategy": {"name": "composite_verdict"}},
        "c": {"strategy": {"name": "composite_verdict"}},
    })
    with pytest.raises(ValidationError):
        load_portfolio_ab_config(ab_path)


def test_loader_rejects_unknown_top_key(tmp_path):
    ab_path = _write_ab(tmp_path, arms={
        "a": {"strategy": {"name": "composite_verdict"}},
        "b": {"strategy": {"name": "composite_verdict"}, "bogus": 1},
    })
    with pytest.raises(ValidationError):
        load_portfolio_ab_config(ab_path)


def test_loader_happy_path(tmp_path):
    ab_path = _write_ab(tmp_path, arms={
        "a": {"strategy": {"name": "composite_verdict"}},
        "b": {"strategy": {"name": "ml_factor"}},
    })
    cfg = load_portfolio_ab_config(ab_path)
    assert isinstance(cfg, PortfolioABConfig)
    assert list(cfg.arms) == ["a", "b"]


def test_build_effective_cfg_replaces_strategy_wholesale():
    base = _base_cfg()
    arm = PortfolioArmOverride(strategy={"name": "composite_verdict"})
    eff = build_effective_cfg(base, arm)
    assert eff.strategy.name == "composite_verdict"
    # base.strategy was ml_factor (per config.yaml) — gone now
    # (we never inherit nested fields when whole-replacing).


def test_build_effective_cfg_field_merges_portfolio_backtest():
    base = _base_cfg()
    arm = PortfolioArmOverride(
        portfolio_backtest={
            "enabled": True,
            "portfolio": {"top_k": 99},     # only override top_k
        },
    )
    eff = build_effective_cfg(base, arm)
    assert eff.portfolio_backtest.enabled is True
    assert eff.portfolio_backtest.portfolio.top_k == 99
    # Other portfolio.* fields inherit from base defaults.
    assert eff.portfolio_backtest.portfolio.rebalance_n_days == 5
    assert eff.portfolio_backtest.staggered_starts == 1


def test_build_effective_cfg_no_override_inherits():
    base = _base_cfg()
    arm = PortfolioArmOverride()  # both fields None
    eff = build_effective_cfg(base, arm)
    assert eff.strategy.name == base.strategy.name
    assert eff.portfolio_backtest.enabled == base.portfolio_backtest.enabled


def test_build_effective_cfg_recomputes_content_hash():
    base = _base_cfg()
    arm_a = PortfolioArmOverride(strategy={"name": "composite_verdict"})
    arm_b = PortfolioArmOverride(strategy={"name": "ml_factor"})
    h_a = build_effective_cfg(base, arm_a).content_hash
    h_b = build_effective_cfg(base, arm_b).content_hash
    assert h_a != h_b, "different strategies should yield different hashes"
    # Same override twice → same hash (canonical sorted-key serialization).
    h_a2 = build_effective_cfg(base, arm_a).content_hash
    assert h_a == h_a2


def test_arm_extra_field_forbidden():
    with pytest.raises(ValidationError):
        PortfolioArmOverride(stocks_filter=["605589"])  # type: ignore
