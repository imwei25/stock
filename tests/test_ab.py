"""Tests for stockpool.ab — config schema, deep-merge, runner, report."""
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from stockpool.ab import (
    ABConfig,
    ArmBacktestOverride,
    ArmOverride,
    build_effective_cfg,
    load_ab_config,
)
from stockpool.config import load_config, StrategyConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_base_config(tmp_path: Path) -> Path:
    """Copy real config.yaml to tmp_path so tests don't tread on it."""
    base_src = PROJECT_ROOT / "config.yaml"
    base_dst = tmp_path / "config.yaml"
    base_dst.write_bytes(base_src.read_bytes())
    return base_dst


# ── Schema ──────────────────────────────────────────────────────────────────


def test_arm_holding_days_must_be_singleton():
    """equity_curve_holding_days enforces length-1 list with N > 0."""
    with pytest.raises(ValidationError):
        ArmBacktestOverride(equity_curve_holding_days=[5, 10])
    with pytest.raises(ValidationError):
        ArmBacktestOverride(equity_curve_holding_days=[])
    with pytest.raises(ValidationError):
        ArmBacktestOverride(equity_curve_holding_days=[0])
    o = ArmBacktestOverride(equity_curve_holding_days=[10])
    assert o.equity_curve_holding_days == [10]


def test_arm_backtest_other_fields_optional():
    """All non-holding-days fields default to None (inherit base)."""
    o = ArmBacktestOverride(equity_curve_holding_days=[7])
    assert o.engine is None
    assert o.position_size is None
    assert o.costs is None


def test_arm_extra_fields_forbidden():
    """Typoed fields raise instead of silently being ignored."""
    with pytest.raises(ValidationError):
        ArmBacktestOverride(equity_curve_holding_days=[10], engin="single")  # typo


def test_arms_must_be_exactly_two():
    """ABConfig requires exactly 2 arms."""
    arm = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    with pytest.raises(ValidationError):
        ABConfig(base_config="config.yaml", arms={"a": arm})
    with pytest.raises(ValidationError):
        ABConfig(base_config="config.yaml", arms={"a": arm, "b": arm, "c": arm})
    ABConfig(base_config="config.yaml", arms={"a": arm, "b": arm})


# ── Deep-merge ──────────────────────────────────────────────────────────────


def test_merge_replaces_strategy_section_wholly(tmp_path):
    """arm.strategy replaces base.strategy with no leakage."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    from stockpool.config import MLFactorConfig
    arm = ArmOverride(
        strategy=StrategyConfig(
            name="ml_factor",
            ml_factor=MLFactorConfig(horizon=7),
        ),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    eff = build_effective_cfg(base, arm)
    assert eff.strategy.name == "ml_factor"
    assert eff.strategy.ml_factor.horizon == 7


def test_merge_backtest_fields_inherit_when_none(tmp_path):
    """arm.backtest fields left as None inherit from base.backtest."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    base_engine = base.backtest.engine
    arm = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    eff = build_effective_cfg(base, arm)
    assert eff.backtest.engine == base_engine
    assert eff.backtest.equity_curve_holding_days == [10]


def test_merge_backtest_fields_override_when_set(tmp_path):
    """arm.backtest fields with explicit values replace base's."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    arm = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(
            equity_curve_holding_days=[10],
            engine="single",
            position_size=0.25,
        ),
    )
    eff = build_effective_cfg(base, arm)
    assert eff.backtest.engine == "single"
    assert eff.backtest.position_size == 0.25


def test_merge_does_not_mutate_base(tmp_path):
    """Merging is non-destructive on the base config object."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    base_engine_before = base.backtest.engine
    base_name_before = base.strategy.name
    arm = ArmOverride(
        strategy=StrategyConfig(name="ml_factor"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10], engine="single"),
    )
    build_effective_cfg(base, arm)
    assert base.backtest.engine == base_engine_before
    assert base.strategy.name == base_name_before


def test_merge_recomputes_content_hash(tmp_path):
    """Different arm overrides → different content_hash (ML cache isolation)."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    arm_a = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[5]),
    )
    arm_b = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
    )
    eff_a = build_effective_cfg(base, arm_a)
    eff_b = build_effective_cfg(base, arm_b)
    assert eff_a.content_hash != eff_b.content_hash


def test_merge_revalidates_pydantic(tmp_path):
    """A merge result that violates AppConfig constraints fails fast."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    arm = ArmOverride(
        strategy=StrategyConfig(name="composite_verdict"),
        backtest=ArmBacktestOverride(
            equity_curve_holding_days=[10],
            position_size=2.0,  # exceeds le=1.0 in BacktestConfig
        ),
    )
    with pytest.raises(ValidationError):
        build_effective_cfg(base, arm)


# ── load_ab_config ──────────────────────────────────────────────────────────


def _write_ab_yaml(tmp_path: Path, base_rel: str = "config.yaml",
                   stocks_filter=None, with_typo: bool = False) -> Path:
    arms = {
        "a": {
            "strategy": {"name": "composite_verdict"},
            "backtest": {"equity_curve_holding_days": [5]},
        },
        "b": {
            "strategy": {"name": "composite_verdict"},
            "backtest": {"equity_curve_holding_days": [10]},
        },
    }
    if with_typo:
        arms["a"]["backtest"]["equity_holding_days"] = [5]  # typo
    raw = {"base_config": base_rel, "arms": arms}
    if stocks_filter is not None:
        raw["stocks_filter"] = stocks_filter
    p = tmp_path / "ab.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_load_ab_config_happy_path(tmp_path):
    _write_base_config(tmp_path)
    ab_path = _write_ab_yaml(tmp_path)
    ab = load_ab_config(ab_path)
    assert list(ab.arms) == ["a", "b"]


def test_load_ab_config_missing_base_raises(tmp_path):
    ab_path = _write_ab_yaml(tmp_path, base_rel="does_not_exist.yaml")
    with pytest.raises(ValueError, match="base_config"):
        load_ab_config(ab_path)


def test_load_ab_config_resolves_base_relative_to_ab_yaml(tmp_path):
    """base_config: ../config.yaml works when ab.yaml is in a subdir."""
    _write_base_config(tmp_path)
    subdir = tmp_path / "experiments"
    subdir.mkdir()
    ab_path = _write_ab_yaml(subdir, base_rel="../config.yaml")
    ab = load_ab_config(ab_path)
    assert ab.base_config == "../config.yaml"


def test_stocks_filter_must_be_subset_of_base(tmp_path):
    _write_base_config(tmp_path)
    ab_path = _write_ab_yaml(tmp_path, stocks_filter=["999999"])  # not in base
    with pytest.raises(ValueError, match="stocks_filter"):
        load_ab_config(ab_path)


def test_extra_field_in_arm_backtest_rejected(tmp_path):
    _write_base_config(tmp_path)
    ab_path = _write_ab_yaml(tmp_path, with_typo=True)
    with pytest.raises(ValidationError):
        load_ab_config(ab_path)
