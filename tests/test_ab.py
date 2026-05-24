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


# ── Pool sharing plan ───────────────────────────────────────────────────────


def _make_cfg(tmp_path, strategy_name, panel_mode=None,
              training_universe=None, factors=None):
    """Helper: build an effective AppConfig with the requested strategy variant."""
    base_path = _write_base_config(tmp_path)
    base = load_config(base_path)
    if strategy_name == "ml_factor":
        from stockpool.config import MLFactorConfig
        kw = {}
        if panel_mode is not None:
            kw["panel_mode"] = panel_mode
        if training_universe is not None:
            kw["training_universe"] = training_universe
        if factors is not None:
            kw["factors"] = factors
        ml_cfg = MLFactorConfig(**kw)
        arm = ArmOverride(
            strategy=StrategyConfig(name="ml_factor", ml_factor=ml_cfg),
            backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
        )
    else:
        arm = ArmOverride(
            strategy=StrategyConfig(name="composite_verdict"),
            backtest=ArmBacktestOverride(equity_curve_holding_days=[10]),
        )
    return build_effective_cfg(base, arm)


def test_pool_plan_both_composite(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    cfgs = [_make_cfg(tmp_path, "composite_verdict")] * 2
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is False
    assert plan["shared_factors"] is None


def test_pool_plan_ml_vs_composite(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    cfgs = [
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all"),
        _make_cfg(tmp_path, "composite_verdict"),
    ]
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is True
    assert plan["shared_factors"] is None


def test_pool_plan_both_ml_pooled_all_same_factors(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    factors = ["momentum_20", "rsi_centered_14"]
    cfgs = [
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all", factors=factors),
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all", factors=factors),
    ]
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is True
    assert plan["shared_factors"] == factors


def test_pool_plan_both_ml_pooled_all_different_factors(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    cfgs = [
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all", factors=["momentum_20"]),
        _make_cfg(tmp_path, "ml_factor", panel_mode="pooled",
                  training_universe="all", factors=["rsi_centered_14"]),
    ]
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is True
    assert plan["shared_factors"] is None


def test_pool_plan_one_ml_per_stock_does_not_load_universe(tmp_path):
    from stockpool.ab.runner import _decide_pool_sharing
    cfgs = [
        _make_cfg(tmp_path, "ml_factor", panel_mode="per_stock"),
        _make_cfg(tmp_path, "composite_verdict"),
    ]
    plan = _decide_pool_sharing(cfgs, stocks=[])
    assert plan["load_universe"] is False


# ── Runner integration ──────────────────────────────────────────────────────


@pytest.fixture
def isolated_cache_two_stocks(tmp_path, monkeypatch):
    """Cache directory with two synthetic stocks ready to load."""
    import numpy as np
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    for code, seed in [("605589", 7), ("300750", 19)]:
        rng = np.random.default_rng(seed)
        n = 220
        returns = rng.normal(0.0005, 0.02, n)
        close = 100.0 * np.cumprod(1 + returns)
        df = __import__("pandas").DataFrame({
            "date": __import__("pandas").date_range("2024-01-02", periods=n, freq="B"),
            "open":  close * 0.998, "high": close * 1.005,
            "low":   close * 0.995, "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        })
        df.to_parquet(cache_dir / f"{code}_daily.parquet", index=False)

    import pandas as pd
    cache_last = pd.date_range("2024-01-02", periods=220, freq="B")[-1]
    fresh_today = pd.Timestamp(cache_last) + pd.Timedelta(days=1)
    monkeypatch.setattr("stockpool.fetcher._today", lambda: fresh_today)
    return cache_dir


def _ab_setup(tmp_path, cache_dir):
    """Build an ab.yaml + base config wired to a synthetic cache."""
    import yaml
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(cache_dir)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    raw["stocks"] = [
        {"code": "605589", "name": "Alpha", "sector": ""},
        {"code": "300750", "name": "Bravo", "sector": ""},
    ]
    base_path = tmp_path / "config.yaml"
    base_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    ab_raw = {
        "base_config": "config.yaml",
        "arms": {
            "single_engine": {
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10], "engine": "single"},
            },
            "multi_lot_engine": {
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10], "engine": "multi_lot"},
            },
        },
    }
    ab_path = tmp_path / "ab.yaml"
    ab_path.write_text(yaml.safe_dump(ab_raw, sort_keys=False), encoding="utf-8")
    ab_cfg = load_ab_config(ab_path)
    base_cfg = load_config(base_path)
    return ab_cfg, base_cfg


def test_run_ab_smoke_two_composite_arms(tmp_path, isolated_cache_two_stocks):
    from stockpool.ab import run_ab
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    result = run_ab(ab_cfg, base_cfg, base_cfg.stocks, refresh=False)
    assert result.arm_a.name == "single_engine"
    assert result.arm_b.name == "multi_lot_engine"
    assert len(result.arm_a.per_stock) == 2
    assert len(result.arm_b.per_stock) == 2
    assert result.arm_a.failed == []
    assert result.arm_b.failed == []


def test_run_ab_per_stock_failure_isolated(tmp_path, isolated_cache_two_stocks,
                                           monkeypatch):
    """Force one stock's walk_forward_verdicts to crash; ABResult still returns,
    crash recorded in `failed`."""
    from stockpool.ab import run_ab
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)

    from stockpool import backtest_runner as br
    real_wf = br.walk_forward_verdicts
    state = {"calls": 0}
    def _maybe_throw(daily, *a, **kw):
        state["calls"] += 1
        # First call (first arm, first stock) throws once
        if state["calls"] == 1:
            raise RuntimeError("simulated crash")
        return real_wf(daily, *a, **kw)
    monkeypatch.setattr(br, "walk_forward_verdicts", _maybe_throw)

    result = run_ab(ab_cfg, base_cfg, base_cfg.stocks, refresh=False)
    # At least one arm had at least one stock fail; total survivors > 0.
    total_failed = len(result.arm_a.failed) + len(result.arm_b.failed)
    total_done = len(result.arm_a.per_stock) + len(result.arm_b.per_stock)
    assert total_failed >= 1
    assert total_done >= 1


def test_run_single_arm_returns_arm_result(tmp_path, isolated_cache_two_stocks):
    from stockpool.ab import run_single_arm
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    result = run_single_arm(ab_cfg, base_cfg, base_cfg.stocks, False, "multi_lot_engine")
    assert result.name == "multi_lot_engine"
    assert len(result.per_stock) == 2


def test_run_single_arm_unknown_name_raises(tmp_path, isolated_cache_two_stocks):
    from stockpool.ab import run_single_arm
    ab_cfg, base_cfg = _ab_setup(tmp_path, isolated_cache_two_stocks)
    with pytest.raises(KeyError):
        run_single_arm(ab_cfg, base_cfg, base_cfg.stocks, False, "no_such_arm")
