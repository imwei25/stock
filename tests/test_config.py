import pytest
import yaml
from pydantic import ValidationError
from stockpool.config import load_config, AppConfig


def _minimal_yaml() -> dict:
    """Smallest valid config — every field explicitly provided."""
    return {
        "stocks": [{"code": "605589", "name": "圣泉集团"}],
        "data": {"history_days": 500, "cache_dir": "data", "force_refresh": False},
        "indicators": {
            "ma_periods": [5, 10, 20, 60],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "kdj": {"n": 9, "m1": 3, "m2": 3},
            "rsi_periods": [6, 12, 24],
            "boll": {"n": 20, "k": 2},
            "volume_ratio_window": 5,
            "breakout_window": 20,
        },
        "weights": {
            "ma_cross_strong": 2, "ma_alignment": 1,
            "macd_cross_above_zero": 2, "macd_cross_below_zero": 1, "macd_histogram_expand": 1,
            "kdj_oversold_cross": 2, "kdj_overbought_cross": 2, "kdj_normal_cross": 1,
            "rsi_oversold": 1, "rsi_overbought": 1,
            "boll_band_touch": 2, "boll_mid_cross": 1,
            "volume_surge_bullish": 1, "volume_surge_bearish": 1,
            "breakout_new_high": 2, "breakout_new_low": 2,
        },
        "scoring": {
            "daily_weight": 0.7, "weekly_weight": 0.3,
            "resonance_bonus": 2, "resonance_daily_threshold": 3, "resonance_weekly_threshold": 1,
        },
        "verdicts": {"strong_buy": 6, "buy": 3, "sell": -3, "strong_sell": -6},
        "backtest": {"forward_days": [5, 10, 20], "equity_curve_holding_days": [5, 10, 20]},
        "report": {"output_dir": "reports", "keep_history": True, "klines_to_show": 120},
    }


def test_load_valid_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(_minimal_yaml()), encoding="utf-8")

    cfg = load_config(cfg_file)

    assert isinstance(cfg, AppConfig)
    assert len(cfg.stocks) == 1
    assert cfg.stocks[0].code == "605589"
    assert cfg.data.history_days == 500
    assert cfg.scoring.daily_weight == 0.7
    assert cfg.backtest.equity_curve_holding_days == [5, 10, 20]


def test_missing_required_field_raises(tmp_path):
    raw = _minimal_yaml()
    del raw["stocks"]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_invalid_type_raises(tmp_path):
    raw = _minimal_yaml()
    raw["data"]["history_days"] = "five hundred"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_config_hash_is_stable(tmp_path):
    """Same config across loads → same content_hash."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(_minimal_yaml()), encoding="utf-8")

    cfg1 = load_config(cfg_file)
    cfg2 = load_config(cfg_file)
    assert cfg1.content_hash == cfg2.content_hash
    assert len(cfg1.content_hash) == 8


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_default_config_yaml_loads():
    """Sanity check: the repo's own config.yaml is valid."""
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    assert len(cfg.stocks) >= 1
    assert all(len(s.code) == 6 for s in cfg.stocks)


def test_equity_curve_holding_days_defaults_when_missing(tmp_path):
    raw = _minimal_yaml()
    del raw["backtest"]["equity_curve_holding_days"]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.backtest.equity_curve_holding_days == [5, 10, 20]


def test_equity_curve_holding_days_rejects_empty(tmp_path):
    raw = _minimal_yaml()
    raw["backtest"]["equity_curve_holding_days"] = []
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_equity_curve_holding_days_rejects_non_positive(tmp_path):
    raw = _minimal_yaml()
    raw["backtest"]["equity_curve_holding_days"] = [5, 0, 10]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(cfg_file)


# === Strategy / MLFactor configuration ===

def test_strategy_defaults_to_composite_verdict(tmp_path):
    """When the yaml omits `strategy:`, the legacy composite path is used."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(_minimal_yaml()), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.strategy.name == "composite_verdict"
    # ml_factor sub-block exists with sensible defaults.
    assert cfg.strategy.ml_factor.horizon == 5
    assert cfg.strategy.ml_factor.train_window == 250
    assert cfg.strategy.ml_factor.panel_mode == "per_stock"


def test_strategy_can_be_set_to_ml_factor(tmp_path):
    raw = _minimal_yaml()
    raw["strategy"] = {
        "name": "ml_factor",
        "ml_factor": {
            "factors": ["momentum_5", "macd_hist"],
            "horizon": 10,
            "train_window": 500,
            "refit_every": 30,
            "panel_mode": "pooled",
            "selector": {"type": "lasso", "lasso": {"alpha": 0.01}},
            "weighter": {"type": "ir", "ir": {"n_chunks": 8}},
            "thresholds": {
                "strong_buy": 0.85, "buy": 0.65, "sell": 0.35, "strong_sell": 0.15,
            },
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.strategy.name == "ml_factor"
    assert cfg.strategy.ml_factor.factors == ["momentum_5", "macd_hist"]
    assert cfg.strategy.ml_factor.panel_mode == "pooled"
    assert cfg.strategy.ml_factor.weighter.type == "ir"
    assert cfg.strategy.ml_factor.weighter.ir.n_chunks == 8
    assert cfg.strategy.ml_factor.thresholds.strong_buy == 0.85


def test_strategy_rejects_unknown_name(tmp_path):
    raw = _minimal_yaml()
    raw["strategy"] = {"name": "neural_net"}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_quantile_thresholds_must_be_ordered(tmp_path):
    raw = _minimal_yaml()
    raw["strategy"] = {
        "name": "ml_factor",
        "ml_factor": {
            "thresholds": {
                "strong_buy": 0.30, "buy": 0.70, "sell": 0.40, "strong_sell": 0.10,
            },
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_selector_lasso_subcfg_explicit():
    """New form: selector.lasso.alpha works."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig.model_validate({
        "type": "lasso",
        "lasso": {"alpha": 0.01, "max_iter": 500, "tol": 1e-5},
    })
    assert cfg.type == "lasso"
    assert cfg.lasso.alpha == 0.01
    assert cfg.lasso.max_iter == 500
    assert cfg.lasso.tol == 1e-5


def test_selector_lasso_subcfg_defaults():
    """selector: {type: lasso} uses LassoConfig defaults."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig.model_validate({"type": "lasso"})
    assert cfg.lasso.alpha == 0.001
    assert cfg.lasso.max_iter == 1000
    assert cfg.lasso.tol == 1e-6


def test_selector_flat_alpha_rejected():
    """Legacy flat alpha field on SelectorConfig must raise ValidationError."""
    import pydantic
    from stockpool.config import SelectorConfig
    with pytest.raises(pydantic.ValidationError) as exc:
        SelectorConfig.model_validate({"type": "lasso", "alpha": 0.01})
    assert "extra" in str(exc.value).lower() or "forbid" in str(exc.value).lower()


def test_mlfactor_embargo_days_default_is_none():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig()
    assert cfg.embargo_days is None


def test_mlfactor_embargo_days_explicit_zero():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig(embargo_days=0)
    assert cfg.embargo_days == 0


def test_mlfactor_embargo_days_explicit_positive():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig(embargo_days=5)
    assert cfg.embargo_days == 5


def test_mlfactor_embargo_days_negative_rejected():
    import pydantic
    from stockpool.config import MLFactorConfig
    with pytest.raises(pydantic.ValidationError):
        MLFactorConfig(embargo_days=-1)


def test_mlfactor_label_type_default_is_return():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig()
    assert cfg.label_type == "return"


def test_mlfactor_label_type_accepts_all_documented():
    from stockpool.config import MLFactorConfig
    for label in ("return", "vol_adjusted", "cross_sec_rank"):
        cfg = MLFactorConfig(label_type=label)
        assert cfg.label_type == label


def test_mlfactor_label_type_unknown_rejected():
    import pydantic
    from stockpool.config import MLFactorConfig
    with pytest.raises(pydantic.ValidationError):
        MLFactorConfig(label_type="momentum")


def test_selector_default_type_is_lasso():
    """Default selector.type was 'lightgbm' in PR-B1 but rolled back to
    'lasso' on 2026-05-24 after A/B validation showed LGB regressed sharpe
    on the 16-stock × 500-bar baseline. See docs/ab_validation_results.md."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig()
    assert cfg.type == "lasso"


def test_selector_lightgbm_subcfg_explicit():
    """selector.lightgbm.num_leaves and friends parse from YAML."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig.model_validate({
        "type": "lightgbm",
        "lightgbm": {
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.1,
            "num_iterations": 100,
            "max_depth": 6,
            "random_state": 7,
            "top_k_factors": 10,
            "min_importance_ratio": 0.05,
            "verbose": 0,
        },
    })
    assert cfg.type == "lightgbm"
    assert cfg.lightgbm.num_leaves == 31
    assert cfg.lightgbm.min_data_in_leaf == 50
    assert cfg.lightgbm.learning_rate == 0.1
    assert cfg.lightgbm.num_iterations == 100
    assert cfg.lightgbm.max_depth == 6
    assert cfg.lightgbm.random_state == 7
    assert cfg.lightgbm.top_k_factors == 10
    assert cfg.lightgbm.min_importance_ratio == 0.05
    assert cfg.lightgbm.verbose == 0


def test_selector_lightgbm_subcfg_defaults():
    """LightGBMSelectorConfig defaults match spec section 3.2."""
    from stockpool.config import SelectorConfig
    cfg = SelectorConfig.model_validate({"type": "lightgbm"})
    assert cfg.lightgbm.num_leaves == 15
    assert cfg.lightgbm.min_data_in_leaf == 20
    assert cfg.lightgbm.learning_rate == 0.05
    assert cfg.lightgbm.num_iterations == 200
    assert cfg.lightgbm.max_depth == 4
    assert cfg.lightgbm.random_state == 42
    assert cfg.lightgbm.top_k_factors == 20
    assert cfg.lightgbm.min_importance_ratio == 0.01
    assert cfg.lightgbm.verbose == -1


def test_selector_lightgbm_flat_num_leaves_rejected():
    """Flat num_leaves at SelectorConfig level is rejected (extra='forbid')."""
    import pydantic
    from stockpool.config import SelectorConfig
    with pytest.raises(pydantic.ValidationError):
        SelectorConfig.model_validate({"type": "lightgbm", "num_leaves": 31})


def test_selector_unknown_type_rejected():
    """type='xgboost' is not in Literal['lasso','lightgbm'] → reject."""
    import pydantic
    from stockpool.config import SelectorConfig
    with pytest.raises(pydantic.ValidationError):
        SelectorConfig.model_validate({"type": "xgboost"})


def test_weighter_default_type_is_ic():
    """Default weighter.type was 'lightgbm' in PR-B2 but rolled back to 'ic'
    on 2026-05-24 after A/B validation showed LGB weighter contributed ~-12%
    return on the 16-stock × 500-bar baseline. See
    docs/ab_validation_results.md."""
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig()
    assert cfg.type == "ic"


def test_weighter_ic_subcfg_explicit():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({
        "type": "ic",
        "ic": {"use_rank": False, "min_abs_ic": 0.05},
    })
    assert cfg.type == "ic"
    assert cfg.ic.use_rank is False
    assert cfg.ic.min_abs_ic == 0.05


def test_weighter_ic_subcfg_defaults():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({"type": "ic"})
    assert cfg.ic.use_rank is True
    assert cfg.ic.min_abs_ic == 0.0


def test_weighter_ir_subcfg_explicit():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({
        "type": "ir",
        "ir": {"n_chunks": 4, "use_rank": False, "min_abs_ir": 0.1},
    })
    assert cfg.type == "ir"
    assert cfg.ir.n_chunks == 4
    assert cfg.ir.use_rank is False
    assert cfg.ir.min_abs_ir == 0.1


def test_weighter_ir_subcfg_defaults():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({"type": "ir"})
    assert cfg.ir.n_chunks == 6
    assert cfg.ir.use_rank is True
    assert cfg.ir.min_abs_ir == 0.0


def test_weighter_equal_subcfg_parses():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({"type": "equal"})
    assert cfg.type == "equal"
    assert cfg.equal is not None


def test_weighter_lightgbm_subcfg_explicit():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({
        "type": "lightgbm",
        "lightgbm": {
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.1,
            "num_iterations": 100,
            "max_depth": 6,
            "random_state": 7,
            "verbose": 0,
        },
    })
    assert cfg.type == "lightgbm"
    assert cfg.lightgbm.num_leaves == 31
    assert cfg.lightgbm.learning_rate == 0.1


def test_weighter_lightgbm_subcfg_defaults():
    from stockpool.config import WeighterConfig
    cfg = WeighterConfig.model_validate({"type": "lightgbm"})
    assert cfg.lightgbm.num_leaves == 15
    assert cfg.lightgbm.min_data_in_leaf == 20
    assert cfg.lightgbm.learning_rate == 0.05
    assert cfg.lightgbm.num_iterations == 200
    assert cfg.lightgbm.max_depth == 4
    assert cfg.lightgbm.random_state == 42
    assert cfg.lightgbm.verbose == -1


def test_weighter_flat_use_rank_rejected():
    import pydantic
    from stockpool.config import WeighterConfig
    with pytest.raises(pydantic.ValidationError):
        WeighterConfig.model_validate({"type": "ic", "use_rank": True})


def test_weighter_unknown_type_rejected():
    import pydantic
    from stockpool.config import WeighterConfig
    with pytest.raises(pydantic.ValidationError):
        WeighterConfig.model_validate({"type": "catboost"})


# ============================================================================
# F3 PR-C — SizingConfig + position_size deprecation
# ============================================================================

import warnings
from stockpool.config import (
    BacktestConfig, SizingConfig, FixedSizingConfig, VolTargetSizingConfig,
)


def test_sizing_config_defaults():
    """Default sizing.type is vol_target with all-default sub-fields."""
    s = SizingConfig()
    assert s.type == "vol_target"
    assert s.fixed.size == 0.1
    assert s.vol_target.reference_vol_annual == 0.30
    assert s.vol_target.vol_window == 20
    assert s.vol_target.min_size == 0.03
    assert s.vol_target.max_size == 0.20
    assert s.vol_target.fallback_to == "fixed"


def test_sizing_config_rejects_extra_fields():
    """SizingConfig has extra='forbid'."""
    with pytest.raises(ValidationError):
        SizingConfig(type="fixed", unknown_field=1)


def test_vol_target_rejects_min_gt_max():
    with pytest.raises(ValidationError):
        VolTargetSizingConfig(min_size=0.5, max_size=0.1)


def test_vol_target_rejects_invalid_fallback():
    with pytest.raises(ValidationError):
        VolTargetSizingConfig(fallback_to="bogus")


def test_backtest_config_default_sizing_is_vol_target():
    bt = BacktestConfig(forward_days=[5], equity_curve_holding_days=[5])
    assert bt.sizing.type == "vol_target"
    assert bt.position_size is None


def test_position_size_alone_migrates_with_deprecation_warning():
    """Setting position_size= triggers DeprecationWarning + auto-migration."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bt = BacktestConfig(
            forward_days=[5], equity_curve_holding_days=[5],
            position_size=0.15,
        )
        depr = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(depr) == 1
        assert "position_size" in str(depr[0].message)
    assert bt.sizing.type == "fixed"
    assert bt.sizing.fixed.size == 0.15
    assert bt.position_size is None  # cleared after migration


def test_position_size_alone_at_default_value_still_migrates():
    """Even position_size=0.1 (= default) triggers migration."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bt = BacktestConfig(
            forward_days=[5], equity_curve_holding_days=[5],
            position_size=0.1,
        )
    assert bt.sizing.type == "fixed"
    assert bt.sizing.fixed.size == 0.1


def test_position_size_plus_explicit_sizing_fixed_size_raises():
    """Explicit non-default sizing.fixed.size alongside position_size → conflict."""
    with pytest.raises(ValidationError, match="position_size"):
        BacktestConfig(
            forward_days=[5], equity_curve_holding_days=[5],
            position_size=0.1,
            sizing=SizingConfig(
                type="fixed",
                fixed=FixedSizingConfig(size=0.2),
            ),
        )


def test_yaml_with_sizing_block_loads(tmp_path):
    """End-to-end: YAML with sizing: block loads cleanly."""
    raw = _minimal_yaml()
    raw["backtest"]["sizing"] = {
        "type": "vol_target",
        "vol_target": {
            "reference_vol_annual": 0.25,
            "vol_window": 30,
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.backtest.sizing.type == "vol_target"
    assert cfg.backtest.sizing.vol_target.reference_vol_annual == 0.25
    assert cfg.backtest.sizing.vol_target.vol_window == 30


def test_yaml_with_legacy_position_size_loads_with_warning(tmp_path):
    """End-to-end: legacy position_size YAML still works, emits warning."""
    raw = _minimal_yaml()
    raw["backtest"]["position_size"] = 0.07
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(cfg_file)
        depr = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(depr) == 1
    assert cfg.backtest.sizing.type == "fixed"
    assert cfg.backtest.sizing.fixed.size == 0.07


# === Portfolio backtest schema (PR-1) ===

def test_portfolio_backtest_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(_minimal_yaml()), encoding="utf-8")
    cfg = load_config(cfg_file)
    pb = cfg.portfolio_backtest
    assert pb.enabled is False
    assert pb.portfolio.top_k == 20
    assert pb.portfolio.rebalance_n_days == 5
    assert pb.portfolio.max_per_industry == 5
    assert pb.portfolio.initial_cash == 1.0
    assert pb.eligibility.min_avg_amount_20d == 5e7
    assert pb.eligibility.exclude_st is True
    assert pb.eligibility.min_history_bars == 60
    assert pb.staggered_starts == 1
    assert pb.score_cache_dir == "data/portfolio_scores"
    # universe_codes default = None → legacy behavior (portfolio universe
    # = training pool)
    assert pb.universe_codes is None


def test_portfolio_backtest_universe_codes_loads_from_yaml(tmp_path):
    raw = _minimal_yaml()
    raw["portfolio_backtest"] = {
        "enabled": True,
        "universe_codes": ["600000", "000001", "300001"],
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.portfolio_backtest.universe_codes == ["600000", "000001", "300001"]


def test_portfolio_backtest_extra_forbid(tmp_path):
    raw = _minimal_yaml()
    raw["portfolio_backtest"] = {"enabled": True, "bogus": 1}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_portfolio_backtest_round_trip(tmp_path):
    raw = _minimal_yaml()
    raw["portfolio_backtest"] = {
        "enabled": True,
        "portfolio": {"top_k": 30, "rebalance_n_days": 10, "max_per_industry": None},
        "eligibility": {"min_avg_amount_20d": 1e8, "exclude_st": False},
        "staggered_starts": 3,
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.portfolio_backtest.enabled is True
    assert cfg.portfolio_backtest.portfolio.top_k == 30
    assert cfg.portfolio_backtest.portfolio.max_per_industry is None
    assert cfg.portfolio_backtest.eligibility.exclude_st is False
    assert cfg.portfolio_backtest.staggered_starts == 3


# ---------------------------------------------------------------------------
# Task 1: MaskConfig schema tests
# ---------------------------------------------------------------------------

def test_mask_config_defaults():
    from stockpool.config import MaskConfig
    cfg = MaskConfig()
    assert cfg.enabled is False
    assert cfg.limit_up_threshold_main == 0.098
    assert cfg.limit_up_threshold_chinext == 0.198
    assert cfg.limit_up_threshold_bse == 0.298
    assert cfg.min_listing_days == 252


def test_mask_config_extra_forbid():
    from stockpool.config import MaskConfig
    import pytest
    with pytest.raises(ValidationError):
        MaskConfig(unknown_field=1)


def test_ml_factor_config_has_mask():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig()
    assert cfg.mask.enabled is False


def test_ml_factor_mask_loaded_from_yaml():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig.model_validate({
        "mask": {"enabled": True, "min_listing_days": 100}
    })
    assert cfg.mask.enabled is True
    assert cfg.mask.min_listing_days == 100
    assert cfg.mask.limit_up_threshold_main == 0.098


def test_ml_factor_content_hash_changes_with_mask(tmp_path):
    from stockpool.config import load_config
    import yaml
    base = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    base["strategy"] = {"name": "ml_factor"}
    f_a = tmp_path / "cfg_a.yaml"
    f_a.write_text(yaml.safe_dump(base), encoding="utf-8")
    cfg_a = load_config(f_a)

    base["strategy"]["ml_factor"] = {"mask": {"enabled": True}}
    f_b = tmp_path / "cfg_b.yaml"
    f_b.write_text(yaml.safe_dump(base), encoding="utf-8")
    cfg_b = load_config(f_b)

    assert cfg_a.content_hash != cfg_b.content_hash
