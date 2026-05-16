import pytest
import yaml
from pydantic import ValidationError
from stockpool.config import load_config, AppConfig


def _minimal_yaml() -> dict:
    """Smallest valid config — every required field present."""
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
        "backtest": {"forward_days": [5, 10, 20]},
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
