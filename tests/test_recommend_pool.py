"""Tests for stockpool.recommend_pool (Pool B).

Strategy + universe loading + industry map are all stubbed so tests run
offline. End-to-end covers the funnel, the greedy industry cap, the ISO-week
cache, and per-stock failure isolation.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from stockpool import recommend_pool
from stockpool.config import load_config


# === helpers ===

def _daily(n: int = 60, close: float = 20.0, volume: float = 100_000.0) -> pd.DataFrame:
    """Synthetic OHLCV. ``volume`` 单位 = 股 (P1-6); amount = volume*close."""
    dates = pd.date_range("2026-01-02", periods=n, freq="B")
    c = np.full(n, close)
    return pd.DataFrame({
        "date": dates,
        "open": c, "high": c + 0.1, "low": c - 0.1, "close": c,
        "volume": np.full(n, volume),
    })


def _minimal_cfg_yaml(tmp_path: Path, **overrides) -> Path:
    """Write a minimal yaml that loads + add a recommend_pool block."""
    raw = {
        "stocks": [{"code": "605589", "name": "圣泉集团"}],
        "data": {"history_days": 500, "cache_dir": str(tmp_path / "cache"),
                 "force_refresh": False},
        "indicators": {
            "ma_periods": [5, 10, 20, 60],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "kdj": {"n": 9, "m1": 3, "m2": 3},
            "rsi_periods": [6, 12, 24],
            "boll": {"n": 20, "k": 2},
            "volume_ratio_window": 5, "breakout_window": 20,
        },
        "weights": {
            "ma_cross_strong": 2, "ma_alignment": 1,
            "macd_cross_above_zero": 2, "macd_cross_below_zero": 1,
            "macd_histogram_expand": 1,
            "kdj_oversold_cross": 2, "kdj_overbought_cross": 2,
            "kdj_normal_cross": 1, "rsi_oversold": 1, "rsi_overbought": 1,
            "boll_band_touch": 2, "boll_mid_cross": 1,
            "volume_surge_bullish": 1, "volume_surge_bearish": 1,
            "breakout_new_high": 2, "breakout_new_low": 2,
        },
        "scoring": {"daily_weight": 0.7, "weekly_weight": 0.3,
                    "resonance_bonus": 2, "resonance_daily_threshold": 3,
                    "resonance_weekly_threshold": 1},
        "verdicts": {"strong_buy": 6, "buy": 3, "sell": -3, "strong_sell": -6},
        "backtest": {"forward_days": [5], "equity_curve_holding_days": [5]},
        "report": {"output_dir": "reports", "keep_history": True,
                   "klines_to_show": 120},
        "recommend_pool": {
            "enabled": True, "top_n": 5, "min_avg_amount_20d": 1.0,
            "max_per_industry": 2, "refresh": "weekly",
            "cache_dir": str(tmp_path / "pool_b"),
        },
    }
    for k, v in overrides.items():
        raw["recommend_pool"][k] = v
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


class _StubStrategy:
    """Returns a per-code score lookup. predict_latest only uses ``code``."""
    def __init__(self, code: str, scores: dict[str, float]):
        self._code = code
        self._scores = scores

    def predict_latest(self, daily_df):
        s = self._scores.get(self._code)
        if s is None:
            return {"signal": "neutral", "final_score": float("nan")}
        verdict = "strong_buy" if s > 0.5 else "buy" if s > 0 else "sell"
        return {"signal": verdict, "final_score": s}


def _patch_strategy(monkeypatch, scores: dict[str, float]):
    def _builder(cfg, pool_data=None, current_stock_code=None,
                 factor_panel=None, close_panel=None, shared_cache=None):
        return _StubStrategy(current_stock_code, scores)
    monkeypatch.setattr(recommend_pool, "build_strategy", _builder)


def _patch_data(
    monkeypatch,
    universe: Mapping[str, pd.DataFrame],
    names: Mapping[str, str],
    industries: Mapping[str, str],
):
    monkeypatch.setattr(recommend_pool, "load_universe_cache",
                        lambda *a, **k: dict(universe))
    monkeypatch.setattr(
        recommend_pool, "list_universe",
        lambda *a, **k: pd.DataFrame(
            [{"code": c, "name": n, "market": "SH"} for c, n in names.items()]
        ),
    )
    monkeypatch.setattr(
        recommend_pool, "load_or_build_industry_map",
        lambda *a, **k: dict(industries),
    )


# === config schema ===

def test_recommend_pool_defaults(tmp_path):
    """When yaml omits recommend_pool, defaults apply (enabled=True, top_n=30)."""
    raw = yaml.safe_load(_minimal_cfg_yaml(tmp_path).read_text())
    del raw["recommend_pool"]
    p = tmp_path / "no_poolb.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.recommend_pool.enabled is True
    assert cfg.recommend_pool.top_n == 30
    assert cfg.recommend_pool.max_per_industry == 5
    assert cfg.recommend_pool.refresh == "weekly"


def test_recommend_pool_rejects_bad_refresh(tmp_path):
    raw = yaml.safe_load(_minimal_cfg_yaml(tmp_path).read_text())
    raw["recommend_pool"]["refresh"] = "hourly"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(p)


# === funnel ===

def test_funnel_liquidity_threshold(monkeypatch, tmp_path):
    """Volume × close → amount in 元(volume 单位 = 股);below threshold is dropped."""
    cfg = load_config(_minimal_cfg_yaml(tmp_path, min_avg_amount_20d=2e7))
    # A: amount = 10M 股 * 10 = 100M 元 → keep
    # B: amount = 100k 股 * 10 = 1M 元 → drop
    universe = {
        "000001": _daily(volume=10_000_000, close=10.0),
        "000002": _daily(volume=100_000, close=10.0),
    }
    _patch_data(monkeypatch, universe,
                names={"000001": "A", "000002": "B"},
                industries={"000001": "X", "000002": "Y"})
    _patch_strategy(monkeypatch, {"000001": 0.9, "000002": 0.9})

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    codes = [e.code for e in out]
    assert "000001" in codes
    assert "000002" not in codes


def test_funnel_drops_st_by_name(monkeypatch, tmp_path):
    cfg = load_config(_minimal_cfg_yaml(tmp_path))
    universe = {
        "000001": _daily(volume=100_000),
        "000002": _daily(volume=100_000),
    }
    _patch_data(monkeypatch, universe,
                names={"000001": "正常股", "000002": "ST 麻烦"},
                industries={"000001": "X", "000002": "X"})
    _patch_strategy(monkeypatch, {"000001": 0.5, "000002": 0.9})

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert [e.code for e in out] == ["000001"]


def test_funnel_drops_short_history(monkeypatch, tmp_path):
    """< 20 bars → cannot compute 20-day avg amount, drop."""
    cfg = load_config(_minimal_cfg_yaml(tmp_path))
    universe = {
        "000001": _daily(n=60, volume=100_000),
        "000002": _daily(n=10, volume=100_000),   # too short
    }
    _patch_data(monkeypatch, universe,
                names={"000001": "A", "000002": "B"},
                industries={"000001": "X", "000002": "Y"})
    _patch_strategy(monkeypatch, {"000001": 0.5, "000002": 0.9})

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert [e.code for e in out] == ["000001"]


# === industry cap ===

def test_industry_cap_greedy(monkeypatch, tmp_path):
    """4 stocks in same industry, cap=2 → top-2 by score only."""
    cfg = load_config(_minimal_cfg_yaml(tmp_path,
                                         top_n=10, max_per_industry=2))
    universe = {c: _daily(volume=100_000) for c in
                ["000001", "000002", "000003", "000004"]}
    _patch_data(monkeypatch, universe,
                names={c: c for c in universe},
                industries={c: "化工" for c in universe})
    _patch_strategy(monkeypatch, {
        "000001": 0.9, "000002": 0.7, "000003": 0.5, "000004": 0.3,
    })

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert [e.code for e in out] == ["000001", "000002"]
    assert [e.rank for e in out] == [1, 2]


def test_all_unmapped_bypasses_cap(monkeypatch, tmp_path):
    """If every stock is unmapped (akshare failed → industry_map empty),
    skip the cap entirely. Otherwise the pool would be truncated to just
    `max_per_industry` items — useless."""
    cfg = load_config(_minimal_cfg_yaml(tmp_path,
                                         top_n=10, max_per_industry=1))
    universe = {c: _daily(volume=100_000) for c in
                ["000001", "000002", "000003"]}
    _patch_data(monkeypatch, universe,
                names={c: c for c in universe},
                industries={})  # all unmapped → "未知"
    _patch_strategy(monkeypatch, {
        "000001": 0.9, "000002": 0.7, "000003": 0.5,
    })

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    # All "未知" → cap bypassed → all 3 survive
    assert [e.code for e in out] == ["000001", "000002", "000003"]
    assert all(e.industry == "未知" for e in out)


def test_mixed_mapped_and_unmapped_still_caps_unknown(monkeypatch, tmp_path):
    """When industry map is partially populated, the 未知 bucket should still
    be capped (those few unmapped stocks shouldn't flood the pool)."""
    cfg = load_config(_minimal_cfg_yaml(tmp_path,
                                         top_n=10, max_per_industry=1))
    universe = {c: _daily(volume=100_000) for c in
                ["000001", "000002", "000003", "000004"]}
    _patch_data(monkeypatch, universe,
                names={c: c for c in universe},
                # 000001 mapped, others unmapped → "未知"
                industries={"000001": "化工"})
    _patch_strategy(monkeypatch, {
        "000001": 0.9, "000002": 0.7, "000003": 0.5, "000004": 0.3,
    })

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    # 化工: cap=1 → 000001; 未知: cap=1 → 000002 (highest unmapped)
    assert [e.code for e in out] == ["000001", "000002"]


# === ranking + pool A overlap ===

def test_overlap_with_pool_a_marked(monkeypatch, tmp_path):
    """605589 is in cfg.stocks → is_in_pool_a=True."""
    cfg = load_config(_minimal_cfg_yaml(tmp_path,
                                         top_n=5, max_per_industry=5))
    codes = ["605589", "000001", "000002"]
    universe = {c: _daily(volume=100_000) for c in codes}
    _patch_data(monkeypatch, universe,
                names={c: c for c in codes},
                industries={c: f"sec-{c}" for c in codes})
    _patch_strategy(monkeypatch, {c: 0.5 for c in codes})

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    by_code = {e.code: e for e in out}
    assert by_code["605589"].is_in_pool_a is True
    assert by_code["000001"].is_in_pool_a is False


# === fail isolation ===

def test_per_stock_predict_failure_skipped(monkeypatch, tmp_path):
    cfg = load_config(_minimal_cfg_yaml(tmp_path,
                                         top_n=5, max_per_industry=5))
    codes = ["000001", "000002", "000003"]
    universe = {c: _daily(volume=100_000) for c in codes}
    _patch_data(monkeypatch, universe,
                names={c: c for c in codes},
                industries={c: f"sec-{c}" for c in codes})

    class _FailingFor000002:
        def __init__(self, code: str):
            self._c = code
        def predict_latest(self, daily_df):
            if self._c == "000002":
                raise RuntimeError("synthetic blow up")
            return {"signal": "buy", "final_score": 0.5}

    def _builder(cfg, pool_data=None, current_stock_code=None,
                 factor_panel=None, close_panel=None, shared_cache=None):
        return _FailingFor000002(current_stock_code)
    monkeypatch.setattr(recommend_pool, "build_strategy", _builder)

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    out_codes = {e.code for e in out}
    assert out_codes == {"000001", "000003"}


def test_nan_score_skipped(monkeypatch, tmp_path):
    cfg = load_config(_minimal_cfg_yaml(tmp_path))
    codes = ["000001", "000002"]
    universe = {c: _daily(volume=100_000) for c in codes}
    _patch_data(monkeypatch, universe,
                names={c: c for c in codes},
                industries={c: "X" for c in codes})
    _patch_strategy(monkeypatch, {
        "000001": 0.5,           # 000002 missing → NaN final_score
    })
    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert [e.code for e in out] == ["000001"]


# === cache ===

def test_cache_hit_same_week_skips_compute(monkeypatch, tmp_path):
    cfg = load_config(_minimal_cfg_yaml(tmp_path))
    universe = {"000001": _daily(volume=100_000)}
    _patch_data(monkeypatch, universe,
                names={"000001": "A"}, industries={"000001": "X"})
    _patch_strategy(monkeypatch, {"000001": 0.5})

    # First call: computes + writes cache
    out1 = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert len(out1) == 1

    # Second call: replace build_strategy with one that explodes
    def _boom(*a, **k):
        raise AssertionError("strategy should not be called on cache hit")
    monkeypatch.setattr(recommend_pool, "build_strategy", _boom)

    out2 = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert [e.code for e in out2] == ["000001"]
    assert out2[0].final_score == 0.5


def test_cache_invalidates_across_iso_weeks(monkeypatch, tmp_path):
    """date in week W and date in week W+1 → different cache files."""
    cfg = load_config(_minimal_cfg_yaml(tmp_path))
    universe = {"000001": _daily(volume=100_000)}
    _patch_data(monkeypatch, universe,
                names={"000001": "A"}, industries={"000001": "X"})
    _patch_strategy(monkeypatch, {"000001": 0.5})

    p_w18 = recommend_pool._cache_path_for(cfg, date(2026, 5, 4))   # W18
    p_w19 = recommend_pool._cache_path_for(cfg, date(2026, 5, 11))  # W19
    assert p_w18 != p_w19


def test_cache_invalidates_on_content_hash_change(tmp_path):
    cfg_a = load_config(_minimal_cfg_yaml(tmp_path))
    # Different yaml → different content_hash
    cfg_b_path = _minimal_cfg_yaml(tmp_path, top_n=42)
    cfg_b = load_config(cfg_b_path)
    assert cfg_a.content_hash != cfg_b.content_hash
    assert (recommend_pool._cache_path_for(cfg_a, date(2026, 5, 4))
            != recommend_pool._cache_path_for(cfg_b, date(2026, 5, 4)))


def test_refresh_always_bypasses_cache(monkeypatch, tmp_path):
    cfg = load_config(_minimal_cfg_yaml(tmp_path, refresh="always"))
    universe = {"000001": _daily(volume=100_000)}
    _patch_data(monkeypatch, universe,
                names={"000001": "A"}, industries={"000001": "X"})

    calls = {"n": 0}
    class _Counting:
        def __init__(self, code): self._c = code
        def predict_latest(self, _df):
            calls["n"] += 1
            return {"signal": "buy", "final_score": 0.5}
    monkeypatch.setattr(
        recommend_pool, "build_strategy",
        lambda cfg, pool_data=None, current_stock_code=None,
               factor_panel=None, close_panel=None,
               shared_cache=None: _Counting(current_stock_code),
    )

    recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert calls["n"] == 2   # both runs hit predict_latest


def test_refresh_never_without_cache_returns_empty(monkeypatch, tmp_path):
    cfg = load_config(_minimal_cfg_yaml(tmp_path, refresh="never"))
    universe = {"000001": _daily(volume=100_000)}
    _patch_data(monkeypatch, universe,
                names={"000001": "A"}, industries={"000001": "X"})
    _patch_strategy(monkeypatch, {"000001": 0.5})

    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert out == []


def test_empty_universe_returns_empty(monkeypatch, tmp_path):
    cfg = load_config(_minimal_cfg_yaml(tmp_path))
    _patch_data(monkeypatch, {}, names={}, industries={})
    _patch_strategy(monkeypatch, {})
    out = recommend_pool.compute_or_load_pool_b(cfg, date(2026, 5, 4))
    assert out == []
