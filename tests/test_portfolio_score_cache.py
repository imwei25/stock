"""score_cache_key: scoring-signature cache so arms differing only in
portfolio params reuse the precomputed score panel."""
from __future__ import annotations

from pathlib import Path

import yaml

import pandas as pd

from stockpool.config import load_config
from stockpool.portfolio.scoring import _set_stock_context, score_cache_key
from stockpool.portfolio_ab.config import build_effective_cfg, load_portfolio_ab_config


class _FakeStrat:
    def __init__(self, panel=None):
        if panel is not None:
            self._factor_panel = panel
        self._current_stock_code = "SENTINEL"


def test_set_stock_context_slices_when_in_panel():
    panel = {"f1": pd.DataFrame(columns=["A", "B"]), "f2": pd.DataFrame(columns=["A", "B"])}
    s = _FakeStrat(panel)
    _set_stock_context(s, "A")
    assert s._current_stock_code == "A"          # in panel → slice that stock
    _set_stock_context(s, "ZZZ")
    assert s._current_stock_code is None          # absent → None → recompute fallback


def test_set_stock_context_noop_without_panel():
    s = _FakeStrat(panel=None)   # composite_verdict-like: no _factor_panel
    _set_stock_context(s, "A")
    assert s._current_stock_code == "SENTINEL"    # untouched, no crash

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODES = ["000001", "000002", "600000", "600519"]


def _ab(tmp_path: Path, arms: dict):
    base_path = tmp_path / "base.yaml"
    base_path.write_bytes((PROJECT_ROOT / "config.yaml").read_bytes())
    ab_path = tmp_path / "ab.yaml"
    ab_path.write_text(yaml.safe_dump({"base_config": "base.yaml", "arms": arms}),
                       encoding="utf-8")
    base = load_config(base_path)
    ab = load_portfolio_ab_config(ab_path)
    return {n: build_effective_cfg(base, ov) for n, ov in ab.arms.items()}


def test_key_shared_when_only_portfolio_params_differ(tmp_path):
    """Arms differing only in top_k / rebalance → identical score cache key."""
    cfgs = _ab(tmp_path, arms={
        "k3": {"strategy": {"name": "composite_verdict"},
               "portfolio_backtest": {"portfolio": {"top_k": 3, "rebalance_n_days": 5}}},
        "k10": {"strategy": {"name": "composite_verdict"},
                "portfolio_backtest": {"portfolio": {"top_k": 10, "rebalance_n_days": 20}}},
    })
    ka = score_cache_key(cfgs["k3"], CODES)
    kb = score_cache_key(cfgs["k10"], CODES)
    assert ka == kb, "arms with identical scoring must share the score cache key"
    # content_hash, by contrast, DOES differ (full-config hash) — the old key.
    assert cfgs["k3"].content_hash != cfgs["k10"].content_hash


def test_key_differs_when_universe_differs(tmp_path):
    cfgs = _ab(tmp_path, arms={
        "a": {"strategy": {"name": "composite_verdict"},
              "portfolio_backtest": {"portfolio": {"top_k": 3}}},
        "b": {"strategy": {"name": "composite_verdict"},
              "portfolio_backtest": {"portfolio": {"top_k": 10}}},
    })
    k_full = score_cache_key(cfgs["a"], CODES)
    k_subset = score_cache_key(cfgs["a"], CODES[:2])
    assert k_full != k_subset
    # order-independent
    assert score_cache_key(cfgs["a"], CODES) == score_cache_key(cfgs["a"], list(reversed(CODES)))


def test_key_differs_when_strategy_differs(tmp_path):
    cfgs = _ab(tmp_path, arms={
        "comp": {"strategy": {"name": "composite_verdict"}},
        "ml": {"strategy": {"name": "ml_factor",
                            "ml_factor": {"factors": ["momentum_20"]}}},
    })
    assert score_cache_key(cfgs["comp"], CODES) != score_cache_key(cfgs["ml"], CODES)
