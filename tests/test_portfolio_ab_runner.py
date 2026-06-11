"""Tests for portfolio_ab.runner: happy path + failure isolation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stockpool.config import load_config
from stockpool.portfolio_ab.config import (
    PortfolioABConfig,
    PortfolioArmOverride,
)
from stockpool.portfolio_ab.runner import ArmResult, run_portfolio_ab


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _seed_panel(codes, n=80):
    rng = np.random.default_rng(13)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    out = {}
    for c in codes:
        ret = rng.normal(0.0005, 0.02, n)
        close = 100.0 * np.cumprod(1 + ret)
        out[c] = pd.DataFrame({
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.005,
            "low":  close * 0.995,
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        })
    return out


@pytest.fixture
def base_cfg(tmp_path):
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    # factors_file 相对配置目录解析(P1-9)→ selection.json 一并复制
    _sel = PROJECT_ROOT / "reports" / "selection.json"
    if _sel.exists():
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
        (tmp_path / "reports" / "selection.json").write_bytes(_sel.read_bytes())
    raw["data"]["cache_dir"] = str(tmp_path / "data")
    raw["data"]["history_days"] = 80
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    raw["strategy"] = {"name": "composite_verdict"}
    raw["portfolio_backtest"] = {
        "enabled": True,
        "portfolio": {"top_k": 2, "rebalance_n_days": 10, "max_per_industry": None},
        "eligibility": {"min_avg_amount_20d": 0, "exclude_st": False, "min_history_bars": 1},
        "staggered_starts": 1,
        "score_cache_dir": str(tmp_path / "scores"),
    }
    (tmp_path / "data").mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(cfg_path)


def test_run_portfolio_ab_happy(base_cfg):
    pool_data = _seed_panel(["A", "B", "C", "D"])
    ab_cfg = PortfolioABConfig(
        base_config="config.yaml",
        arms={
            "a": PortfolioArmOverride(strategy={"name": "composite_verdict"}),
            "b": PortfolioArmOverride(
                strategy={"name": "composite_verdict"},
                portfolio_backtest={"portfolio": {"top_k": 1}},
            ),
        },
    )
    res = run_portfolio_ab(
        ab_cfg, base_cfg, pool_data=pool_data,
        sector_map={}, name_map={c: c for c in pool_data},
    )
    assert set(res.arms) == {"a", "b"}
    assert not res.arms["a"].failed
    assert not res.arms["b"].failed
    # Different top_k → different content_hash → independent score cache files.
    a_eff = res.arms["a"].effective_cfg
    b_eff = res.arms["b"].effective_cfg
    assert a_eff.content_hash != b_eff.content_hash


def test_run_portfolio_ab_failure_isolation(base_cfg, monkeypatch):
    """If arm A's score panel computation explodes, arm B still runs."""
    pool_data = _seed_panel(["A", "B"])
    ab_cfg = PortfolioABConfig(
        base_config="config.yaml",
        arms={
            "broken": PortfolioArmOverride(strategy={"name": "composite_verdict"}),
            "good":   PortfolioArmOverride(strategy={"name": "composite_verdict"},
                                            portfolio_backtest={"portfolio": {"top_k": 1}}),
        },
    )

    # Make `precompute_scores_from_legacy` raise iff the arm's score cache
    # dir contains "broken_marker" — set up the marker by patching the
    # function to raise based on arm-specific hash.
    real = __import__("stockpool.portfolio_ab.runner", fromlist=["precompute_scores_from_legacy"])
    real_fn = real.precompute_scores_from_legacy
    call_count = {"n": 0}
    arm_a_hash = None  # will populate below

    from stockpool.portfolio_ab.config import build_effective_cfg
    arm_a_hash = build_effective_cfg(base_cfg, ab_cfg.arms["broken"]).content_hash

    def fake(*args, **kwargs):
        call_count["n"] += 1
        # First call corresponds to "broken" arm; raise.
        if call_count["n"] == 1:
            raise RuntimeError("simulated failure")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(
        "stockpool.portfolio_ab.runner.precompute_scores_from_legacy", fake,
    )
    res = run_portfolio_ab(
        ab_cfg, base_cfg, pool_data=pool_data,
        sector_map={}, name_map={c: c for c in pool_data},
    )
    assert res.arms["broken"].failed
    assert "simulated failure" in (res.arms["broken"].error or "")
    assert not res.arms["good"].failed


def test_armresult_primary_curve_empty_when_failed():
    arm = ArmResult(name="x", effective_cfg=None, failed=True, error="boom")
    assert arm.primary_curve.empty
    assert arm.primary_metrics == {}
    assert list(arm.trades) == []
