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


def test_score_memo_shares_panel_across_arms(base_cfg, monkeypatch):
    """Two arms with identical scoring config (differ only in top_k) compute the
    score panel ONCE — the in-process memo shares it even under refresh_scores
    (which bypasses the on-disk cache)."""
    pool_data = _seed_panel(["A", "B", "C", "D"])
    ab_cfg = PortfolioABConfig(
        base_config="config.yaml",
        arms={
            "k2": PortfolioArmOverride(strategy={"name": "composite_verdict"}),
            "k1": PortfolioArmOverride(
                strategy={"name": "composite_verdict"},
                portfolio_backtest={"portfolio": {"top_k": 1}},
            ),
        },
    )
    import stockpool.portfolio_ab.runner as R
    real_fn = R.precompute_scores_from_legacy
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real_fn(*a, **k)

    monkeypatch.setattr(R, "precompute_scores_from_legacy", counting)
    res = run_portfolio_ab(
        ab_cfg, base_cfg, pool_data=pool_data,
        sector_map={}, name_map={c: c for c in pool_data},
        refresh_scores=True,  # bypass disk cache → memo is the only sharing path
    )
    assert not res.arms["k2"].failed and not res.arms["k1"].failed
    assert calls["n"] == 1, f"expected 1 shared precompute across arms, got {calls['n']}"


def test_armresult_primary_curve_empty_when_failed():
    arm = ArmResult(name="x", effective_cfg=None, failed=True, error="boom")
    assert arm.primary_curve.empty
    assert arm.primary_metrics == {}
    assert list(arm.trades) == []


def _build_minimal_ab(base_cfg, tmp_path):
    """Build a 2-arm PortfolioABConfig + companion data for parity tests.

    Uses the same pool_data as test_run_portfolio_ab_happy so both arms are
    deterministic composite_verdict arms with different top_k only — the
    score panel cache key differs between arms (different content_hash) but
    each arm's serial vs parallel result must be bitwise equal.

    score_cache_dir is forced to ``tmp_path / "scores"`` so that the
    serial run writes the parquet files and the parallel run reads them
    (refresh_scores=False), guaranteeing identical inputs to PortfolioEngine.
    """
    import copy
    import yaml

    # Rebuild base_cfg with score_cache_dir under tmp_path so caches land
    # in the right place.  The fixture already sets score_cache_dir, so we
    # can use base_cfg directly — both the serial and parallel call share the
    # same object reference, which is what we want.
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
    sector_map: dict = {}
    name_map = {c: c for c in pool_data}
    return ab_cfg, pool_data, sector_map, name_map


def test_parallel_arms_matches_serial(base_cfg, tmp_path):
    """parallel_arms=True should yield metrics + equity curves identical to serial.

    PR-T1.4: Both arms run concurrently via ProcessPoolExecutor.  Sub-ULP FP
    drift from subprocess BLAS state is absorbed by rtol=1e-12 (well below
    cost/slippage noise).

    Strategy: run serial first so score parquet caches are written, then run
    parallel with refresh_scores=False so each subprocess reads from parquet
    rather than recomputing, ensuring identical inputs → identical outputs.
    """
    import numpy as np

    ab_cfg, pool_data, sector_map, name_map = _build_minimal_ab(base_cfg, tmp_path)

    # Serial run — writes score parquet files under base_cfg.score_cache_dir.
    res_serial = run_portfolio_ab(
        ab_cfg, base_cfg, pool_data=pool_data,
        sector_map=sector_map, name_map=name_map,
        parallel_arms=False,
    )
    # Sanity: serial run succeeded for both arms.
    for arm_name in ab_cfg.arms:
        assert not res_serial.arms[arm_name].failed, (
            f"serial arm '{arm_name}' unexpectedly failed: "
            f"{res_serial.arms[arm_name].error}"
        )

    # Parallel run — reads score parquet files written above (refresh_scores=False).
    res_parallel = run_portfolio_ab(
        ab_cfg, base_cfg, pool_data=pool_data,
        sector_map=sector_map, name_map=name_map,
        parallel_arms=True,
        refresh_scores=False,
    )
    # Sanity: parallel run succeeded for both arms.
    for arm_name in ab_cfg.arms:
        assert not res_parallel.arms[arm_name].failed, (
            f"parallel arm '{arm_name}' failed: "
            f"{res_parallel.arms[arm_name].error}"
        )

    # Metric parity — allow rtol=1e-12 to absorb any sub-ULP FP drift.
    for arm_name in ab_cfg.arms:
        m_s = res_serial.arms[arm_name].primary_metrics
        m_p = res_parallel.arms[arm_name].primary_metrics
        assert set(m_s) == set(m_p), (
            f"arm '{arm_name}' metric keys differ: serial={set(m_s)} parallel={set(m_p)}"
        )
        for k in m_s:
            v_s, v_p = m_s[k], m_p[k]
            if isinstance(v_s, float) and np.isnan(v_s):
                assert isinstance(v_p, float) and np.isnan(v_p), (
                    f"arm '{arm_name}' metric '{k}': serial=NaN but parallel={v_p}"
                )
            else:
                assert np.isclose(v_s, v_p, rtol=1e-12, atol=0), (
                    f"arm '{arm_name}' metric '{k}': serial={v_s!r} vs parallel={v_p!r}"
                )

    # Equity-curve parity.
    for arm_name in ab_cfg.arms:
        curve_s = res_serial.arms[arm_name].primary_curve["equity"].values
        curve_p = res_parallel.arms[arm_name].primary_curve["equity"].values
        np.testing.assert_allclose(
            curve_s, curve_p,
            rtol=1e-12, atol=0, equal_nan=True,
            err_msg=f"arm '{arm_name}' equity curve differs between serial and parallel",
        )
