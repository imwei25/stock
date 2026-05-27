"""Smoke test for `python -m stockpool portfolio-ab`."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _seed_cache(cache_dir: Path, codes, n=200):
    rng = np.random.default_rng(7)
    for code in codes:
        returns = rng.normal(0.0005, 0.02, n)
        close = 100.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "open":  close * 0.998,
            "high":  close * 1.005,
            "low":   close * 0.995,
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        })
        df.to_parquet(cache_dir / f"{code}_daily.parquet", index=False)


@pytest.fixture
def ab_setup(tmp_path, monkeypatch):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    codes = ["605589", "000001", "000002", "000003"]
    _seed_cache(cache_dir, codes)
    cache_last = pd.date_range("2024-01-02", periods=200, freq="B")[-1]
    monkeypatch.setattr(
        "stockpool.fetcher._today",
        lambda: pd.Timestamp(cache_last) + pd.Timedelta(days=1),
    )
    monkeypatch.setattr(
        "stockpool.industry_map.load_or_build_industry_map",
        lambda cache_dir, source="auto": {c: f"行业{i % 2}" for i, c in enumerate(codes)},
    )

    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(cache_dir)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    raw["stocks"] = [{"code": c, "name": c, "sector": ""} for c in codes]
    raw["data"]["force_refresh"] = False
    raw["strategy"] = {"name": "composite_verdict"}
    raw["portfolio_backtest"] = {
        "enabled": True,
        "portfolio": {"top_k": 2, "rebalance_n_days": 20, "max_per_industry": None},
        "eligibility": {"min_avg_amount_20d": 0, "exclude_st": False, "min_history_bars": 1},
        "staggered_starts": 1,
        "score_cache_dir": str(tmp_path / "scores"),
    }
    base_path = tmp_path / "config.yaml"
    base_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    ab_raw = {
        "base_config": "config.yaml",
        "arms": {
            "a_topk2": {"strategy": {"name": "composite_verdict"}},
            "b_topk1": {
                "strategy": {"name": "composite_verdict"},
                "portfolio_backtest": {"portfolio": {"top_k": 1}},
            },
        },
    }
    ab_path = tmp_path / "ab.yaml"
    ab_path.write_text(yaml.safe_dump(ab_raw), encoding="utf-8")
    return ab_path, tmp_path


def test_portfolio_ab_cli_happy(ab_setup):
    ab_path, tmp_path = ab_setup
    rc = main(["portfolio-ab", "--config", str(ab_path)])
    assert rc == 0
    out = tmp_path / "reports" / "portfolio_ab" / "latest.html"
    assert out.exists()
    assert out.stat().st_size > 1024
    html = out.read_text(encoding="utf-8")
    assert "Portfolio A/B" in html
    assert "a_topk2" in html and "b_topk1" in html


def test_portfolio_ab_cli_unknown_arm(ab_setup):
    ab_path, _ = ab_setup
    rc = main(["portfolio-ab", "--config", str(ab_path), "--arm", "doesnt_exist"])
    assert rc == 2


def test_portfolio_ab_cli_single_arm_debug(ab_setup, capsys):
    """--arm runs only that arm and prints to stdout; no HTML emitted."""
    ab_path, tmp_path = ab_setup
    rc = main(["portfolio-ab", "--config", str(ab_path), "--arm", "a_topk2"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Portfolio arm: a_topk2" in captured.out
    # No HTML in --arm mode.
    out = tmp_path / "reports" / "portfolio_ab" / "latest.html"
    assert not out.exists()


def test_portfolio_ab_cli_refuses_when_base_disabled(ab_setup):
    ab_path, tmp_path = ab_setup
    base_path = tmp_path / "config.yaml"
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    raw["portfolio_backtest"]["enabled"] = False
    base_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    rc = main(["portfolio-ab", "--config", str(ab_path)])
    assert rc == 2
