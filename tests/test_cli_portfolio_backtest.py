"""Smoke test for `python -m stockpool portfolio-backtest`."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _seed_cache(cache_dir: Path, codes: list[str], n: int = 200):
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
def base_setup(tmp_path, monkeypatch):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    codes = ["605589", "000001", "000002"]
    _seed_cache(cache_dir, codes)
    cache_last = pd.date_range("2024-01-02", periods=200, freq="B")[-1]
    monkeypatch.setattr(
        "stockpool.fetcher._today",
        lambda: pd.Timestamp(cache_last) + pd.Timedelta(days=1),
    )
    # PR-2: stub the industry map loader so no baostock/akshare call happens.
    monkeypatch.setattr(
        "stockpool.industry_map.load_or_build_industry_map",
        lambda cache_dir, source="auto": {c: f"行业{i % 2}" for i, c in enumerate(codes)},
    )

    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(cache_dir)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    raw["stocks"] = [{"code": c, "name": c, "sector": ""} for c in codes]
    # Make sure the cached fetcher is used (no network).
    raw["data"]["force_refresh"] = False
    # Use composite_verdict to avoid loading the industry map (baostock call)
    # and training ML pipelines in a smoke test.
    raw["strategy"] = {"name": "composite_verdict"}
    # PR-1 portfolio_backtest defaults; just enable + small top_k.
    raw["portfolio_backtest"] = {
        "enabled": True,
        "portfolio": {
            "top_k": 2,
            "rebalance_n_days": 20,
            "max_per_industry": None,
        },
        "staggered_starts": 1,
        "score_cache_dir": str(tmp_path / "scores"),
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_file, tmp_path


def test_portfolio_backtest_smoke(base_setup):
    cfg_file, tmp_path = base_setup
    rc = main(["portfolio-backtest", "--config", str(cfg_file)])
    assert rc == 0
    out = tmp_path / "reports" / "portfolio" / "latest.html"
    assert out.exists()
    assert out.stat().st_size > 1024
    html = out.read_text(encoding="utf-8")
    assert "Portfolio Backtest" in html
    assert "Equity Curve" in html


def test_portfolio_backtest_workers_flag(base_setup):
    """`--workers` is accepted and plumbed into precompute (serial path here)."""
    cfg_file, tmp_path = base_setup
    rc = main(["portfolio-backtest", "--config", str(cfg_file), "--workers", "1"])
    assert rc == 0
    assert (tmp_path / "reports" / "portfolio" / "latest.html").exists()


def test_portfolio_backtest_refuses_when_disabled(tmp_path, base_setup, monkeypatch):
    cfg_file, _ = base_setup
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    raw["portfolio_backtest"]["enabled"] = False
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    rc = main(["portfolio-backtest", "--config", str(cfg_file)])
    assert rc == 2


def test_portfolio_backtest_uses_universe_cache(base_setup, tmp_path):
    """When universe.parquet exists, engine loads beyond cfg.stocks."""
    cfg_file, tmp_path = base_setup
    # Add a 4th code "888888" to universe.parquet (but NOT to cfg.stocks).
    cache_dir = tmp_path / "data"
    _seed_cache(cache_dir, ["888888"])
    universe = pd.DataFrame({
        "code": ["605589", "000001", "000002", "888888"],
        "name": ["A", "B", "C", "Extra"],
        "market": ["sh", "sz", "sz", "sh"],
    })
    universe.to_parquet(cache_dir / "universe.parquet", index=False)
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    # Force a fresh score panel (different content_hash).
    raw["portfolio_backtest"]["portfolio"]["top_k"] = 3
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    rc = main(["portfolio-backtest", "--config", str(cfg_file)])
    assert rc == 0
    # Score cache file should reference 4 stocks (a 888888 column shows up).
    score_files = list((tmp_path / "scores").glob("*.parquet"))
    assert len(score_files) == 1
    sp = pd.read_parquet(score_files[0])
    assert "888888" in sp.columns


def test_portfolio_backtest_staggered_ensemble(base_setup):
    """staggered_starts=3 → ensemble HTML with k=3 cards + envelope band."""
    cfg_file, tmp_path = base_setup
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    raw["portfolio_backtest"]["staggered_starts"] = 3
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    rc = main(["portfolio-backtest", "--config", str(cfg_file)])
    assert rc == 0
    html = (tmp_path / "reports" / "portfolio" / "latest.html").read_text(encoding="utf-8")
    assert "ensemble" in html.lower()
    assert "Per-offset metrics" in html


def test_portfolio_backtest_refresh_scores_recomputes(base_setup, monkeypatch):
    cfg_file, tmp_path = base_setup
    # First run populates the cache.
    rc = main(["portfolio-backtest", "--config", str(cfg_file)])
    assert rc == 0
    score_files = list((tmp_path / "scores").glob("*.parquet"))
    assert len(score_files) == 1
    mtime_first = score_files[0].stat().st_mtime

    # Track precompute calls on the second run.
    call_count = {"n": 0}
    from stockpool.portfolio import scoring as scoring_mod
    real_fn = scoring_mod.precompute_scores_from_legacy

    def tracked(*args, **kwargs):
        call_count["n"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(
        "stockpool.cli.precompute_scores_from_legacy", tracked, raising=False,
    )
    # The CLI does `from ... import precompute_scores_from_legacy` *inside* the
    # function, so patch the source module:
    monkeypatch.setattr(
        "stockpool.portfolio.scoring.precompute_scores_from_legacy", tracked,
    )

    # Re-run WITHOUT --refresh-scores → should hit cache (call_count stays 0).
    rc = main(["portfolio-backtest", "--config", str(cfg_file)])
    assert rc == 0
    assert call_count["n"] == 0

    # Re-run WITH --refresh-scores → should recompute (call_count = 1).
    rc = main([
        "portfolio-backtest", "--config", str(cfg_file), "--refresh-scores",
    ])
    assert rc == 0
    assert call_count["n"] == 1
