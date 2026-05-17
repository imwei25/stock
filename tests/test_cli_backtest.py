"""Smoke test for `python -m stockpool backtest`."""
from pathlib import Path

import pandas as pd
import pytest

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Seed the cache with synthetic daily data so no network call happens."""
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()

    # Build 200 days of synthetic data so weekly bars >= 30
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
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
    df.to_parquet(cache_dir / "605589_daily.parquet", index=False)
    return cache_dir


def test_backtest_cli_produces_html(tmp_path, isolated_cache, monkeypatch):
    """End-to-end: backtest CLI produces a non-trivial HTML report."""
    # Build a config pointing at the seeded cache + tmp output dir
    import yaml
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(isolated_cache)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    rc = main(["backtest", "--config", str(cfg_file), "--stocks", "605589"])
    assert rc == 0

    backtest_dir = tmp_path / "reports" / "backtest"
    latest = backtest_dir / "latest.html"
    assert latest.exists()
    assert latest.stat().st_size > 1024
    html = latest.read_text(encoding="utf-8")
    assert "N=5" in html and "N=10" in html and "N=20" in html
    assert "605589" in html
