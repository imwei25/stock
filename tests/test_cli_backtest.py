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
    # Pin today to 1 day after the last cached bar so the staleness check passes.
    cache_last = pd.date_range("2024-01-02", periods=200, freq="B")[-1]
    fresh_today = pd.Timestamp(cache_last) + pd.Timedelta(days=1)
    monkeypatch.setattr("stockpool.fetcher._today", lambda: fresh_today)

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
    # Assert against the actual configured holding-day caps rather than
    # hard-coded values, so config tuning doesn't break this smoke test.
    for N in raw["backtest"]["equity_curve_holding_days"]:
        assert f"N={N}" in html
    assert "605589" in html


def test_backtest_continues_after_per_stock_failure(tmp_path, isolated_cache, monkeypatch):
    """A mid-loop stock failure must not abort the run; warning is logged."""
    cache_last = pd.date_range("2024-01-02", periods=200, freq="B")[-1]
    fresh_today = pd.Timestamp(cache_last) + pd.Timedelta(days=1)
    monkeypatch.setattr("stockpool.fetcher._today", lambda: fresh_today)

    import yaml
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(isolated_cache)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    raw["stocks"] = [
        {"code": "605589", "name": "Cached", "sector": ""},
        {"code": "000001", "name": "Missing", "sector": ""},
    ]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    # Patch backtest_runner.fetch_daily: serve 605589 from disk, raise for others.
    real_read = pd.read_parquet
    cached_codes = {"605589"}
    def _selective_fetch(code, *a, **kw):
        if code in cached_codes:
            return real_read(isolated_cache / f"{code}_daily.parquet")
        raise RuntimeError("network disabled in test")
    monkeypatch.setattr("stockpool.backtest_runner.fetch_daily", _selective_fetch)

    rc = main(["backtest", "--config", str(cfg_file)])
    assert rc == 0  # cached stock succeeded → report renders
    latest = tmp_path / "reports" / "backtest" / "latest.html"
    assert latest.exists()
    html = latest.read_text(encoding="utf-8")
    assert "605589" in html
    assert "000001" not in html  # failed stock not in report


def test_backtest_stocks_trims_warmup_bars(monkeypatch):
    """When cfg.data.warmup_days > 0, daily passed to simulate_* is trimmed."""
    import numpy as np
    from stockpool.config import load_config, StrategyConfig
    from stockpool import backtest_runner

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    # Force composite_verdict so simulate_equity_curve is the code path taken
    # (ml_factor requires a sector_map and full pool context which we don't have here).
    cfg = cfg.model_copy(update={
        "data": cfg.data.model_copy(update={"warmup_days": 100}),
        "strategy": StrategyConfig(name="composite_verdict"),
    })

    captured_lens = []
    real_sim = backtest_runner.simulate_equity_curve
    def spy_sim(wf, **kw):
        captured_lens.append(len(wf))
        return real_sim(wf, **kw)
    monkeypatch.setattr(backtest_runner, "simulate_equity_curve", spy_sim)

    dates = pd.date_range("2022-01-01", periods=600, freq="B")
    close = 100 + np.cumsum(np.random.default_rng(0).standard_normal(600))
    pool_data = {
        cfg.stocks[0].code: pd.DataFrame({
            "date": dates, "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": [1_000_000] * 600,
        }),
    }

    per_stock, failed = backtest_runner.backtest_stocks(
        cfg, cfg.stocks[:1], pool_data, None, shared_cache={}, refresh=False,
    )
    assert len(per_stock) == 1, f"expected 1 success, got {failed}"
    # daily was trimmed from 600 to 500 (600 - warmup_days=100) before walk_forward_verdicts.
    # wf may further drop indicator warmup; just verify it's <= 500.
    assert captured_lens, "simulate_equity_curve was not called"
    assert captured_lens[0] <= 500, (
        f"simulate got wf len {captured_lens[0]} — expected ≤ 500 after warmup trim"
    )
