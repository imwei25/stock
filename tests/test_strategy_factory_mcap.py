"""Integration tests for log_mcap_panel wiring in build_factor_panel."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pool_data(n_stocks=300, n_days=80, seed=0):
    """Synthetic pool_data: {code: daily_df with date/open/high/low/close/volume}."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-09-01", periods=n_days, freq="B")
    pool = {}
    for i in range(n_stocks):
        code = f"{600000 + i:06d}"
        close = 10.0 + rng.standard_normal(n_days).cumsum() * 0.5
        close = np.clip(close, 1.0, None)
        df = pd.DataFrame({
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.97,
            "close": close,
            "volume": rng.integers(1e5, 1e7, size=n_days),
        })
        pool[code] = df
    return pool


def test_build_factor_panel_builds_log_mcap_when_enabled(monkeypatch, tmp_path):
    """mcap_neutralize=True triggers build_log_mcap_panel + passes to pipeline."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import build_factor_panel

    pool = _pool_data(n_stocks=300, n_days=80, seed=1)
    # Stub balance fundamentals (300 stocks × 1 quarter each)
    fake_balance = pd.DataFrame({
        "code": list(pool.keys()),
        "pubDate": pd.to_datetime(["2024-06-30"] * len(pool)),
        "statDate": pd.to_datetime(["2024-03-31"] * len(pool)),
        "totalShare": [1e9] * len(pool),
    })
    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals",
        lambda table, cache_dir=None: fake_balance,
    )

    call_log = {"log_mcap_panel": None}
    original_apply = None

    from stockpool.ml import preprocess as preproc_mod
    original_apply = preproc_mod.apply_preprocess_pipeline

    def spy_apply(factor_panel, cfg, **kwargs):
        call_log["log_mcap_panel"] = kwargs.get("log_mcap_panel")
        return original_apply(factor_panel, cfg, **kwargs)

    monkeypatch.setattr(preproc_mod, "apply_preprocess_pipeline", spy_apply)

    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
        min_pool_size=200,
    )
    result = build_factor_panel(
        ["momentum_20"], pool, preprocess_cfg=cfg, cache_dir=str(tmp_path),
    )
    assert call_log["log_mcap_panel"] is not None
    assert call_log["log_mcap_panel"].shape[1] == len(pool)
    assert "momentum_20" in result


def test_build_factor_panel_skips_log_mcap_when_disabled(monkeypatch, tmp_path):
    """mcap_neutralize=False → log_mcap_panel must be None (no balance fetch)."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import build_factor_panel

    pool = _pool_data(n_stocks=300, n_days=80, seed=2)

    # Should NOT be called at all
    fetch_calls = {"count": 0}

    def fake_loader(table, cache_dir=None):
        fetch_calls["count"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(
        "stockpool.fundamentals_loader.load_or_build_fundamentals", fake_loader,
    )

    cfg = PreprocessConfig(
        winsorize=None, zscore=True,
        industry_neutralize=False, mcap_neutralize=False,
        min_pool_size=200,
    )
    build_factor_panel(
        ["momentum_20"], pool, preprocess_cfg=cfg, cache_dir=str(tmp_path),
    )
    assert fetch_calls["count"] == 0


def test_build_factor_panel_warns_when_mcap_on_but_cache_dir_none(monkeypatch, caplog):
    """mcap_neutralize=True with cache_dir=None → warning, skips mcap (no crash)."""
    import logging
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import build_factor_panel

    pool = _pool_data(n_stocks=300, n_days=80, seed=3)
    cfg = PreprocessConfig(
        winsorize=None, zscore=False,
        industry_neutralize=False, mcap_neutralize=True,
        min_pool_size=200,
    )
    with caplog.at_level(logging.WARNING):
        result = build_factor_panel(
            ["momentum_20"], pool, preprocess_cfg=cfg, cache_dir=None,
        )
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "mcap_neutralize=True" in msgs and "cache_dir" in msgs
    assert "momentum_20" in result
