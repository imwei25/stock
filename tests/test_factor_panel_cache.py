"""PR-2: load_or_build_factor_panel disk cache.

`<cache_dir>/factor_panels/<sig>/{manifest.json, close.parquet, <factor>.parquet}`.
Key hashes (sorted factor names, sorted universe codes, last_date). Any input
change invalidates the cache.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import stockpool.strategy_factory as sf
from stockpool.strategy_factory import load_or_build_factor_panel


def _stock_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "open": close * 0.998,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.uniform(5e5, 2e6, n),
    })


def _pool(codes: list[str]) -> dict[str, pd.DataFrame]:
    return {c: _stock_df(seed=i + 1) for i, c in enumerate(codes)}


def test_first_call_writes_manifest_and_parquets(tmp_path):
    pool = _pool(["A", "B"])
    factors = ["momentum_5", "alpha_003"]
    fp, cp = load_or_build_factor_panel(factors, pool, tmp_path)

    root = tmp_path / "factor_panels"
    sigs = list(root.iterdir())
    assert len(sigs) == 1
    sig_dir = sigs[0]
    assert (sig_dir / "manifest.json").exists()
    assert (sig_dir / "close.parquet").exists()
    for name in fp.keys():
        assert (sig_dir / f"{name}.parquet").exists()

    meta = json.loads((sig_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(meta["factors"]) == set(fp.keys())
    assert meta["n_codes"] == 2


def test_second_call_hits_cache(tmp_path, monkeypatch):
    pool = _pool(["A", "B"])
    factors = ["momentum_5"]
    load_or_build_factor_panel(factors, pool, tmp_path)

    # On second call, build_factor_panel must NOT be invoked.
    def fail(*a, **kw):
        raise AssertionError("build_factor_panel should not be called on cache hit")
    monkeypatch.setattr(sf, "build_factor_panel", fail)
    monkeypatch.setattr(sf, "build_close_panel", fail)

    fp, cp = load_or_build_factor_panel(factors, pool, tmp_path)
    assert "momentum_5" in fp
    assert not cp.empty


def test_changed_factor_list_rebuilds(tmp_path):
    pool = _pool(["A", "B"])
    load_or_build_factor_panel(["momentum_5"], pool, tmp_path)
    load_or_build_factor_panel(["momentum_5", "alpha_003"], pool, tmp_path)
    sigs = list((tmp_path / "factor_panels").iterdir())
    assert len(sigs) == 2  # 不同 sig 各自一份


def test_changed_universe_rebuilds(tmp_path):
    load_or_build_factor_panel(["momentum_5"], _pool(["A", "B"]), tmp_path)
    load_or_build_factor_panel(["momentum_5"], _pool(["A", "B", "C"]), tmp_path)
    sigs = list((tmp_path / "factor_panels").iterdir())
    assert len(sigs) == 2


def test_refresh_bypasses_cache(tmp_path, monkeypatch):
    pool = _pool(["A", "B"])
    factors = ["momentum_5"]
    load_or_build_factor_panel(factors, pool, tmp_path)

    calls = {"n": 0}
    real_bfp = sf.build_factor_panel
    real_bcp = sf.build_close_panel
    def spy_fp(*a, **kw):
        calls["n"] += 1
        return real_bfp(*a, **kw)
    monkeypatch.setattr(sf, "build_factor_panel", spy_fp)
    # close panel build is fine either way

    load_or_build_factor_panel(factors, pool, tmp_path, refresh=True)
    assert calls["n"] == 1


def test_cached_values_match_fresh(tmp_path):
    pool = _pool(["A", "B", "C"])
    factors = ["momentum_5", "alpha_003"]
    fp1, cp1 = load_or_build_factor_panel(factors, pool, tmp_path)
    fp2, cp2 = load_or_build_factor_panel(factors, pool, tmp_path)
    for name in fp1:
        pd.testing.assert_frame_equal(fp1[name], fp2[name])
    pd.testing.assert_frame_equal(cp1, cp2)


def test_empty_pool_returns_empty(tmp_path):
    fp, cp = load_or_build_factor_panel(["momentum_5"], {}, tmp_path)
    assert fp == {}
    assert cp.empty
