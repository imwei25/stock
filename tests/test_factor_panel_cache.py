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


def test_build_log_mcap_panel_static_broadcast(tmp_path):
    """build_log_mcap_panel = log(close × static totalShare), missing codes NaN."""
    pool = _pool(["600001", "000002", "300003"])
    snap = pd.DataFrame({
        "code": ["600001", "000002"],  # 300003 deliberately absent
        "totalShare": [1.0e9, 2.0e9],
        "pubDate": pd.to_datetime(["2026-04-30", "2026-04-30"]),
        "statDate": pd.to_datetime(["2026-03-31", "2026-03-31"]),
    })
    snap.to_parquet(tmp_path / "mcap_shares.parquet", index=False)

    out = sf.build_log_mcap_panel(pool, tmp_path)
    assert out is not None
    cp = sf.build_close_panel(pool)
    # mapped codes = log(close × shares); unmapped (300003) = all NaN.
    expected = np.log(cp["600001"] * 1.0e9)
    pd.testing.assert_series_equal(out["600001"], expected, check_names=False)
    assert out["300003"].isna().all()


def test_build_log_mcap_panel_missing_snapshot_returns_none(tmp_path):
    """No mcap_shares.parquet → None (caller skips the step)."""
    pool = _pool(["S001"])
    assert sf.build_log_mcap_panel(pool, tmp_path) is None


def test_maybe_inject_mcap_panel_noop_when_disabled(tmp_path):
    """maybe_inject_mcap_panel does nothing when market_cap_neutralize is off."""
    from stockpool.config import PreprocessConfig
    from stockpool.factors.context import set_mcap_panel, get_mcap_panel
    set_mcap_panel(None)
    try:
        sf.maybe_inject_mcap_panel(PreprocessConfig(), _pool(["S001"]), tmp_path)
        assert get_mcap_panel() is None
    finally:
        set_mcap_panel(None)


def test_build_factor_panel_picks_up_context_mcap_panel(monkeypatch):
    """market_cap_neutralize=True → build_factor_panel reads context mcap panel."""
    from stockpool.config import PreprocessConfig
    from stockpool.factors.context import set_mcap_panel, get_mcap_panel
    from stockpool.ml import preprocess as preproc_mod

    pool = _pool([f"S{i:03d}" for i in range(8)])
    # Build a matching log_mcap panel over the union of dates/codes.
    cp = sf.build_close_panel(pool)
    rng = np.random.default_rng(7)
    log_mcap = pd.DataFrame(
        rng.uniform(20, 26, cp.shape), index=cp.index, columns=cp.columns,
    )
    set_mcap_panel(log_mcap)
    try:
        captured = {}
        real = preproc_mod.apply_preprocess_pipeline

        def spy(fp, cfg, **kw):
            captured["log_mcap_panel"] = kw.get("log_mcap_panel")
            return real(fp, cfg, **kw)

        monkeypatch.setattr(preproc_mod, "apply_preprocess_pipeline", spy)
        cfg = PreprocessConfig(market_cap_neutralize=True, min_pool_size=0)
        sf.build_factor_panel(["momentum_5"], pool, preprocess_cfg=cfg)
        # The context panel was threaded into the pipeline.
        assert captured["log_mcap_panel"] is get_mcap_panel()
        assert captured["log_mcap_panel"] is not None
    finally:
        set_mcap_panel(None)


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


def test_load_or_build_factor_panel_no_mask_param(tmp_path):
    """load_or_build_factor_panel 不再接受 mask_config —
    factor panel 与 mask 解耦(2026-05-31 重构),mask 只影响标签层。"""
    from stockpool.strategy_factory import load_or_build_factor_panel
    from stockpool.config import MaskConfig
    import json

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    fp, _ = load_or_build_factor_panel(
        ["momentum_5"], pool_data, cache_dir=tmp_path,
    )
    assert "momentum_5" in fp
    panels_dir = tmp_path / "factor_panels"
    sig_dirs = list(panels_dir.iterdir())
    assert len(sig_dirs) == 1
    manifest = json.loads((sig_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    # manifest 不再写 mask_* 字段
    assert "mask_enabled" not in manifest
    assert "mask_threshold_main" not in manifest
    # mask_config kwarg 已移除 → 传它会 TypeError
    with pytest.raises(TypeError, match="mask_config"):
        load_or_build_factor_panel(
            ["momentum_5"], pool_data, cache_dir=tmp_path,
            mask_config=MaskConfig(enabled=True),
        )


def test_load_or_build_factor_panel_mask_independent_sig(tmp_path):
    """两次调用相同因子和股池 → 同一个 sig 目录(mask 状态无关,
    因 mask 已不参与 factor_panel sig 计算)。"""
    from stockpool.strategy_factory import load_or_build_factor_panel
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    load_or_build_factor_panel(["momentum_5"], pool_data, cache_dir=tmp_path)
    load_or_build_factor_panel(["momentum_5"], pool_data, cache_dir=tmp_path)
    panels_dir = tmp_path / "factor_panels"
    # 第二次是 cache hit,只有一个 sig 目录
    assert len(list(panels_dir.iterdir())) == 1


# ---------------------------------------------------------------------------
# Task 15: fundamentals refresh invalidates the factor_panel cache.
# ---------------------------------------------------------------------------


def test_fundamentals_latest_mtime_helper_none_cache_dir():
    """Helper returns None for None cache_dir."""
    from stockpool.strategy_factory import _fundamentals_latest_mtime
    assert _fundamentals_latest_mtime(None) is None


def test_fundamentals_latest_mtime_helper_no_files(tmp_path):
    """Helper returns None when no fundamentals_*.parquet files exist."""
    from stockpool.strategy_factory import _fundamentals_latest_mtime
    # tmp_path is empty
    assert _fundamentals_latest_mtime(tmp_path) is None


def test_fundamentals_latest_mtime_helper_with_files(tmp_path):
    """Helper returns ISO mtime when files exist."""
    from stockpool.strategy_factory import _fundamentals_latest_mtime
    # Write 2 dummy fundamentals parquet files
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(tmp_path / "fundamentals_profit.parquet")
    pd.DataFrame({"y": [4, 5, 6]}).to_parquet(tmp_path / "fundamentals_balance.parquet")
    iso = _fundamentals_latest_mtime(tmp_path)
    assert iso is not None
    assert isinstance(iso, str)
    # ISO datetime format
    assert "T" in iso or " " in iso


def test_factor_panel_manifest_includes_fundamentals_snapshot_date(tmp_path):
    """New build writes fundamentals_snapshot_date to manifest (None if no fundamentals)."""
    # minimal pool_data with momentum-friendly fixture
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "open": 100.0 + np.arange(80) * 0.1,
        "high": 101.0 + np.arange(80) * 0.1,
        "low": 99.0 + np.arange(80) * 0.1,
        "close": 100.0 + np.arange(80) * 0.1,
        "volume": 1e6,
    })
    pool_data = {"A": df, "B": df.copy()}

    load_or_build_factor_panel(["momentum_20"], pool_data, tmp_path)

    # Find written manifest
    panels_dir = tmp_path / "factor_panels"
    sigs = list(panels_dir.iterdir())
    assert len(sigs) == 1
    manifest = json.loads((sigs[0] / "manifest.json").read_text())
    assert "fundamentals_snapshot_date" in manifest
    # No fundamentals files in tmp_path → None
    assert manifest["fundamentals_snapshot_date"] is None


def test_factor_panel_cache_invalidated_when_fundamentals_newer(tmp_path):
    """Cache hit, then fundamentals parquet appears with newer mtime → rebuild."""
    import os
    import time

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "open": 100.0 + np.arange(80) * 0.1,
        "high": 101.0 + np.arange(80) * 0.1,
        "low": 99.0 + np.arange(80) * 0.1,
        "close": 100.0 + np.arange(80) * 0.1,
        "volume": 1e6,
    })
    pool_data = {"A": df, "B": df.copy()}

    # First build — cache empty, will write to disk
    load_or_build_factor_panel(["momentum_20"], pool_data, tmp_path)

    panels_dir = tmp_path / "factor_panels"
    sigs = list(panels_dir.iterdir())
    sig_dir = sigs[0]
    manifest_path = sig_dir / "manifest.json"
    manifest_before = json.loads(manifest_path.read_text())
    built_at_before = manifest_before["built_at"]

    # Now write a fundamentals parquet with a NEWER mtime
    time.sleep(0.05)  # ensure mtime differs
    fund_path = tmp_path / "fundamentals_profit.parquet"
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(fund_path)
    # Force mtime to be later than manifest's built_at by setting to now+1
    new_time = time.time() + 1.0
    os.utime(fund_path, (new_time, new_time))

    # Second build — should detect newer fundamentals mtime and rebuild.
    # Spy on build_factor_panel to confirm it's actually invoked again
    # (built_at timestamps may collide at sub-ms resolution on fast machines).
    calls = {"n": 0}
    real_bfp = sf.build_factor_panel

    def spy_fp(*a, **kw):
        calls["n"] += 1
        return real_bfp(*a, **kw)

    import unittest.mock as _mock
    with _mock.patch.object(sf, "build_factor_panel", spy_fp):
        load_or_build_factor_panel(["momentum_20"], pool_data, tmp_path)

    assert calls["n"] == 1, "rebuild must invoke build_factor_panel"

    manifest_after = json.loads(manifest_path.read_text())
    # New manifest should record the new fundamentals_snapshot_date (no longer None)
    assert manifest_after["fundamentals_snapshot_date"] is not None
    # Sanity: the snapshot date in the rebuilt manifest reflects the new mtime
    # (was None before since no fundamentals files existed at first build).
    assert manifest_before["fundamentals_snapshot_date"] is None


# ---------------------------------------------------------------------------
# Task 7: preprocess_cfg wired into _factor_panel_sig and build_factor_panel
# ---------------------------------------------------------------------------


def test_cache_sig_all_off_backwards_compat():
    """Default PreprocessConfig sig matches pre-PR baseline (preprocess=None in dict)."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    pool = _pool(["S001"])
    sig_no_arg, _ = _factor_panel_sig(["momentum_20"], pool)
    sig_default, _ = _factor_panel_sig(["momentum_20"], pool, preprocess_cfg=PreprocessConfig())
    assert sig_no_arg == sig_default


def test_cache_sig_with_preprocess_isolated_from_baseline():
    """Enabling preprocess changes the sig (cache key)."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    pool = _pool(["S001"])
    sig_off, _ = _factor_panel_sig(["momentum_20"], pool, preprocess_cfg=PreprocessConfig())
    sig_on, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(zscore=True),
    )
    assert sig_off != sig_on


def test_cache_invalidates_on_preprocess_change():
    """Two different preprocess settings produce distinct sigs."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    pool = _pool(["S001"])
    sig_a, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(zscore=True),
    )
    sig_b, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(winsorize=(0.01, 0.99)),
    )
    assert sig_a != sig_b


def test_cache_sig_changes_on_symmetric_orthogonalize():
    """Flipping symmetric_orthogonalize yields a distinct factor-panel sig."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import _factor_panel_sig
    pool = _pool(["S001"])
    sig_off, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(zscore=True),
    )
    sig_on, _ = _factor_panel_sig(
        ["momentum_20"], pool,
        preprocess_cfg=PreprocessConfig(zscore=True, symmetric_orthogonalize=True),
    )
    assert sig_off != sig_on


def test_build_factor_panel_passes_preprocess(monkeypatch):
    """build_factor_panel routes preprocess_cfg through apply_preprocess_pipeline."""
    from stockpool.config import PreprocessConfig
    from stockpool import strategy_factory
    from stockpool.ml import preprocess as preproc_mod

    pool = _pool(["S001", "S002"])

    called = {}
    real_apply = preproc_mod.apply_preprocess_pipeline

    def spy(fp, cfg, sector_map=None, factor_types=None, n_codes=None, log_mcap_panel=None):
        called["cfg"] = cfg
        called["n_factors"] = len(fp)
        return real_apply(fp, cfg, sector_map=sector_map, factor_types=factor_types, n_codes=n_codes)

    monkeypatch.setattr(preproc_mod, "apply_preprocess_pipeline", spy)

    # Spy is set on the module; build_factor_panel imports inside the function
    # so it'll see the patched version.
    cfg = PreprocessConfig(zscore=True)
    strategy_factory.build_factor_panel(["momentum_20"], pool, preprocess_cfg=cfg)
    assert called["cfg"] is cfg
    assert called["n_factors"] == 1

    # All-off should NOT invoke the pipeline (short-circuit before call).
    called.clear()
    strategy_factory.build_factor_panel(
        ["momentum_20"], pool, preprocess_cfg=PreprocessConfig(),
    )
    assert called == {}


def test_load_or_build_factor_panel_threads_preprocess(tmp_path):
    """When preprocess_cfg is on, two calls produce different sig dirs."""
    from stockpool.config import PreprocessConfig
    from stockpool.strategy_factory import load_or_build_factor_panel
    pool = _pool(["S001", "S002"])

    # Off
    fp_off, _ = load_or_build_factor_panel(
        ["momentum_20"], pool, str(tmp_path),
        preprocess_cfg=PreprocessConfig(),
    )
    # On
    fp_on, _ = load_or_build_factor_panel(
        ["momentum_20"], pool, str(tmp_path),
        preprocess_cfg=PreprocessConfig(zscore=True),
    )
    # Two distinct sig dirs created
    sig_dirs = list((tmp_path / "factor_panels").iterdir())
    assert len(sig_dirs) == 2


def test_build_factor_panel_passes_n_codes_to_pipeline(monkeypatch):
    """build_factor_panel forwards n_codes=len(pool_data) so the size guard
    activates downstream."""
    from stockpool.config import PreprocessConfig
    from stockpool import strategy_factory
    from stockpool.ml import preprocess as preproc_mod

    pool = _pool(["S001", "S002", "S003"])

    captured = {}
    real = preproc_mod.apply_preprocess_pipeline

    def spy(fp, cfg, sector_map=None, factor_types=None, n_codes=None, log_mcap_panel=None):
        captured["n_codes"] = n_codes
        return real(fp, cfg, sector_map=sector_map,
                    factor_types=factor_types, n_codes=n_codes)

    monkeypatch.setattr(preproc_mod, "apply_preprocess_pipeline", spy)

    cfg = PreprocessConfig(zscore=True, min_pool_size=0)  # guard off so spy returns transformed values
    strategy_factory.build_factor_panel(["momentum_20"], pool, preprocess_cfg=cfg)
    assert captured["n_codes"] == 3, (
        f"build_factor_panel should pass n_codes=len(pool_data)=3, got {captured['n_codes']}"
    )
