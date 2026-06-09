"""Unit tests for cross-sectional factor preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_panel(n_days=5, n_stocks=30, seed=0):
    """Synthetic factor panel: T × N DataFrame, dates index, codes columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    values = rng.standard_normal((n_days, n_stocks))
    return pd.DataFrame(values, index=dates, columns=codes)


def test_winsorize_clips_to_quantile():
    """Values outside [lo quantile, hi quantile] are clipped per day."""
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel(n_days=3, n_stocks=100, seed=1)
    out = winsorize_panel(df, 0.05, 0.95)
    for d in df.index:
        row = df.loc[d]
        lo_q = row.quantile(0.05)
        hi_q = row.quantile(0.95)
        assert out.loc[d].min() >= lo_q - 1e-9
        assert out.loc[d].max() <= hi_q + 1e-9


def test_winsorize_all_nan_row_passthrough():
    """A day with all-NaN cross-section is returned unchanged."""
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel(n_days=3, n_stocks=10, seed=2)
    df.iloc[1] = np.nan
    out = winsorize_panel(df, 0.01, 0.99)
    assert out.iloc[1].isna().all()
    assert out.shape == df.shape


def test_winsorize_invalid_bounds_raises():
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel()
    with pytest.raises(ValueError):
        winsorize_panel(df, 0.99, 0.01)
    with pytest.raises(ValueError):
        winsorize_panel(df, 0.5, 0.5)


def test_winsorize_preserves_index_columns():
    from stockpool.ml.preprocess import winsorize_panel
    df = _make_panel()
    out = winsorize_panel(df, 0.01, 0.99)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape


def test_cs_zscore_mean_zero_std_one():
    """After per-day cs zscore, each row has μ ≈ 0 and σ ≈ 1."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=3, n_stocks=50, seed=3)
    out = cs_zscore_panel(df)
    for d in df.index:
        row = out.loc[d].dropna()
        assert abs(row.mean()) < 1e-9
        assert abs(row.std(ddof=0) - 1.0) < 1e-9


def test_cs_zscore_constant_row_returns_zero():
    """A day where every stock has identical value → returns zeros (σ < 1e-12)."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=3, n_stocks=10, seed=4)
    df.iloc[1] = 7.5  # constant row
    out = cs_zscore_panel(df)
    assert (out.iloc[1] == 0.0).all()


def test_cs_zscore_handles_nan():
    """Partial NaN row: zscore computed on non-NaN values, NaN positions stay NaN."""
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel(n_days=2, n_stocks=10, seed=5)
    df.iloc[0, :3] = np.nan
    out = cs_zscore_panel(df)
    assert out.iloc[0, :3].isna().all()
    valid = out.iloc[0, 3:]
    assert abs(valid.mean()) < 1e-9
    assert abs(valid.std(ddof=0) - 1.0) < 1e-9


def test_cs_zscore_preserves_index_columns():
    from stockpool.ml.preprocess import cs_zscore_panel
    df = _make_panel()
    out = cs_zscore_panel(df)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape


def test_industry_neutralize_within_group_mean_zero():
    """Per-day, within each industry group, mean of values is 0."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=3, n_stocks=12, seed=6)
    # 3 industries × 4 stocks each.
    sector_map = {f"S{i:03d}": f"ind_{i // 4}" for i in range(12)}
    out = industry_neutralize_panel(df, sector_map)
    for d in df.index:
        row = out.loc[d]
        for ind in {"ind_0", "ind_1", "ind_2"}:
            members = [c for c, s in sector_map.items() if s == ind]
            assert abs(row[members].mean()) < 1e-9


def test_industry_neutralize_unknown_code_bucket():
    """Codes not in sector_map go to '_unknown_' bucket and are demeaned together."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=6, seed=7)
    sector_map = {"S000": "A", "S001": "A"}  # only 2 of 6 mapped
    out = industry_neutralize_panel(df, sector_map)
    unknown_cols = [f"S{i:03d}" for i in range(2, 6)]
    for d in df.index:
        assert abs(out.loc[d, unknown_cols].mean()) < 1e-9


def test_industry_neutralize_empty_sector_map_raises():
    """Empty sector_map raises — caller (apply_pipeline) wraps in try/skip."""
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel()
    with pytest.raises(ValueError):
        industry_neutralize_panel(df, {})


def test_industry_neutralize_preserves_index_columns():
    from stockpool.ml.preprocess import industry_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=8)
    sector_map = {f"S{i:03d}": f"ind_{i % 2}" for i in range(8)}
    out = industry_neutralize_panel(df, sector_map)
    assert (out.index == df.index).all()
    assert list(out.columns) == list(df.columns)
    assert out.shape == df.shape


def test_pipeline_all_off_returns_input():
    """All-off cfg returns a shallow copy with same values (no transform)."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    fp = {"f1": _make_panel(n_days=2, n_stocks=6),
          "f2": _make_panel(n_days=2, n_stocks=6, seed=99)}
    out = apply_preprocess_pipeline(fp, PreprocessConfig())
    assert set(out.keys()) == {"f1", "f2"}
    for k in out:
        pd.testing.assert_frame_equal(out[k], fp[k])
    # Shallow-copy semantics: dict is a fresh object, but frames are the same.
    assert out is not fp


def test_pipeline_all_on_three_steps_order():
    """Pipeline runs winsorize → zscore → neutralize in that order."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _make_panel(n_days=3, n_stocks=20, seed=10)
    fp = {"x": df}
    sector_map = {f"S{i:03d}": f"ind_{i % 4}" for i in range(20)}
    cfg = PreprocessConfig(
        winsorize=(0.01, 0.99), zscore=True, industry_neutralize=True,
        min_pool_size=0,  # disable size guard for unit-test panel
    )
    out = apply_preprocess_pipeline(fp, cfg, sector_map=sector_map)
    # After neutralize, within-industry mean = 0 each day.
    for d in df.index:
        row = out["x"].loc[d]
        for ind in {f"ind_{i}" for i in range(4)}:
            members = [c for c, s in sector_map.items() if s == ind]
            assert abs(row[members].mean()) < 1e-9


def test_pipeline_skips_neutralize_when_no_sector_map(caplog):
    """industry_neutralize=true but empty sector_map → warning, skip step."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _make_panel(n_days=2, n_stocks=10, seed=11)
    fp = {"x": df}
    cfg = PreprocessConfig(industry_neutralize=True, min_pool_size=0)  # disable size guard for unit-test panel
    import logging
    with caplog.at_level(logging.WARNING, logger="stockpool.ml.preprocess"):
        out = apply_preprocess_pipeline(fp, cfg, sector_map=None)
    # No neutralize done → values unchanged (no other step enabled).
    pd.testing.assert_frame_equal(out["x"], df)
    assert any("sector_map" in r.message.lower() for r in caplog.records)


def test_pipeline_skips_neutralize_for_fundamental_types():
    """Factors tagged 'fundamental' bypass industry_neutralize."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _make_panel(n_days=2, n_stocks=10, seed=12)
    fp = {"pe_ratio": df, "momentum_20": df.copy()}
    factor_types = {
        "pe_ratio": ("fundamental",),
        "momentum_20": ("momentum", "time_series"),
    }
    sector_map = {f"S{i:03d}": f"ind_{i % 2}" for i in range(10)}
    cfg = PreprocessConfig(industry_neutralize=True, min_pool_size=0)  # disable size guard for unit-test panel
    out = apply_preprocess_pipeline(
        fp, cfg, sector_map=sector_map, factor_types=factor_types,
    )
    # pe_ratio untouched
    pd.testing.assert_frame_equal(out["pe_ratio"], df)
    # momentum_20 demeaned per industry
    for d in df.index:
        row = out["momentum_20"].loc[d]
        for ind in {"ind_0", "ind_1"}:
            members = [c for c, s in sector_map.items() if s == ind]
            assert abs(row[members].mean()) < 1e-9


def test_pipeline_partial_steps_independent():
    """Each step independently togglable: zscore-only doesn't trigger others."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline, cs_zscore_panel
    df = _make_panel(n_days=2, n_stocks=10, seed=13)
    fp = {"x": df}
    cfg = PreprocessConfig(zscore=True, min_pool_size=0)  # disable size guard for unit-test panel
    out = apply_preprocess_pipeline(fp, cfg)
    pd.testing.assert_frame_equal(out["x"], cs_zscore_panel(df))


def test_pipeline_preserves_factor_keys_and_shapes():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    fp = {"a": _make_panel(seed=20), "b": _make_panel(seed=21), "c": _make_panel(seed=22)}
    cfg = PreprocessConfig(winsorize=(0.05, 0.95), zscore=True, min_pool_size=0)  # disable size guard for unit-test panel
    out = apply_preprocess_pipeline(fp, cfg)
    assert set(out.keys()) == {"a", "b", "c"}
    for k in out:
        assert out[k].shape == fp[k].shape


def test_is_all_off_true_for_defaults():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    assert _is_all_off(PreprocessConfig()) is True


def test_is_all_off_false_for_any_step_enabled():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    assert _is_all_off(PreprocessConfig(zscore=True)) is False
    assert _is_all_off(PreprocessConfig(winsorize=(0.01, 0.99))) is False
    assert _is_all_off(PreprocessConfig(industry_neutralize=True)) is False


def test_pipeline_skips_all_when_pool_below_min(caplog):
    """n_codes < min_pool_size triggers full skip with single warning."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline
    df = _make_panel(n_days=2, n_stocks=10, seed=20)
    fp = {"x": df}
    sector_map = {f"S{i:03d}": f"ind_{i % 2}" for i in range(10)}
    cfg = PreprocessConfig(
        winsorize=(0.01, 0.99), zscore=True, industry_neutralize=True,
        min_pool_size=200,  # explicit; default is 200 already
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="stockpool.ml.preprocess"):
        out = apply_preprocess_pipeline(
            fp, cfg, sector_map=sector_map, n_codes=10,
        )
    # All preprocess steps skipped → values pass through unchanged.
    pd.testing.assert_frame_equal(out["x"], df)
    # Single warning about size guard
    skip_warnings = [r for r in caplog.records if "min_pool_size" in r.message]
    assert len(skip_warnings) == 1


def test_pipeline_n_codes_none_bypasses_size_guard():
    """n_codes=None disables the guard (legacy / unit-test path)."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline, cs_zscore_panel
    df = _make_panel(n_days=2, n_stocks=10, seed=21)
    fp = {"x": df}
    cfg = PreprocessConfig(zscore=True, min_pool_size=200)
    # No n_codes argument → guard not consulted → transforms apply.
    out = apply_preprocess_pipeline(fp, cfg)
    pd.testing.assert_frame_equal(out["x"], cs_zscore_panel(df))


def test_pipeline_n_codes_above_min_runs_normally():
    """n_codes >= min_pool_size proceeds with full pipeline."""
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline, cs_zscore_panel
    df = _make_panel(n_days=2, n_stocks=300, seed=22)
    fp = {"x": df}
    cfg = PreprocessConfig(zscore=True, min_pool_size=200)
    out = apply_preprocess_pipeline(fp, cfg, n_codes=300)
    pd.testing.assert_frame_equal(out["x"], cs_zscore_panel(df))


# ---------------------------------------------------------------------------
# market_cap_neutralize_panel (Phase 2)
# ---------------------------------------------------------------------------

def test_mcap_neutralize_residual_orthogonal_to_logmcap():
    """After neutralize, per-day residual is ~uncorrelated with log_mcap."""
    from stockpool.ml.preprocess import market_cap_neutralize_panel
    rng = np.random.default_rng(30)
    df = _make_panel(n_days=4, n_stocks=80, seed=30)
    # log_mcap correlated structure; factor = 2*log_mcap + noise so size loads.
    log_mcap = _make_panel(n_days=4, n_stocks=80, seed=31).abs() + 5.0
    df = 2.0 * log_mcap + pd.DataFrame(
        rng.standard_normal(df.shape), index=df.index, columns=df.columns,
    )
    out = market_cap_neutralize_panel(df, log_mcap)
    for d in df.index:
        r = out.loc[d]
        m = log_mcap.loc[d]
        # residual cross-sectional mean ~0 and corr with size ~0
        assert abs(r.mean()) < 1e-9
        assert abs(np.corrcoef(r.values, m.values)[0, 1]) < 1e-6


def test_mcap_neutralize_removes_pure_size_factor_to_zero():
    """A factor that IS exactly a*log_mcap+b residualises to ~0."""
    from stockpool.ml.preprocess import market_cap_neutralize_panel
    log_mcap = _make_panel(n_days=3, n_stocks=50, seed=32).abs() + 3.0
    df = 1.7 * log_mcap + 0.5  # perfectly linear in size
    out = market_cap_neutralize_panel(df, log_mcap)
    assert (out.abs() < 1e-9).all().all()


def test_mcap_neutralize_factor_nan_stays_nan():
    """NaN factor cells remain NaN after neutralize."""
    from stockpool.ml.preprocess import market_cap_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=20, seed=33)
    log_mcap = _make_panel(n_days=2, n_stocks=20, seed=34).abs() + 2.0
    df.iloc[0, :3] = np.nan
    out = market_cap_neutralize_panel(df, log_mcap)
    assert out.iloc[0, :3].isna().all()


def test_mcap_neutralize_missing_size_passes_through():
    """A stock with NaN log_mcap keeps its raw factor value (no residual)."""
    from stockpool.ml.preprocess import market_cap_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=20, seed=35)
    log_mcap = _make_panel(n_days=2, n_stocks=20, seed=36).abs() + 2.0
    log_mcap.iloc[:, 5] = np.nan  # one stock has no size data
    out = market_cap_neutralize_panel(df, log_mcap)
    # That column unchanged (size unavailable → pass through raw factor).
    pd.testing.assert_series_equal(
        out.iloc[:, 5], df.iloc[:, 5], check_names=False,
    )


def test_mcap_neutralize_degenerate_size_falls_back_to_demean():
    """Constant log_mcap (var≈0) → plain demean (slope 0)."""
    from stockpool.ml.preprocess import market_cap_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=30, seed=37)
    log_mcap = pd.DataFrame(8.0, index=df.index, columns=df.columns)
    out = market_cap_neutralize_panel(df, log_mcap)
    for d in df.index:
        pd.testing.assert_series_equal(
            out.loc[d], df.loc[d] - df.loc[d].mean(), check_names=False,
        )


def test_mcap_neutralize_reindexes_superset_panel():
    """log_mcap covering extra rows/cols is reindexed to df."""
    from stockpool.ml.preprocess import market_cap_neutralize_panel
    df = _make_panel(n_days=2, n_stocks=10, seed=38)
    big = _make_panel(n_days=4, n_stocks=20, seed=39).abs() + 4.0
    big = big.reindex(index=big.index, columns=list(df.columns) + ["X999"])
    big.loc[:, df.columns] = (
        _make_panel(n_days=4, n_stocks=10, seed=39).abs().reindex(big.index) + 4.0
    ).values
    out = market_cap_neutralize_panel(df, big)
    assert out.shape == df.shape
    assert list(out.columns) == list(df.columns)


def test_pipeline_mcap_neutralize_skips_fundamental_and_warns(caplog):
    """market_cap_neutralize on: fundamental factors skipped; missing panel warns."""
    import logging
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import apply_preprocess_pipeline, market_cap_neutralize_panel
    df = _make_panel(n_days=3, n_stocks=300, seed=40)
    log_mcap = _make_panel(n_days=3, n_stocks=300, seed=41).abs() + 3.0
    fp = {"momentum_20": df.copy(), "pe": df.copy()}
    factor_types = {"momentum_20": ("momentum",), "pe": ("fundamental",)}
    cfg = PreprocessConfig(market_cap_neutralize=True)
    out = apply_preprocess_pipeline(
        fp, cfg, factor_types=factor_types, n_codes=300, log_mcap_panel=log_mcap,
    )
    # momentum neutralised; pe (fundamental) untouched.
    pd.testing.assert_frame_equal(
        out["momentum_20"], market_cap_neutralize_panel(df, log_mcap),
    )
    pd.testing.assert_frame_equal(out["pe"], df)

    # Missing panel → step skipped with warning.
    with caplog.at_level(logging.WARNING, logger="stockpool.ml.preprocess"):
        out2 = apply_preprocess_pipeline(
            {"x": df.copy()}, cfg, n_codes=300, log_mcap_panel=None,
        )
    pd.testing.assert_frame_equal(out2["x"], df)
    assert any("log_mcap_panel is None" in r.message for r in caplog.records)


def test_is_all_off_false_for_market_cap_neutralize():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    assert _is_all_off(PreprocessConfig(market_cap_neutralize=True)) is False


def test_is_all_off_false_for_symmetric_orthogonalize():
    from stockpool.config import PreprocessConfig
    from stockpool.ml.preprocess import _is_all_off
    assert _is_all_off(PreprocessConfig(symmetric_orthogonalize=True)) is False
