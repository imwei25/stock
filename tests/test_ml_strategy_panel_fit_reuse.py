"""PR-1 速度优化:`_try_fit` pooled 分支在拿到预算好的 factor_panel + close_panel
时,走快路径 (`_build_pooled_xy_from_panel`),不再每个 refit_bar 重算因子。

测试覆盖:
  * 快路径与慢路径(legacy `build_panel`)产出的 (X, y) 训练集等价(同一 shape,
    同一 index 集合,数值近似相等,只考虑 dropna 后的样本)。
  * 没注入 close_panel 时 _try_fit 自动回退到慢路径。
  * close_panel 注入后 with_stock 仍能透传。
  * panel 注入后调 generate_signals 不抛异常,且与不注入 panel 跑出来的 signals
    数量一致 —— 行为不退化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.backtesting.strategies import MLFactorStrategy
from stockpool.config import (
    MaskConfig, MLFactorConfig, PreprocessConfig, SelectorConfig, WeighterConfig,
)
from stockpool.ml.dataset import build_panel, compute_factor_panel
from stockpool.strategy_factory import build_close_panel, build_factor_panel


def _stock_df(close: list[float], seed: int = 0) -> pd.DataFrame:
    n = len(close)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "open": [c * 0.998 for c in close],
        "high": [c * 1.005 for c in close],
        "low": [c * 0.995 for c in close],
        "close": close,
        "volume": rng.uniform(5e5, 2e6, n),
    })


def _make_pool(n_bars: int = 120, n_stocks: int = 4) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    return {
        chr(ord("A") + i): _stock_df(
            list(100 + np.cumsum(rng.standard_normal(n_bars))), seed=i + 1,
        )
        for i in range(n_stocks)
    }


def _make_panels(pool: dict[str, pd.DataFrame], factors: list[str]):
    per_stock = {c: d.set_index(pd.to_datetime(d["date"])).sort_index()
                 for c, d in pool.items()}
    dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(dates, name="date")
    ohlcv_panel = {
        f: pd.DataFrame({c: d[f].reindex(idx) for c, d in per_stock.items()}, index=idx)
        for f in ("open", "high", "low", "close", "volume")
    }
    factor_panel = compute_factor_panel(ohlcv_panel, factors)
    close_panel = build_close_panel(pool)
    return factor_panel, close_panel


def _cfg(factors: list[str], train_window: int = 40) -> MLFactorConfig:
    return MLFactorConfig(
        factors=factors, horizon=3, train_window=train_window,
        min_train_samples=20, refit_every=10, panel_mode="pooled",
        embargo_days=0, share_pool_fit=True,
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )


def test_fast_path_equivalent_to_legacy_build_panel():
    """快路径与 legacy build_panel 在相同 cutoff 下产出的训练集等价。"""
    factors = ["momentum_5", "alpha_003"]
    pool = _make_pool(n_bars=120)
    factor_panel, close_panel = _make_panels(pool, factors)
    cfg = _cfg(factors)

    # 选 host="A",cutoff_bar 选 90(中段,保证 label_end 落在有效区间)
    host_daily = pool["A"]
    current_bar = 90

    # 快路径
    strat_fast = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel, close_panel=close_panel,
    )
    X_fast, y_fast = strat_fast._build_pooled_xy_from_panel(host_daily, current_bar)

    # 慢路径:用同一 strategy 但去掉 close_panel,触发 build_panel
    from stockpool.ml.dataset import build_panel
    strat_slow = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
    )
    current_date = host_daily["date"].iloc[current_bar]
    pool_trunc = strat_slow._build_truncated_pool(host_daily, current_date, current_bar)
    X_slow, y_slow = build_panel(pool_trunc, factors, cfg.horizon)
    if len(X_slow) > 0 and cfg.train_window > 0:
        X_slow = X_slow.groupby(level="stock", group_keys=False, sort=False).tail(cfg.train_window)
        y_slow = y_slow.loc[X_slow.index]

    # 同 shape
    assert X_fast.shape == X_slow.shape, f"fast={X_fast.shape}, slow={X_slow.shape}"
    # 同 index 集合(可能 row order 略有差异,排序后比对)
    assert set(X_fast.index) == set(X_slow.index)
    # 数值上等价(按 index 排序后)
    X_fast_s = X_fast.sort_index()
    X_slow_s = X_slow.sort_index()
    pd.testing.assert_frame_equal(
        X_fast_s, X_slow_s, check_like=True, atol=1e-9, rtol=1e-9,
    )
    y_fast_s = y_fast.sort_index()
    y_slow_s = y_slow.sort_index()
    pd.testing.assert_series_equal(
        y_fast_s, y_slow_s, check_names=False, atol=1e-9, rtol=1e-9,
    )


def test_try_fit_falls_back_when_close_panel_missing(monkeypatch):
    """没注入 close_panel → _try_fit 走 build_panel 慢路径,不抛异常。"""
    factors = ["momentum_5"]
    pool = _make_pool(n_bars=120, n_stocks=3)
    factor_panel, _ = _make_panels(pool, factors)
    cfg = _cfg(factors)

    strat = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel, close_panel=None,
    )
    # spy: build_panel 必须被调用
    calls = {"n": 0}
    import stockpool.backtesting.strategies as mod
    real_build = mod.build_panel
    def spy(*a, **kw):
        calls["n"] += 1
        return real_build(*a, **kw)
    monkeypatch.setattr(mod, "build_panel", spy)

    out = strat._try_fit(
        pool["A"], strat._build_x_full(pool["A"]),
        mod.forward_return(pool["A"], cfg.horizon), 80,
    )
    assert out is not None
    assert calls["n"] >= 1


def test_try_fit_uses_fast_path_when_both_panels_set(monkeypatch):
    """注入 close_panel + factor_panel → _try_fit 不再调用 build_panel。"""
    factors = ["momentum_5"]
    pool = _make_pool(n_bars=120, n_stocks=3)
    factor_panel, close_panel = _make_panels(pool, factors)
    cfg = _cfg(factors)

    strat = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel, close_panel=close_panel,
    )
    import stockpool.backtesting.strategies as mod
    calls = {"n": 0}
    def spy(*a, **kw):
        calls["n"] += 1
        raise AssertionError("build_panel should NOT be called on fast path")
    monkeypatch.setattr(mod, "build_panel", spy)

    out = strat._try_fit(
        pool["A"], strat._build_x_full(pool["A"]),
        mod.forward_return(pool["A"], cfg.horizon), 80,
    )
    assert out is not None
    assert calls["n"] == 0


def test_with_stock_propagates_close_panel():
    factors = ["momentum_5"]
    pool = _make_pool(n_bars=80, n_stocks=3)
    factor_panel, close_panel = _make_panels(pool, factors)
    cfg = _cfg(factors)

    strat = MLFactorStrategy(
        cfg=cfg, pool_data=pool, factor_panel=factor_panel,
        close_panel=close_panel, current_stock_code="A",
    )
    strat_b = strat.with_stock("B")
    assert strat_b._close_panel is not None
    pd.testing.assert_frame_equal(strat_b._close_panel, close_panel)


def test_prestack_cache_equivalent_to_per_call_stack():
    """shared_cache 路径(``_ensure_pooled_xy_long`` 一次性 stack)与每次 refit
    重新切片+stack 的旧路径在多个 cutoff 上产出 bitwise 相同的训练集。

    防止后续重构悄悄破坏快/慢路径等价契约 (PR-3 引入)。
    """
    factors = ["momentum_5", "alpha_003"]
    pool = _make_pool(n_bars=120)
    factor_panel, close_panel = _make_panels(pool, factors)
    cfg = _cfg(factors)

    host_daily = pool["A"]
    shared = {}
    strat_fast = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel, close_panel=close_panel,
        shared_cache=shared,
    )
    strat_legacy = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel, close_panel=close_panel,
        shared_cache=None,
    )

    # 多个 cutoff 都应等价 —— 早段/中段/末段
    for current_bar in (60, 90, 110):
        X_f, y_f = strat_fast._build_pooled_xy_from_panel(host_daily, current_bar)
        X_l, y_l = strat_legacy._build_pooled_xy_from_panel(host_daily, current_bar)
        assert X_f.shape == X_l.shape, f"bar={current_bar}: {X_f.shape} vs {X_l.shape}"
        assert set(X_f.index) == set(X_l.index)
        X_f_s = X_f.sort_index(); X_l_s = X_l.sort_index()
        pd.testing.assert_frame_equal(
            X_f_s, X_l_s, check_like=True, atol=0.0, rtol=0.0,
        )
        pd.testing.assert_series_equal(
            y_f.sort_index(), y_l.sort_index(),
            check_names=False, atol=0.0, rtol=0.0,
        )

    # shared_cache 被 seed 了 pre-stacked panel
    keys = [k for k in shared.keys() if isinstance(k, tuple) and k[0] == "__pooled_xy_long__"]
    assert len(keys) == 1, f"expected 1 pre-stack cache entry, got {keys}"


def test_prestack_cache_skips_redundant_stack_calls(monkeypatch):
    """shared_cache 拿到 pre-stacked panel 后,后续 refit 不再调用 stack_panel_to_xy。"""
    factors = ["momentum_5"]
    pool = _make_pool(n_bars=120, n_stocks=4)
    factor_panel, close_panel = _make_panels(pool, factors)
    cfg = _cfg(factors)

    shared = {}
    strat = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel, close_panel=close_panel,
        shared_cache=shared,
    )

    import stockpool.backtesting.strategies as mod
    calls = {"n": 0}
    real_stack = mod.stack_panel_to_xy
    def spy(*a, **kw):
        calls["n"] += 1
        return real_stack(*a, **kw)
    monkeypatch.setattr(mod, "stack_panel_to_xy", spy)

    # 三次连续调用,不同 cutoff;stack 应只发生 1 次。
    for bar in (60, 80, 100):
        strat._build_pooled_xy_from_panel(pool["A"], bar)
    assert calls["n"] == 1, f"expected 1 stack call, got {calls['n']}"


def _stock_df_with_limit_up(seed: int, n_bars: int, hit_bars: tuple[int, ...]) -> pd.DataFrame:
    """Generate OHLCV with one or more limit-up days at given iloc positions."""
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal(n_bars) * 0.02
    close = 100 * np.exp(np.cumsum(rets))
    for hit in hit_bars:
        if hit + 1 < n_bars:
            close[hit] = close[hit - 1] * 1.099
            close[hit + 1] = close[hit] * 0.998
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n_bars, freq="B"),
        "open":  close * 0.998,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": rng.uniform(5e5, 2e6, n_bars),
    })


def test_fast_fallback_applies_mask_when_enabled():
    """Bug A regression: ``_build_pooled_xy_from_panel`` 的 fallback(无 shared_cache)
    必须和 pre-stack 路径以及 legacy build_panel 一样应用 mask。

    历史背景:commit 1582b52 只在 ``_ensure_pooled_xy_long`` 加了 mask,fallback 漏掉。
    导致 shared_cache=None 时(单元测试 + 任何不传 shared_cache 的 caller)
    fallback 静默忽略 ``cfg.mask``,与生产 fast path 输出不一致。
    """
    factors = ["momentum_5"]
    # Put limit-up days INSIDE the train_window tail so mask actually bites.
    # current_bar=90, horizon=3, embargo=0 → label_cutoff iloc=86, tail(40)=[47..86].
    pool = {
        f"S{i:02d}": _stock_df_with_limit_up(seed=i + 1, n_bars=120, hit_bars=(60, 75))
        for i in range(5)
    }
    factor_panel, close_panel = _make_panels(pool, factors)
    cfg = MLFactorConfig(
        factors=factors, horizon=3, train_window=40,
        min_train_samples=10, refit_every=10, panel_mode="pooled",
        embargo_days=0, share_pool_fit=True,
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
        mask=MaskConfig(enabled=True, min_listing_days=0),
    )
    host_daily = pool["S00"]
    current_bar = 90

    # Fast pre-stack (with shared_cache → mask applied)
    strat_pre = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="S00",
        factor_panel=factor_panel, close_panel=close_panel, shared_cache={},
    )
    X_pre, y_pre = strat_pre._build_pooled_xy_from_panel(host_daily, current_bar)

    # Fast fallback (without shared_cache → must still apply mask)
    strat_fb = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="S00",
        factor_panel=factor_panel, close_panel=close_panel, shared_cache=None,
    )
    X_fb, y_fb = strat_fb._build_pooled_xy_from_panel(host_daily, current_bar)

    # Legacy build_panel with mask
    strat_legacy = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="S00",
    )
    cur_date = host_daily["date"].iloc[current_bar]
    pool_t = strat_legacy._build_truncated_pool(host_daily, cur_date, current_bar)
    X_l, y_l = build_panel(pool_t, factors, cfg.horizon, mask_config=cfg.mask)
    if len(X_l) > 0 and cfg.train_window > 0:
        X_l = X_l.groupby(level="stock", group_keys=False, sort=False).tail(cfg.train_window)
        y_l = y_l.loc[X_l.index]

    # Normalize index order: pre-stack returns (date, stock), legacy returns (stock, date).
    def norm(X, y):
        if X.index.names[0] == "date":
            X = X.swaplevel("date", "stock")
            y = y.swaplevel("date", "stock")
        return X.sort_index(), y.sort_index()

    Xp, yp = norm(X_pre, y_pre)
    Xf, yf = norm(X_fb,  y_fb)
    Xl, yl = norm(X_l,   y_l)

    # All three should have identical row sets.
    assert set(Xp.index) == set(Xl.index), "pre-stack vs legacy diverge"
    assert set(Xf.index) == set(Xp.index), (
        f"fallback skipped mask: {len(set(Xf.index) ^ set(Xp.index))} rows differ"
    )
    pd.testing.assert_frame_equal(Xf, Xp, atol=0.0, rtol=0.0)
    pd.testing.assert_series_equal(yf, yp, check_names=False, atol=0.0, rtol=0.0)


def test_legacy_matches_fast_when_share_pool_fit_false():
    """Bug B regression: 非共享模式下 ``_build_truncated_pool`` 必须也 drop host,
    与 ``_build_pooled_xy_from_panel`` 一致。

    历史 legacy 通过 ``out[host_key] = daily_df.iloc[:host_slice_end]`` 把 host 重新
    塞回去,但 fast path 直接 drop host;两路径在非共享下输出不一致。Bug B 已对齐到
    "两路径都 drop host"(非共享语义本意:训练完全不见 host)。
    """
    factors = ["momentum_5"]
    pool = _make_pool(n_bars=120, n_stocks=5)
    factor_panel, close_panel = _make_panels(pool, factors)
    cfg = MLFactorConfig(
        factors=factors, horizon=3, train_window=40,
        min_train_samples=10, refit_every=10, panel_mode="pooled",
        embargo_days=0, share_pool_fit=False,
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )
    host_daily = pool["A"]
    current_bar = 90

    # Fast path (drops host)
    strat_fast = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel, close_panel=close_panel, shared_cache={},
    )
    X_fast, y_fast = strat_fast._build_pooled_xy_from_panel(host_daily, current_bar)

    # Legacy (must also drop host after Bug B fix)
    strat_legacy = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
    )
    cur_date = host_daily["date"].iloc[current_bar]
    pool_t = strat_legacy._build_truncated_pool(host_daily, cur_date, current_bar)
    assert "A" not in pool_t, "host must not be in legacy truncated pool when not sharing"
    X_l, y_l = build_panel(pool_t, factors, cfg.horizon)
    if len(X_l) > 0 and cfg.train_window > 0:
        X_l = X_l.groupby(level="stock", group_keys=False, sort=False).tail(cfg.train_window)
        y_l = y_l.loc[X_l.index]

    def norm(X, y):
        if X.index.names[0] == "date":
            X = X.swaplevel("date", "stock")
            y = y.swaplevel("date", "stock")
        return X.sort_index(), y.sort_index()

    Xf, yf = norm(X_fast, y_fast)
    Xl, yl = norm(X_l, y_l)
    assert "A" not in Xf.index.get_level_values("stock"), "fast must not include host"
    assert "A" not in Xl.index.get_level_values("stock"), "legacy must not include host"
    assert set(Xf.index) == set(Xl.index)
    pd.testing.assert_frame_equal(Xf, Xl, atol=0.0, rtol=0.0)
    pd.testing.assert_series_equal(yf, yl, check_names=False, atol=0.0, rtol=0.0)


def test_preprocess_applied_consistently_across_paths():
    """Bug C regression: preprocess (zscore) 必须在 fast pre-stack / fast fallback /
    legacy ``build_panel`` 三路径下产出相同 (X, y)。

    历史背景:``ml/dataset.py:build_panel`` 不接 preprocess_cfg,只算 raw 因子;
    ``prepare_pool`` 把 preprocess 包进 ``factor_panel`` 灌给 fast path。
    当用户开 preprocess 时 fast 用预处理值、legacy 用 raw 值,语义分裂。
    现在 ``build_panel`` 也接 preprocess_cfg → 三路径对齐。
    """
    factors = ["momentum_5", "rsi_centered_14"]
    pool = _make_pool(n_bars=120, n_stocks=10)  # ≥ default min_pool_size? no, but we override
    preprocess = PreprocessConfig(zscore=True, min_pool_size=1)
    # Fast path inputs: preprocessed factor_panel (as prepare_pool builds it).
    factor_panel_processed = build_factor_panel(
        factors, pool, preprocess_cfg=preprocess,
    )
    close_panel = build_close_panel(pool)
    cfg = MLFactorConfig(
        factors=factors, horizon=3, train_window=40,
        min_train_samples=10, refit_every=10, panel_mode="pooled",
        embargo_days=0, share_pool_fit=True,
        preprocess=preprocess,
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )
    host_daily = pool["A"]
    current_bar = 90

    # Fast pre-stack (shared_cache)
    strat_pre = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel_processed, close_panel=close_panel,
        shared_cache={},
    )
    X_pre, y_pre = strat_pre._build_pooled_xy_from_panel(host_daily, current_bar)

    # Fast fallback (no shared_cache)
    strat_fb = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=factor_panel_processed, close_panel=close_panel,
        shared_cache=None,
    )
    X_fb, y_fb = strat_fb._build_pooled_xy_from_panel(host_daily, current_bar)

    # Legacy build_panel with preprocess
    strat_legacy = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
    )
    cur_date = host_daily["date"].iloc[current_bar]
    pool_t = strat_legacy._build_truncated_pool(host_daily, cur_date, current_bar)
    X_l, y_l = build_panel(
        pool_t, factors, cfg.horizon, preprocess_cfg=preprocess,
    )
    if len(X_l) > 0 and cfg.train_window > 0:
        X_l = X_l.groupby(level="stock", group_keys=False, sort=False).tail(cfg.train_window)
        y_l = y_l.loc[X_l.index]

    def norm(X, y):
        if X.index.names[0] == "date":
            X = X.swaplevel("date", "stock")
            y = y.swaplevel("date", "stock")
        return X.sort_index(), y.sort_index()

    Xp, yp = norm(X_pre, y_pre)
    Xf, yf = norm(X_fb,  y_fb)
    Xl, yl = norm(X_l,   y_l)

    assert set(Xp.index) == set(Xl.index), "pre-stack vs legacy diverge"
    assert set(Xf.index) == set(Xl.index), "fallback vs legacy diverge"
    pd.testing.assert_frame_equal(Xp, Xl, atol=1e-9, rtol=1e-9)
    pd.testing.assert_frame_equal(Xf, Xl, atol=1e-9, rtol=1e-9)
    pd.testing.assert_series_equal(yp, yl, check_names=False, atol=1e-9, rtol=1e-9)
    pd.testing.assert_series_equal(yf, yl, check_names=False, atol=1e-9, rtol=1e-9)

    # Sanity: with zscore on, the factor values are NOT the same as the raw
    # build_panel(preprocess_cfg=None) — otherwise this test wouldn't be testing
    # anything meaningful.
    X_raw, _ = build_panel(pool_t, factors, cfg.horizon)
    if len(X_raw) > 0 and cfg.train_window > 0:
        X_raw = X_raw.groupby(level="stock", group_keys=False, sort=False).tail(cfg.train_window)
    X_raw = X_raw.sort_index()
    assert not X_raw.equals(Xl), "raw vs preprocessed should differ when zscore=True"
