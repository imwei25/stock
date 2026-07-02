"""Training-data hygiene (2026-07-02, RV session).

Three fixes born from the alpha_083 inf poisoning + delisted-backfill frozen
windows (WORKLOG「RV」):

  * ``compute_factor_panel`` sanitizes ±inf → NaN at the single chokepoint
    every factor passes through (bare-division factors emit inf on H==L==C
    一字板 bars; ≥1% inf on one day also defeats the winsorize q99 clip).
  * ``stack_panel_to_xy(dropna=True)`` screens non-finite values (defense in
    depth — one inf row silently NaNs the whole Lasso fit).
  * pooled training slices bound rows to the host's trailing ``train_window``
    dates, so a stock whose data ends mid-history (delisted) ages out of the
    pool instead of contributing its frozen final rows to every later fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.backtesting.strategies import MLFactorStrategy
from stockpool.config import MLFactorConfig, SelectorConfig, WeighterConfig
from stockpool.ml.dataset import compute_factor_panel, stack_panel_to_xy
from stockpool.strategy_factory import build_close_panel


# ---- helpers (mirroring test_ml_strategy_panel_fit_reuse fixtures) ----

def _stock_df(n: int, seed: int, start: str = "2024-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="B"),
        "open": close * 0.998,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.uniform(5e5, 2e6, n),
    })


def _ohlcv_panel(pool: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    per_stock = {c: d.set_index(pd.to_datetime(d["date"])).sort_index()
                 for c, d in pool.items()}
    dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(dates, name="date")
    return {
        f: pd.DataFrame({c: d[f].reindex(idx) for c, d in per_stock.items()},
                        index=idx)
        for f in ("open", "high", "low", "close", "volume")
    }


def _factor_panel(pool, factors):
    return compute_factor_panel(_ohlcv_panel(pool), factors)


def _cfg(factors, train_window=40) -> MLFactorConfig:
    return MLFactorConfig(
        factors=factors, horizon=3, train_window=train_window,
        min_train_samples=20, refit_every=10, panel_mode="pooled",
        embargo_days=0, share_pool_fit=True, label_basis="close",
        selector=SelectorConfig(type="lasso"),
        weighter=WeighterConfig(type="ic"),
    )


# ---- inf sanitization ----

def test_compute_factor_panel_sanitizes_inf(monkeypatch):
    """A factor emitting ±inf (bare division on an H==L==C bar) must come out
    of compute_factor_panel as NaN."""
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    dirty = pd.DataFrame(
        {"A": [1.0, np.inf, 3.0, -np.inf, 5.0], "B": [1.0] * 5}, index=idx,
    )

    class _FakeFactor:
        name = "fake_inf"

        def compute(self, panel):
            return dirty

    import stockpool.ml.dataset as ds
    monkeypatch.setattr(ds, "make_factor", lambda name: _FakeFactor())
    out = ds.compute_factor_panel({"close": dirty}, ["fake_inf"])["fake_inf"]
    assert not np.isinf(out.to_numpy()).any()
    assert out["A"].isna().tolist() == [False, True, False, True, False]
    # finite values untouched
    assert out.loc[idx[0], "A"] == 1.0
    assert (out["B"] == 1.0).all()


def test_stack_panel_to_xy_drops_nonfinite_rows():
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    fp = {"f1": pd.DataFrame({"A": [1.0, np.inf, 3.0, 4.0],
                              "B": [1.0, 2.0, 3.0, 4.0]}, index=idx)}
    fwd = pd.DataFrame({"A": [0.1, 0.1, 0.1, np.inf],
                        "B": [0.1, 0.1, 0.1, 0.1]}, index=idx)
    X, y = stack_panel_to_xy(fp, fwd, dropna=True)
    assert np.isfinite(X.to_numpy()).all()
    assert np.isfinite(y.to_numpy()).all()
    # A@idx[1] (inf factor) and A@idx[3] (inf label) must be gone.
    assert ("A", idx[1]) not in X.index
    assert ("A", idx[3]) not in X.index
    # B rows survive in full.
    assert sum(1 for s, _ in X.index if s == "B") == 4


# ---- stale frozen-window exclusion ----

def _delisted_pool(n_bars=160, delist_at=60):
    pool = {c: _stock_df(n_bars, seed=i + 1)
            for i, c in enumerate(["A", "B", "C"])}
    # D "delists": data ends at bar `delist_at`.
    pool["D"] = _stock_df(delist_at, seed=9)
    return pool


def test_pooled_fast_path_excludes_stale_delisted_rows():
    """At a late cutoff, a stock whose data ended >train_window dates ago must
    contribute zero training rows (fast path)."""
    factors = ["momentum_5"]
    pool = _delisted_pool()
    fp = _factor_panel(pool, factors)
    cp = build_close_panel(pool)
    cfg = _cfg(factors, train_window=40)

    strat = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=fp, close_panel=cp, shared_cache={},
    )
    X, _y = strat._build_pooled_xy_from_panel(pool["A"], 150)
    stocks = set(X.index.get_level_values("stock"))
    assert "D" not in stocks, (
        "delisted stock's frozen final rows leaked into a much-later "
        "training set"
    )
    assert {"B", "C"} <= stocks


def test_pooled_fast_path_keeps_recently_ended_stock():
    """A stock whose data ends *inside* the trailing window still trains."""
    factors = ["momentum_5"]
    pool = _delisted_pool(n_bars=160, delist_at=140)  # ends inside tail(40) of bar 150
    fp = _factor_panel(pool, factors)
    cp = build_close_panel(pool)
    cfg = _cfg(factors, train_window=40)

    strat = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=fp, close_panel=cp, shared_cache={},
    )
    X, _y = strat._build_pooled_xy_from_panel(pool["A"], 150)
    assert "D" in set(X.index.get_level_values("stock"))


def test_pooled_legacy_fallback_matches_fast_on_delisted_pool():
    """Fast path (pre-stacked) and per-call fallback agree on the delisted
    pool — the date-bound must live in both."""
    factors = ["momentum_5"]
    pool = _delisted_pool()
    fp = _factor_panel(pool, factors)
    cp = build_close_panel(pool)
    cfg = _cfg(factors, train_window=40)

    fast = MLFactorStrategy(
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=fp, close_panel=cp, shared_cache={},
    )
    slow = MLFactorStrategy(  # no shared_cache → per-call fallback
        cfg=cfg, pool_data=pool, current_stock_code="A",
        factor_panel=fp, close_panel=cp,
    )
    Xf, yf = fast._build_pooled_xy_from_panel(pool["A"], 150)
    Xs, ys = slow._build_pooled_xy_from_panel(pool["A"], 150)
    assert set(Xf.index) == set(Xs.index)
    pd.testing.assert_frame_equal(
        Xf.sort_index(), Xs.sort_index(), check_like=True, atol=1e-9, rtol=1e-9,
    )
    pd.testing.assert_series_equal(
        yf.sort_index(), ys.sort_index(), check_names=False, atol=1e-9, rtol=1e-9,
    )
