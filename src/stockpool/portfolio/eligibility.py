"""Per-bar eligibility filter for the portfolio engine.

Decides "which codes are even allowed to enter the portfolio today", before
the engine applies score-based ranking. Three checks:

  * ``min_history_bars`` — enough bars to compute factors / metrics
  * ``exclude_st`` — name contains "ST" / "*ST" / etc.
  * ``min_avg_amount_20d`` — last-20-bar mean of ``close * volume * 100``
    (mootdx volume unit = 手; 1 手 = 100 股 → multiply by 100 for amount in 元)

Industry cap is *not* here: it depends on the engine's evolving target set
(per-target greedy walk), so it lives in the engine.

Performance
-----------
``eligible`` is called once per *rebalance bar* (every ``rebalance_n_days``)
over the whole universe. The naive implementation re-parsed
``pd.to_datetime(daily["date"])`` and re-sliced + re-aggregated every stock's
full history on *every* call — ``O(rebalances × N × T)`` and dominated by
pandas ``to_datetime`` (profiled at ~38 s of a 100-stock engine run).

The eligibility verdict at ``date_t`` is a pure as-of function of each stock's
own history, so we precompute it **once per panel** into per-code arrays of
``(sorted_date_ns, eligible_bool_per_bar)`` and answer each ``eligible(date_t)``
with an ``O(N log T)`` ``searchsorted``. The result is bit-identical to the old
path (same ``tail(20).mean()`` fresh-window mean, same NaN / missing-column /
ST semantics); locked in by ``tests/test_portfolio_eligibility.py`` and
``tests/test_portfolio_eligibility_equiv.py``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.config import PortfolioEligibilityConfig


def _trailing_mean_20(amount: np.ndarray) -> np.ndarray:
    """Per-bar mean of the trailing ≤20 values — bit-exact vs ``df.tail(20).mean()``.

    Full windows (bar ≥ 19) use a sliding-window view; the first 19 partial
    windows are computed directly. NaN in any window propagates (matches the
    original ``(close*volume*100).mean()`` returning NaN).
    """
    n = len(amount)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    if n >= 20:
        sw = np.lib.stride_tricks.sliding_window_view(amount, 20).mean(axis=1)
        out[19:] = sw
    last = min(19, n)
    for j in range(last):
        out[j] = amount[: j + 1].mean()
    return out


class EligibilityFilter:
    """Decide eligible codes per bar.

    Args:
        cfg: ``PortfolioEligibilityConfig`` from the loaded yaml.
        name_map: ``{code: display_name}``. Used for ST detection only.
            Codes absent from ``name_map`` are *not* assumed ST (they pass
            the ST check; the only way to filter them is via the other rules).
    """

    def __init__(
        self,
        cfg: PortfolioEligibilityConfig,
        name_map: Mapping[str, str] | None = None,
    ):
        self.cfg = cfg
        self.name_map = dict(name_map or {})
        # Per-panel precompute cache (keyed by id(panel_data) — the engine reuses
        # the same dict object across all rebalance bars in a run).
        self._cache_key: int | None = None
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def eligible(
        self,
        date_t: pd.Timestamp,
        panel_data: Mapping[str, pd.DataFrame],
    ) -> set[str]:
        """Return the set of codes that pass all three checks at ``date_t``."""
        date_t = pd.Timestamp(date_t)
        if self._cache_key != id(panel_data):
            self._build_cache(panel_data)
        ts = date_t.value  # int64 nanoseconds
        out: set[str] = set()
        for code, (dates_ns, elig) in self._cache.items():
            if dates_ns.size == 0:
                continue
            # j = index of the last bar with date <= date_t.
            j = int(np.searchsorted(dates_ns, ts, side="right")) - 1
            if j < 0:
                continue
            if elig[j]:
                out.add(code)
        return out

    def _build_cache(self, panel_data: Mapping[str, pd.DataFrame]) -> None:
        cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        min_hist = self.cfg.min_history_bars
        thr = self.cfg.min_avg_amount_20d
        check_liq = thr > 0
        empty = (np.empty(0, dtype="int64"), np.empty(0, dtype=bool))
        for code, daily in panel_data.items():
            if self.cfg.exclude_st and _is_st(self.name_map.get(code, "")):
                cache[code] = empty
                continue
            if "date" not in daily.columns or "close" not in daily.columns:
                cache[code] = empty
                continue
            # Sort by date (defensive; matches the engine's sorted pivots and the
            # original positional tail() on already-sorted cache data).
            dates_ns = pd.to_datetime(daily["date"]).to_numpy(dtype="datetime64[ns]").astype("int64")
            order = np.argsort(dates_ns, kind="stable")
            dates_ns = dates_ns[order]
            n = len(dates_ns)
            if n == 0:
                cache[code] = empty
                continue
            hist_ok = np.arange(1, n + 1) >= min_hist
            if check_liq:
                if "volume" not in daily.columns:
                    cache[code] = (dates_ns, np.zeros(n, dtype=bool))
                    continue
                close = daily["close"].to_numpy(dtype=float)[order]
                vol = daily["volume"].to_numpy(dtype=float)[order]
                amount = close * vol * 100.0
                avg20 = _trailing_mean_20(amount)
                liq_ok = ~np.isnan(avg20) & (avg20 >= thr)
                elig = hist_ok & liq_ok
            else:
                elig = hist_ok
            cache[code] = (dates_ns, elig)
        self._cache = cache
        self._cache_key = id(panel_data)


def _is_st(name: str) -> bool:
    """ST detection: case-insensitive substring on the display name.

    Matches '*ST', 'ST', 'st' — anywhere in the name. Same heuristic as
    ``recommend_pool``.
    """
    if not name:
        return False
    return "ST" in name.upper()
