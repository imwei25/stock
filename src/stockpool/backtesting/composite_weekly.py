"""Fast, bit-exact weekly-score computation for ``CompositeVerdictStrategy``.

Background
----------
The naive walk-forward inside ``CompositeVerdictStrategy.generate_signals`` was
``O(T^2)``: for *every* daily bar ``i`` it re-resampled the entire growing daily
history to weekly **and** recomputed all weekly indicators via ``add_all`` —

    for i in range(...):
        weekly = resample_to_weekly(daily_df.iloc[:i + 1])   # O(i) resample
        enriched_w = add_all(weekly, cfg)                    # ~7 ms fixed cost
        weekly_score = score_triggers(detect_signals(enriched_w, weights))

Profiling (1 600-bar A-share history) showed ~17 s/stock, of which ~16.6 s was
this weekly block (resample ~5 s + ``add_all`` ~11.5 s).  ``add_all`` costs a
*fixed* ~7 ms per call regardless of frame size, so the killer was the **number
of calls** (~1 570 per stock), not their size.

Optimisation
------------
The weekly series for daily bar ``i`` (in week ``k``) is

    [ completed weeks 0 .. k-1 ]  ++  [ partial week k up to bar i ]

Every indicator is *causal*: simple-MA / BOLL / breakout / vol-ratio are finite
rolling windows, and MACD / KDJ / RSI are ``adjust=False`` EWMAs (purely
recursive — ``ema[t] = a*x[t] + (1-a)*ema[t-1]``).  Therefore indicator values at
**completed** weeks never change as more daily bars arrive, so

    enriched_full = add_all(resample_to_weekly(daily_df))     # computed ONCE

gives the exact indicator values for every completed week.  Only the **partial**
week's row changes per daily bar, and it can be derived incrementally from the
completed-week state (recursive EWMAs) and as-of rolling windows.

Bit-exactness
-------------
``detect_signals`` reads only the last 2-3 rows of the weekly frame plus the
``len(df) >= 4`` gate, so we reuse the *real* ``detect_signals`` on a 4-row tail
``[enriched_full[k-3 .. k-1], partial_row]`` — no signal logic is re-implemented.

The partial-row values are reproduced **bit-for-bit** vs the slow path:

* MACD / KDJ / RSI: the one-step ``adjust=False`` EWMA extension is exactly what
  pandas computes (verified bit-exact), seeded from the completed-week states.
* ``rolling.min`` / ``rolling.max`` (KDJ window, breakout) are order-independent,
  so a closed-form ``min``/``max`` over completed window + partial is exact.
* ``rolling.mean`` / ``rolling.std`` use an *online* Kahan/Welford accumulator in
  pandas that is **history-dependent** (a trailing window gives a different
  last value than the full series).  So MA / BOLL / vol-ratio are reproduced by
  rolling pandas over the full as-of series — not a closed form.  This is the
  one place we keep per-bar pandas rolling, but it is cheap relative to the
  former ``add_all`` per bar.

Validated to 0 mismatches vs the slow loop over 30 real A-share histories
(~47 000 bars) and 3 synthetic seeds; net ~11x faster on the weekly block.
The equivalence is locked in by ``tests/test_composite_weekly_fast.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.signals import detect_signals, score_triggers


def weekly_scores_by_bar(
    daily_df: pd.DataFrame,
    indicators_cfg,
    weights,
    start: int,
    weekly_warmup: int,
) -> dict[int, int]:
    """Return ``{daily_bar_index: weekly_score}`` for ``i in [start, len(daily_df))``.

    Bit-exact replacement for the per-bar
    ``score_triggers(detect_signals(add_all(resample_to_weekly(daily_df[:i+1]))))``
    block (yielding ``0`` when fewer than ``weekly_warmup`` weekly bars exist).

    Args mirror the strategy config: ``indicators_cfg`` is an ``IndicatorsConfig``
    and ``weights`` a ``WeightsConfig``.
    """
    n = len(daily_df)
    out: dict[int, int] = {}

    # Week index per daily bar (0-based, dense). Dates are sorted, so the
    # W-FRI period is non-decreasing and factorize gives the week counter.
    dates = pd.DatetimeIndex(pd.to_datetime(daily_df["date"]))
    k_arr = pd.factorize(dates.to_period("W-FRI"), sort=False)[0]

    high = daily_df["high"].to_numpy(dtype=float)
    low = daily_df["low"].to_numpy(dtype=float)
    vol = daily_df["volume"].to_numpy(dtype=float)
    openp = daily_df["open"].to_numpy(dtype=float)
    pc = daily_df["close"].to_numpy(dtype=float)

    # Partial-week OHLCV per daily bar (aggregation within the bar's week).
    ph = pd.Series(high).groupby(k_arr).cummax().to_numpy()
    pl = pd.Series(low).groupby(k_arr).cummin().to_numpy()
    pv = pd.Series(vol).groupby(k_arr).cumsum().to_numpy()
    po = pd.Series(openp).groupby(k_arr).transform("first").to_numpy()

    # Completed-week reference frame + enriched indicators (computed ONCE).
    # Use ALL columns so the per-bar tail frame is structurally identical to the
    # slow path's weekly frame (detect_signals reads literal column names like
    # ``rsi6`` / ``vol_ratio5``; mirroring the column set keeps behaviour exact
    # even for non-default indicator windows).
    wf = resample_to_weekly(daily_df)
    if wf.empty:
        return {i: 0 for i in range(start, n)}
    enriched_full = add_all(wf, indicators_cfg)
    all_cols = list(enriched_full.columns)
    ef = {c: enriched_full[c].to_numpy() for c in all_cols}

    wfc = wf["close"].to_numpy(dtype=float)
    wfh = wf["high"].to_numpy(dtype=float)
    wfl = wf["low"].to_numpy(dtype=float)
    wfv = wf["volume"].to_numpy(dtype=float)

    # Recursive (adjust=False) EWMA states over the completed-week series — the
    # exact same calls indicators.py uses, so values match bit-for-bit.
    fast, slow, sigp = indicators_cfg.macd.fast, indicators_cfg.macd.slow, indicators_cfg.macd.signal
    ema_fast = wf["close"].ewm(span=fast, adjust=False).mean().to_numpy()
    ema_slow = wf["close"].ewm(span=slow, adjust=False).mean().to_numpy()
    dif_full = ema_fast - ema_slow
    dea_full = pd.Series(dif_full).ewm(span=sigp, adjust=False).mean().to_numpy()

    nk, m1, m2 = indicators_cfg.kdj.n, indicators_cfg.kdj.m1, indicators_cfg.kdj.m2
    low_nf = wf["low"].rolling(nk).min()
    high_nf = wf["high"].rolling(nk).max()
    rsv_full = ((wf["close"] - low_nf) / (high_nf - low_nf) * 100).fillna(50)
    k_full = rsv_full.ewm(alpha=1 / m1, adjust=False).mean().to_numpy()
    d_full = pd.Series(k_full).ewm(alpha=1 / m2, adjust=False).mean().to_numpy()

    rsi_periods = indicators_cfg.rsi_periods
    rsi_p = 6 if 6 in rsi_periods else (rsi_periods[0] if rsi_periods else 6)
    delta = wf["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_p, adjust=False).mean().to_numpy()
    avg_loss = loss.ewm(alpha=1 / rsi_p, adjust=False).mean().to_numpy()

    # Completed-window rolling min/max (order-independent → exact).
    bk_max = pd.Series(wfc).rolling(indicators_cfg.breakout_window - 1).max().to_numpy()
    bk_min = pd.Series(wfc).rolling(indicators_cfg.breakout_window - 1).min().to_numpy()
    kd_min_l = pd.Series(wfl).rolling(nk - 1).min().to_numpy()
    kd_max_h = pd.Series(wfh).rolling(nk - 1).max().to_numpy()

    a_f = 2 / (fast + 1)
    a_s = 2 / (slow + 1)
    a_sig = 2 / (sigp + 1)
    a1 = 1 / m1
    a2 = 1 / m2
    a_rsi = 1 / rsi_p
    bw = indicators_cfg.boll.k
    vw = indicators_cfg.volume_ratio_window

    for i in range(start, n):
        k = int(k_arr[i])
        if k + 1 < weekly_warmup:
            out[i] = 0
            continue
        c = pc[i]

        # MA / BOLL: pandas rolling.mean/std are online (history-dependent), so
        # roll over the full as-of close series to stay bit-exact.
        asof_close = np.empty(k + 1)
        asof_close[:k] = wfc[:k]
        asof_close[k] = c
        s_close = pd.Series(asof_close)
        ma5 = s_close.rolling(5).mean().iloc[-1]
        ma10 = s_close.rolling(10).mean().iloc[-1]
        ma20 = s_close.rolling(20).mean().iloc[-1]
        ma60 = s_close.rolling(60).mean().iloc[-1]
        std = s_close.rolling(20).std(ddof=0).iloc[-1]
        boll_mid = ma20
        boll_up = ma20 + bw * std
        boll_low = ma20 - bw * std

        # vol_ratio = volume / rolling(w).mean(volume).shift(1); at row k the
        # shift makes it the mean of the *completed* weeks [k-w .. k-1].
        if k >= vw:
            avgv = pd.Series(wfv[:k]).rolling(vw).mean().iloc[-1]
            vr = pv[i] / avgv if avgv != 0 else np.nan
        else:
            vr = np.nan

        # MACD: one-step adjust=False EWMA extension off the completed states.
        ef_ = a_f * c + (1 - a_f) * ema_fast[k - 1]
        es_ = a_s * c + (1 - a_s) * ema_slow[k - 1]
        dif = ef_ - es_
        dea = a_sig * dif + (1 - a_sig) * dea_full[k - 1]
        hist = 2 * (dif - dea)

        # KDJ: RSV from completed+partial window min/max, then recursive K/D.
        if k >= nk - 1:
            ln = min(kd_min_l[k - 1], pl[i])
            hn = max(kd_max_h[k - 1], ph[i])
            denom = hn - ln
            rsv = (c - ln) / denom * 100 if denom != 0 else np.nan
            if not np.isfinite(rsv):
                rsv = 50.0
            kk = a1 * rsv + (1 - a1) * k_full[k - 1]
            dd = a2 * kk + (1 - a2) * d_full[k - 1]
            jj = 3 * kk - 2 * dd
        else:
            kk = dd = jj = np.nan

        # RSI (Wilder, adjust=False) one-step extension.
        d_ = c - wfc[k - 1]
        ag = a_rsi * max(d_, 0.0) + (1 - a_rsi) * avg_gain[k - 1]
        al = a_rsi * max(-d_, 0.0) + (1 - a_rsi) * avg_loss[k - 1]
        if al == 0:
            rsi6 = 50.0
        else:
            rsi6 = 100 - 100 / (1 + ag / al)
        if not np.isfinite(rsi6):
            rsi6 = 50.0
        if k < rsi_p:
            rsi6 = np.nan

        # Breakout: close == rolling(bw).max/min including itself → compare
        # against the completed-window extreme.
        bwn = indicators_cfg.breakout_window
        if k >= bwn - 1 and not np.isnan(bk_max[k - 1]):
            bh = bool(c >= bk_max[k - 1])
            bl = bool(c <= bk_min[k - 1])
        else:
            bh = bl = False

        # Reuse the real detect_signals on a 4-row tail (3 completed + partial).
        # The partial row only needs the columns detect_signals reads; any other
        # enriched column is unread, so NaN there is harmless.
        partial = {col: np.nan for col in all_cols}
        partial.update({
            "open": po[i], "high": ph[i], "low": pl[i], "close": c, "volume": pv[i],
            "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
            "macd_dif": dif, "macd_dea": dea, "macd_hist": hist,
            "kdj_k": kk, "kdj_d": dd, "kdj_j": jj,
            "boll_mid": boll_mid, "boll_up": boll_up, "boll_low": boll_low,
            "is_breakout_high": bh, "is_breakout_low": bl,
        })
        if "rsi6" in partial:
            partial["rsi6"] = rsi6
        if "vol_ratio5" in partial:
            partial["vol_ratio5"] = vr
        idx = (k - 3, k - 2, k - 1)
        data = {
            col: [ef[col][idx[0]], ef[col][idx[1]], ef[col][idx[2]], partial[col]]
            for col in all_cols
        }
        frame = pd.DataFrame(data, columns=all_cols)
        out[i] = score_triggers(detect_signals(frame, weights))

    return out
