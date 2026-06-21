"""Cross-sectional preprocessing pipeline for ML factor panels.

Three stateless steps, applied per-day (cross-sectional):

  * ``winsorize_panel(df, lo, hi)``  — clip to per-day [lo, hi] quantiles
  * ``cs_zscore_panel(df)``           — per-day (x - μ_t) / σ_t
  * ``industry_neutralize_panel(df, sector_map)``
                                      — per-day within-industry demean

Wrapped by ``apply_preprocess_pipeline`` which honors a ``PreprocessConfig``.

Look-ahead safe: each function consumes only per-day cross-sectional info,
never references other rows. See spec
``docs/superpowers/specs/2026-06-06-factor-preprocessing-phase1-design.md``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stockpool.config import PreprocessConfig

log = logging.getLogger(__name__)


def winsorize_panel(
    df: pd.DataFrame, lower: float, upper: float,
) -> pd.DataFrame:
    """Per-day cross-sectional clip to ``[lower quantile, upper quantile]``.

    Args:
        df: T × N factor wide-frame (date index, code columns).
        lower: lower quantile bound, e.g. ``0.01``.
        upper: upper quantile bound, e.g. ``0.99``.

    Returns:
        Same-shape DataFrame with values outside [q_lo(t), q_hi(t)] clipped.
        All-NaN rows are returned unchanged (shape preserved).

    Raises:
        ValueError: if not ``0 < lower < upper < 1``.
    """
    if not (0 < lower < upper < 1):
        raise ValueError(
            f"winsorize bounds must satisfy 0 < lower < upper < 1, "
            f"got ({lower}, {upper})"
        )
    lo_q = df.quantile(lower, axis=1)
    hi_q = df.quantile(upper, axis=1)
    out = df.clip(lower=lo_q, upper=hi_q, axis=0)
    return out


def cs_zscore_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day cross-sectional z-score: ``(x - μ_t) / σ_t``.

    Args:
        df: T × N factor wide-frame.

    Returns:
        Same-shape DataFrame. Rows where ``σ_t < 1e-12`` (constant
        cross-section, all-NaN, or single non-NaN cell) return 0 — this
        deterministically neutralizes a degenerate day rather than producing
        ``±inf``/``NaN``. NaN cells stay NaN.

        ``σ`` uses ``ddof=0`` (matches ``standardize_fit`` upstream).
    """
    mu = df.mean(axis=1, skipna=True)
    sigma = df.std(axis=1, ddof=0, skipna=True)
    # Avoid div-by-zero: replace tiny σ with 1, then zero those rows out.
    sigma_safe = sigma.where(sigma >= 1e-12, 1.0)
    out = df.sub(mu, axis=0).div(sigma_safe, axis=0)
    degenerate = sigma < 1e-12
    if degenerate.any():
        # For degenerate rows, force non-NaN cells to 0 (NaN cells stay NaN).
        for d in df.index[degenerate]:
            out.loc[d] = out.loc[d].where(df.loc[d].isna(), 0.0)
    return out


def _per_day_ols_residual(
    y: pd.Series, X: pd.DataFrame,
) -> tuple[pd.Series, bool]:
    """Per-day OLS residualisation. Returns (residual, used_ols).

    ``y`` and ``X`` share the same index (codes for one date). Drops rows where
    y is NaN or any X column is NaN. Requires >= 10 valid rows AND X to be
    strictly tall (more rows than columns) — otherwise returns y unchanged with
    used_ols=False so callers can fall back / count degenerate days.

    Single-member dummy columns (column-sum == 1) are dropped along with the
    corresponding row to avoid those codes being demeaned to their own value
    (a silent zero-out, see Phase 1.5 incident).
    """
    valid_mask = y.notna() & X.notna().all(axis=1)
    y_v = y[valid_mask]
    X_v = X.loc[valid_mask]

    if len(y_v) < 10:
        return y, False

    # Drop dummies that have only one member among the valid rows;
    # drop those rows too (those single-member codes get y unchanged).
    col_sums = X_v.sum(axis=0)
    single_member_cols = col_sums.index[col_sums == 1].tolist()
    if single_member_cols:
        # The row(s) where the single-member dummy is hot — these codes are not
        # represented in the regression and keep their original y.
        single_member_rows = X_v.index[
            X_v[single_member_cols].any(axis=1)
        ].tolist()
        X_v = X_v.drop(columns=single_member_cols).drop(index=single_member_rows)
        y_v = y_v.drop(index=single_member_rows)

    if len(y_v) < 10 or X_v.shape[0] <= X_v.shape[1]:
        return y, False

    coef, *_ = np.linalg.lstsq(X_v.values, y_v.values, rcond=None)
    resid_v = y_v.values - X_v.values @ coef

    out = y.copy()
    out.loc[y_v.index] = resid_v
    return out, True


def mcap_neutralize_panel(
    df: pd.DataFrame, log_mcap: pd.DataFrame,
) -> pd.DataFrame:
    """Per-day residualise Y ~ 1 + log_mcap (no industry).

    Vectorised closed-form OLS: single regressor + intercept has an analytical
    solution identical to ``numpy.linalg.lstsq`` (β = cov(y, x) / var(x),
    α = ȳ − β·x̄, ε = y − α − β·x). Operates on the entire (T × N) panel via
    masked numpy sums — no per-day Python loop, no lstsq calls.

    Args:
        df: T × N factor wide-frame (date index, code columns).
        log_mcap: T × N log-market-cap aligned to df's index. Codes that appear
            in ``df.columns`` but not in ``log_mcap.columns`` are treated as NaN
            and dropped per day.

    Returns:
        Same shape as df. NaN cells stay NaN. Days that fail the OLS preconditions
        (< 10 valid codes or var(log_mcap) ≈ 0) return their original df rows
        unchanged; aggregate fallback count is logged at WARNING level once per call.
        Cells where log_mcap is NaN but the factor is valid keep the original
        factor value (matches the per-row OLS path which excludes those codes).
    """
    if df.empty:
        return df.copy()

    log_mcap_aligned = log_mcap.reindex(index=df.index, columns=df.columns)
    Y = df.astype(float).values         # (T, N)
    X = log_mcap_aligned.values          # (T, N)

    valid = ~(np.isnan(Y) | np.isnan(X))  # (T, N)
    n_valid = valid.sum(axis=1)            # (T,)

    # Per-day means via masked sums; invalid cells zeroed before sum so they
    # contribute nothing. ``safe_n`` keeps division well-defined on rows with
    # n_valid == 0 (those rows fall back below anyway).
    Y_z = np.where(valid, Y, 0.0)
    X_z = np.where(valid, X, 0.0)
    safe_n = np.where(n_valid > 0, n_valid, 1)
    mu_y = Y_z.sum(axis=1) / safe_n        # (T,)
    mu_x = X_z.sum(axis=1) / safe_n        # (T,)

    Y_c = np.where(valid, Y - mu_y[:, None], 0.0)
    X_c = np.where(valid, X - mu_x[:, None], 0.0)
    var_x = (X_c ** 2).sum(axis=1) / safe_n       # (T,)
    cov_xy = (X_c * Y_c).sum(axis=1) / safe_n     # (T,)

    safe_var = np.where(var_x > 1e-12, var_x, 1.0)
    beta = cov_xy / safe_var               # (T,)
    alpha = mu_y - beta * mu_x             # (T,)

    resid = Y - alpha[:, None] - beta[:, None] * X   # NaN-in-X/Y → NaN

    fallback_mask = (n_valid < 10) | (var_x <= 1e-12)
    out = np.where(fallback_mask[:, None], Y, resid)
    # Preserve original Y where only X is missing on a non-fallback day — the
    # per-row OLS variant excluded those codes from the regression, leaving
    # their original values untouched.
    cell_preserve = np.isnan(X) & ~np.isnan(Y)
    out = np.where(cell_preserve, Y, out)

    fallback_days = int(fallback_mask.sum())
    if fallback_days:
        log.warning(
            "mcap_neutralize_panel: fallback on %d / %d days "
            "(degenerate cross-section: < 10 valid codes or var(log_mcap) ≈ 0)",
            fallback_days, len(df.index),
        )
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def industry_neutralize_panel(
    df: pd.DataFrame,
    sector_map: Mapping[str, str],
    log_mcap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-day within-industry demean OR joint OLS Y ~ industry + log_mcap.

    Args:
        df: T × N factor wide-frame (columns = codes).
        sector_map: ``{code: industry_label}``. Codes absent from the map
            fall into a single ``"_unknown_"`` bucket and are demeaned together.
        log_mcap: optional T × N log(market_cap). When provided, the per-day
            transform switches from group demean to OLS residualisation against
            ``[industry_dummies(drop_first), log_mcap]``. Days that fail OLS
            preconditions (< 10 valid codes or rank deficient) fall back to
            the legacy group-demean path for that day.

    Returns:
        Same-shape DataFrame.

    Raises:
        ValueError: if ``sector_map`` is empty (caller catches and skips).
    """
    if not sector_map:
        raise ValueError("sector_map is empty; cannot industry-neutralize")

    if log_mcap is None:
        # Legacy fast path — group demean (bit-for-bit unchanged).
        industries = pd.Series(
            {c: sector_map.get(c, "_unknown_") for c in df.columns},
            name="industry",
        )
        transposed = df.T.copy()
        transposed["__industry__"] = industries
        date_cols = [c for c in transposed.columns if c != "__industry__"]
        demeaned = transposed.groupby("__industry__")[date_cols].transform(
            lambda s: s - s.mean()
        )
        return demeaned.T

    # OLS path: build one-hot industry dummies (drop first to avoid singularity)
    industries = pd.Series(
        {c: sector_map.get(c, "_unknown_") for c in df.columns},
    )
    dummies = pd.get_dummies(industries, prefix="ind", drop_first=True, dtype=float)
    # dummies index = codes; columns = ind_<label> minus reference

    log_mcap_aligned = log_mcap.reindex(index=df.index, columns=df.columns)
    out = df.astype(float).copy()
    fallback_days = 0

    # Pre-compute group-demean output once for fallback rows
    legacy_fallback = industry_neutralize_panel(df, sector_map, log_mcap=None)

    for date in df.index:
        y = df.loc[date]
        m = log_mcap_aligned.loc[date]
        X = dummies.copy()
        X["intercept"] = 1.0
        X["log_mcap"] = m.values
        resid, used_ols = _per_day_ols_residual(y, X)
        if used_ols:
            out.loc[date] = resid
        else:
            out.loc[date] = legacy_fallback.loc[date]
            fallback_days += 1

    if fallback_days:
        log.warning(
            "industry_neutralize_panel(log_mcap=...): OLS fallback on %d / %d days "
            "(degenerate cross-section); used group demean for those days",
            fallback_days, len(df.index),
        )
    return out


def _industry_neutralize_per_day_loop(
    df: pd.DataFrame,
    sector_map: Mapping[str, str],
    log_mcap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """LEGACY per-day OLS loop. Kept ONLY as a test oracle (see
    tests/test_ml_preprocess_mcap.py::test_industry_log_mcap_batch_matches_legacy).
    Not used in production after PR-T1.1."""
    if not sector_map:
        raise ValueError("sector_map is empty; cannot industry-neutralize")

    if log_mcap is None:
        # Legacy fast path — group demean (bit-for-bit unchanged).
        industries = pd.Series(
            {c: sector_map.get(c, "_unknown_") for c in df.columns},
            name="industry",
        )
        transposed = df.T.copy()
        transposed["__industry__"] = industries
        date_cols = [c for c in transposed.columns if c != "__industry__"]
        demeaned = transposed.groupby("__industry__")[date_cols].transform(
            lambda s: s - s.mean()
        )
        return demeaned.T

    # OLS path: build one-hot industry dummies (drop first to avoid singularity)
    industries = pd.Series(
        {c: sector_map.get(c, "_unknown_") for c in df.columns},
    )
    dummies = pd.get_dummies(industries, prefix="ind", drop_first=True, dtype=float)
    # dummies index = codes; columns = ind_<label> minus reference

    log_mcap_aligned = log_mcap.reindex(index=df.index, columns=df.columns)
    out = df.astype(float).copy()
    fallback_days = 0

    # Pre-compute group-demean output once for fallback rows
    legacy_fallback = industry_neutralize_panel(df, sector_map, log_mcap=None)

    for date in df.index:
        y = df.loc[date]
        m = log_mcap_aligned.loc[date]
        X = dummies.copy()
        X["intercept"] = 1.0
        X["log_mcap"] = m.values
        resid, used_ols = _per_day_ols_residual(y, X)
        if used_ols:
            out.loc[date] = resid
        else:
            out.loc[date] = legacy_fallback.loc[date]
            fallback_days += 1

    if fallback_days:
        log.warning(
            "industry_neutralize_panel(log_mcap=...): OLS fallback on %d / %d days "
            "(degenerate cross-section); used group demean for those days",
            fallback_days, len(df.index),
        )
    return out


def _is_all_off(cfg: "PreprocessConfig") -> bool:
    """True when every step is disabled (cfg semantically a no-op)."""
    return (
        cfg.winsorize is None
        and cfg.zscore is False
        and cfg.industry_neutralize is False
        and cfg.mcap_neutralize is False
    )


def apply_preprocess_pipeline(
    factor_panel: dict[str, pd.DataFrame],
    cfg: "PreprocessConfig",
    sector_map: Mapping[str, str] | None = None,
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
    n_codes: int | None = None,
    log_mcap_panel: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run winsorize → cs_zscore → (industry/mcap) neutralize per factor.

    Args:
        factor_panel: ``{factor_name: T × N DataFrame}``.
        cfg: ``PreprocessConfig`` controlling which steps run.
        sector_map: ``{code: industry}``. Required when ``cfg.industry_neutralize``;
            empty/missing skips that step with a warning.
        factor_types: ``{factor_name: (type_tag, ...)}``.
            * ``"fundamental"`` → skip industry neutralize (legacy bank-low-PE rule).
            * ``"contains_mcap"`` → also skip mcap neutralize (PE/PB are
              collinear with close × shares).
        n_codes: actual panel width. When below ``cfg.min_pool_size`` every step
            is skipped with a warning (Phase 1.5 size guard).
        log_mcap_panel: T × N log(market_cap). Required when
            ``cfg.mcap_neutralize``; ``None`` → skip mcap with a warning.

    Returns:
        New dict with same keys; values transformed (or shallow-copied if cfg
        all-off or size guard tripped). Original input is never mutated.
    """
    if _is_all_off(cfg):
        return dict(factor_panel)

    if n_codes is not None and n_codes < cfg.min_pool_size:
        log.warning(
            "preprocess pipeline skipped: n_codes=%d < min_pool_size=%d "
            "(cross-sectional preprocessing requires a wider panel)",
            n_codes, cfg.min_pool_size,
        )
        return dict(factor_panel)

    do_industry = cfg.industry_neutralize and bool(sector_map)
    if cfg.industry_neutralize and not sector_map:
        log.warning(
            "industry_neutralize=True but sector_map is empty/None; "
            "skipping that step (winsorize/zscore/mcap still applied if enabled)"
        )

    do_mcap = cfg.mcap_neutralize and log_mcap_panel is not None
    if cfg.mcap_neutralize and log_mcap_panel is None:
        log.warning(
            "mcap_neutralize=True but log_mcap_panel is None; skipping mcap step "
            "(caller must build log(market_cap) and pass it in)"
        )

    out: dict[str, pd.DataFrame] = {}
    for name, df in factor_panel.items():
        work = df
        if cfg.winsorize is not None:
            lo, hi = cfg.winsorize
            work = winsorize_panel(work, lo, hi)
        if cfg.zscore:
            work = cs_zscore_panel(work)

        tags = factor_types.get(name, ()) if factor_types else ()
        is_fundamental = "fundamental" in tags
        is_contains_mcap = "contains_mcap" in tags

        # industry: legacy rule — skip on fundamental tag.
        run_industry = do_industry and not is_fundamental
        # mcap: skip only on contains_mcap tag (PE/PB). Other fundamentals OK.
        run_mcap = do_mcap and not is_contains_mcap

        if run_industry and run_mcap:
            work = industry_neutralize_panel(
                work, sector_map, log_mcap=log_mcap_panel,
            )
        elif run_industry:
            work = industry_neutralize_panel(work, sector_map)
        elif run_mcap:
            work = mcap_neutralize_panel(work, log_mcap_panel)

        out[name] = work
    return out
