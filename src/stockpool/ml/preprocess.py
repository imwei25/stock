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

    Args:
        df: T x N factor wide-frame (date index, code columns).
        log_mcap: T x N log-market-cap aligned to df's index. Codes that appear
            in ``df.columns`` but not in ``log_mcap.columns`` are treated as NaN
            and dropped per day.

    Returns:
        Same shape as df. NaN cells stay NaN. Days that fail the OLS preconditions
        (< 10 valid codes or rank deficiency) return their original df rows
        unchanged; aggregate fallback count is logged at WARNING level once per call.
    """
    if df.empty:
        return df.copy()
    log_mcap_aligned = log_mcap.reindex(index=df.index, columns=df.columns)
    out = df.copy()
    fallback_days = 0
    for date in df.index:
        y = df.loc[date]
        m = log_mcap_aligned.loc[date]
        X = pd.DataFrame({"intercept": 1.0, "log_mcap": m.values}, index=y.index)
        resid, used_ols = _per_day_ols_residual(y, X)
        if used_ols:
            out.loc[date] = resid
        else:
            fallback_days += 1
    if fallback_days:
        log.warning(
            "mcap_neutralize_panel: fallback on %d / %d days "
            "(degenerate cross-section: < 10 valid codes or rank deficient)",
            fallback_days, len(df.index),
        )
    return out


def industry_neutralize_panel(
    df: pd.DataFrame, sector_map: Mapping[str, str],
) -> pd.DataFrame:
    """Per-day within-industry demean.

    Args:
        df: T × N factor wide-frame (columns = codes).
        sector_map: ``{code: industry_label}``. Codes absent from the map
            fall into a single ``"_unknown_"`` bucket and are demeaned together.

    Returns:
        Same-shape DataFrame, each cell ``= x - mean(x within industry on day)``.

    Raises:
        ValueError: if ``sector_map`` is empty (caller catches and skips).
    """
    if not sector_map:
        raise ValueError("sector_map is empty; cannot industry-neutralize")
    industries = pd.Series(
        {c: sector_map.get(c, "_unknown_") for c in df.columns},
        name="industry",
    )
    # Transpose so each industry is contiguous rows; groupby + transform demean.
    transposed = df.T.copy()
    transposed["__industry__"] = industries
    # For each day column, subtract per-industry mean.
    date_cols = [c for c in transposed.columns if c != "__industry__"]
    demeaned = transposed.groupby("__industry__")[date_cols].transform(
        lambda s: s - s.mean()
    )
    return demeaned.T


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
) -> dict[str, pd.DataFrame]:
    """Run winsorize → cs_zscore → industry_neutralize on each factor.

    Args:
        factor_panel: ``{factor_name: T × N DataFrame}``.
        cfg: ``PreprocessConfig`` controlling which steps run.
        sector_map: ``{code: industry}``. Required when
            ``cfg.industry_neutralize=True``; if missing/empty, that step is
            skipped with a warning (other steps still run).
        factor_types: ``{factor_name: (type_tag, ...)}``. Factors whose tag
            tuple includes ``"fundamental"`` skip industry neutralize
            (preserves sector-intrinsic signal like bank-low-PE).
        n_codes: actual panel width (number of stocks). When provided AND
            below ``cfg.min_pool_size``, every preprocess step is skipped
            with a single warning — cross-sec preprocessing is unstable
            on small pools and produces silent zero-demean bugs in
            single-member industries (Phase 1.5 size guard). Pass ``None``
            to bypass the guard entirely (used by unit tests of the
            transform logic itself).

    Returns:
        New dict with same keys; values are transformed (or shallow-copied
        if cfg is all-off OR size guard tripped). Original input is never mutated.
    """
    if _is_all_off(cfg):
        return dict(factor_panel)

    # Phase 1.5 size guard: cross-sec preprocessing on small pools is
    # mathematically degenerate (μ/σ unstable; single-member industries
    # demean to 0). When caller supplies n_codes and it falls below the
    # configured threshold, skip every step with one warning per call.
    if n_codes is not None and n_codes < cfg.min_pool_size:
        log.warning(
            "preprocess pipeline skipped: n_codes=%d < min_pool_size=%d "
            "(cross-sectional preprocessing requires a wider panel)",
            n_codes, cfg.min_pool_size,
        )
        return dict(factor_panel)

    out: dict[str, pd.DataFrame] = {}
    do_neutralize = cfg.industry_neutralize and bool(sector_map)
    if cfg.industry_neutralize and not sector_map:
        log.warning(
            "industry_neutralize=True but sector_map is empty/None; "
            "skipping that step (winsorize/zscore still applied if enabled)"
        )

    for name, df in factor_panel.items():
        work = df
        if cfg.winsorize is not None:
            lo, hi = cfg.winsorize
            work = winsorize_panel(work, lo, hi)
        if cfg.zscore:
            work = cs_zscore_panel(work)
        if do_neutralize:
            tags = factor_types.get(name, ()) if factor_types else ()
            if "fundamental" not in tags:
                work = industry_neutralize_panel(work, sector_map)
        out[name] = work
    return out
