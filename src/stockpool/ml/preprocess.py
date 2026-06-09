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


def market_cap_neutralize_panel(
    df: pd.DataFrame, log_mcap: pd.DataFrame,
) -> pd.DataFrame:
    """Per-day cross-sectional OLS residual of factor on ``log(market_cap)``.

    For each day (row) ``t`` the factor cross-section ``f`` is regressed on
    ``[1, m]`` where ``m`` is that day's ``log_mcap`` cross-section, and the
    residual ``f - (a + b·m)`` replaces the factor value. This strips the
    linear size exposure (large-cap vs small-cap tilt) from the factor while
    preserving everything orthogonal to size.

    Vectorised over all days at once (no python loop): per-row slope/intercept
    are computed from masked cross-sectional moments.

    Args:
        df: T × N factor wide-frame (date index, code columns).
        log_mcap: T × N ``log(total_market_cap)`` wide-frame. Reindexed to
            ``df``'s index/columns internally, so a superset panel is fine.

    Returns:
        Same-shape DataFrame.
          * Cells where ``f`` is NaN stay NaN.
          * Cells where ``f`` is valid but ``log_mcap`` is NaN keep the
            **original** ``f`` (cannot residualise without size → pass through).
          * Days with < 2 jointly-valid stocks, or a degenerate size cross-
            section (``var(m) ≈ 0``), fall back to a plain demean
            (``f - mean(f)``), i.e. slope ``b = 0``.
    """
    m = log_mcap.reindex(index=df.index, columns=df.columns)
    # Joint-valid mask per cell.
    valid = df.notna() & m.notna()
    vmask = valid.astype(float)
    fv = df.where(valid, 0.0)
    mv = m.where(valid, 0.0)

    n = vmask.sum(axis=1)
    Sm = mv.sum(axis=1)
    Sf = fv.sum(axis=1)
    Smm = (mv * mv).sum(axis=1)
    Smf = (mv * fv).sum(axis=1)

    n_safe = n.where(n > 0, np.nan)
    mean_m = Sm / n_safe
    mean_f = Sf / n_safe
    var_m = Smm / n_safe - mean_m**2
    cov_mf = Smf / n_safe - mean_m * mean_f

    # slope only where the size cross-section is non-degenerate and n >= 2.
    ok = (var_m > 1e-12) & (n >= 2)
    b = (cov_mf / var_m.where(ok, np.nan)).where(ok, 0.0)
    a = mean_f - b * mean_m  # demean fallback when b == 0

    fitted = m.mul(b, axis=0).add(a, axis=0)  # a + b·m (NaN where m NaN)
    resid = df - fitted
    # Where m is NaN (but f valid), fitted is NaN → resid NaN; restore raw f.
    resid = resid.where(m.notna(), df)
    return resid


def symmetric_orthogonalize_panel(
    factor_panel: dict[str, pd.DataFrame],
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-day cross-sectional symmetric (Löwdin) orthogonalization.

    Jointly decorrelates the **non-fundamental** factors so that, on each day,
    the cross-sectional correlation between any two output factors is ~0, while
    each output factor stays maximally close to its (standardised) input
    (order-independent — unlike Gram-Schmidt). Fundamental-tagged factors pass
    through untouched (orthogonalising PE/PB against momentum muddies the
    intrinsic valuation signal — same rationale as the neutralize steps).

    Stateless per day (each day computes its own transform from that day's
    cross-section only) → look-ahead safe; the predict path reading the same
    cached panel is automatically consistent with training.

    Args:
        factor_panel: ``{factor_name: T × N DataFrame}`` (date index, code cols).
        factor_types: ``{factor_name: (type_tag, ...)}``; names whose tags include
            ``"fundamental"`` are excluded from orthogonalization (copied through).

    Returns:
        New dict, same keys. Non-fundamental factors decorrelated per day;
        fundamental factors and the input frames are never mutated.

        Per-day fallbacks (day returned unchanged for the affected factors):
          * fewer jointly-valid stocks than factors (``N_valid < K``) → cannot
            form a full-rank correlation matrix;
          * all-NaN / empty day.
        On a non-degenerate day, cells where any non-fundamental factor is NaN
        are set NaN across all non-fundamental factors (they leave the valid
        subset and are dropped downstream at stack_panel_to_xy anyway).
    """
    types = factor_types or {}
    nf_names = [
        n for n in factor_panel
        if "fundamental" not in types.get(n, ())
    ]
    out: dict[str, pd.DataFrame] = {n: factor_panel[n].copy() for n in factor_panel}
    K = len(nf_names)
    if K == 0:
        return out

    ref = factor_panel[nf_names[0]]
    dates, codes = ref.index, ref.columns
    cube = np.stack(
        [factor_panel[n].reindex(index=dates, columns=codes).to_numpy(dtype=float)
         for n in nf_names],
        axis=-1,
    )  # shape (T, N, K)

    transformed = cube.copy()
    for ti in range(cube.shape[0]):
        day = cube[ti]                          # (N, K)
        valid = ~np.isnan(day).any(axis=1)      # stocks with all K factors present
        n_valid = int(valid.sum())
        if n_valid < K or n_valid == 0:
            continue                            # passthrough this day (keep raw)
        F = day[valid]                          # (n_valid, K)
        mu = F.mean(axis=0)
        sigma = F.std(axis=0, ddof=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        Fs = (F - mu) / sigma                   # per-day z-score
        M = (Fs.T @ Fs) / n_valid               # (K, K) correlation matrix
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-10)    # floor to keep S finite
        S = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
        transformed[ti][valid] = Fs @ S
        # Stocks missing any non-fundamental factor leave the valid subset →
        # NaN them across ALL non-fundamental factors so the day is consistent
        # (dropped downstream at stack_panel_to_xy anyway).
        transformed[ti][~valid] = np.nan

    for k, name in enumerate(nf_names):
        out[name] = pd.DataFrame(transformed[:, :, k], index=dates, columns=codes)
    return out


def _is_all_off(cfg: "PreprocessConfig") -> bool:
    """True when every step is disabled (cfg semantically a no-op)."""
    return (
        cfg.winsorize is None
        and cfg.zscore is False
        and cfg.industry_neutralize is False
        and cfg.market_cap_neutralize is False
        and cfg.symmetric_orthogonalize is False
    )


def apply_preprocess_pipeline(
    factor_panel: dict[str, pd.DataFrame],
    cfg: "PreprocessConfig",
    sector_map: Mapping[str, str] | None = None,
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
    n_codes: int | None = None,
    log_mcap_panel: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run winsorize → cs_zscore → industry_neutralize → market_cap_neutralize.

    Args:
        factor_panel: ``{factor_name: T × N DataFrame}``.
        cfg: ``PreprocessConfig`` controlling which steps run.
        sector_map: ``{code: industry}``. Required when
            ``cfg.industry_neutralize=True``; if missing/empty, that step is
            skipped with a warning (other steps still run).
        factor_types: ``{factor_name: (type_tag, ...)}``. Factors whose tag
            tuple includes ``"fundamental"`` skip BOTH neutralize steps
            (preserves sector/size-intrinsic signal like bank-low-PE).
        n_codes: actual panel width (number of stocks). When provided AND
            below ``cfg.min_pool_size``, every preprocess step is skipped
            with a single warning — cross-sec preprocessing is unstable
            on small pools and produces silent zero-demean bugs in
            single-member industries (Phase 1.5 size guard). Pass ``None``
            to bypass the guard entirely (used by unit tests of the
            transform logic itself).
        log_mcap_panel: T × N ``log(total_market_cap)`` wide-frame. Required
            when ``cfg.market_cap_neutralize=True``; if missing, that step is
            skipped with a warning (other steps still run).

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
            "skipping that step (other steps still applied if enabled)"
        )
    do_mcap = cfg.market_cap_neutralize and log_mcap_panel is not None
    if cfg.market_cap_neutralize and log_mcap_panel is None:
        log.warning(
            "market_cap_neutralize=True but log_mcap_panel is None; "
            "skipping that step (other steps still applied if enabled)"
        )

    for name, df in factor_panel.items():
        work = df
        tags = factor_types.get(name, ()) if factor_types else ()
        is_fundamental = "fundamental" in tags
        if cfg.winsorize is not None:
            lo, hi = cfg.winsorize
            work = winsorize_panel(work, lo, hi)
        if cfg.zscore:
            work = cs_zscore_panel(work)
        if do_neutralize and not is_fundamental:
            work = industry_neutralize_panel(work, sector_map)
        if do_mcap and not is_fundamental:
            work = market_cap_neutralize_panel(work, log_mcap_panel)
        out[name] = work
    return out
