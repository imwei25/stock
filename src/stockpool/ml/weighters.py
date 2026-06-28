"""Factor weighters: re-weight selected factors by predictive power.

A ``FactorWeighter`` consumes the (already filtered) factor matrix and target,
learns one weight per factor, and produces a per-bar prediction via a simple
weighted sum on standardised inputs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from stockpool.ml.dataset import standardize_apply, standardize_fit


class FactorWeighter(ABC):
    """Assign a weight to each factor and produce a composite score."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def weights(self) -> pd.Series: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-bar per-factor contribution to ``predict(X)``.

        Linear weighters return ``standardised(X) * weights`` (row sums equal
        ``predict(X)`` by construction). Non-linear weighters (e.g. LightGBM)
        return their model-specific decomposition, e.g. SHAP values.
        """


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Returns 0 if either side is constant."""
    if len(a) < 2:
        return 0.0
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa = ra.std(ddof=0)
    sb = rb.std(ddof=0)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    sa = a.std(ddof=0)
    sb = b.std(ddof=0)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


class _StandardisingMixin:
    """Shared standardisation helpers — fit on training, replay on predict."""

    def _fit_standardiser(self, X: pd.DataFrame) -> np.ndarray:
        self._feature_names = list(X.columns)
        Xn = X.to_numpy(dtype=float, copy=True)
        self._x_mean, self._x_std = standardize_fit(Xn)
        return standardize_apply(Xn, self._x_mean, self._x_std)

    def _apply_standardiser(self, X: pd.DataFrame) -> np.ndarray:
        if self._x_mean is None or self._x_std is None or self._feature_names is None:
            raise RuntimeError("Weighter not fitted yet")
        # Reorder columns to fit-time order; missing cols filled with 0 (mean).
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(
                f"predict() missing columns from fit: {missing}"
            )
        Xn = X[self._feature_names].to_numpy(dtype=float, copy=True)
        return standardize_apply(Xn, self._x_mean, self._x_std)


class _LinearWeighterContributionsMixin:
    """Shared ``contributions()`` impl for linear-combination weighters
    (IC / IR / Equal). Returns ``standardised(X) * weights`` per cell.

    Depends on the class to provide:
      * ``self._weights`` — pd.Series of per-factor weights
      * ``self._feature_names`` — list[str] of fit-time feature names
      * ``self._apply_standardiser(X)`` — z-score apply (from _StandardisingMixin)
    """

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._weights is None or self._weights.empty:
            return pd.DataFrame(index=X.index)
        Xs = self._apply_standardiser(X)
        w = self._weights.to_numpy()
        return pd.DataFrame(
            Xs * w, index=X.index, columns=self._feature_names,
        )


class EqualWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Equal weight (1/k) on standardised factors. Useful as a baseline."""

    def __init__(self):
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            return
        k = len(self._feature_names)
        self._weights = pd.Series(
            [1.0 / k] * k, index=self._feature_names, name="weight",
        )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class ICWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Weight each factor by its IC with the target.

    By default uses Spearman rank IC (more robust to outliers in returns).
    Set ``use_rank=False`` for Pearson IC. Weights are L1-normalised
    (``Σ|w| = 1``) so prediction magnitudes are comparable across refits.

    Sign of the weight follows the sign of the IC (a negative-IC factor is
    inverted), so the resulting score is monotone-increasing in expected return.
    """

    def __init__(self, use_rank: bool = True, min_abs_ic: float = 0.0):
        self.use_rank = use_rank
        self.min_abs_ic = min_abs_ic
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None
        self._ic: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            self._ic = pd.Series(dtype=float)
            return

        y_np = y.to_numpy(dtype=float)
        corr_fn = _spearman_corr if self.use_rank else _pearson_corr
        ic_values = np.array([corr_fn(Xs[:, j], y_np) for j in range(Xs.shape[1])])
        self._ic = pd.Series(ic_values, index=self._feature_names, name="ic")

        masked = np.where(np.abs(ic_values) >= self.min_abs_ic, ic_values, 0.0)
        total = np.sum(np.abs(masked))
        if total < 1e-12:
            # Fall back to equal weight if no factor crosses the IC threshold.
            k = len(self._feature_names)
            self._weights = pd.Series(
                [1.0 / k] * k, index=self._feature_names, name="weight",
            )
        else:
            self._weights = pd.Series(
                masked / total, index=self._feature_names, name="weight",
            )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    @property
    def ic(self) -> pd.Series:
        """The fit-time IC of each factor (signed)."""
        assert self._ic is not None
        return self._ic.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class IRWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Weight each factor by its information ratio over rolling sub-windows.

    IR_i = mean(IC_i) / std(IC_i) computed over equal-sized chunks of the
    training window. Falls back to ICWeighter behaviour when only one chunk
    fits.

    Lower the noise sensitivity of plain IC weighting at the cost of needing
    a longer training window (recommend ≥ 5 * n_chunks bars).
    """

    def __init__(
        self, n_chunks: int = 6, use_rank: bool = True, min_abs_ir: float = 0.0,
    ):
        if n_chunks <= 0:
            raise ValueError(f"n_chunks must be > 0, got {n_chunks}")
        self.n_chunks = n_chunks
        self.use_rank = use_rank
        self.min_abs_ir = min_abs_ir
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None
        self._ir: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            self._ir = pd.Series(dtype=float)
            return

        n = Xs.shape[0]
        n_chunks = min(self.n_chunks, max(1, n // 5))
        bounds = np.linspace(0, n, n_chunks + 1, dtype=int)
        corr_fn = _spearman_corr if self.use_rank else _pearson_corr
        y_np = y.to_numpy(dtype=float)

        ir_values = np.zeros(Xs.shape[1])
        for j in range(Xs.shape[1]):
            chunk_ics = []
            for c in range(n_chunks):
                lo, hi = bounds[c], bounds[c + 1]
                if hi - lo < 2:
                    continue
                chunk_ics.append(corr_fn(Xs[lo:hi, j], y_np[lo:hi]))
            if len(chunk_ics) < 2:
                # Fall back to single IC (= IR with denominator dropped).
                ir_values[j] = corr_fn(Xs[:, j], y_np)
                continue
            arr = np.array(chunk_ics)
            std = arr.std(ddof=0)
            if std < 1e-12:
                ir_values[j] = arr.mean()
            else:
                ir_values[j] = arr.mean() / std

        self._ir = pd.Series(ir_values, index=self._feature_names, name="ir")
        masked = np.where(np.abs(ir_values) >= self.min_abs_ir, ir_values, 0.0)
        total = np.sum(np.abs(masked))
        if total < 1e-12:
            k = len(self._feature_names)
            self._weights = pd.Series(
                [1.0 / k] * k, index=self._feature_names, name="weight",
            )
        else:
            self._weights = pd.Series(
                masked / total, index=self._feature_names, name="weight",
            )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    @property
    def ir(self) -> pd.Series:
        """Per-factor information ratio (mean(IC) / std(IC) across chunks)."""
        assert self._ir is not None
        return self._ir.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class HalfLifeICWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Weight each factor by its **time-decayed** per-day cross-sectional IC.

    Unlike :class:`ICWeighter` which pools all (stock, date) samples into one
    Spearman correlation, this weighter computes a per-date cross-sectional IC
    and aggregates with exponential decay so recent dates dominate. Standard
    Barra/AlphaPortfolio convention.

    For each factor j:
      * On every training date t with >= ``min_stocks_per_day`` stocks,
        ``IC_{j,t} = corr(rank(factor_j[:, t]), rank(y[:, t]))``.
      * Weight ``IC_{j,t}`` by ``exp(-ln2 * (T - t) / halflife)`` — the
        most recent date carries weight 1.0, a date ``halflife`` business days
        earlier carries 0.5, etc.
      * ``IC_j = Σ_t w_t · IC_{j,t} / Σ_t w_t``.

    Weights are sign-preserving and L1-normalised — same final-score contract
    as :class:`ICWeighter`. Requires pooled cross-sectional input (MultiIndex
    with ``"date"`` level); raises in per-stock mode.
    """

    def __init__(
        self,
        halflife: float = 60.0,
        use_rank: bool = True,
        min_stocks_per_day: int = 10,
        min_abs_ic: float = 0.0,
    ):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        if min_stocks_per_day < 4:
            raise ValueError(
                f"min_stocks_per_day must be >= 4, got {min_stocks_per_day}"
            )
        if min_abs_ic < 0:
            raise ValueError(f"min_abs_ic must be >= 0, got {min_abs_ic}")
        self.halflife = halflife
        self.use_rank = use_rank
        self.min_stocks_per_day = min_stocks_per_day
        self.min_abs_ic = min_abs_ic
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None
        self._ic: pd.Series | None = None

    @staticmethod
    def _extract_dates(idx: pd.Index) -> np.ndarray:
        if isinstance(idx, pd.MultiIndex) and "date" in (idx.names or []):
            return idx.get_level_values("date").to_numpy()
        raise ValueError(
            "HalfLifeICWeighter requires y/X with a MultiIndex containing a "
            "'date' level (pooled cross-sectional data); got "
            f"index names={getattr(idx, 'names', None)!r}"
        )

    def _per_day_ic(
        self, fvals: np.ndarray, y: np.ndarray, dates: np.ndarray,
    ) -> pd.Series:
        """Per-date cross-sectional Spearman (or Pearson) IC. Index = date."""
        df = pd.DataFrame({"f": fvals, "y": y, "d": dates})
        sizes = df.groupby("d", sort=True)["d"].transform("size")
        df = df[sizes >= self.min_stocks_per_day]
        if df.empty:
            return pd.Series(dtype=float)
        if self.use_rank:
            df = df.assign(
                f=df.groupby("d", sort=False)["f"].rank(method="average"),
                y=df.groupby("d", sort=False)["y"].rank(method="average"),
            )
        # Pearson corr on (possibly ranked) values per group.
        def _corr_per_group(g: pd.DataFrame) -> float:
            a, b = g["f"].to_numpy(), g["y"].to_numpy()
            sa, sb = a.std(ddof=0), b.std(ddof=0)
            if sa < 1e-12 or sb < 1e-12:
                return np.nan
            return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
        return df.groupby("d", sort=True)[["f", "y"]].apply(_corr_per_group).dropna()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            self._ic = pd.Series(dtype=float)
            return

        dates = self._extract_dates(y.index)
        y_np = y.to_numpy(dtype=float)
        decay = np.log(2.0) / self.halflife

        ic_values = np.zeros(Xs.shape[1])
        for j, name in enumerate(self._feature_names):
            ic_series = self._per_day_ic(Xs[:, j], y_np, dates)
            if len(ic_series) < 2:
                ic_values[j] = 0.0
                continue
            # Most recent date → weight 1.0; ages measured in business-day
            # positions within the training window (not calendar days).
            ages = np.arange(len(ic_series) - 1, -1, -1, dtype=float)
            w = np.exp(-decay * ages)
            ic_values[j] = float(np.average(ic_series.to_numpy(), weights=w))

        self._ic = pd.Series(
            ic_values, index=self._feature_names, name="halflife_ic",
        )
        masked = np.where(np.abs(ic_values) >= self.min_abs_ic, ic_values, 0.0)
        total = np.sum(np.abs(masked))
        if total < 1e-12:
            k = len(self._feature_names)
            self._weights = pd.Series(
                [1.0 / k] * k, index=self._feature_names, name="weight",
            )
        else:
            self._weights = pd.Series(
                masked / total, index=self._feature_names, name="weight",
            )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    @property
    def ic(self) -> pd.Series:
        """Per-factor time-decayed IC (signed)."""
        assert self._ic is not None
        return self._ic.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class SharpeWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Weight each factor by the Sharpe of its quantile long-short portfolio.

    For each factor j, on every cross-section (date) with at least
    ``min_stocks_per_day`` valid stocks, rank stocks by the standardised factor
    value. The "factor portfolio" is long the top ``quantile`` of stocks and
    short the bottom ``quantile``; its single-date return is
    ``mean(y_top) - mean(y_bot)``. Aggregating across the training window gives
    a daily return series whose ``mean / std`` is the factor's Sharpe.

    Weights are sign-preserving (positive Sharpe → positive weight, negative
    Sharpe inverts the factor) and L1-normalised so prediction magnitudes are
    comparable across refits — mirrors :class:`ICWeighter` semantics.

    Differs from :class:`IRWeighter` in *what* is being averaged: IR aggregates
    cross-window IC, this aggregates a cross-sectional long-short P&L per date,
    which is closer to the economic value a portfolio strategy would realise.

    Requires the training ``y`` (and ``X``) to carry a ``MultiIndex`` with a
    ``"date"`` level (i.e. pooled mode); raises in per-stock mode.

    No annualisation factor is applied — the sqrt(252) multiplier would cancel
    in the subsequent L1 normalisation.
    """

    def __init__(
        self,
        quantile: float = 0.2,
        min_stocks_per_day: int = 10,
        min_valid_days: int = 5,
        min_abs_sharpe: float = 0.0,
    ):
        if not 0.0 < quantile < 0.5:
            raise ValueError(
                f"quantile must be in (0, 0.5), got {quantile}"
            )
        if min_stocks_per_day < 4:
            raise ValueError(
                f"min_stocks_per_day must be >= 4, got {min_stocks_per_day}"
            )
        if min_valid_days < 2:
            raise ValueError(
                f"min_valid_days must be >= 2, got {min_valid_days}"
            )
        if min_abs_sharpe < 0.0:
            raise ValueError(
                f"min_abs_sharpe must be >= 0, got {min_abs_sharpe}"
            )
        self.quantile = quantile
        self.min_stocks_per_day = min_stocks_per_day
        self.min_valid_days = min_valid_days
        self.min_abs_sharpe = min_abs_sharpe
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None
        self._sharpe: pd.Series | None = None

    @staticmethod
    def _extract_dates(idx: pd.Index) -> np.ndarray:
        if isinstance(idx, pd.MultiIndex) and "date" in (idx.names or []):
            return idx.get_level_values("date").to_numpy()
        raise ValueError(
            "SharpeWeighter requires y/X with a MultiIndex containing a 'date' "
            "level (pooled cross-sectional data); got "
            f"index names={getattr(idx, 'names', None)!r}"
        )

    def _single_factor_sharpe(
        self, fvals: np.ndarray, y: np.ndarray, dates: np.ndarray,
    ) -> float:
        df = pd.DataFrame({"f": fvals, "y": y, "d": dates})
        # Cross-sectional rank within each date in [0, 1].
        ranks = df.groupby("d", sort=False)["f"].rank(pct=True, method="average")
        counts = df.groupby("d", sort=False)["d"].transform("size")
        eligible = counts >= self.min_stocks_per_day
        if not eligible.any():
            return 0.0
        top_mask = eligible & (ranks >= 1 - self.quantile)
        bot_mask = eligible & (ranks <= self.quantile)
        top_per_day = df.loc[top_mask].groupby("d", sort=True)["y"].mean()
        bot_per_day = df.loc[bot_mask].groupby("d", sort=True)["y"].mean()
        ls = (top_per_day - bot_per_day).dropna()
        if len(ls) < self.min_valid_days:
            return 0.0
        arr = ls.to_numpy()
        std = arr.std(ddof=0)
        if std < 1e-12:
            return 0.0
        return float(arr.mean() / std)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            self._sharpe = pd.Series(dtype=float)
            return

        dates = self._extract_dates(y.index)
        y_np = y.to_numpy(dtype=float)
        sharpe_values = np.array([
            self._single_factor_sharpe(Xs[:, j], y_np, dates)
            for j in range(Xs.shape[1])
        ])
        self._sharpe = pd.Series(
            sharpe_values, index=self._feature_names, name="sharpe",
        )

        masked = np.where(
            np.abs(sharpe_values) >= self.min_abs_sharpe, sharpe_values, 0.0,
        )
        total = np.sum(np.abs(masked))
        if total < 1e-12:
            k = len(self._feature_names)
            self._weights = pd.Series(
                [1.0 / k] * k, index=self._feature_names, name="weight",
            )
        else:
            self._weights = pd.Series(
                masked / total, index=self._feature_names, name="weight",
            )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    @property
    def sharpe(self) -> pd.Series:
        """Per-factor Sharpe of the quantile long-short portfolio (signed)."""
        assert self._sharpe is not None
        return self._sharpe.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class RidgeWeighter(_LinearWeighterContributionsMixin, FactorWeighter, _StandardisingMixin):
    """Ridge regression weighter (L2-regularised linear regression on factors).

    Sister to :class:`ICWeighter`: instead of per-factor marginal IC, fits
    ``y = X·β + ε`` jointly with L2 penalty ``alpha``. Joint fit naturally
    suppresses redundant factors (correlated factors share weight) — useful
    when the Lasso selector still leaves residual collinearity.

    Weights are the ridge coefficients on standardised X, then L1-normalised
    so predictions are scale-comparable to other weighters. Set ``alpha=0``
    for plain OLS (no regularisation); typical values 0.1–10.

    Linear weighter contract: ``contributions(X) = standardise(X) * weights``,
    row sums equal ``predict(X)``. No look-ahead — fit only consumes (X, y).
    Closed-form solver (no sklearn dependency), matches LassoSelector pattern.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = False,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._feature_names: list[str] | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._weights: pd.Series | None = None
        self._raw_coef: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = self._fit_standardiser(X)
        if not self._feature_names:
            self._weights = pd.Series(dtype=float)
            self._raw_coef = pd.Series(dtype=float)
            return

        y_arr = y.to_numpy(dtype=float)
        if self.fit_intercept:
            y_arr = y_arr - y_arr.mean()

        # Closed-form ridge: (XᵀX + α·I)⁻¹·Xᵀy. Use lstsq-style solve via
        # np.linalg.solve for stability; pseudo-inverse fallback on singular.
        n, p = Xs.shape
        gram = Xs.T @ Xs + self.alpha * np.eye(p)
        rhs = Xs.T @ y_arr
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(gram) @ rhs

        self._raw_coef = pd.Series(coef, index=self._feature_names, name="ridge_coef")

        total = np.sum(np.abs(coef))
        if total < 1e-12:
            k = len(self._feature_names)
            self._weights = pd.Series(
                [1.0 / k] * k, index=self._feature_names, name="weight",
            )
        else:
            self._weights = pd.Series(
                coef / total, index=self._feature_names, name="weight",
            )

    def weights(self) -> pd.Series:
        assert self._weights is not None
        return self._weights.copy()

    @property
    def raw_coef(self) -> pd.Series:
        """Unnormalised ridge coefficients on standardised X (signed)."""
        assert self._raw_coef is not None
        return self._raw_coef.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._weights is None or self._weights.empty:
            return pd.Series(0.0, index=X.index)
        Xs = self._apply_standardiser(X)
        scores = Xs @ self._weights.to_numpy()
        return pd.Series(scores, index=X.index, name="score")


class LightGBMWeighter(FactorWeighter):
    """Tree-based weighter using LightGBM.

    ``fit(X, y)`` trains a regression LGB and caches mean|SHAP| as ``_weights``
    (computed once on training data, returned by ``weights()``).
    ``predict(X)`` runs ``booster.predict(X.values)``.
    ``contributions(X)`` runs ``booster.predict(X.values, pred_contrib=True)``
    and returns per-feature SHAP values (drops the trailing base-value column).

    Unlike linear weighters, this class does NOT inherit
    ``_StandardisingMixin`` — LightGBM is scale-invariant. Look-ahead safety
    rests on the same ABC contract: predict only consumes X, never y.

    Lazy import: ``import lightgbm`` happens inside ``fit()`` so the module
    can be imported without lightgbm installed.
    """

    def __init__(
        self,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        learning_rate: float = 0.05,
        num_iterations: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
        verbose: int = -1,
    ):
        if num_leaves <= 1:
            raise ValueError(f"num_leaves must be > 1, got {num_leaves}")
        if min_data_in_leaf <= 0:
            raise ValueError(f"min_data_in_leaf must be > 0, got {min_data_in_leaf}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be > 0, got {num_iterations}")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {max_depth}")

        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose

        self._booster = None
        self._feature_names: list[str] | None = None
        self._weights: pd.Series | None = None  # cached mean|SHAP|

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb  # lazy import

        if X.empty or len(y) == 0:
            self._feature_names = list(X.columns)
            self._weights = pd.Series(dtype=float)
            self._booster = None
            return

        self._feature_names = list(X.columns)
        dataset = lgb.Dataset(
            X.values, label=y.values, feature_name=self._feature_names,
        )
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "seed": self.random_state,
            "verbose": self.verbose,
        }
        self._booster = lgb.train(
            params, dataset, num_boost_round=self.num_iterations,
        )

        # Cache mean|SHAP| as weights (per Q1+Q5 design decisions).
        # pred_contrib returns shape (n, n_features + 1); last col is base value.
        contribs = self._booster.predict(X.values, pred_contrib=True)
        feature_contribs = contribs[:, :-1]
        mean_abs = np.abs(feature_contribs).mean(axis=0)
        self._weights = pd.Series(
            mean_abs, index=self._feature_names, name="lgb_mean_abs_shap",
        )

    def weights(self) -> pd.Series:
        if self._weights is None:
            raise RuntimeError("Weighter not fitted yet")
        return self._weights.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._booster is None:
            return pd.Series(0.0, index=X.index)
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"predict() missing columns: {missing}")
        Xn = X[self._feature_names].values
        preds = self._booster.predict(Xn)
        return pd.Series(preds, index=X.index, name="score")

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._booster is None:
            return pd.DataFrame(index=X.index)
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"contributions() missing columns: {missing}")
        Xn = X[self._feature_names].values
        contribs = self._booster.predict(Xn, pred_contrib=True)
        return pd.DataFrame(
            contribs[:, :-1], index=X.index, columns=self._feature_names,
        )
