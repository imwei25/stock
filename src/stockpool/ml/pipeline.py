"""Two-step factor combination pipeline.

Step 1 — selector (Lasso): drops factors whose L1-regularised coefficient is 0.
Step 2 — weighter (IC / IR / equal): re-weights the survivors.

The pipeline keeps the selector around for inspection (``coef_``,
``selected_factors``) but only the weighter participates in ``predict``.

If the selector drops *every* factor, the pipeline silently uses all input
factors with equal weight as a fallback — better than predicting zeros and
producing a flat backtest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from stockpool.ml.selectors import FactorSelector, LassoSelector
from stockpool.ml.weighters import EqualWeighter, FactorWeighter, ICWeighter

log = logging.getLogger(__name__)


@dataclass
class FitInfo:
    """Diagnostic summary of one ``TwoStepPipeline.fit`` call."""
    n_samples: int
    n_input_factors: int
    selected_factors: list[str]
    fallback_used: bool          # True if selector dropped everything
    coef: pd.Series              # raw lasso coefs (standardised space)
    weights: pd.Series           # final per-factor weights


class TwoStepPipeline:
    """Lasso selection → IC (or IR / equal) weighting → score."""

    def __init__(
        self,
        selector: FactorSelector | None = None,
        weighter: FactorWeighter | None = None,
    ):
        self.selector = selector if selector is not None else LassoSelector()
        self.weighter = weighter if weighter is not None else ICWeighter()
        self.fit_info_: FitInfo | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FitInfo:
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        self.selector.fit(X, y)
        selected = self.selector.selected_factors()
        fallback = False
        if not selected:
            # Selector dropped everything → fall back to all input factors,
            # weighter still does the heavy lifting.
            selected = list(X.columns)
            fallback = True
            log.warning(
                "TwoStepPipeline: selector 一个因子都没选中,回退全部 %d 个"
                "因子等权进 weighter(行为突变,检查 alpha/数据质量)",
                len(selected),
            )

        X_sub = X[selected]
        self.weighter.fit(X_sub, y)

        # `coef_` is set by LassoSelector but other selectors may skip it.
        coef = getattr(self.selector, "coef_", pd.Series(dtype=float))
        self.fit_info_ = FitInfo(
            n_samples=len(X),
            n_input_factors=len(X.columns),
            selected_factors=selected,
            fallback_used=fallback,
            coef=coef.copy() if coef is not None else pd.Series(dtype=float),
            weights=self.weighter.weights(),
        )
        return self.fit_info_

    def fill_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """P3-9:用 fit 时各因子的均值填 NaN,而不是 0。

        weighter 的 standardiser 存了 fit-time μ;以 μ 填充,标准化后恰为
        0(真正的中性值)。``fillna(0)`` 只在 cs_zscore 预处理开启(μ≈0)
        时近似中性 —— 关掉 zscore 后 WQ rank 类因子 μ≈0.5,缺一个因子
        等于注入恒定方向偏置。weighter 没有 μ(如 LightGBM)时回退 0。
        """
        means = getattr(self.weighter, "_x_mean", None)
        names = getattr(self.weighter, "_feature_names", None)
        out = X.copy()
        if means is not None and names is not None and len(names) == len(means):
            mu = pd.Series(means, index=names)
            common = [c for c in out.columns if c in mu.index]
            if common:
                out[common] = out[common].fillna(mu[common])
        return out.fillna(0.0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.fit_info_ is None:
            raise RuntimeError("Pipeline not fitted yet")
        selected = self.fit_info_.selected_factors
        missing = [c for c in selected if c not in X.columns]
        if missing:
            raise KeyError(f"predict() missing columns: {missing}")
        return self.weighter.predict(X[selected])

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-bar per-factor contribution. Delegates to ``weighter.contributions``.

        Row sums equal ``self.predict(X)`` for linear weighters; for non-linear
        weighters the convention is weighter-specific (see weighter docstring).
        """
        if self.fit_info_ is None:
            raise RuntimeError("Pipeline not fitted yet")
        selected = self.fit_info_.selected_factors
        if not selected:
            return pd.DataFrame(index=X.index)
        missing = [c for c in selected if c not in X.columns]
        if missing:
            raise KeyError(f"contributions() missing columns: {missing}")
        return self.weighter.contributions(X[selected])
