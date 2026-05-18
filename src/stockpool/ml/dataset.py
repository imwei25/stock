"""Build (X, y) panels from OHLCV history for ML factor combination.

Conventions:

  * ``X``: each column is one factor's value; index is daily ``date``.
  * ``y``: the forward-N return ``close[t+N] / close[t] - 1``; aligned to ``X``
    by the *current* date ``t`` (so row ``t`` of ``y`` is the future).
  * Rows with any NaN (insufficient warmup or insufficient future) are dropped
    by ``align_xy``.
  * Pooled mode (multi-stock) concatenates row-wise and keeps a ``stock`` column
    as the second-level index so callers can group.

Look-ahead safety: ``X`` rows depend only on past data (factors are
look-ahead-safe by contract); ``y`` is computed from future closes and is
*only* used during ``fit``, never during ``predict``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from stockpool.factors.registry import make_factor


def build_factor_matrix(
    df: pd.DataFrame, factor_names: Sequence[str],
) -> pd.DataFrame:
    """Compute every named factor on ``df`` and return one DataFrame.

    Args:
        df: daily OHLCV with a ``date`` column.
        factor_names: registered factor names.

    Returns:
        DataFrame indexed by ``df["date"]`` with one column per factor.
    """
    if "date" not in df.columns:
        raise ValueError("df must have a 'date' column")
    cols: dict[str, pd.Series] = {}
    for name in factor_names:
        factor = make_factor(name)
        series = factor.compute(df)
        cols[factor.name] = series.reset_index(drop=True)
    out = pd.DataFrame(cols)
    out.index = pd.Index(df["date"].reset_index(drop=True), name="date")
    return out


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """``close[t+horizon] / close[t] - 1``, indexed by current date.

    Last ``horizon`` rows are NaN (no future data).
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    closes = df["close"].reset_index(drop=True)
    future = closes.shift(-horizon)
    y = future / closes - 1.0
    y.index = pd.Index(df["date"].reset_index(drop=True), name="date")
    return y


def align_xy(
    X: pd.DataFrame, y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with any NaN in either side. Return aligned (X, y)."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]


def build_panel(
    stocks_data: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str],
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Pool multi-stock data into a single (X, y) panel.

    Args:
        stocks_data: mapping ``{stock_code: daily_df}``. Each frame must have
            ``date`` and ``close`` columns.
        factor_names: registered factor names.
        horizon: forward-return window in bars.

    Returns:
        (X, y) where the rows are pooled across stocks and a MultiIndex of
        ``(stock, date)`` is set. Rows with any NaN are dropped.
    """
    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    for code, df in stocks_data.items():
        Xi = build_factor_matrix(df, factor_names)
        yi = forward_return(df, horizon)
        Xi, yi = align_xy(Xi, yi)
        if len(Xi) == 0:
            continue
        Xi = Xi.copy()
        Xi.index = pd.MultiIndex.from_product(
            [[code], Xi.index], names=["stock", "date"],
        )
        yi = yi.copy()
        yi.index = Xi.index
        X_parts.append(Xi)
        y_parts.append(yi)
    if not X_parts:
        empty_idx = pd.MultiIndex.from_arrays(
            [[], []], names=["stock", "date"],
        )
        return (
            pd.DataFrame(columns=list(factor_names), index=empty_idx),
            pd.Series(dtype=float, index=empty_idx),
        )
    X = pd.concat(X_parts, axis=0)
    y = pd.concat(y_parts, axis=0)
    return X, y


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (mean, std) for each column; std=0 columns get std=1."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def standardize_apply(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray,
) -> np.ndarray:
    return (X - mean) / std
