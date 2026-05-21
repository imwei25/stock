"""Build (X, y) panels from OHLCV history for ML factor combination.

Panel-first 接口(WQ101 cross-sectional 因子需要):

  * ``compute_factor_panel(panel, factor_names) -> dict[name, T×N DataFrame]``
    在整张 OHLCV Panel 上一次性算完所有因子。
  * ``forward_return_panel(close, horizon) -> T×N DataFrame``。
  * ``stack_panel_to_xy(factor_panel, fwd_ret, dropna=True) -> (X, y)``
    长表化为 (T·N) × F 的训练样本,index 是 (stock, date)。

兼容层(per-stock):

  * ``build_factor_matrix(df, factor_names) -> X`` 把单只票当 1-列 panel 跑,
    cross-sectional 因子的 rank 会退化为常数 —— 仅适合纯时间序列因子。
  * ``build_panel(pool, factor_names, horizon)`` 把 ``{code: daily_df}`` 字典
    封成 Panel,然后调用 panel-first 路径。
  * ``forward_return(df, horizon)`` 单股版。

Look-ahead 安全: factor row ``t`` 只依赖 ``[:t+1]``;forward return 用未来 close
计算,**只在训练时见 y,predict 时不见**。
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from stockpool.factors.registry import make_factor


# ─────────────────────────────────────────────────────────────────────────────
# Panel-first API
# ─────────────────────────────────────────────────────────────────────────────

def compute_factor_panel(
    panel: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """在 OHLCV Panel 上算所有因子,返回 ``{name: T×N DataFrame}``。"""
    out: dict[str, pd.DataFrame] = {}
    for name in factor_names:
        f = make_factor(name)
        out[f.name] = f.compute(panel)
    return out


def forward_return_panel(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """T×N forward return: ``close[t+h] / close[t] - 1``,末 h 行 NaN。"""
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    return close.shift(-horizon) / close - 1.0


def stack_panel_to_xy(
    factor_panel: Mapping[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """把因子宽表 + 收益宽表堆成长表 ``X (T·N × F)`` + ``y (T·N,)``。

    Index 是 ``MultiIndex[(stock, date)]``。``dropna=True`` 会移除任一因子或 y 为
    NaN 的行(用于训练)。
    """
    names = list(factor_panel.keys())
    if not names:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["stock", "date"])
        return pd.DataFrame(columns=names, index=empty_idx), pd.Series(dtype=float, index=empty_idx)

    ref = next(iter(factor_panel.values()))
    # 每个因子: stack 成 Series indexed by (date, code)
    parts: dict[str, pd.Series] = {}
    for nm in names:
        s = factor_panel[nm].stack(future_stack=True)
        s.index.set_names(["date", "stock"], inplace=True)
        parts[nm] = s.swaplevel("date", "stock").sort_index()
    X = pd.DataFrame(parts)
    y = fwd_ret.stack(future_stack=True)
    y.index.set_names(["date", "stock"], inplace=True)
    y = y.swaplevel("date", "stock").sort_index()
    y = y.reindex(X.index)

    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
    return X, y


def slice_stock_factor_row(
    factor_panel: Mapping[str, pd.DataFrame],
    code: str,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """从因子 panel 取出某只票在某日的 1 行 X(用于 predict)。"""
    cols: dict[str, float] = {}
    for nm, wide in factor_panel.items():
        if code in wide.columns and date in wide.index:
            cols[nm] = float(wide.at[date, code])
        else:
            cols[nm] = float("nan")
    return pd.DataFrame([cols], index=pd.Index([date], name="date"))


def slice_stock_factor_matrix(
    factor_panel: Mapping[str, pd.DataFrame],
    code: str,
) -> pd.DataFrame:
    """取某只票完整 T × F 因子矩阵(行索引 date)。"""
    out: dict[str, pd.Series] = {}
    for nm, wide in factor_panel.items():
        if code in wide.columns:
            out[nm] = wide[code]
        else:
            out[nm] = pd.Series(np.nan, index=wide.index)
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# 兼容层:per-stock single-frame API(把单股 df 包成 1-列 panel)
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_singleton_panel(df: pd.DataFrame, code: str = "_self_") -> dict[str, pd.DataFrame]:
    """把 long-form 单股 df 转成 1-列 panel。cross-sectional 因子会退化。"""
    if "date" not in df.columns:
        raise ValueError("df must have a 'date' column")
    idx = pd.DatetimeIndex(pd.to_datetime(df["date"]).values, name="date")
    return {
        field: pd.DataFrame({code: df[field].values}, index=idx)
        for field in ("open", "high", "low", "close", "volume")
    }


def build_factor_matrix(
    df: pd.DataFrame, factor_names: Sequence[str],
) -> pd.DataFrame:
    """Compute every named factor on one stock's ``df``, return T × F.

    Wraps ``df`` into a 1-stock panel; cross-sectional factors (rank,
    indneutralize) will return degenerate constants — use the pooled path
    (``build_panel`` / ``stack_panel_to_xy``) for those.
    """
    panel = _df_to_singleton_panel(df)
    code = next(iter(panel["close"].columns))
    cols: dict[str, pd.Series] = {}
    for name in factor_names:
        f = make_factor(name)
        wide = f.compute(panel)
        cols[f.name] = wide[code].reset_index(drop=True)
    out = pd.DataFrame(cols)
    out.index = pd.Index(df["date"].reset_index(drop=True), name="date")
    return out


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """``close[t+horizon] / close[t] - 1``,index = current date."""
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

    把 ``{code: daily_df}`` 装成 OHLCV Panel → 在 Panel 上算所有因子 →
    stack 成长表 ``(stock, date) × F``。这样 cross-sectional 因子拿到的是完整
    横截面,与 WQ101 论文语义一致。
    """
    if not stocks_data:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["stock", "date"])
        return (
            pd.DataFrame(columns=list(factor_names), index=empty_idx),
            pd.Series(dtype=float, index=empty_idx),
        )

    # 1) 构造 Panel:列 = stock code,行 = union dates
    per_stock: dict[str, pd.DataFrame] = {}
    for code, df in stocks_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        per_stock[code] = d.set_index("date").sort_index()
    all_dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")
    panel: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        panel[field] = pd.DataFrame(
            {code: d[field].reindex(idx) for code, d in per_stock.items()},
            index=idx,
        )

    # 2) 算因子
    fp = compute_factor_panel(panel, factor_names)
    fwd = forward_return_panel(panel["close"], horizon)
    X, y = stack_panel_to_xy(fp, fwd, dropna=True)
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
