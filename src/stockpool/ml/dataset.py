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
    *,
    mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """在 OHLCV Panel 上算所有因子,返回 ``{name: T×N DataFrame}``。

    Args:
        panel: OHLCV panel.
        factor_names: 因子名列表。
        mask: 可选 T × N bool。若提供,会在算因子前 ``apply_mask``,
              mask=False 位置变 NaN,通过算子自然传播。default None → 旧行为不变。
    """
    if mask is not None:
        from stockpool.panel import apply_mask
        panel = apply_mask(panel, mask)
    out: dict[str, pd.DataFrame] = {}
    for name in factor_names:
        f = make_factor(name)
        out[f.name] = f.compute(panel)
    return out


def forward_return_panel(
    close: pd.DataFrame,
    horizon: int,
    label_type: str = "return",
) -> pd.DataFrame:
    """T×N forward-return panel with configurable label transform.

    Args:
        close: T × N 收盘价宽表 (date index, code columns).
        horizon: 前瞻天数 h。
        label_type:
            "return"          — close[t+h] / close[t] - 1 (legacy, default).
            "vol_adjusted"    — NotImplementedError (placeholder for future PR).
            "cross_sec_rank"  — NotImplementedError (placeholder for future PR).
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if label_type == "return":
        return close.shift(-horizon) / close - 1.0
    if label_type in ("vol_adjusted", "cross_sec_rank"):
        raise NotImplementedError(
            f"label_type={label_type!r} is not implemented in PR-A; "
            f"interface stub only."
        )
    raise ValueError(
        f"unknown label_type={label_type!r}; "
        f"expected one of: return, vol_adjusted, cross_sec_rank"
    )


def stack_panel_to_xy(
    factor_panel: Mapping[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """把因子宽表 + 收益宽表堆成长表 ``X (T·N × F)`` + ``y (T·N,)``。

    Index 是 ``MultiIndex[(stock, date)]`` 按 ``(stock, date)`` 字典序排列。
    ``dropna=True`` 会移除任一因子或 y 为 NaN 的行(用于训练)。

    Numpy-fast 实现:对每个因子做 ``reindex(dates, stocks).to_numpy().ravel('F')``
    并 ``column_stack``;MultiIndex 用 ``np.repeat`` + ``np.tile`` 构造。比逐因子
    ``DataFrame.stack()`` + ``swaplevel`` + ``sort_index`` 快 5-10×,语义一致。
    """
    names = list(factor_panel.keys())
    if not names:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["stock", "date"])
        return pd.DataFrame(columns=names, index=empty_idx), pd.Series(dtype=float, index=empty_idx)

    ref = factor_panel[names[0]]
    # 排序 dates / stocks 以保证输出 MultiIndex 是 (stock, date) 字典序。
    dates = pd.DatetimeIndex(ref.index).sort_values()
    stocks = pd.Index(sorted(ref.columns.tolist()))
    T, N = len(dates), len(stocks)

    if T == 0 or N == 0:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["stock", "date"])
        return pd.DataFrame(columns=names, index=empty_idx), pd.Series(dtype=float, index=empty_idx)

    # MultiIndex(stock, date) 字典序: stocks 每个重复 T 次,dates 平铺 N 次。
    stock_arr = np.repeat(stocks.to_numpy(), T)
    date_arr = np.tile(dates.to_numpy(), N)
    idx = pd.MultiIndex.from_arrays([stock_arr, date_arr], names=["stock", "date"])

    # X: 每个因子 reindex 后 F-order ravel(列优先,stock 慢、date 快),与 idx 顺序对齐。
    col_arrays = [
        factor_panel[nm].reindex(index=dates, columns=stocks).to_numpy(dtype=float).ravel(order="F")
        for nm in names
    ]
    X_arr = np.column_stack(col_arrays) if col_arrays else np.empty((T * N, 0))
    y_arr = fwd_ret.reindex(index=dates, columns=stocks).to_numpy(dtype=float).ravel(order="F")

    if dropna:
        mask = ~np.isnan(X_arr).any(axis=1) & ~np.isnan(y_arr)
        if not mask.all():
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            idx = idx[mask]

    X_df = pd.DataFrame(X_arr, index=idx, columns=names)
    y_s = pd.Series(y_arr, index=idx)
    return X_df, y_s


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


def forward_return(
    df: pd.DataFrame,
    horizon: int,
    label_type: str = "return",
) -> pd.Series:
    """单股 forward return,带 label_type 接口(与 forward_return_panel 一致)。

    Only ``label_type='return'`` is implemented in PR-A; other documented
    options raise NotImplementedError as interface placeholders.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if label_type == "return":
        closes = df["close"].reset_index(drop=True)
        y = closes.shift(-horizon) / closes - 1.0
        y.index = pd.Index(df["date"].reset_index(drop=True), name="date")
        return y
    if label_type in ("vol_adjusted", "cross_sec_rank"):
        raise NotImplementedError(
            f"label_type={label_type!r} is not implemented in PR-A; "
            f"interface stub only."
        )
    raise ValueError(
        f"unknown label_type={label_type!r}; "
        f"expected one of: return, vol_adjusted, cross_sec_rank"
    )


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
