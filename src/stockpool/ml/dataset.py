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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd

from stockpool.factors.registry import make_factor

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from stockpool.config import MaskConfig, PreprocessConfig


# ─────────────────────────────────────────────────────────────────────────────
# Panel-first API
# ─────────────────────────────────────────────────────────────────────────────

def compute_factor_panel(
    panel: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """在 OHLCV Panel 上算所有因子,返回 ``{name: T×N DataFrame}``。

    **不对 panel 应用 tradability mask** — 时间序列因子(ts_corr/ts_rank/
    argmin 等)应该看到真实价格(包括涨停日的 +9.9%),那本身是有用信号。
    Mask 仅在标签 (``forward_return_panel``) 和模型训练样本筛选(通过
    label NaN 自然 dropna)上生效。详见
    ``docs/handoff/2026-05-31-mask-ab-investigation.md``。
    """
    out: dict[str, pd.DataFrame] = {}
    try:
        from tqdm import tqdm
        factor_iter = tqdm(
            list(factor_names),
            desc="compute_factor_panel",
            unit="factor",
            mininterval=1.0,
        )
    except ImportError:
        factor_iter = factor_names
    for name in factor_iter:
        f = make_factor(name)
        out[f.name] = f.compute(panel)
    return out


def forward_return_panel(
    close: pd.DataFrame,
    horizon: int,
    label_type: str = "return",
    *,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """T×N forward-return panel with configurable label transform.

    Args:
        close: T × N 收盘价宽表 (date index, code columns).
        horizon: 前瞻天数 h。
        label_type:
            "return"          — close[t+h] / close[t] - 1 (legacy, default).
            "vol_adjusted"    — NotImplementedError (placeholder for future PR).
            "cross_sec_rank"  — NotImplementedError (placeholder for future PR).
        mask: 可选 T × N bool。若提供做双向检查 — 要求 mask[t]=True ∧ mask[t+horizon]=True;
              不满足的 t 位置 y 值变 NaN。
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if label_type == "return":
        y = close.shift(-horizon) / close - 1.0
        if mask is not None:
            shifted = mask.shift(-horizon)
            label_valid = mask & shifted.where(shifted.notna(), False).astype(bool)
            y = y.where(label_valid)
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
    # Build a (F, T, N) 3D array for Rust dispatch (or numpy fallback).
    panels_3d = np.stack(
        [factor_panel[nm].reindex(index=dates, columns=stocks).to_numpy(dtype=float) for nm in names],
        axis=0,  # (F, T, N)
    )
    panels_3d = np.ascontiguousarray(panels_3d)

    # Try Rust dispatch for the column-stack reshape.
    _use_rust = False
    try:
        import stockpool_ops_rs as _rust_mod
        if hasattr(_rust_mod, "stack_factors_long"):
            import os as _os
            if _os.environ.get("STOCKPOOL_USE_PYTHON_OPS") != "1":
                _use_rust = True
    except ImportError:
        pass

    if _use_rust:
        # Rust: (F, T, N) → (T*N, F) with rayon parallelism over stocks.
        X_arr = _rust_mod.stack_factors_long(panels_3d)
    else:
        # Numpy fallback: transpose (F,T,N)→(N,T,F) + contiguous copy + reshape.
        # Equivalent to per-factor F-order ravel + column_stack but ~3× faster.
        # (N,T,F) contiguous reshape gives (N*T, F) in C order, matching idx layout.
        if len(names) > 0:
            X_arr = np.ascontiguousarray(panels_3d.transpose(2, 1, 0)).reshape(T * N, len(names))
        else:
            X_arr = np.empty((T * N, 0))
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
    df: pd.DataFrame,
    factor_names: Sequence[str],
) -> pd.DataFrame:
    """Compute every named factor on one stock's ``df``, return T × F.

    Wraps ``df`` into a 1-stock panel; cross-sectional factors (rank,
    indneutralize) will return degenerate constants — use the pooled path
    (``build_panel`` / ``stack_panel_to_xy``) for those.

    Panel 不应用 tradability mask — 时间序列因子需要真实价格(包括涨停日),
    详见 ``compute_factor_panel`` docstring。

    Args:
        df: 单股 daily DataFrame(含 date + OHLCV 列)。
        factor_names: 因子名列表。
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
    *,
    mask_config: "MaskConfig | None" = None,
    ipo_dates: Mapping[str, pd.Timestamp] | None = None,
    preprocess_cfg: "PreprocessConfig | None" = None,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Pool multi-stock data into a single (X, y) panel.

    把 ``{code: daily_df}`` 装成 OHLCV Panel → 在 Panel 上算所有因子 →
    (可选) 截面预处理 → stack 成长表 ``(stock, date) × F``。

    Args:
        stocks_data: ``{code: daily_df}``.
        factor_names: 因子名列表。
        horizon: forward return 前瞻天数。
        mask_config: 可选 MaskConfig,启用 tradability mask。
        ipo_dates: 可选 ``{code: IPO timestamp}``,传给 listing_mask 防止
            缓存历史短的成熟股被误标新上市;建议从
            ``stockpool.ipo_dates.load_or_build_ipo_dates`` 加载。
        preprocess_cfg: 可选 PreprocessConfig。非 None 且非全关时,在
            ``compute_factor_panel`` 之后对因子 panel 跑 winsorize / cs_zscore /
            industry_neutralize / mcap_neutralize 流水线 —— 与
            ``strategy_factory.build_factor_panel`` 行为一致,保证 fast path 和
            legacy fallback 拿到同一份预处理因子。``sector_map`` 从
            ``factors.context.get_sector_map()`` 取(caller 责任注入)。
        cache_dir: 仅当 ``preprocess_cfg.mcap_neutralize=True`` 时需要,用于
            ``stockpool.ml.mcap.build_log_mcap_panel`` 取 baostock balance 缓存。
            ``mcap_neutralize=True`` 且 ``cache_dir=None`` 时 log warning 并跳过
            mcap 步(其他步骤照常)。
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

    # 2) 算因子 — panel 不 mask 化(时间序列因子需要看真实价格)
    fp = compute_factor_panel(panel, factor_names)

    # 2.5) 截面预处理(可选)— 与 strategy_factory.build_factor_panel 行为对齐。
    if preprocess_cfg is not None:
        from stockpool.ml import preprocess as preproc_mod
        if not preproc_mod._is_all_off(preprocess_cfg):
            from stockpool.factors.context import get_sector_map
            from stockpool.factors.registry import list_specs
            sector_map = get_sector_map() or None
            types_map = {
                s.base_name: s.types
                for s in list_specs() if s.base_name in factor_names
            }
            log_mcap_panel = None
            if preprocess_cfg.mcap_neutralize:
                if cache_dir is None:
                    log.warning(
                        "build_panel: mcap_neutralize=True but cache_dir=None; "
                        "skipping mcap step (winsorize/zscore/industry still applied)"
                    )
                else:
                    from stockpool.ml.mcap import build_log_mcap_panel
                    log_mcap_panel = build_log_mcap_panel(panel, cache_dir=cache_dir)
            fp = preproc_mod.apply_preprocess_pipeline(
                fp, preprocess_cfg, sector_map=sector_map,
                factor_types=types_map, n_codes=len(stocks_data),
                log_mcap_panel=log_mcap_panel,
            )

    # 3) 算 forward-return,可选 mask 做双向标签检查
    mask: pd.DataFrame | None = None
    if mask_config is not None and mask_config.enabled:
        from stockpool.panel import compute_tradability_mask
        mask = compute_tradability_mask(panel, mask_config, ipo_dates=ipo_dates)

    fwd = forward_return_panel(panel["close"], horizon, mask=mask)
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
