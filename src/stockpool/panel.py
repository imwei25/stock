"""Panel: 跨股票宽表数据结构,WQ101 类横截面因子的底座。

约定:
  - 一个 Panel 是 ``Mapping[str, pd.DataFrame]``,key 是字段名(open/high/low/close/volume),
    value 是 T × N 宽表 —— 行索引 ``date``(``DatetimeIndex``),列索引 ``code``。
  - 所有字段共享同一组 (index, columns),便于做 ``rank(axis=1)`` 等横截面算子。
  - 上市前的行用 NaN 填充;实现者按需 ``dropna`` 或 ``ffill``。

构造方式: ``build_panel_from_cache(codes, history_days, cache_dir)`` 从 fetcher
已经缓存的 per-stock parquet 直接读取并对齐。不会触发网络请求 —— 调用方应先保证
缓存就绪(``fetcher.fetch_daily`` 跑过)。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

log = logging.getLogger(__name__)

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


def build_panel_from_cache(
    codes: Sequence[str],
    history_days: int,
    cache_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """从 ``data/<code>_daily.parquet`` 装一个 OHLCV 宽表 Panel。

    Args:
        codes: 股票代码列表。
        history_days: 取末尾 N 个交易日(对齐后的 union 日期再截尾)。
        cache_dir: parquet 缓存目录(通常 ``cfg.data.cache_dir``)。

    Returns:
        ``{field: DataFrame(T × N)}``,字段固定为 OHLCV_FIELDS。

    Raises:
        FileNotFoundError: 任一 code 的缓存文件不存在。
    """
    cache_dir = Path(cache_dir)
    per_stock: dict[str, pd.DataFrame] = {}
    for code in codes:
        p = cache_dir / f"{code}_daily.parquet"
        if not p.exists():
            raise FileNotFoundError(f"cache missing for {code}: {p}")
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        per_stock[code] = df.set_index("date").sort_index()

    # 取所有股票日期的并集,缺失填 NaN(上市前/停牌不强制 ffill,由因子自己决定)
    all_dates = sorted(set().union(*(df.index for df in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")

    panel: dict[str, pd.DataFrame] = {}
    for field in OHLCV_FIELDS:
        wide = pd.DataFrame(
            {code: df[field].reindex(idx) for code, df in per_stock.items()},
            index=idx,
        )
        wide.columns.name = "code"
        panel[field] = wide

    # 截尾到 history_days(按非全 NaN 的有效日)
    valid_days = panel["close"].dropna(how="all").index
    if len(valid_days) > history_days:
        keep = valid_days[-history_days:]
        for k in panel:
            panel[k] = panel[k].loc[keep]
    return panel


def panel_shape(panel: Mapping[str, pd.DataFrame]) -> tuple[int, int]:
    """(T, N) 形状,以 close 为准。"""
    return panel["close"].shape


def assert_panel_valid(panel: Mapping[str, pd.DataFrame]) -> None:
    """开发期断言:所有字段同形状、同索引、同列。"""
    ref = panel["close"]
    for k, v in panel.items():
        if v.shape != ref.shape:
            raise ValueError(f"panel field {k!r} shape {v.shape} != close {ref.shape}")
        if not v.index.equals(ref.index):
            raise ValueError(f"panel field {k!r} index mismatch with close")
        if not v.columns.equals(ref.columns):
            raise ValueError(f"panel field {k!r} columns mismatch with close")
