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
from typing import TYPE_CHECKING, Mapping, Sequence

import pandas as pd

if TYPE_CHECKING:
    from stockpool.config import MaskConfig

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


def _limit_threshold(code: str) -> float:
    """A 股按板块判定涨跌停幅度阈值。

    返回值是"abs 当日 ret 超过它即视为涨/跌停日"的阈值。略小于规则上限
    (0.098 < 0.10)是为了让真实涨停(实际 ret ≈ 0.099 因 round-to-cent)
    也能被命中。
    """
    if code.startswith(("300", "301", "688")):
        return 0.198  # 创业板 + 科创板 ±20%
    if code.startswith(("82", "83", "87", "43")):
        return 0.298  # 北交所 ±30%(项目 universe 不含,留兜底)
    return 0.098      # 主板沪深 ±10%


def _listing_mask(close: pd.DataFrame, min_days: int = 252) -> pd.DataFrame:
    """Mask=False 对每只股 panel 内"新上市后头 min_days 个交易日"。

    成熟股(panel 起点就有 close)视为已经上市 >1 年,全程 True。
    全 NaN 的股全程 False。
    """
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    for code in close.columns:
        series = close[code]
        first_valid = series.first_valid_index()
        if first_valid is None:
            mask[code] = False
            continue
        first_pos = close.index.get_loc(first_valid)
        if first_pos == 0:
            continue
        end_pos = min(first_pos + min_days, len(close))
        col_pos = mask.columns.get_loc(code)
        mask.iloc[first_pos:end_pos, col_pos] = False
    return mask


def _limit_threshold_for_config(code: str, config: "MaskConfig") -> float:
    """Like ``_limit_threshold`` but reads thresholds from a ``MaskConfig``."""
    if code.startswith(("300", "301", "688")):
        return config.limit_up_threshold_chinext
    if code.startswith(("82", "83", "87", "43")):
        return config.limit_up_threshold_bse
    return config.limit_up_threshold_main


def compute_tradability_mask(
    panel: Mapping[str, pd.DataFrame],
    config: "MaskConfig",
) -> pd.DataFrame:
    """从 OHLCV panel 计算可交易性 mask(close-side, paper B mask-first)。

    三条件 AND:
      1. |close ret| < per-code 涨跌停阈值
      2. volume > 0 (非停牌)
      3. 上市天数 ≥ min_listing_days
    """
    close = panel["close"]
    volume = panel["volume"]

    thresholds = pd.Series(
        {code: _limit_threshold_for_config(code, config) for code in close.columns}
    )

    ret = close / close.shift(1) - 1
    cond_not_limit = ret.abs().lt(thresholds, axis=1)
    cond_has_volume = volume > 0
    cond_listed = _listing_mask(close, min_days=config.min_listing_days)

    return cond_not_limit & cond_has_volume & cond_listed


def apply_mask(
    panel: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return a new panel with mask=False positions set to NaN across all fields.

    原 panel 不被修改 (``DataFrame.where`` 返回新对象)。
    """
    return {field: df.where(mask) for field, df in panel.items()}
