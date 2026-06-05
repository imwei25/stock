"""基本面因子族 (论文 A 启发,baostock 5 张季度表).

PIT 对齐: 按 pubDate (公告日,**非** statDate 报告期末) 前向填充到日频,
防 ~1 个月未来泄露。Field 名以 Task 0 调研笔记为准。

7 个 base class:
  - 直接字段:roe, roa, gross_margin, net_margin
  - YOY 字段:revenue_yoy
  - 复合计算:pe, pb (close × totalShare / TTM(netProfit | equity))
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _default_cache_dir() -> Path:
    """项目 data/ 目录;fundamentals_loader 缓存在这里。"""
    # 本文件在 src/stockpool/factors/fundamentals.py
    # parents[3] = 项目根目录
    return Path(__file__).resolve().parents[3] / "data"


def _pit_align(
    raw: pd.DataFrame,
    field: str,
    panel_close: pd.DataFrame,
) -> pd.DataFrame:
    """long-form raw (code/pubDate/<field>) → T×N panel,按 pubDate ffill。

    PIT 保证: 日 t 只看到 pubDate ≤ t 的财报。
    """
    if raw is None or raw.empty or field not in raw.columns:
        return pd.DataFrame(
            np.nan, index=panel_close.index, columns=panel_close.columns
        )

    sub = raw[["code", "pubDate", field]].copy()
    sub[field] = pd.to_numeric(sub[field], errors="coerce")
    sub = sub.dropna(subset=["pubDate", field])
    if sub.empty:
        return pd.DataFrame(
            np.nan, index=panel_close.index, columns=panel_close.columns
        )

    # 同股同 pubDate 取最后一份(防重复)
    sub = sub.sort_values(["code", "pubDate"]).drop_duplicates(
        subset=["code", "pubDate"], keep="last"
    )
    pivot = sub.pivot(index="pubDate", columns="code", values=field)
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot = pivot.sort_index()

    # reindex(method='ffill') 保证 PIT: 日 t 只能看到 pubDate ≤ t 的财报
    aligned = pivot.reindex(panel_close.index, method="ffill")
    return aligned.reindex(columns=panel_close.columns)


class _ScalarFundamentalFactor(Factor):
    """直接字段类的共享逻辑: 一表一字段 PIT 对齐。"""

    _table: str = ""
    _field: str = ""

    def __init__(self):
        pass

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        raw = load_or_build_fundamentals(
            self._table, cache_dir=_default_cache_dir()
        )
        return _pit_align(raw, self._field, panel["close"])


@register(
    "roe",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="ROE (return on equity, profit.roeAvg) PIT 前向填充",
)
class ROEFactor(_ScalarFundamentalFactor):
    _table = "profit"
    _field = "roeAvg"

    @property
    def name(self) -> str:
        return "roe"


@register(
    "roa",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="ROA (return on assets, profit.roaAvg) PIT 前向填充",
)
class ROAFactor(_ScalarFundamentalFactor):
    _table = "profit"
    _field = "roaAvg"

    @property
    def name(self) -> str:
        return "roa"


@register(
    "gross_margin",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="毛利率 profit.gpMargin",
)
class GrossMarginFactor(_ScalarFundamentalFactor):
    _table = "profit"
    _field = "gpMargin"

    @property
    def name(self) -> str:
        return "gross_margin"


@register(
    "net_margin",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="净利率 profit.npMargin",
)
class NetMarginFactor(_ScalarFundamentalFactor):
    _table = "profit"
    _field = "npMargin"

    @property
    def name(self) -> str:
        return "net_margin"


@register(
    "revenue_yoy",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="营收同比 growth.YOYIncome",
)
class RevenueYoYFactor(_ScalarFundamentalFactor):
    _table = "growth"
    _field = "YOYIncome"

    @property
    def name(self) -> str:
        return "revenue_yoy"


@register(
    "pe",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="PE = close × totalShare / TTM(netProfit),亏损 → NaN",
)
class PEFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "pe"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        profit = load_or_build_fundamentals("profit", cache_dir=_default_cache_dir())
        balance = load_or_build_fundamentals("balance", cache_dir=_default_cache_dir())
        if profit is None or profit.empty or balance is None or balance.empty:
            return pd.DataFrame(
                np.nan, index=panel["close"].index, columns=panel["close"].columns
            )

        # TTM netProfit: 对每股按 pubDate 排序后 rolling 4 季 sum
        profit = profit.sort_values(["code", "pubDate"]).copy()
        profit["netProfit"] = pd.to_numeric(profit["netProfit"], errors="coerce")
        profit["_ttm"] = (
            profit.groupby("code")["netProfit"]
            .rolling(4, min_periods=4).sum()
            .reset_index(level=0, drop=True)
        )

        ni_panel = _pit_align(profit, "_ttm", panel["close"])
        shares_panel = _pit_align(balance, "totalShare", panel["close"])

        pe = panel["close"] * shares_panel / ni_panel
        # 亏损 / 缺数据 → NaN
        return pe.where((ni_panel > 0) & shares_panel.notna())


@register(
    "pb",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="PB = close × totalShare / totalShareholdersEquity",
)
class PBFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "pb"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        balance = load_or_build_fundamentals("balance", cache_dir=_default_cache_dir())
        if balance is None or balance.empty:
            return pd.DataFrame(
                np.nan, index=panel["close"].index, columns=panel["close"].columns
            )

        shares_panel = _pit_align(balance, "totalShare", panel["close"])
        equity_panel = _pit_align(balance, "totalShareholdersEquity", panel["close"])

        pb = panel["close"] * shares_panel / equity_panel
        return pb.where((equity_panel > 0) & shares_panel.notna())
