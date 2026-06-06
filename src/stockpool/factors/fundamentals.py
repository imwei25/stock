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

    # ffill within pivot so that a code announced on date A fills later rows
    # that previously had NaN (e.g. a different code announced on date B > A).
    # Without this, per-column reindex ffill misses values in sparse pivots
    # where each code has a different pubDate.
    pivot = pivot.ffill()

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
    description="净资产收益率,股东每投 1 元年化能赚多少。长期 > 15% 通常是优质资产;严格按公告日 PIT 对齐(只看已披露)。",
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
    description="总资产收益率,每 1 元资产年化产出多少利润。衡量经营效率;与 ROE 对比可看出杠杆水平。",
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
    description="毛利率(销售毛利占收入比例)。高且稳 = 产品有定价权或成本控制好,常见于消费/高科技龙头。",
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
    description="净利率(净利润占收入比例),企业最终留下的赚钱能力,综合反映成本+费用+税负+其他损益。",
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
    description="营收同比增长率,看企业当下的成长性。配合毛利率可判断“高增长是否伴随毛利下滑”。",
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
    types=("fundamental", "cross_sectional", "contains_mcap"),
    description="市盈率(总市值 / 滚动 4 季净利润)。值越低越便宜,但要警惕周期顶反转;亏损公司返回 NaN。",
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
    types=("fundamental", "cross_sectional", "contains_mcap"),
    description="市净率(总市值 / 股东权益)。低 PB 常见于金融/周期股,需要配合 ROE 才能判断是否真便宜。",
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


@register(
    "market_cap",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "size"),
    description="总市值 (close × 总股本)。规模因子,小盘溢价 / 大盘稳定的常用代理。严格按公告日 PIT 对齐。",
)
class MarketCapFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "market_cap"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        balance = load_or_build_fundamentals("balance", cache_dir=_default_cache_dir())
        if balance is None or balance.empty:
            return pd.DataFrame(
                np.nan, index=panel["close"].index, columns=panel["close"].columns,
            )
        shares_panel = _pit_align(balance, "totalShare", panel["close"])
        return panel["close"] * shares_panel


@register(
    "log_market_cap",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "size"),
    description="log(总市值)。剥离市值 β 时常用;线性回归更稳定。NaN 出现在停牌 / 无股本数据 / mcap ≤ 0 时。",
)
class LogMarketCapFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "log_market_cap"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mcap = MarketCapFactor().compute(panel)
        return np.log(mcap.where(mcap > 0))
