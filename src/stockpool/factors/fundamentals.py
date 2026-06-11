"""基本面因子族 (baostock 季度表,严格 PIT 对齐).

字段以实测 schema 为准(docs/handoff/2026-05-31-baostock-fundamentals-schema.md):

7 个 base class:
  - 直接字段:roe(profit.roeAvg)、gross_margin(profit.gpMargin)、
    net_margin(profit.npMargin)、netprofit_yoy(growth.YOYNI)
  - 复合计算:
    - roa = dupont.dupontROE / dupont.dupontAssetStoEquity(杜邦恒等式;
      dupont 表没有 dupontROA 字段)
    - pe  = close / profit.epsTTM(epsTTM 已是滚动 12 月口径)
    - pb  = pe_ttm × roe_ttm(P/E × E/B = P/B;roe_ttm 由 roeAvg YTD 差分构造)

PIT 对齐: 按 pubDate (公告日,**非** statDate 报告期末) 前向填充到日频,
防 ~1 个月未来泄露。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register

log = logging.getLogger(__name__)

# ffill 上限(交易日)。正常披露节奏下 pubDate 间隔最长约 6 个月(≈120 交易日),
# 250 容忍一次延迟披露;超过即视为停止披露(退市/长期停牌等),不再沿用旧值。
_FFILL_LIMIT = 250


def _default_cache_dir() -> Path:
    """项目 data/ 目录;fundamentals_loader 缓存在这里。"""
    # 本文件在 src/stockpool/factors/fundamentals.py
    # parents[3] = 项目根目录
    return Path(__file__).resolve().parents[3] / "data"


def _nan_panel(panel_close: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.nan, index=panel_close.index, columns=panel_close.columns
    )


def _pit_align(
    raw: pd.DataFrame,
    field: str,
    panel_close: pd.DataFrame,
    *,
    table: str = "",
) -> pd.DataFrame:
    """long-form raw (code/pubDate[/statDate]/<field>) → T×N panel,按 pubDate ffill。

    PIT 保证: 日 t 只看到 pubDate ≤ t 的财报。

    - 整表为空(拉取彻底失败)→ 优雅降级,返回全 NaN panel。
    - 表非空但缺请求字段 → **fail loud**,raise KeyError(防字段名拼错被
      静默吞成全 NaN,见 P1-1 审计)。
    - 同 code 同 pubDate 多份报告(年报+一季报同日披露)→ 保留最新 statDate
      的报告期(P2-18)。
    - ffill 有上限 `_FFILL_LIMIT`,停止披露后不无限沿用(P3-13)。
    """
    if raw is None or raw.empty:
        return _nan_panel(panel_close)
    if field not in raw.columns:
        raise KeyError(
            f"fundamentals table {table or '<unknown>'!r} has no field "
            f"{field!r}; actual columns: {list(raw.columns)}"
        )

    has_stat = "statDate" in raw.columns
    cols = ["code", "pubDate"] + (["statDate"] if has_stat else []) + [field]
    sub = raw[cols].copy()
    sub[field] = pd.to_numeric(sub[field], errors="coerce")
    sub["pubDate"] = pd.to_datetime(sub["pubDate"], errors="coerce")
    sub = sub.dropna(subset=["pubDate", field])
    if sub.empty:
        return _nan_panel(panel_close)

    # 同股同 pubDate 可能含多个报告期(年报 + 一季报同日披露):
    # 按 statDate 排序后 keep="last" → 保留最新报告期(P2-18)。
    sort_cols = ["code", "pubDate"] + (["statDate"] if has_stat else [])
    sub = sub.sort_values(sort_cols).drop_duplicates(
        subset=["code", "pubDate"], keep="last"
    )

    # P2-17: panel 起始早于多数 code 的最早 pubDate → 基本面覆盖不足,
    # 前段将为 NaN(quarters 拉取窗口不够长)。
    earliest = sub.groupby("code")["pubDate"].min()
    if len(earliest) and panel_close.index[0] < earliest.median():
        log.warning(
            "基本面覆盖不足: panel 起始 %s 早于多数 code 的最早 pubDate %s "
            "(table=%s field=%s) — 前段因子值将为 NaN,考虑增大 quarters 拉取窗口",
            panel_close.index[0].date(), earliest.median().date(), table, field,
        )

    pivot = sub.pivot(index="pubDate", columns="code", values=field)
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot = pivot.sort_index()

    # reindex(method='ffill') 保证 PIT: 日 t 只能看到 pubDate ≤ t 的财报;
    # limit=_FFILL_LIMIT: 停止披露超 ~1 年后不再沿用旧值(P3-13)。
    aligned = pivot.reindex(
        panel_close.index, method="ffill", limit=_FFILL_LIMIT
    )
    return aligned.reindex(columns=panel_close.columns)


def _ytd_ratio_to_ttm(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """YTD 累计比率 → TTM(近 4 个单季之和)。

    baostock profit 表的 roeAvg 等比率是"年初至今累计"口径(YTD)。
    构造:单季值 = 当季 YTD − 上季 YTD(同年内;Q1 即 YTD 自身),
    TTM = 近 4 个连续单季之和,按 statDate 对齐到季度网格。

    **近似**:对 YTD 比率直接差分,忽略了分母(平均净资产等)在年内各季
    之间的漂移 —— 严格意义上 YTD 比率不可加,但季内净资产变动通常远小于
    利润波动,误差可接受。

    缺季处理:把 statDate 落到完整季度网格上,任何缺季会让"单季差分"与
    随后 4 季窗口的 TTM 自动变 NaN,不会跨期错配。

    Args:
        raw: long-form,需含 code / pubDate / statDate / <field>
        field: YTD 累计比率字段名(如 roeAvg)

    Returns:
        long-form DataFrame: code / pubDate / statDate / <field>_ttm,
        每行对应 raw 中一份(去重后的)季报。
    """
    out_field = f"{field}_ttm"
    required = {"code", "pubDate", "statDate", field}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(
            f"_ytd_ratio_to_ttm requires columns {sorted(required)}; "
            f"missing {sorted(missing)}; actual columns: {list(raw.columns)}"
        )

    sub = raw[["code", "pubDate", "statDate", field]].copy()
    sub[field] = pd.to_numeric(sub[field], errors="coerce")
    sub["pubDate"] = pd.to_datetime(sub["pubDate"], errors="coerce")
    sub["statDate"] = pd.to_datetime(sub["statDate"], errors="coerce")
    sub = sub.dropna(subset=["pubDate", "statDate", field])
    if sub.empty:
        return pd.DataFrame(columns=["code", "pubDate", "statDate", out_field])

    # 同 code 同 statDate 多份(更正公告)→ 保留最新 pubDate 那份
    sub = sub.sort_values(["code", "statDate", "pubDate"]).drop_duplicates(
        subset=["code", "statDate"], keep="last"
    )

    parts: list[pd.DataFrame] = []
    for code, g in sub.groupby("code", sort=False):
        quarters = g["statDate"].dt.to_period("Q")
        ytd = pd.Series(g[field].to_numpy(), index=quarters)
        # 完整季度网格:缺季留 NaN,差分/rolling 自动失效而非错配
        grid = ytd.reindex(
            pd.period_range(quarters.iloc[0], quarters.iloc[-1], freq="Q")
        )
        single = grid.where(
            grid.index.quarter == 1,  # Q1: 单季 = YTD 自身
            grid - grid.shift(1),     # 其余: 当季 YTD − 上季 YTD(同年内)
        )
        ttm = single.rolling(4, min_periods=4).sum()
        parts.append(pd.DataFrame({
            "code": code,
            "pubDate": g["pubDate"].to_numpy(),
            "statDate": g["statDate"].to_numpy(),
            out_field: ttm.reindex(quarters).to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


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
        return _pit_align(raw, self._field, panel["close"], table=self._table)


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
class ROAFactor(Factor):
    """ROA = dupontROE / dupontAssetStoEquity(杜邦恒等式)。

    baostock dupont 表没有 dupontROA 直接字段(实测 schema 见
    docs/handoff/2026-05-31-baostock-fundamentals-schema.md §5),
    按 ROA = ROE / 权益乘数 = (NI/equity) / (asset/equity) = NI/asset 派生。
    两个字段同表同行,PIT 对齐口径一致。
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "roa"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        dupont = load_or_build_fundamentals(
            "dupont", cache_dir=_default_cache_dir()
        )
        if dupont is None or dupont.empty:
            return _nan_panel(panel["close"])

        roe = _pit_align(dupont, "dupontROE", panel["close"], table="dupont")
        mult = _pit_align(
            dupont, "dupontAssetStoEquity", panel["close"], table="dupont"
        )
        roa = roe / mult
        # 权益乘数 ≤ 0(资不抵债)无意义 → NaN
        return roa.where(mult > 0)


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
    "netprofit_yoy",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="净利润同比增长率(growth.YOYNI),看企业当下的成长性。原 revenue_yoy 因子因 baostock 无营收同比字段(YOYIncome 不存在)改名并换源。",
)
class NetProfitYoYFactor(_ScalarFundamentalFactor):
    _table = "growth"
    _field = "YOYNI"

    @property
    def name(self) -> str:
        return "netprofit_yoy"


def _pe_panel(
    profit: pd.DataFrame, panel_close: pd.DataFrame
) -> pd.DataFrame:
    """PE_TTM = close / epsTTM;epsTTM ≤ 0(亏损)→ NaN。"""
    eps = _pit_align(profit, "epsTTM", panel_close, table="profit")
    pe = panel_close / eps
    return pe.where(eps > 0)


@register(
    "pe",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="市盈率 TTM(close / epsTTM,baostock epsTTM 已是滚动 12 月口径)。值越低越便宜,但要警惕周期顶反转;亏损公司返回 NaN。",
)
class PEFactor(Factor):
    """PE_TTM = close / profit.epsTTM。

    epsTTM 是 baostock 现成的滚动 12 月每股收益,直接用,
    避免对"年初累计值 netProfit"做 rolling(4).sum 的口径错误
    (YTD 累计值滚动求和会双重计数,≈2.5 倍年利润)。
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "pe"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        profit = load_or_build_fundamentals(
            "profit", cache_dir=_default_cache_dir()
        )
        if profit is None or profit.empty:
            return _nan_panel(panel["close"])
        return _pe_panel(profit, panel["close"])


@register(
    "pb",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="市净率(PE_TTM × ROE_TTM,数学上 P/E × E/B = P/B)。低 PB 常见于金融/周期股,需要配合 ROE 才能判断是否真便宜。",
)
class PBFactor(Factor):
    """PB = PE_TTM × ROE_TTM(P/E × E/B = P/B)。

    balance 表没有 totalShareholdersEquity / totalShare 字段,无法直接算
    市值/净资产;改用恒等式 P/B = P/E × E/B = (close/epsTTM) × roe_ttm。

    roe_ttm 由 profit.roeAvg(YTD 累计比率)经 ``_ytd_ratio_to_ttm`` 构造:
    单季 roe = 当季 YTD − 上季 YTD(同年内,Q1 即自身),TTM = 近 4 个单季
    之和,按 statDate 对齐。**近似**:YTD 比率差分忽略了分母平均净资产的
    季内漂移,严格意义上比率不可加,误差通常很小。

    亏损(epsTTM ≤ 0)或 roe_ttm ≤ 0 → NaN。
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "pb"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        profit = load_or_build_fundamentals(
            "profit", cache_dir=_default_cache_dir()
        )
        if profit is None or profit.empty:
            return _nan_panel(panel["close"])

        pe = _pe_panel(profit, panel["close"])
        ttm = _ytd_ratio_to_ttm(profit, "roeAvg")
        roe_ttm = _pit_align(
            ttm, "roeAvg_ttm", panel["close"], table="profit(roeAvg→TTM)"
        )
        pb = pe * roe_ttm
        return pb.where(roe_ttm > 0)
