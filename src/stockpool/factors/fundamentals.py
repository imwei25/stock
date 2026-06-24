"""基本面因子族 (baostock 季度表,严格 PIT 对齐).

PIT 对齐: 按 pubDate (公告日,**非** statDate 报告期末) 前向填充到日频,
防 ~1 个月未来泄露。

直接字段类:roe / gross_margin / net_margin / netprofit_yoy / asset_yoy /
eps_yoy / debt_to_asset / cfo_to_np。复合计算:roa / pe / pb / market_cap /
log_market_cap / revenue_yoy / ep / bp。
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

# ffill 上限(交易日/并集行)。正常披露节奏下 pubDate 间隔最长约 6 个月,
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
      静默吞成全 NaN)。
    - 同 code 同 pubDate 多份报告(年报+一季报同日披露)→ 保留最新 statDate
      的报告期。
    - ffill 有上限 `_FFILL_LIMIT`,停止披露后不无限沿用。
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
    # 按 statDate 排序后 keep="last" → 保留最新报告期。
    sort_cols = ["code", "pubDate"] + (["statDate"] if has_stat else [])
    sub = sub.sort_values(sort_cols).drop_duplicates(
        subset=["code", "pubDate"], keep="last"
    )

    pivot = sub.pivot(index="pubDate", columns="code", values=field)
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot = pivot.sort_index()

    # PIT 前向填充。**不能**用 ``reindex(method='ffill')``:pivot 是稀疏的
    # (每个 pubDate 行只对当天披露的 code 有值,其余 code 为 NaN),reindex-ffill
    # 按"最近的前一个 pubDate 行"取值、不跳过 NaN —— 某 code 只有在"它自己的
    # pubDate 恰是全市场最近一次披露日"时才取到值,导致覆盖率塌到 ~1%。
    #
    # 正确做法:reindex 到 (pubDate ∪ 目标日) 并集,再**逐列** ffill(列内跳过 NaN、
    # 传播每个 code 的最近一次披露值),最后切回目标日。PIT 保证:并集已含 pubDate
    # 行,ffill 只向后传,日 t 只取 pubDate ≤ t 的值。limit=_FFILL_LIMIT 对停止
    # 披露的 code 起截断作用。
    union = pivot.index.union(panel_close.index)
    aligned = (
        pivot.reindex(union)
        .ffill(limit=_FFILL_LIMIT)
        .reindex(panel_close.index)
    )
    return aligned.reindex(columns=panel_close.columns)


def _ytd_ratio_to_ttm(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """YTD 累计比率 → TTM(近 4 个单季之和)。

    baostock profit 表的 roeAvg 等比率是"年初至今累计"口径(YTD)。
    构造:单季值 = 当季 YTD − 上季 YTD(同年内;Q1 即 YTD 自身),
    TTM = 近 4 个连续单季之和,按 statDate 对齐到季度网格。

    Returns:
        long-form DataFrame: code / pubDate / statDate / <field>_ttm。
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
    description="总资产收益率 = ROE / (Asset/Equity)。baostock 没有直接 ROA 字段;从 profit.roeAvg + balance.assetToEquity 联算(数学等价于 NI/Asset)。",
)
class ROAFactor(Factor):
    """ROA derived = ROE / (Asset/Equity).

    数学上: ROE = NI/E, A/E = Asset/E → ROA = NI/Asset = ROE/(A/E)。
    baostock query_profit_data 不返回 roaAvg 字段(只有 roeAvg),
    所以靠 balance.assetToEquity 把 ROE 杠杆比例除掉。
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "roa"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        profit = load_or_build_fundamentals("profit", cache_dir=_default_cache_dir())
        balance = load_or_build_fundamentals("balance", cache_dir=_default_cache_dir())
        if profit is None or profit.empty or balance is None or balance.empty:
            return pd.DataFrame(
                np.nan, index=panel["close"].index, columns=panel["close"].columns
            )

        roe_panel = _pit_align(profit, "roeAvg", panel["close"])
        a2e_panel = _pit_align(balance, "assetToEquity", panel["close"])
        # A/E > 0 才有意义(亏损公司 ROE 可能为负,这里只看资产杠杆,> 0)
        return roe_panel / a2e_panel.where(a2e_panel > 0)


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
    description="营收同比增长率,看企业当下的成长性。baostock growth 表没 YOYIncome 字段;从 profit.MBRevenue 自己 shift(4) 季算 YoY(银行/保险类 MBRevenue 可能为空 → NaN)。",
)
class RevenueYoYFactor(Factor):
    """Revenue YoY computed from profit.MBRevenue (4-quarter lag).

    baostock query_growth_data 只返回 YOYEquity/YOYAsset/YOYNI/YOYEPSBasic/YOYPNI,
    没有直接的 revenue YoY。从 profit.MBRevenue 自己算:同股按 pubDate 排序后
    shift(4) 拿同期上年 4 个季度前的值,YoY = (now - prior) / |prior|。

    覆盖率 caveat:profit.MBRevenue 在银行/保险类股票上经常为空字符串(主营业务
    收入对金融机构定义模糊),导致这些 code 整段 NaN。
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "revenue_yoy"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        profit = load_or_build_fundamentals("profit", cache_dir=_default_cache_dir())
        if profit is None or profit.empty:
            return pd.DataFrame(
                np.nan, index=panel["close"].index, columns=panel["close"].columns
            )

        profit = profit.sort_values(["code", "pubDate"]).copy()
        profit["_mbrev"] = pd.to_numeric(profit["MBRevenue"], errors="coerce")
        profit["_mbrev_lag4"] = profit.groupby("code")["_mbrev"].shift(4)
        # YoY 防 division-by-zero
        denom = profit["_mbrev_lag4"].abs()
        profit["_yoy"] = (profit["_mbrev"] - profit["_mbrev_lag4"]) / denom.where(denom > 1e-9)
        return _pit_align(profit, "_yoy", panel["close"])


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
        # NOTE: baostock 的 totalShare 在 *profit* 表里(不是误称的 balance)
        profit = load_or_build_fundamentals("profit", cache_dir=_default_cache_dir())
        if profit is None or profit.empty:
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
        if "totalShare" not in profit.columns:
            return _nan_panel(panel["close"])
        shares_panel = _pit_align(profit, "totalShare", panel["close"])

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
        # baostock balance 表没有 totalShare / totalShareholdersEquity 字段;
        # 用 profit 同期(YTD)反推 equity = netProfit / roeAvg。
        # baostock 的 netProfit 与 roeAvg 都是 YTD 累计(Q1=3 月,Q4=全年),
        # 比例消掉年内尺度差异 → 给出稳定的 equity 估算。**不要 rolling 4 季 TTM**,
        # 因为 4 行 YTD 相加 = Q1+H1+9个月+全年 ≈ 2.5× 年度,会把 equity 高估同样倍数。
        profit = load_or_build_fundamentals("profit", cache_dir=_default_cache_dir())
        if profit is None or profit.empty:
            return pd.DataFrame(
                np.nan, index=panel["close"].index, columns=panel["close"].columns
            )

        profit = profit.sort_values(["code", "pubDate"]).copy()
        profit["_netProfit"] = pd.to_numeric(profit["netProfit"], errors="coerce")
        profit["_roeAvg"] = pd.to_numeric(profit["roeAvg"], errors="coerce")
        positive_roe = profit["_roeAvg"].where(profit["_roeAvg"] > 1e-9)
        profit["_equity_implied"] = (profit["_netProfit"] / positive_roe).where(
            lambda s: s > 0
        )

        equity_panel = _pit_align(profit, "_equity_implied", panel["close"])
        if "totalShare" not in profit.columns:
            return _nan_panel(panel["close"])
        shares_panel = _pit_align(profit, "totalShare", panel["close"])

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
        # NOTE: baostock 的 totalShare 在 *profit* 表里(不是误称的 balance)
        profit = load_or_build_fundamentals("profit", cache_dir=_default_cache_dir())
        if profit is None or profit.empty:
            return pd.DataFrame(
                np.nan, index=panel["close"].index, columns=panel["close"].columns,
            )
        if "totalShare" not in profit.columns:
            return _nan_panel(panel["close"])
        shares_panel = _pit_align(profit, "totalShare", panel["close"])
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


# ── 成长(growth 表,YoY 同比;ported from composite-backtest 2026-06-24)──────


@register(
    "netprofit_yoy",
    sources=("custom",),
    types=("growth", "fundamental", "cross_sectional"),
    description="净利润同比增长率(growth.YOYNI),看企业当下的成长性。直接取 baostock growth 表字段。",
)
class NetProfitYoYFactor(_ScalarFundamentalFactor):
    _table = "growth"
    _field = "YOYNI"

    @property
    def name(self) -> str:
        return "netprofit_yoy"


@register(
    "asset_yoy",
    sources=("custom",),
    types=("growth", "fundamental", "cross_sectional"),
    description="总资产同比增速(growth.YOYAsset,投资/扩张速度)。多为反向因子:激进扩表者随后常跑输(投资因子)。方向由 IC 定。",
)
class AssetYoYFactor(_ScalarFundamentalFactor):
    _table = "growth"
    _field = "YOYAsset"

    @property
    def name(self) -> str:
        return "asset_yoy"


@register(
    "eps_yoy",
    sources=("custom",),
    types=("growth", "fundamental", "cross_sectional"),
    description="基本每股收益同比(growth.YOYEPSBasic),剔除股本摊薄的纯成长。与 netprofit_yoy 互补。",
)
class EPSYoYFactor(_ScalarFundamentalFactor):
    _table = "growth"
    _field = "YOYEPSBasic"

    @property
    def name(self) -> str:
        return "eps_yoy"


# ── 杠杆(balance 表,Barra Leverage)────────────────────────────────────────


@register(
    "debt_to_asset",
    sources=("custom",),
    types=("leverage", "fundamental", "cross_sectional"),
    description="资产负债率(balance.liabilityToAsset)。Barra Leverage 风格;高杠杆放大风险与顺/逆周期弹性。",
)
class DebtToAssetFactor(_ScalarFundamentalFactor):
    _table = "balance"
    _field = "liabilityToAsset"

    @property
    def name(self) -> str:
        return "debt_to_asset"


# ── 盈利质量(cash_flow 表,应计利润代理)────────────────────────────────────
#
# CFOToNP = 经营活动现金流 / 净利润。比值高 = 利润有真金白银现金支撑(应计成分
# 低、盈余质量高);低/为负 = 利润多为应计(应收/存货),盈余质量差。


@register(
    "cfo_to_np",
    sources=("custom",),
    types=("quality", "fundamental", "cross_sectional"),
    description="盈利质量:经营现金流/净利润(cash_flow.CFOToNP)。高=利润有现金支撑、应计成分低、更可持续。",
)
class CFOToNetProfitFactor(_ScalarFundamentalFactor):
    _table = "cash_flow"
    _field = "CFOToNP"

    @property
    def name(self) -> str:
        return "cfo_to_np"


# ── 估值(收益/账面 收益率口径,Barra Earnings Yield / Book-to-Price)─────────
#
# EP/BP 是 PE/PB 的倒数,但截面建模上更优:EP=epsTTM/close 在亏损时仍有定义
# (负盈利收益率);1/x 把 PE/PB 的右厚尾压成有界,winsorize+zscore 后更线性。


def _pe_ttm_panel(profit: pd.DataFrame, panel_close: pd.DataFrame) -> pd.DataFrame:
    """PE_TTM = close / epsTTM;epsTTM ≤ 0(亏损)→ NaN。"""
    eps = _pit_align(profit, "epsTTM", panel_close, table="profit")
    pe = panel_close / eps
    return pe.where(eps > 0)


@register(
    "ep",
    sources=("custom",),
    types=("value", "fundamental", "cross_sectional"),
    description="盈利收益率 EP = epsTTM/close(PE 的倒数,Barra Earnings Yield)。值大=便宜;亏损股为负值仍保留(比 PE 多覆盖)。",
)
class EarningsYieldFactor(Factor):
    """EP = epsTTM / close。与 PE 不同,亏损(eps<0)时返回负值而非 NaN。"""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "ep"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        profit = load_or_build_fundamentals(
            "profit", cache_dir=_default_cache_dir()
        )
        if profit is None or profit.empty:
            return _nan_panel(panel["close"])
        eps = _pit_align(profit, "epsTTM", panel["close"], table="profit")
        close = panel["close"].replace(0.0, np.nan)
        return eps / close


@register(
    "bp",
    sources=("custom",),
    types=("value", "fundamental", "cross_sectional"),
    description="账面市值比 BP = 1/PB(Barra Book-to-Price)。值大=便宜;比 PB 的右厚尾更线性。roe_ttm≤0 → NaN。",
)
class BookToPriceFactor(Factor):
    """BP = 1/PB = 1/(PE_TTM × ROE_TTM)。"""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "bp"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        profit = load_or_build_fundamentals(
            "profit", cache_dir=_default_cache_dir()
        )
        if profit is None or profit.empty:
            return _nan_panel(panel["close"])
        pe = _pe_ttm_panel(profit, panel["close"])
        ttm = _ytd_ratio_to_ttm(profit, "roeAvg")
        roe_ttm = _pit_align(
            ttm, "roeAvg_ttm", panel["close"], table="profit(roeAvg→TTM)"
        )
        pb = (pe * roe_ttm).where(roe_ttm > 0)
        pb = pb.where(pb > 0)  # 防 1/0 与负 PB
        return 1.0 / pb
