"""市值 (Size) 因子 —— Barra CNE5/CNE6 头号风格因子。

A 股长期存在显著的小市值溢价(也有阶段性反转),Size 是几乎所有商业风险
模型的第一大风格因子,本库此前只把市值用于 ``market_cap_neutralize`` 预处理,
缺一个可直接进选因子池的 Size 因子,这里补上。

数据来源:``factors.context.get_mcap_panel()`` —— 即 ``build_log_mcap_panel``
产出的 T×N ``log(总市值)`` 面板(用 hfq close × PIT 股本估算,精度见 CLAUDE.md
"已知不支持的能力" 一节)。注入时机:回测/AB/日报路径在 ``market_cap_neutralize``
开启**或**因子集含 ``log_mcap`` 时注入(``maybe_inject_mcap_panel``);
``factors analyze`` 无条件注入(它计算全部因子含 log_mcap)。仅当**无股本数据源**
(profit 表与快照都缺)时 ``build_log_mcap_panel`` 返回 None,此时本因子**优雅降级
为全 NaN**(不 fail loud),避免带崩 analyze 的全因子计算。

⚠️ 与预处理的交互:``market_cap_neutralize`` 会把所有因子对 log_mcap 做截面
回归取残差。若同时把 ``log_mcap`` 选进因子池又开 neutralize,本因子会被自身
残差化成 ≈0 噪声 —— 用 log_mcap 时应关闭 ``market_cap_neutralize``。
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.context import get_mcap_panel
from stockpool.factors.registry import register

log = logging.getLogger(__name__)


@register(
    "log_mcap",
    sources=("custom",),
    types=("size", "cross_sectional"),
    description="对数总市值 (Barra Size)。截面上小市值股取小值;A 股长期有小市值溢价。需与 market_cap_neutralize 关闭配合,否则会被残差化成 0。",
)
class LogMarketCapFactor(Factor):
    """Size = log(总市值)。直接取 context 注入的 log-mcap 面板。"""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "log_mcap"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        mcap = get_mcap_panel()
        if mcap is None or mcap.empty:
            # mcap 未注入(如 analyze 关 neutralize):优雅降级为全 NaN,
            # 不 fail loud。生产路径始终注入,不会走到这里。
            log.warning(
                "log_mcap: mcap_panel 未注入,返回全 NaN(若在生产路径出现"
                "请检查 set_mcap_panel 注入)"
            )
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        # 对齐到当前 panel 的日期 / 代码网格(mcap 可能覆盖不同范围)
        return mcap.reindex(index=close.index, columns=close.columns)
