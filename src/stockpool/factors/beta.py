"""市场敏感度 (Beta) 与特质波动率 (IVOL) 因子。

Barra 风格因子家族里 Beta 与 Residual Volatility 是两个核心维度,本库此前
完全缺失,这里补上。

**市场基准 = 截面等权平均日收益**(panel 各列当日收益的行均值),而非外部指数。
理由:
  * 因子契约是"panel 纯函数",注入外部指数会破坏 factor_panel 缓存签名
    (sig 不跟踪指数内容)、且需 caller wiring;
  * 全市场(training_universe=all,~4350 票)的等权平均收益本就是优良的市场
    代理,自洽、look-ahead 安全。
代价:在小票池(如 48 股评估池 / 单股日报)上,截面均值不是好的市场代理 ——
因此两个因子都标 ``cross_sectional``,只在全市场 / pooled 模式下有意义,与
breadth / industry_relative 等现有截面因子的约束一致。

数学(滚动窗 N,总体口径 ddof=0):
  rm        = ret.mean(axis=1)                          市场收益
  beta_i    = Cov(r_i, rm) / Var(rm)
  ivol_i    = sqrt( Var(r_i) − Cov(r_i, rm)² / Var(rm) )  残差标准差
            = sqrt( Var(r_i) · (1 − corr²) )
全部用滚动一阶/二阶矩拼出,无逐股回归循环。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _rolling_market_moments(
    panel: Mapping[str, pd.DataFrame], n: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """返回 (cov(r_i,rm), var(rm), 供复用的 ret) 的滚动估计 (总体口径)。"""
    ret = panel["close"].pct_change(fill_method=None)
    rm = ret.mean(axis=1)  # 截面等权市场收益 (Series)

    mean_i = ret.rolling(n, min_periods=n).mean()
    mean_m = rm.rolling(n, min_periods=n).mean()
    # E[r_i·rm] − E[r_i]·E[rm]
    cross = ret.mul(rm, axis=0).rolling(n, min_periods=n).mean()
    cov = cross.sub(mean_i.mul(mean_m, axis=0))
    var_m = (rm.pow(2)).rolling(n, min_periods=n).mean() - mean_m.pow(2)
    return cov, var_m.to_frame().iloc[:, 0], ret


@register(
    "beta",
    sources=("custom",),
    types=("beta", "cross_sectional", "time_series"),
    description="对截面等权市场收益的滚动 Beta(N≈60)。>1 进攻、<1 防守。仅全市场/pooled 模式有意义。",
)
class BetaFactor(Factor):
    def __init__(self, n: int = 60):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"beta_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        cov, var_m, _ = _rolling_market_moments(panel, self.n)
        var_m = var_m.replace(0.0, np.nan)
        return cov.div(var_m, axis=0)


@register(
    "ivol",
    sources=("custom",),
    types=("volatility", "cross_sectional", "time_series"),
    description="特质波动率:剔除市场 Beta 后的残差收益标准差(N≈60)。高 IVOL 常对应低预期收益(IVOL 之谜)。仅全市场/pooled 模式有意义。",
)
class IdiosyncraticVolFactor(Factor):
    def __init__(self, n: int = 60):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"ivol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        cov, var_m, ret = _rolling_market_moments(panel, self.n)
        mean_i = ret.rolling(self.n, min_periods=self.n).mean()
        var_i = (ret.pow(2)).rolling(self.n, min_periods=self.n).mean() - mean_i.pow(2)
        var_m_safe = var_m.replace(0.0, np.nan)
        resid_var = var_i.sub(cov.pow(2).div(var_m_safe, axis=0))
        # 数值误差可能让残差方差略小于 0 → clip 到 0 再开方
        return np.sqrt(resid_var.clip(lower=0.0))
