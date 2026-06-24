"""Cross-sectional rank-IC of a strategy's per-day score vs forward returns.

A/B 此前只比 equity 曲线指标(Sharpe/return/回撤/胜率)—— 端到端、含执行/成本/
sizing,小样本下噪声大。本模块附带计算 **预测信号质量**:每股每日 ``final_score``
的横截面 rank-IC,复用选因子那套 IC 数学:

  * ``compute_daily_ic`` —— 逐日横截面 Spearman 秩相关;
  * ``ic_ir = mean_ic / NW-σ`` —— ``_newey_west_std`` 的 Bartlett 核(lag = h−1)
    修正重叠 horizon-h forward return 的机械正自相关。

IC 衡量 score 的预测力,**不含 sizing/成本/执行** —— 与 Sharpe(衡量该信号能否
变现)互补。统计上 IC 信噪比远高于小样本 Sharpe,因子侧 A/B 应以 IC 为主判据、
Sharpe 作变现确认;执行/sizing 侧改动不改 score → IC 看不出差异,只能看 Sharpe。

注:本分支用收盘到收盘的 forward return(``forward_return_panel`` 无 open 基准),
故 label 口径为 close-to-close,与回测的 T+1 开盘成交略有错位,但对 score 预测力的
横截面排序衡量足够;open-to-open 口径见 composite-backtest 分支。
"""
from __future__ import annotations

import pandas as pd

from stockpool.factors_analysis import _newey_west_std, compute_daily_ic
from stockpool.ml.dataset import forward_return_panel


def cross_sectional_score_ic(
    score: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int,
    *,
    method: str = "spearman",
) -> dict:
    """单 horizon 的 score 横截面 rank-IC 汇总。

    Args:
        score: T×N score 宽表(date index, code columns),如 ``final_score``。
        close: T×N 收盘价宽表;自动对齐到 score 的 index/columns。
        horizon: 前瞻天数 h(bars),收盘到收盘 forward return。
        method: ``"spearman"``(秩 IC)或 ``"pearson"``。

    Returns:
        dict: ``mean_ic`` / ``ic_ir`` / ``abs_ic_mean`` / ``n_days`` / ``n_stocks``。
        score 为空或无有效日 → 三个 IC 值为 None。
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    empty = {
        "mean_ic": None, "ic_ir": None, "abs_ic_mean": None,
        "n_days": 0, "n_stocks": int(score.shape[1]) if score.ndim == 2 else 0,
    }
    if score.empty:
        return empty

    score = score.sort_index()
    close = close.reindex(index=score.index, columns=score.columns)
    fwd = forward_return_panel(close, horizon, "return")
    daily = compute_daily_ic(score, fwd, method=method)
    valid = daily.dropna()
    if valid.empty:
        return empty

    mean_ic = float(valid.mean())
    abs_ic = float(valid.abs().mean())
    # 重叠标签的日 IC 序列带机械正自相关 → IR 分母用 NW σ(口径同 analyze_factors)。
    nw = _newey_west_std(daily, lag=max(1, horizon - 1))
    ic_ir = float(mean_ic / nw) if (nw == nw and nw > 1e-12) else None
    return {
        "mean_ic": mean_ic, "ic_ir": ic_ir, "abs_ic_mean": abs_ic,
        "n_days": int(valid.shape[0]), "n_stocks": int(score.shape[1]),
    }


def panels_from_per_stock(
    per_stock: list[tuple[str, str, object]],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """把 per-stock A/B 的 ``EquityResult.score_frame`` 汇成 (score, close) 宽表。

    每个 ``score_frame`` 是 ``[date, open, close, final_score]`` 的逐日帧(回测时
    透出)。缺失/空帧的股票跳过;全空 → (None, None)。
    """
    scores: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    for code, _name, res in per_stock:
        sf = getattr(res, "score_frame", None)
        if sf is None or len(sf) == 0:
            continue
        sf = sf.copy()
        sf["date"] = pd.to_datetime(sf["date"])
        sf = sf.set_index("date")
        scores[code] = sf["final_score"]
        closes[code] = sf["close"]
    if not scores:
        return None, None
    return pd.DataFrame(scores).sort_index(), pd.DataFrame(closes).sort_index()


def arm_score_ic(
    per_stock: list[tuple[str, str, object]],
    horizons: list[int],
) -> dict[int, dict]:
    """逐 horizon 计算一个 arm 的 score 横截面 rank-IC。

    Returns ``{horizon: ic_dict}``;无可用 score_frame 时每个 horizon 返回空 dict
    (mean_ic 等为 None)。
    """
    score, close = panels_from_per_stock(per_stock)
    out: dict[int, dict] = {}
    for h in horizons:
        if score is None:
            out[h] = {"mean_ic": None, "ic_ir": None, "abs_ic_mean": None,
                      "n_days": 0, "n_stocks": 0}
        else:
            out[h] = cross_sectional_score_ic(score, close, h)
    return out
