"""Adapter: per-stock ``Strategy`` → portfolio (T × N) score panel.

For each code, calls ``legacy.generate_signals(daily)`` and extracts the
``score_field`` column (default ``"final_score"`` — emitted by both
``CompositeVerdictStrategy`` and ``MLFactorStrategy``). Walk-forward training
happens inside the legacy strategy, so the resulting panel is look-ahead-safe
by construction.

Failure isolation: any per-stock exception is logged at WARNING and the code
is skipped — the panel still builds for the survivors.
"""
from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd

log = logging.getLogger("stockpool")


def precompute_scores_from_legacy(
    legacy_strategy,
    panel_data: Mapping[str, pd.DataFrame],
    score_field: str = "final_score",
) -> pd.DataFrame:
    """Build a (T × N) score panel by calling ``legacy.generate_signals`` per stock.

    Args:
        legacy_strategy: a per-stock ``Strategy`` whose ``generate_signals``
            output frame contains ``date`` and ``score_field`` columns.
        panel_data: ``{code: daily_df}`` — typically loaded from cache.
        score_field: column to extract (default ``"final_score"``).

    Returns:
        ``pd.DataFrame`` indexed by date, columns = codes, values = score.
        Codes whose ``generate_signals`` raises or omits ``score_field`` are
        skipped. If *all* codes fail, returns an empty frame.
    """
    # P0-5: ml_factor 等池化策略必须先 with_stock(code) 绑定当前股票,
    # 否则 _build_x_full 走单股退化路径 —— cross-sec 因子变常数/NaN、
    # 预处理(winsorize/zscore/mcap 中性化)全部缺失,训练(预处理面板)
    # 与预测(原始单股因子)特征不一致,score panel 的截面排序基本失效。
    can_bind = hasattr(legacy_strategy, "with_stock")
    series_by_code: dict[str, pd.Series] = {}
    for code, daily in panel_data.items():
        try:
            strat = legacy_strategy.with_stock(code) if can_bind else legacy_strategy
            sig = strat.generate_signals(daily)
        except Exception as e:  # noqa: BLE001 — failure-isolation contract
            log.warning("score panel: %s generate_signals failed (%s); skip", code, e)
            continue
        if score_field not in sig.columns:
            log.warning(
                "score panel: %s missing %r in generate_signals output; skip",
                code, score_field,
            )
            continue
        if "date" not in sig.columns:
            log.warning("score panel: %s missing 'date' column; skip", code)
            continue
        s = sig.set_index("date")[score_field]
        # Drop duplicate dates (defensive — shouldn't happen, but if it does
        # pivoting will explode).
        s = s[~s.index.duplicated(keep="last")]
        series_by_code[code] = s

    if not series_by_code:
        return pd.DataFrame()
    panel = pd.DataFrame(series_by_code)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()
