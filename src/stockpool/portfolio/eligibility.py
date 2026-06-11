"""Per-bar eligibility filter for the portfolio engine.

Decides "which codes are even allowed to enter the portfolio today", before
the engine applies score-based ranking. Three checks:

  * ``min_history_bars`` — enough bars to compute factors / metrics
  * ``exclude_st`` — name contains "ST" / "*ST" / etc.
  * ``min_avg_amount_20d`` — last-20-bar mean of ``close * volume``
    (volume 单位已在数据层统一为"股", P1-6; amount 单位为元)

Industry cap is *not* here: it depends on the engine's evolving target set
(per-target greedy walk), so it lives in the engine.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.config import PortfolioEligibilityConfig


class EligibilityFilter:
    """Decide eligible codes per bar.

    Args:
        cfg: ``PortfolioEligibilityConfig`` from the loaded yaml.
        name_map: ``{code: display_name}``. Used for ST detection only.
            Codes absent from ``name_map`` are *not* assumed ST (they pass
            the ST check; the only way to filter them is via the other rules).
    """

    def __init__(
        self,
        cfg: PortfolioEligibilityConfig,
        name_map: Mapping[str, str] | None = None,
    ):
        self.cfg = cfg
        self.name_map = dict(name_map or {})

    def eligible(
        self,
        date_t: pd.Timestamp,
        panel_data: Mapping[str, pd.DataFrame],
    ) -> set[str]:
        """Return the set of codes that pass all three checks at ``date_t``."""
        date_t = pd.Timestamp(date_t)
        out: set[str] = set()
        for code, daily in panel_data.items():
            if self.cfg.exclude_st and _is_st(self.name_map.get(code, "")):
                continue
            # Slice to bars <= date_t. Defensive: ensure date column exists.
            if "date" not in daily.columns or "close" not in daily.columns:
                continue
            df = daily[pd.to_datetime(daily["date"]) <= date_t]
            if len(df) < self.cfg.min_history_bars:
                continue
            if self.cfg.min_avg_amount_20d > 0:
                if "volume" not in df.columns:
                    # No volume → can't compute amount; treat as un-investable
                    # (matches Pool B behavior — no volume = not tradeable).
                    continue
                recent = df.tail(20)
                if len(recent) == 0:
                    continue
                # volume 单位已在数据层统一为"股"(P1-6),amount = close*volume (元)
                avg_amount = float(
                    (recent["close"].astype(float)
                     * recent["volume"].astype(float)).mean()
                )
                if pd.isna(avg_amount) or avg_amount < self.cfg.min_avg_amount_20d:
                    continue
            out.add(code)
        return out


def _is_st(name: str) -> bool:
    """ST detection: case-insensitive substring on the display name.

    Matches '*ST', 'ST', 'st' — anywhere in the name. Same heuristic as
    ``recommend_pool``.
    """
    if not name:
        return False
    return "ST" in name.upper()
