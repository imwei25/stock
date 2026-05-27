"""Portfolio-level strategy ABC + precomputed-panel adapter.

A ``PortfolioStrategy`` sees the full universe at each bar and emits a score
per code. The engine handles top-K selection, eligibility, and rebalancing.

This is intentionally *not* a subclass of the per-stock ``Strategy`` — the
I/O semantics are different (single-stock generator vs cross-sectional
predictor) and forcing inheritance produces a dirty interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PortfolioStrategy(ABC):
    """Cross-sectional scoring abstract base.

    Look-ahead contract: ``predict_scores`` for ``date_t`` may only use
    ``panel_data[code]`` rows with ``date <= date_t``. ``PrecomputedScoreStrategy``
    satisfies this trivially because the panel is built once via
    ``precompute_scores_from_legacy`` (walk-forward happens inside the legacy
    strategy's ``generate_signals``).
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def predict_scores(
        self,
        date_t: pd.Timestamp,
        panel_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Return ``{code: score}``. Higher score = more attractive.

        Codes absent from the dict (or with NaN scores) are treated as
        un-investable for this bar.
        """


class PrecomputedScoreStrategy(PortfolioStrategy):
    """Wrap a (T × N) score panel built ahead of time.

    Decouples the engine from any legacy ``generate_signals`` training cost —
    the panel is computed once (and cacheable to parquet) per config hash.
    """

    def __init__(self, score_panel: pd.DataFrame, name: str = "precomputed"):
        # Defensive: ensure DatetimeIndex sorted ascending so .loc lookups
        # are well-defined.
        if not isinstance(score_panel.index, pd.DatetimeIndex):
            score_panel = score_panel.copy()
            score_panel.index = pd.to_datetime(score_panel.index)
        self._panel = score_panel.sort_index()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def panel(self) -> pd.DataFrame:
        return self._panel

    def predict_scores(
        self,
        date_t: pd.Timestamp,
        panel_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        date_t = pd.Timestamp(date_t)
        if date_t not in self._panel.index:
            return {}
        row = self._panel.loc[date_t].dropna()
        return {
            code: float(v)
            for code, v in row.items()
            if code in panel_data
        }
