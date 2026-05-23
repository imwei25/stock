"""Factor analysis library: rolling IC / IR / half-life / correlation / regime.

The pipeline is intentionally panel-first — every analytic function takes the
already-built OHLCV Panel and factor name list. This keeps the heavy lifting
(panel construction, factor computation) at the call site and makes the core
testable on synthetic data without touching the cache.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class FactorAnalysisResult:
    """Aggregate output of ``analyze_factors``.

    All Series are indexed by factor name (in input order).
    ``daily_ic`` and ``regime_ic`` keys are factor names / regime names.
    """
    factor_names: list[str]
    daily_ic: dict[str, pd.Series]
    mean_ic: pd.Series
    ic_ir: pd.Series
    abs_ic_mean: pd.Series
    half_life: pd.Series
    ic_correlation: pd.DataFrame
    regime_ic: dict[str, pd.Series]
    horizon: int
    ic_window: int
    n_stocks: int
    n_days: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "factor_names": list(self.factor_names),
            "daily_ic": {
                k: {
                    "index": [d.isoformat() for d in v.index],
                    "values": v.tolist(),
                } for k, v in self.daily_ic.items()
            },
            "mean_ic": self.mean_ic.to_dict(),
            "ic_ir": self.ic_ir.to_dict(),
            "abs_ic_mean": self.abs_ic_mean.to_dict(),
            "half_life": self.half_life.to_dict(),
            "ic_correlation": {
                "index": list(self.ic_correlation.index),
                "columns": list(self.ic_correlation.columns),
                "values": self.ic_correlation.values.tolist(),
            },
            "regime_ic": {
                k: v.to_dict() for k, v in self.regime_ic.items()
            },
            "horizon": self.horizon,
            "ic_window": self.ic_window,
            "n_stocks": self.n_stocks,
            "n_days": self.n_days,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, d: dict) -> "FactorAnalysisResult":
        ic_corr = pd.DataFrame(
            d["ic_correlation"]["values"],
            index=d["ic_correlation"]["index"],
            columns=d["ic_correlation"]["columns"],
        )
        return cls(
            factor_names=list(d["factor_names"]),
            daily_ic={
                k: pd.Series(v["values"], index=pd.to_datetime(v["index"]))
                for k, v in d["daily_ic"].items()
            },
            mean_ic=pd.Series(d["mean_ic"]),
            ic_ir=pd.Series(d["ic_ir"]),
            abs_ic_mean=pd.Series(d["abs_ic_mean"]),
            half_life=pd.Series(d["half_life"]),
            ic_correlation=ic_corr,
            regime_ic={k: pd.Series(v) for k, v in d["regime_ic"].items()},
            horizon=int(d["horizon"]),
            ic_window=int(d["ic_window"]),
            n_stocks=int(d["n_stocks"]),
            n_days=int(d["n_days"]),
            start_date=pd.Timestamp(d["start_date"]),
            end_date=pd.Timestamp(d["end_date"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FactorAnalysisResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Placeholder forward declarations — implemented in later tasks.
def compute_daily_ic(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 2")


def classify_regimes(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 3")


def analyze_factors(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 5")


def pick_top_factors(*args, **kwargs):  # noqa: D401
    raise NotImplementedError("implemented in Task 6")
