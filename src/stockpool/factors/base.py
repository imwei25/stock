"""Factor ABC.

A Factor is a pure function from a daily OHLCV DataFrame to a per-bar
Series of continuous values. Subclasses must NEVER mutate the input frame.
Look-ahead safety is the implementer's responsibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """Compute a continuous factor series from daily OHLCV."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in configs, registries, and column names."""
        ...

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series aligned to ``df.index`` with the factor's values.

        Implementers must:
          * not mutate ``df``;
          * ensure row ``i`` depends only on ``df.iloc[:i+1]``;
          * mark insufficient-warmup rows as ``NaN``.
        """
        ...

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "Factor":
        """Instantiate from positional integer args parsed from a factor name.

        Default behaviour treats every suffix part as an int. Override if your
        factor takes non-int parameters or a different parsing scheme.
        """
        return cls(*[int(a) for a in args])  # type: ignore[call-arg]
