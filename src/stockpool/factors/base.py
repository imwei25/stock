"""Factor ABC (panel-in / panel-out).

A Factor 是一个纯函数,从 ``Panel`` (``Mapping[str, pd.DataFrame]``,T × N OHLCV
宽表) 计算出同形状的因子值宽表。Subclasses MUST NOT mutate the input panel.

Look-ahead safety 由实现者保证: 第 ``i`` 行只能依赖前 ``i`` 行的数据。

每个因子有三类元数据,用于 HTML 选择器和粗筛:
  * ``sources``  : 因子来源, e.g. ("builtin",) / ("wq101",) / ("custom",)
  * ``types``    : 类型多标签, e.g. ("momentum", "time_series") / ("cross_sectional", "volume")
  * ``description`` : 一行人话说明,供 HTML / CLI 展示
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import pandas as pd


class Factor(ABC):
    """Compute a continuous factor wide-frame from an OHLCV panel."""

    # 子类用类属性覆盖即可。空 tuple 表示未声明。
    sources: tuple[str, ...] = ("builtin",)
    types: tuple[str, ...] = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in configs, registries, and column names."""
        ...

    @property
    def description(self) -> str:
        """Human-readable one-liner. Override on subclass."""
        return ""

    @abstractmethod
    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a T × N DataFrame aligned to ``panel['close']``.

        Implementers must:
          * not mutate ``panel`` or any of its fields;
          * ensure row ``i`` depends only on rows ``[:i+1]`` (no look-ahead);
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
