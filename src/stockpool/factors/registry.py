"""Factor registry: ``@register("base_name")`` decorator + name-based factory.

Naming convention:

  * ``"base_name"`` (exact match) returns the factor with its default args.
  * ``"base_name_20"`` → ``cls.from_suffix_args(["20"])``.
  * ``"base_name_20_5"`` → ``cls.from_suffix_args(["20", "5"])``.

The base ``Factor.from_suffix_args`` converts each suffix part to ``int`` and
passes them positionally to ``__init__``. Override on subclasses if you need
a non-int parameter or a different parsing scheme.

Resolution prefers exact match, then the longest prefix match.
"""
from __future__ import annotations

from typing import Type

from stockpool.factors.base import Factor


_REGISTRY: dict[str, Type[Factor]] = {}


def register(name: str):
    """Decorator: register a ``Factor`` subclass under ``name``."""
    def _wrap(cls: Type[Factor]) -> Type[Factor]:
        if name in _REGISTRY:
            raise ValueError(f"Factor name already registered: {name!r}")
        _REGISTRY[name] = cls
        return cls
    return _wrap


def make_factor(name: str) -> Factor:
    """Instantiate a registered factor by canonical name.

    Examples:
        ``make_factor("macd_hist")`` → ``MACDHistFactor()``
        ``make_factor("momentum_20")`` → ``MomentumFactor(20)``
        ``make_factor("ma_slope_20_5")`` → ``MASlopeFactor(20, 5)``
    """
    if name in _REGISTRY:
        return _REGISTRY[name]()
    # Longest-prefix match (e.g. "ma_distance" beats "ma" if both existed).
    best_prefix: str | None = None
    for k in _REGISTRY:
        if name.startswith(k + "_"):
            if best_prefix is None or len(k) > len(best_prefix):
                best_prefix = k
    if best_prefix is not None:
        suffix = name[len(best_prefix) + 1:]
        return _REGISTRY[best_prefix].from_suffix_args(suffix.split("_"))
    raise KeyError(
        f"Unknown factor {name!r}. Registered base names: {sorted(_REGISTRY)}"
    )


def list_factors() -> list[str]:
    """All registered factor base names, sorted."""
    return sorted(_REGISTRY)
