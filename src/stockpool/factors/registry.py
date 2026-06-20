"""Factor registry: ``@register(name, ...)`` 装饰器 + 名字派发 + 双轴筛选。

注册时附带元数据(sources / types / description)。

Naming convention 不变:
  * ``"base_name"`` (exact match) 返回默认参数实例
  * ``"base_name_20"`` → ``cls.from_suffix_args(["20"])``
  * ``"base_name_20_5"`` → ``cls.from_suffix_args(["20", "5"])``

筛选: ``filter_specs(sources=..., types=...)`` 支持按来源 / 类型粗筛(集合相交逻辑)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Type

from stockpool.factors.base import Factor


@dataclass(frozen=True)
class FactorSpec:
    """注册表里一个因子条目的元数据。

    instance 是用默认参数构造的样例,主要给 HTML / CLI 展示用 ``name`` 和
    ``description``;真正喂给 pipeline 的实例由 ``make_factor(name)`` 现造。
    """
    base_name: str
    cls: Type[Factor]
    sources: tuple[str, ...]
    types: tuple[str, ...]
    description: str

    @property
    def default_name(self) -> str:
        """默认参数实例的展示名(即 ``cls().name``)。"""
        return self.cls().name


_REGISTRY: dict[str, FactorSpec] = {}


def register(
    name: str,
    *,
    sources: Iterable[str] = ("builtin",),
    types: Iterable[str] = (),
    description: str = "",
):
    """Decorator: register a ``Factor`` subclass under ``name`` with metadata.

    Args:
        name: base name (no suffix args).
        sources: factor origin tags, e.g. ("builtin",) / ("wq101",).
        types: type tags (multi-label), e.g. ("momentum", "time_series").
        description: one-line human-readable summary.
    """
    sources_t = tuple(sources)
    types_t = tuple(types)

    def _wrap(cls: Type[Factor]) -> Type[Factor]:
        if name in _REGISTRY:
            raise ValueError(f"Factor name already registered: {name!r}")
        # 把元数据也回写到类属性上,方便 instance.sources/types/description 直接读取
        cls.sources = sources_t
        cls.types = types_t
        if description and not getattr(cls, "description", None):
            # 用 property 覆盖,保持 description 是 instance attribute
            cls.description = property(lambda self, _d=description: _d)  # type: ignore[assignment]
        _REGISTRY[name] = FactorSpec(
            base_name=name, cls=cls,
            sources=sources_t, types=types_t, description=description,
        )
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
        return _REGISTRY[name].cls()
    # 最长前缀匹配
    best_prefix: str | None = None
    for k in _REGISTRY:
        if name.startswith(k + "_"):
            if best_prefix is None or len(k) > len(best_prefix):
                best_prefix = k
    if best_prefix is not None:
        suffix = name[len(best_prefix) + 1:]
        return _REGISTRY[best_prefix].cls.from_suffix_args(suffix.split("_"))
    raise KeyError(
        f"Unknown factor {name!r}. Registered base names: {sorted(_REGISTRY)}"
    )


def list_factors() -> list[str]:
    """All registered factor base names, sorted."""
    return sorted(_REGISTRY)


def list_specs() -> list[FactorSpec]:
    """All FactorSpec entries, sorted by base_name."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get_spec(name: str) -> FactorSpec:
    """Lookup a spec by base name."""
    if name not in _REGISTRY:
        raise KeyError(name)
    return _REGISTRY[name]


def filter_specs(
    sources: Iterable[str] | None = None,
    types: Iterable[str] | None = None,
    match: str = "any",
) -> list[FactorSpec]:
    """按来源 / 类型粗筛因子。

    Args:
        sources: 若给出,只保留 ``spec.sources`` 与之有交集(``match='any'``)
                 或包含全部(``match='all'``)的因子。
        types: 同上,对 ``spec.types`` 应用。
        match: ``"any"`` (默认, 集合相交) 或 ``"all"`` (包含全部要求标签)。
    """
    src_q = set(sources) if sources is not None else None
    typ_q = set(types) if types is not None else None
    out: list[FactorSpec] = []
    for spec in list_specs():
        if src_q is not None:
            spec_src = set(spec.sources)
            if match == "all" and not src_q.issubset(spec_src):
                continue
            if match == "any" and spec_src.isdisjoint(src_q):
                continue
        if typ_q is not None:
            spec_typ = set(spec.types)
            if match == "all" and not typ_q.issubset(spec_typ):
                continue
            if match == "any" and spec_typ.isdisjoint(typ_q):
                continue
        out.append(spec)
    return out


def all_sources() -> list[str]:
    """All unique source tags across the registry, sorted."""
    s: set[str] = set()
    for spec in _REGISTRY.values():
        s.update(spec.sources)
    return sorted(s)


def all_types() -> list[str]:
    """All unique type tags across the registry, sorted."""
    s: set[str] = set()
    for spec in _REGISTRY.values():
        s.update(spec.types)
    return sorted(s)
