"""Continuous-valued factor library for ML-based factor combination.

A ``Factor`` consumes a Panel (``Mapping[str, pd.DataFrame]`` of OHLCV wide frames,
T × N) and returns a T × N ``pd.DataFrame`` of factor values aligned to
``panel['close']``. Factors are look-ahead-safe by construction (row ``i`` may
only depend on rows ``[:i+1]``) so they can be used in walk-forward backtests.

Adding a new factor:

    from stockpool.factors import Factor, register

    @register(
        "my_factor",
        sources=("custom",),
        types=("momentum", "time_series"),
        description="说明一句话",
    )
    class MyFactor(Factor):
        @property
        def name(self) -> str:
            return "my_factor"

        def compute(self, panel):
            return panel["close"].pct_change(10, fill_method=None)

Built-in factors are auto-registered when this package is imported.
"""
from stockpool.factors.base import Factor
from stockpool.factors.registry import (
    FactorSpec,
    all_sources,
    all_types,
    filter_specs,
    get_spec,
    list_factors,
    list_specs,
    make_factor,
    register,
)

# Side-effect: register built-in factors.
from stockpool.factors import technical  # noqa: F401
from stockpool.factors import wq101  # noqa: F401
from stockpool.factors import custom  # noqa: F401

__all__ = [
    "Factor",
    "FactorSpec",
    "register",
    "make_factor",
    "list_factors",
    "list_specs",
    "get_spec",
    "filter_specs",
    "all_sources",
    "all_types",
]
