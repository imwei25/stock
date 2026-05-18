"""Continuous-valued factor library for ML-based factor combination.

A ``Factor`` consumes a daily OHLCV DataFrame and returns a per-bar
``pd.Series`` of factor values aligned to ``df.index``. Factors are
look-ahead-safe by construction (row ``i`` may only depend on
``df.iloc[:i+1]``) so they can be used in walk-forward backtests.

Adding a new factor:

    from stockpool.factors import Factor, register

    @register("my_factor")
    class MyFactor(Factor):
        @property
        def name(self) -> str:
            return "my_factor"

        def compute(self, df: pd.DataFrame) -> pd.Series:
            return df["close"].pct_change(10)

Built-in factors are auto-registered when this package is imported.
"""
from stockpool.factors.base import Factor
from stockpool.factors.registry import (
    list_factors,
    make_factor,
    register,
)

# Side-effect: register built-in factors.
from stockpool.factors import technical  # noqa: F401

__all__ = ["Factor", "register", "make_factor", "list_factors"]
