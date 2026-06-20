"""ML factor-combination pipeline.

Two-step factor combination:

  1. ``LassoSelector`` — L1-regularised linear regression on (factors → forward
     return); zero-weight factors are dropped.
  2. ``ICWeighter`` / ``IRWeighter`` / ``EqualWeighter`` — re-weight the
     surviving factors by their predictive correlation with the target.

``TwoStepPipeline`` orchestrates both into a single ``fit / predict`` API.

See ``MLFactorStrategy`` in ``stockpool.backtesting.strategies`` for the
walk-forward backtest integration.
"""
from stockpool.ml.dataset import (
    build_factor_matrix,
    build_panel,
    compute_factor_panel,
    forward_return,
    forward_return_panel,
    slice_stock_factor_matrix,
    slice_stock_factor_row,
    stack_panel_to_xy,
)
from stockpool.ml.pipeline import TwoStepPipeline
from stockpool.ml.selectors import FactorSelector, LassoSelector
from stockpool.ml.weighters import (
    EqualWeighter,
    FactorWeighter,
    ICWeighter,
    IRWeighter,
)

__all__ = [
    "build_factor_matrix",
    "build_panel",
    "compute_factor_panel",
    "forward_return",
    "forward_return_panel",
    "slice_stock_factor_matrix",
    "slice_stock_factor_row",
    "stack_panel_to_xy",
    "FactorSelector",
    "LassoSelector",
    "FactorWeighter",
    "EqualWeighter",
    "ICWeighter",
    "IRWeighter",
    "TwoStepPipeline",
]
