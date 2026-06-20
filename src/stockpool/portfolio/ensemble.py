"""Staggered ensemble runner (PR-3 of the portfolio framework spec).

Runs the same ``PortfolioEngine`` configuration across ``N`` ``start_offset``
values (``0..N-1``). For ``rebalance_n_days=n`` and ``N=n``, the rebalance
bar index sets are non-overlapping — each offset is a separate copy of "if
you'd happened to deploy on day k of the cycle".

Three aggregations come back:

  * ``ensemble_curve`` — equal-weighted mean of the N equity curves. This is
    the curve you'd see if you split capital into N tranches and started
    one per day — i.e. the actually-deployable smooth-rollover portfolio.
  * ``envelope`` — per-bar quantile summary (min / p25 / median / p75 / max)
    used to draw the sensitivity band in the HTML report.
  * ``aggregated_metrics`` — median / min / max of headline metrics across
    offsets, plus the ensemble curve's own metrics.

PR-3 runs offsets serially. ``ProcessPoolExecutor`` parallelisation is in
scope only as a Spec §12 followup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from stockpool.backtesting.metrics import compute_metrics
from stockpool.portfolio.engine import PortfolioEngine
from stockpool.portfolio.result import PortfolioBacktestResult


@dataclass
class EnsembleResult:
    """Aggregated result over a staggered ensemble run."""
    individual_results: list[PortfolioBacktestResult]
    ensemble_curve: pd.DataFrame              # date / equity (mean of all offsets)
    envelope: pd.DataFrame                    # date / min / p25 / median / p75 / max
    aggregated_metrics: dict
    strategy_name: str

    @property
    def n_offsets(self) -> int:
        return len(self.individual_results)


class StaggeredRunner:
    """Run a ``PortfolioEngine`` across ``N`` ``start_offset`` values.

    Uses an engine factory rather than a single engine because each run
    needs fresh portfolio state (cash + open positions reset).
    """

    def __init__(
        self,
        engine_factory: Callable[[], PortfolioEngine],
        risk_free_rate: float = 0.02,
    ):
        self._engine_factory = engine_factory
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        panel_data: Mapping[str, pd.DataFrame],
        n_offsets: int,
    ) -> EnsembleResult:
        if n_offsets < 1:
            raise ValueError(f"n_offsets must be >= 1, got {n_offsets}")
        results: list[PortfolioBacktestResult] = []
        for k in range(n_offsets):
            engine = self._engine_factory()
            results.append(engine.run(panel_data, start_offset=k))
        return self._aggregate(results)

    # ---- aggregation ----

    def _aggregate(
        self, results: list[PortfolioBacktestResult],
    ) -> EnsembleResult:
        if not results:
            empty_curve = pd.DataFrame({"date": [], "equity": []})
            empty_env = pd.DataFrame(
                columns=["date", "min", "p25", "median", "p75", "max"],
            )
            return EnsembleResult(
                individual_results=[],
                ensemble_curve=empty_curve,
                envelope=empty_env,
                aggregated_metrics={},
                strategy_name="ensemble",
            )

        # Build a wide equity frame: rows = dates (union, sorted),
        # columns = offset_k. Reindex each curve to that union; ffill any
        # gap (cheap & defensive — engine emits a curve point per bar so
        # all offsets should share the same date axis).
        all_dates = sorted(set().union(
            *[set(pd.to_datetime(r.curve["date"])) for r in results if not r.curve.empty]
        ))
        all_dates = pd.DatetimeIndex(all_dates)
        wide_cols: dict[str, pd.Series] = {}
        for k, r in enumerate(results):
            if r.curve.empty:
                wide_cols[f"k{k}"] = pd.Series(np.nan, index=all_dates)
                continue
            s = pd.Series(
                r.curve["equity"].values,
                index=pd.to_datetime(r.curve["date"]),
            ).reindex(all_dates).ffill()
            wide_cols[f"k{k}"] = s
        wide = pd.DataFrame(wide_cols, index=all_dates)

        ensemble_equity = wide.mean(axis=1).values
        envelope = pd.DataFrame({
            "date": all_dates,
            "min": wide.min(axis=1).values,
            "p25": wide.quantile(0.25, axis=1).values,
            "median": wide.median(axis=1).values,
            "p75": wide.quantile(0.75, axis=1).values,
            "max": wide.max(axis=1).values,
        })
        ensemble_curve = pd.DataFrame({
            "date": all_dates,
            "equity": ensemble_equity,
        })

        # Ensemble metrics: treat the mean curve as a stand-alone strategy.
        ensemble_metrics = compute_metrics(
            pd.Series(ensemble_equity), trades=[],
            risk_free_rate=self.risk_free_rate,
        )

        # Per-offset headline metrics for the cross-offset summary.
        headline_keys = ("total_return", "annualized_return", "sharpe", "max_drawdown")
        per_offset = {k: {} for k in headline_keys}
        for r in results:
            for k in headline_keys:
                per_offset[k].setdefault("vals", []).append(r.metrics.get(k))
        agg: dict = {"ensemble": ensemble_metrics, "per_offset": {}}
        for k, slot in per_offset.items():
            vals = [v for v in slot["vals"] if v is not None]
            if not vals:
                agg["per_offset"][k] = {"median": None, "min": None, "max": None}
                continue
            arr = np.asarray(vals, dtype=float)
            agg["per_offset"][k] = {
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        return EnsembleResult(
            individual_results=results,
            ensemble_curve=ensemble_curve,
            envelope=envelope,
            aggregated_metrics=agg,
            strategy_name=results[0].strategy_name,
        )
