"""Portfolio AB runner — execute two arms over shared universe + sector data.

For each arm:
  1. Compute effective AppConfig (merge of base + arm override).
  2. Build legacy per-stock strategy → precompute (T × N) score panel
     keyed by ``effective_cfg.content_hash`` (cache isolated across arms).
  3. Wrap in ``PrecomputedScoreStrategy`` + ``PortfolioEngine`` (+ optional
     ``StaggeredRunner`` when ``staggered_starts > 1``).
  4. Catch any exception → ``ArmResult.failed = True`` with the error string;
     the other arm still runs and the report still renders (half a chart is
     better than zero).
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from stockpool.backtesting.framework import TradeCosts
from stockpool.config import AppConfig
from stockpool.portfolio.eligibility import EligibilityFilter
from stockpool.portfolio.engine import PortfolioEngine
from stockpool.portfolio.ensemble import EnsembleResult, StaggeredRunner
from stockpool.portfolio.result import PortfolioBacktestResult
from stockpool.portfolio.scoring import precompute_scores_from_legacy
from stockpool.portfolio.strategy import PrecomputedScoreStrategy
from stockpool.portfolio_ab.config import PortfolioABConfig, build_effective_cfg
from stockpool.strategy_factory import build_strategy, load_or_build_factor_panel

log = logging.getLogger("stockpool")


@dataclass
class ArmResult:
    """Outcome of one arm in a portfolio AB run."""
    name: str
    effective_cfg: AppConfig | None
    single: PortfolioBacktestResult | None = None
    ensemble: EnsembleResult | None = None
    failed: bool = False
    error: str | None = None

    @property
    def primary_curve(self) -> pd.DataFrame:
        """The curve to plot — ensemble.ensemble_curve when staggered, else single.curve."""
        if self.ensemble is not None:
            return self.ensemble.ensemble_curve
        if self.single is not None:
            return self.single.curve
        return pd.DataFrame({"date": [], "equity": []})

    @property
    def primary_metrics(self) -> dict:
        if self.ensemble is not None:
            return self.ensemble.aggregated_metrics.get("ensemble", {})
        if self.single is not None:
            return self.single.metrics
        return {}

    @property
    def trades(self):
        """Iterate all closed trades. For ensemble, flatten across offsets."""
        if self.ensemble is not None:
            for r in self.ensemble.individual_results:
                yield from r.trades
        elif self.single is not None:
            yield from self.single.trades


@dataclass
class ABResult:
    """Pair of ArmResult objects, indexed by arm name."""
    arms: dict[str, ArmResult] = field(default_factory=dict)

    @property
    def arm_names(self) -> list[str]:
        return list(self.arms)


def run_single_arm(
    arm_name: str,
    effective_cfg: AppConfig,
    pool_data: Mapping[str, pd.DataFrame],
    sector_map: Mapping[str, str],
    name_map: Mapping[str, str],
    refresh_scores: bool = False,
    portfolio_pool_data: Mapping[str, pd.DataFrame] | None = None,
) -> ArmResult:
    """Execute one portfolio arm. Exceptions are caught and packed into ArmResult.

    Args:
        arm_name: human-readable arm name (for logs and the result label).
        effective_cfg: AppConfig produced by ``build_effective_cfg``.
        pool_data: **training pool** — used to build factor_panel and to
            inject into MLFactorStrategy (so training_universe=all sees the
            full set). Both arms share the same training pool.
        sector_map, name_map: shared lookups.
        refresh_scores: bypass the per-arm score-panel parquet cache.
        portfolio_pool_data: **portfolio universe** — used for
            precompute_scores_from_legacy and PortfolioEngine execution.
            Defaults to ``pool_data`` (legacy behavior when not decoupled).
            Setting this to a subset lets training stay on the full pool
            while portfolio top-K only picks from this subset (avoids
            OOM/segfault from precompute on full 4358-stock predict).
    """
    if portfolio_pool_data is None:
        portfolio_pool_data = pool_data
    log.info(
        "[%s] running portfolio arm (training pool=%d, portfolio universe=%d)",
        arm_name, len(pool_data), len(portfolio_pool_data),
    )
    try:
        # ML factor panel only if the arm's strategy actually needs it.
        # Built on the *training* pool so training_universe=all sees full set.
        factor_panel = None
        close_panel = None
        if (
            effective_cfg.strategy.name == "ml_factor"
            and effective_cfg.strategy.ml_factor.panel_mode == "pooled"
        ):
            factor_panel, close_panel = load_or_build_factor_panel(
                effective_cfg.strategy.ml_factor.factors, pool_data,
                effective_cfg.data.cache_dir,
                preprocess_cfg=effective_cfg.strategy.ml_factor.preprocess,
            )
        shared_cache: dict = {}
        legacy = build_strategy(
            effective_cfg,
            pool_data=pool_data,
            factor_panel=factor_panel,
            close_panel=close_panel,
            shared_cache=shared_cache,
        )

        # Per-arm score panel cache — keyed by the *arm's* content_hash +
        # 数据 last_date(同 cmd_portfolio_backtest,防数据更新后命中旧 panel
        # 致组合尾段静默停止调仓)。
        score_dir = Path(effective_cfg.portfolio_backtest.score_cache_dir)
        score_dir.mkdir(parents=True, exist_ok=True)
        _last_dates = [
            pd.to_datetime(df["date"]).max()
            for df in portfolio_pool_data.values() if len(df)
        ]
        _last_iso = max(_last_dates).date().isoformat() if _last_dates else "nodata"
        score_path = score_dir / f"{effective_cfg.content_hash}_{_last_iso}.parquet"
        if score_path.exists() and not refresh_scores:
            log.info("[%s] cache hit: %s", arm_name, score_path)
            score_panel = pd.read_parquet(score_path)
        else:
            log.info("[%s] precomputing score panel ...", arm_name)
            # Score only the portfolio universe (subset of training pool).
            score_panel = precompute_scores_from_legacy(legacy, portfolio_pool_data)
            if score_panel.empty:
                raise RuntimeError("score panel is empty — all stocks failed")
            score_panel.to_parquet(score_path)

        portfolio_strat = PrecomputedScoreStrategy(
            score_panel, name=effective_cfg.strategy.name,
        )
        eligibility = EligibilityFilter(
            effective_cfg.portfolio_backtest.eligibility, name_map=name_map,
        )
        costs = TradeCosts(
            buy_cost=effective_cfg.backtest.costs.buy_cost,
            sell_cost=effective_cfg.backtest.costs.sell_cost,
        )

        def _factory() -> PortfolioEngine:
            eng = PortfolioEngine(
                strategy=portfolio_strat,
                portfolio_cfg=effective_cfg.portfolio_backtest.portfolio,
                costs=costs,
                risk_free_rate=effective_cfg.backtest.risk_free_rate,
                eligibility=eligibility,
                sector_map=sector_map,
            )
            try:
                from stockpool.ipo_dates import load_st_codes
                eng.st_codes = load_st_codes(effective_cfg.data.cache_dir)
            except Exception:  # noqa: BLE001
                eng.st_codes = None
            return eng

        n_offsets = effective_cfg.portfolio_backtest.staggered_starts
        if n_offsets > 1:
            runner = StaggeredRunner(
                _factory, risk_free_rate=effective_cfg.backtest.risk_free_rate,
            )
            # Engine runs over portfolio_pool_data (needs OHLCV of the stocks
            # it actually trades, not the full training pool).
            ensemble = runner.run(portfolio_pool_data, n_offsets=n_offsets)
            return ArmResult(
                name=arm_name, effective_cfg=effective_cfg, ensemble=ensemble,
            )
        single = _factory().run(portfolio_pool_data, start_offset=0)
        return ArmResult(
            name=arm_name, effective_cfg=effective_cfg, single=single,
        )

    except Exception as e:  # noqa: BLE001 — per-arm failure isolation
        log.error(
            "[%s] arm failed: %s\n%s",
            arm_name, e, traceback.format_exc(),
        )
        return ArmResult(
            name=arm_name, effective_cfg=effective_cfg,
            failed=True, error=str(e),
        )


def run_portfolio_ab(
    ab_cfg: PortfolioABConfig,
    base_cfg: AppConfig,
    pool_data: Mapping[str, pd.DataFrame],
    sector_map: Mapping[str, str],
    name_map: Mapping[str, str],
    refresh_scores: bool = False,
    portfolio_pool_data: Mapping[str, pd.DataFrame] | None = None,
) -> ABResult:
    """Run both arms in sequence; return ``ABResult`` with both outcomes.

    Each arm is independent — if arm A throws, arm B still runs and the
    report still renders with a red banner for the failed side.

    See ``run_single_arm`` for ``portfolio_pool_data`` semantics
    (training pool vs portfolio universe decoupling).
    """
    arms: dict[str, ArmResult] = {}
    for arm_name, override in ab_cfg.arms.items():
        try:
            effective = build_effective_cfg(base_cfg, override)
        except Exception as e:  # noqa: BLE001
            log.error("[%s] effective-cfg merge failed: %s", arm_name, e)
            arms[arm_name] = ArmResult(
                name=arm_name, effective_cfg=None,
                failed=True, error=f"merge failed: {e}",
            )
            continue
        arms[arm_name] = run_single_arm(
            arm_name, effective,
            pool_data=pool_data, sector_map=sector_map, name_map=name_map,
            refresh_scores=refresh_scores,
            portfolio_pool_data=portfolio_pool_data,
        )
    return ABResult(arms=arms)
