"""CLI entry: `python -m stockpool run`."""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd

from stockpool import __version__
from stockpool.ab import (
    load_ab_config,
    render_ab_report,
    run_ab,
    run_single_arm,
)
from stockpool.backtest import compute_hit_rates
from stockpool.backtest_composite import verdict_bucket_stats, walk_forward_verdicts
from stockpool.backtest_report import render_backtest_report
from stockpool.backtest_runner import backtest_stocks, prepare_pool as _prepare_ml_pool
from stockpool.config import AppConfig, load_config
from stockpool.fetcher import (
    check_source_change,
    fetch_daily,
    fetch_index_daily,
    fetch_sector_daily,
    fetch_universe,
    list_universe,
    load_universe_cache,
    resample_to_weekly,
    validate_ohlcv,
)
from stockpool.indicators import add_all
from stockpool.recommend_pool import PoolBEntry, compute_or_load_pool_b
from stockpool.report import ContextSignal, StockAnalysis, render_report
from stockpool.signals import (
    combine_daily_weekly,
    detect_signals,
    score_triggers,
    verdict_of,
)
from stockpool.strategy_factory import build_strategy

log = logging.getLogger("stockpool")


def _compute_verdict(df: pd.DataFrame, cfg: AppConfig):
    """Run indicator+signal pipeline. Returns (d_score, w_score, final, verdict, trig_d, trig_w)."""
    from stockpool.indicators import add_all as _add_all
    enriched = _add_all(df, cfg.indicators)
    trig_d = detect_signals(enriched, cfg.weights)
    d_score = score_triggers(trig_d)

    weekly = resample_to_weekly(df)
    trig_w: list = []
    w_score = 0
    if len(weekly) >= 30:
        trig_w = detect_signals(_add_all(weekly, cfg.indicators), cfg.weights)
        w_score = score_triggers(trig_w)

    final = combine_daily_weekly(d_score, w_score, cfg.scoring)
    verdict = verdict_of(final, cfg.verdicts)
    return d_score, w_score, final, verdict, trig_d, trig_w


def _is_trading_day(today: date) -> bool:
    """Use AKShare trading-day calendar to check if today is an A-share trading day."""
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(cal["trade_date"]).dt.date
        return today in set(dates)
    except Exception as e:
        log.warning("Trading day check failed (%s) — assuming trading day", e)
        return True


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [file_h, stream_h]


def _analyze_one(
    stock, cfg: AppConfig, force_refresh: bool,
    market_context: list[ContextSignal] | None = None,
    pool_data: dict[str, pd.DataFrame] | None = None,
    factor_panel: dict | None = None,
    close_panel: pd.DataFrame | None = None,
    shared_cache: dict | None = None,
    sector_context_cache: dict[str, "ContextSignal | str"] | None = None,
) -> StockAnalysis:
    """Full per-stock pipeline. Single failures are caught → verdict=neutral + warnings."""
    warnings: list[str] = []
    daily_score = 0
    weekly_score = 0
    triggers_daily: list = []
    triggers_weekly: list = []
    hit_rates: dict = {}
    enriched_daily = None
    context = list(market_context or [])

    try:
        daily = fetch_daily(stock.code, cfg.data.history_days,
                            cfg.data.cache_dir, force_refresh=force_refresh,
                            source=cfg.data.source)
    except Exception as e:
        warnings.append(f"数据拉取失败: {e}")
        return StockAnalysis(
            code=stock.code, name=stock.name,
            daily_score=0, weekly_score=0,
            final_score=0.0, verdict="neutral",
            warnings=warnings,
            context=context,
        )

    if len(daily) < 30:
        warnings.append(f"历史数据不足 ({len(daily)} 根),指标可能不可靠")

    warnings.extend(validate_ohlcv(daily))

    enriched_daily = add_all(daily, cfg.indicators)
    triggers_daily = detect_signals(enriched_daily, cfg.weights)
    daily_score = score_triggers(triggers_daily)

    weekly = resample_to_weekly(daily)
    if len(weekly) >= 30:
        enriched_weekly = add_all(weekly, cfg.indicators)
        triggers_weekly = detect_signals(enriched_weekly, cfg.weights)
        weekly_score = score_triggers(triggers_weekly)
    else:
        warnings.append("周 K 样本不足,本股不计算周 K 信号")

    # Verdict + triggers come from the configured strategy. ml_factor's
    # pipeline is refit at most once per calendar month per stock — see
    # MLFactorStrategy.predict_latest.
    strategy_name = cfg.strategy.name
    try:
        strategy = build_strategy(
            cfg, pool_data=pool_data, current_stock_code=stock.code,
            factor_panel=factor_panel, close_panel=close_panel,
            shared_cache=shared_cache,
        )
        latest = strategy.predict_latest(daily)
        verdict = latest.get("signal", "neutral")
        final_score = float(latest.get("final_score", 0.0))
        if strategy_name == "composite_verdict":
            daily_score = int(latest.get("daily_score", daily_score))
            weekly_score = int(latest.get("weekly_score", weekly_score))
        else:
            # ml_factor: 替换 trigger 列表为因子贡献; 老的 indicator triggers/
            # scores 在 ml 路径下没有语义,清零避免误导。
            triggers_daily = list(latest.get("triggers_daily", []))
            triggers_weekly = list(latest.get("triggers_weekly", []))
            daily_score = 0
            weekly_score = 0
    except Exception as e:
        warnings.append(f"策略 {strategy_name} 评级失败,回退到综合评级: {e}")
        final_score = combine_daily_weekly(daily_score, weekly_score, cfg.scoring)
        verdict = verdict_of(final_score, cfg.verdicts)
        strategy_name = "composite_verdict"

    try:
        hit_rates = compute_hit_rates(enriched_daily, cfg.weights, cfg.backtest.forward_days)
    except Exception as e:
        warnings.append(f"单信号回测失败: {e}")

    verdict_hit_rates: dict = {}
    try:
        wf = walk_forward_verdicts(
            daily, cfg.weights, cfg.scoring, cfg.verdicts, cfg.indicators
        )
        verdict_hit_rates = verdict_bucket_stats(wf, cfg.backtest.forward_days)
    except Exception as e:
        warnings.append(f"综合评级回测失败: {e}")

    if stock.sector:
        cached = (sector_context_cache or {}).get(stock.sector)
        if isinstance(cached, ContextSignal):
            context.append(cached)
        elif isinstance(cached, str):
            warnings.append(f"板块({stock.sector})数据失败: {cached}")
        else:
            try:
                sector_df = fetch_sector_daily(
                    stock.sector, cfg.data.history_days,
                    cfg.data.cache_dir, force_refresh,
                    source=cfg.data.source,
                )
                d_s, w_s, f_s, v, trig_d, _ = _compute_verdict(sector_df, cfg)
                context.append(ContextSignal(
                    label=f"{stock.sector}板块",
                    daily_score=d_s, weekly_score=w_s,
                    final_score=f_s, verdict=v,
                    triggers_daily=trig_d,
                ))
            except Exception as e:
                warnings.append(f"板块({stock.sector})数据失败: {e}")

    return StockAnalysis(
        code=stock.code, name=stock.name,
        daily_score=daily_score, weekly_score=weekly_score,
        final_score=final_score, verdict=verdict,
        triggers_daily=triggers_daily, triggers_weekly=triggers_weekly,
        hit_rates=hit_rates,
        verdict_hit_rates=verdict_hit_rates,
        daily_with_indicators=enriched_daily,
        warnings=warnings,
        context=context,
        strategy_name=strategy_name,
    )


def _wire_refresh_fundamentals(args: argparse.Namespace) -> None:
    """--refresh-fundamentals 透传:设置 fundamentals_loader 模块级 force flag。

    设置后本进程内所有基本面表加载无视 30 天缓存直接重拉(P2-26)。
    """
    from stockpool import fundamentals_loader
    fundamentals_loader.set_force_refresh(
        bool(getattr(args, "refresh_fundamentals", False))
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    _wire_refresh_fundamentals(args)

    run_date = date.today().isoformat()
    backtest_root = Path(cfg.report.output_dir) / "backtest"
    _setup_logging(backtest_root / run_date)
    log.info("stockpool backtest v%s starting for %s", __version__, run_date)

    stocks = cfg.stocks
    if args.stocks:
        wanted = set(args.stocks.split(","))
        stocks = [s for s in stocks if s.code in wanted]
        if not stocks:
            log.error("No stocks match --stocks filter: %s", args.stocks)
            return 2

    # For ml_factor in pooled mode, preload every stock's history once so
    # the strategy can build a panel at each refit point. Pool composition is
    # decided by ``training_universe``.
    pool_data, factor_panel, close_panel = _prepare_ml_pool(
        cfg, stocks, args.refresh,
        refresh_factor_panel=getattr(args, "refresh_factor_panel", False),
    )
    # One shared cache for the whole backtest run — lets MLFactorStrategy
    # reuse the stacked (X, y) panel across stocks and share monthly fits.
    shared_cache: dict = {}

    per_stock, failed = backtest_stocks(
        cfg, stocks, pool_data, factor_panel,
        shared_cache=shared_cache, refresh=args.refresh,
        close_panel=close_panel,
    )
    for code, err in failed:
        log.warning("Skipped %s: %s", code, err)

    if not per_stock:
        log.error("No stocks could be backtested.")
        return 1

    if cfg.backtest.engine == "multi_lot":
        sizing = cfg.backtest.sizing
        if sizing.type == "fixed":
            sizing_desc = f"fixed {sizing.fixed.size:.0%}"
        else:
            vt = sizing.vol_target
            sizing_desc = (
                f"vol_target ref={vt.reference_vol_annual:.0%} "
                f"window={vt.vol_window} clip=[{vt.min_size:.0%},{vt.max_size:.0%}]"
            )
        engine_label = f"multi_lot · {sizing_desc}"
    else:
        engine_label = "single · 同时只持一只票,信号反转换仓"
    out = render_backtest_report(
        per_stock, run_date=run_date, output_dir=backtest_root,
        engine_label=engine_label,
    )
    log.info("Backtest report written: %s", out)
    log.info("Latest also at: %s", backtest_root / "latest.html")
    return 0


def _apply_stocks_filter(stocks, codes):
    if not codes:
        return list(stocks)
    keep = set(codes)
    return [s for s in stocks if s.code in keep]


def _fmt_opt(v, spec: str) -> str:
    """Format an optional metric; None (undefined, e.g. short window) → '—'."""
    return "—" if v is None else format(v, spec)


def _print_single_arm_stdout(arm_result) -> None:
    print(f"=== Arm: {arm_result.name} ===")
    print(f"Stocks succeeded: {len(arm_result.per_stock)}; "
          f"failed: {len(arm_result.failed)}")
    for code, name, res in arm_result.per_stock:
        N = next(iter(res.metrics))
        m = res.metrics[N]
        print(f"  {code} {name}: total_ret={m['total_return']:+.3f} "
              f"ann={_fmt_opt(m.get('annualized_return'), '+.3f')} "
              f"sharpe={_fmt_opt(m.get('sharpe'), '+.2f')} "
              f"max_dd={m['max_drawdown']:.3f}")


def cmd_ab(args: argparse.Namespace) -> int:
    try:
        ab_cfg = load_ab_config(args.config)
        base_cfg_path = (Path(args.config).parent / ab_cfg.base_config).resolve()
        base_cfg = load_config(base_cfg_path)
    except Exception:
        log.exception("ab config invalid")
        return 2

    run_date = date.today().isoformat()
    out_root = Path(base_cfg.report.output_dir) / "ab"
    _setup_logging(out_root / run_date)
    log.info("stockpool ab v%s for %s", __version__, run_date)

    stocks = _apply_stocks_filter(base_cfg.stocks, ab_cfg.stocks_filter)

    if args.arm:
        if args.arm not in ab_cfg.arms:
            log.error("--arm %r not in %s", args.arm, list(ab_cfg.arms))
            return 2
        arm_result = run_single_arm(
            ab_cfg, base_cfg, stocks, args.refresh, args.arm,
        )
        _print_single_arm_stdout(arm_result)
        return 0

    result = run_ab(
        ab_cfg, base_cfg, stocks, args.refresh,
        share_pool=not args.no_share_pool,
    )
    if not result.arm_a.per_stock and not result.arm_b.per_stock:
        log.error("Both arms produced no results.")
        return 1
    out = render_ab_report(result, output_dir=out_root)
    log.info("AB report written: %s", out)
    log.info("Latest also at: %s", out_root / "latest.html")
    return 0


def cmd_portfolio_backtest(args: argparse.Namespace) -> int:
    """Portfolio-level backtest (PR-1 skeleton).

    Universe = cfg.stocks (PR-2 will switch to load_universe_cache + eligibility).
    Calls cfg.strategy's per-stock generate_signals to precompute a (T × N)
    score panel, then runs PortfolioEngine to produce one equity curve.
    """
    from stockpool.industry_map import load_or_build_industry_map
    from stockpool.portfolio.eligibility import EligibilityFilter
    from stockpool.portfolio.engine import PortfolioEngine
    from stockpool.portfolio.report import render_portfolio_report
    from stockpool.portfolio.scoring import precompute_scores_from_legacy
    from stockpool.portfolio.strategy import PrecomputedScoreStrategy

    cfg = load_config(args.config)
    _wire_refresh_fundamentals(args)
    if not cfg.portfolio_backtest.enabled:
        log.error(
            "portfolio_backtest.enabled=false in %s — set to true to opt in.",
            args.config,
        )
        return 2

    run_date = date.today().isoformat()
    out_root = Path(cfg.report.output_dir) / "portfolio"
    _setup_logging(out_root / run_date)
    log.info("stockpool portfolio-backtest v%s for %s", __version__, run_date)

    # ---- universe (PR-2) ----
    # Prefer the full A-share cache (universe.parquet + load_universe_cache).
    # Falls back to cfg.stocks with a warning when the universe cache is empty
    # (e.g. user hasn't run fetch-universe yet).
    cache_dir = Path(cfg.data.cache_dir)
    universe_path = cache_dir / "universe.parquet"
    name_map: dict[str, str] = {}
    if universe_path.exists():
        universe_df = pd.read_parquet(universe_path)
        name_map = dict(zip(universe_df["code"], universe_df.get("name", universe_df["code"])))
    pool_data = load_universe_cache(cache_dir, cfg.data.history_days)
    if not pool_data:
        log.warning(
            "Universe cache is empty (run `stockpool fetch-universe` first to "
            "unlock the full A-share universe). Falling back to cfg.stocks."
        )
        pool_data = {}
        for s in cfg.stocks:
            try:
                pool_data[s.code] = fetch_daily(
                    s.code, cfg.data.history_days, cfg.data.cache_dir,
                    force_refresh=args.refresh, source=cfg.data.source,
                )
                name_map.setdefault(s.code, s.name)
            except Exception as e:
                log.warning("Skipped %s: %s", s.code, e)
    else:
        log.info("Universe cache: %d stocks", len(pool_data))
        # cfg.stocks still merged in so any application-pinned code missing
        # from the universe cache (e.g. 北交) is still tradeable.
        for s in cfg.stocks:
            if s.code not in pool_data:
                try:
                    pool_data[s.code] = fetch_daily(
                        s.code, cfg.data.history_days, cfg.data.cache_dir,
                        force_refresh=args.refresh, source=cfg.data.source,
                    )
                except Exception as e:
                    log.warning("Skipped %s: %s", s.code, e)
            name_map.setdefault(s.code, s.name)
    if not pool_data:
        log.error("No usable stock data; aborting.")
        return 1

    # ---- sector map (PR-2) ----
    # Only meaningful if max_per_industry is set; load anyway for stability
    # so report has the data even if cap is None.
    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    log.info("Sector map: %d codes mapped", len(sector_map))
    from stockpool.factors.context import set_sector_map
    set_sector_map(sector_map)

    # ---- Decouple portfolio universe from training pool (optional) ----
    # See PortfolioBacktestConfig.universe_codes. Default None = use full
    # pool_data as both training and portfolio universe (legacy behavior).
    portfolio_codes = cfg.portfolio_backtest.universe_codes
    if portfolio_codes:
        portfolio_pool_data = {c: pool_data[c] for c in portfolio_codes if c in pool_data}
        missing = [c for c in portfolio_codes if c not in pool_data]
        if missing:
            log.warning(
                "%d portfolio universe codes not in training pool (skipped): %s",
                len(missing), missing[:5],
            )
        log.info(
            "Portfolio universe decoupled: training pool=%d, portfolio universe=%d",
            len(pool_data), len(portfolio_pool_data),
        )
    else:
        portfolio_pool_data = pool_data

    # ---- ML factor panel (only for ml_factor + pooled) ----
    # Always built on the full training pool so ml_factor with
    # training_universe=all still sees all 4358 stocks.
    factor_panel = None
    close_panel = None
    if cfg.strategy.name == "ml_factor" and cfg.strategy.ml_factor.panel_mode == "pooled":
        from stockpool.strategy_factory import (
            load_or_build_factor_panel,
            maybe_inject_mcap_panel,
        )
        maybe_inject_mcap_panel(
            cfg.strategy.ml_factor.preprocess, pool_data, cfg.data.cache_dir,
        )
        factor_panel, close_panel = load_or_build_factor_panel(
            cfg.strategy.ml_factor.factors, pool_data, cfg.data.cache_dir,
            preprocess_cfg=cfg.strategy.ml_factor.preprocess,
        )

    shared_cache: dict = {}
    legacy = build_strategy(
        cfg,
        pool_data=pool_data,
        factor_panel=factor_panel,
        close_panel=close_panel,
        shared_cache=shared_cache,
    )

    # Score-panel cache: keyed by cfg.content_hash. Spec §6.5 accepts the
    # known suboptimality that changing top_k also invalidates (no partial
    # hash) — first version trades cache hit rate for simplicity.
    score_dir = Path(cfg.portfolio_backtest.score_cache_dir)
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f"{cfg.content_hash}.parquet"
    if score_path.exists() and not args.refresh_scores:
        log.info("Loading cached score panel: %s", score_path)
        score_panel = pd.read_parquet(score_path)
    else:
        log.info(
            "Precomputing score panel over %d stocks (portfolio universe) ...",
            len(portfolio_pool_data),
        )
        score_panel = precompute_scores_from_legacy(legacy, portfolio_pool_data)
        if score_panel.empty:
            log.error("Score panel is empty (all stocks failed).")
            return 1
        score_panel.to_parquet(score_path)
        log.info("Score panel cached: %s", score_path)

    from stockpool.backtesting.framework import TradeCosts
    portfolio_strat = PrecomputedScoreStrategy(
        score_panel, name=cfg.strategy.name,
    )
    eligibility = EligibilityFilter(
        cfg.portfolio_backtest.eligibility, name_map=name_map,
    )
    costs = TradeCosts(
        buy_cost=cfg.backtest.costs.buy_cost,
        sell_cost=cfg.backtest.costs.sell_cost,
    )

    def _make_engine() -> PortfolioEngine:
        # Fresh engine instance each call so staggered runs don't share
        # internal state. PrecomputedScoreStrategy is stateless and safely
        # reused.
        return PortfolioEngine(
            strategy=portfolio_strat,
            portfolio_cfg=cfg.portfolio_backtest.portfolio,
            costs=costs,
            risk_free_rate=cfg.backtest.risk_free_rate,
            eligibility=eligibility,
            sector_map=sector_map,
        )

    n_offsets = cfg.portfolio_backtest.staggered_starts
    if n_offsets > 1:
        from stockpool.portfolio.ensemble import StaggeredRunner
        from stockpool.portfolio.report import render_ensemble_report
        log.info("Running staggered ensemble: %d offsets", n_offsets)
        runner = StaggeredRunner(
            engine_factory=_make_engine,
            risk_free_rate=cfg.backtest.risk_free_rate,
        )
        ensemble = runner.run(portfolio_pool_data, n_offsets=n_offsets)
        log.info(
            "Ensemble done: %d offsets, ensemble total_return=%+.3f",
            ensemble.n_offsets,
            ensemble.aggregated_metrics.get("ensemble", {}).get("total_return", 0.0),
        )
        out = render_ensemble_report(
            ensemble, panel_data=portfolio_pool_data,
            run_date=run_date, output_dir=out_root,
            config_hash=cfg.content_hash,
            initial_equity=cfg.portfolio_backtest.portfolio.initial_cash,
        )
    else:
        result = _make_engine().run(portfolio_pool_data, start_offset=0)
        log.info(
            "Backtest done: %d bars, %d trades, total_return=%+.3f",
            len(result.curve), len(result.trades),
            result.metrics.get("total_return", 0.0),
        )
        out = render_portfolio_report(
            result, panel_data=portfolio_pool_data,
            run_date=run_date, output_dir=out_root,
            config_hash=cfg.content_hash,
        )
    log.info("Portfolio report written: %s", out)
    log.info("Latest also at: %s", out_root / "latest.html")
    return 0


def cmd_portfolio_ab(args: argparse.Namespace) -> int:
    """Portfolio-level A/B comparison (PR-4 of the portfolio framework spec).

    Builds a shared universe + sector/name maps once, then runs both arms
    against them (each arm gets its own score panel keyed by per-arm
    content_hash). Failures are isolated per arm — the report still renders
    if one side fails.
    """
    from stockpool.industry_map import load_or_build_industry_map
    from stockpool.portfolio_ab import (
        load_portfolio_ab_config,
        render_portfolio_ab_report,
        run_portfolio_ab,
        run_single_arm,
    )
    from stockpool.portfolio_ab.config import build_effective_cfg

    try:
        ab_cfg = load_portfolio_ab_config(args.config)
        base_path = (Path(args.config).parent / ab_cfg.base_config).resolve()
        base_cfg = load_config(base_path)
    except Exception:
        log.exception("portfolio-ab config invalid")
        return 2
    if not base_cfg.portfolio_backtest.enabled:
        log.error(
            "base config %s has portfolio_backtest.enabled=false; "
            "set to true to opt in.", base_path,
        )
        return 2

    if args.arm and args.arm not in ab_cfg.arms:
        log.error("--arm %r not in %s", args.arm, list(ab_cfg.arms))
        return 2

    run_date = date.today().isoformat()
    out_root = Path(base_cfg.report.output_dir) / "portfolio_ab"
    _setup_logging(out_root / run_date)
    log.info("stockpool portfolio-ab v%s for %s", __version__, run_date)

    # ---- shared universe (mirrors cmd_portfolio_backtest) ----
    cache_dir = Path(base_cfg.data.cache_dir)
    universe_path = cache_dir / "universe.parquet"
    name_map: dict[str, str] = {}
    if universe_path.exists():
        universe_df = pd.read_parquet(universe_path)
        name_map = dict(zip(universe_df["code"], universe_df.get("name", universe_df["code"])))
    pool_data = load_universe_cache(cache_dir, base_cfg.data.history_days)
    if not pool_data:
        log.warning(
            "Universe cache is empty; falling back to cfg.stocks. "
            "Run `stockpool fetch-universe` first for the full A-share universe."
        )
        pool_data = {}
        for s in base_cfg.stocks:
            try:
                pool_data[s.code] = fetch_daily(
                    s.code, base_cfg.data.history_days, base_cfg.data.cache_dir,
                    force_refresh=args.refresh, source=base_cfg.data.source,
                )
                name_map.setdefault(s.code, s.name)
            except Exception as e:
                log.warning("Skipped %s: %s", s.code, e)
    else:
        log.info("Universe cache: %d stocks", len(pool_data))
        for s in base_cfg.stocks:
            if s.code not in pool_data:
                try:
                    pool_data[s.code] = fetch_daily(
                        s.code, base_cfg.data.history_days, base_cfg.data.cache_dir,
                        force_refresh=args.refresh, source=base_cfg.data.source,
                    )
                except Exception as e:
                    log.warning("Skipped %s: %s", s.code, e)
            name_map.setdefault(s.code, s.name)
    if not pool_data:
        log.error("No usable stock data; aborting.")
        return 1

    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    log.info("Sector map: %d codes mapped", len(sector_map))
    from stockpool.factors.context import set_sector_map
    set_sector_map(sector_map)

    # Decouple training pool (= full pool_data) from portfolio universe
    # (= optional explicit subset). When universe_codes is None, both are
    # the same (legacy behavior). See PortfolioBacktestConfig.universe_codes.
    portfolio_codes = base_cfg.portfolio_backtest.universe_codes
    if portfolio_codes:
        portfolio_pool_data = {c: pool_data[c] for c in portfolio_codes if c in pool_data}
        missing = [c for c in portfolio_codes if c not in pool_data]
        if missing:
            log.warning(
                "%d portfolio universe codes not in training pool (skipped): %s",
                len(missing), missing[:5],
            )
        log.info(
            "Portfolio universe decoupled: training pool=%d, portfolio universe=%d",
            len(pool_data), len(portfolio_pool_data),
        )
    else:
        portfolio_pool_data = None  # use full pool_data (legacy)

    if args.arm:
        # Single-arm debug mode: stdout only, no HTML.
        effective = build_effective_cfg(base_cfg, ab_cfg.arms[args.arm])
        arm_result = run_single_arm(
            args.arm, effective,
            pool_data=pool_data, sector_map=sector_map, name_map=name_map,
            refresh_scores=args.refresh_scores,
            portfolio_pool_data=portfolio_pool_data,
        )
        _print_portfolio_arm_stdout(arm_result)
        return 0

    result = run_portfolio_ab(
        ab_cfg, base_cfg, pool_data=pool_data,
        sector_map=sector_map, name_map=name_map,
        refresh_scores=args.refresh_scores,
        portfolio_pool_data=portfolio_pool_data,
    )
    out = render_portfolio_ab_report(result, run_date=run_date, output_dir=out_root)
    log.info("Portfolio AB report written: %s", out)
    log.info("Latest also at: %s", out_root / "latest.html")
    return 0


def _print_portfolio_arm_stdout(arm_result) -> None:
    print(f"=== Portfolio arm: {arm_result.name} ===")
    if arm_result.failed:
        print(f"FAILED: {arm_result.error}")
        return
    m = arm_result.primary_metrics
    print(
        f"  total_ret={m.get('total_return') or 0.0:+.3f} "
        f"ann={_fmt_opt(m.get('annualized_return'), '+.3f')} "
        f"sharpe={_fmt_opt(m.get('sharpe'), '+.2f')} "
        f"max_dd={m.get('max_drawdown') or 0.0:.3f} "
        f"trades={m.get('trade_count', 0)}"
    )


def cmd_fetch_universe(args: argparse.Namespace) -> int:
    """Bulk-fetch all A-shares (ex-ST/科创/北交) into the data cache.

    Used to build the training universe for ML strategies. The application
    universe (`config.yaml:stocks`) is unaffected.
    """
    cfg = load_config(args.config)
    _setup_logging(Path(cfg.report.output_dir) / date.today().isoformat())

    # Listing the whole A-share universe is only implemented for mootdx
    # (baostock/akshare lack a clean equivalent), but the per-stock K-line
    # pull should follow cfg.data.source unless the user overrode it on CLI.
    log.info("Listing A-share universe via mootdx ...")
    listing = list_universe(source="mootdx")
    log.info("Universe size: %d stocks (含 ST,训练池不按当前名称剔除)", len(listing))

    # P0-4 轻量 / P3-4: 合入 baostock 干净名单 —— 干净中文名、is_st、
    # out_date/status。mootdx 名是乱码,且 ST/退市状态只有 baostock 可靠。
    try:
        from stockpool.ipo_dates import load_or_build_stock_basics
        basics = load_or_build_stock_basics(cfg.data.cache_dir)
        if not basics.empty:
            merged = listing.merge(
                basics[["code", "name", "ipo_date", "out_date", "status", "is_st"]],
                on="code", how="left", suffixes=("_tdx", ""),
            )
            merged["name"] = merged["name"].fillna(merged["name_tdx"])
            listing = merged.drop(columns=["name_tdx"])
            log.info("Merged stock_basics: %d ST flagged, %d missing in baostock",
                     int(listing["is_st"].fillna(False).sum()),
                     int(listing["name"].isna().sum()))
    except Exception as e:  # noqa: BLE001
        log.warning("stock_basics merge failed (%s); universe keeps mootdx names", e)

    if args.limit > 0:
        listing = listing.head(args.limit)
        log.info("--limit %d → pulling first %d", args.limit, len(listing))

    codes = listing["code"].tolist()
    Path(cfg.data.cache_dir).mkdir(parents=True, exist_ok=True)
    listing_path = Path(cfg.data.cache_dir) / "universe.parquet"
    listing.to_parquet(listing_path, index=False)
    log.info("Universe listing cached: %s", listing_path)

    effective_source = args.source or cfg.data.source
    force_refresh = args.refresh
    if check_source_change(cfg.data.cache_dir, effective_source):
        log.warning(
            "Data source changed → forcing full refresh of universe cache "
            "(use --refresh false has no effect here; mixing sources would "
            "corrupt volume units and adjustment baselines)."
        )
        force_refresh = True
    log.info("Fetching per-stock daily bars via %s ...", effective_source)
    result = fetch_universe(
        codes,
        history_days=cfg.data.history_days,
        cache_dir=cfg.data.cache_dir,
        source=effective_source,
        force_refresh=force_refresh,
        max_workers=args.workers,
    )
    log.info("Fetched %d/%d stocks successfully.", len(result), len(codes))
    return 0


def cmd_factors_analyze(args: argparse.Namespace) -> int:
    """Analyze factors on the pooled panel and write HTML + JSON reports."""
    from stockpool.factors import list_factors
    from stockpool.factors_analysis import analyze_factors
    from stockpool.factors_analysis_report import render_factor_analysis_report
    from stockpool.panel import build_panel_from_cache

    cfg = load_config(args.config)
    cache_dir = Path(cfg.data.cache_dir)

    if args.universe == "all":
        universe_file = cache_dir / "universe.parquet"
        if not universe_file.exists():
            log.error(
                "universe=all but %s does not exist; "
                "run `python -m stockpool fetch-universe` first",
                universe_file,
            )
            return 1
        all_codes = pd.read_parquet(universe_file)["code"].tolist()
        # Skip codes whose per-stock parquet is missing (fetch-universe may have
        # failed on a handful of codes — e.g. newly listed with no history).
        codes = [c for c in all_codes if (cache_dir / f"{c}_daily.parquet").exists()]
        missing = len(all_codes) - len(codes)
        if missing:
            log.warning("Skipping %d codes with no daily cache", missing)
        if not codes:
            log.error("No usable per-stock cache under %s; re-run fetch-universe", cache_dir)
            return 1
    else:
        codes = [s.code for s in cfg.stocks]

    factor_names = list(args.factors) if args.factors else list_factors()
    log.info(
        "Analyzing %d factors over %d stocks (universe=%s)",
        len(factor_names), len(codes), args.universe,
    )

    # Sector-aware factors (industry_neutral / industry_relative_strength)
    # need sector_map. Loading is cheap (baostock cache, ~30 day TTL).
    from stockpool.factors.context import set_sector_map
    from stockpool.industry_map import load_or_build_industry_map

    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    if not sector_map:
        log.warning(
            "Industry map unavailable; sector-aware factors will be NaN"
        )
    set_sector_map(sector_map)

    panel = build_panel_from_cache(codes, cfg.data.history_days, cache_dir)

    # P2-4 口径对齐:与生产训练同一套 mask + 预处理 + 标签基准。
    ml_cfg = cfg.strategy.ml_factor
    mask = None
    if ml_cfg.mask.enabled:
        from stockpool.ipo_dates import load_or_build_ipo_dates, load_st_codes
        from stockpool.panel import compute_tradability_mask
        mask = compute_tradability_mask(
            panel, ml_cfg.mask,
            ipo_dates=load_or_build_ipo_dates(cache_dir) or None,
            st_codes=load_st_codes(cache_dir) or None,
        )
    if ml_cfg.preprocess.market_cap_neutralize:
        from stockpool.factors.context import set_mcap_panel
        from stockpool.strategy_factory import build_log_mcap_panel_from_close
        mcap = build_log_mcap_panel_from_close(panel["close"], cache_dir)
        if mcap is not None:
            set_mcap_panel(mcap)

    end_date = pd.Timestamp(args.end_date) if getattr(args, "end_date", None) else None

    regime_close = None
    if not args.no_regime:
        idx_code = cfg.context.indices[0].code if cfg.context.indices else None
        if idx_code:
            idx_path = cache_dir / f"idx_{idx_code}.parquet"
            if idx_path.exists():
                idx_df = pd.read_parquet(idx_path)
                idx_df["date"] = pd.to_datetime(idx_df["date"])
                regime_close = idx_df.set_index("date").sort_index()["close"]
            else:
                log.warning(
                    "regime index cache missing (%s); skipping regime split", idx_path
                )

    result = analyze_factors(
        panel=panel,
        factor_names=factor_names,
        horizon=args.horizon,
        ic_window=args.ic_window,
        regime_index_close=regime_close,
        end_date=end_date,
        label_basis=ml_cfg.label_basis,
        mask=mask,
        preprocess_cfg=ml_cfg.preprocess,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    json_path = out_dir / f"{stamp}.json"
    html_path = out_dir / f"{stamp}.html"
    latest_html = out_dir / "latest.html"

    result.to_json(json_path)
    render_factor_analysis_report(result, html_path)
    if latest_html.exists() or latest_html.is_symlink():
        latest_html.unlink()
    latest_html.write_bytes(html_path.read_bytes())

    log.info("Wrote %s and %s", json_path, html_path)
    return 0


def cmd_factors_pick_by_ic(args: argparse.Namespace) -> int:
    """Pick a de-correlated top-N from a FactorAnalysisResult JSON."""
    import json
    from stockpool.factors_analysis import FactorAnalysisResult, pick_top_factors

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("input JSON not found: %s", input_path)
        return 1

    result = FactorAnalysisResult.from_json(input_path)
    picked = pick_top_factors(
        result,
        top_n=args.top_n,
        max_correlation=args.max_corr,
        min_ir=args.min_ir,
        score_by=args.score_by,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"factors": picked}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Picked %d factors → %s", len(picked), output_path)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _t_total = time.perf_counter()
    _t = time.perf_counter()
    cfg = load_config(args.config)
    _wire_refresh_fundamentals(args)

    today = date.today()
    run_date = today.isoformat()
    log_dir = Path(cfg.report.output_dir) / run_date
    _setup_logging(log_dir)

    log.info("stockpool v%s starting run for %s", __version__, run_date)

    if not args.skip_trading_day_check and not _is_trading_day(today):
        log.info("Today (%s) is not an A-share trading day. Exit 0.", run_date)
        return 0

    stocks = cfg.stocks
    if args.stocks:
        wanted = set(args.stocks.split(","))
        stocks = [s for s in stocks if s.code in wanted]
        if not stocks:
            log.error("No stocks match --stocks filter: %s", args.stocks)
            return 2
    print(f"[TIME] setup+config: {time.perf_counter()-_t:.2f}s", flush=True)

    _t = time.perf_counter()
    market_context: list[ContextSignal] = []
    for idx_cfg in cfg.context.indices:
        try:
            idx_df = fetch_index_daily(
                idx_cfg.code, cfg.data.history_days,
                cfg.data.cache_dir, force_refresh=args.refresh,
                source=cfg.data.source,
            )
            d_s, w_s, f_s, v, trig_d, _ = _compute_verdict(idx_df, cfg)
            market_context.append(ContextSignal(
                label=idx_cfg.name,
                daily_score=d_s, weekly_score=w_s,
                final_score=f_s, verdict=v,
                triggers_daily=trig_d,
            ))
            log.info("Market index %s: %s (%+.1f)", idx_cfg.name, v, f_s)
        except Exception as e:
            log.warning("Market index %s failed: %s", idx_cfg.code, e)
    print(f"[TIME] market_index_context ({len(cfg.context.indices)} indices): "
          f"{time.perf_counter()-_t:.2f}s", flush=True)

    # For ml_factor in pooled mode the strategy needs every stock's history
    # to build cross-sectional factors at predict time. Pool composition is
    # decided by ``training_universe`` (pool vs full A-share cache).
    _t = time.perf_counter()
    pool_data, factor_panel, close_panel = _prepare_ml_pool(cfg, stocks, args.refresh)
    print(f"[TIME] _prepare_ml_pool (pool_data={len(pool_data) if pool_data else 0} stocks, "
          f"factor_panel={len(factor_panel) if factor_panel else 0} factors): "
          f"{time.perf_counter()-_t:.2f}s", flush=True)
    shared_cache: dict = {}

    # Pre-fetch + compute each unique sector once. Same ContextSignal is shared
    # across all stocks in that sector, saving redundant fetch_sector_daily calls.
    _t = time.perf_counter()
    sector_context_cache: dict[str, "ContextSignal | str"] = {}
    unique_sectors = sorted({s.sector for s in stocks if s.sector})
    for sector in unique_sectors:
        try:
            sector_df = fetch_sector_daily(
                sector, cfg.data.history_days,
                cfg.data.cache_dir, args.refresh,
                source=cfg.data.source,
            )
            d_s, w_s, f_s, v, trig_d, _ = _compute_verdict(sector_df, cfg)
            sector_context_cache[sector] = ContextSignal(
                label=f"{sector}板块",
                daily_score=d_s, weekly_score=w_s,
                final_score=f_s, verdict=v,
                triggers_daily=trig_d,
            )
        except Exception as e:  # noqa: BLE001
            sector_context_cache[sector] = str(e)
            log.warning("Sector %s pre-fetch failed: %s", sector, e)
    print(f"[TIME] sector_context prefetch ({len(unique_sectors)} unique sectors): "
          f"{time.perf_counter()-_t:.2f}s", flush=True)

    _t = time.perf_counter()
    analyses: list[StockAnalysis] = []
    for s in stocks:
        log.info("Analyzing %s (%s)...", s.code, s.name)
        try:
            analyses.append(_analyze_one(
                s, cfg, force_refresh=args.refresh,
                market_context=market_context,
                pool_data=pool_data,
                factor_panel=factor_panel,
                close_panel=close_panel,
                shared_cache=shared_cache,
                sector_context_cache=sector_context_cache,
            ))
        except Exception as e:
            log.error("Unexpected failure on %s: %s\n%s", s.code, e, traceback.format_exc())
            analyses.append(StockAnalysis(
                code=s.code, name=s.name,
                daily_score=0, weekly_score=0, final_score=0.0,
                verdict="neutral",
                warnings=[f"未预期错误: {e}"],
                context=list(market_context),
            ))
    print(f"[TIME] per_stock_loop ({len(stocks)} stocks): "
          f"{time.perf_counter()-_t:.2f}s", flush=True)

    _t = time.perf_counter()
    pool_b: list[PoolBEntry] = []
    if cfg.recommend_pool.enabled:
        try:
            pool_b = compute_or_load_pool_b(
                cfg, today,
                pool_data=pool_data, factor_panel=factor_panel,
                close_panel=close_panel,
            )
            log.info("Pool B: %d stocks (top_n=%d)",
                     len(pool_b), cfg.recommend_pool.top_n)
        except Exception as e:
            log.error("Pool B failed (continuing without it): %s\n%s",
                      e, traceback.format_exc())
    print(f"[TIME] pool_b (len={len(pool_b)}): {time.perf_counter()-_t:.2f}s", flush=True)

    _t = time.perf_counter()
    out = render_report(
        analyses, run_date=run_date,
        config_path=Path(args.config), config_hash=cfg.content_hash,
        output_dir=cfg.report.output_dir,
        keep_history=cfg.report.keep_history,
        klines_to_show=cfg.report.klines_to_show,
        market_context=market_context,
        pool_b=pool_b or None,
    )
    print(f"[TIME] render_report: {time.perf_counter()-_t:.2f}s", flush=True)
    print(f"[TIME] TOTAL cmd_run: {time.perf_counter()-_t_total:.2f}s", flush=True)
    log.info("Report written: %s", out)
    log.info("Latest also at: %s", Path(cfg.report.output_dir) / "latest.html")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stockpool", description="A-share signal analyzer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Full pipeline: fetch → analyze → report")
    p_run.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_run.add_argument("--refresh", action="store_true", help="Bypass cache, refetch all")
    p_run.add_argument("--refresh-fundamentals", action="store_true",
                       help="强制重拉 baostock 财务数据(绕过 30 天缓存)")
    p_run.add_argument("--stocks", default="", help="Only run listed codes (comma-separated)")
    p_run.add_argument("--skip-trading-day-check", action="store_true",
                       help="Run even on non-trading days (debug)")
    p_run.set_defaults(func=cmd_run)

    p_bt = sub.add_parser("backtest", help="Composite-strategy equity-curve backtest")
    p_bt.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_bt.add_argument("--refresh", action="store_true", help="Bypass cache, refetch all")
    p_bt.add_argument("--refresh-factor-panel", action="store_true",
                      help="Bypass data/factor_panels/ cache, recompute factors")
    p_bt.add_argument("--refresh-fundamentals", action="store_true",
                      help="强制重拉 baostock 财务数据(绕过 30 天缓存)")
    p_bt.add_argument("--stocks", default="", help="Only run listed codes (comma-separated)")
    p_bt.set_defaults(func=cmd_backtest)

    p_ab = sub.add_parser("ab", help="A/B-compare two strategies on the same universe")
    p_ab.add_argument("--config", default="ab.yaml")
    p_ab.add_argument("--refresh", action="store_true")
    p_ab.add_argument("--arm", default=None, help="Debug: run only one arm by name")
    p_ab.add_argument("--no-share-pool", action="store_true",
                      help="Force each arm to load its own universe / factor panel")
    p_ab.set_defaults(func=cmd_ab)

    p_pb = sub.add_parser(
        "portfolio-backtest",
        help="Portfolio-level backtest (top-K equal-weight, periodic rebalance)",
    )
    p_pb.add_argument("--config", default="config.yaml")
    p_pb.add_argument("--refresh", action="store_true",
                      help="Bypass per-stock OHLCV cache, refetch all")
    p_pb.add_argument("--refresh-scores", action="store_true",
                      help="Bypass data/portfolio_scores cache, recompute scores")
    p_pb.add_argument("--refresh-fundamentals", action="store_true",
                      help="强制重拉 baostock 财务数据(绕过 30 天缓存)")
    p_pb.set_defaults(func=cmd_portfolio_backtest)

    p_pab = sub.add_parser(
        "portfolio-ab",
        help="Portfolio-level A/B comparison of two strategy configs",
    )
    p_pab.add_argument("--config", default="portfolio_ab.yaml")
    p_pab.add_argument("--refresh", action="store_true",
                       help="Bypass per-stock OHLCV cache")
    p_pab.add_argument("--refresh-scores", action="store_true",
                       help="Bypass data/portfolio_scores cache for both arms")
    p_pab.add_argument("--arm", default=None,
                       help="Debug: run only one arm; prints metrics to stdout, no HTML")
    p_pab.set_defaults(func=cmd_portfolio_ab)

    p_fu = sub.add_parser(
        "fetch-universe",
        help="Bulk-fetch all A-shares (ex-ST/科创/北交) into the data cache (for ML training)",
    )
    p_fu.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_fu.add_argument(
        "--source", default=None,
        choices=["mootdx", "baostock", "akshare"],
        help="Override cfg.data.source for the per-stock fetch step. "
             "Default: use cfg.data.source. The universe *listing* always "
             "uses mootdx (only impl).",
    )
    p_fu.add_argument("--workers", type=int, default=8, help="Parallel threads (default 8)")
    p_fu.add_argument("--refresh", action="store_true", help="Bypass cache, refetch all")
    p_fu.add_argument("--limit", type=int, default=0,
                      help="Limit to first N stocks (for smoke testing; 0 = all)")
    p_fu.set_defaults(func=cmd_fetch_universe)

    # `factors` sub-tree: list / show / pick
    p_factors = sub.add_parser("factors", help="Inspect or select registered factors")
    fsub = p_factors.add_subparsers(dest="factors_cmd", required=True)
    from stockpool.factors_picker import cli_list, cli_pick, cli_show
    p_list = fsub.add_parser("list", help="List all registered factors")
    p_list.add_argument("--source", default=None, help="Filter by source tag (e.g. wq101)")
    p_list.add_argument("--type", default=None, help="Filter by type tag (e.g. momentum)")
    p_list.set_defaults(func=cli_list)
    p_show = fsub.add_parser("show", help="Show one factor's metadata")
    p_show.add_argument("name", help="Factor name (with or without suffix args)")
    p_show.set_defaults(func=cli_show)
    p_pick = fsub.add_parser(
        "pick",
        help="Open the HTML factor picker; '应用' button writes selection.json",
    )
    p_pick.add_argument(
        "--output", default=None,
        help="Selection JSON path (default: reports/selection.json). "
             "Written when '应用' is clicked in the browser.",
    )
    p_pick.add_argument(
        "--port", type=int, default=0,
        help="Server port (default: 0 = auto-pick a free port)",
    )
    p_pick.add_argument(
        "--no-open", action="store_true",
        help="Don't auto-open the browser",
    )
    p_pick.add_argument(
        "--static", action="store_true",
        help="Fallback: write the picker as a static HTML file instead of running "
             "a local server. '应用' button auto-degrades to 'download'.",
    )
    p_pick.add_argument(
        "--html-output", default=None,
        help="(--static only) HTML file path (default: reports/factors_picker.html)",
    )
    p_pick.set_defaults(func=cli_pick)

    p_analyze = fsub.add_parser(
        "analyze",
        help="Compute rolling IC / IR / half-life / correlation across factors",
    )
    p_analyze.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_analyze.add_argument(
        "--universe", choices=["pool", "all"], default="pool",
        help="pool = cfg.stocks; all = data/universe.parquet (needs fetch-universe first)",
    )
    p_analyze.add_argument(
        "--factors", nargs="*", default=None,
        help="Factor names (default: all registered factors)",
    )
    p_analyze.add_argument("--horizon", type=int, default=3)
    p_analyze.add_argument(
        "--ic-window", type=int, default=252,
        help="Metadata only — daily IC uses the full window",
    )
    p_analyze.add_argument(
        "--no-regime", action="store_true",
        help="Skip the bull/bear/sideways regime split",
    )
    p_analyze.add_argument(
        "--end-date", type=str, default=None,
        help="分析窗口截止日 (YYYY-MM-DD)。selection 必须截止在回测起点之前"
             "(P0-6),否则因子清单带 in-sample 选择偏差",
    )
    p_analyze.add_argument(
        "--output", default="reports/factor_analysis",
        help="Output directory (HTML + JSON written here)",
    )
    p_analyze.set_defaults(func=cmd_factors_analyze)

    p_pick_ic = fsub.add_parser(
        "pick-by-ic",
        help="From a factors-analyze JSON, pick a top-N de-correlated selection.json",
    )
    p_pick_ic.add_argument(
        "--input", required=True,
        help="Path to a factors-analyze JSON (e.g. reports/factor_analysis/2026-05-23.json)",
    )
    p_pick_ic.add_argument(
        "--output", default="reports/selection.json",
        help="Output selection.json (consumed by MLFactorConfig.factors_file)",
    )
    p_pick_ic.add_argument("--top-n", type=int, default=20)
    p_pick_ic.add_argument("--max-corr", type=float, default=0.6)
    p_pick_ic.add_argument("--min-ir", type=float, default=0.05)
    p_pick_ic.add_argument(
        "--score-by", choices=["ir", "mean_ic", "abs_ic"], default="ir",
    )
    p_pick_ic.set_defaults(func=cmd_factors_pick_by_ic)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
