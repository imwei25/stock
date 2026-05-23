"""CLI entry: `python -m stockpool run`."""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd

from stockpool import __version__
from stockpool.backtest import compute_hit_rates
from stockpool.backtest_composite import simulate_equity_curve, verdict_bucket_stats, walk_forward_verdicts
from stockpool.backtest_report import render_backtest_report
from stockpool.config import AppConfig, load_config
from stockpool.fetcher import (
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
from stockpool.report import ContextSignal, StockAnalysis, render_report
from stockpool.signals import (
    combine_daily_weekly,
    detect_signals,
    score_triggers,
    verdict_of,
)
from stockpool.strategy_factory import build_factor_panel, build_strategy, simulate_strategy_equity_curve

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


def _prepare_ml_pool(
    cfg: AppConfig, stocks, force_refresh: bool,
) -> tuple[dict[str, pd.DataFrame] | None, dict | None]:
    """Build (pool_data, factor_panel) for ml_factor strategies, or (None, None).

    Pool composition depends on ``cfg.strategy.ml_factor.training_universe``:
      * ``pool``: only ``cfg.stocks`` (legacy, ~10 stocks).
      * ``all``: full A-share cache from ``data/`` (~4000 stocks, requires a
        prior ``fetch-universe`` run). Application stocks are merged in so any
        cfg.stocks entry missing from the universe cache (e.g. 北交) is still
        usable. Cross-sec factors only become meaningful at panel widths in
        the hundreds, so ``all`` is recommended whenever WQ101 alphas are used.

    The factor panel is computed once on the combined pool and reused across
    every per-stock predict — the panel-wide computation can be expensive on
    the all-universe path.
    """
    if (
        cfg.strategy.name != "ml_factor"
        or cfg.strategy.ml_factor.panel_mode != "pooled"
    ):
        return None, None

    ml_cfg = cfg.strategy.ml_factor
    pool_data: dict[str, pd.DataFrame] = {}

    if ml_cfg.training_universe == "all":
        log.info("Loading universe cache (training_universe=all) ...")
        pool_data = load_universe_cache(cfg.data.cache_dir, cfg.data.history_days)
        if not pool_data:
            log.warning(
                "training_universe=all but data/ has no cached stocks. "
                "Run `python -m stockpool fetch-universe` first; falling back to pool."
            )
        else:
            log.info("Universe cache loaded: %d stocks", len(pool_data))

    # Ensure every application stock is in the pool (fetch fresh if missing
    # or stale; also picks up today's bar for already-cached stocks).
    for s in stocks:
        try:
            pool_data[s.code] = fetch_daily(
                s.code, cfg.data.history_days, cfg.data.cache_dir,
                force_refresh=force_refresh, source=cfg.data.source,
            )
        except Exception as e:
            log.warning("Pool preload skipped for %s: %s", s.code, e)

    log.info("Building factor panel over %d stocks × %d factors ...",
             len(pool_data), len(ml_cfg.factors))
    factor_panel = build_factor_panel(ml_cfg.factors, pool_data)
    log.info("Factor panel built: %d factors", len(factor_panel))
    return pool_data, factor_panel


def _analyze_one(
    stock, cfg: AppConfig, force_refresh: bool,
    market_context: list[ContextSignal] | None = None,
    pool_data: dict[str, pd.DataFrame] | None = None,
    factor_panel: dict | None = None,
    shared_cache: dict | None = None,
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
            factor_panel=factor_panel, shared_cache=shared_cache,
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


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

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
    pool_data, factor_panel = _prepare_ml_pool(cfg, stocks, args.refresh)
    needs_pool = pool_data is not None
    # One shared cache for the whole backtest run — lets MLFactorStrategy
    # reuse the stacked (X, y) panel across stocks and share monthly fits.
    shared_cache: dict = {}

    per_stock: list = []
    for s in stocks:
        log.info("Backtesting %s (%s)...", s.code, s.name)
        try:
            daily = pool_data.get(s.code) if needs_pool else None
            if daily is None:
                daily = fetch_daily(
                    s.code, cfg.data.history_days, cfg.data.cache_dir,
                    force_refresh=args.refresh, source=cfg.data.source,
                )
            if cfg.strategy.name == "composite_verdict":
                wf = walk_forward_verdicts(
                    daily, cfg.weights, cfg.scoring, cfg.verdicts, cfg.indicators
                )
                if len(wf) == 0:
                    log.warning("%s: insufficient history, skipping", s.code)
                    continue
                result = simulate_equity_curve(
                    wf,
                    holding_days_list=cfg.backtest.equity_curve_holding_days,
                    with_buy_and_hold=True,
                    buy_cost=cfg.backtest.costs.buy_cost,
                    sell_cost=cfg.backtest.costs.sell_cost,
                    risk_free_rate=cfg.backtest.risk_free_rate,
                    engine=cfg.backtest.engine,
                    position_size=cfg.backtest.position_size,
                    max_concurrent_lots=cfg.backtest.max_concurrent_lots,
                )
            else:
                strategy = build_strategy(
                    cfg,
                    pool_data=pool_data if needs_pool else None,
                    current_stock_code=s.code,
                    factor_panel=factor_panel,
                    shared_cache=shared_cache,
                )
                result = simulate_strategy_equity_curve(
                    daily, strategy,
                    holding_days_list=cfg.backtest.equity_curve_holding_days,
                    with_buy_and_hold=True,
                    buy_cost=cfg.backtest.costs.buy_cost,
                    sell_cost=cfg.backtest.costs.sell_cost,
                    risk_free_rate=cfg.backtest.risk_free_rate,
                    engine=cfg.backtest.engine,
                    position_size=cfg.backtest.position_size,
                    max_concurrent_lots=cfg.backtest.max_concurrent_lots,
                )
            per_stock.append((s.code, s.name, result))
        except Exception as e:
            log.error("Backtest failed for %s: %s\n%s", s.code, e, traceback.format_exc())

    if not per_stock:
        log.error("No stocks could be backtested.")
        return 1

    if cfg.backtest.engine == "multi_lot":
        engine_label = (
            f"multi_lot · 每次买入 {cfg.backtest.position_size:.0%} 起始资本独立一单"
        )
    else:
        engine_label = "single · 同时只持一只票,信号反转换仓"
    out = render_backtest_report(
        per_stock, run_date=run_date, output_dir=backtest_root,
        engine_label=engine_label,
    )
    log.info("Backtest report written: %s", out)
    log.info("Latest also at: %s", backtest_root / "latest.html")
    return 0


def cmd_fetch_universe(args: argparse.Namespace) -> int:
    """Bulk-fetch all A-shares (ex-ST/科创/北交) into the data cache.

    Used to build the training universe for ML strategies. The application
    universe (`config.yaml:stocks`) is unaffected.
    """
    cfg = load_config(args.config)
    _setup_logging(Path(cfg.report.output_dir) / date.today().isoformat())

    log.info("Listing A-share universe via %s ...", args.source)
    listing = list_universe(source=args.source)
    log.info("Universe size: %d stocks", len(listing))

    if args.limit > 0:
        listing = listing.head(args.limit)
        log.info("--limit %d → pulling first %d", args.limit, len(listing))

    codes = listing["code"].tolist()
    listing_path = Path(cfg.data.cache_dir) / "universe.parquet"
    Path(cfg.data.cache_dir).mkdir(parents=True, exist_ok=True)
    listing.to_parquet(listing_path, index=False)
    log.info("Universe listing cached: %s", listing_path)

    result = fetch_universe(
        codes,
        history_days=cfg.data.history_days,
        cache_dir=cfg.data.cache_dir,
        source=args.source,
        force_refresh=args.refresh,
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
        codes = list_universe(cache_dir)
        if not codes:
            log.error(
                "universe=all but data/universe.parquet is empty; "
                "run `python -m stockpool fetch-universe` first"
            )
            return 1
    else:
        codes = [s.code for s in cfg.stocks]

    factor_names = list(args.factors) if args.factors else list_factors()
    log.info(
        "Analyzing %d factors over %d stocks (universe=%s)",
        len(factor_names), len(codes), args.universe,
    )

    panel = build_panel_from_cache(codes, cfg.data.history_days, cache_dir)

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


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

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

    # For ml_factor in pooled mode the strategy needs every stock's history
    # to build cross-sectional factors at predict time. Pool composition is
    # decided by ``training_universe`` (pool vs full A-share cache).
    pool_data, factor_panel = _prepare_ml_pool(cfg, stocks, args.refresh)
    shared_cache: dict = {}

    analyses: list[StockAnalysis] = []
    for s in stocks:
        log.info("Analyzing %s (%s)...", s.code, s.name)
        try:
            analyses.append(_analyze_one(
                s, cfg, force_refresh=args.refresh,
                market_context=market_context,
                pool_data=pool_data,
                factor_panel=factor_panel,
                shared_cache=shared_cache,
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

    out = render_report(
        analyses, run_date=run_date,
        config_path=Path(args.config), config_hash=cfg.content_hash,
        output_dir=cfg.report.output_dir,
        keep_history=cfg.report.keep_history,
        klines_to_show=cfg.report.klines_to_show,
        market_context=market_context,
    )
    log.info("Report written: %s", out)
    log.info("Latest also at: %s", Path(cfg.report.output_dir) / "latest.html")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stockpool", description="A-share signal analyzer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Full pipeline: fetch → analyze → report")
    p_run.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_run.add_argument("--refresh", action="store_true", help="Bypass cache, refetch all")
    p_run.add_argument("--stocks", default="", help="Only run listed codes (comma-separated)")
    p_run.add_argument("--skip-trading-day-check", action="store_true",
                       help="Run even on non-trading days (debug)")
    p_run.set_defaults(func=cmd_run)

    p_bt = sub.add_parser("backtest", help="Composite-strategy equity-curve backtest")
    p_bt.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_bt.add_argument("--refresh", action="store_true", help="Bypass cache, refetch all")
    p_bt.add_argument("--stocks", default="", help="Only run listed codes (comma-separated)")
    p_bt.set_defaults(func=cmd_backtest)

    p_fu = sub.add_parser(
        "fetch-universe",
        help="Bulk-fetch all A-shares (ex-ST/科创/北交) into the data cache (for ML training)",
    )
    p_fu.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_fu.add_argument("--source", default="mootdx", choices=["mootdx"],
                      help="Data source (currently only mootdx is supported)")
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
        "--output", default="reports/factor_analysis",
        help="Output directory (HTML + JSON written here)",
    )
    p_analyze.set_defaults(func=cmd_factors_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
