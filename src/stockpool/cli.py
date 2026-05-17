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
from stockpool.fetcher import fetch_daily, resample_to_weekly
from stockpool.indicators import add_all
from stockpool.report import StockAnalysis, render_report
from stockpool.signals import (
    combine_daily_weekly,
    detect_signals,
    score_triggers,
    verdict_of,
)

log = logging.getLogger("stockpool")


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


def _analyze_one(stock, cfg: AppConfig, force_refresh: bool) -> StockAnalysis:
    """Full per-stock pipeline. Single failures are caught → verdict=neutral + warnings."""
    warnings: list[str] = []
    daily_score = 0
    weekly_score = 0
    triggers_daily: list = []
    triggers_weekly: list = []
    hit_rates: dict = {}
    enriched_daily = None

    try:
        daily = fetch_daily(stock.code, cfg.data.history_days,
                            cfg.data.cache_dir, force_refresh=force_refresh)
    except Exception as e:
        warnings.append(f"数据拉取失败: {e}")
        return StockAnalysis(
            code=stock.code, name=stock.name,
            daily_score=0, weekly_score=0,
            final_score=0.0, verdict="neutral",
            warnings=warnings,
        )

    if len(daily) < 30:
        warnings.append(f"历史数据不足 ({len(daily)} 根),指标可能不可靠")

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

    final_score = combine_daily_weekly(daily_score, weekly_score, cfg.scoring)
    verdict = verdict_of(final_score, cfg.verdicts)

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

    return StockAnalysis(
        code=stock.code, name=stock.name,
        daily_score=daily_score, weekly_score=weekly_score,
        final_score=final_score, verdict=verdict,
        triggers_daily=triggers_daily, triggers_weekly=triggers_weekly,
        hit_rates=hit_rates,
        verdict_hit_rates=verdict_hit_rates,
        daily_with_indicators=enriched_daily,
        warnings=warnings,
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

    per_stock: list = []
    for s in stocks:
        log.info("Backtesting %s (%s)...", s.code, s.name)
        try:
            daily = fetch_daily(
                s.code, cfg.data.history_days, cfg.data.cache_dir,
                force_refresh=args.refresh,
            )
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
            )
            per_stock.append((s.code, s.name, result))
        except Exception as e:
            log.error("Backtest failed for %s: %s\n%s", s.code, e, traceback.format_exc())

    if not per_stock:
        log.error("No stocks could be backtested.")
        return 1

    out = render_backtest_report(per_stock, run_date=run_date, output_dir=backtest_root)
    log.info("Backtest report written: %s", out)
    log.info("Latest also at: %s", backtest_root / "latest.html")
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

    analyses: list[StockAnalysis] = []
    for s in stocks:
        log.info("Analyzing %s (%s)...", s.code, s.name)
        try:
            analyses.append(_analyze_one(s, cfg, force_refresh=args.refresh))
        except Exception as e:
            log.error("Unexpected failure on %s: %s\n%s", s.code, e, traceback.format_exc())
            analyses.append(StockAnalysis(
                code=s.code, name=s.name,
                daily_score=0, weekly_score=0, final_score=0.0,
                verdict="neutral",
                warnings=[f"未预期错误: {e}"],
            ))

    out = render_report(
        analyses, run_date=run_date,
        config_path=Path(args.config), config_hash=cfg.content_hash,
        output_dir=cfg.report.output_dir,
        keep_history=cfg.report.keep_history,
        klines_to_show=cfg.report.klines_to_show,
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
