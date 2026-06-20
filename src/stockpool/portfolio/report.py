"""Minimal HTML rendering for a portfolio backtest result (PR-1).

Outputs:
  * Equity curve vs equal-weight buy-and-hold baseline
  * Metrics table
  * Per-bar holdings count line

PR-3 will overlay ensemble bands; PR-4 adds AB-style diff. Keep this file
deliberately small.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line

from stockpool.portfolio.ensemble import EnsembleResult
from stockpool.portfolio.result import PortfolioBacktestResult


_CSS = """
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif;
         max-width: 1400px; margin: 1em auto; padding: 0 1em; color: #222; }
  h1 { margin-bottom: 0.3em; }
  .meta { color: #666; margin-bottom: 1em; }
  table { border-collapse: collapse; width: 100%; margin: 0.5em 0 1.5em; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 0.95em; }
  th { background: #f6f6f6; text-align: left; }
  .chart-wrap { margin: 1em 0; }
  footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #eee;
           color: #888; font-size: 0.85em; }
"""


def _equal_weight_baseline(
    panel_data: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Equal-weight buy-and-hold across all codes from bar 0; no costs."""
    closes_wide: dict[str, pd.Series] = {}
    for code, df in panel_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date").sort_index()
        closes_wide[code] = d["close"].astype(float)
    if not closes_wide:
        return np.ones(len(dates))
    wide = pd.DataFrame(closes_wide).reindex(dates).ffill()
    # Normalize each column to first valid bar, then mean across columns.
    base = wide.iloc[0].replace(0, np.nan)
    norm = wide.divide(base, axis=1)
    return norm.mean(axis=1).ffill().fillna(1.0).values


def _equity_chart(
    result: PortfolioBacktestResult,
    baseline: np.ndarray | None,
) -> Line:
    dates = pd.DatetimeIndex(result.curve["date"]).strftime("%Y-%m-%d").tolist()
    line = (
        Line(init_opts=opts.InitOpts(width="100%", height="480px"))
        .add_xaxis(dates)
    )
    eq_vals = [round(float(v), 4) for v in result.curve["equity"].values]
    line.add_yaxis(
        f"Portfolio · {result.strategy_name}", eq_vals,
        is_smooth=True, is_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),
    )
    if baseline is not None and len(baseline) == len(dates):
        # Scale baseline to initial portfolio equity so the two curves start
        # at the same value (otherwise initial_cash=1.0 vs baseline=1.0 align
        # naturally, but for initial_cash≠1.0 we'd need to rescale).
        scale = float(result.curve["equity"].iloc[0]) if len(result.curve) else 1.0
        bh_vals = [round(float(v * scale), 4) for v in baseline]
        line.add_yaxis(
            "Equal-Weight B&H", bh_vals,
            is_smooth=True, is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(type_="dashed", width=2),
        )
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="Equity Curve", pos_left="center", pos_top="2%"),
        xaxis_opts=opts.AxisOpts(
            is_scale=True,
            axislabel_opts=opts.LabelOpts(rotate=30, font_size=10, margin=8),
        ),
        yaxis_opts=opts.AxisOpts(is_scale=True, name="净值"),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            opts.DataZoomOpts(type_="slider", pos_bottom="2%", pos_top="92%"),
        ],
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_top="10%", pos_left="center"),
    )
    line.options["grid"] = {
        "top": "22%", "bottom": "16%", "left": "8%", "right": "4%",
        "containLabel": True,
    }
    return line


def _holdings_chart(result: PortfolioBacktestResult) -> Line:
    dates = pd.DatetimeIndex(result.curve["date"]).strftime("%Y-%m-%d").tolist()
    vals = [int(v) for v in result.curve["num_positions"].values]
    line = (
        Line(init_opts=opts.InitOpts(width="100%", height="280px"))
        .add_xaxis(dates)
        .add_yaxis("num_positions", vals, is_step=True, is_symbol_show=False,
                   label_opts=opts.LabelOpts(is_show=False))
    )
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="Holdings over time", pos_left="center", pos_top="2%"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30, font_size=10)),
        yaxis_opts=opts.AxisOpts(name="# positions", min_=0),
        legend_opts=opts.LegendOpts(is_show=False),
    )
    line.options["grid"] = {
        "top": "20%", "bottom": "20%", "left": "8%", "right": "4%",
        "containLabel": True,
    }
    return line


def _metrics_table(result: PortfolioBacktestResult) -> str:
    m = result.metrics
    rows = [
        ("Strategy", result.strategy_name),
        ("Total return", f"{m.get('total_return', 0.0):+.3f}"),
        ("Annualized return", f"{m.get('annualized_return', 0.0):+.3f}"),
        ("Sharpe", f"{m.get('sharpe', 0.0) or 0.0:+.2f}"),
        ("Max drawdown", f"{m.get('max_drawdown', 0.0):.3f}"),
        ("# trades", str(m.get("trade_count", len(result.trades)))),
        ("# rebalances", str(len(result.rebalance_log))),
    ]
    body = "\n".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _ensemble_chart(
    ensemble: EnsembleResult,
    baseline: np.ndarray | None,
    initial_equity: float,
) -> Line:
    """Envelope band (min/max) + median + ensemble (mean) + B&H.

    pyecharts Line doesn't have a native band primitive; we approximate it
    by stacking ``min`` as a transparent line and ``max - min`` as a stacked
    filled area on top. ``stack="band"`` keeps them grouped so toggling one
    series from the legend hides the band as a unit.
    """
    dates = pd.DatetimeIndex(ensemble.envelope["date"]).strftime("%Y-%m-%d").tolist()
    env = ensemble.envelope
    min_vals = [round(float(v), 4) for v in env["min"].values]
    max_vals = [round(float(v), 4) for v in env["max"].values]
    band_top = [round(float(b - a), 4) for a, b in zip(env["min"].values, env["max"].values)]
    median_vals = [round(float(v), 4) for v in env["median"].values]
    ensemble_vals = [round(float(v), 4) for v in ensemble.ensemble_curve["equity"].values]

    line = (
        Line(init_opts=opts.InitOpts(width="100%", height="520px"))
        .add_xaxis(dates)
    )
    # Band base (transparent, only to anchor the stack).
    line.add_yaxis(
        "_band_base", min_vals,
        is_smooth=False, is_symbol_show=False, stack="band",
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=0, opacity=0),
        areastyle_opts=opts.AreaStyleOpts(opacity=0),
    )
    # Band fill = max - min, stacked on top of the base → fills [min, max].
    line.add_yaxis(
        "min-max band", band_top,
        is_smooth=False, is_symbol_show=False, stack="band",
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=0, opacity=0),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.18, color="#888"),
    )
    line.add_yaxis(
        f"median (k={ensemble.n_offsets})", median_vals,
        is_smooth=True, is_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5, type_="dashed", color="#000"),
    )
    line.add_yaxis(
        "ensemble (mean)", ensemble_vals,
        is_smooth=True, is_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=2.5, color="#c0392b"),
    )
    if baseline is not None and len(baseline) == len(dates):
        bh_vals = [round(float(v * initial_equity), 4) for v in baseline]
        line.add_yaxis(
            "Equal-Weight B&H", bh_vals,
            is_smooth=True, is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(type_="dashed", width=2, color="#2563eb"),
        )

    line.set_global_opts(
        title_opts=opts.TitleOpts(title="Ensemble net asset value", pos_left="center", pos_top="2%"),
        xaxis_opts=opts.AxisOpts(
            is_scale=True,
            axislabel_opts=opts.LabelOpts(rotate=30, font_size=10, margin=8),
        ),
        yaxis_opts=opts.AxisOpts(is_scale=True, name="净值"),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            opts.DataZoomOpts(type_="slider", pos_bottom="2%", pos_top="92%"),
        ],
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(
            pos_top="10%", pos_left="center",
            # Hide the band-base legend entry (it's a layout helper).
            selected_map={"_band_base": False},
        ),
    )
    line.options["grid"] = {
        "top": "22%", "bottom": "16%", "left": "8%", "right": "4%",
        "containLabel": True,
    }
    return line


def _per_offset_cards_html(ensemble: EnsembleResult) -> str:
    rows = []
    for k, r in enumerate(ensemble.individual_results):
        m = r.metrics
        rows.append(
            f"<tr><td>k={k}</td>"
            f"<td>{m.get('total_return', 0.0):+.3f}</td>"
            f"<td>{m.get('annualized_return', 0.0):+.3f}</td>"
            f"<td>{m.get('sharpe', 0.0) or 0.0:+.2f}</td>"
            f"<td>{m.get('max_drawdown', 0.0):.3f}</td>"
            f"<td>{m.get('trade_count', len(r.trades))}</td>"
            f"</tr>"
        )
    body = "\n".join(rows)
    return (
        "<details><summary>Per-offset metrics</summary>"
        "<table><thead><tr><th>offset</th><th>total</th><th>ann</th>"
        "<th>sharpe</th><th>max_dd</th><th>#trades</th></tr></thead>"
        f"<tbody>{body}</tbody></table></details>"
    )


def _ensemble_metrics_table(ensemble: EnsembleResult) -> str:
    """Aggregated metrics for ensemble runs: ensemble row + median/min/max row."""
    em = ensemble.aggregated_metrics.get("ensemble", {})
    po = ensemble.aggregated_metrics.get("per_offset", {})

    def _fmt(v, fmt):
        if v is None:
            return "—"
        return format(v, fmt)

    rows = [
        ("ensemble (mean curve)",
         _fmt(em.get('total_return'), '+.3f'),
         _fmt(em.get('annualized_return'), '+.3f'),
         _fmt(em.get('sharpe') or 0.0, '+.2f'),
         _fmt(em.get('max_drawdown'), '.3f')),
        ("offset median",
         _fmt(po.get('total_return', {}).get('median'), '+.3f'),
         _fmt(po.get('annualized_return', {}).get('median'), '+.3f'),
         _fmt(po.get('sharpe', {}).get('median'), '+.2f'),
         _fmt(po.get('max_drawdown', {}).get('median'), '.3f')),
        ("offset min",
         _fmt(po.get('total_return', {}).get('min'), '+.3f'),
         _fmt(po.get('annualized_return', {}).get('min'), '+.3f'),
         _fmt(po.get('sharpe', {}).get('min'), '+.2f'),
         _fmt(po.get('max_drawdown', {}).get('min'), '.3f')),
        ("offset max",
         _fmt(po.get('total_return', {}).get('max'), '+.3f'),
         _fmt(po.get('annualized_return', {}).get('max'), '+.3f'),
         _fmt(po.get('sharpe', {}).get('max'), '+.2f'),
         _fmt(po.get('max_drawdown', {}).get('max'), '.3f')),
    ]
    head = (
        "<thead><tr><th>scope</th><th>total return</th>"
        "<th>annualized</th><th>sharpe</th><th>max DD</th></tr></thead>"
    )
    body = "<tbody>" + "".join(
        f"<tr><th>{n}</th><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>"
        for n, a, b, c, d in rows
    ) + "</tbody>"
    return f"<table>{head}{body}</table>"


def render_ensemble_report(
    ensemble: EnsembleResult,
    panel_data: dict[str, pd.DataFrame],
    run_date: str,
    output_dir: str | Path,
    config_hash: str = "",
    initial_equity: float = 1.0,
) -> Path:
    """Render an ensemble (staggered) portfolio backtest HTML."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    html_path = out_root / f"{run_date}.html"

    if not ensemble.individual_results or ensemble.ensemble_curve.empty:
        html_path.write_text(
            "<!doctype html><html><body><h1>Empty ensemble result</h1></body></html>",
            encoding="utf-8",
        )
    else:
        dates = pd.DatetimeIndex(ensemble.envelope["date"])
        baseline = _equal_weight_baseline(panel_data, dates)
        chart = _ensemble_chart(ensemble, baseline=baseline, initial_equity=initial_equity)
        body = (
            f"<h1>Portfolio Backtest (ensemble · k={ensemble.n_offsets}) — {run_date}</h1>"
            f"<p class='meta'>strategy: <code>{ensemble.strategy_name}</code> · "
            f"config hash: <code>{config_hash}</code></p>"
            f"{_ensemble_metrics_table(ensemble)}"
            f"<div class='chart-wrap'>{chart.render_embed()}</div>"
            f"{_per_offset_cards_html(ensemble)}"
            f"<footer>stockpool portfolio-backtest · ensemble · {run_date}</footer>"
        )
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Portfolio ensemble {run_date}</title><style>{_CSS}</style>"
            "</head><body>"
            f"{body}"
            "</body></html>"
        )
        html_path.write_text(html, encoding="utf-8")

    latest = out_root / "latest.html"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    shutil.copyfile(html_path, latest)
    return html_path


def render_portfolio_report(
    result: PortfolioBacktestResult,
    panel_data: dict[str, pd.DataFrame],
    run_date: str,
    output_dir: str | Path,
    config_hash: str = "",
) -> Path:
    """Render a single-arm portfolio backtest HTML.

    Writes ``<output_dir>/<run_date>.html`` and copies it to
    ``<output_dir>/latest.html`` for stable linking.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    html_path = out_root / f"{run_date}.html"

    if result.curve.empty:
        html_path.write_text(
            "<!doctype html><html><body><h1>Empty result</h1>"
            "<p>No bars to render — check universe/score panel.</p></body></html>",
            encoding="utf-8",
        )
    else:
        dates = pd.DatetimeIndex(result.curve["date"])
        baseline = _equal_weight_baseline(panel_data, dates)
        eq = _equity_chart(result, baseline=baseline)
        hold = _holdings_chart(result)

        body = (
            f"<h1>Portfolio Backtest — {run_date}</h1>"
            f"<p class='meta'>config hash: <code>{config_hash}</code></p>"
            f"{_metrics_table(result)}"
            f"<div class='chart-wrap'>{eq.render_embed()}</div>"
            f"<div class='chart-wrap'>{hold.render_embed()}</div>"
            f"<footer>stockpool portfolio-backtest · {run_date}</footer>"
        )
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Portfolio {run_date}</title><style>{_CSS}</style>"
            "</head><body>"
            f"{body}"
            "</body></html>"
        )
        html_path.write_text(html, encoding="utf-8")

    latest = out_root / "latest.html"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    shutil.copyfile(html_path, latest)
    return html_path
