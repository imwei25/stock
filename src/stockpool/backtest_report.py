"""HTML rendering for the composite-strategy backtest (B)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line

from stockpool.backtest_composite import EquityResult


_CSS = """
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 1400px;
         margin: 1em auto; padding: 0 1em; color: #222; }
  h1 { margin-bottom: 0.3em; }
  .meta { color: #666; margin-bottom: 1em; }
  table { border-collapse: collapse; width: 100%; margin: 0.5em 0 1.5em; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 0.95em; }
  th { background: #f6f6f6; text-align: left; }
  details { border-top: 2px solid #e6e6e6; padding: 1em 0; margin-top: 1em; }
  details summary { cursor: pointer; padding: 0.3em 0; }
  .chart-wrap { margin: 1em 0; }
  footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #eee;
           color: #888; font-size: 0.85em; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
"""


def _equity_chart(result: EquityResult, title: str) -> Line:
    """One line chart, one series per N + buy-and-hold."""
    any_curve = next(iter(result.curves.values()))
    dates = pd.DatetimeIndex(any_curve["date"]).strftime("%Y-%m-%d").tolist()

    line = (
        Line(init_opts=opts.InitOpts(width="100%", height="380px"))
        .add_xaxis(dates)
    )
    for N in sorted(result.curves.keys()):
        series_vals = [round(float(v), 4) for v in result.curves[N]["equity"].values]
        line.add_yaxis(
            f"N={N}", series_vals,
            is_smooth=True, is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
        )
    if result.buy_and_hold is not None:
        bh_vals = [round(float(v), 4) for v in result.buy_and_hold["equity"].values]
        line.add_yaxis(
            "Buy & Hold", bh_vals,
            is_smooth=True, is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(type_="dashed", width=2),
        )

    line.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        xaxis_opts=opts.AxisOpts(is_scale=True),
        yaxis_opts=opts.AxisOpts(is_scale=True, name="净值"),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside"),
            opts.DataZoomOpts(type_="slider"),
        ],
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_top="6%"),
    )
    return line


def _fmt_pct(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "—"
    if signed:
        return f"{x*100:+.2f}%"
    return f"{x*100:.2f}%"


def _metrics_table(result: EquityResult) -> str:
    rows = []
    for N in sorted(result.curves.keys()):
        m = result.metrics[N]
        rows.append(
            f"<tr>"
            f"<td>N={N}</td>"
            f"<td>{_fmt_pct(m['total_return'], signed=True)}</td>"
            f"<td>{_fmt_pct(m['annualized_return'], signed=True)}</td>"
            f"<td>{_fmt_pct(m['max_drawdown'])}</td>"
            f"<td>{m['trade_count']}</td>"
            f"<td>{_fmt_pct(m['win_rate'])}</td>"
            f"<td>{m['avg_trade_return_pct']:+.2f}%</td>"
            f"</tr>"
        )
    if result.buy_and_hold_metrics is not None:
        m = result.buy_and_hold_metrics
        rows.append(
            f"<tr>"
            f"<td>Buy &amp; Hold</td>"
            f"<td>{_fmt_pct(m['total_return'], signed=True)}</td>"
            f"<td>{_fmt_pct(m['annualized_return'], signed=True)}</td>"
            f"<td>{_fmt_pct(m['max_drawdown'])}</td>"
            f"<td>{m['trade_count']}</td>"
            f"<td>—</td>"
            f"<td>—</td>"
            f"</tr>"
        )
    return f"""
      <table>
        <thead><tr>
          <th>策略</th><th>总收益</th><th>年化</th><th>最大回撤</th>
          <th>交易次数</th><th>胜率</th><th>平均单笔</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    """


def _stock_section(code: str, name: str, result: EquityResult) -> str:
    try:
        chart_html = _equity_chart(result, f"{code} {name}").render_embed()
    except Exception as e:
        chart_html = f"<p style='color:#a00'>图表生成失败: {e}</p>"
    return f"""
    <details id="stock-{code}" open>
      <summary>
        <span style="font-size:1.2em; font-weight:bold">{code} {name}</span>
      </summary>
      <div class="chart-wrap">{chart_html}</div>
      {_metrics_table(result)}
    </details>
    """


def render_backtest_report(
    per_stock: list[tuple[str, str, EquityResult]],
    run_date: str,
    output_dir: str | Path,
) -> Path:
    """Render the backtest HTML page.

    per_stock: list of (code, name, EquityResult) tuples.
    Returns the path to <output_dir>/<run_date>.html. Also writes latest.html.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_date}.html"

    index_rows = "".join(
        f'<li><a href="#stock-{code}">{code} {name}</a></li>'
        for code, name, _ in per_stock
    )
    sections = "".join(_stock_section(c, n, r) for c, n, r in per_stock)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>综合策略回测 · {run_date}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>综合策略回测 · {run_date}</h1>
  <p class="meta">基于当前权重对历史每日重建综合评级,模拟 N=5/10/20 持有期策略与 Buy &amp; Hold 基准。</p>
  <h2>索引</h2>
  <ul>{index_rows}</ul>
  {sections}
  <footer>
    <p>⚠️ <strong>免责声明:</strong>回测假设无手续费、无 T+1、无滑点,与真实交易存在差距,仅供技术参考。</p>
  </footer>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    shutil.copyfile(out_path, output_dir / "latest.html")
    return out_path
