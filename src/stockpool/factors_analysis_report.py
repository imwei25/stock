"""HTML report for FactorAnalysisResult.

Renders a single self-contained HTML file with:
  * summary header (n_factors / n_stocks / n_days / date range / horizon)
  * ranking table (factor / mean_ic / ic_ir / abs_ic_mean / half_life)
  * IC time-series multi-line chart (top-10 by |ic_ir|)
  * correlation heatmap (F × F)
  * regime IC table (factor × regime)
  * picked selection box

Uses pyecharts (already a project dep) — no new dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import HeatMap, Line, Page

from stockpool.factors_analysis import FactorAnalysisResult


def _summary_html(result: FactorAnalysisResult) -> str:
    return (
        '<div style="font-family:sans-serif;padding:12px 18px;background:#f5f7fa;'
        'border-radius:6px;margin-bottom:18px">'
        f'<b>因子分析报告</b> &nbsp;|&nbsp; '
        f'因子数 {len(result.factor_names)} &nbsp;|&nbsp; '
        f'股票数 {result.n_stocks} &nbsp;|&nbsp; '
        f'交易日 {result.n_days} &nbsp;|&nbsp; '
        f'horizon {result.horizon} &nbsp;|&nbsp; '
        f'区间 {result.start_date.date()} → {result.end_date.date()}'
        '</div>'
    )


def _ranking_table_html(result: FactorAnalysisResult) -> str:
    has_degenerate = (
        isinstance(result.degenerate_day_ratio, pd.Series)
        and len(result.degenerate_day_ratio) > 0
    )
    rows = []
    for n in result.factor_names:
        row = {
            "factor": n,
            "mean_ic": float(result.mean_ic[n]),
            "ic_ir": float(result.ic_ir[n]),
            "abs_ic_mean": float(result.abs_ic_mean[n]),
            "half_life": float(result.half_life[n]),
        }
        if has_degenerate:
            row["degenerate"] = float(result.degenerate_day_ratio.get(n, float("nan")))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("ic_ir", key=lambda s: s.abs(), ascending=False)
    if not has_degenerate:
        body = df.to_html(
            index=False, float_format="%.4f", border=0,
            classes="ranking-table",
        )
    else:
        # Render manually so we can apply a .warn class to the degenerate cell
        # when ratio > 0.20 (red text via CSS).
        cols = ["factor", "mean_ic", "ic_ir", "abs_ic_mean", "half_life", "degenerate"]
        header_labels = {
            "factor": "factor", "mean_ic": "mean_ic", "ic_ir": "ic_ir",
            "abs_ic_mean": "abs_ic_mean", "half_life": "half_life",
            "degenerate": "degenerate %",
        }
        thead = "".join(f"<th>{header_labels[c]}</th>" for c in cols)
        body_rows = []
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if c == "factor":
                    cells.append(f"<td>{v}</td>")
                elif c == "degenerate":
                    if pd.isna(v):
                        cells.append('<td>nan</td>')
                    else:
                        cls = ' class="warn"' if v > 0.20 else ""
                        cells.append(f'<td{cls}>{v * 100:.2f}%</td>')
                else:
                    cells.append(
                        f"<td>nan</td>" if pd.isna(v) else f"<td>{v:.4f}</td>"
                    )
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        body = (
            '<table border="0" class="ranking-table">'
            f"<thead><tr>{thead}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
        )
    return (
        '<h3 style="font-family:sans-serif">因子排名 (按 |IC IR| 降序)</h3>'
        + body
    )


def _ic_timeseries_chart(result: FactorAnalysisResult, top_k: int = 10) -> Line:
    top = result.ic_ir.abs().sort_values(ascending=False).head(top_k).index.tolist()
    line = Line(init_opts=opts.InitOpts(width="1100px", height="380px"))
    if not top:
        return line
    dates = result.daily_ic[top[0]].index
    line.add_xaxis([d.strftime("%Y-%m-%d") for d in dates])
    for n in top:
        smooth_ic = result.daily_ic[n].rolling(20, min_periods=5).mean()
        line.add_yaxis(
            n, [None if pd.isna(v) else round(v, 4) for v in smooth_ic.tolist()],
            is_symbol_show=False, is_smooth=True,
        )
    line.set_global_opts(
        title_opts=opts.TitleOpts(title=f"Top-{len(top)} 因子 20 日滚动 IC"),
        xaxis_opts=opts.AxisOpts(type_="category"),
        yaxis_opts=opts.AxisOpts(type_="value", min_=-0.3, max_=0.3),
        legend_opts=opts.LegendOpts(pos_top="bottom"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts(type_="inside")],
    )
    return line


def _correlation_heatmap(result: FactorAnalysisResult) -> HeatMap:
    corr = result.ic_correlation
    names = list(corr.index)
    data = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            data.append([i, j, round(float(corr.iloc[i, j]), 3)])
    hm = HeatMap(init_opts=opts.InitOpts(width="900px", height="650px"))
    hm.add_xaxis(names)
    hm.add_yaxis("IC corr", names, data)
    hm.set_global_opts(
        title_opts=opts.TitleOpts(title="因子 IC 相关性热图"),
        visualmap_opts=opts.VisualMapOpts(
            min_=-1, max_=1, range_color=["#3060cf", "#ffffff", "#c4463a"],
            pos_left="right",
        ),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60)),
    )
    return hm


def _regime_table_html(result: FactorAnalysisResult) -> str:
    if not result.regime_ic:
        return ""
    df = pd.DataFrame(result.regime_ic)[list(result.regime_ic.keys())]
    df.index.name = "factor"
    return (
        '<h3 style="font-family:sans-serif">不同 regime 下的均值 IC</h3>'
        + df.reset_index().to_html(
            index=False, float_format="%.4f", border=0,
            classes="regime-table",
        )
    )


def _picked_box_html(picked: Sequence[str]) -> str:
    if not picked:
        return ""
    items = ", ".join(f"<code>{n}</code>" for n in picked)
    return (
        '<div style="font-family:sans-serif;padding:12px 18px;'
        'background:#fffceb;border-left:4px solid #f5c518;margin:18px 0">'
        f'<b>pick_top_factors 默认参数下选出</b>({len(picked)} 个):<br>{items}'
        '</div>'
    )


def render_factor_analysis_report(
    result: FactorAnalysisResult,
    out_path: str | Path,
    picked: Sequence[str] = (),
) -> Path:
    """Render `result` to a single self-contained HTML file at `out_path`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page = Page(layout=Page.SimplePageLayout)
    page.add(_ic_timeseries_chart(result))
    page.add(_correlation_heatmap(result))

    html_chunks = [
        _summary_html(result),
        _picked_box_html(picked),
        _ranking_table_html(result),
        _regime_table_html(result),
    ]

    base_html = page.render_embed()
    head = (
        '<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"/>'
        '<title>因子分析报告</title>'
        '<style>'
        'body{font-family:Segoe UI, "PingFang SC", sans-serif;margin:24px;}'
        'table{border-collapse:collapse;font-size:13px;margin:8px 0 24px;}'
        'th{background:#f0f3f7;padding:6px 10px;text-align:left;}'
        'td{padding:4px 10px;border-bottom:1px solid #e6e9ed;}'
        'td.warn{color:#c4463a;font-weight:600;}'
        '</style></head><body>'
    )
    out_path.write_text(
        head + "\n".join(html_chunks) + base_html + "</body></html>",
        encoding="utf-8",
    )
    return out_path
