"""HTML rendering for portfolio AB results.

Layout:
  * Banner (arm names + headline diff)
  * Aggregated metrics table — two columns + Δ + Δ%
  * Equity curves overlay — both arms' primary curves + B&H baseline
  * Per-stock contribution decomposition:
      - top 15 contributors per arm (table)
      - set analysis: only A / only B / both (counts + total contribution sum)
  * Red banner per failed arm
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line

from stockpool.portfolio_ab.runner import ABResult, ArmResult


_CSS = """
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif;
         max-width: 1400px; margin: 1em auto; padding: 0 1em; color: #222; }
  h1 { margin-bottom: 0.3em; }
  .meta { color: #666; margin-bottom: 1em; }
  .banner { padding: 0.6em 1em; border-radius: 6px; margin: 0.8em 0; }
  .banner-info { background: #eef5ff; border: 1px solid #b6d4fe; }
  .banner-fail { background: #fde2e1; border: 1px solid #f5b7b1; color: #842029; }
  table { border-collapse: collapse; width: 100%; margin: 0.5em 0 1.5em; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 0.95em; }
  th { background: #f6f6f6; text-align: left; }
  .pos { color: #166534; }
  .neg { color: #991b1b; }
  .two-col { display: flex; gap: 1.5em; }
  .two-col > div { flex: 1; }
  .chart-wrap { margin: 1em 0; }
  footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #eee;
           color: #888; font-size: 0.85em; }
"""


def _delta_cell(va, vb, fmt=".3f") -> str:
    if va is None or vb is None:
        return "<td>—</td><td>—</td>"
    d = vb - va
    cls = "pos" if d > 0 else ("neg" if d < 0 else "")
    pct = (d / abs(va) * 100) if va not in (0, None) else None
    pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
    return f"<td class='{cls}'>{d:+{fmt}}{pct_str}</td>"


def _aggregated_table(arm_a: ArmResult, arm_b: ArmResult) -> str:
    ma = arm_a.primary_metrics
    mb = arm_b.primary_metrics
    rows = []
    keys = [
        ("total_return", ".3f"),
        ("annualized_return", ".3f"),
        ("sharpe", ".2f"),
        ("max_drawdown", ".3f"),
        ("trade_count", ".0f"),
        ("win_rate", ".3f"),
    ]
    for k, fmt in keys:
        va, vb = ma.get(k), mb.get(k)
        va_str = f"{va:{fmt}}" if va is not None else "—"
        vb_str = f"{vb:{fmt}}" if vb is not None else "—"
        # Δ shown for numeric, non-None values.
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta_cell = _delta_cell(float(va), float(vb), fmt)
        else:
            delta_cell = "<td>—</td>"
        rows.append(
            f"<tr><th>{k}</th><td>{va_str}</td><td>{vb_str}</td>{delta_cell}</tr>"
        )
    head = (
        "<thead><tr><th>metric</th>"
        f"<th>{arm_a.name}</th><th>{arm_b.name}</th>"
        f"<th>Δ (B − A)</th></tr></thead>"
    )
    return f"<table>{head}<tbody>{''.join(rows)}</tbody></table>"


def _equity_overlay(arm_a: ArmResult, arm_b: ArmResult) -> Line:
    """Both arms' primary equity curves overlaid on the same x-axis."""
    curve_a = arm_a.primary_curve
    curve_b = arm_b.primary_curve
    # Build the union date axis so both curves render aligned even if one
    # arm dropped a few bars.
    da = pd.to_datetime(curve_a["date"]) if not curve_a.empty else pd.DatetimeIndex([])
    db = pd.to_datetime(curve_b["date"]) if not curve_b.empty else pd.DatetimeIndex([])
    union = pd.DatetimeIndex(sorted(set(da).union(set(db))))
    series_a = (
        pd.Series(curve_a["equity"].values, index=da).reindex(union).ffill()
        if not curve_a.empty else pd.Series([], dtype=float)
    )
    series_b = (
        pd.Series(curve_b["equity"].values, index=db).reindex(union).ffill()
        if not curve_b.empty else pd.Series([], dtype=float)
    )
    dates = union.strftime("%Y-%m-%d").tolist()
    line = (
        Line(init_opts=opts.InitOpts(width="100%", height="480px"))
        .add_xaxis(dates)
    )
    if len(series_a) > 0:
        line.add_yaxis(
            arm_a.name, [round(float(v), 4) for v in series_a.values],
            is_smooth=True, is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#2563eb"),
        )
    if len(series_b) > 0:
        line.add_yaxis(
            arm_b.name, [round(float(v), 4) for v in series_b.values],
            is_smooth=True, is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#c0392b"),
        )
    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title="Arm net asset value", pos_left="center", pos_top="2%",
        ),
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


def _per_stock_contribution(arm: ArmResult) -> pd.DataFrame:
    """Aggregate trade returns per code → contribution share.

    Returns a DataFrame: code / trade_count / total_ret_contrib / share_pct
    Sorted descending by absolute contribution.
    """
    rows: dict[str, dict] = {}
    for t in arm.trades:
        # Each trade's contribution to the portfolio is weight_at_entry * ret
        # (approximate; ignores compounding interactions between rebalances).
        slot = rows.setdefault(t.code, {"count": 0, "contrib": 0.0})
        slot["count"] += 1
        slot["contrib"] += t.weight_at_entry * t.ret
    if not rows:
        return pd.DataFrame(columns=["code", "trade_count", "contrib", "share_pct"])
    total_abs = sum(abs(v["contrib"]) for v in rows.values()) or 1.0
    out = pd.DataFrame([
        {
            "code": code,
            "trade_count": v["count"],
            "contrib": v["contrib"],
            "share_pct": v["contrib"] / total_abs * 100,
        }
        for code, v in rows.items()
    ])
    out = out.reindex(out["contrib"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def _contribution_table_html(arm: ArmResult, n: int = 15) -> str:
    df = _per_stock_contribution(arm)
    if df.empty:
        return f"<p>{arm.name}: no closed trades.</p>"
    head = df.head(n)
    rows = "".join(
        f"<tr><td>{r['code']}</td><td>{int(r['trade_count'])}</td>"
        f"<td class='{'pos' if r['contrib'] > 0 else 'neg'}'>{r['contrib']:+.4f}</td>"
        f"<td>{r['share_pct']:+.1f}%</td></tr>"
        for _, r in head.iterrows()
    )
    return (
        f"<h3>{arm.name} — top {min(n, len(df))} contributors</h3>"
        "<table><thead><tr><th>code</th><th>#trades</th>"
        "<th>contrib</th><th>share</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _set_analysis(arm_a: ArmResult, arm_b: ArmResult) -> str:
    """Set analysis of traded codes — 'only A' / 'only B' / 'both'."""
    codes_a = set(_per_stock_contribution(arm_a)["code"].tolist())
    codes_b = set(_per_stock_contribution(arm_b)["code"].tolist())
    only_a = codes_a - codes_b
    only_b = codes_b - codes_a
    both = codes_a & codes_b
    rows = [
        ("Only A", arm_a.name, len(only_a)),
        ("Only B", arm_b.name, len(only_b)),
        ("Both", "—", len(both)),
    ]
    body = "".join(
        f"<tr><th>{label}</th><td>{name}</td><td>{n}</td></tr>"
        for label, name, n in rows
    )
    return (
        "<h3>Traded-code set analysis</h3>"
        "<table><thead><tr><th>bucket</th><th>arm</th>"
        "<th>codes</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _banner_html(ab_result: ABResult) -> str:
    names = list(ab_result.arms)
    if len(names) != 2:
        return ""
    a, b = ab_result.arms[names[0]], ab_result.arms[names[1]]
    bits = [f"<div class='banner banner-info'>Comparing <b>{a.name}</b> vs <b>{b.name}</b></div>"]
    if a.failed:
        bits.append(
            f"<div class='banner banner-fail'>Arm <b>{a.name}</b> FAILED: {a.error}</div>"
        )
    if b.failed:
        bits.append(
            f"<div class='banner banner-fail'>Arm <b>{b.name}</b> FAILED: {b.error}</div>"
        )
    return "".join(bits)


def render_portfolio_ab_report(
    ab_result: ABResult,
    run_date: str,
    output_dir: str | Path,
) -> Path:
    """Write the AB comparison HTML to ``output_dir/<run_date>.html`` (+ latest.html)."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    html_path = out_root / f"{run_date}.html"

    names = list(ab_result.arms)
    if len(names) != 2:
        html_path.write_text(
            "<!doctype html><html><body><h1>Invalid AB result</h1>"
            f"<p>Expected 2 arms, got {len(names)}.</p></body></html>",
            encoding="utf-8",
        )
    else:
        arm_a = ab_result.arms[names[0]]
        arm_b = ab_result.arms[names[1]]

        banner = _banner_html(ab_result)
        metrics_table = _aggregated_table(arm_a, arm_b)
        chart = _equity_overlay(arm_a, arm_b)
        contrib_a = _contribution_table_html(arm_a)
        contrib_b = _contribution_table_html(arm_b)
        set_analysis = _set_analysis(arm_a, arm_b)

        body = (
            f"<h1>Portfolio A/B — {run_date}</h1>"
            f"{banner}"
            "<h2>Aggregated metrics</h2>"
            f"{metrics_table}"
            f"<div class='chart-wrap'>{chart.render_embed()}</div>"
            "<h2>Per-stock contribution</h2>"
            f"<div class='two-col'><div>{contrib_a}</div><div>{contrib_b}</div></div>"
            f"{set_analysis}"
            f"<footer>stockpool portfolio-ab · {run_date}</footer>"
        )
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Portfolio AB {run_date}</title>"
            f"<style>{_CSS}</style></head><body>"
            f"{body}</body></html>"
        )
        html_path.write_text(html, encoding="utf-8")

    latest = out_root / "latest.html"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    shutil.copyfile(html_path, latest)
    return html_path
