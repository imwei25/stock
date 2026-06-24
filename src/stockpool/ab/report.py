"""HTML report for A/B test results.

Layout (top → bottom):
  1. Metadata banner (arm names, differing fields, base hash, per-arm counts).
  2. Aggregate diff table over common stocks.
  3. Sharpe scatter (A.sharpe x B.sharpe per common stock).
  4. Sharpe diff histogram (B.sharpe - A.sharpe).
  5. Per-stock cards (3-series equity chart + side-by-side metric table).
  6. Failure detail + full effective_cfg dumps (folded).
"""
from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd
import yaml
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Scatter

from stockpool.ab.runner import ABResult, ArmResult
from stockpool.ab.score_ic import arm_score_ic
from stockpool.backtest_composite import EquityResult
from stockpool.backtest_report import _CSS

_METRIC_DEFS = [
    ("total_return",        "Total return",        True,  "pct"),
    ("annualized_return",   "Annualized return",   True,  "pct"),
    ("sharpe",              "Sharpe",              True,  "num"),
    ("max_drawdown",        "Max drawdown",        False, "pct"),
    ("win_rate",            "Win rate",            True,  "pct"),
    ("avg_trade_return_pct","Avg trade ret %",     True,  "raw"),
    ("trade_count",         "Trade count",         None,  "int"),
]


def _fmt(val, kind: str) -> str:
    if val is None:
        return "—"
    if kind == "pct":
        return f"{val*100:+.2f}%"
    if kind == "num":
        return f"{val:+.3f}"
    if kind == "raw":
        return f"{val:+.2f}"
    if kind == "int":
        return str(int(val))
    return str(val)


def _safe_num(v) -> float:
    """Coerce metric value to float, treating None as 0.0."""
    return 0.0 if v is None else float(v)


def _arm_metrics(arm_result: ArmResult) -> dict[str, dict[str, float]]:
    """Map code → metrics dict (the single-N dict from EquityResult)."""
    out: dict[str, dict[str, float]] = {}
    for code, _name, res in arm_result.per_stock:
        N = next(iter(res.metrics))
        out[code] = res.metrics[N]
    return out


def compute_diff_table(arm_a: ArmResult, arm_b: ArmResult) -> dict:
    """Aggregate per-metric stats over stocks present in BOTH arms.

    Returns a dict:
      {
        "common_stocks_count": int,
        "common": list[str],
        "rows": list[dict],   # one per metric
      }
    Each row: label / kind / higher_better / a_mean / a_median / b_mean /
    b_median / diff_mean / a_wins / b_wins.
    """
    a_metrics = _arm_metrics(arm_a)
    b_metrics = _arm_metrics(arm_b)
    common = sorted(set(a_metrics) & set(b_metrics))

    rows = []
    for key, label, higher_better, kind in _METRIC_DEFS:
        if not common:
            rows.append({
                "label": label, "kind": kind, "higher_better": higher_better,
                "a_mean": None, "a_median": None,
                "b_mean": None, "b_median": None,
                "diff_mean": None, "a_wins": 0, "b_wins": 0,
            })
            continue
        a_vals = [_safe_num(a_metrics[c].get(key)) for c in common]
        b_vals = [_safe_num(b_metrics[c].get(key)) for c in common]
        a_mean = sum(a_vals) / len(a_vals)
        b_mean = sum(b_vals) / len(b_vals)
        a_med = statistics.median(a_vals)
        b_med = statistics.median(b_vals)
        if higher_better is None:
            a_wins = b_wins = 0
        elif higher_better:
            a_wins = sum(1 for a, b in zip(a_vals, b_vals) if a > b)
            b_wins = sum(1 for a, b in zip(a_vals, b_vals) if b > a)
        else:
            a_wins = sum(1 for a, b in zip(a_vals, b_vals) if a < b)
            b_wins = sum(1 for a, b in zip(a_vals, b_vals) if b < a)
        rows.append({
            "label": label, "kind": kind, "higher_better": higher_better,
            "a_mean": a_mean, "a_median": a_med,
            "b_mean": b_mean, "b_median": b_med,
            "diff_mean": b_mean - a_mean,
            "a_wins": a_wins, "b_wins": b_wins,
        })
    return {"common_stocks_count": len(common), "common": common, "rows": rows}


def _diff_table_html(table: dict, arm_a_name: str, arm_b_name: str) -> str:
    header = (
        f"<tr><th>Metric</th>"
        f"<th>{arm_a_name} mean</th><th>{arm_a_name} median</th>"
        f"<th>{arm_b_name} mean</th><th>{arm_b_name} median</th>"
        f"<th>Δ mean (B−A)</th>"
        f"<th>{arm_a_name} wins</th><th>{arm_b_name} wins</th></tr>"
    )
    body_rows = []
    for row in table["rows"]:
        label = row["label"]
        if row["higher_better"] is False:
            label += " <span style='color:#888;font-size:.85em'>(lower better)</span>"
        body_rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{_fmt(row['a_mean'], row['kind'])}</td>"
            f"<td>{_fmt(row['a_median'], row['kind'])}</td>"
            f"<td>{_fmt(row['b_mean'], row['kind'])}</td>"
            f"<td>{_fmt(row['b_median'], row['kind'])}</td>"
            f"<td><strong>{_fmt(row['diff_mean'], row['kind'])}</strong></td>"
            f"<td>{row['a_wins']}</td><td>{row['b_wins']}</td></tr>"
        )
    return (
        f"<table><thead>{header}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _sharpe_scatter(arm_a: ArmResult, arm_b: ArmResult) -> str:
    a_metrics = _arm_metrics(arm_a)
    b_metrics = _arm_metrics(arm_b)
    common = sorted(set(a_metrics) & set(b_metrics))
    if not common:
        return "<p>No common stocks — scatter omitted.</p>"
    points = [[a_metrics[c]["sharpe"], b_metrics[c]["sharpe"], c] for c in common]
    vals = [p[0] for p in points] + [p[1] for p in points]
    lo, hi = min(vals), max(vals)
    if lo == hi:
        hi = lo + 1e-6

    sc = (
        Scatter(init_opts=opts.InitOpts(width="100%", height="420px"))
        .add_xaxis([p[0] for p in points])
        .add_yaxis(
            f"{arm_b.name} vs {arm_a.name}",
            [p[1] for p in points],
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Sharpe scatter — above diagonal = {arm_b.name} wins",
                pos_left="center",
            ),
            xaxis_opts=opts.AxisOpts(
                name=f"{arm_a.name} Sharpe", min_=lo, max_=hi, type_="value",
            ),
            yaxis_opts=opts.AxisOpts(
                name=f"{arm_b.name} Sharpe", min_=lo, max_=hi, type_="value",
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
        )
    )
    sc.options["series"][0]["markLine"] = {
        "symbol": "none",
        "lineStyle": {"color": "#999", "type": "dashed"},
        "data": [[{"coord": [lo, lo]}, {"coord": [hi, hi]}]],
    }
    return sc.render_embed()


def _diff_histogram(arm_a: ArmResult, arm_b: ArmResult) -> str:
    a_metrics = _arm_metrics(arm_a)
    b_metrics = _arm_metrics(arm_b)
    common = sorted(set(a_metrics) & set(b_metrics))
    diffs = [b_metrics[c]["sharpe"] - a_metrics[c]["sharpe"] for c in common]
    if not diffs:
        return "<p>No common stocks — histogram omitted.</p>"

    lo, hi = min(diffs), max(diffs)
    if lo == hi:
        hi = lo + 1e-6
    n_bins = min(12, max(4, len(diffs) // 2 or 4))
    width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for d in diffs:
        i = min(int((d - lo) / width), n_bins - 1)
        bins[i] += 1
    labels = [f"{lo + width*i:+.2f}" for i in range(n_bins)]

    bar = (
        Bar(init_opts=opts.InitOpts(width="100%", height="320px"))
        .add_xaxis(labels)
        .add_yaxis(
            f"{arm_b.name} − {arm_a.name} (Sharpe)", bins,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Sharpe diff distribution",
                                      pos_left="center"),
            xaxis_opts=opts.AxisOpts(name="Δ Sharpe"),
            yaxis_opts=opts.AxisOpts(name="stocks"),
            legend_opts=opts.LegendOpts(pos_top="6%"),
        )
    )
    return bar.render_embed()


def _ab_equity_chart(
    a_result: EquityResult | None,
    b_result: EquityResult | None,
    a_name: str,
    b_name: str,
    title: str,
) -> str:
    ref = a_result if a_result is not None else b_result
    if ref is None:
        return ""
    any_curve = next(iter(ref.curves.values()))
    dates = pd.DatetimeIndex(any_curve["date"]).strftime("%Y-%m-%d").tolist()

    line = Line(init_opts=opts.InitOpts(width="100%", height="420px")).add_xaxis(dates)
    if a_result is not None:
        N = next(iter(a_result.curves))
        vals = [round(float(v), 4) for v in a_result.curves[N]["equity"].values]
        line.add_yaxis(a_name, vals, is_smooth=True, is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False))
    if b_result is not None:
        N = next(iter(b_result.curves))
        vals = [round(float(v), 4) for v in b_result.curves[N]["equity"].values]
        line.add_yaxis(b_name, vals, is_smooth=True, is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False))
    if ref.buy_and_hold is not None:
        bh = [round(float(v), 4) for v in ref.buy_and_hold["equity"].values]
        line.add_yaxis("Buy & Hold", bh, is_smooth=True, is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False),
                       linestyle_opts=opts.LineStyleOpts(type_="dashed", width=2))

    line.set_global_opts(
        title_opts=opts.TitleOpts(title=title, pos_left="center"),
        xaxis_opts=opts.AxisOpts(is_scale=True,
                                 axislabel_opts=opts.LabelOpts(rotate=30, font_size=10)),
        yaxis_opts=opts.AxisOpts(is_scale=True, name="净值"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_top="6%"),
        datazoom_opts=[opts.DataZoomOpts(type_="inside"),
                       opts.DataZoomOpts(type_="slider", pos_bottom="2%")],
    )
    line.options["grid"] = {"top": "18%", "bottom": "16%", "left": "8%", "right": "4%",
                            "containLabel": True}
    return line.render_embed()


def _per_stock_cards(arm_a: ArmResult, arm_b: ArmResult) -> str:
    a_map = {code: (name, res) for code, name, res in arm_a.per_stock}
    b_map = {code: (name, res) for code, name, res in arm_b.per_stock}
    all_codes = sorted(set(a_map) | set(b_map))
    sections = []
    for i, code in enumerate(all_codes):
        a_entry = a_map.get(code)
        b_entry = b_map.get(code)
        name = (a_entry or b_entry)[0]
        a_res = a_entry[1] if a_entry else None
        b_res = b_entry[1] if b_entry else None
        title = f"{code} {name}"
        if a_res is None:
            title += f" [Arm {arm_a.name} failed]"
        if b_res is None:
            title += f" [Arm {arm_b.name} failed]"
        chart = _ab_equity_chart(a_res, b_res, arm_a.name, arm_b.name, title)
        rows = []
        for key, label, higher_better, kind in _METRIC_DEFS:
            a_v = None if a_res is None else a_res.metrics[next(iter(a_res.metrics))].get(key)
            b_v = None if b_res is None else b_res.metrics[next(iter(b_res.metrics))].get(key)
            d = (b_v - a_v) if (a_v is not None and b_v is not None) else None
            rows.append(
                f"<tr><td>{label}</td>"
                f"<td>{_fmt(a_v, kind)}</td>"
                f"<td>{_fmt(b_v, kind)}</td>"
                f"<td>{_fmt(d, kind)}</td></tr>"
            )
        table = (
            f"<table><thead><tr><th>Metric</th>"
            f"<th>{arm_a.name}</th><th>{arm_b.name}</th><th>Δ</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
        open_attr = "open" if i < 3 else ""
        sections.append(
            f"<details {open_attr}><summary>"
            f"<span style='font-size:1.1em;font-weight:bold'>{title}</span>"
            f"</summary><div class='chart-wrap'>{chart}</div>{table}</details>"
        )
    return "".join(sections)


def _metadata_banner(ab_result: ABResult) -> str:
    a, b = ab_result.arm_a, ab_result.arm_b
    a_arm = ab_result.ab_cfg.arms[a.name]
    b_arm = ab_result.ab_cfg.arms[b.name]
    return (
        f"<h1>A/B Test Report — {ab_result.run_date}</h1>"
        f"<p class='meta'>Base config: {ab_result.ab_cfg.base_config} "
        f"(hash: {ab_result.base_cfg.content_hash})</p>"
        f"<div class='banner'>"
        f"  <h3>Arm A: {a.name}</h3>"
        f"  <pre>{yaml.safe_dump(a_arm.model_dump(), sort_keys=False)}</pre>"
        f"  <p>{len(a.per_stock)} succeeded, {len(a.failed)} failed.</p>"
        f"  <h3>Arm B: {b.name}</h3>"
        f"  <pre>{yaml.safe_dump(b_arm.model_dump(), sort_keys=False)}</pre>"
        f"  <p>{len(b.per_stock)} succeeded, {len(b.failed)} failed.</p>"
        f"</div>"
    )


def _failure_detail(ab_result: ABResult) -> str:
    def _format_one(arm: ArmResult) -> str:
        if not arm.failed:
            return f"<p>{arm.name}: no failures.</p>"
        rows = "".join(
            f"<li><code>{code}</code>: {err}</li>" for code, err in arm.failed
        )
        return f"<p>{arm.name}:</p><ul>{rows}</ul>"
    return (
        f"<details><summary>Failure detail</summary>"
        f"{_format_one(ab_result.arm_a)}{_format_one(ab_result.arm_b)}"
        f"</details>"
    )


def _full_cfg_dump(ab_result: ABResult) -> str:
    a_yaml = yaml.safe_dump(ab_result.arm_a.effective_cfg.model_dump(), sort_keys=False)
    b_yaml = yaml.safe_dump(ab_result.arm_b.effective_cfg.model_dump(), sort_keys=False)
    return (
        f"<details><summary>Full effective configs</summary>"
        f"<h4>Arm A: {ab_result.arm_a.name}</h4><pre>{a_yaml}</pre>"
        f"<h4>Arm B: {ab_result.arm_b.name}</h4><pre>{b_yaml}</pre>"
        f"</details>"
    )


def _arm_label_basis(arm: ArmResult) -> str:
    """ml_factor arm 用其训练标签口径,否则默认 open(与 T+1 执行对齐)。"""
    cfg = arm.effective_cfg
    if cfg.strategy.name == "ml_factor":
        return cfg.strategy.ml_factor.label_basis
    return "open"


def _score_ic_section(arm_a: ArmResult, arm_b: ArmResult) -> str:
    """final_score 横截面 rank-IC 表(两 arm × 各 holding-day horizon)。

    衡量预测信号质量,**不含 sizing/成本/执行** —— 与 Sharpe 表互补。小样本下 IC
    信噪比高于 Sharpe;因子侧改动以 IC 为主判据,执行/sizing 侧改动看不出 IC 差。
    """
    horizons = sorted(set(arm_a.effective_cfg.backtest.equity_curve_holding_days)
                      | set(arm_b.effective_cfg.backtest.equity_curve_holding_days))
    ic_a = arm_score_ic(arm_a.per_stock, horizons, label_basis=_arm_label_basis(arm_a))
    ic_b = arm_score_ic(arm_b.per_stock, horizons, label_basis=_arm_label_basis(arm_b))

    def _num(v, fmt="{:+.4f}"):
        return fmt.format(v) if v is not None else "—"

    rows = []
    for h in horizons:
        a, b = ic_a.get(h, {}), ic_b.get(h, {})
        a_ic, b_ic = a.get("mean_ic"), b.get("mean_ic")
        d_ic = (b_ic - a_ic) if (a_ic is not None and b_ic is not None) else None
        rows.append(
            f"<tr><td>h={h}</td>"
            f"<td>{_num(a_ic)}</td><td>{_num(a.get('ic_ir'), '{:+.3f}')}</td>"
            f"<td>{_num(a.get('abs_ic_mean'), '{:.4f}')}</td>"
            f"<td>{_num(b_ic)}</td><td>{_num(b.get('ic_ir'), '{:+.3f}')}</td>"
            f"<td>{_num(b.get('abs_ic_mean'), '{:.4f}')}</td>"
            f"<td><strong>{_num(d_ic)}</strong></td>"
            f"<td>{a.get('n_days', 0)}/{a.get('n_stocks', 0)}</td></tr>"
        )
    header = (
        f"<tr><th>Horizon</th>"
        f"<th>{arm_a.name} IC</th><th>{arm_a.name} ICIR</th><th>{arm_a.name} |IC|</th>"
        f"<th>{arm_b.name} IC</th><th>{arm_b.name} ICIR</th><th>{arm_b.name} |IC|</th>"
        f"<th>Δ IC (B−A)</th><th>days/stocks</th></tr>"
    )
    note = (
        "<p class='meta'>横截面 rank-IC(Spearman)of <code>final_score</code> vs "
        "前瞻 h 日 forward return(各 arm 按其 label_basis:ml_factor open 基准用 "
        "open[t+1+h]/open[t+1]−1,否则收盘到收盘;ICIR 带 Newey-West 修正)。衡量预测力,"
        "不含 sizing/成本/执行 —— 与 Sharpe 互补;小样本下比 Sharpe 更可靠,但执行/sizing "
        "侧改动不改 score、IC 看不出差异。</p>"
    )
    return (
        f"{note}<table><thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_ab_report(ab_result: ABResult, output_dir: str | Path) -> Path:
    """Render the full A/B HTML report. Writes <date>.html and latest.html."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ab_result.run_date}.html"

    banner = _metadata_banner(ab_result)
    table = compute_diff_table(ab_result.arm_a, ab_result.arm_b)
    table_html = _diff_table_html(table, ab_result.arm_a.name, ab_result.arm_b.name)
    scatter = _sharpe_scatter(ab_result.arm_a, ab_result.arm_b)
    histogram = _diff_histogram(ab_result.arm_a, ab_result.arm_b)
    cards = _per_stock_cards(ab_result.arm_a, ab_result.arm_b)
    score_ic_html = _score_ic_section(ab_result.arm_a, ab_result.arm_b)
    failures = _failure_detail(ab_result)
    cfg_dump = _full_cfg_dump(ab_result)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
  <meta charset="utf-8">
  <title>A/B Report · {ab_result.run_date}</title>
  <style>{_CSS}
    .banner {{ border: 1px solid #e6e6e6; padding: 1em; margin: 1em 0; }}
    .banner pre {{ background: #f6f6f6; padding: 0.6em; overflow-x: auto; }}
  </style>
</head><body>
  {banner}
  <h2>Aggregate (over {table['common_stocks_count']} common stocks)</h2>
  {table_html}
  <h2>Predictive IC (cross-sectional rank-IC of final_score)</h2>
  {score_ic_html}
  <h2>Sharpe scatter</h2>
  <div class='chart-wrap'>{scatter}</div>
  <h2>Sharpe diff distribution</h2>
  <div class='chart-wrap'>{histogram}</div>
  <h2>Per-stock comparison</h2>
  {cards}
  <h2>Failures &amp; reproducibility</h2>
  {failures}
  {cfg_dump}
  <footer><p>Generated by stockpool ab.</p></footer>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    latest = output_dir / "latest.html"
    latest.write_bytes(out_path.read_bytes())
    return out_path
