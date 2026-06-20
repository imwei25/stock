"""HTML 报告生成 — pyecharts driver."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Kline, Line

from stockpool.signals import Trigger

if TYPE_CHECKING:
    from stockpool.recommend_pool import PoolBEntry


_VERDICT_LABEL = {
    "strong_buy":   ("🟢🟢", "强烈买入", "#0a7d24"),
    "buy":          ("🟢",   "买入观察", "#37b54a"),
    "neutral":      ("⚪",   "观望",     "#999999"),
    "sell":         ("🔴",   "卖出观察", "#d96b6b"),
    "strong_sell":  ("🔴🔴", "强烈卖出", "#a01818"),
}


# 5 个子图的纵向布局（百分比基于 Grid 总高度 1100px）
# 每个子图前留 4% 给该子图的 legend；slider 放底部 4%
_LAYOUT = {
    "kline":  {"top": "5%",  "height": "25%", "legend": "1%"},   # K线占大头
    "volume": {"top": "34%", "height": "8%",  "legend": None},   # 成交量无 legend
    "macd":   {"top": "47%", "height": "12%", "legend": "44%"},
    "kdj":    {"top": "64%", "height": "12%", "legend": "61%"},
    "rsi":    {"top": "81%", "height": "12%", "legend": "78%"},
}


def _kline_main(code: str, name: str, df: pd.DataFrame) -> Kline:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    ohlc = df[["open", "close", "low", "high"]].values.tolist()

    kline = (
        Kline()
        .add_xaxis(dates)
        .add_yaxis(
            f"{code} {name}",
            ohlc,
            itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(is_scale=True,
                                     splitarea_opts=opts.SplitAreaOpts(is_show=True)),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1, 2, 3, 4]),
                opts.DataZoomOpts(type_="slider", xaxis_index=[0, 1, 2, 3, 4],
                                  pos_top="94%", pos_bottom="2%"),
            ],
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_top=_LAYOUT["kline"]["legend"]),
        )
    )
    return kline


def _ma_boll_overlay(df: pd.DataFrame) -> Line:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    line = Line().add_xaxis(dates)
    for col, label in [("ma5", "MA5"), ("ma10", "MA10"), ("ma20", "MA20"), ("ma60", "MA60"),
                       ("boll_up", "BOLL上"), ("boll_mid", "BOLL中"), ("boll_low", "BOLL下")]:
        if col in df.columns:
            vals = df[col].round(3).where(df[col].notna(), None).tolist()
            line.add_yaxis(label, vals,
                           is_smooth=True, is_symbol_show=False,
                           label_opts=opts.LabelOpts(is_show=False),
                           linestyle_opts=opts.LineStyleOpts(width=1))
    line.set_global_opts(legend_opts=opts.LegendOpts(pos_top=_LAYOUT["kline"]["legend"]))
    return line


def _volume_bar(df: pd.DataFrame) -> Bar:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    colors = ["#ec0000" if c >= o else "#00da3c"
              for c, o in zip(df["close"], df["open"])]
    bar = (
        Bar()
        .add_xaxis(dates)
        .add_yaxis("成交量", df["volume"].tolist(),
                   label_opts=opts.LabelOpts(is_show=False),
                   itemstyle_opts=opts.ItemStyleOpts(color="#999"))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(grid_index=1,
                                     axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(grid_index=1, is_scale=True),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    bar.options["series"][0]["data"] = [
        {"value": v, "itemStyle": {"color": c}}
        for v, c in zip(df["volume"].tolist(), colors)
    ]
    return bar


def _macd_chart(df: pd.DataFrame) -> Bar:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    hist = df["macd_hist"].round(4).fillna(0).tolist()

    bar = (
        Bar()
        .add_xaxis(dates)
        .add_yaxis("MACD", hist,
                   label_opts=opts.LabelOpts(is_show=False),
                   itemstyle_opts=opts.ItemStyleOpts(color="#999"))
    )
    bar.options["series"][0]["data"] = [
        {"value": v, "itemStyle": {"color": "#ec0000" if v >= 0 else "#00da3c"}}
        for v in hist
    ]
    bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(grid_index=2,
                                 axislabel_opts=opts.LabelOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(grid_index=2, is_scale=True),
        legend_opts=opts.LegendOpts(pos_top=_LAYOUT["macd"]["legend"]),
    )

    line = (
        Line()
        .add_xaxis(dates)
        .add_yaxis("DIF", df["macd_dif"].round(4).fillna(0).tolist(),
                   is_smooth=True, is_symbol_show=False,
                   label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("DEA", df["macd_dea"].round(4).fillna(0).tolist(),
                   is_smooth=True, is_symbol_show=False,
                   label_opts=opts.LabelOpts(is_show=False))
    )
    return bar.overlap(line)


def _kdj_chart(df: pd.DataFrame) -> Line:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    line = (
        Line()
        .add_xaxis(dates)
        .add_yaxis("K", df["kdj_k"].round(2).fillna(50).tolist(),
                   is_smooth=True, is_symbol_show=False,
                   label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("D", df["kdj_d"].round(2).fillna(50).tolist(),
                   is_smooth=True, is_symbol_show=False,
                   label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("J", df["kdj_j"].round(2).fillna(50).tolist(),
                   is_smooth=True, is_symbol_show=False,
                   label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(grid_index=3,
                                     axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(grid_index=3, is_scale=True),
            legend_opts=opts.LegendOpts(pos_top=_LAYOUT["kdj"]["legend"]),
        )
    )
    return line


def _rsi_chart(df: pd.DataFrame) -> Line:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    line = Line().add_xaxis(dates)
    for col, label in [("rsi6", "RSI6"), ("rsi12", "RSI12"), ("rsi24", "RSI24")]:
        if col in df.columns:
            line.add_yaxis(label, df[col].round(2).fillna(50).tolist(),
                           is_smooth=True, is_symbol_show=False,
                           label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(grid_index=4, is_scale=True,
                                 axislabel_opts=opts.LabelOpts(rotate=30, font_size=10)),
        yaxis_opts=opts.AxisOpts(grid_index=4, is_scale=True),
        legend_opts=opts.LegendOpts(pos_top=_LAYOUT["rsi"]["legend"]),
    )
    return line


def build_stock_chart(code: str, name: str, df: pd.DataFrame, klines_to_show: int) -> Grid:
    """5-row synchronized grid: K-line + volume + MACD + KDJ + RSI."""
    show = df.tail(klines_to_show).reset_index(drop=True)

    kline = _kline_main(code, name, show).overlap(_ma_boll_overlay(show))
    volume = _volume_bar(show)
    macd = _macd_chart(show)
    kdj = _kdj_chart(show)
    rsi = _rsi_chart(show)

    grid = (
        Grid(init_opts=opts.InitOpts(width="100%", height="1100px"))
        .add(kline,
             grid_opts=opts.GridOpts(pos_top=_LAYOUT["kline"]["top"],
                                     height=_LAYOUT["kline"]["height"]))
        .add(volume,
             grid_opts=opts.GridOpts(pos_top=_LAYOUT["volume"]["top"],
                                     height=_LAYOUT["volume"]["height"]))
        .add(macd,
             grid_opts=opts.GridOpts(pos_top=_LAYOUT["macd"]["top"],
                                     height=_LAYOUT["macd"]["height"]))
        .add(kdj,
             grid_opts=opts.GridOpts(pos_top=_LAYOUT["kdj"]["top"],
                                     height=_LAYOUT["kdj"]["height"]))
        .add(rsi,
             grid_opts=opts.GridOpts(pos_top=_LAYOUT["rsi"]["top"],
                                     height=_LAYOUT["rsi"]["height"]))
    )
    return grid


# ===== Full-page report =====

@dataclass
class ContextSignal:
    """Buy/sell signal for a market index or sector board."""
    label: str           # e.g. "上证指数", "化工板块"
    daily_score: int
    weekly_score: int
    final_score: float
    verdict: str
    triggers_daily: list[Trigger] = field(default_factory=list)


@dataclass
class StockAnalysis:
    code: str
    name: str
    daily_score: int
    weekly_score: int
    final_score: float
    verdict: str
    triggers_daily: list[Trigger] = field(default_factory=list)
    triggers_weekly: list[Trigger] = field(default_factory=list)
    hit_rates: dict[str, Any] = field(default_factory=dict)
    verdict_hit_rates: dict[str, Any] = field(default_factory=dict)
    daily_with_indicators: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)
    context: list[ContextSignal] = field(default_factory=list)
    strategy_name: str = "composite_verdict"


def _overview_row(a: StockAnalysis) -> str:
    emoji, label, color = _VERDICT_LABEL.get(a.verdict, ("⚪", "观望", "#999"))
    top_triggers = a.triggers_daily[:3] if a.triggers_daily else []
    trigger_text = " / ".join(t.description for t in top_triggers) if top_triggers else "—"
    return f"""
      <tr>
        <td><a href="#stock-{a.code}">{a.code}</a></td>
        <td>{a.name}</td>
        <td style="text-align:right">{a.daily_score:+d}</td>
        <td style="text-align:right">{a.weekly_score:+d}</td>
        <td style="text-align:right; font-weight:bold; color:{color}">{a.final_score:+.1f}</td>
        <td><span style="color:{color}">{emoji} {label}</span></td>
        <td style="color:#666; font-size:0.9em">{trigger_text}</td>
      </tr>
    """


def _triggers_section_html(a: "StockAnalysis") -> str:
    """Strategy-aware trigger section.

    composite_verdict → 日 K + 周 K 双列, 含分数加权公式。
    ml_factor → 单列 "主要因子贡献" (按 |z×w| 降序), 不显示加权公式。
    """
    if a.strategy_name == "ml_factor":
        return (
            "<div class='signal-cols'>"
            "<div>"
            f"<h4>主要因子贡献 (预测分 {a.final_score:+.3f})</h4>"
            f"<ul>{_trigger_list_html(a.triggers_daily)}</ul>"
            "</div></div>"
        )
    return (
        "<div class='signal-cols'>"
        "<div>"
        f"<h4>触发信号(日 K)— 日分 {a.daily_score:+d} × 0.7 = {a.daily_score * 0.7:+.2f}</h4>"
        f"<ul>{_trigger_list_html(a.triggers_daily)}</ul>"
        "</div>"
        "<div>"
        f"<h4>触发信号(周 K)— 周分 {a.weekly_score:+d} × 0.3 = {a.weekly_score * 0.3:+.2f}</h4>"
        f"<ul>{_trigger_list_html(a.triggers_weekly)}</ul>"
        "</div></div>"
    )


def _trigger_list_html(triggers: list[Trigger]) -> str:
    if not triggers:
        return "<li><em>无触发信号</em></li>"
    rows = []
    for t in triggers:
        sign = "+" if t.direction > 0 else "-"
        rows.append(
            f"<li>{t.description} <span style='color:#888'>"
            f"({sign}{abs(t.weight)})</span></li>"
        )
    return "\n".join(rows)


def _hit_rate_table(hit_rates: dict[str, Any]) -> str:
    if not hit_rates:
        return "<p style='color:#888'>本股历史窗口内无同类信号样本。</p>"
    rows = []
    for sig, data in hit_rates.items():
        cells = [f"<td>{sig}</td>", f"<td>{data['count']}</td>"]
        for n in (5, 10, 20):
            key = f"forward_{n}"
            if key in data:
                d = data[key]
                cells.append(
                    f"<td>{d['mean_return_pct']:+.2f}% / "
                    f"<span style='color:#666'>{d['win_rate']*100:.0f}%</span></td>"
                )
            else:
                cells.append("<td>—</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
      <table class="hit-rate">
        <thead><tr>
          <th>信号</th><th>次数</th>
          <th>5 日 均涨幅/胜率</th>
          <th>10 日</th>
          <th>20 日</th>
        </tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    """


def _verdict_bucket_table(stats: dict[str, Any]) -> str:
    if not stats:
        return "<p style='color:#888'>本股历史窗口内无综合评级样本。</p>"

    label_map = {
        "strong_buy":   ("🟢🟢", "强烈买入"),
        "buy":          ("🟢",   "买入"),
        "neutral":      ("⚪",   "中性"),
        "sell":         ("🔴",   "卖出"),
        "strong_sell":  ("🔴🔴", "强烈卖出"),
    }
    rows = []
    for key in ("strong_buy", "buy", "neutral", "sell", "strong_sell"):
        data = stats.get(key)
        if not data:
            continue
        emoji, label = label_map[key]
        cells = [
            f"<td>{emoji} {label}</td>",
            f"<td>{data['count']}</td>",
        ]
        for n in (5, 10, 20):
            d = data.get(f"forward_{n}")
            if d and d["sample_size"] > 0:
                cells.append(
                    f"<td>{d['mean_return_pct']:+.2f}%</td>"
                    f"<td><span style='color:#666'>{d['win_rate']*100:.0f}%</span></td>"
                )
            else:
                cells.append("<td>—</td><td>—</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
      <table class="hit-rate">
        <thead><tr>
          <th>评级</th><th>样本</th>
          <th>5 日均涨幅</th><th>5 日胜率</th>
          <th>10 日均涨幅</th><th>10 日胜率</th>
          <th>20 日均涨幅</th><th>20 日胜率</th>
        </tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    """


def _context_bar_html(context: list[ContextSignal]) -> str:
    if not context:
        return ""
    parts = []
    for c in context:
        emoji, label, color = _VERDICT_LABEL.get(c.verdict, ("⚪", "观望", "#999"))
        top = c.triggers_daily[0].description if c.triggers_daily else ""
        tip = f' title="{top}"' if top else ""
        parts.append(
            f"<span class='ctx-pill' style='border-color:{color}'{tip}>"
            f"<strong>{c.label}</strong>&nbsp;"
            f"<span style='color:{color}'>{emoji}&nbsp;{label}&nbsp;{c.final_score:+.1f}</span>"
            f"</span>"
        )
    return "<div class='context-bar'>" + "".join(parts) + "</div>"


def _stock_section_html(a: StockAnalysis, klines_to_show: int) -> str:
    emoji, label, color = _VERDICT_LABEL.get(a.verdict, ("⚪", "观望", "#999"))
    warnings_html = ""
    if a.warnings:
        warnings_html = "<div class='warning'>⚠️ " + " / ".join(a.warnings) + "</div>"

    chart_html = ""
    if a.daily_with_indicators is not None and len(a.daily_with_indicators) > 0:
        try:
            grid = build_stock_chart(a.code, a.name, a.daily_with_indicators, klines_to_show)
            chart_html = grid.render_embed()
        except Exception as e:
            chart_html = f"<p style='color:#a00'>图表生成失败: {e}</p>"

    context_bar = _context_bar_html(a.context)

    return f"""
    <details id="stock-{a.code}">
      <summary>
        <span style="font-size:1.3em; font-weight:bold">{a.code} {a.name}</span>
        <span style="color:{color}; margin-left:1em">{emoji} {label} 终分 {a.final_score:+.1f}</span>
      </summary>
      {context_bar}
      {warnings_html}
      <div class="chart-wrap">{chart_html}</div>

      {_triggers_section_html(a)}

      <h4>单信号历史命中率(过去 500 日)</h4>
      {_hit_rate_table(a.hit_rates)}

      <h4>综合评级历史回测(过去 500 日)</h4>
      {_verdict_bucket_table(a.verdict_hit_rates)}
    </details>
    """


_CSS = """
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 1400px;
         margin: 1em auto; padding: 0 1em; color: #222; }
  h1 { margin-bottom: 0.3em; }
  .meta { color: #666; margin-bottom: 1em; }
  table { border-collapse: collapse; width: 100%; margin: 0.5em 0 1.5em; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 0.95em; }
  th { background: #f6f6f6; text-align: left; }
  .overview tr:hover { background: #fafafa; }
  details { border-top: 2px solid #e6e6e6; padding: 1em 0; margin-top: 1em; }
  details summary { cursor: pointer; padding: 0.3em 0; }
  .chart-wrap { margin: 1em 0; }
  .signal-cols { display: flex; gap: 2em; margin: 1em 0; }
  .signal-cols > div { flex: 1; }
  .signal-cols ul { margin: 0.3em 0; padding-left: 1.3em; }
  .hit-rate { font-size: 0.9em; }
  .hit-rate th { background: #fafafa; }
  .warning { background: #fff4e5; padding: 0.5em 1em; border-left: 3px solid #f80;
             margin: 0.5em 0; font-size: 0.9em; }
  .context-bar { display: flex; flex-wrap: wrap; gap: 0.5em; margin: 0.6em 0 0.8em; }
  .ctx-pill { border: 1px solid #ddd; border-radius: 4px; padding: 0.25em 0.7em;
              font-size: 0.88em; cursor: default; background: #fafafa; }
  footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #eee;
           color: #888; font-size: 0.85em; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
"""


def _render_pool_b_section(pool_b: list["PoolBEntry"], run_date: str) -> str:
    """Pool B (全市场量化推荐池) 底部段。每周刷新,允许与 Pool A 重叠 (⭐)。"""
    from datetime import date as _date
    iso = _date.fromisoformat(run_date).isocalendar()
    rows = []
    for e in pool_b:
        emoji, label, color = _VERDICT_LABEL.get(e.verdict, ("⚪", "观望", "#999"))
        star = "⭐" if e.is_in_pool_a else ""
        rows.append(
            f"<tr>"
            f"<td style='text-align:right'>{e.rank}</td>"
            f"<td>{e.code}</td>"
            f"<td>{e.name} {star}</td>"
            f"<td>{e.industry}</td>"
            f"<td style='text-align:right; font-weight:bold; color:{color}'>"
            f"{e.final_score:+.3f}</td>"
            f"<td><span style='color:{color}'>{emoji} {label}</span></td>"
            f"</tr>"
        )
    return f"""
  <h2>Pool B · 全市场量化推荐池 (top {len(pool_b)})</h2>
  <p class="meta">
    本池每周刷新 (ISO {iso.year}-W{iso.week:02d}),按当前策略
    <code>predict_latest</code> 全市场打分,经流动性 + ST + 行业上限漏斗后取 top-N。
    标 ⭐ 的代码同时在自选 Pool A 中。
  </p>
  <table class="overview">
    <thead><tr>
      <th>排名</th><th>代码</th><th>名称</th><th>行业</th>
      <th>终分</th><th>判定</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
"""


_ECHARTS_LIB_RE = re.compile(
    r'\s*<script type="text/javascript" src="[^"]*echarts\.min\.js"></script>'
)
_ECHARTS_INIT_RE = re.compile(
    r'<script>(\s*var chart_[0-9a-f]+ = echarts\.init\b.*?)</script>',
    re.DOTALL,
)

_LAZY_LOADER_JS = """
<script>
(function () {
  function activate(root) {
    root.querySelectorAll('script[type="text/echarts-pending"]').forEach(function (s) {
      if (s.dataset.done) return;
      var ns = document.createElement('script');
      ns.text = s.textContent;
      s.parentNode.insertBefore(ns, s.nextSibling);
      s.dataset.done = '1';
    });
  }
  function openHashTarget() {
    if (!location.hash) return;
    var el = document.getElementById(location.hash.slice(1));
    if (el && el.tagName === 'DETAILS') el.open = true;
  }
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('details').forEach(function (d) {
      d.addEventListener('toggle', function () { if (d.open) activate(d); });
    });
    openHashTarget();
  });
  window.addEventListener('hashchange', openHashTarget);
})();
</script>
"""


def _optimize_html(html: str) -> str:
    """单次 echarts lib + 懒加载图表 init,改善大报告首屏速度。

    1. 删除每章重复的 ``<script src=".../echarts.min.js">``,在 ``<head>`` 放一份带
       ``defer`` 的引用。
    2. 把 ``<script>var chart_UUID = echarts.init(...)...</script>`` 改成
       ``type="text/echarts-pending"``,避免页面加载时立即执行。
    3. 注入小段 loader:``<details>`` 第一次展开(或被锚点定位)时再执行其内
       的延迟脚本。
    """
    lib_tags = _ECHARTS_LIB_RE.findall(html)
    if lib_tags:
        html = _ECHARTS_LIB_RE.sub('', html)
        lib_tag = lib_tags[0].strip().replace(
            '></script>', ' defer></script>', 1
        )
        html = html.replace('</head>', f'  {lib_tag}\n</head>', 1)

    html = _ECHARTS_INIT_RE.sub(
        r'<script type="text/echarts-pending">\1</script>',
        html,
    )

    return html.replace('</body>', _LAZY_LOADER_JS + '</body>', 1)


def _summary_counts(analyses: list[StockAnalysis]) -> str:
    counts = {k: 0 for k in _VERDICT_LABEL}
    for a in analyses:
        counts[a.verdict] = counts.get(a.verdict, 0) + 1
    parts = []
    for key in ["strong_buy", "buy", "neutral", "sell", "strong_sell"]:
        emoji, label, color = _VERDICT_LABEL[key]
        parts.append(f"<span style='color:{color}'>{emoji} {label} {counts[key]}</span>")
    return " &nbsp; ".join(parts)


def render_report(
    analyses: list[StockAnalysis],
    run_date: str,
    config_path: Path,
    config_hash: str,
    output_dir: str | Path,
    keep_history: bool,
    klines_to_show: int = 120,
    market_context: list[ContextSignal] | None = None,
    pool_b: list["PoolBEntry"] | None = None,
) -> Path:
    """Render full-page HTML report, return file path."""
    output_dir = Path(output_dir)
    day_dir = output_dir / run_date
    day_dir.mkdir(parents=True, exist_ok=True)
    out_path = day_dir / "index.html"

    analyses_sorted = sorted(analyses, key=lambda a: -a.final_score)

    overview_rows = "".join(_overview_row(a) for a in analyses_sorted)
    stock_sections = "".join(_stock_section_html(a, klines_to_show) for a in analyses_sorted)

    market_html = ""
    if market_context:
        market_html = f"<h2>市场环境</h2>{_context_bar_html(market_context)}"

    pool_b_html = _render_pool_b_section(pool_b, run_date) if pool_b else ""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>养龙股池每日信号 · {run_date}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>养龙股池每日信号 · {run_date}</h1>
  <p class="meta">
    扫描 {len(analyses)} 只 &nbsp; &nbsp; {_summary_counts(analyses_sorted)}
  </p>
  {market_html}

  <h2>总览(按终分降序)</h2>
  <table class="overview">
    <thead><tr>
      <th>代码</th><th>名称</th><th>日分</th><th>周分</th><th>终分</th>
      <th>判定</th><th>主要触发</th>
    </tr></thead>
    <tbody>{overview_rows}</tbody>
  </table>

  <h2>单股详情</h2>
  {stock_sections}

  {pool_b_html}

  <footer>
    <p>Config: <code>{config_path}</code> &nbsp; hash <code>{config_hash}</code></p>
    <p>⚠️ <strong>免责声明:</strong>本报告基于公开行情数据的技术指标计算,
       信号与打分仅供个人技术分析参考,<strong>不构成任何投资建议</strong>。
       使用者应自行承担交易决策的全部责任。</p>
  </footer>
</body>
</html>
"""
    html = _optimize_html(html)
    out_path.write_text(html, encoding="utf-8")

    latest = output_dir / "latest.html"
    shutil.copyfile(out_path, latest)

    return out_path
