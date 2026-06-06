"""Static HTML renderer for the AB candidate pool.

Outputs a single HTML file with inline JSON data + vanilla-JS client-side
filtering (industry select, code prefix, name substring). No HTTP server,
no jinja, no framework — matches `factors_picker._render_html` style.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>AB Candidate Pool</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 12px; }}
  .filters {{ position: sticky; top: 0; background: #fff; padding: 8px 0;
              border-bottom: 1px solid #ddd; display: flex; gap: 12px; align-items: center; }}
  .filters label {{ font-size: 13px; }}
  .filters input, .filters select {{ padding: 4px 6px; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2),
  th:nth-child(3), td:nth-child(3), th:last-child, td:last-child {{ text-align: left; }}
  th {{ background: #f5f5f5; cursor: pointer; user-select: none; }}
  th:hover {{ background: #e8e8e8; }}
  tfoot {{ font-size: 12px; color: #666; }}
</style>
</head>
<body>
<h1>AB Candidate Pool — built {build_date}</h1>
<div class="filters">
  <label>行业 <select id="filter-industry"><option value="">全部</option>{industry_options}</select></label>
  <label>代码 <input id="filter-code" type="text" placeholder="6字头..."></label>
  <label>名称 <input id="filter-name" type="text" placeholder="子串..."></label>
</div>
<table>
  <thead><tr>
    <th data-col="code">代码</th>
    <th data-col="name">名称</th>
    <th data-col="industry">行业</th>
    <th data-col="circ_mv">流通市值(亿)</th>
    <th data-col="avg_amount_20d">20日均额(亿)</th>
    <th data-col="source_tag">source_tag</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table>
<tfoot id="footer">显示 <span id="shown">0</span> / 共 <span id="total">0</span> 票 | build_date: {build_date}</tfoot>
<script>
const POOL_DATA = {pool_json};
let sortCol = "circ_mv";
let sortDesc = true;

function fmtY(v) {{ return (v / 1e8).toFixed(2); }}
function applyFilters() {{
  const ind = document.getElementById("filter-industry").value;
  const code = document.getElementById("filter-code").value.trim();
  const name = document.getElementById("filter-name").value.trim();
  let rows = POOL_DATA.filter(r =>
    (!ind || r.industry === ind) &&
    (!code || r.code.startsWith(code)) &&
    (!name || r.name.indexOf(name) >= 0)
  );
  rows.sort((a, b) => {{
    const va = a[sortCol], vb = b[sortCol];
    if (typeof va === "number" && typeof vb === "number") return sortDesc ? vb - va : va - vb;
    return sortDesc ? String(vb).localeCompare(String(va)) : String(va).localeCompare(String(vb));
  }});
  const tbody = document.getElementById("rows");
  tbody.innerHTML = rows.map(r =>
    `<tr><td>${{r.code}}</td><td>${{r.name}}</td><td>${{r.industry}}</td>` +
    `<td>${{fmtY(r.circ_mv)}}</td><td>${{fmtY(r.avg_amount_20d)}}</td>` +
    `<td>${{r.source_tag}}</td></tr>`
  ).join("");
  document.getElementById("shown").textContent = rows.length;
  document.getElementById("total").textContent = POOL_DATA.length;
}}
document.querySelectorAll("th[data-col]").forEach(th => {{
  th.addEventListener("click", () => {{
    const c = th.dataset.col;
    if (sortCol === c) sortDesc = !sortDesc; else {{ sortCol = c; sortDesc = true; }}
    applyFilters();
  }});
}});
["filter-industry", "filter-code", "filter-name"].forEach(id => {{
  document.getElementById(id).addEventListener("input", applyFilters);
  document.getElementById(id).addEventListener("change", applyFilters);
}});
applyFilters();
</script>
</body>
</html>
"""


def render_ab_pool_html(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Render the AB candidate pool to a static HTML page.

    Embeds rows as inline JSON. Client-side filter via vanilla JS. No HTTP server.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    industries = sorted(df["industry"].unique()) if not df.empty else []
    industry_options = "".join(f'<option value="{i}">{i}</option>' for i in industries)
    build_date = str(df["build_date"].iloc[0]) if not df.empty else ""

    records = df.assign(
        circ_mv=df.get("circ_mv", pd.Series(dtype=float)).astype(float),
        avg_amount_20d=df.get("avg_amount_20d", pd.Series(dtype=float)).astype(float),
    ).to_dict(orient="records")
    # build_date column not needed in JSON (already in footer)
    for r in records:
        r.pop("build_date", None)
    pool_json = json.dumps(records, ensure_ascii=False)

    html = _TEMPLATE.format(
        build_date=build_date,
        industry_options=industry_options,
        pool_json=pool_json,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
