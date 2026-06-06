"""HTML 因子选择器 + CLI 子命令的服务端。

工作流:
  1. ``python -m stockpool factors pick`` 默认起一个本地 HTTP 服务 (127.0.0.1) 并
     用浏览器打开。
  2. 用户按 "来源" 和 "类型" 双轴筛选,卡片上勾选因子。
  3. 点 **"应用"** → 直接 POST 到 ``/save``,服务端把选择写到
     ``reports/selection.json``(或 ``--output`` 指定的路径)。无需手动下载/移动。
  4. 也支持 "下载 selection.json" 或 "复制 YAML" 作为离线兜底。
  5. Ctrl-C 退出服务。``--static`` 切回老的"生成静态 HTML 文件"模式
     (适合需要把 HTML 单独保存归档时)。

页面状态用 localStorage 持久化(刷新不丢);打开页面时也会从服务端 ``/selection.json``
读取已有的选择并合并。
"""
from __future__ import annotations

import html
import http.server
import inspect
import json
import textwrap
import threading
import webbrowser
from pathlib import Path

from stockpool.factors import all_sources, all_types, list_specs


def _extract_formula(cls: type) -> str:
    """提取 Factor 子类的 ``compute`` 方法源码作为"公式"展示。

    源码即公式 — 对于公式因子这是最直接的呈现方式。失败时返回空串。
    """
    try:
        src = inspect.getsource(cls.compute)
    except (OSError, TypeError):
        return ""
    return textwrap.dedent(src).rstrip()


def _factor_payload() -> list[dict]:
    """把注册表序列化为 HTML 需要的纯字典列表。"""
    out: list[dict] = []
    for spec in list_specs():
        out.append({
            "name": spec.default_name,
            "base": spec.base_name,
            "sources": list(spec.sources),
            "types": list(spec.types),
            "description": spec.description,
            "formula": _extract_formula(spec.cls),
        })
    return out


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>因子选择器 — stockpool</title>
<style>
  body { font-family: -apple-system, "Segoe UI", "PingFang SC", sans-serif;
         margin: 0; background: #f6f7f9; color: #1f2328; }
  header { background: #1f2328; color: #fff; padding: 14px 22px;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  header .meta { font-size: 13px; color: #c8ccd2; }
  header .actions { margin-left: auto; display: flex; gap: 8px; }
  button { padding: 6px 14px; font-size: 13px; cursor: pointer;
           border: 1px solid #c8ccd2; background: #fff; border-radius: 4px; }
  button.primary { background: #0969da; color: #fff; border-color: #0969da; }
  button:hover { opacity: .9; }
  main { display: grid; grid-template-columns: 260px 1fr; gap: 16px;
         padding: 16px; max-width: 1400px; margin: 0 auto; }
  aside { background: #fff; padding: 12px 16px; border-radius: 6px;
          border: 1px solid #e1e4e8; height: fit-content; position: sticky; top: 16px; }
  aside h3 { font-size: 13px; text-transform: uppercase; color: #57606a;
             margin: 16px 0 8px; }
  aside h3:first-child { margin-top: 0; }
  aside label { display: flex; align-items: center; gap: 6px; padding: 4px 0;
                font-size: 14px; cursor: pointer; }
  aside .count { color: #57606a; font-size: 12px; margin-left: auto; }
  #factors { display: grid;
             grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
             gap: 10px; align-content: start; }
  .card { background: #fff; border: 1px solid #e1e4e8; border-radius: 6px;
          padding: 12px 14px; transition: border-color .12s; }
  .card.selected { border-color: #0969da; background: #f6faff; }
  .card .top { display: flex; align-items: flex-start; gap: 8px; }
  .card .name { font-family: ui-monospace, Menlo, Consolas, monospace;
                font-weight: 600; font-size: 14px; }
  .card .tabs { display: flex; gap: 0; margin: 8px 0 0;
                border-bottom: 1px solid #e1e4e8; }
  .card .tab-btn { background: transparent; border: none; padding: 4px 10px;
                   font-size: 12px; color: #57606a; cursor: pointer;
                   border-bottom: 2px solid transparent; margin-bottom: -1px;
                   border-radius: 0; }
  .card .tab-btn.active { color: #0969da; border-bottom-color: #0969da;
                          font-weight: 600; }
  .card .tab-btn:hover { background: #f6f8fa; }
  .card .tab-panel { display: none; padding-top: 8px; }
  .card .tab-panel.active { display: block; }
  .card .desc { color: #57606a; font-size: 12.5px; margin: 0 0 8px;
                line-height: 1.45; }
  .card pre.formula { background: #f6f8fa; color: #1f2328; padding: 8px 10px;
                      border-radius: 4px; font-size: 11.5px; line-height: 1.4;
                      margin: 0 0 8px; overflow-x: auto;
                      font-family: ui-monospace, Menlo, Consolas, monospace;
                      max-height: 220px; }
  .card pre.formula.empty { color: #8b949e; font-style: italic; padding: 8px;
                            font-family: inherit; background: transparent; }
  .tag { display: inline-block; padding: 1px 7px; font-size: 11px;
         border-radius: 10px; background: #eef2f6; color: #1f2328;
         margin-right: 4px; }
  .tag.src { background: #ddf4ff; color: #0550ae; }
  .tag.type { background: #fff8c5; color: #6f5a00; }
  .empty { padding: 40px; text-align: center; color: #57606a;
           grid-column: 1 / -1; }
  .status { margin-left: 12px; font-size: 13px; color: #57606a; }
  details summary { cursor: pointer; padding: 6px 0; font-size: 13px;
                    color: #0969da; }
  pre.yaml-out { background: #161b22; color: #e6edf3; padding: 12px;
                 border-radius: 6px; font-size: 12px; overflow-x: auto;
                 white-space: pre-wrap; word-break: break-all; }
</style>
</head>
<body>
<header>
  <h1>因子选择器</h1>
  <span class="meta">共 __TOTAL__ 个因子,选中 <b id="selCount">0</b></span>
  <span class="status" id="status"></span>
  <div class="actions">
    <button onclick="selectVisible()">勾选当前可见</button>
    <button onclick="clearAll()">清空</button>
    <button class="primary" id="applyBtn" onclick="applySelection()" title="保存到服务端 selection.json (server 模式)">应用</button>
    <button onclick="downloadJson()">下载 selection.json</button>
    <button onclick="copyYaml()">复制 YAML</button>
  </div>
</header>
<main>
  <aside>
    <h3>来源 (sources)</h3>
    <div id="filterSources"></div>
    <h3>类型 (types)</h3>
    <div id="filterTypes"></div>
    <h3>筛选模式</h3>
    <label><input type="radio" name="matchMode" value="any" checked> 任一标签匹配</label>
    <label><input type="radio" name="matchMode" value="all"> 必须全部标签</label>
    <h3>导出 YAML 预览</h3>
    <details>
      <summary>展开/收起</summary>
      <pre class="yaml-out" id="yamlOut"></pre>
    </details>
  </aside>
  <section id="factors"></section>
</main>

<script>
const FACTORS = __FACTORS_JSON__;
const ALL_SOURCES = __SOURCES__;
const ALL_TYPES = __TYPES__;
const STORAGE_KEY = "stockpool_factor_selection_v1";

let selected = new Set();
let activeSources = new Set();
let activeTypes = new Set();

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (Array.isArray(saved.selected)) selected = new Set(saved.selected);
    if (Array.isArray(saved.sources)) activeSources = new Set(saved.sources);
    if (Array.isArray(saved.types)) activeTypes = new Set(saved.types);
  } catch (e) { /* ignore */ }
}
function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    selected: [...selected],
    sources: [...activeSources],
    types: [...activeTypes],
  }));
}

function getMatchMode() {
  const r = document.querySelector("input[name=matchMode]:checked");
  return r ? r.value : "any";
}

function passesFilter(f) {
  const mode = getMatchMode();
  if (activeSources.size > 0) {
    const set = new Set(f.sources);
    const ok = mode === "all"
      ? [...activeSources].every(s => set.has(s))
      : [...activeSources].some(s => set.has(s));
    if (!ok) return false;
  }
  if (activeTypes.size > 0) {
    const set = new Set(f.types);
    const ok = mode === "all"
      ? [...activeTypes].every(t => set.has(t))
      : [...activeTypes].some(t => set.has(t));
    if (!ok) return false;
  }
  return true;
}

function visibleFactors() { return FACTORS.filter(passesFilter); }

function renderFilters() {
  const fs = document.getElementById("filterSources");
  fs.innerHTML = ALL_SOURCES.map(s => {
    const count = FACTORS.filter(f => f.sources.includes(s)).length;
    const checked = activeSources.has(s) ? "checked" : "";
    return `<label><input type="checkbox" data-src="${s}" ${checked}>${s}<span class="count">${count}</span></label>`;
  }).join("");
  fs.querySelectorAll("input").forEach(el => el.onchange = e => {
    if (e.target.checked) activeSources.add(e.target.dataset.src);
    else activeSources.delete(e.target.dataset.src);
    saveState(); renderFactors();
  });

  const ft = document.getElementById("filterTypes");
  ft.innerHTML = ALL_TYPES.map(t => {
    const count = FACTORS.filter(f => f.types.includes(t)).length;
    const checked = activeTypes.has(t) ? "checked" : "";
    return `<label><input type="checkbox" data-typ="${t}" ${checked}>${t}<span class="count">${count}</span></label>`;
  }).join("");
  ft.querySelectorAll("input").forEach(el => el.onchange = e => {
    if (e.target.checked) activeTypes.add(e.target.dataset.typ);
    else activeTypes.delete(e.target.dataset.typ);
    saveState(); renderFactors();
  });

  document.querySelectorAll("input[name=matchMode]").forEach(el =>
    el.onchange = () => { saveState(); renderFactors(); });
}

function renderFactors() {
  const vis = visibleFactors();
  const container = document.getElementById("factors");
  if (vis.length === 0) {
    container.innerHTML = '<div class="empty">没有匹配的因子,放宽筛选试试</div>';
  } else {
    container.innerHTML = vis.map(f => {
      const sel = selected.has(f.name) ? "selected" : "";
      const nameEsc = escapeHtml(f.name);
      const descEsc = f.description ? escapeHtml(f.description) : "(无描述)";
      const srcTags = f.sources.map(s => `<span class="tag src">${escapeHtml(s)}</span>`).join("");
      const typTags = f.types.map(t => `<span class="tag type">${escapeHtml(t)}</span>`).join("");
      const formulaBody = f.formula
        ? `<pre class="formula">${escapeHtml(f.formula)}</pre>`
        : `<pre class="formula empty">(未提取到公式)</pre>`;
      return `<div class="card ${sel}" data-name="${nameEsc}">
        <div class="top">
          <input type="checkbox" ${selected.has(f.name)?"checked":""} data-name="${nameEsc}">
          <span class="name">${nameEsc}</span>
        </div>
        <div class="tabs">
          <button type="button" class="tab-btn active" data-tab="intro">简介</button>
          <button type="button" class="tab-btn" data-tab="formula">公式</button>
        </div>
        <div class="tab-panel active" data-panel="intro">
          <div class="desc">${descEsc}</div>
        </div>
        <div class="tab-panel" data-panel="formula">${formulaBody}</div>
        <div>${srcTags}${typTags}</div>
      </div>`;
    }).join("");
    container.querySelectorAll("input[type=checkbox]").forEach(cb => {
      cb.onchange = e => {
        const nm = e.target.dataset.name;
        if (e.target.checked) selected.add(nm); else selected.delete(nm);
        saveState(); updateCount(); renderFactors();
      };
    });
    container.querySelectorAll(".tab-btn").forEach(btn => {
      btn.onclick = e => {
        const card = e.target.closest(".card");
        const tab = e.target.dataset.tab;
        card.querySelectorAll(".tab-btn").forEach(b =>
          b.classList.toggle("active", b.dataset.tab === tab));
        card.querySelectorAll(".tab-panel").forEach(p =>
          p.classList.toggle("active", p.dataset.panel === tab));
      };
    });
  }
  updateCount();
  updateYamlPreview();
}

function updateCount() {
  document.getElementById("selCount").textContent = selected.size;
}

function selectVisible() {
  visibleFactors().forEach(f => selected.add(f.name));
  saveState(); renderFactors();
}

function clearAll() {
  if (!confirm(`确定清空 ${selected.size} 个选择?`)) return;
  selected = new Set();
  saveState(); renderFactors();
}

function selectionPayload() {
  return { factors: [...selected].sort(), generated_at: new Date().toISOString() };
}

function selectionYaml() {
  const list = [...selected].sort();
  if (list.length === 0) return "strategy:\\n  ml_factor:\\n    factors: []";
  return "strategy:\\n  ml_factor:\\n    factors:\\n" +
         list.map(n => `      - ${n}`).join("\\n");
}

function updateYamlPreview() {
  document.getElementById("yamlOut").textContent = selectionYaml();
}

function downloadJson() {
  if (selected.size === 0) { alert("还没选任何因子"); return; }
  const blob = new Blob([JSON.stringify(selectionPayload(), null, 2)],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "selection.json";
  a.click();
  setStatus(`已下载 selection.json (${selected.size} 个因子)`);
}

// 通过服务端 POST /save 把当前选择写到 selection.json。
// 仅在 "server 模式" (factors pick 默认) 下可用;静态文件模式下会失败,
// 自动回退到下载。
async function applySelection() {
  if (selected.size === 0) { alert("还没选任何因子"); return; }
  try {
    const r = await fetch("/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(selectionPayload()),
    });
    const data = await r.json();
    if (data.ok) {
      setStatus(`✓ 已应用 ${selected.size} 个因子 → ${data.path}`);
    } else {
      setStatus(`保存失败: ${data.error}`);
    }
  } catch (e) {
    setStatus(`无服务端连接 (静态文件模式?),已自动改为下载`);
    downloadJson();
  }
}

// 首次加载时,从服务端读已有 selection.json,并入 localStorage 状态。
// 服务端没起 / 文件不存在时静默忽略,只用 localStorage。
async function syncFromServer() {
  try {
    const r = await fetch("/selection.json", {cache: "no-store"});
    if (!r.ok) return;
    const data = await r.json();
    if (Array.isArray(data.factors) && data.factors.length > 0) {
      // 服务端文件优先(选择记录的权威源),覆盖 localStorage
      selected = new Set(data.factors);
      saveState();
      renderFactors();
      setStatus(`已从 selection.json 载入 ${selected.size} 个因子`);
    }
  } catch (e) { /* 静态模式,忽略 */ }
}

function copyYaml() {
  navigator.clipboard.writeText(selectionYaml()).then(
    () => setStatus("YAML 已复制到剪贴板"),
    () => setStatus("复制失败,请手动展开预览复制"),
  );
}

function setStatus(msg) {
  const el = document.getElementById("status");
  el.textContent = msg;
  setTimeout(() => { el.textContent = ""; }, 4000);
}

loadState();
renderFilters();
renderFactors();
syncFromServer();
</script>
</body>
</html>
"""


def render_picker_html() -> str:
    """渲染静态 HTML 字符串。"""
    factors = _factor_payload()
    return (
        _HTML_TEMPLATE
        .replace("__FACTORS_JSON__", json.dumps(factors, ensure_ascii=False))
        .replace("__SOURCES__", json.dumps(all_sources()))
        .replace("__TYPES__", json.dumps(all_types()))
        .replace("__TOTAL__", str(len(factors)))
    )


def write_and_open_picker(out_path: Path, open_browser: bool = True) -> Path:
    """[Static 模式] 生成 HTML 选择器文件并(可选)用浏览器打开。返回写入路径。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_picker_html(), encoding="utf-8")
    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


def _make_handler(html_body: str, selection_path: Path):
    """构造一个 BaseHTTPRequestHandler 子类,闭包绑定 HTML 内容和落盘路径。

    路由:
      GET  /                 → HTML 页面
      GET  /selection.json   → 当前 selection.json 内容(用于页面加载时同步)
      POST /save             → 写入 selection.json,响应 {"ok": true, "path": "..."}
    """
    class PickerHandler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = html_body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/selection.json"):
                if selection_path.exists():
                    try:
                        data = json.loads(selection_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        return self._send_json(500, {"ok": False, "error": str(e)})
                    return self._send_json(200, data)
                return self._send_json(200, {"factors": []})
            self.send_response(404)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.path != "/save":
                self.send_response(404)
                self.end_headers()
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
                if not isinstance(data.get("factors"), list):
                    raise ValueError("payload must contain a 'factors' list")
                selection_path.parent.mkdir(parents=True, exist_ok=True)
                selection_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._send_json(200, {"ok": True, "path": str(selection_path.resolve())})
            except Exception as e:
                self._send_json(400, {"ok": False, "error": str(e)})

        def log_message(self, format, *args):  # 静音默认 access log
            return
    return PickerHandler


def serve_picker(
    selection_path: Path,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    """起一个本地 HTTP 服务器伺服 HTML 选择器,阻塞到 Ctrl-C。

    Args:
        selection_path: ``应用`` 按下时写入的 JSON 路径。
        port: 监听端口;``0`` 表示让系统自动分配。
        open_browser: 是否自动用默认浏览器打开。
    """
    html_body = render_picker_html()
    handler_cls = _make_handler(html_body, selection_path)
    server = http.server.HTTPServer(("127.0.0.1", port), handler_cls)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"因子选择器: {url}")
    print(f"应用按钮 → 写入: {selection_path.resolve()}")
    print("Ctrl-C 停止。")
    if open_browser:
        # 用线程打开,避免某些环境下 webbrowser.open 阻塞
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


def cli_list(args) -> int:
    """``stockpool factors list`` —— 文本列出所有因子。"""
    specs = list_specs()
    if args.source:
        specs = [s for s in specs if args.source in s.sources]
    if args.type:
        specs = [s for s in specs if args.type in s.types]
    print(f"# {len(specs)} factor(s)")
    for spec in specs:
        srcs = ",".join(spec.sources)
        typs = ",".join(spec.types) or "-"
        print(f"  {spec.default_name:<22}  [{srcs}]  ({typs})  {spec.description}")
    return 0


def cli_show(args) -> int:
    """``stockpool factors show NAME`` —— 展示单个因子元数据。"""
    from stockpool.factors.registry import get_spec
    try:
        spec = get_spec(args.name)
    except KeyError:
        # 不是 base name 时,尝试 make_factor(可能带后缀),取它的元数据
        from stockpool.factors import make_factor
        try:
            inst = make_factor(args.name)
        except KeyError as e:
            print(f"ERROR: {e}", flush=True)
            return 1
        print(f"name:        {inst.name}")
        print(f"sources:     {list(inst.sources)}")
        print(f"types:       {list(inst.types)}")
        print(f"description: {inst.description}")
        return 0
    print(f"base_name:   {spec.base_name}")
    print(f"default:     {spec.default_name}")
    print(f"sources:     {list(spec.sources)}")
    print(f"types:       {list(spec.types)}")
    print(f"description: {spec.description}")
    print(f"class:       {spec.cls.__module__}.{spec.cls.__name__}")
    return 0


def cli_pick(args) -> int:
    """``stockpool factors pick`` —— 起本地服务伺服选择器,默认行为。

    模式:
      * 默认 (server):本地 HTTP 服务,"应用" 按钮直写 ``selection.json``。
      * ``--static``:回退到老的"写静态 HTML 文件"模式;此时 "应用" 按钮会
        自动降级为"下载"。
    """
    selection_path = Path(args.output) if args.output else Path("reports") / "selection.json"
    if args.static:
        html_path = (
            Path(args.html_output)
            if getattr(args, "html_output", None)
            else Path("reports") / "factors_picker.html"
        )
        write_and_open_picker(html_path, open_browser=not args.no_open)
        print(f"已生成静态 HTML: {html_path.resolve()}")
        print(f"勾选完成后点 '下载 selection.json',保存到: {selection_path}")
        print("然后在 config.yaml 设置:")
        print("  strategy:")
        print("    ml_factor:")
        print(f"      factors_file: {selection_path}")
        return 0

    print("启动因子选择器服务 (Ctrl-C 退出)...")
    print(f"在 config.yaml 设置:")
    print(f"  strategy:")
    print(f"    ml_factor:")
    print(f"      factors_file: {selection_path}")
    serve_picker(selection_path, port=args.port, open_browser=not args.no_open)
    return 0
