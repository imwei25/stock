# AB Candidate Pool — Design

**Status:** spec / awaiting review
**Date:** 2026-06-06
**Branch (intended):** `feat/ab-candidate-pool`

## Problem

A/B testing today has only two universe options, and both are awkward:

- **`cfg.stocks`** (8–30 票) — statistical power too low. Per-stock AB
  aggregates (sharpe mean / win count) wobble enough at this sample size that
  small genuine improvements can flip sign run-to-run. Portfolio AB on this
  pool is even worse because the portfolio engine has almost no eligible
  universe to pick from.
- **`universe.parquet`** (~4350 票, via `fetch-universe`) — too slow and too
  noisy. Per-stock AB on 4350 takes hours, and a meaningful fraction of those
  stocks are illiquid / new / ST junk that dominates dispersion without
  carrying signal. Portfolio AB on the full market works, but score panel
  precomputation is slow and the cross-section is dominated by stocks the
  strategy will never realistically trade.

What's missing is a **middle-ground, stable, reproducible candidate pool**
sized for "enough samples for meaningful aggregates, small enough to iterate"
— ~100 stocks, industry-balanced, refreshed only when the user explicitly
asks.

## Goals (MVP)

- A new `python -m stockpool ab-pool build` command that constructs a
  stratified candidate pool: ~28 SW level-1 industries × (top-2 by 流通市值 +
  top-2 by 20 日均额, **no dedup**) ≈ 100 stocks, with hard pre-filters
  (ST / 上市不满 252 日 / 流动性 < 5e7) applied before stratification.
- The pool is persisted to `data/ab_pool.parquet` and **never auto-refreshes**
  — `build` is idempotent-guarded (refuses to overwrite without `--refresh`),
  so historical AB runs remain reproducible until the user explicitly rebuilds.
- A new `python -m stockpool ab-pool show` command renders an HTML page
  (`reports/ab_pool.html`) with industry / code / name filtering for
  human inspection of the pool composition.
- Both `ab.yaml` and `portfolio_ab.yaml` gain a top-level `use_ab_pool: bool`
  flag (default false). When true:
  - **per-stock AB**: ab_pool codes replace `cfg.stocks` iteration; existing
    `stocks_filter` still applies (intersects with ab_pool).
  - **portfolio AB**: ab_pool codes are injected into each arm's
    `portfolio_backtest.universe_codes` (per-arm explicit `universe_codes`
    still wins). Training pool is **not** touched.
- Akshare snapshot (`ak.stock_zh_a_spot_em()`) is the data source for 流通市值
  at build time — one-shot call, no caching of the snapshot itself, snapshot
  values are baked into the parquet.

## Non-Goals (explicit follow-ups)

- **Auto-refresh on a schedule.** The user explicitly wants the pool static
  unless rebuilt by hand — reproducibility of historical AB runs is the
  hard constraint. No cron, no "pool is stale" warnings, no implicit rebuild.
- **Multiple named pools** (e.g., `ab_pool_smallcap`, `ab_pool_growth`).
  One canonical `data/ab_pool.parquet` only. Multi-pool support is a clean
  follow-up if/when it's needed: extend `cache_path` to a dict, add a
  `pool_name: str` selector to ab.yaml. Not in this spec.
- **`stats` subcommand** (industry distribution chart, mcap quantiles, etc.).
  Dropped during brainstorming — `show` HTML already exposes enough for human
  inspection. Can be revisited if a use case emerges.
- **Per-stock or per-industry custom weighting in the stratification.**
  Selection is strictly "top-2 by circ_mv + top-2 by avg_amount_20d per
  industry, allow overlap" — no industry-size weights, no minimum-stocks-per-industry
  flooring beyond skipping empty buckets. Adds complexity without clear win.
- **AB pool driving the training universe.** Decoupling training pool from
  application pool is a deliberate architectural decision aligned with
  existing `portfolio_backtest.universe_codes` semantics. Cross-sec factors
  and IC weights need the full market cross-section to compute stably; the
  AB pool is "what arms get compared on", not "what the model learns from".
- **Diff between two ab_pool builds** (e.g., `ab-pool diff old.parquet new.parquet`).
  Rare workflow; can git-diff the parquet's code column manually if needed.
- **`stocks_filter` that adds codes not in ab_pool.** Filter remains
  subtractive only (matches existing `ab.yaml` semantics).
- **Mootdx / akshare 板块行业 path.** Industry stratification uses SW
  level-1 via existing `load_or_build_industry_map(source="auto")` chain
  (baostock first, akshare fallback). Mootdx industry path is documented as
  broken (`block_hy.dat` returns 0 bytes) and isn't an option here.

## Architecture

### New module layout

```
src/stockpool/ab_pool.py             # NEW
  AbPoolConfig (pydantic)            # build params, owned by AppConfig.ab_pool
  build_ab_pool(cfg, refresh) → Path # main build entry point
  load_ab_pool(cache_path) → DataFrame
  _fetch_circ_mv_snapshot() → DataFrame   # akshare wrapper, mockable
  _compute_avg_amount_20d(universe, cache_dir) → DataFrame
  _apply_hard_filters(df, cfg) → DataFrame
  _stratified_select(df, cfg) → DataFrame  # per-industry 2+2 with overlap

src/stockpool/ab_pool_report.py      # NEW
  render_ab_pool_html(df, output_path) → Path
  # Embeds the parquet rows as inline JSON; client-side filter via vanilla JS.
  # No HTTP server. Mirrors `factors_picker._render_html` static-mode style.

src/stockpool/cli.py                 # MODIFIED
  + cmd_ab_pool_build(args)
  + cmd_ab_pool_show(args)
  # ab-pool top-level subcommand with build/show sub-subcommands

src/stockpool/config.py              # MODIFIED
  + AbPoolConfig
  AppConfig.ab_pool: AbPoolConfig = Field(default_factory=AbPoolConfig)

src/stockpool/ab/config.py           # MODIFIED
  ABConfig.use_ab_pool: bool = False
  load_ab_config now reads parquet (if use_ab_pool) and synthesizes a
  StockEntry list, applied AFTER stocks_filter intersection.

src/stockpool/portfolio_ab/config.py # MODIFIED
  PortfolioABConfig.use_ab_pool: bool = False
  build_effective_cfg injects ab_pool codes into
  portfolio_backtest.universe_codes (per-arm explicit override still wins).
```

### Data flow — `ab-pool build`

```
universe.parquet               akshare snapshot           industry_map cache
(code/name/market)              (code → circ_mv,           (code → SW-1)
                                 total_mv, name)
       \                              |                          /
        \_____________________________|__________________________/
                                      |
                                 inner join
                                      |
                          per-stock 20d avg_amount
                          from <code>_daily.parquet
                                      |
                              hard filters:
                       - ST (name match)
                       - ipo_dates < 252 days
                       - avg_amount_20d < 5e7
                       - circ_mv NaN
                                      |
                          groupby(industry) :
                            top-2 by circ_mv  → tag mcap
                            top-2 by avg_amt  → tag liq
                            union, merge tags
                                      |
                              ab_pool.parquet
```

### Parquet schema (`data/ab_pool.parquet`)

| column | type | source |
|---|---|---|
| `code` | str | universe.parquet |
| `name` | str | akshare snapshot (authoritative, matches build-time naming) |
| `industry` | str | industry_map; falls back to `"未知"` |
| `circ_mv` | float64 | akshare snapshot (yuan) |
| `avg_amount_20d` | float64 | computed from per-stock parquet cache |
| `source_tag` | str | `"mcap"` / `"liq"` / `"mcap+liq"` |
| `build_date` | date | constant per file, ISO date of build invocation |

Row count: typically ~100. Per industry: top-2 mcap ∪ top-2 liq with
**row-level merge** (overlap → single row, `source_tag="mcap+liq"`) yields 3
or 4 rows depending on overlap (~3.5 avg). With ~28 SW-1 industries → ~98
rows; "未知" bucket contributes up to 4 more if `include_unknown_industry=true`.

### `AbPoolConfig` schema

```python
class AbPoolConfig(BaseModel):
    cache_path: Path = Path("data/ab_pool.parquet")
    industry_source: Literal["auto", "baostock", "akshare"] = "auto"
    min_listing_days: int = 252
    min_avg_amount_20d: float = 5.0e7
    per_industry_top_mcap: int = 2
    per_industry_top_liq: int = 2
    exclude_st: bool = True
    include_unknown_industry: bool = True

    model_config = ConfigDict(extra="forbid")
```

Section is fully optional in `config.yaml`; defaults reproduce the
brainstormed recipe exactly. Pydantic `extra=forbid` keeps typos loud.

### CLI surface

```
python -m stockpool ab-pool build [--config config.yaml] [--refresh]
python -m stockpool ab-pool show  [--config config.yaml]
```

- `build` — see Data flow above. Exit codes:
  - **0** success
  - **1** preflight failure (universe.parquet missing OR cache_path exists
    without `--refresh`)
  - **2** data source failure (akshare snapshot raises, industry_map load
    fails, all industry buckets empty)
- `show` — loads parquet, calls `render_ab_pool_html` → `reports/ab_pool.html`
  + `reports/ab_pool_latest.html` (latest pointer mirrors other report
  conventions). Calls `webbrowser.open()` on the output path. Exit 1 if
  parquet missing.

`show` is a thin read-only renderer — no flags, no filters server-side
(all filtering is client-side JS).

### HTML page (`reports/ab_pool.html`)

Static HTML. Inline `<script>const POOL_DATA = [...];</script>` with the
parquet rows. Vanilla JS, no framework dependency, no HTTP server.

Layout (top to bottom):

1. **Header**: `AB Candidate Pool — built 2026-06-06`
2. **Filter bar** (sticky):
   - 行业 `<select>` with distinct industries (`全部` default)
   - 代码 `<input type="text">` (prefix match)
   - 名称 `<input type="text">` (substring match, Chinese-safe)
3. **Table**: columns `代码 / 名称 / 行业 / 流通市值(亿) / 20日均额(亿) / source_tag`
   - Click column header to sort (default descending);
     remember last sort across filter updates.
   - 流通市值 / 20 日均额 rendered as `亿` (= yuan / 1e8), 2 decimals.
4. **Footer**: `显示 X / 共 Y 票 | build_date: YYYY-MM-DD`
   - `X` updates live as filters change; `Y` is total row count.

Filter logic: three filters AND-combined; re-render on `input` / `change` event.

Implementation lives in `ab_pool_report.py` via plain string templates
(matches `factors_picker._render_html` style — no jinja dependency).

### Config integration — `ab.yaml`

```yaml
base_config: config.yaml
use_ab_pool: true                # NEW; default false
stocks_filter: ["600519", ...]   # optional, still subtractive
arms:
  baseline:  { strategy: { ... }, backtest: { equity_curve_holding_days: [10] } }
  challenger: { strategy: { ... }, backtest: { equity_curve_holding_days: [10] } }
```

Load order in `load_ab_config`:

1. Parse `ab.yaml` and base config as today.
2. If `use_ab_pool=true`:
   - Read `data/ab_pool.parquet` (path from `base_cfg.ab_pool.cache_path`).
   - Build `StockEntry(code, name, sector=industry)` per row.
   - **Replace** `base_cfg.stocks` with this list.
   - If parquet missing → raise → CLI exits 2 with explicit "run ab-pool build first" message.
3. If `stocks_filter` present → intersect with current stocks list.
4. Build per-arm `effective_cfg` as today (`build_effective_cfg`).

`effective_cfg.content_hash` already covers `stocks` list — swapping stocks
naturally yields a different hash, so ml caches (factor panels, model pkls)
isolate correctly without any new hash logic.

### Config integration — `portfolio_ab.yaml`

```yaml
base_config: config.yaml
use_ab_pool: true                # NEW; default false
arms:
  baseline:    { strategy: { ... }, portfolio_backtest: { ... } }
  challenger:  { strategy: { ... }, portfolio_backtest: { ... } }
```

Load order in `build_effective_cfg`:

1. Parse `portfolio_ab.yaml` and base config as today.
2. If `use_ab_pool=true`:
   - Read `data/ab_pool.parquet`.
   - For each arm's `effective_cfg`, set
     `portfolio_backtest.universe_codes` = list of ab_pool codes,
     **unless** the arm's override has an explicit non-None
     `portfolio_backtest.universe_codes` (per-arm always wins).
3. Training pool (`strategy.ml_factor.training_universe`) is **untouched**.
4. Each arm's `content_hash` recomputes (universe_codes changed) → score
   panel caches isolate automatically.

Decoupling rationale (already documented in CLAUDE.md / handoff notes):
training pool feeds cross-sec factor computation and IC/IR weight estimation
— both need the full market (~4350) for stable statistics. AB pool is "where
arms get compared", a separate concern. Conflating them would crash IC
stability at ~100 stocks.

## Error handling

| Trigger | Behavior | Exit |
|---|---|---|
| `universe.parquet` missing | print "run fetch-universe first" | build:1 |
| `ab_pool.parquet` exists, no `--refresh` | print "add --refresh" | build:1 |
| akshare snapshot raises | print error, no partial write | build:2 |
| industry_map load fails | print error, no partial write | build:2 |
| One industry bucket empty (e.g., only 1 stock passes filters) | warn, contribute what's available, continue | build:0 |
| **All** industry buckets empty | print error | build:2 |
| Single stock's daily parquet missing/corrupt for 20d calc | mark NaN, gets caught by liquidity filter, skip stock | build:0 |
| `ab_pool.parquet` missing on `show` | print "run ab-pool build first" | show:1 |
| `webbrowser.open()` fails (headless) | swallow, print output path | show:0 |
| `use_ab_pool=true` + parquet unreadable | runner aborts before any arm runs | ab/portfolio-ab:2 |

No fallback to `cfg.stocks` when `use_ab_pool=true` but parquet missing —
silent fallback would make AB results non-reproducible.

## Caching & invalidation

- `ab_pool.parquet` is the **only** new persistent artifact.
- Akshare snapshot is **not** cached separately — it's a one-shot inside
  `build_ab_pool`. Re-running `build --refresh` always hits akshare again.
- `industry_map` cache (`data/stock_industry_map.parquet`) follows existing
  30-day staleness via `load_or_build_industry_map` — no change.
- `ipo_dates.parquet` cache likewise unchanged.
- `effective_cfg.content_hash` change cascades to ml caches:
  - per-stock AB: stocks list changes → hash changes → factor panel and
    `ml_models/<sig>_*.pkl` keyed by hash automatically refresh.
  - portfolio AB: `portfolio_backtest.universe_codes` change → hash change
    → `data/portfolio_scores/<hash>/...` cache key isolates per arm.

No new hash-key derivation logic is required.

## Testing

New file `tests/test_ab_pool.py` (16 cases). All akshare / baostock /
webbrowser interactions mocked via `monkeypatch`. No live HTTP.

| Test | Validates |
|---|---|
| `test_build_basic` | synthetic 5-industry × 8-stock universe + mock akshare + mock industry_map → parquet exists, expected columns, ~20 rows |
| `test_build_overlap_dedup_tag` | top-2 mcap ∩ top-2 liq = 1 stock → that stock has `source_tag="mcap+liq"`; industry contributes 3 rows (overlap merges into one row, no row duplication) |
| `test_build_hard_filters` | ST / new IPO / illiquid / NaN circ_mv inputs are all excluded |
| `test_build_unknown_industry` | missing industry → `"未知"` bucket selected when `include_unknown_industry=true`, dropped when false |
| `test_build_idempotent_guard` | existing parquet + no `--refresh` → exit 1, file unchanged |
| `test_build_refresh` | existing parquet + `--refresh` → file overwritten, new build_date |
| `test_build_akshare_failure` | mock akshare raises → exit 2, parquet not written |
| `test_show_html_smoke` | mock parquet → `render_ab_pool_html` returns HTML containing all 3 filter inputs, ≥1 table row, build_date string |
| `test_show_missing_parquet` | no parquet → exit 1 |
| `test_ab_yaml_use_ab_pool_per_stock` | `use_ab_pool=true` + mock parquet → loaded stocks equal parquet codes (not cfg.stocks) |
| `test_ab_yaml_stocks_filter_on_ab_pool` | both set → final stocks = intersection |
| `test_portfolio_ab_yaml_use_ab_pool` | `use_ab_pool=true` → both arms' `effective_cfg.portfolio_backtest.universe_codes` == parquet codes; `strategy.ml_factor.training_universe` unchanged |
| `test_portfolio_ab_per_arm_override` | arm with explicit non-None `universe_codes` keeps its own value even when `use_ab_pool=true` |
| `test_use_ab_pool_missing_parquet` | `use_ab_pool=true` + no parquet → ab runner exits 2 before any arm runs |
| `test_cli_ab_pool_build` | full `python -m stockpool ab-pool build` smoke (all data mocked) |
| `test_cli_ab_pool_show` | full `python -m stockpool ab-pool show` smoke; `webbrowser.open` monkeypatched to no-op; output file exists |

Not tested: real akshare HTTP, real webbrowser launch, real industry_map
network fetch.

## Documentation updates

Per project rule "改动后更新文档(CLAUDE.md + README.md)":

- **CLAUDE.md**:
  - "快速命令" — add `ab-pool build` / `ab-pool show` lines
  - "模块地图" — add rows for `ab_pool.py` and `ab_pool_report.py`
  - "配置" — add `ab_pool:` section description; add `use_ab_pool` to
    `ab.yaml` / `portfolio_ab.yaml` sections
  - "数据流" / "缓存" — add `ab_pool.parquet` to the cache list
  - "测试" — add `test_ab_pool.py` row
- **README.md**:
  - "快速开始" — add a short paragraph on building the AB pool before
    running AB tests
  - "常用命令" — `ab-pool` subcommand entries

## Open questions / risks

- **Akshare snapshot reliability.** `stock_zh_a_spot_em` is the canonical
  full-A spot endpoint and has been stable historically, but akshare upstream
  changes occasionally. Mitigation: snapshot lives only inside `build_ab_pool`;
  failure is non-destructive (no partial parquet). User reruns `build --refresh`
  when akshare recovers. No 24/7 dependency.
- **SW-1 granularity drift.** baostock returns SW level-1 (~28 industries);
  akshare 东财 returns ~80. With `industry_source: auto` (baostock first),
  default behavior is stable at ~28 buckets → ~100-stock pool. If a user
  explicitly forces `industry_source: akshare`, bucket count balloons to ~80
  → ~250-stock pool. Acceptable; documented in `AbPoolConfig` docstring.
- **"未知" bucket size.** Stocks without industry map entry (typically very
  new IPOs or delisted) — with `include_unknown_industry=true` they can add
  up to 4 codes. Almost always benign; setting `false` is a clean escape.
- **Naming asymmetry with `stocks[].sector`.** `cfg.stocks` uses `sector`
  (free-form Chinese name); ab_pool uses `industry` (SW-1 from industry_map).
  Synthesizing `StockEntry(sector=industry)` for per-stock AB display works,
  but the field-name mismatch is a small wart. Acceptable for MVP.
