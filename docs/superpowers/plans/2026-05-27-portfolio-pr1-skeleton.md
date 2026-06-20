# PR-1 Plan: Portfolio Backtest Skeleton (MVP)

> **Plan date**: 2026-05-27
> **Spec**: `docs/superpowers/specs/2026-05-24-portfolio-framework-design.md` §10.2
> **Scope**: PR-1 only — core skeleton. Universe = `cfg.stocks`, no eligibility/industry cap.

---

## 1. 目标

Land a runnable `python -m stockpool portfolio-backtest --config config.yaml` that:
- Uses `cfg.stocks` as universe (PR-2 will switch to `load_universe_cache`)
- Precomputes a (T × N) score panel by calling `legacy_strategy.generate_signals()` per stock
- Top-K by score every `rebalance_n_days` bars, equal-weight
- T+1 fill at `open[t+1]`
- Outputs `reports/portfolio/<date>.html` with one equity curve + holdings timeline
- Zero regression: existing `backtest` / `ab` CLI untouched

PR-1 **deliberately omits** (deferred to PR-2/3/4):
- Eligibility filter (schema lands in PR-1, engine doesn't read it)
- Industry cap (top-K just takes top by score)
- Staggered ensemble (`start_offset` parameter present but `staggered_starts` enforced = 1)
- Portfolio AB

## 2. New files

| File | Contents |
|---|---|
| `src/stockpool/portfolio/__init__.py` | Public surface re-exports |
| `src/stockpool/portfolio/strategy.py` | `PortfolioStrategy` ABC + `PrecomputedScoreStrategy` |
| `src/stockpool/portfolio/scoring.py` | `precompute_scores_from_legacy(legacy, panel_data)` |
| `src/stockpool/portfolio/result.py` | `PortfolioTrade`, `PortfolioBacktestResult` |
| `src/stockpool/portfolio/engine.py` | `PortfolioConfig`, `Portfolio` (internal), `PortfolioEngine` |
| `src/stockpool/portfolio/report.py` | Single-arm HTML render (equity + holdings timeline) |
| `tests/test_portfolio_strategy.py` | `PrecomputedScoreStrategy.predict_scores` semantics |
| `tests/test_portfolio_scoring.py` | precompute happy path + failure isolation + look-ahead |
| `tests/test_portfolio_engine.py` | T+1, cash conservation, rebalance diff, determinism |
| `tests/test_cli_portfolio_backtest.py` | CLI smoke (monkeypatched fetcher) |

## 3. Modified files

- `src/stockpool/config.py` — add `PortfolioBacktestConfig` (full schema including `eligibility`, so PR-2 doesn't migrate user yaml)
- `src/stockpool/cli.py` — register `portfolio-backtest` subcommand + `cmd_portfolio_backtest`
- `CLAUDE.md`, `README.md` — module-map / commands / config / tests sections
- `docs/strategy_improvement_2026.md` §6 — mark PR-1 ✅ (will do once plan completes)

## 4. Design notes

### 4.1 `PortfolioStrategy` ABC

Per spec §5.1 — `predict_scores(date_t, panel_data) -> dict[code, score]`. No inheritance from per-stock `Strategy`.

### 4.2 `PrecomputedScoreStrategy`

Wraps a `(T × N)` `pd.DataFrame` score panel. `predict_scores(date_t, panel_data)` returns `panel.loc[date_t].dropna()` filtered to codes present in `panel_data`.

### 4.3 `precompute_scores_from_legacy(legacy, panel_data, score_field="final_score")`

Iterates `panel_data.items()`, calls `legacy.generate_signals(daily)`, extracts `signals.set_index("date")[score_field]`, builds `pd.DataFrame(series_by_code)`. Failures (any per-stock exception) are caught, logged at WARNING, skipped.

Both `CompositeVerdictStrategy` and `MLFactorStrategy` already write `final_score` in `generate_signals()` output — confirmed via grep on `backtesting/strategies.py`.

### 4.4 `PortfolioEngine.run(panel_data, start_offset=0)`

Pseudocode per spec §6.2. Key invariants:
- All-dates union sorted ascending → bar index
- `rebalance_bars = {start_offset, start_offset+n, start_offset+2n, ...}`
- At each rebalance bar `t`:
  1. `scores = self.strategy.predict_scores(date_t, panel_data)`
  2. PR-1: `eligible = set(scores.keys())` (no filter)
  3. `ranked = sorted(scores.items(), key=-score)[:top_k]` (no industry cap in PR-1)
  4. `portfolio.rebalance_to(target_codes=set(ranked), exec_bar_idx=t+1, ...)`
- Mark-to-market every bar with `close[t]`
- T+1: fills at `open[t+1]`. Last bar (no t+1) ⇒ skip the trade.

Internal `Portfolio`:
- `cash: float`
- `positions: dict[code, {entry_idx, entry_price, shares, weight_at_entry}]`
- `rebalance_to(target_codes, exec_bar_idx, panel_data, costs)`:
  - Sells positions in `current - target` at `open[exec]` minus `sell_cost`
  - For survivors, redistribute total equity equally across target → buy/top-up at `open[exec]` minus `buy_cost`
  - PR-1 simplification: **full rebalance** (sell ALL, rebuild equal-weight from cash). Cleaner, matches "等权 top-K" semantics in spec. Later PRs can add turnover_cap.

### 4.5 Metrics

Reuse `stockpool.backtesting.metrics.compute_metrics(equity_series, trades, risk_free_rate)`. The function takes `list[Trade]` (per-stock). For portfolio, we'll pass per-`PortfolioTrade`-like adapter or compute portfolio-specific metrics inline (sharpe / max_dd / total_return are equity-only; win_rate/avg_trade_ret use trades).

Decision: build a thin adapter — turn each `PortfolioTrade` into a `Trade(entry_idx, exit_idx, entry_price, exit_price, ret, days_held)` purely for `compute_metrics` consumption. Avoids duplicating math.

### 4.6 `PortfolioBacktestConfig` schema

```python
class PortfolioConfig(BaseModel):  # rename to avoid collision: PortfolioRunConfig
    top_k: int = 20
    rebalance_n_days: int = 5
    max_per_industry: int | None = 5
    initial_cash: float = 1.0

class EligibilityConfig(BaseModel):
    min_avg_amount_20d: float = 5e7
    exclude_st: bool = True
    min_history_bars: int = 60

class PortfolioBacktestConfig(BaseModel):
    enabled: bool = False
    portfolio: PortfolioRunConfig = Field(default_factory=PortfolioRunConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    staggered_starts: int = Field(default=1, ge=1, le=20)
    score_cache_dir: str = "data/portfolio_scores"
    model_config = ConfigDict(extra="forbid")
```

**Name collision**: spec uses `PortfolioConfig` for the inner block, but our backtest framework has no such name. To avoid confusion with `PortfolioBacktestConfig`, the inner block class is named `PortfolioRunConfig` in code but the YAML key remains `portfolio:` (per spec). Class name is internal-only.

### 4.7 CLI

```bash
python -m stockpool portfolio-backtest --config config.yaml [--refresh-scores]
```

`--refresh-scores` bypasses `data/portfolio_scores/<content_hash>.parquet` cache (only operational flag, doesn't change config semantics).

PR-1 ignores `cfg.portfolio_backtest.enabled` (since the command is opt-in by invocation). Or: command refuses if `enabled=false`? Spec §6.1 step 1 says "检查 cfg.portfolio_backtest.enabled". Decision: **honor `enabled=false`** — error out with exit code 2 if false. This forces users to opt-in in yaml, prevents accidental runs.

### 4.8 Report

Minimal HTML: pyecharts line chart for equity vs B&H (simple equal-weight buy-and-hold of cfg.stocks at bar 0) + a metrics table + a "holdings over time" stacked area or simple line plot of `num_positions`.

PR-1 keeps the report bare. PR-3 will add ensemble envelope.

## 5. Tests

### 5.1 `test_portfolio_strategy.py`

- `PrecomputedScoreStrategy.predict_scores(date, panel_data)` returns dict
- Returns `{}` for date not in panel
- Filters to codes present in `panel_data`
- NaN scores dropped

### 5.2 `test_portfolio_scoring.py`

- Happy path: stub legacy returns `(date, final_score)` per code → assemble correct T×N panel
- One stock throws → logged warning + skipped, others present
- Look-ahead: feed legacy a stub that records the daily_df it sees; assert no future bars leaked (PR-1: easy test — legacy gets whole `daily`, no truncation needed; the panel is built once. Real look-ahead lives in legacy's `generate_signals`; we just verify our helper doesn't subset it.)

### 5.3 `test_portfolio_engine.py`

- Constructed score panel with deterministic top-K → assert positions match
- Cash conservation: `cash + Σ(shares_i * close[t])` ≈ `equity[t]` to 1e-9
- T+1: fills at `open[t+1]`, not `open[t]`
- Rebalance diff: from {A,B,C} → {B,C,D} sells A buys D; B,C re-weighted
- Determinism: two runs on same inputs → identical curves + trades
- `rebalance_n_days=5, start_offset=2` → rebalance bar indices {2,7,12,...}
- Last bar skip: target diff at last bar → no trade emitted

### 5.4 `test_cli_portfolio_backtest.py`

- Monkeypatch `fetch_daily` to return synthetic OHLCV; assert reports/portfolio/<date>.html exists
- `enabled=false` → exit code 2
- `--refresh-scores` bypasses cache (assert legacy strategy called twice across two runs)

## 6. Task breakdown

1. Add `PortfolioBacktestConfig` schema to `config.py` (+ test in `test_config.py`)
2. New module: `portfolio/strategy.py` + `result.py` + tests
3. New module: `portfolio/scoring.py` + tests
4. New module: `portfolio/engine.py` (skeleton + Portfolio internal class) + tests
5. New module: `portfolio/report.py` (minimal HTML)
6. Wire `cli.cmd_portfolio_backtest` + score caching
7. CLI smoke test
8. Docs sync: CLAUDE.md, README.md, strategy_improvement_2026.md §6
9. Full `pytest tests/ -q` green

## 7. Acceptance

```bash
python -m stockpool portfolio-backtest --config config.yaml
# → reports/portfolio/<date>.html exists, equity curve renders, no crashes
pytest tests/ -q                        # all green, ~380 tests
python -m stockpool backtest            # unchanged behavior (regression check)
python -m stockpool ab                  # unchanged behavior (regression check)
```
