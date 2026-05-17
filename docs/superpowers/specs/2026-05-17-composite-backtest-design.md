# Composite-Score Historical Backtest — Design

**Date:** 2026-05-17
**Status:** Approved, ready for implementation plan
**Scope:** Add historical backtest of the composite weighted scoring system. Two outputs:
- **A.** Verdict-bucketed forward-return statistics, integrated into the existing per-stock HTML report.
- **B.** Equity-curve simulation under a fixed-holding-period trading rule, produced by a new CLI subcommand.

The existing per-signal hit-rate backtest in `src/stockpool/backtest.py` is preserved unchanged.

---

## 1. Goals

Answer the question: *"If I had been using the current weights (`config.yaml::weights` + `scoring` + `verdicts`) historically, how would the composite signal have performed?"*

Two complementary views:

- **A — Bucket stats:** For every historical day, bucket by that day's `verdict` (`strong_buy` / `buy` / `neutral` / `sell` / `strong_sell`) and report forward 5/10/20-day mean return and win rate per bucket. Lets the user see whether `strong_buy` days actually outperform `neutral` days.
- **B — Equity curve:** Simulate a simple long-only strategy (enter on `buy`/`strong_buy`, exit after N days or on `sell`/`strong_sell`) for several N values, plotted alongside a buy-and-hold baseline.

Per-stock only. No portfolio-level aggregation in this iteration.

---

## 2. Module Layout

**New file:** `src/stockpool/backtest_composite.py`

Public surface:

- `walk_forward_verdicts(daily_df, weights, scoring_cfg, verdicts_cfg, indicators_cfg) -> pd.DataFrame`
  Returns one row per daily bar (after warmup) with columns: `date, close, daily_score, weekly_score, final_score, verdict`.
- `verdict_bucket_stats(wf_df, forward_days) -> dict`
  Used by A.
- `simulate_equity_curve(wf_df, holding_days_list, with_buy_and_hold=True) -> EquityResult`
  Used by B. `EquityResult` is a dataclass containing the equity series (per N and buy-and-hold) and per-strategy metrics.

**New CLI subcommand:** `python -m stockpool backtest [--config ...] [--stocks ...] [--refresh]`
Produces `reports/backtest/YYYY-MM-DD.html` and `reports/backtest/latest.html`.

**Unchanged modules:** `signals.py`, `backtest.py` (per-signal), `fetcher.py`, `indicators.py`, the existing `run` subcommand.

**A's integration into the existing report:**
- `cli.py::_analyze_one` calls `verdict_bucket_stats` and stores the result on `StockAnalysis` (new field `verdict_hit_rates`).
- `report.py` renders a new table beneath the existing per-signal hit-rate section.

**Config extension (`config.yaml`):**
```yaml
backtest:
  forward_days: [5, 10, 20]                # existing — used by A and per-signal stats
  equity_curve_holding_days: [5, 10, 20]   # new — used by B
```

`config.py::BacktestConfig` adds the `equity_curve_holding_days: list[int]` field with default `[5, 10, 20]`.

---

## 3. Walk-Forward Verdict Computation (shared by A and B)

This is the load-bearing piece. The full composite score (daily + weekly + resonance bonus) must be reconstructed for every historical bar **without future-data leakage**, so the resulting verdict matches what the live `_analyze_one` would have produced on that date.

### 3.1 Algorithm

```
enriched_daily = add_all(daily_df, indicators_cfg)   # precompute once; left-to-right indicators have no leakage

cache = (last_week_key, last_weekly_score)
for i in range(warmup_start, len(daily_df)):
    daily_window = enriched_daily.iloc[max(0, i-1) : i+1]
    daily_triggers = detect_signals(daily_window, weights)
    daily_score = score_triggers(daily_triggers)

    week_key = (date_i.year, ISO-week-of(date_i))
    if cache.last_week_key == week_key:
        weekly_score = cache.last_weekly_score
    else:
        weekly_full = resample_to_weekly(daily_df.iloc[:i+1])      # critical: slice first
        if len(weekly_full) >= 30:
            enriched_w = add_all(weekly_full, indicators_cfg)
            weekly_score = score_triggers(detect_signals(enriched_w, weights))
        else:
            weekly_score = 0
        cache = (week_key, weekly_score)

    final_score = combine_daily_weekly(daily_score, weekly_score, scoring_cfg)
    verdict = verdict_of(final_score, verdicts_cfg)
    emit (date_i, close_i, daily_score, weekly_score, final_score, verdict)
```

### 3.2 No-leakage invariants

- The weekly DataFrame is always derived from `daily_df.iloc[:i+1]` — never from a precomputed full-history weekly aggregation. Reason: the most recent weekly bar in a precomputed full-history series would aggregate days *after* `date_i` whenever `date_i` falls mid-week.
- Daily indicators (`enriched_daily`) are precomputed once on the full daily history. This is safe because all indicators used (MA, MACD, KDJ, RSI, BOLL, vol ratio, breakout flags) are computed left-to-right; their value at bar `i` depends only on bars `≤ i`.
- The week-key cache reuses the previous bar's `weekly_score` only when `date_i` and `date_{i-1}` share the same `(year, ISO-week)`. A week boundary forces re-aggregation.

### 3.3 Warmup

- Daily warmup: same as live (`len(daily) >= 30`). Earlier bars are skipped.
- Weekly warmup: if `len(weekly) < 30`, `weekly_score = 0`. This matches `cli.py::_analyze_one` behavior (a warning is logged live; here we silently emit verdict using `daily_score` only).

### 3.4 Performance budget

500-day history × 8 stocks. Daily loop ≈ 4000 iterations. Weekly re-aggregations ≈ 100 weeks × 8 = 800 (thanks to the week-key cache). Expected runtime: low single-digit seconds total. No need for further optimization in this iteration.

### 3.5 Test for leakage

Strongest possible test: for any bar `i`, the verdict returned by `walk_forward_verdicts` at row `i` must exactly equal the verdict produced by running the constituent pipeline (`add_all` → `detect_signals` → `score_triggers` for both daily and weekly slices, then `combine_daily_weekly` → `verdict_of`) on `daily_df.iloc[:i+1]`. Test the final bar and three random middle bars. Note: this calls the underlying functions directly, not the CLI wrapper `_analyze_one`, to keep the test free of data-fetch and logging side effects.

---

## 4. A — Verdict-Bucket Stats

### 4.1 Algorithm

```python
def verdict_bucket_stats(wf_df, forward_days):
    closes = wf_df["close"].values
    verdicts = wf_df["verdict"].values
    buckets = {label: {"count": 0, "returns": {n: [] for n in forward_days}}
               for label in ("strong_buy","buy","neutral","sell","strong_sell")}

    for i in range(len(wf_df)):
        v = verdicts[i]
        buckets[v]["count"] += 1
        for n in forward_days:
            j = i + n
            if j >= len(closes):
                continue
            ret_pct = (closes[j] / closes[i] - 1) * 100
            buckets[v]["returns"][n].append(ret_pct)

    # Aggregate per bucket → {count, forward_N: {mean_return_pct, win_rate, sample_size}}
```

### 4.2 Win-rate direction

- `strong_buy`, `buy`: a sample is a "win" if `ret_pct > 0`
- `strong_sell`, `sell`: a sample is a "win" if `ret_pct < 0`
- `neutral`: win = `ret_pct > 0` (used as a baseline reference)

### 4.3 Output shape

```python
{
  "strong_buy":  {"count": 12, "forward_5": {"mean_return_pct": 3.2, "win_rate": 0.75, "sample_size": 12}, "forward_10": {...}, "forward_20": {...}},
  "buy":         {"count": 45, ...},
  "neutral":     {"count": 320, ...},
  "sell":        {"count": 38, ...},
  "strong_sell": {"count":  8, ...},
}
```

### 4.4 Report rendering

`report.py` adds a 5-row × (1 + 1 + 3×2)-column table beneath the existing per-signal hit-rate section:

| 评级 | 样本 | 5 日均收益 | 5 日胜率 | 10 日均收益 | 10 日胜率 | 20 日均收益 | 20 日胜率 |
|---|---|---|---|---|---|---|---|

Empty buckets (e.g. zero `strong_sell` days in the window) render as `—`.

---

## 5. B — Equity-Curve Simulation

### 5.1 State machine (per stock per N)

`position[t]` ∈ {0, 1} represents whether the strategy is long during the period from `close[t-1]` to `close[t]`. The decision for `position[t]` is made at end-of-day `t-1` based on `verdict[t-1]`.

```python
position[start_idx] = 0
days_held = 0
entry_idx = None

for t in range(start_idx + 1, len(wf_df)):
    prev_verdict = verdicts[t-1]

    if position[t-1] == 0:
        if prev_verdict in {"buy", "strong_buy"}:
            position[t] = 1
            entry_idx = t - 1
            days_held = 0
        else:
            position[t] = 0
    else:  # currently long
        held_now = days_held + 1
        if held_now >= N or prev_verdict in {"sell", "strong_sell"}:
            position[t] = 0
            record trade: (entry_idx → t-1, ret = close[t-1]/close[entry_idx] - 1)
        else:
            position[t] = 1
            days_held = held_now

    daily_ret = close[t] / close[t-1] - 1
    equity[t] = equity[t-1] * (1 + position[t] * daily_ret)
```

### 5.2 Rules and edge cases

- **No T+1, no fees, no slippage** — B1 simplification.
- **Same-day expiry-and-sell:** if `days_held >= N` and `prev_verdict ∈ {sell, strong_sell}`, the bar exits cleanly either way (no conflict — both paths produce `position[t] = 0`).
- **Buy signal while already long:** ignored. No pyramiding.
- **Open position at end of series:** the trade is NOT counted in the closed-trade ledger (no exit price), but the unrealized equity is preserved in `equity[-1]`.
- **`start_idx`:** the first bar where `walk_forward_verdicts` emitted a row. Equity is normalized: `equity[start_idx] = 1.0`.

### 5.3 Metrics (per strategy curve)

- `total_return = equity[-1] / equity[start_idx] - 1`
- `annualized_return = (1 + total_return) ** (252 / num_trading_days) - 1`
- `max_drawdown = max((running_peak - equity[t]) / running_peak)`
- `trade_count` = number of completed (entry, exit) pairs
- `win_rate = wins / trade_count` (wins = trades with positive return)
- `avg_trade_return_pct = mean of per-trade returns`

### 5.4 Buy-and-hold baseline

`equity_bh[t] = close[t] / close[start_idx]`, starting from the same `start_idx`. Reported metrics: `total_return`, `annualized_return`, `max_drawdown` computed normally. `trade_count = 1` (the single buy-and-hold position). `win_rate` and `avg_trade_return_pct` render as `—` in the comparison table (single-trade win-rate is degenerate).

### 5.5 Report rendering

`reports/backtest/YYYY-MM-DD.html` (plus a `latest.html` symlink/copy):

- Top: index of stocks with anchor links.
- Per stock, one section containing:
  - One pyecharts line chart with shared X axis, 4 series: `N=5`, `N=10`, `N=20`, `Buy & Hold`. Y axis shows normalized equity (start = 1.0).
  - One comparison table:

    | 策略 | 总收益 | 年化 | 最大回撤 | 交易次数 | 胜率 | 平均单笔 |
    |---|---|---|---|---|---|---|

---

## 6. CLI

```
python -m stockpool backtest [--config config.yaml] [--stocks 000001,000002] [--refresh]
```

Flow inside `cmd_backtest`:
1. Load config.
2. For each stock in config (filtered by `--stocks` if provided):
   - `fetch_daily` (reuses cache, `--refresh` forces refetch).
   - `walk_forward_verdicts`.
   - For each N in `config.backtest.equity_curve_holding_days`: `simulate_equity_curve` for that N.
   - Buy-and-hold baseline.
3. Render one HTML page combining all stocks.
4. Write `reports/backtest/YYYY-MM-DD.html`; copy to `reports/backtest/latest.html`.

The existing trading-day check is **not** applied to `backtest` (intentional — you should be able to run a backtest anytime, weekends included).

---

## 7. Testing

`tests/test_backtest_composite.py`:

1. **`walk_forward_verdicts` — no leakage**
   - Generate ~100-bar synthetic OHLCV that produces deterministic indicator values.
   - Run walk-forward across the full series.
   - Assert the final-bar verdict equals the verdict from running the live `_analyze_one` pipeline on the same data (strongest leakage test).
   - For 3 randomly-chosen middle bars, slice `daily_df.iloc[:i+1]`, run the live pipeline, and assert verdict equality with the walk-forward row at `i`.

2. **`verdict_bucket_stats` — numerical correctness**
   - Hand-construct a 10-row series with known verdicts and closes.
   - Assert per-bucket `count`, per-`forward_N` `mean_return_pct`, `win_rate` (with direction), and `sample_size` (`i + n >= len` rows are not counted).

3. **`simulate_equity_curve` — state-machine edges**
   - All-neutral series → flat equity at 1.0, `trade_count = 0`.
   - Buy on day 0, then all neutral, N=10 → exit on day 10 at `close[10]`, equity matches.
   - Buy on day 0, sell on day 3, N=10 → early exit on day 3, days_held = 3.
   - Buy on day 0, buy again on day 2 → second buy ignored, no double-entry.
   - Buy on day-(len-3) (open at end of series) → trade not counted, equity reflects unrealized P&L.
   - Buy-and-hold baseline equals `close[t] / close[start_idx]` for all `t`.

4. **`tests/test_cli_backtest.py` — smoke**
   - Use a small fixture CSV (or one already in `tests/fixtures/`).
   - Run `python -m stockpool backtest --stocks <code>`.
   - Assert `reports/backtest/<date>.html` exists, > 1 KB, and contains the literal strings `N=5`, `N=10`, `N=20`, `Buy & Hold`.

TDD discipline: each function gets its failing test before its implementation.

---

## 8. Non-Goals (out of scope, possible follow-ups)

- T+1 enforcement, A-share fees and slippage (would be a "B3" follow-up).
- Portfolio-level aggregation across the 8 stocks.
- Walk-forward weight optimization or parameter search.
- Short-side simulation (A-share retail can't short, kept long-only).
- Sharpe / Sortino / Calmar. Easy to add later if useful.
