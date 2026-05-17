# Composite-Score Historical Backtest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a historical backtest of the composite weighted scoring system: (A) per-stock verdict-bucket forward-return stats embedded in the existing report, and (B) a new `backtest` CLI subcommand that produces an equity-curve HTML report with N=5/10/20 strategies and a buy-and-hold baseline.

**Architecture:** New module `backtest_composite.py` reconstructs the composite verdict for every historical day without future-data leakage (slice-then-resample for weekly + week-key cache). A and B both consume the same walk-forward output. Backtest HTML rendering lives in a new `backtest_report.py` to keep `report.py` focused.

**Tech Stack:** Python 3, pandas, pydantic, pyecharts (already in use), pytest.

**Spec:** `docs/superpowers/specs/2026-05-17-composite-backtest-design.md`

---

## File Structure

**New files:**
- `src/stockpool/backtest_composite.py` — walk-forward verdicts, verdict bucket stats, equity-curve simulator
- `src/stockpool/backtest_report.py` — pyecharts equity chart + metrics table → HTML
- `tests/test_backtest_composite.py` — unit tests for all three functions
- `tests/test_cli_backtest.py` — smoke test for the new CLI subcommand

**Modified files:**
- `src/stockpool/config.py` — add `equity_curve_holding_days` field to `BacktestConfig`
- `config.yaml` — add `equity_curve_holding_days: [5, 10, 20]`
- `src/stockpool/report.py` — add `verdict_hit_rates` field to `StockAnalysis`; render new table in stock section
- `src/stockpool/cli.py` — call `walk_forward_verdicts` + `verdict_bucket_stats` inside `_analyze_one`; add `cmd_backtest` subcommand
- `tests/test_config.py` — update `_minimal_yaml()` to include new field
- `tests/test_report_smoke.py` — extend smoke assertions for the new table

---

## Task 1: Extend `BacktestConfig` with `equity_curve_holding_days`

**Files:**
- Modify: `src/stockpool/config.py:84-85`
- Modify: `tests/test_config.py:35`
- Modify: `config.yaml:56-57`

- [ ] **Step 1: Update `_minimal_yaml()` in test_config.py to include the new field**

Modify line 35 from:
```python
        "backtest": {"forward_days": [5, 10, 20]},
```
to:
```python
        "backtest": {"forward_days": [5, 10, 20], "equity_curve_holding_days": [5, 10, 20]},
```

- [ ] **Step 2: Add a new failing test in `tests/test_config.py`**

Append to `tests/test_config.py`:
```python
def test_equity_curve_holding_days_loads(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(_minimal_yaml()), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.backtest.equity_curve_holding_days == [5, 10, 20]


def test_equity_curve_holding_days_defaults_when_missing(tmp_path):
    raw = _minimal_yaml()
    del raw["backtest"]["equity_curve_holding_days"]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.backtest.equity_curve_holding_days == [5, 10, 20]
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `pytest tests/test_config.py::test_equity_curve_holding_days_loads tests/test_config.py::test_equity_curve_holding_days_defaults_when_missing -v`
Expected: FAIL — `AttributeError: 'BacktestConfig' object has no attribute 'equity_curve_holding_days'`.

- [ ] **Step 4: Add the field to `BacktestConfig`**

Modify `src/stockpool/config.py` lines 84-85:
```python
class BacktestConfig(BaseModel):
    forward_days: list[int]
    equity_curve_holding_days: list[int] = Field(default_factory=lambda: [5, 10, 20])
```

- [ ] **Step 5: Add the entry to `config.yaml`**

Modify `config.yaml` lines 56-57 from:
```yaml
backtest:
  forward_days: [5, 10, 20]
```
to:
```yaml
backtest:
  forward_days: [5, 10, 20]
  equity_curve_holding_days: [5, 10, 20]
```

- [ ] **Step 6: Run all config tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: all PASS (including the new two and the existing `test_default_config_yaml_loads`).

- [ ] **Step 7: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py config.yaml
git commit -m "feat(config): add equity_curve_holding_days for composite backtest"
```

---

## Task 2: `walk_forward_verdicts` — historical verdict reconstruction without leakage

**Files:**
- Create: `src/stockpool/backtest_composite.py`
- Create: `tests/test_backtest_composite.py`

This is the core function. Three tests guard against look-ahead bias and warmup edge cases.

- [ ] **Step 1: Create `tests/test_backtest_composite.py` with the leakage test**

```python
"""Walk-forward composite verdict tests — guard against look-ahead bias."""
import numpy as np
import pandas as pd
import pytest

from stockpool.backtest_composite import walk_forward_verdicts
from stockpool.config import (
    BOLLConfig, IndicatorsConfig, KDJConfig, MACDConfig,
    ScoringConfig, VerdictsConfig, WeightsConfig,
)
from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.signals import (
    combine_daily_weekly, detect_signals, score_triggers, verdict_of,
)


@pytest.fixture
def weights() -> WeightsConfig:
    return WeightsConfig(
        ma_cross_strong=2, ma_alignment=1,
        macd_cross_above_zero=2, macd_cross_below_zero=1, macd_histogram_expand=1,
        kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1,
        boll_band_touch=2, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1,
        breakout_new_high=2, breakout_new_low=2,
    )


@pytest.fixture
def scoring() -> ScoringConfig:
    return ScoringConfig(
        daily_weight=0.7, weekly_weight=0.3,
        resonance_bonus=2, resonance_daily_threshold=3, resonance_weekly_threshold=1,
    )


@pytest.fixture
def verdicts_cfg() -> VerdictsConfig:
    return VerdictsConfig(strong_buy=6, buy=3, sell=-3, strong_sell=-6)


@pytest.fixture
def indicators_cfg() -> IndicatorsConfig:
    return IndicatorsConfig(
        ma_periods=[5, 10, 20, 60],
        macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3),
        rsi_periods=[6, 12, 24],
        boll=BOLLConfig(n=20, k=2.0),
        volume_ratio_window=5,
        breakout_window=20,
    )


def _synthetic_history(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """n trading days of pseudo-realistic OHLCV with embedded volatility."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
    })


def _live_verdict_at(daily, i, weights, scoring, verdicts_cfg, indicators_cfg):
    """Reproduce _analyze_one's verdict on daily.iloc[:i+1]."""
    sub_daily = daily.iloc[:i + 1].copy()
    enriched_d = add_all(sub_daily, indicators_cfg)
    daily_triggers = detect_signals(enriched_d, weights)
    daily_score = score_triggers(daily_triggers)

    weekly = resample_to_weekly(sub_daily)
    if len(weekly) >= 30:
        enriched_w = add_all(weekly, indicators_cfg)
        weekly_score = score_triggers(detect_signals(enriched_w, weights))
    else:
        weekly_score = 0

    final = combine_daily_weekly(daily_score, weekly_score, scoring)
    return verdict_of(final, verdicts_cfg), daily_score, weekly_score


def test_walk_forward_matches_live_at_final_bar(weights, scoring, verdicts_cfg, indicators_cfg):
    """The verdict at the last bar must equal the live pipeline's verdict on the full data."""
    daily = _synthetic_history(n=300)
    wf = walk_forward_verdicts(daily, weights, scoring, verdicts_cfg, indicators_cfg)

    assert len(wf) > 0, "walk-forward returned no rows"
    last = wf.iloc[-1]
    expected_verdict, expected_d, expected_w = _live_verdict_at(
        daily, len(daily) - 1, weights, scoring, verdicts_cfg, indicators_cfg
    )
    assert last["verdict"] == expected_verdict
    assert last["daily_score"] == expected_d
    assert last["weekly_score"] == expected_w
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_backtest_composite.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockpool.backtest_composite'`.

- [ ] **Step 3: Create the initial implementation of `backtest_composite.py`**

Create `src/stockpool/backtest_composite.py`:
```python
"""Walk-forward composite-score backtest (A: bucket stats, B: equity curve).

Distinct from backtest.py, which computes per-signal hit rates. This module
reconstructs the full composite verdict (daily + weekly + resonance) for every
historical day, without future-data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpool.config import (
    IndicatorsConfig, ScoringConfig, VerdictsConfig, WeightsConfig,
)
from stockpool.fetcher import resample_to_weekly
from stockpool.indicators import add_all
from stockpool.signals import (
    combine_daily_weekly, detect_signals, score_triggers, verdict_of,
)


_DAILY_WARMUP = 30  # match cli.py::_analyze_one's threshold


def _iso_week_key(ts) -> tuple[int, int]:
    iso = pd.Timestamp(ts).isocalendar()
    return (int(iso.year), int(iso.week))


def walk_forward_verdicts(
    daily_df: pd.DataFrame,
    weights: WeightsConfig,
    scoring_cfg: ScoringConfig,
    verdicts_cfg: VerdictsConfig,
    indicators_cfg: IndicatorsConfig,
) -> pd.DataFrame:
    """For each daily bar (after warmup) compute the composite verdict that
    the live pipeline would have produced on data available at that bar.

    Returns a DataFrame with columns:
        date, close, daily_score, weekly_score, final_score, verdict
    """
    if len(daily_df) < _DAILY_WARMUP:
        return pd.DataFrame(columns=[
            "date", "close", "daily_score", "weekly_score", "final_score", "verdict",
        ])

    enriched_daily = add_all(daily_df, indicators_cfg)

    rows: list[dict] = []
    cached_week_key: tuple[int, int] | None = None
    cached_weekly_score: int = 0

    for i in range(_DAILY_WARMUP - 1, len(daily_df)):
        daily_window = enriched_daily.iloc[max(0, i - 1): i + 1]
        daily_triggers = detect_signals(daily_window, weights)
        daily_score = score_triggers(daily_triggers)

        week_key = _iso_week_key(daily_df["date"].iloc[i])
        if cached_week_key == week_key:
            weekly_score = cached_weekly_score
        else:
            weekly = resample_to_weekly(daily_df.iloc[:i + 1])
            if len(weekly) >= 30:
                enriched_w = add_all(weekly, indicators_cfg)
                weekly_score = score_triggers(detect_signals(enriched_w, weights))
            else:
                weekly_score = 0
            cached_week_key = week_key
            cached_weekly_score = weekly_score

        final_score = combine_daily_weekly(daily_score, weekly_score, scoring_cfg)
        verdict = verdict_of(final_score, verdicts_cfg)

        rows.append({
            "date": daily_df["date"].iloc[i],
            "close": float(daily_df["close"].iloc[i]),
            "daily_score": int(daily_score),
            "weekly_score": int(weekly_score),
            "final_score": float(final_score),
            "verdict": verdict,
        })

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_backtest_composite.py::test_walk_forward_matches_live_at_final_bar -v`
Expected: PASS.

- [ ] **Step 5: Add the middle-bar leakage test**

Append to `tests/test_backtest_composite.py`:
```python
def test_walk_forward_matches_live_at_middle_bars(weights, scoring, verdicts_cfg, indicators_cfg):
    """For three random middle bars, walk-forward output must equal live pipeline."""
    daily = _synthetic_history(n=300, seed=7)
    wf = walk_forward_verdicts(daily, weights, scoring, verdicts_cfg, indicators_cfg)

    rng = np.random.default_rng(123)
    # wf row k corresponds to daily index (29 + k); pick 3 rows away from edges
    middle_indices = rng.choice(range(20, len(wf) - 20), size=3, replace=False)

    for k in middle_indices:
        daily_idx = 29 + int(k)
        expected_verdict, expected_d, expected_w = _live_verdict_at(
            daily, daily_idx, weights, scoring, verdicts_cfg, indicators_cfg
        )
        row = wf.iloc[int(k)]
        assert row["verdict"] == expected_verdict, f"verdict mismatch at k={k}, daily_idx={daily_idx}"
        assert row["daily_score"] == expected_d
        assert row["weekly_score"] == expected_w
```

- [ ] **Step 6: Run the middle-bar test**

Run: `pytest tests/test_backtest_composite.py::test_walk_forward_matches_live_at_middle_bars -v`
Expected: PASS. If FAIL, the week-key cache logic has a bug — the most likely cause is reusing a stale `weekly_score` across a week boundary.

- [ ] **Step 7: Add the warmup edge-case test**

Append to `tests/test_backtest_composite.py`:
```python
def test_walk_forward_handles_short_history(weights, scoring, verdicts_cfg, indicators_cfg):
    """Less than 30 daily bars returns an empty DataFrame."""
    daily = _synthetic_history(n=20)
    wf = walk_forward_verdicts(daily, weights, scoring, verdicts_cfg, indicators_cfg)
    assert len(wf) == 0
    assert list(wf.columns) == [
        "date", "close", "daily_score", "weekly_score", "final_score", "verdict",
    ]


def test_walk_forward_weekly_score_zero_when_insufficient_weekly_bars(
    weights, scoring, verdicts_cfg, indicators_cfg
):
    """When weekly bars < 30, weekly_score must be 0 (matches _analyze_one)."""
    daily = _synthetic_history(n=50)  # ~10 weeks → too few
    wf = walk_forward_verdicts(daily, weights, scoring, verdicts_cfg, indicators_cfg)
    assert len(wf) > 0
    assert (wf["weekly_score"] == 0).all()
```

- [ ] **Step 8: Run the warmup tests**

Run: `pytest tests/test_backtest_composite.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/stockpool/backtest_composite.py tests/test_backtest_composite.py
git commit -m "feat(backtest): walk-forward composite verdicts with leakage guards"
```

---

## Task 3: `verdict_bucket_stats` — bucketed forward-return stats for A

**Files:**
- Modify: `src/stockpool/backtest_composite.py` (append)
- Modify: `tests/test_backtest_composite.py` (append)

- [ ] **Step 1: Add failing tests for the bucket stats**

Append to `tests/test_backtest_composite.py`:
```python
from stockpool.backtest_composite import verdict_bucket_stats


def _wf_from_verdicts(verdicts: list[str], closes: list[float]) -> pd.DataFrame:
    """Build a synthetic walk-forward DataFrame from manually-set verdicts."""
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=len(verdicts), freq="B"),
        "close": closes,
        "daily_score": [0] * len(verdicts),
        "weekly_score": [0] * len(verdicts),
        "final_score": [0.0] * len(verdicts),
        "verdict": verdicts,
    })


def test_verdict_bucket_stats_counts():
    wf = _wf_from_verdicts(
        ["buy", "buy", "neutral", "sell", "buy", "neutral", "strong_buy", "strong_sell", "neutral", "neutral"],
        [100, 102, 103, 104, 100, 105, 110, 108, 105, 106],
    )
    stats = verdict_bucket_stats(wf, forward_days=[2])

    assert stats["buy"]["count"] == 3
    assert stats["neutral"]["count"] == 4
    assert stats["sell"]["count"] == 1
    assert stats["strong_buy"]["count"] == 1
    assert stats["strong_sell"]["count"] == 1


def test_verdict_bucket_stats_forward_return_and_win_rate():
    """buy at idx 0 (close 100), idx 1 (close 102), idx 4 (close 100).
    Forward 2 returns: idx 0 → close[2]=103 → +3.0%; idx 1 → close[3]=104 → +1.96%;
    idx 4 → close[6]=110 → +10.0%.
    All positive → win_rate 1.0 (buy wins on positive return).
    Mean ≈ (3.0 + 1.96 + 10.0) / 3 ≈ 4.99%
    """
    wf = _wf_from_verdicts(
        ["buy", "buy", "neutral", "sell", "buy", "neutral", "strong_buy", "strong_sell", "neutral", "neutral"],
        [100, 102, 103, 104, 100, 105, 110, 108, 105, 106],
    )
    stats = verdict_bucket_stats(wf, forward_days=[2])
    buy = stats["buy"]["forward_2"]
    assert buy["sample_size"] == 3
    assert buy["mean_return_pct"] == pytest.approx((3.0 + (104/102 - 1) * 100 + 10.0) / 3, rel=1e-4)
    assert buy["win_rate"] == 1.0


def test_verdict_bucket_stats_sell_win_rate_direction():
    """sell at idx 0 (close 100), close[2]=95 → -5% → win for sell (negative is good)."""
    wf = _wf_from_verdicts(
        ["sell", "neutral", "neutral", "neutral"],
        [100, 99, 95, 96],
    )
    stats = verdict_bucket_stats(wf, forward_days=[2])
    assert stats["sell"]["forward_2"]["win_rate"] == 1.0
    assert stats["sell"]["forward_2"]["mean_return_pct"] == pytest.approx(-5.0, rel=1e-4)


def test_verdict_bucket_stats_omits_out_of_range_forward():
    """Last 2 rows can't have forward_2; sample_size reflects that."""
    wf = _wf_from_verdicts(["buy", "buy", "buy"], [100, 101, 102])
    stats = verdict_bucket_stats(wf, forward_days=[2])
    # Only idx 0 has close[2]=102 available; idx 1 and 2 are out of range.
    assert stats["buy"]["forward_2"]["sample_size"] == 1
```

- [ ] **Step 2: Run the new tests, expect failure**

Run: `pytest tests/test_backtest_composite.py -k bucket -v`
Expected: FAIL — `ImportError: cannot import name 'verdict_bucket_stats'`.

- [ ] **Step 3: Implement `verdict_bucket_stats`**

Append to `src/stockpool/backtest_composite.py`:
```python
_VERDICT_LABELS = ("strong_buy", "buy", "neutral", "sell", "strong_sell")
_BULL_VERDICTS = {"strong_buy", "buy"}
_BEAR_VERDICTS = {"strong_sell", "sell"}


def verdict_bucket_stats(
    wf_df: pd.DataFrame, forward_days: list[int]
) -> dict[str, dict]:
    """For each verdict bucket, aggregate forward-N return stats.

    Returns:
        {
          "strong_buy": {"count": N, "forward_5": {"mean_return_pct", "win_rate", "sample_size"}, ...},
          "buy":        {...},
          "neutral":    {...},
          "sell":       {...},
          "strong_sell":{...},
        }
    Every bucket key is always present, even with count=0.
    """
    closes = wf_df["close"].values
    verdicts = wf_df["verdict"].values

    buckets: dict[str, dict] = {
        label: {"count": 0, "_returns": {n: [] for n in forward_days}}
        for label in _VERDICT_LABELS
    }

    for i in range(len(wf_df)):
        v = verdicts[i]
        if v not in buckets:
            continue
        buckets[v]["count"] += 1
        for n in forward_days:
            j = i + n
            if j >= len(closes):
                continue
            ret_pct = (closes[j] / closes[i] - 1) * 100
            buckets[v]["_returns"][n].append(ret_pct)

    result: dict[str, dict] = {}
    for label, b in buckets.items():
        entry: dict = {"count": b["count"]}
        for n in forward_days:
            rets = b["_returns"][n]
            if rets:
                mean_ret = sum(rets) / len(rets)
                if label in _BULL_VERDICTS:
                    wins = sum(1 for r in rets if r > 0)
                elif label in _BEAR_VERDICTS:
                    wins = sum(1 for r in rets if r < 0)
                else:
                    wins = sum(1 for r in rets if r > 0)
                win_rate = wins / len(rets)
                entry[f"forward_{n}"] = {
                    "mean_return_pct": mean_ret,
                    "win_rate": win_rate,
                    "sample_size": len(rets),
                }
            else:
                entry[f"forward_{n}"] = {
                    "mean_return_pct": 0.0,
                    "win_rate": 0.0,
                    "sample_size": 0,
                }
        result[label] = entry
    return result
```

- [ ] **Step 4: Run all bucket tests, expect pass**

Run: `pytest tests/test_backtest_composite.py -k bucket -v`
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/backtest_composite.py tests/test_backtest_composite.py
git commit -m "feat(backtest): verdict_bucket_stats for composite-score report"
```

---

## Task 4: Integrate A into `_analyze_one` and the per-stock report

**Files:**
- Modify: `src/stockpool/report.py:190-202` (StockAnalysis dataclass)
- Modify: `src/stockpool/report.py:265-302` (stock section HTML)
- Modify: `src/stockpool/cli.py:56-109` (`_analyze_one`)
- Modify: `tests/test_report_smoke.py`

- [ ] **Step 1: Add `verdict_hit_rates` field to `StockAnalysis`**

Modify `src/stockpool/report.py` lines 190-202:
```python
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
```

- [ ] **Step 2: Add a helper that renders the verdict-bucket table**

Append to `src/stockpool/report.py` after `_hit_rate_table` (after line 262):
```python
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
```

- [ ] **Step 3: Render the new table in `_stock_section_html`**

Modify `src/stockpool/report.py` lines 299-300 from:
```python
      <h4>历史命中率(过去 500 日)</h4>
      {_hit_rate_table(a.hit_rates)}
```
to:
```python
      <h4>单信号历史命中率(过去 500 日)</h4>
      {_hit_rate_table(a.hit_rates)}

      <h4>综合评级历史回测(过去 500 日)</h4>
      {_verdict_bucket_table(a.verdict_hit_rates)}
```

- [ ] **Step 4: Wire up `_analyze_one` to compute verdict-bucket stats**

Modify `src/stockpool/cli.py` — update imports near line 15:
```python
from stockpool.backtest import compute_hit_rates
from stockpool.backtest_composite import verdict_bucket_stats, walk_forward_verdicts
```

Modify the body of `_analyze_one` (replace lines 96-109) from:
```python
    try:
        hit_rates = compute_hit_rates(enriched_daily, cfg.weights, cfg.backtest.forward_days)
    except Exception as e:
        warnings.append(f"回测计算失败: {e}")

    return StockAnalysis(
        code=stock.code, name=stock.name,
        daily_score=daily_score, weekly_score=weekly_score,
        final_score=final_score, verdict=verdict,
        triggers_daily=triggers_daily, triggers_weekly=triggers_weekly,
        hit_rates=hit_rates,
        daily_with_indicators=enriched_daily,
        warnings=warnings,
    )
```
to:
```python
    try:
        hit_rates = compute_hit_rates(enriched_daily, cfg.weights, cfg.backtest.forward_days)
    except Exception as e:
        warnings.append(f"单信号回测失败: {e}")

    verdict_hit_rates: dict = {}
    try:
        wf = walk_forward_verdicts(
            daily, cfg.weights, cfg.scoring, cfg.verdicts, cfg.indicators
        )
        verdict_hit_rates = verdict_bucket_stats(wf, cfg.backtest.forward_days)
    except Exception as e:
        warnings.append(f"综合评级回测失败: {e}")

    return StockAnalysis(
        code=stock.code, name=stock.name,
        daily_score=daily_score, weekly_score=weekly_score,
        final_score=final_score, verdict=verdict,
        triggers_daily=triggers_daily, triggers_weekly=triggers_weekly,
        hit_rates=hit_rates,
        verdict_hit_rates=verdict_hit_rates,
        daily_with_indicators=enriched_daily,
        warnings=warnings,
    )
```

- [ ] **Step 5: Extend the report smoke test**

Open `tests/test_report_smoke.py`, find the test that asserts HTML content, and add assertions that the new heading and table render. If unsure, add a new test:
```python
def test_report_includes_verdict_bucket_section():
    from stockpool.report import _stock_section_html, StockAnalysis
    a = StockAnalysis(
        code="000001", name="测试",
        daily_score=2, weekly_score=1, final_score=3.0, verdict="buy",
        verdict_hit_rates={
            "buy": {
                "count": 5,
                "forward_5":  {"mean_return_pct": 1.2, "win_rate": 0.6, "sample_size": 5},
                "forward_10": {"mean_return_pct": 0.0, "win_rate": 0.0, "sample_size": 0},
                "forward_20": {"mean_return_pct": 0.0, "win_rate": 0.0, "sample_size": 0},
            },
        },
    )
    html = _stock_section_html(a, klines_to_show=60)
    assert "综合评级历史回测" in html
    assert "🟢 买入" in html
    assert "+1.20%" in html
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/stockpool/report.py src/stockpool/cli.py tests/test_report_smoke.py
git commit -m "feat(report): embed composite-verdict bucket stats per stock"
```

---

## Task 5: `simulate_equity_curve` and `EquityResult` dataclass

**Files:**
- Modify: `src/stockpool/backtest_composite.py` (append)
- Modify: `tests/test_backtest_composite.py` (append)

- [ ] **Step 1: Add state-machine edge tests**

Append to `tests/test_backtest_composite.py`:
```python
from stockpool.backtest_composite import simulate_equity_curve


def test_simulate_all_neutral_flat_equity():
    wf = _wf_from_verdicts(["neutral"] * 10, [100, 101, 102, 99, 100, 103, 105, 104, 106, 108])
    result = simulate_equity_curve(wf, holding_days_list=[5], with_buy_and_hold=False)
    curve = result.curves[5]
    assert (curve["equity"] == 1.0).all()
    assert result.metrics[5]["trade_count"] == 0


def test_simulate_hold_to_n_exit():
    """Buy at idx 0 (close 100), neutral after, N=3 → exit at idx 3 close 130.
    Equity should be 1.30 by end."""
    closes = [100, 110, 120, 130, 125, 125, 125]
    wf = _wf_from_verdicts(["buy"] + ["neutral"] * 6, closes)
    result = simulate_equity_curve(wf, holding_days_list=[3], with_buy_and_hold=False)
    curve = result.curves[3]
    # Day 0: position[0]=0, equity=1.0
    # Day 1: prev_verdict=buy, flat → long; entry at close[0]=100; equity = 1.0 * (110/100) = 1.10
    # Day 2: held 1 day; equity = 1.10 * (120/110) = 1.20
    # Day 3: held 2 days; equity = 1.20 * (130/120) = 1.30
    # Day 4: held 3 days → exit; position[4]=0; equity stays at 1.30
    assert curve["equity"].iloc[3] == pytest.approx(1.30, rel=1e-6)
    assert curve["equity"].iloc[-1] == pytest.approx(1.30, rel=1e-6)
    assert result.metrics[3]["trade_count"] == 1
    assert result.metrics[3]["win_rate"] == 1.0


def test_simulate_sell_signal_early_exit():
    """Buy at idx 0, sell at idx 2, N=10 → exit on idx 3 (prev_verdict=sell)."""
    closes = [100, 110, 105, 100, 95, 90]
    wf = _wf_from_verdicts(["buy", "neutral", "sell", "neutral", "neutral", "neutral"], closes)
    result = simulate_equity_curve(wf, holding_days_list=[10], with_buy_and_hold=False)
    curve = result.curves[10]
    # Day 3: prev_verdict=sell → exit. Final equity should equal close[2]/close[0]=1.05.
    assert curve["equity"].iloc[3] == pytest.approx(1.05, rel=1e-6)
    assert curve["equity"].iloc[-1] == pytest.approx(1.05, rel=1e-6)
    assert result.metrics[10]["trade_count"] == 1


def test_simulate_buy_while_long_ignored():
    """Second buy signal while already long must not reopen."""
    closes = [100, 110, 110, 110, 110, 100, 100]
    wf = _wf_from_verdicts(
        ["buy", "neutral", "buy", "neutral", "neutral", "neutral", "neutral"], closes
    )
    result = simulate_equity_curve(wf, holding_days_list=[10], with_buy_and_hold=False)
    # Held continuously from day 1 onward (no exit triggered before len-1 because N=10>len).
    assert result.metrics[10]["trade_count"] == 0  # open position at end → not counted


def test_simulate_buy_and_hold_baseline():
    closes = [100, 110, 120, 130]
    wf = _wf_from_verdicts(["neutral"] * 4, closes)
    result = simulate_equity_curve(wf, holding_days_list=[5], with_buy_and_hold=True)
    bh = result.buy_and_hold
    assert bh is not None
    assert bh["equity"].iloc[0] == pytest.approx(1.0)
    assert bh["equity"].iloc[-1] == pytest.approx(1.30)
    assert result.buy_and_hold_metrics["total_return"] == pytest.approx(0.30, rel=1e-6)


def test_simulate_metrics_max_drawdown():
    """Hand-built drawdown: equity 1.0 → 2.0 → 1.0 → 1.5. Max DD = (2.0-1.0)/2.0 = 0.5."""
    closes = [100, 200, 100, 150]
    wf = _wf_from_verdicts(["buy", "neutral", "neutral", "neutral"], closes)
    result = simulate_equity_curve(wf, holding_days_list=[10], with_buy_and_hold=False)
    assert result.metrics[10]["max_drawdown"] == pytest.approx(0.5, rel=1e-6)
```

- [ ] **Step 2: Run the new tests, expect failure**

Run: `pytest tests/test_backtest_composite.py -k simulate -v`
Expected: FAIL — `ImportError: cannot import name 'simulate_equity_curve'`.

- [ ] **Step 3: Implement `EquityResult` and `simulate_equity_curve`**

Append to `src/stockpool/backtest_composite.py`:
```python
@dataclass
class EquityResult:
    """Output of simulate_equity_curve.

    curves[N] -> DataFrame with columns: date, equity, position
    metrics[N] -> dict with total_return, annualized_return, max_drawdown,
                  trade_count, win_rate, avg_trade_return_pct
    buy_and_hold -> DataFrame[date, equity] or None
    buy_and_hold_metrics -> dict (win_rate / avg_trade_return_pct are None)
    """
    curves: dict[int, pd.DataFrame]
    metrics: dict[int, dict]
    buy_and_hold: pd.DataFrame | None = None
    buy_and_hold_metrics: dict | None = None


_TRADING_DAYS_PER_YEAR = 252


def _compute_metrics(equity_series, trades: list[dict]) -> dict:
    eq = equity_series.values
    total_return = float(eq[-1] / eq[0] - 1) if len(eq) > 0 else 0.0
    n_days = len(eq)
    if n_days > 1 and total_return > -1:
        ann = (1 + total_return) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1
    else:
        ann = 0.0

    running_peak = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > running_peak:
            running_peak = v
        dd = (running_peak - v) / running_peak if running_peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    if trades:
        wins = sum(1 for t in trades if t["ret"] > 0)
        win_rate = wins / len(trades)
        avg_trade = sum(t["ret"] for t in trades) / len(trades) * 100
    else:
        win_rate = 0.0
        avg_trade = 0.0

    return {
        "total_return": total_return,
        "annualized_return": float(ann),
        "max_drawdown": float(max_dd),
        "trade_count": len(trades),
        "win_rate": float(win_rate),
        "avg_trade_return_pct": float(avg_trade),
    }


def _simulate_one(wf_df: pd.DataFrame, N: int) -> tuple[pd.DataFrame, dict]:
    closes = wf_df["close"].values
    verdicts = wf_df["verdict"].values
    n = len(wf_df)

    position = [0] * n
    equity = [1.0] * n
    days_held = 0
    entry_idx: int | None = None
    trades: list[dict] = []

    for t in range(1, n):
        prev_v = verdicts[t - 1]

        if position[t - 1] == 0:
            if prev_v in ("buy", "strong_buy"):
                position[t] = 1
                entry_idx = t - 1
                days_held = 0
            else:
                position[t] = 0
        else:
            held_now = days_held + 1
            if held_now >= N or prev_v in ("sell", "strong_sell"):
                position[t] = 0
                exit_idx = t - 1
                ret = closes[exit_idx] / closes[entry_idx] - 1
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": exit_idx, "ret": float(ret),
                })
                entry_idx = None
                days_held = 0
            else:
                position[t] = 1
                days_held = held_now

        daily_ret = closes[t] / closes[t - 1] - 1
        equity[t] = equity[t - 1] * (1 + position[t] * daily_ret)

    curve = pd.DataFrame({
        "date": wf_df["date"].values,
        "equity": equity,
        "position": position,
    })
    return curve, _compute_metrics(curve["equity"], trades)


def _simulate_buy_and_hold(wf_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    closes = wf_df["close"].values
    equity = closes / closes[0]
    curve = pd.DataFrame({"date": wf_df["date"].values, "equity": equity})
    total_return = float(equity[-1] - 1)
    n_days = len(equity)
    ann = (1 + total_return) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1 if n_days > 1 else 0.0

    running_peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > running_peak:
            running_peak = v
        dd = (running_peak - v) / running_peak if running_peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    metrics = {
        "total_return": total_return,
        "annualized_return": float(ann),
        "max_drawdown": float(max_dd),
        "trade_count": 1,
        "win_rate": None,
        "avg_trade_return_pct": None,
    }
    return curve, metrics


def simulate_equity_curve(
    wf_df: pd.DataFrame,
    holding_days_list: list[int],
    with_buy_and_hold: bool = True,
) -> EquityResult:
    """Simulate the B1 strategy for each N in holding_days_list.

    For each N, equity starts at 1.0 on the first walk-forward bar. Decisions
    are made at end-of-day t-1 based on verdict[t-1], realized at close[t-1].
    Long-only; no fees; no T+1; no slippage.
    """
    curves: dict[int, pd.DataFrame] = {}
    metrics: dict[int, dict] = {}
    for N in holding_days_list:
        curve, m = _simulate_one(wf_df, N)
        curves[N] = curve
        metrics[N] = m

    bh_curve = None
    bh_metrics = None
    if with_buy_and_hold and len(wf_df) > 0:
        bh_curve, bh_metrics = _simulate_buy_and_hold(wf_df)

    return EquityResult(
        curves=curves, metrics=metrics,
        buy_and_hold=bh_curve, buy_and_hold_metrics=bh_metrics,
    )
```

- [ ] **Step 4: Run the simulate tests**

Run: `pytest tests/test_backtest_composite.py -k simulate -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/backtest_composite.py tests/test_backtest_composite.py
git commit -m "feat(backtest): equity-curve simulator with B1 trading rule"
```

---

## Task 6: Backtest report renderer (pyecharts + HTML)

**Files:**
- Create: `src/stockpool/backtest_report.py`
- Create: `tests/test_backtest_report.py`

- [ ] **Step 1: Create a smoke test for the renderer**

Create `tests/test_backtest_report.py`:
```python
"""Smoke test for backtest_report.render_backtest_report."""
import pandas as pd

from stockpool.backtest_composite import EquityResult
from stockpool.backtest_report import render_backtest_report


def _result_for(closes: list[float]) -> EquityResult:
    dates = pd.date_range("2026-01-02", periods=len(closes), freq="B")
    curve = pd.DataFrame({
        "date": dates,
        "equity": [c / closes[0] for c in closes],
        "position": [1] * len(closes),
    })
    bh = pd.DataFrame({"date": dates, "equity": [c / closes[0] for c in closes]})
    return EquityResult(
        curves={5: curve, 10: curve, 20: curve},
        metrics={
            N: {
                "total_return": 0.1, "annualized_return": 0.05,
                "max_drawdown": 0.02, "trade_count": 3,
                "win_rate": 0.67, "avg_trade_return_pct": 1.2,
            } for N in (5, 10, 20)
        },
        buy_and_hold=bh,
        buy_and_hold_metrics={
            "total_return": 0.1, "annualized_return": 0.05,
            "max_drawdown": 0.02, "trade_count": 1,
            "win_rate": None, "avg_trade_return_pct": None,
        },
    )


def test_render_backtest_report_smoke(tmp_path):
    closes = [100, 102, 105, 103, 108, 110]
    per_stock = [
        ("605589", "圣泉集团", _result_for(closes)),
        ("603986", "兆易创新", _result_for(closes)),
    ]
    out = render_backtest_report(
        per_stock, run_date="2026-05-17", output_dir=tmp_path
    )
    html = out.read_text(encoding="utf-8")
    assert out.exists()
    assert out.stat().st_size > 1024
    assert "N=5" in html and "N=10" in html and "N=20" in html
    assert "Buy &amp; Hold" in html or "Buy & Hold" in html
    assert "605589" in html and "603986" in html
    assert (tmp_path / "latest.html").exists()
```

- [ ] **Step 2: Run the test, expect failure**

Run: `pytest tests/test_backtest_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockpool.backtest_report'`.

- [ ] **Step 3: Implement `backtest_report.py`**

Create `src/stockpool/backtest_report.py`:
```python
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
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/test_backtest_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/backtest_report.py tests/test_backtest_report.py
git commit -m "feat(report): backtest HTML with equity curves and metrics table"
```

---

## Task 7: New CLI subcommand `backtest`

**Files:**
- Modify: `src/stockpool/cli.py:160-173` (argparse setup)
- Create: `tests/test_cli_backtest.py`

- [ ] **Step 1: Create the CLI smoke test**

Create `tests/test_cli_backtest.py`:
```python
"""Smoke test for `python -m stockpool backtest`."""
from pathlib import Path
import shutil

import pandas as pd
import pytest

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Seed the cache with synthetic daily data so no network call happens."""
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()

    # Build 200 days of synthetic data so weekly bars >= 30
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    returns = rng.normal(0.0005, 0.02, n)
    close = 100.0 * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "open":  close * 0.998,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
    })
    df.to_parquet(cache_dir / "605589_daily.parquet", index=False)
    return cache_dir


def test_backtest_cli_produces_html(tmp_path, isolated_cache, monkeypatch):
    """End-to-end: backtest CLI produces a non-trivial HTML report."""
    # Build a config pointing at the seeded cache + tmp output dir
    import yaml
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(isolated_cache)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    rc = main(["backtest", "--config", str(cfg_file), "--stocks", "605589"])
    assert rc == 0

    backtest_dir = tmp_path / "reports" / "backtest"
    latest = backtest_dir / "latest.html"
    assert latest.exists()
    assert latest.stat().st_size > 1024
    html = latest.read_text(encoding="utf-8")
    assert "N=5" in html and "N=10" in html and "N=20" in html
    assert "605589" in html
```

- [ ] **Step 2: Run the test, expect failure**

Run: `pytest tests/test_cli_backtest.py -v`
Expected: FAIL — `argparse error: invalid choice 'backtest'`.

- [ ] **Step 3: Add `cmd_backtest` and register the subcommand**

Modify `src/stockpool/cli.py`:

Add imports near the top (after existing imports):
```python
from stockpool.backtest_composite import simulate_equity_curve, walk_forward_verdicts
from stockpool.backtest_report import render_backtest_report
```

Add a new function before `def main(...)` (before line 160):
```python
def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    run_date = date.today().isoformat()
    backtest_root = Path(cfg.report.output_dir) / "backtest"
    _setup_logging(backtest_root / run_date)
    log.info("stockpool backtest v%s starting for %s", __version__, run_date)

    stocks = cfg.stocks
    if args.stocks:
        wanted = set(args.stocks.split(","))
        stocks = [s for s in stocks if s.code in wanted]
        if not stocks:
            log.error("No stocks match --stocks filter: %s", args.stocks)
            return 2

    per_stock: list = []
    for s in stocks:
        log.info("Backtesting %s (%s)...", s.code, s.name)
        try:
            daily = fetch_daily(
                s.code, cfg.data.history_days, cfg.data.cache_dir,
                force_refresh=args.refresh,
            )
            wf = walk_forward_verdicts(
                daily, cfg.weights, cfg.scoring, cfg.verdicts, cfg.indicators
            )
            if len(wf) == 0:
                log.warning("%s: insufficient history, skipping", s.code)
                continue
            result = simulate_equity_curve(
                wf,
                holding_days_list=cfg.backtest.equity_curve_holding_days,
                with_buy_and_hold=True,
            )
            per_stock.append((s.code, s.name, result))
        except Exception as e:
            log.error("Backtest failed for %s: %s\n%s", s.code, e, traceback.format_exc())

    if not per_stock:
        log.error("No stocks could be backtested.")
        return 1

    out = render_backtest_report(per_stock, run_date=run_date, output_dir=backtest_root)
    log.info("Backtest report written: %s", out)
    log.info("Latest also at: %s", backtest_root / "latest.html")
    return 0
```

Modify `def main(...)` to register the subcommand. After the existing `p_run.set_defaults(...)` block (around line 170), add:
```python
    p_bt = sub.add_parser("backtest", help="Composite-strategy equity-curve backtest")
    p_bt.add_argument("--config", default="config.yaml", help="Path (default: config.yaml)")
    p_bt.add_argument("--refresh", action="store_true", help="Bypass cache, refetch all")
    p_bt.add_argument("--stocks", default="", help="Only run listed codes (comma-separated)")
    p_bt.set_defaults(func=cmd_backtest)
```

- [ ] **Step 4: Run the CLI smoke test**

Run: `pytest tests/test_cli_backtest.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Manual smoke test from the project root**

Run: `python -m stockpool backtest --stocks 605589`
Expected: Exit 0, file `reports/backtest/<today>.html` exists. Open the file in a browser; you should see one stock section, a line chart with 4 series (N=5, N=10, N=20, Buy & Hold), and a metrics table.

- [ ] **Step 7: Commit**

```bash
git add src/stockpool/cli.py tests/test_cli_backtest.py
git commit -m "feat(cli): add 'backtest' subcommand for composite-strategy equity curves"
```

---

## Self-Review Notes

After drafting the plan I reviewed it against the spec:

- **§1 Goals (A and B):** covered by Tasks 4 (A integration) and 5+6+7 (B simulation + renderer + CLI).
- **§2 Module layout:** Tasks 2/3/5 create `backtest_composite.py`; Task 6 creates `backtest_report.py`; Task 4 integrates A; Task 7 adds the CLI.
- **§3 Walk-forward + no-leakage invariants:** Task 2 covers them and has explicit leakage tests at final and middle bars; the week-key cache and `daily.iloc[:i+1]` slicing are both implemented and tested.
- **§3.3 Warmup:** Task 2 step 7 tests both short-history (returns empty) and "few weekly bars" (weekly_score = 0).
- **§4 A — bucket stats:** Task 3 covers the math; Task 4 covers report integration.
- **§5 B — state machine and metrics:** Task 5 tests all five edge cases listed in the spec.
- **§5.4 Buy-and-hold:** Task 5 tests it; Task 6 renders `—` for win-rate/avg-trade per the spec.
- **§6 CLI:** Task 7 covers it; trading-day check is correctly omitted.
- **§7 Testing:** every named test in §7 has a concrete code block in the plan.
- **§8 Non-goals:** intentionally not in the plan.

No placeholders, all step bodies are concrete code/commands. Function/method names are consistent across tasks (`walk_forward_verdicts`, `verdict_bucket_stats`, `simulate_equity_curve`, `EquityResult`, `render_backtest_report`, `cmd_backtest`).
