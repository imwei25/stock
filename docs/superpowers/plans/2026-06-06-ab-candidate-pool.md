# AB Candidate Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stratified ~100-stock candidate pool for AB tests (per-stock and portfolio), sized between `cfg.stocks` and full market, static unless rebuilt by hand, opt-in via `use_ab_pool` flag in `ab.yaml` / `portfolio_ab.yaml`.

**Architecture:** New `ab_pool.py` module persists `data/ab_pool.parquet` from `universe.parquet` + akshare 流通市值 snapshot + industry map + 20-day liquidity. Per-industry top-2 by 流通市值 + top-2 by 20日均额 with row-level merge on overlap. New `ab-pool build/show` CLI subcommand. `ABConfig` / `PortfolioABConfig` gain `use_ab_pool: bool` that swaps stocks list (per-stock) or injects `universe_codes` (portfolio).

**Tech Stack:** Python 3.11+, pandas, pydantic, pyarrow, akshare, baostock (transitively via industry_map), pytest.

**Spec:** `docs/superpowers/specs/2026-06-06-ab-candidate-pool-design.md`

---

## File Structure

**Created:**
- `src/stockpool/ab_pool.py` — `AbPoolConfig` (pydantic), `build_ab_pool`, `load_ab_pool`, internal helpers `_fetch_circ_mv_snapshot`, `_compute_avg_amount_20d`, `_apply_hard_filters`, `_stratified_select`
- `src/stockpool/ab_pool_report.py` — `render_ab_pool_html` (static HTML w/ inline JSON + vanilla JS filter)
- `tests/test_ab_pool.py` — 16 cases covering build / show / yaml integration

**Modified:**
- `src/stockpool/config.py` — add `AbPoolConfig` import path + `AppConfig.ab_pool: AbPoolConfig` field
- `src/stockpool/cli.py` — add `ab-pool` subparser w/ `build` / `show` sub-subcommands + `cmd_ab_pool_build` / `cmd_ab_pool_show`; modify `cmd_ab` to swap stocks when `use_ab_pool`
- `src/stockpool/ab/config.py` — `ABConfig.use_ab_pool: bool = False`; `_resolve_stocks(ab_cfg, base_cfg)` helper; relax stocks_filter membership check to post-swap codes
- `src/stockpool/portfolio_ab/config.py` — `PortfolioABConfig.use_ab_pool: bool = False`; injection logic in `build_effective_cfg` (per-arm explicit `universe_codes` always wins)
- `CLAUDE.md`, `README.md` — quick command, module map, config section, test row entries

---

## Task 1: AbPoolConfig schema + AppConfig wiring

**Files:**
- Create: `src/stockpool/ab_pool.py` (skeleton)
- Modify: `src/stockpool/config.py:586-601` (`AppConfig`)
- Test: `tests/test_ab_pool.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_ab_pool.py`:

```python
"""Tests for stockpool.ab_pool — AB candidate pool build / load / yaml integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from stockpool.ab_pool import AbPoolConfig
from stockpool.config import AppConfig, load_config


def test_ab_pool_config_defaults():
    cfg = AbPoolConfig()
    assert cfg.cache_path == Path("data/ab_pool.parquet")
    assert cfg.industry_source == "auto"
    assert cfg.min_listing_days == 252
    assert cfg.min_avg_amount_20d == 5.0e7
    assert cfg.per_industry_top_mcap == 2
    assert cfg.per_industry_top_liq == 2
    assert cfg.exclude_st is True
    assert cfg.include_unknown_industry is True


def test_ab_pool_config_extra_forbidden():
    with pytest.raises(Exception):  # pydantic ValidationError
        AbPoolConfig(unknown_field=42)


def test_app_config_has_ab_pool_default(tmp_path: Path):
    yaml_text = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert isinstance(cfg.ab_pool, AbPoolConfig)
    assert cfg.ab_pool.cache_path == Path("data/ab_pool.parquet")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: FAIL with `ImportError: cannot import name 'AbPoolConfig' from 'stockpool.ab_pool'`

- [ ] **Step 3: Implement minimal `AbPoolConfig` in new module**

Create `src/stockpool/ab_pool.py`:

```python
"""AB candidate pool — stratified ~100-stock pool for AB tests.

Build: industry-stratified top-2-mcap + top-2-liquidity selection from
universe.parquet, with akshare 流通市值 snapshot. Persisted to
data/ab_pool.parquet; static unless rebuilt by hand.

See docs/superpowers/specs/2026-06-06-ab-candidate-pool-design.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class AbPoolConfig(BaseModel):
    """Build parameters for `python -m stockpool ab-pool build`.

    Defaults reproduce the spec's recipe exactly (28 SW-1 industries × 2 mcap
    + 2 liq ≈ 100 stocks). Section is fully optional in config.yaml.
    """
    model_config = ConfigDict(extra="forbid")

    cache_path: Path = Path("data/ab_pool.parquet")
    industry_source: Literal["auto", "baostock", "akshare"] = "auto"
    min_listing_days: int = 252
    min_avg_amount_20d: float = 5.0e7
    per_industry_top_mcap: int = 2
    per_industry_top_liq: int = 2
    exclude_st: bool = True
    include_unknown_industry: bool = True
```

- [ ] **Step 4: Wire into AppConfig**

Edit `src/stockpool/config.py:586-601` — add import at top of file (find existing pydantic imports area, ~line 6-10):

```python
# Add to imports section
from stockpool.ab_pool import AbPoolConfig
```

⚠ Circular import risk: `ab_pool.py` will eventually import from `config.py` (for AppConfig). Avoid this by NOT importing AppConfig at module top of ab_pool.py; only import inside functions that need it (build_ab_pool will need `cfg: AppConfig` — use TYPE_CHECKING guard).

Modify `AppConfig` class (insert before `content_hash: str = ""`):

```python
class AppConfig(BaseModel):
    """Root config. `content_hash` is set post-load, not in YAML."""
    stocks: list[Stock]
    data: DataConfig
    indicators: IndicatorsConfig
    weights: WeightsConfig
    scoring: ScoringConfig
    verdicts: VerdictsConfig
    backtest: BacktestConfig
    report: ReportConfig
    context: ContextConfig = Field(default_factory=ContextConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    recommend_pool: RecommendPoolConfig = Field(default_factory=RecommendPoolConfig)
    portfolio_backtest: PortfolioBacktestConfig = Field(
        default_factory=PortfolioBacktestConfig,
    )
    ab_pool: AbPoolConfig = Field(default_factory=AbPoolConfig)

    content_hash: str = ""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Run full test suite to check no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -x`
Expected: All existing tests pass (no AppConfig regression from new optional field)

- [ ] **Step 7: Commit**

```bash
git add src/stockpool/ab_pool.py src/stockpool/config.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): add AbPoolConfig schema + AppConfig.ab_pool wiring

Task 1 of AB candidate pool spec. Defaults reproduce stratified 28×4
recipe. Section optional in config.yaml.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `_apply_hard_filters` helper

**Files:**
- Modify: `src/stockpool/ab_pool.py` (add helper)
- Test: `tests/test_ab_pool.py` (add cases)

- [ ] **Step 1: Write failing test**

Append to `tests/test_ab_pool.py`:

```python
import pandas as pd
from datetime import date, timedelta

from stockpool.ab_pool import _apply_hard_filters


def _make_candidate_df(rows: list[dict]) -> pd.DataFrame:
    """Build the input DataFrame shape that _apply_hard_filters expects.

    Columns: code / name / industry / circ_mv / avg_amount_20d / ipo_date
    """
    return pd.DataFrame(rows)


def test_hard_filters_drops_st():
    cfg = AbPoolConfig()
    today = date.today()
    df = _make_candidate_df([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "000001", "name": "ST平安", "industry": "银行",
         "circ_mv": 1e11, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["600519"]


def test_hard_filters_drops_new_ipo():
    cfg = AbPoolConfig()
    today = date.today()
    df = _make_candidate_df([
        {"code": "688001", "name": "新股", "industry": "电子",
         "circ_mv": 1e10, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=100)},  # < 252 days
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["600519"]


def test_hard_filters_drops_illiquid():
    cfg = AbPoolConfig(min_avg_amount_20d=5e7)
    today = date.today()
    df = _make_candidate_df([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "300999", "name": "小盘", "industry": "电子",
         "circ_mv": 5e8, "avg_amount_20d": 1e7,  # below 5e7 floor
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["600519"]


def test_hard_filters_drops_nan_circ_mv():
    cfg = AbPoolConfig()
    today = date.today()
    import numpy as np
    df = _make_candidate_df([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "999999", "name": "无快照", "industry": "未知",
         "circ_mv": np.nan, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["600519"]


def test_hard_filters_st_variants():
    """ST detection should catch ST / *ST / 退 (delisting marker)."""
    cfg = AbPoolConfig()
    today = date.today()
    df = _make_candidate_df([
        {"code": "1", "name": "正常股", "industry": "银行",
         "circ_mv": 1e11, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "2", "name": "ST某某", "industry": "银行",
         "circ_mv": 1e10, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "3", "name": "*ST某某", "industry": "银行",
         "circ_mv": 1e10, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
        {"code": "4", "name": "某某退", "industry": "银行",
         "circ_mv": 1e10, "avg_amount_20d": 1e9,
         "ipo_date": today - timedelta(days=10000)},
    ])
    out = _apply_hard_filters(df, cfg, today=today)
    assert list(out["code"]) == ["1"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py::test_hard_filters_drops_st -v`
Expected: FAIL with `ImportError: cannot import name '_apply_hard_filters'`

- [ ] **Step 3: Implement `_apply_hard_filters`**

Append to `src/stockpool/ab_pool.py`:

```python
from datetime import date as _date

import pandas as pd


def _apply_hard_filters(
    df: pd.DataFrame,
    cfg: AbPoolConfig,
    today: _date | None = None,
) -> pd.DataFrame:
    """Apply pre-stratification hard filters.

    Drops in order:
      1. NaN circ_mv (stock missing from akshare snapshot)
      2. ST / *ST / 退 names (if cfg.exclude_st)
      3. IPO date within min_listing_days
      4. avg_amount_20d below min_avg_amount_20d

    Expects columns: code, name, industry, circ_mv, avg_amount_20d, ipo_date.
    ``today`` is injectable for deterministic tests.
    """
    if today is None:
        today = _date.today()
    out = df.copy()
    out = out[out["circ_mv"].notna()]
    if cfg.exclude_st:
        name_str = out["name"].astype(str)
        is_st = (
            name_str.str.upper().str.contains("ST", na=False)
            | name_str.str.contains("退", na=False)
        )
        out = out[~is_st]
    cutoff = pd.Timestamp(today) - pd.Timedelta(days=cfg.min_listing_days)
    ipo_ts = pd.to_datetime(out["ipo_date"], errors="coerce")
    out = out[ipo_ts <= cutoff]
    out = out[out["avg_amount_20d"] >= cfg.min_avg_amount_20d]
    return out.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: 8 PASSED (3 from Task 1 + 5 new)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ab_pool.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): add _apply_hard_filters helper

Drops NaN mcap / ST / new IPO / illiquid candidates before stratification.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `_stratified_select` helper (industry top-2+2 with overlap merge)

**Files:**
- Modify: `src/stockpool/ab_pool.py`
- Test: `tests/test_ab_pool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ab_pool.py`:

```python
from stockpool.ab_pool import _stratified_select


def test_stratified_no_overlap():
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        # 银行: 4 stocks, mcap-rank distinct from liq-rank
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 1},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 2},
        {"code": "B3", "name": "B3", "industry": "银行", "circ_mv": 1, "avg_amount_20d": 9},
        {"code": "B4", "name": "B4", "industry": "银行", "circ_mv": 2, "avg_amount_20d": 8},
    ])
    out = _stratified_select(df, cfg)
    # Expect 4 rows: top-2 mcap = {B1, B2}, top-2 liq = {B3, B4}, no overlap
    assert set(out["code"]) == {"B1", "B2", "B3", "B4"}
    assert dict(zip(out["code"], out["source_tag"])) == {
        "B1": "mcap", "B2": "mcap", "B3": "liq", "B4": "liq",
    }


def test_stratified_full_overlap_3_rows():
    """Top-2 mcap fully overlaps top-2 liq → 1 shared + bucket yields 3 rows."""
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        # 银行: top-2 by mcap = {B1, B2}; top-2 by liq = {B1, B3}
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 1},
        {"code": "B3", "name": "B3", "industry": "银行", "circ_mv": 1, "avg_amount_20d": 8},
    ])
    out = _stratified_select(df, cfg)
    assert set(out["code"]) == {"B1", "B2", "B3"}
    tags = dict(zip(out["code"], out["source_tag"]))
    assert tags["B1"] == "mcap+liq"
    assert tags["B2"] == "mcap"
    assert tags["B3"] == "liq"


def test_stratified_multiple_industries():
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 8},
        {"code": "F1", "name": "F1", "industry": "食品", "circ_mv": 5, "avg_amount_20d": 5},
        {"code": "F2", "name": "F2", "industry": "食品", "circ_mv": 4, "avg_amount_20d": 4},
    ])
    out = _stratified_select(df, cfg)
    assert set(out["code"]) == {"B1", "B2", "F1", "F2"}
    assert set(out[out["industry"] == "银行"]["code"]) == {"B1", "B2"}
    assert set(out[out["industry"] == "食品"]["code"]) == {"F1", "F2"}


def test_stratified_small_bucket_partial_fill():
    """Bucket with only 1 stock contributes 1 row (no error, no warning escalation)."""
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        {"code": "X1", "name": "X1", "industry": "稀有", "circ_mv": 1, "avg_amount_20d": 1},
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 8},
    ])
    out = _stratified_select(df, cfg)
    assert "X1" in set(out["code"])


def test_stratified_unknown_industry_included():
    cfg = AbPoolConfig(include_unknown_industry=True)
    df = pd.DataFrame([
        {"code": "U1", "name": "U1", "industry": "未知", "circ_mv": 5, "avg_amount_20d": 5},
        {"code": "U2", "name": "U2", "industry": "未知", "circ_mv": 4, "avg_amount_20d": 4},
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
    ])
    out = _stratified_select(df, cfg)
    assert {"U1", "U2"}.issubset(set(out["code"]))


def test_stratified_unknown_industry_excluded():
    cfg = AbPoolConfig(include_unknown_industry=False)
    df = pd.DataFrame([
        {"code": "U1", "name": "U1", "industry": "未知", "circ_mv": 5, "avg_amount_20d": 5},
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
    ])
    out = _stratified_select(df, cfg)
    assert "U1" not in set(out["code"])
    assert "B1" in set(out["code"])


def test_stratified_output_columns():
    cfg = AbPoolConfig()
    df = pd.DataFrame([
        {"code": "B1", "name": "B1", "industry": "银行", "circ_mv": 9, "avg_amount_20d": 9},
        {"code": "B2", "name": "B2", "industry": "银行", "circ_mv": 8, "avg_amount_20d": 8},
    ])
    out = _stratified_select(df, cfg)
    assert set(out.columns) >= {"code", "name", "industry", "circ_mv",
                                "avg_amount_20d", "source_tag"}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k stratified`
Expected: FAIL with `ImportError: cannot import name '_stratified_select'`

- [ ] **Step 3: Implement `_stratified_select`**

Append to `src/stockpool/ab_pool.py`:

```python
import logging

log = logging.getLogger("stockpool")


def _stratified_select(df: pd.DataFrame, cfg: AbPoolConfig) -> pd.DataFrame:
    """Per-industry top-N by 流通市值 ∪ top-N by 20日均额, row-merged on overlap.

    Overlap semantics: a stock that appears in both top lists yields a SINGLE
    output row with source_tag="mcap+liq" (no row duplication). Buckets
    smaller than 2N contribute what they have.

    Skips "未知" bucket entirely when cfg.include_unknown_industry=False.
    """
    rows: list[dict] = []
    for industry, bucket in df.groupby("industry", sort=False):
        if industry == "未知" and not cfg.include_unknown_industry:
            continue
        top_mcap = set(
            bucket.nlargest(cfg.per_industry_top_mcap, "circ_mv")["code"]
        )
        top_liq = set(
            bucket.nlargest(cfg.per_industry_top_liq, "avg_amount_20d")["code"]
        )
        selected = top_mcap | top_liq
        if not selected:
            log.warning("ab_pool: industry %r yielded 0 selections", industry)
            continue
        for r in bucket[bucket["code"].isin(selected)].itertuples(index=False):
            in_mcap = r.code in top_mcap
            in_liq = r.code in top_liq
            tag = "mcap+liq" if (in_mcap and in_liq) else (
                "mcap" if in_mcap else "liq"
            )
            rows.append({
                "code": r.code,
                "name": r.name,
                "industry": industry,
                "circ_mv": r.circ_mv,
                "avg_amount_20d": r.avg_amount_20d,
                "source_tag": tag,
            })
    if not rows:
        return pd.DataFrame(
            columns=["code", "name", "industry", "circ_mv",
                     "avg_amount_20d", "source_tag"]
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: 15 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ab_pool.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): add _stratified_select helper

Per-industry top-2 mcap + top-2 liq with row-level overlap merge
(source_tag='mcap+liq' on intersection, no row duplication).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Data ingest helpers — akshare snapshot + 20d liquidity

**Files:**
- Modify: `src/stockpool/ab_pool.py`
- Test: `tests/test_ab_pool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ab_pool.py`:

```python
from unittest.mock import MagicMock, patch

from stockpool.ab_pool import _fetch_circ_mv_snapshot, _compute_avg_amount_20d


def test_fetch_circ_mv_snapshot_normalizes(monkeypatch):
    """Mock akshare; verify shape: columns code/name/circ_mv (float, yuan)."""
    fake_df = pd.DataFrame({
        "代码": ["600519", "000001"],
        "名称": ["贵州茅台", "平安银行"],
        "流通市值": [2.1e12, 3.2e11],  # akshare already returns yuan
    })
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.return_value = fake_df
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)

    out = _fetch_circ_mv_snapshot()

    assert list(out.columns) == ["code", "name", "circ_mv"]
    assert list(out["code"]) == ["600519", "000001"]
    assert list(out["name"]) == ["贵州茅台", "平安银行"]
    assert out["circ_mv"].dtype.kind == "f"
    assert out["circ_mv"].iloc[0] == pytest.approx(2.1e12)


def test_fetch_circ_mv_snapshot_propagates_error(monkeypatch):
    def raise_err():
        raise RuntimeError("akshare timeout")
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.side_effect = raise_err
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)
    with pytest.raises(RuntimeError, match="akshare"):
        _fetch_circ_mv_snapshot()


def test_compute_avg_amount_20d_basic(tmp_path: Path):
    """Synthesize per-stock parquet, verify avg_amount = mean(vol*close*100) tail-20."""
    cache_dir = tmp_path
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    # vol*close*100 average of last 20 should be 100 * (100 * 10) = 100000
    df = pd.DataFrame({
        "date": dates,
        "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
        "volume": 100.0,
    })
    df.to_parquet(cache_dir / "600519_daily.parquet")

    out = _compute_avg_amount_20d(["600519"], cache_dir)
    assert list(out["code"]) == ["600519"]
    assert out["avg_amount_20d"].iloc[0] == pytest.approx(100.0 * 10.0 * 100)


def test_compute_avg_amount_20d_missing_file_nan(tmp_path: Path):
    """Missing parquet → NaN, not crash."""
    out = _compute_avg_amount_20d(["600519"], tmp_path)
    assert list(out["code"]) == ["600519"]
    import math
    assert math.isnan(out["avg_amount_20d"].iloc[0])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k "fetch_circ or compute_avg"`
Expected: FAIL with `ImportError` on `_fetch_circ_mv_snapshot` / `_compute_avg_amount_20d`

- [ ] **Step 3: Implement helpers**

Append to `src/stockpool/ab_pool.py`:

```python
from pathlib import Path as _Path


def _import_akshare():
    """Indirection seam for tests — mock this function, not `import akshare`."""
    import akshare
    return akshare


def _fetch_circ_mv_snapshot() -> pd.DataFrame:
    """Pull全 A 股流通市值 snapshot from akshare's stock_zh_a_spot_em.

    Returns columns: code (str, 6-digit zero-padded), name (str), circ_mv (yuan).
    Raises on akshare failure — caller decides exit semantics.
    """
    ak = _import_akshare()
    raw = ak.stock_zh_a_spot_em()
    out = pd.DataFrame({
        "code": raw["代码"].astype(str).str.zfill(6),
        "name": raw["名称"].astype(str),
        "circ_mv": pd.to_numeric(raw["流通市值"], errors="coerce"),
    })
    return out


def _compute_avg_amount_20d(
    codes: list[str],
    cache_dir: str | _Path,
) -> pd.DataFrame:
    """For each code, compute mean(volume * close * 100) over last 20 bars.

    Reads from <cache_dir>/<code>_daily.parquet. Missing files yield NaN.
    Returns columns: code, avg_amount_20d.

    Note: mootdx volume unit is 手 (= 100 股), so multiply by 100 to get 元.
    Matches recommend_pool._apply_funnel:172-174.
    """
    cache_dir = _Path(cache_dir)
    rows: list[dict] = []
    for code in codes:
        path = cache_dir / f"{code}_daily.parquet"
        if not path.exists():
            rows.append({"code": code, "avg_amount_20d": float("nan")})
            continue
        try:
            daily = pd.read_parquet(path)
            tail = daily.tail(20)
            avg = float((tail["volume"] * tail["close"] * 100).mean())
        except Exception as e:
            log.warning("ab_pool: avg_amount calc failed for %s (%s)", code, e)
            avg = float("nan")
        rows.append({"code": code, "avg_amount_20d": avg})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k "fetch_circ or compute_avg"`
Expected: 4 PASSED

Run full file: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: 19 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ab_pool.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): add akshare circ_mv snapshot + 20d liquidity helpers

_fetch_circ_mv_snapshot wraps stock_zh_a_spot_em with normalized columns.
_compute_avg_amount_20d reads per-stock parquet cache, NaN-on-miss.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `build_ab_pool` orchestration + `load_ab_pool` reader

**Files:**
- Modify: `src/stockpool/ab_pool.py`
- Test: `tests/test_ab_pool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ab_pool.py`:

```python
from stockpool.ab_pool import build_ab_pool, load_ab_pool


def _stub_app_cfg(tmp_path: Path) -> "AppConfig":
    """Build a minimal AppConfig with cache_dir = tmp_path / 'data'."""
    yaml_text = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    cfg.data.cache_dir = str(data_dir)
    cfg.ab_pool.cache_path = data_dir / "ab_pool.parquet"
    return cfg


def _seed_universe_and_daily(cfg, codes_industries: list[tuple[str, str, str, float]]):
    """Seed universe.parquet + per-stock parquets + industry_map cache.

    Each tuple = (code, name, industry, daily_amount_yuan)
    """
    data_dir = Path(cfg.data.cache_dir)
    universe = pd.DataFrame([
        {"code": c, "name": n, "market": "sh" if c.startswith("6") else "sz"}
        for c, n, _, _ in codes_industries
    ])
    universe.to_parquet(data_dir / "universe.parquet")

    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    for code, _, _, daily_amt in codes_industries:
        # daily_amt = volume * close * 100  →  set volume=daily_amt/(close*100)
        close = 10.0
        volume = daily_amt / (close * 100)
        df = pd.DataFrame({
            "date": dates, "open": close, "high": close, "low": close,
            "close": close, "volume": volume,
        })
        df.to_parquet(data_dir / f"{code}_daily.parquet")

    industry_df = pd.DataFrame([
        {"code": c, "industry": ind}
        for c, _, ind, _ in codes_industries
    ])
    industry_df.to_parquet(data_dir / "stock_industry_map.parquet")


def test_build_basic(tmp_path, monkeypatch):
    cfg = _stub_app_cfg(tmp_path)
    cfg.ab_pool.min_listing_days = 0  # disable IPO filter for synthetic data
    _seed_universe_and_daily(cfg, [
        ("600001", "Bank1", "银行", 1e9),
        ("600002", "Bank2", "银行", 1e9),
        ("600003", "Bank3", "银行", 1e9),
        ("600004", "Bank4", "银行", 1e9),
        ("600005", "Food1", "食品", 1e9),
        ("600006", "Food2", "食品", 1e9),
    ])
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.return_value = pd.DataFrame({
        "代码": ["600001", "600002", "600003", "600004", "600005", "600006"],
        "名称": ["Bank1", "Bank2", "Bank3", "Bank4", "Food1", "Food2"],
        "流通市值": [9e10, 8e10, 7e10, 6e10, 5e10, 4e10],
    })
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)
    monkeypatch.setattr("stockpool.ab_pool._load_industry_map",
                        lambda *_a, **_k: {"600001": "银行", "600002": "银行",
                                            "600003": "银行", "600004": "银行",
                                            "600005": "食品", "600006": "食品"})
    monkeypatch.setattr("stockpool.ab_pool._load_ipo_dates",
                        lambda *_a, **_k: {})

    out_path = build_ab_pool(cfg, refresh=False)

    assert out_path.exists()
    df = load_ab_pool(out_path)
    assert set(df.columns) >= {"code", "name", "industry", "circ_mv",
                                "avg_amount_20d", "source_tag", "build_date"}
    assert set(df["industry"]) == {"银行", "食品"}


def test_build_idempotent_guard(tmp_path, monkeypatch):
    cfg = _stub_app_cfg(tmp_path)
    cfg.ab_pool.cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ab_pool.cache_path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_ab_pool(cfg, refresh=False)
    assert cfg.ab_pool.cache_path.read_bytes() == b"existing"


def test_build_refresh_overwrites(tmp_path, monkeypatch):
    cfg = _stub_app_cfg(tmp_path)
    cfg.ab_pool.min_listing_days = 0
    _seed_universe_and_daily(cfg, [
        ("600001", "Bank1", "银行", 1e9),
        ("600002", "Bank2", "银行", 1e9),
    ])
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.return_value = pd.DataFrame({
        "代码": ["600001", "600002"], "名称": ["Bank1", "Bank2"],
        "流通市值": [9e10, 8e10],
    })
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)
    monkeypatch.setattr("stockpool.ab_pool._load_industry_map",
                        lambda *_a, **_k: {"600001": "银行", "600002": "银行"})
    monkeypatch.setattr("stockpool.ab_pool._load_ipo_dates",
                        lambda *_a, **_k: {})

    cfg.ab_pool.cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ab_pool.cache_path.write_bytes(b"old")
    build_ab_pool(cfg, refresh=True)
    # File should be overwritten with a valid parquet
    df = load_ab_pool(cfg.ab_pool.cache_path)
    assert "600001" in set(df["code"])


def test_build_universe_missing(tmp_path):
    cfg = _stub_app_cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="universe.parquet"):
        build_ab_pool(cfg, refresh=False)


def test_build_all_buckets_empty(tmp_path, monkeypatch):
    cfg = _stub_app_cfg(tmp_path)
    _seed_universe_and_daily(cfg, [("600001", "Only", "银行", 1e3)])  # below floor
    mock_ak = MagicMock()
    mock_ak.stock_zh_a_spot_em.return_value = pd.DataFrame({
        "代码": ["600001"], "名称": ["Only"], "流通市值": [1e10],
    })
    monkeypatch.setattr("stockpool.ab_pool._import_akshare", lambda: mock_ak)
    monkeypatch.setattr("stockpool.ab_pool._load_industry_map",
                        lambda *_a, **_k: {"600001": "银行"})
    monkeypatch.setattr("stockpool.ab_pool._load_ipo_dates",
                        lambda *_a, **_k: {})
    with pytest.raises(RuntimeError, match="empty"):
        build_ab_pool(cfg, refresh=False)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k "test_build_"`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement orchestration**

Append to `src/stockpool/ab_pool.py`:

```python
from datetime import date as _date_today  # already imported above as _date; use that
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stockpool.config import AppConfig


def _load_industry_map(cache_dir: _Path, source: str) -> dict[str, str]:
    """Thin wrapper over industry_map.load_or_build_industry_map for mockability."""
    from stockpool.industry_map import load_or_build_industry_map
    return load_or_build_industry_map(cache_dir=cache_dir, source=source)


def _load_ipo_dates(cache_dir: _Path) -> dict[str, pd.Timestamp]:
    """Thin wrapper over ipo_dates.load_or_build_ipo_dates for mockability."""
    from stockpool.ipo_dates import load_or_build_ipo_dates
    return load_or_build_ipo_dates(cache_dir=cache_dir)


def build_ab_pool(cfg: "AppConfig", refresh: bool = False) -> _Path:
    """Build the AB candidate pool and persist to cfg.ab_pool.cache_path.

    Raises:
      FileNotFoundError — universe.parquet missing
      FileExistsError — cache_path exists without refresh=True
      RuntimeError — akshare snapshot empty / all industry buckets empty

    Returns: the cache_path Path on success.
    """
    out_path = _Path(cfg.ab_pool.cache_path)
    if out_path.exists() and not refresh:
        raise FileExistsError(
            f"{out_path} already exists. Pass --refresh to rebuild."
        )

    cache_dir = _Path(cfg.data.cache_dir)
    universe_path = cache_dir / "universe.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(
            f"{universe_path} not found. Run `python -m stockpool fetch-universe` first."
        )
    universe = pd.read_parquet(universe_path)
    universe["code"] = universe["code"].astype(str).str.zfill(6)

    snapshot = _fetch_circ_mv_snapshot()
    industry = _load_industry_map(cache_dir, cfg.ab_pool.industry_source)
    ipo_dates = _load_ipo_dates(cache_dir)
    liq = _compute_avg_amount_20d(list(universe["code"]), cache_dir)

    # Assemble candidate table — left-join universe ← snapshot ← industry ← ipo ← liq
    candidates = universe[["code", "name"]].merge(
        snapshot[["code", "circ_mv", "name"]].rename(
            columns={"name": "snapshot_name"}
        ),
        on="code", how="left",
    )
    # Prefer akshare name when present (more authoritative for ST tagging)
    candidates["name"] = candidates["snapshot_name"].fillna(candidates["name"])
    candidates = candidates.drop(columns=["snapshot_name"])
    candidates["industry"] = candidates["code"].map(industry).fillna("未知")
    candidates["ipo_date"] = candidates["code"].map(
        lambda c: ipo_dates.get(c, pd.Timestamp("1900-01-01"))
    )
    candidates = candidates.merge(liq, on="code", how="left")

    filtered = _apply_hard_filters(candidates, cfg.ab_pool)
    selected = _stratified_select(filtered, cfg.ab_pool)

    if selected.empty:
        raise RuntimeError(
            "ab_pool: all industry buckets empty after filters — "
            "check liquidity floor / ST filter / IPO cutoff"
        )

    selected = selected.copy()
    selected["build_date"] = _date.today()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out_path, index=False)
    log.info("ab_pool: built %d codes across %d industries → %s",
             len(selected), selected["industry"].nunique(), out_path)
    return out_path


def load_ab_pool(cache_path: str | _Path) -> pd.DataFrame:
    """Read the persisted AB pool parquet. Raises FileNotFoundError if absent."""
    cache_path = _Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Run `python -m stockpool ab-pool build` first."
        )
    return pd.read_parquet(cache_path)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: 24 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ab_pool.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): add build_ab_pool orchestration + load_ab_pool

Wires universe.parquet + akshare snapshot + industry_map + ipo_dates +
20d liquidity through hard filters → stratified selection → parquet.
Idempotent guard, refresh override, empty-bucket error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: HTML renderer (`render_ab_pool_html`)

**Files:**
- Create: `src/stockpool/ab_pool_report.py`
- Test: `tests/test_ab_pool.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_ab_pool.py`:

```python
from stockpool.ab_pool_report import render_ab_pool_html


def test_render_html_smoke(tmp_path):
    df = pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2.1e12, "avg_amount_20d": 5e9,
         "source_tag": "mcap+liq", "build_date": "2026-06-06"},
        {"code": "000001", "name": "平安银行", "industry": "银行",
         "circ_mv": 3e11, "avg_amount_20d": 8e8,
         "source_tag": "liq", "build_date": "2026-06-06"},
    ])
    out_path = tmp_path / "ab_pool.html"
    render_ab_pool_html(df, out_path)
    html = out_path.read_text(encoding="utf-8")

    # Inline JSON data
    assert "POOL_DATA" in html
    assert "600519" in html
    assert "贵州茅台" in html
    # Three filter inputs
    assert 'id="filter-industry"' in html
    assert 'id="filter-code"' in html
    assert 'id="filter-name"' in html
    # Build date footer
    assert "2026-06-06" in html
    # Table header
    assert "代码" in html and "流通市值" in html


def test_render_html_empty_df(tmp_path):
    """Empty df should still produce a valid HTML page."""
    df = pd.DataFrame(columns=["code", "name", "industry", "circ_mv",
                                "avg_amount_20d", "source_tag", "build_date"])
    out_path = tmp_path / "ab_pool.html"
    render_ab_pool_html(df, out_path)
    html = out_path.read_text(encoding="utf-8")
    assert "POOL_DATA" in html
    assert "[]" in html  # empty JSON array
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k render`
Expected: FAIL with ImportError on `ab_pool_report`

- [ ] **Step 3: Implement renderer**

Create `src/stockpool/ab_pool_report.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k render`
Expected: 2 PASSED

Run full file: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: 26 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ab_pool_report.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): add HTML renderer with client-side filtering

Static HTML w/ inline JSON, three filters (industry/code/name) AND-combined,
sortable columns, footer count. No HTTP server, no jinja.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: CLI subcommand `ab-pool build` / `ab-pool show`

**Files:**
- Modify: `src/stockpool/cli.py` (add subparser + commands)
- Test: `tests/test_ab_pool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ab_pool.py`:

```python
import subprocess
import sys


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run `python -m stockpool` with the given args from a tmp cwd."""
    proj_root = Path(__file__).parent.parent
    env = {"PYTHONPATH": str(proj_root / "src")}
    import os
    env.update(os.environ)
    return subprocess.run(
        [sys.executable, "-m", "stockpool", *args],
        cwd=cwd, capture_output=True, text=True, env=env,
    )


def test_cli_ab_pool_build_missing_universe(tmp_path):
    """Build without universe.parquet → exit 1, helpful message."""
    cfg = _stub_app_cfg(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    # Re-dump cfg with updated cache_dir
    import yaml as _yaml
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    res = _run_cli(["ab-pool", "build", "--config", str(cfg_path)], cwd=tmp_path)
    assert res.returncode == 1
    assert "universe.parquet" in (res.stderr + res.stdout)


def test_cli_ab_pool_build_idempotent_guard(tmp_path):
    cfg = _stub_app_cfg(tmp_path)
    cfg.ab_pool.cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ab_pool.cache_path.write_bytes(b"old")
    cfg_path = tmp_path / "config.yaml"
    import yaml as _yaml
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    res = _run_cli(["ab-pool", "build", "--config", str(cfg_path)], cwd=tmp_path)
    assert res.returncode == 1
    assert "--refresh" in (res.stderr + res.stdout)


def test_cli_ab_pool_show_missing_parquet(tmp_path):
    cfg = _stub_app_cfg(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    import yaml as _yaml
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    res = _run_cli(["ab-pool", "show", "--config", str(cfg_path)], cwd=tmp_path)
    assert res.returncode == 1
    assert "ab-pool build" in (res.stderr + res.stdout)


def test_cli_ab_pool_show_renders(tmp_path, monkeypatch):
    """End-to-end: write a parquet directly, call show, assert HTML created."""
    cfg = _stub_app_cfg(tmp_path)
    cfg.ab_pool.cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
                        "circ_mv": 2e12, "avg_amount_20d": 5e9,
                        "source_tag": "mcap+liq", "build_date": "2026-06-06"}])
    df.to_parquet(cfg.ab_pool.cache_path)

    cfg_path = tmp_path / "config.yaml"
    import yaml as _yaml
    _yaml.safe_dump(cfg.model_dump(mode="python"),
                    open(cfg_path, "w", encoding="utf-8"), allow_unicode=True)
    # Disable browser auto-open via env var (will be read by cmd_ab_pool_show)
    import os
    env = {**os.environ, "STOCKPOOL_NO_BROWSER": "1"}
    proj_root = Path(__file__).parent.parent
    env["PYTHONPATH"] = str(proj_root / "src")
    res = subprocess.run(
        [sys.executable, "-m", "stockpool", "ab-pool", "show",
         "--config", str(cfg_path)],
        cwd=tmp_path, capture_output=True, text=True, env=env,
    )
    assert res.returncode == 0, res.stderr
    # Output: reports/ab_pool.html relative to cwd
    out_html = tmp_path / "reports" / "ab_pool.html"
    assert out_html.exists()
    assert "贵州茅台" in out_html.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k "cli_ab_pool"`
Expected: FAIL with exit code != expected or subprocess errors

- [ ] **Step 3: Implement CLI commands**

Edit `src/stockpool/cli.py`. Find the existing imports area (top of file) and add:

```python
from stockpool.ab_pool import build_ab_pool, load_ab_pool
from stockpool.ab_pool_report import render_ab_pool_html
```

Add new command functions (placement: after `cmd_ab` ~line 313, before `cmd_portfolio_backtest` ~line 352):

```python
def cmd_ab_pool_build(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except Exception:
        log.exception("config invalid")
        return 2
    try:
        out_path = build_ab_pool(cfg, refresh=args.refresh)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception:
        log.exception("ab-pool build failed")
        return 2
    df = load_ab_pool(out_path)
    print(f"✓ Built ab_pool: {len(df)} codes across "
          f"{df['industry'].nunique()} industries, saved to {out_path}")
    return 0


def cmd_ab_pool_show(args: argparse.Namespace) -> int:
    import os
    import webbrowser

    try:
        cfg = load_config(args.config)
    except Exception:
        log.exception("config invalid")
        return 2
    try:
        df = load_ab_pool(cfg.ab_pool.cache_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_dir = Path(cfg.report.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ab_pool.html"
    render_ab_pool_html(df, out_path)
    # Also write a "latest" alias
    latest = out_dir / "ab_pool_latest.html"
    latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"AB pool HTML: {out_path}")

    if not os.environ.get("STOCKPOOL_NO_BROWSER"):
        try:
            webbrowser.open(out_path.resolve().as_uri())
        except Exception as e:
            log.warning("webbrowser.open failed: %s", e)
    return 0
```

Add subparser wiring. Find the `p_ab = sub.add_parser("ab", ...)` block (~line 1037) and add the new `ab-pool` subparser nearby:

```python
    p_ab_pool = sub.add_parser(
        "ab-pool",
        help="Build / show the AB candidate pool (stratified ~100-stock pool)",
    )
    ab_pool_sub = p_ab_pool.add_subparsers(dest="ab_pool_action", required=True)

    p_ab_pool_build = ab_pool_sub.add_parser(
        "build", help="Build data/ab_pool.parquet (manual rebuild only)",
    )
    p_ab_pool_build.add_argument("--config", default="config.yaml")
    p_ab_pool_build.add_argument(
        "--refresh", action="store_true",
        help="Overwrite existing ab_pool.parquet",
    )
    p_ab_pool_build.set_defaults(func=cmd_ab_pool_build)

    p_ab_pool_show = ab_pool_sub.add_parser(
        "show", help="Render reports/ab_pool.html and open in browser",
    )
    p_ab_pool_show.add_argument("--config", default="config.yaml")
    p_ab_pool_show.set_defaults(func=cmd_ab_pool_show)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k "cli_ab_pool"`
Expected: 4 PASSED

Run full file: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: 30 PASSED

- [ ] **Step 5: Manual smoke (optional, requires real akshare access)**

Skip this step if network unavailable. From repo root:

```bash
.venv/Scripts/python.exe -m stockpool ab-pool build --config config.yaml
.venv/Scripts/python.exe -m stockpool ab-pool show --config config.yaml
```

Expected: parquet at `data/ab_pool.parquet` (~100 rows); HTML opens in browser.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/cli.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): add ab-pool build/show CLI subcommands

build: idempotent (refuses overwrite without --refresh), exit 1 on
preflight, exit 2 on data source failure. show: renders HTML to
reports/ab_pool.html and opens browser (suppressible via env).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `ab.yaml` integration — `use_ab_pool` flag in per-stock AB

**Files:**
- Modify: `src/stockpool/ab/config.py` (add field + stocks resolver)
- Modify: `src/stockpool/cli.py:313-349` (`cmd_ab` swap point)
- Test: `tests/test_ab_pool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ab_pool.py`:

```python
from stockpool.ab.config import ABConfig, load_ab_config


def _write_ab_yaml(tmp_path: Path, use_ab_pool: bool, stocks_filter: list[str] | None = None):
    base_cfg_text = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_cfg_text, encoding="utf-8")

    ab_text = f"""
base_config: config.yaml
use_ab_pool: {str(use_ab_pool).lower()}
{("stocks_filter: " + repr(stocks_filter)) if stocks_filter else ""}
arms:
  baseline:
    strategy:
      name: composite_verdict
    backtest:
      equity_curve_holding_days: [10]
  challenger:
    strategy:
      name: composite_verdict
    backtest:
      equity_curve_holding_days: [10]
"""
    ab_path = tmp_path / "ab.yaml"
    ab_path.write_text(ab_text, encoding="utf-8")
    return ab_path


def test_ab_config_use_ab_pool_default_false(tmp_path):
    ab_path = _write_ab_yaml(tmp_path, use_ab_pool=False)
    ab_cfg = load_ab_config(ab_path)
    assert ab_cfg.use_ab_pool is False


def test_ab_config_use_ab_pool_true_field(tmp_path):
    ab_path = _write_ab_yaml(tmp_path, use_ab_pool=True)
    # Seed an ab_pool.parquet so load_ab_config doesn't fail on membership check
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 5e9,
         "source_tag": "mcap+liq", "build_date": "2026-06-06"},
    ]).to_parquet(data_dir / "ab_pool.parquet")
    ab_cfg = load_ab_config(ab_path)
    assert ab_cfg.use_ab_pool is True


def test_ab_config_use_ab_pool_missing_parquet_raises(tmp_path):
    ab_path = _write_ab_yaml(tmp_path, use_ab_pool=True)
    # No parquet exists
    with pytest.raises(Exception, match="ab_pool.parquet"):
        load_ab_config(ab_path)


def test_resolve_stocks_use_ab_pool_replaces(tmp_path):
    from stockpool.ab.config import _resolve_stocks
    base_cfg = load_config(tmp_path / "_dummy")  # will set up below
    # Need a proper setup
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    base_cfg.data.cache_dir = str(data_dir)
    base_cfg.ab_pool.cache_path = data_dir / "ab_pool.parquet"

    pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 5e9,
         "source_tag": "mcap+liq", "build_date": "2026-06-06"},
        {"code": "000001", "name": "平安银行", "industry": "银行",
         "circ_mv": 3e11, "avg_amount_20d": 8e8,
         "source_tag": "liq", "build_date": "2026-06-06"},
    ]).to_parquet(base_cfg.ab_pool.cache_path)

    ab_cfg = ABConfig(
        base_config="config.yaml", use_ab_pool=True, stocks_filter=[],
        arms={
            "a": ArmOverride.model_validate({
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10]},
            }),
            "b": ArmOverride.model_validate({
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10]},
            }),
        },
    )
    stocks = _resolve_stocks(ab_cfg, base_cfg)
    assert [s.code for s in stocks] == ["600519", "000001"]
    assert [s.sector for s in stocks] == ["食品饮料", "银行"]


def test_resolve_stocks_filter_intersect_with_ab_pool(tmp_path):
    from stockpool.ab.config import _resolve_stocks
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    base_cfg.data.cache_dir = str(data_dir)
    base_cfg.ab_pool.cache_path = data_dir / "ab_pool.parquet"
    pd.DataFrame([
        {"code": "600519", "name": "贵州茅台", "industry": "食品饮料",
         "circ_mv": 2e12, "avg_amount_20d": 5e9,
         "source_tag": "mcap+liq", "build_date": "2026-06-06"},
        {"code": "000001", "name": "平安银行", "industry": "银行",
         "circ_mv": 3e11, "avg_amount_20d": 8e8,
         "source_tag": "liq", "build_date": "2026-06-06"},
    ]).to_parquet(base_cfg.ab_pool.cache_path)
    ab_cfg = ABConfig(
        base_config="config.yaml", use_ab_pool=True,
        stocks_filter=["600519"],
        arms={
            "a": ArmOverride.model_validate({
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10]},
            }),
            "b": ArmOverride.model_validate({
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10]},
            }),
        },
    )
    stocks = _resolve_stocks(ab_cfg, base_cfg)
    assert [s.code for s in stocks] == ["600519"]
```

Also import `ArmOverride` at top of test file (after existing imports):

```python
from stockpool.ab.config import ABConfig, ArmOverride, load_ab_config, _resolve_stocks
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k "ab_config_ or resolve_stocks"`
Expected: FAIL — `use_ab_pool` is not a field on ABConfig

- [ ] **Step 3: Add `use_ab_pool` + `_resolve_stocks`**

Edit `src/stockpool/ab/config.py`. Change `ABConfig` class (replace lines 58-72):

```python
class ABConfig(BaseModel):
    """Top-level A/B test config (loaded from ab.yaml)."""
    model_config = ConfigDict(extra="forbid")
    base_config: str
    use_ab_pool: bool = False
    stocks_filter: list[str] = Field(default_factory=list)
    arms: dict[str, ArmOverride]

    @field_validator("arms")
    @classmethod
    def _exactly_two(cls, v: dict[str, ArmOverride]) -> dict[str, ArmOverride]:
        if len(v) != 2:
            raise ValueError(
                f"arms must contain exactly 2 entries, got {len(v)}: {list(v)}"
            )
        return v
```

Add `_resolve_stocks` helper (after `build_effective_cfg`, before `load_ab_config`):

```python
def _resolve_stocks(ab_cfg: ABConfig, base_cfg: AppConfig) -> list[Stock]:
    """Return the per-stock iteration list for this AB run.

    Precedence:
      1. If ``ab_cfg.use_ab_pool``, read ab_pool.parquet → synthesize Stock list
         (sector = industry from parquet).
      2. Else use base_cfg.stocks as-is.
      3. Then intersect with ``ab_cfg.stocks_filter`` if non-empty.

    Raises FileNotFoundError if use_ab_pool=True but parquet absent.
    """
    from stockpool.config import Stock as _Stock
    from stockpool.ab_pool import load_ab_pool

    if ab_cfg.use_ab_pool:
        df = load_ab_pool(base_cfg.ab_pool.cache_path)
        stocks = [
            _Stock(code=str(r["code"]).zfill(6), name=str(r["name"]),
                   sector=str(r["industry"]))
            for _, r in df.iterrows()
        ]
    else:
        stocks = list(base_cfg.stocks)

    if ab_cfg.stocks_filter:
        wanted = set(ab_cfg.stocks_filter)
        stocks = [s for s in stocks if s.code in wanted]
    return stocks
```

Also need to import `Stock` at top of ab/config.py (the import from `stockpool.config` already exists — add `Stock`):

```python
# Update existing import (~line 11-17)
from stockpool.config import (
    AppConfig,
    BacktestCostConfig,
    SizingConfig,
    Stock,
    StrategyConfig,
    load_config,
)
```

Update `load_ab_config` (line 110-146) to defer stocks_filter membership when `use_ab_pool`:

```python
def load_ab_config(ab_path: str | Path) -> ABConfig:
    """Load and validate ab.yaml. Performs post-pydantic checks that need
    side info (base config existence, stocks resolution, deep-merge validity).

    Raises pydantic.ValidationError or ValueError on any failure.
    """
    ab_path = Path(ab_path)
    raw = yaml.safe_load(ab_path.read_text(encoding="utf-8"))
    ab_cfg = ABConfig.model_validate(raw)

    base_path = (ab_path.parent / ab_cfg.base_config).resolve()
    if not base_path.exists():
        raise ValueError(
            f"base_config {ab_cfg.base_config!r} (resolved to {base_path}) "
            f"does not exist"
        )

    base_cfg = load_config(base_path)

    # Resolve the effective stocks list (ab_pool swap + stocks_filter)
    # and validate that stocks_filter codes exist in the resolved pool.
    try:
        resolved = _resolve_stocks(ab_cfg, base_cfg)
    except FileNotFoundError as e:
        raise ValueError(
            f"use_ab_pool=true but {e}"
        ) from e

    if ab_cfg.stocks_filter:
        resolved_codes = {s.code for s in resolved}
        # _resolve_stocks already intersected; check whether any filter code
        # didn't make it (i.e., wasn't in the pre-filter pool)
        if ab_cfg.use_ab_pool:
            df = pd.read_parquet(base_cfg.ab_pool.cache_path)
            pool_codes = {str(c).zfill(6) for c in df["code"]}
            unknown = [c for c in ab_cfg.stocks_filter if c not in pool_codes]
        else:
            base_codes = {s.code for s in base_cfg.stocks}
            unknown = [c for c in ab_cfg.stocks_filter if c not in base_codes]
        if unknown:
            raise ValueError(
                f"stocks_filter references codes not in resolved pool: {unknown}"
            )

    for name, arm in ab_cfg.arms.items():
        try:
            build_effective_cfg(base_cfg, arm)
        except ValidationError as e:
            raise ValueError(
                f"arm {name!r} fails effective-config validation: {e}"
            ) from e

    return ab_cfg
```

Add `import pandas as pd` to imports section.

Update `cmd_ab` in `cli.py:313-349` to use `_resolve_stocks`. Replace the `stocks = _apply_stocks_filter(...)` line (~327):

```python
    from stockpool.ab.config import _resolve_stocks
    try:
        stocks = _resolve_stocks(ab_cfg, base_cfg)
    except FileNotFoundError as e:
        log.error("ab pool load failed: %s", e)
        return 2
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v`
Expected: 35 PASSED

Run existing ab tests to confirm no regression:

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab.py tests/test_cli_ab.py -v`
Expected: All existing PASS (use_ab_pool defaults false → behavior identical)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/ab/config.py src/stockpool/cli.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): wire use_ab_pool into per-stock AB

ABConfig.use_ab_pool: bool = False; when true, _resolve_stocks loads
ab_pool.parquet and synthesizes Stock(sector=industry). stocks_filter
intersects with resolved pool. cmd_ab swaps via _resolve_stocks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `portfolio_ab.yaml` integration — `use_ab_pool` for portfolio AB

**Files:**
- Modify: `src/stockpool/portfolio_ab/config.py`
- Test: `tests/test_ab_pool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ab_pool.py`:

```python
from stockpool.portfolio_ab.config import (
    PortfolioABConfig, PortfolioArmOverride, load_portfolio_ab_config,
    build_effective_cfg as portfolio_build_effective_cfg,
)


def _seed_ab_pool(tmp_path: Path, codes: list[str]):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame([
        {"code": c, "name": f"N{c}", "industry": "test",
         "circ_mv": 1e10, "avg_amount_20d": 1e8,
         "source_tag": "mcap", "build_date": "2026-06-06"}
        for c in codes
    ]).to_parquet(data_dir / "ab_pool.parquet")
    return data_dir / "ab_pool.parquet"


def test_portfolio_ab_use_ab_pool_default_false():
    cfg = PortfolioABConfig(
        base_config="config.yaml",
        arms={"a": PortfolioArmOverride(), "b": PortfolioArmOverride()},
    )
    assert cfg.use_ab_pool is False


def test_portfolio_ab_injects_universe_codes(tmp_path):
    """use_ab_pool=true → both arms' effective_cfg.portfolio_backtest.universe_codes
    == parquet codes; training_universe unchanged."""
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    base_cfg.ab_pool.cache_path = _seed_ab_pool(tmp_path, ["600001", "600002"])

    arm = PortfolioArmOverride(
        strategy={"name": "composite_verdict"},
        portfolio_backtest={"enabled": True},
    )
    eff = portfolio_build_effective_cfg(
        base_cfg, arm, use_ab_pool=True,
    )
    assert eff.portfolio_backtest.universe_codes == ["600001", "600002"]
    # Training pool field on ml_factor should be untouched
    assert eff.strategy.ml_factor.training_universe == base_cfg.strategy.ml_factor.training_universe


def test_portfolio_ab_per_arm_override_wins(tmp_path):
    """Per-arm explicit universe_codes wins over use_ab_pool."""
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    base_cfg.ab_pool.cache_path = _seed_ab_pool(tmp_path, ["600001", "600002"])

    arm = PortfolioArmOverride(
        strategy={"name": "composite_verdict"},
        portfolio_backtest={"enabled": True, "universe_codes": ["999999"]},
    )
    eff = portfolio_build_effective_cfg(base_cfg, arm, use_ab_pool=True)
    assert eff.portfolio_backtest.universe_codes == ["999999"]


def test_portfolio_ab_use_ab_pool_missing_parquet_raises(tmp_path):
    base_yaml = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(base_yaml, encoding="utf-8")
    base_cfg = load_config(tmp_path / "config.yaml")
    base_cfg.ab_pool.cache_path = tmp_path / "data" / "ab_pool.parquet"
    arm = PortfolioArmOverride(strategy={"name": "composite_verdict"})
    with pytest.raises(FileNotFoundError, match="ab_pool"):
        portfolio_build_effective_cfg(base_cfg, arm, use_ab_pool=True)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k "portfolio_ab"`
Expected: FAIL — `use_ab_pool` not on PortfolioABConfig; build_effective_cfg lacks `use_ab_pool` kwarg

- [ ] **Step 3: Add `use_ab_pool` + injection**

Edit `src/stockpool/portfolio_ab/config.py`. Update `PortfolioABConfig` class (lines 42-55):

```python
class PortfolioABConfig(BaseModel):
    """Top-level portfolio AB config (loaded from portfolio_ab.yaml)."""
    model_config = ConfigDict(extra="forbid")
    base_config: str
    use_ab_pool: bool = False
    arms: dict[str, PortfolioArmOverride] = Field(..., min_length=2, max_length=2)

    @model_validator(mode="after")
    def _check_arms_count(self) -> "PortfolioABConfig":
        if len(self.arms) != 2:
            raise ValueError(
                f"portfolio AB requires exactly 2 arms, got {len(self.arms)}: "
                f"{list(self.arms)}"
            )
        return self
```

Update `build_effective_cfg` signature and body (lines 70-99):

```python
def build_effective_cfg(
    base: AppConfig,
    arm: PortfolioArmOverride,
    use_ab_pool: bool = False,
) -> AppConfig:
    """Deep-merge an arm's overrides into the base config.

    Rules:
      * ``arm.strategy`` (if set) replaces ``base.strategy`` *wholesale*.
      * ``arm.portfolio_backtest`` (if set) field-merges into
        ``base.portfolio_backtest`` (recursive for nested dicts).
      * If ``use_ab_pool``: inject ab_pool codes into
        ``portfolio_backtest.universe_codes`` unless the merged result
        already has a non-None universe_codes (per-arm override always wins).
      * All other top-level fields pass through unchanged.

    Returns a fresh ``AppConfig`` with ``content_hash`` recomputed from a
    canonical sorted-key yaml dump of the merged dict.
    """
    merged = base.model_dump(mode="python")
    if arm.strategy is not None:
        merged["strategy"] = dict(arm.strategy)
    if arm.portfolio_backtest is not None:
        merged["portfolio_backtest"] = _deep_merge_dict(
            merged.get("portfolio_backtest", {}) or {},
            arm.portfolio_backtest,
        )

    if use_ab_pool:
        from stockpool.ab_pool import load_ab_pool
        pool_df = load_ab_pool(base.ab_pool.cache_path)
        pool_codes = [str(c).zfill(6) for c in pool_df["code"]]
        pb = merged.setdefault("portfolio_backtest", {})
        # Per-arm override wins: only inject if universe_codes is None / absent
        if pb.get("universe_codes") is None:
            pb["universe_codes"] = pool_codes

    out = AppConfig.model_validate(merged)
    canonical = yaml.safe_dump(merged, sort_keys=True).encode("utf-8")
    out.content_hash = hashlib.sha256(canonical).hexdigest()[:8]
    return out
```

Update `load_portfolio_ab_config` to pass `use_ab_pool` to `build_effective_cfg` (lines 122-128):

```python
    for name, arm in ab_cfg.arms.items():
        try:
            build_effective_cfg(base_cfg, arm, use_ab_pool=ab_cfg.use_ab_pool)
        except ValidationError as e:
            raise ValueError(
                f"arm {name!r} fails effective-config validation: {e}"
            ) from e
```

Also update the portfolio_ab runner to pass the flag. Find the runner (`src/stockpool/portfolio_ab/runner.py`) — search for `build_effective_cfg` calls:

```bash
.venv/Scripts/python.exe -c "import subprocess; print(open('src/stockpool/portfolio_ab/runner.py').read().count('build_effective_cfg'))"
```

For each `build_effective_cfg(base_cfg, arm)` call, add `use_ab_pool=ab_cfg.use_ab_pool`:

```python
# in runner.py — replace existing build_effective_cfg(base_cfg, arm) calls
effective = build_effective_cfg(base_cfg, arm, use_ab_pool=ab_cfg.use_ab_pool)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ab_pool.py -v -k "portfolio_ab"`
Expected: 4 PASSED

Run existing portfolio_ab tests:

Run: `.venv/Scripts/python.exe -m pytest tests/test_portfolio_ab_config.py tests/test_portfolio_ab_runner.py tests/test_cli_portfolio_ab.py -v`
Expected: All PASS (default false → unchanged behavior)

Run full suite:

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -x`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/portfolio_ab/config.py src/stockpool/portfolio_ab/runner.py tests/test_ab_pool.py
git commit -m "feat(ab-pool): wire use_ab_pool into portfolio AB

PortfolioABConfig.use_ab_pool: bool = False; when true,
build_effective_cfg injects parquet codes into
portfolio_backtest.universe_codes (per-arm explicit override wins).
Training pool untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md "快速命令"**

In `CLAUDE.md`, find the section ending with `python -m stockpool portfolio-ab` (the portfolio-ab subsection). Add a new block above the test command:

```bash
# AB candidate pool — stratified ~100-stock pool for AB tests (static, manual rebuild)
python -m stockpool ab-pool build [--refresh]
python -m stockpool ab-pool show    # 渲染 reports/ab_pool.html + 浏览器打开
```

- [ ] **Step 2: Update CLAUDE.md "模块地图"**

In the module map table, add new rows (alphabetically near `recommend_pool.py`):

```markdown
| `src/stockpool/ab_pool.py` | **AB 候选池**:行业分层 top-2 mcap + top-2 liq(允许 source_tag 合并的行内 union)+ akshare 流通市值快照 + ipo/流动性硬过滤;`build_ab_pool` 生成 `data/ab_pool.parquet`,`load_ab_pool` 读盘。静态、手动重建 |
| `src/stockpool/ab_pool_report.py` | AB 池 HTML 渲染器(client-side JS 筛选,无 HTTP server) |
```

- [ ] **Step 3: Update CLAUDE.md "配置" 段**

Add a new bullet (near the `recommend_pool` section description):

```markdown
- **`ab_pool`** — AB 候选池构建参数。`cache_path`(`data/ab_pool.parquet`)/ `industry_source`(`auto`/`baostock`/`akshare`)/ `min_listing_days`(252)/ `min_avg_amount_20d`(5e7)/ `per_industry_top_mcap`(2)/ `per_industry_top_liq`(2)/ `exclude_st`(true)/ `include_unknown_industry`(true)。整段可选,默认值复现 spec 28 行业 × 4 配方。前置:必须先 `fetch-universe`。生成命令:`python -m stockpool ab-pool build`
- **AB 池开关**(独立配置文件 `ab.yaml` / `portfolio_ab.yaml`):新增顶层 `use_ab_pool: bool`(默认 false)。`ab.yaml` 设 true 时,per-stock AB 用 ab_pool 替换 `cfg.stocks` 迭代(`stocks_filter` 仍生效,作为子集过滤)。`portfolio_ab.yaml` 设 true 时,把 ab_pool codes 注入到每个 arm 的 `portfolio_backtest.universe_codes`(per-arm 显式 `universe_codes` 仍优先)。训练池(`training_universe`)**不**受影响 — AB 池是"对比所用样本",训练池是"模型学习的横截面",两者解耦。
```

- [ ] **Step 4: Update CLAUDE.md "缓存" 段**

Add to the `data/` cache list:

```markdown
- `ab_pool.parquet` — AB 候选池(`code/name/industry/circ_mv/avg_amount_20d/source_tag/build_date`),`ab-pool build` 生成,静态不变除非 `--refresh`
```

- [ ] **Step 5: Update CLAUDE.md "测试" 表**

Add row:

```markdown
| `test_ab_pool.py` | AB 候选池: AbPoolConfig + 硬过滤(ST/IPO/流动性/NaN) + 行业分层 top-2+2 with overlap merge + akshare snapshot mock + 20d liquidity calc + build orchestration(idempotent guard, refresh, empty buckets)+ HTML 渲染 smoke + CLI build/show smoke + ab.yaml & portfolio_ab.yaml `use_ab_pool` 集成 |
```

- [ ] **Step 6: Update README.md**

In `README.md`, locate the AB testing command section (search for `ab --config`). Insert just above it:

```markdown
### AB 候选池(可选)

per-stock AB 默认在 `cfg.stocks`(几只)上对比,样本太小;全市场又太慢。
中间方案:构建一个 ~100 票的行业分层候选池,AB 对比时通过开关复用。

\```bash
# 一次性构建(需先 fetch-universe)
python -m stockpool ab-pool build
# 浏览器查看池子内容(支持行业/代码/名称筛选)
python -m stockpool ab-pool show
\```

在 `ab.yaml` 或 `portfolio_ab.yaml` 顶层加 `use_ab_pool: true` 即启用。
池子静态、手动重建(`--refresh`),保证历史 AB 结果可复现。
```

(Replace the escaped backticks with real ones — they're escaped here only because this plan doc is markdown.)

- [ ] **Step 7: Run final smoke and full test suite**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: All ~631 tests pass (615 existing + ~16 new).

- [ ] **Step 8: Commit docs**

```bash
git add CLAUDE.md README.md
git commit -m "docs(ab-pool): document ab-pool build/show + use_ab_pool flag

CLAUDE.md: quick command, module map, config section, cache list, test table.
README.md: AB candidate pool section before AB CLI usage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Checklist (for plan author, run before handing off)

- [x] Every task ends with a commit step
- [x] Every code step shows the actual code (no "implement X" without body)
- [x] Test code is concrete (no "write a test for this")
- [x] Spec coverage: AbPoolConfig (T1) ✓ / hard filters (T2) ✓ / stratification (T3) ✓ / akshare + liquidity (T4) ✓ / build + load (T5) ✓ / HTML render (T6) ✓ / CLI build+show (T7) ✓ / per-stock AB integration (T8) ✓ / portfolio AB integration (T9) ✓ / docs (T10) ✓
- [x] Exit codes match spec table: build=0/1/2, show=0/1, runner=2 on missing parquet ✓
- [x] Idempotent guard tested + refresh override tested ✓
- [x] Per-arm explicit `universe_codes` wins over `use_ab_pool` ✓ (T9 test)
- [x] Training pool untouched ✓ (T9 test asserts `training_universe` unchanged)
- [x] `stocks_filter` intersection with ab_pool ✓ (T8 test)
- [x] Naming consistency: `AbPoolConfig`, `build_ab_pool`, `load_ab_pool`, `render_ab_pool_html`, `use_ab_pool`, `_resolve_stocks`, `_apply_hard_filters`, `_stratified_select`, `_fetch_circ_mv_snapshot`, `_compute_avg_amount_20d` — all consistent across tasks ✓
- [x] No placeholders / TBDs / "similar to Task N" references ✓
