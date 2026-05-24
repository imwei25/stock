# F1 plan-2 — Custom factors + WQ101 ranking + A/B 验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 3 个 A 股 custom 因子(`industry_relative_strength` / `limit_up_count` / `turnover_zscore`),把 sector_map 从 `factors/wq101.py` 提升为 `factors/context.py` 共享上下文,跑全市场 ranking 出 `selection.json`,通过 A/B 决定是否更新 `config.yaml` 默认 `factors` 列表。

**Architecture:** sector_map 通过新的 `factors/context.py` 模块以 ClassVar + `set_sector_map`/`get_sector_map` 注入,WQ101 兼容 re-export 旧 import 路径。3 个 custom 因子在 `factors/custom.py` 注册(`sources=("custom",)`)。5 处入口前调 `set_sector_map`(`cli factors analyze` / `prepare_pool` / `build_factor_panel` / `recommend_pool` / 测试 fixture)。A/B 在 `stockpool ab` 工具上以 `docs/ab_runs/p5_1_old8_vs_top20.yaml` 跑 baseline(8 手挑因子)vs candidate(`factors_file: reports/selection.json`)。

**Tech Stack:** pandas / numpy 现有,无新依赖。`industry_map.py`(baostock 证监会分类,30 日缓存)、`factors_analysis.py` + CLI 子命令 + pyecharts(F1 plan-1 已就绪)。

**Spec:** `docs/superpowers/specs/2026-05-24-f1-plan-2-custom-factors-and-ranking-design.md`

---

## File Structure

**Create:**
- `src/stockpool/factors/context.py` — 共享 sector_map ClassVar + set/get + `indneutralize_with_context` helper
- `src/stockpool/factors/custom.py` — 3 个 A 股 custom 因子
- `tests/test_factors_context.py` — context 行为 + wq101 re-export 兼容
- `tests/test_factors_custom.py` — 3 因子数值正确 + look-ahead + 注册
- `docs/ab_runs/p5_1_old8_vs_top20.yaml` — A/B 配置

**Modify:**
- `src/stockpool/factors/wq101.py` — 删除 `_Wq101Context` / 本地 `set_sector_map` / `_indneutralize`,改为从 `factors.context` import + re-export
- `src/stockpool/factors/__init__.py` — 加 `from stockpool.factors import custom` 触发注册
- `src/stockpool/cli.py:cmd_factors_analyze` — build panel 前注入 sector_map
- `src/stockpool/backtest_runner.py:prepare_pool` — 训练前注入 sector_map(ml_factor 路径)
- `src/stockpool/recommend_pool.py:compute_or_load_pool_b` — load industry_map 后调 `set_sector_map`
- `tests/test_factors_analysis.py` — 加一个含 industry_neutral 因子的用例

**Conditional Modify(Step 7 根据 A/B verdict):**
- `config.yaml` — `factors:` 列表 → `factors_file: reports/selection.json`(仅 ✅ pass)
- `CLAUDE.md` — 模块地图 + sweet spot
- `README.md` — factors workflow
- `docs/strategy_improvement_2026.md` §6 — 落到"已完成"行
- `docs/ab_validation_results.md` — 加 P5-1 段
- `reports/factor_analysis/<date>.{json,html}` — 分析存档(任何 verdict 都 commit)
- `reports/selection.json` — top-20(✅ pass / ⚠️ tied 时 commit,❌ regression 时不 commit)
- `reports/ab/p5_1_old8_vs_top20_<date>.html` — A/B 报告

---

## Task 1: Shared sector context + WQ101 refactor

**Files:**
- Create: `src/stockpool/factors/context.py`
- Modify: `src/stockpool/factors/wq101.py:32-52`
- Test: `tests/test_factors_context.py`

- [ ] **Step 1.1: Write the failing test for sector context**

Create `tests/test_factors_context.py`:

```python
"""Tests for shared factor context (sector_map injection)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _reset_sector_map():
    """Isolate ClassVar between tests to avoid pollution."""
    from stockpool.factors.context import set_sector_map
    set_sector_map({})
    yield
    set_sector_map({})


def test_set_get_sector_map_roundtrip():
    from stockpool.factors.context import set_sector_map, get_sector_map
    set_sector_map({"600000": "银行", "000001": "银行"})
    assert get_sector_map() == {"600000": "银行", "000001": "银行"}


def test_get_sector_map_returns_copy():
    """Mutation of returned dict must not affect internal state."""
    from stockpool.factors.context import set_sector_map, get_sector_map
    set_sector_map({"600000": "银行"})
    snapshot = get_sector_map()
    snapshot["FAKE"] = "X"
    assert get_sector_map() == {"600000": "银行"}


def test_empty_sector_map_default():
    from stockpool.factors.context import get_sector_map
    assert get_sector_map() == {}


def test_indneutralize_with_context_empty_map():
    """Empty sector_map → cross-sec demean (subtract daily row mean)."""
    from stockpool.factors.context import indneutralize_with_context
    x = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=pd.date_range("2024-01-01", periods=2),
        columns=["A", "B", "C"],
    )
    out = indneutralize_with_context(x)
    expected = pd.DataFrame(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        index=x.index, columns=x.columns,
    )
    pd.testing.assert_frame_equal(out, expected)


def test_indneutralize_with_context_nonempty():
    """Non-empty sector_map → group demean within each sector."""
    from stockpool.factors.context import set_sector_map, indneutralize_with_context
    set_sector_map({"A": "X", "B": "X", "C": "Y"})
    x = pd.DataFrame(
        [[1.0, 3.0, 10.0]],
        index=pd.date_range("2024-01-01", periods=1),
        columns=["A", "B", "C"],
    )
    out = indneutralize_with_context(x)
    # X-sector mean = (1+3)/2 = 2; Y-sector solo → mean = 10 → demean to 0
    expected = pd.DataFrame(
        [[-1.0, 1.0, 0.0]],
        index=x.index, columns=x.columns,
    )
    pd.testing.assert_frame_equal(out, expected)


def test_wq101_set_sector_map_reexport():
    """Old import path 'from stockpool.factors.wq101 import set_sector_map' still works."""
    from stockpool.factors.wq101 import set_sector_map as wq_set
    from stockpool.factors.context import get_sector_map
    wq_set({"600519": "白酒"})
    assert get_sector_map() == {"600519": "白酒"}
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `python -m pytest tests/test_factors_context.py -v`
Expected: ImportError / ModuleNotFoundError for `stockpool.factors.context`

- [ ] **Step 1.3: Create the context module**

Create `src/stockpool/factors/context.py`:

```python
"""Shared factor context (sector_map et al.).

Lifted from ``factors/wq101.py`` to make sector-aware factors outside wq101
share the same injection point. ``factors/wq101.py`` re-exports
``set_sector_map`` and ``get_sector_map`` for backward compatibility.
"""
from __future__ import annotations

from typing import ClassVar, Mapping

import pandas as pd

from stockpool.factors import ops


class _FactorContext:
    """Module-wide context for sector-aware factors.

    Set via ``set_sector_map`` at the strategy / analysis entry point;
    factors read via ``get_sector_map`` (returns a copy).
    """
    sector_map: ClassVar[dict[str, str]] = {}


def set_sector_map(mapping: Mapping[str, str]) -> None:
    """Inject ``{code: sector_name}`` for downstream factors."""
    _FactorContext.sector_map = dict(mapping)


def get_sector_map() -> dict[str, str]:
    """Return a snapshot of the current sector_map (empty if unset)."""
    return dict(_FactorContext.sector_map)


def indneutralize_with_context(x: pd.DataFrame) -> pd.DataFrame:
    """Industry-neutralise ``x`` (T×N) using current sector context.

    If sector_map is empty, falls back to cross-sectional demean
    (subtract per-day row mean, equivalent to ``ops.cs_demean``).
    """
    if _FactorContext.sector_map:
        return ops.indneutralize(x, _FactorContext.sector_map)
    return ops.cs_demean(x)
```

- [ ] **Step 1.4: Refactor wq101.py to use shared context**

Edit `src/stockpool/factors/wq101.py`. Replace lines 22-52 (imports + `_Wq101Context` + `set_sector_map` + `_indneutralize`) with the following. The rest of the file is unchanged.

```python
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.context import (  # noqa: F401 — re-export for back-compat
    _FactorContext,
    get_sector_map,
    indneutralize_with_context as _indneutralize,
    set_sector_map,
)
from stockpool.factors.registry import register


# Legacy alias for any code that imported _Wq101Context directly.
# All references now go through _FactorContext via factors.context.
_Wq101Context = _FactorContext
```

Note: keep the rest of `wq101.py` (the `WqAlpha` base class and all `Alpha001`..`Alpha101` subclasses, starting at the original line 55+) untouched. The function `_indneutralize` is now imported (it's `indneutralize_with_context` under an alias) and behaves identically.

- [ ] **Step 1.5: Run new tests and full factor test suite to verify no regression**

Run: `python -m pytest tests/test_factors_context.py tests/test_wq101.py tests/test_factors.py -v`
Expected: PASS (all new tests + wq101 unchanged + registry unchanged)

- [ ] **Step 1.6: Commit**

```bash
git add src/stockpool/factors/context.py src/stockpool/factors/wq101.py tests/test_factors_context.py
git commit -m "$(cat <<'EOF'
feat(factors): hoist sector_map to factors.context as shared module

把原本局部于 wq101.py 的 _Wq101Context.sector_map 提升到
factors/context.py,wq101.py 通过 re-export 保留旧 import 路径。
为 F1 plan-2 的 industry_relative_strength custom factor 做准备。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: IndustryRelativeStrengthFactor (custom 因子第一个)

**Files:**
- Create: `src/stockpool/factors/custom.py`
- Modify: `src/stockpool/factors/__init__.py:42-43`
- Test: `tests/test_factors_custom.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_factors_custom.py`:

```python
"""Tests for A-share custom factors."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _reset_sector_map():
    from stockpool.factors.context import set_sector_map
    set_sector_map({})
    yield
    set_sector_map({})


def _make_panel(prices_dict, vols_dict=None):
    """Build minimal OHLCV panel from {code: [close_series]}."""
    codes = list(prices_dict.keys())
    n_bars = len(next(iter(prices_dict.values())))
    dates = pd.date_range("2024-01-01", periods=n_bars)
    close = pd.DataFrame(prices_dict, index=dates)
    volume = (
        pd.DataFrame(vols_dict, index=dates)
        if vols_dict is not None
        else pd.DataFrame(1.0, index=dates, columns=codes)
    )
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": volume,
    }


def test_industry_relative_strength_basic():
    """Factor = own n-day return minus sector median n-day return."""
    from stockpool.factors.context import set_sector_map
    from stockpool.factors.registry import make_factor

    # 2 sectors (X, Y), 3 stocks each. 20-bar prices designed so that
    # at t=20, returns are known.
    n = 20
    base = np.linspace(10.0, 20.0, n + 1)  # +100% over n bars for everyone
    # Stock A (sector X): final price 22 → return = (22/10) - 1 = 1.2
    # Stock B (sector X): final price 20 → return = 1.0
    # Stock C (sector Y): final price 30 → return = 2.0
    # Stock D (sector Y): final price 15 → return = 0.5
    prices = {
        "A": np.r_[base[:-1], [22.0]],
        "B": np.r_[base[:-1], [20.0]],
        "C": np.r_[base[:-1], [30.0]],
        "D": np.r_[base[:-1], [15.0]],
    }
    panel = _make_panel(prices)
    set_sector_map({"A": "X", "B": "X", "C": "Y", "D": "Y"})

    factor = make_factor(f"industry_relative_strength_{n}")
    out = factor.compute(panel)

    last = out.iloc[-1]
    # X median = (1.2 + 1.0) / 2 = 1.1
    # Y median = (2.0 + 0.5) / 2 = 1.25
    assert last["A"] == pytest.approx(1.2 - 1.1)
    assert last["B"] == pytest.approx(1.0 - 1.1)
    assert last["C"] == pytest.approx(2.0 - 1.25)
    assert last["D"] == pytest.approx(0.5 - 1.25)


def test_industry_relative_strength_no_sector_map():
    """Empty sector_map → entire output is NaN."""
    from stockpool.factors.registry import make_factor

    prices = {"A": np.linspace(10, 12, 25), "B": np.linspace(10, 11, 25)}
    panel = _make_panel(prices)

    factor = make_factor("industry_relative_strength_20")
    out = factor.compute(panel)
    assert out.isna().all().all()


def test_industry_relative_strength_singleton_sector():
    """Sector with only 1 stock → that column NaN at last bar."""
    from stockpool.factors.context import set_sector_map
    from stockpool.factors.registry import make_factor

    n = 20
    prices = {
        "A": np.r_[np.linspace(10, 19, n), [22.0]],
        "B": np.r_[np.linspace(10, 19, n), [20.0]],
        "C": np.r_[np.linspace(10, 19, n), [30.0]],  # solo in sector Y
    }
    panel = _make_panel(prices)
    set_sector_map({"A": "X", "B": "X", "C": "Y"})

    factor = make_factor(f"industry_relative_strength_{n}")
    out = factor.compute(panel).iloc[-1]
    assert not np.isnan(out["A"])
    assert not np.isnan(out["B"])
    assert np.isnan(out["C"])


def test_industry_relative_strength_unmapped_stock():
    """Stock not in sector_map → that column NaN at last bar; others unaffected."""
    from stockpool.factors.context import set_sector_map
    from stockpool.factors.registry import make_factor

    n = 20
    prices = {
        "A": np.r_[np.linspace(10, 19, n), [22.0]],
        "B": np.r_[np.linspace(10, 19, n), [20.0]],
        "MISSING": np.r_[np.linspace(10, 19, n), [99.0]],
    }
    panel = _make_panel(prices)
    set_sector_map({"A": "X", "B": "X"})  # MISSING omitted

    factor = make_factor(f"industry_relative_strength_{n}")
    out = factor.compute(panel).iloc[-1]
    assert np.isnan(out["MISSING"])
    assert not np.isnan(out["A"])
    assert not np.isnan(out["B"])


def test_industry_relative_strength_look_ahead():
    """Truncating panel must not change earlier rows."""
    from stockpool.factors.context import set_sector_map
    from stockpool.factors.registry import make_factor

    n = 20
    rng = np.random.RandomState(42)
    prices = {
        "A": np.cumsum(rng.normal(0, 0.5, 50)) + 100,
        "B": np.cumsum(rng.normal(0, 0.5, 50)) + 100,
        "C": np.cumsum(rng.normal(0, 0.5, 50)) + 100,
    }
    panel_full = _make_panel(prices)
    panel_trunc = {k: v.iloc[:-5] for k, v in panel_full.items()}
    set_sector_map({"A": "X", "B": "X", "C": "Y"})

    factor = make_factor(f"industry_relative_strength_{n}")
    full_out = factor.compute(panel_full).iloc[:-5]
    trunc_out = factor.compute(panel_trunc)

    pd.testing.assert_frame_equal(full_out, trunc_out)


def test_industry_relative_strength_registered():
    from stockpool.factors.registry import get_spec
    spec = get_spec("industry_relative_strength")
    assert spec.sources == ("custom",)
    assert "industry_neutral" in spec.types
    assert "momentum" in spec.types
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `python -m pytest tests/test_factors_custom.py -v`
Expected: FAIL — "Factor name not registered: industry_relative_strength"

- [ ] **Step 2.3: Create factors/custom.py with IndustryRelativeStrengthFactor**

Create `src/stockpool/factors/custom.py`:

```python
"""A-share specific custom factors (panel-native).

补 WQ101 没覆盖的 A 股专属信号:同业超额收益、涨停频次、异常活跃度。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.context import get_sector_map
from stockpool.factors.registry import register


@register(
    "industry_relative_strength",
    sources=("custom",),
    types=("momentum", "industry_neutral", "cross_sectional"),
    description="N 日动量减去同行业中位动量 (sector_map 通过 factors.context 注入)",
)
class IndustryRelativeStrengthFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"industry_relative_strength_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ret = close.pct_change(self.n, fill_method=None)  # T×N

        sector_map = get_sector_map()
        if not sector_map:
            return pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)

        # Label every column with its sector ("__unknown__" for codes not in map).
        sector_series = pd.Series(
            {code: sector_map.get(code, "__unknown__") for code in ret.columns},
            name="sector",
        )

        # groupby column → sector; transform within each sector daily.
        groups = ret.T.groupby(sector_series)
        sector_median = groups.transform("median").T          # T×N
        sector_count = groups.transform("count").T            # T×N count of non-NaN
        # singleton sector (count<2 on that day) → NaN to avoid self-minus-self
        sector_median = sector_median.where(sector_count >= 2, np.nan)

        result = ret - sector_median

        # codes not in sector_map → entire column NaN
        unknown_cols = [
            c for c in result.columns
            if sector_map.get(c, "__unknown__") == "__unknown__"
        ]
        if unknown_cols:
            result.loc[:, unknown_cols] = np.nan
        return result
```

- [ ] **Step 2.4: Wire custom module into factors package**

Edit `src/stockpool/factors/__init__.py`. After line 43 (the `import wq101` side-effect line), add:

```python
from stockpool.factors import custom  # noqa: F401
```

So that the section becomes:

```python
# Side-effect: register built-in factors.
from stockpool.factors import technical  # noqa: F401
from stockpool.factors import wq101  # noqa: F401
from stockpool.factors import custom  # noqa: F401
```

- [ ] **Step 2.5: Run tests**

Run: `python -m pytest tests/test_factors_custom.py -v -k industry_relative_strength`
Expected: PASS (5 industry_relative_strength tests + registered test)

- [ ] **Step 2.6: Commit**

```bash
git add src/stockpool/factors/custom.py src/stockpool/factors/__init__.py tests/test_factors_custom.py
git commit -m "$(cat <<'EOF'
feat(factors): add IndustryRelativeStrengthFactor custom factor

A 股专属:N 日动量减去同行业中位动量, sector_map 通过
factors.context.set_sector_map 注入。singleton sector / 未映射股
→ NaN, 空 sector_map → 整列 NaN (退化路径)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: LimitUpCountFactor + TurnoverZScoreFactor

**Files:**
- Modify: `src/stockpool/factors/custom.py` (append two more factor classes)
- Modify: `tests/test_factors_custom.py` (append tests for these two)

- [ ] **Step 3.1: Append failing tests**

Append to `tests/test_factors_custom.py`:

```python
# ── LimitUpCountFactor ──────────────────────────────────────────────────────

def test_limit_up_count_basic():
    """Count of bars with pct_change > 0.099 in rolling N window."""
    from stockpool.factors.registry import make_factor

    # Daily returns design:
    #   bar 0:  NaN (no prev)
    #   bar 1:  +5%  → 0
    #   bar 2:  +10% → 1 (0.10 > 0.099)
    #   bar 3:  +9.9% → 0 (0.099 not strictly > 0.099)
    #   bar 4:  +11% → 1
    #   bar 5:  +9.95% → 1
    # With n=3:
    #   bars 0,1,2 → NaN (warmup, need 3 non-NaN observations)
    #   bar 3: window [bar1, bar2, bar3] is_limit = [0, 1, 0] → sum = 1
    #   bar 4: window [bar2, bar3, bar4] is_limit = [1, 0, 1] → sum = 2
    #   bar 5: window [bar3, bar4, bar5] is_limit = [0, 1, 1] → sum = 2
    rets = [np.nan, 0.05, 0.10, 0.099, 0.11, 0.0995]
    close = [100.0]
    for r in rets[1:]:
        close.append(close[-1] * (1 + r))
    prices = {"A": np.array(close)}
    panel = _make_panel(prices)

    factor = make_factor("limit_up_count_3")
    out = factor.compute(panel)["A"].tolist()
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    assert out[3] == pytest.approx(1.0)
    assert out[4] == pytest.approx(2.0)
    assert out[5] == pytest.approx(2.0)


def test_limit_up_count_warmup_nan():
    """First n bars must be NaN due to rolling min_periods=n."""
    from stockpool.factors.registry import make_factor

    prices = {"A": np.linspace(100, 110, 25)}  # smooth, no limit-ups
    panel = _make_panel(prices)
    factor = make_factor("limit_up_count_20")
    out = factor.compute(panel)["A"]
    # bars [0, 19] are warmup (need 20 non-NaN; bar 0 has NaN pct_change)
    assert out.iloc[:20].isna().all()
    # later bars are valid
    assert out.iloc[20:].notna().all()


def test_limit_up_count_look_ahead():
    from stockpool.factors.registry import make_factor

    rng = np.random.RandomState(7)
    prices = {"A": np.cumsum(rng.normal(0, 1, 50)) + 100}
    panel_full = _make_panel(prices)
    panel_trunc = {k: v.iloc[:-5] for k, v in panel_full.items()}

    factor = make_factor("limit_up_count_20")
    full_out = factor.compute(panel_full).iloc[:-5]
    trunc_out = factor.compute(panel_trunc)
    pd.testing.assert_frame_equal(full_out, trunc_out)


def test_limit_up_count_registered():
    from stockpool.factors.registry import get_spec
    spec = get_spec("limit_up_count")
    assert spec.sources == ("custom",)
    assert "momentum" in spec.types


# ── TurnoverZScoreFactor ────────────────────────────────────────────────────

def test_turnover_zscore_basic():
    """log(volume) z-scored over rolling N=3 window."""
    from stockpool.factors.registry import make_factor

    # Volumes: [1, 1, 1, 100, 1, 1]
    # log:      [0, 0, 0, log(100), 0, 0]
    # At bar 3 (window 1..3 = [0, 0, log(100)]): mean=log(100)/3, std≠0
    vols = [1.0, 1.0, 1.0, 100.0, 1.0, 1.0]
    prices = {"A": np.full(len(vols), 100.0)}
    panel = _make_panel(prices, vols_dict={"A": np.array(vols)})

    factor = make_factor("turnover_zscore_3")
    out = factor.compute(panel)["A"]
    # bars 0..1 NaN (warmup), bar 2 OK (window [0,0,0] → std=0 → NaN via replace)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])  # std=0 → NaN
    # bar 3: log(100) is a positive outlier, z-score > 0
    assert out.iloc[3] > 0


def test_turnover_zscore_zero_volume():
    """volume=0 (suspension) → log undefined → NaN; non-zero rows unaffected."""
    from stockpool.factors.registry import make_factor

    vols = [1.0, 2.0, 0.0, 4.0, 5.0]  # bar 2 suspended
    prices = {"A": np.full(len(vols), 100.0)}
    panel = _make_panel(prices, vols_dict={"A": np.array(vols)})

    factor = make_factor("turnover_zscore_3")
    out = factor.compute(panel)["A"]
    # bar 2 has volume=0 → log → NaN → window calc fails for bars containing it
    assert np.isnan(out.iloc[2])


def test_turnover_zscore_warmup_nan():
    """rolling(60) → first 60 rows NaN."""
    from stockpool.factors.registry import make_factor

    rng = np.random.RandomState(11)
    vols = np.abs(rng.normal(1000, 100, 80))
    prices = {"A": np.full(len(vols), 100.0)}
    panel = _make_panel(prices, vols_dict={"A": vols})

    factor = make_factor("turnover_zscore_60")
    out = factor.compute(panel)["A"]
    assert out.iloc[:60].isna().all()
    # at least some later bars should be finite
    assert out.iloc[60:].notna().any()


def test_turnover_zscore_look_ahead():
    from stockpool.factors.registry import make_factor

    rng = np.random.RandomState(13)
    vols = np.abs(rng.normal(1000, 100, 80))
    prices = {"A": np.full(len(vols), 100.0)}
    panel_full = _make_panel(prices, vols_dict={"A": vols})
    panel_trunc = {k: v.iloc[:-5] for k, v in panel_full.items()}

    factor = make_factor("turnover_zscore_60")
    full_out = factor.compute(panel_full).iloc[:-5]
    trunc_out = factor.compute(panel_trunc)
    pd.testing.assert_frame_equal(full_out, trunc_out)


def test_turnover_zscore_registered():
    from stockpool.factors.registry import get_spec
    spec = get_spec("turnover_zscore")
    assert spec.sources == ("custom",)
    assert "volume" in spec.types
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `python -m pytest tests/test_factors_custom.py -v -k "limit_up or turnover"`
Expected: FAIL — `limit_up_count` and `turnover_zscore` not registered

- [ ] **Step 3.3: Append both factors to custom.py**

Append to `src/stockpool/factors/custom.py`:

```python
@register(
    "limit_up_count",
    sources=("custom",),
    types=("momentum", "time_series"),
    description="近 N 日触及涨停 (close > prev_close × 1.099) 的次数",
)
class LimitUpCountFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"limit_up_count_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ret = close.pct_change(fill_method=None)
        # 主板涨停 10%, 留 0.1% tolerance 免 round-off
        # ST/科创/北交已被 fetch-universe 过滤, 此处不区分
        is_limit_up = (ret > 0.099).astype(float)
        # bar 0's pct_change is NaN → astype(float) → 0.0; force to NaN
        # so rolling.sum with min_periods=n properly warmups
        is_limit_up.iloc[0] = np.nan
        return is_limit_up.rolling(self.n, min_periods=self.n).sum()


@register(
    "turnover_zscore",
    sources=("custom",),
    types=("volume", "time_series"),
    description="log(volume) 的 N 日时间序列 z-score, 反映异常活跃度",
)
class TurnoverZScoreFactor(Factor):
    def __init__(self, n: int = 60):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"turnover_zscore_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.rolling(self.n, min_periods=self.n).mean()
        std = lv.rolling(self.n, min_periods=self.n).std(ddof=0)
        std = std.replace(0.0, np.nan)
        return (lv - mean) / std
```

- [ ] **Step 3.4: Run all custom factor tests**

Run: `python -m pytest tests/test_factors_custom.py -v`
Expected: PASS (all 14+ tests)

- [ ] **Step 3.5: Run full test suite to check no regressions**

Run: `python -m pytest tests/ -q`
Expected: All tests pass (374+ tests)

- [ ] **Step 3.6: Commit**

```bash
git add src/stockpool/factors/custom.py tests/test_factors_custom.py
git commit -m "$(cat <<'EOF'
feat(factors): add LimitUpCountFactor + TurnoverZScoreFactor custom factors

limit_up_count_N: 近 N 日触及涨停 (>=10%) 次数, 用主板 0.099 阈值.
turnover_zscore_N: log(volume) 的 N 日时间序列 z-score, 异常活跃度.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: sector_map injection at 4 entry points

**Files:**
- Modify: `src/stockpool/cli.py:380-420` (cmd_factors_analyze)
- Modify: `src/stockpool/backtest_runner.py:28-80` (prepare_pool)
- Modify: `src/stockpool/recommend_pool.py:96-100` (compute_or_load_pool_b)
- Modify: `tests/test_factors_analysis.py` (add industry-neutral coverage)

- [ ] **Step 4.1: Add coverage test for industry-neutral factor in analyze pipeline**

Append to `tests/test_factors_analysis.py` (one new test function):

```python
def test_analyze_factors_with_industry_neutral_factor():
    """When the factor list contains an industry_neutral factor, callers must
    inject sector_map first via factors.context.set_sector_map."""
    import numpy as np
    import pandas as pd
    from stockpool.factors.context import set_sector_map
    from stockpool.factors_analysis import analyze_factors

    n_bars = 60
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=n_bars)
    codes = ["A", "B", "C", "D"]
    closes = pd.DataFrame(
        {c: 100 + np.cumsum(rng.normal(0, 1, n_bars)) for c in codes},
        index=dates,
    )
    panel = {
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": pd.DataFrame(1000.0, index=dates, columns=codes),
    }
    set_sector_map({"A": "X", "B": "X", "C": "Y", "D": "Y"})

    result = analyze_factors(
        panel=panel,
        factor_names=["industry_relative_strength_20", "momentum_5"],
        horizon=3,
        ic_window=20,
    )
    # Both factors should produce *some* finite daily IC values
    assert "industry_relative_strength_20" in result.daily_ic
    assert result.daily_ic["industry_relative_strength_20"].notna().any()
    set_sector_map({})  # cleanup
```

- [ ] **Step 4.2: Run new test to verify it currently fails (or has incorrect behaviour)**

Run: `python -m pytest tests/test_factors_analysis.py::test_analyze_factors_with_industry_neutral_factor -v`
Expected: PASS now actually — the test only verifies the pipeline runs end-to-end with sector_map set. If existing `analyze_factors` works, this test should already pass. Run it to confirm. If it FAILS, debug before continuing.

(If the test passes immediately, that's fine — it's a regression guard, not a TDD-driven new feature.)

- [ ] **Step 4.3: Inject sector_map in cmd_factors_analyze**

Edit `src/stockpool/cli.py`. After the imports block of `cmd_factors_analyze` (line 386) and before `panel = build_panel_from_cache(...)` (line 418), add the injection. The full modified function section becomes:

Find this in `cmd_factors_analyze`:

```python
    factor_names = list(args.factors) if args.factors else list_factors()
    log.info(
        "Analyzing %d factors over %d stocks (universe=%s)",
        len(factor_names), len(codes), args.universe,
    )

    panel = build_panel_from_cache(codes, cfg.data.history_days, cache_dir)
```

Insert before `panel = build_panel_from_cache(...)`:

```python
    # Sector-aware factors (industry_neutral / industry_relative_strength)
    # need sector_map. Loading is cheap (baostock cache, ~30 day TTL).
    from stockpool.factors.context import set_sector_map
    from stockpool.industry_map import load_or_build_industry_map

    sector_map = load_or_build_industry_map(cache_dir, source="auto")
    if not sector_map:
        log.warning(
            "Industry map unavailable; sector-aware factors will be NaN"
        )
    set_sector_map(sector_map)

```

- [ ] **Step 4.4: Inject sector_map in prepare_pool**

Edit `src/stockpool/backtest_runner.py`. In `prepare_pool`, after the universe + per-stock fetch loop and before `factor_panel = build_factor_panel(...)` (around line 76-78), add:

Find this:

```python
    log.info("Building factor panel over %d stocks × %d factors ...",
             len(pool_data), len(ml_cfg.factors))
    factor_panel = build_factor_panel(ml_cfg.factors, pool_data)
    log.info("Factor panel built: %d factors", len(factor_panel))
    return pool_data, factor_panel
```

Insert before the `log.info("Building factor panel ...")` line:

```python
    # Inject sector_map so industry_neutral factors (WQ101 + custom
    # industry_relative_strength_N) get group context. Empty map → factors
    # fall back to cross-sec demean / NaN.
    from stockpool.factors.context import set_sector_map
    from stockpool.industry_map import load_or_build_industry_map

    sector_map = load_or_build_industry_map(cfg.data.cache_dir, source="auto")
    set_sector_map(sector_map)

```

- [ ] **Step 4.5: Use set_sector_map in compute_or_load_pool_b**

Edit `src/stockpool/recommend_pool.py:96-100`. Find this block:

```python
    industry_map = load_or_build_industry_map(
        cfg.data.cache_dir,
        max_age_days=cfg_pool.industry_map_max_age_days,
        source=cfg_pool.industry_source,
    )
```

Replace with:

```python
    industry_map = load_or_build_industry_map(
        cfg.data.cache_dir,
        max_age_days=cfg_pool.industry_map_max_age_days,
        source=cfg_pool.industry_source,
    )
    # Make the same map available to factors that consume sector context
    # (industry_relative_strength_N + WQ101 indneutralize).
    from stockpool.factors.context import set_sector_map
    set_sector_map(industry_map)
```

- [ ] **Step 4.6: Run targeted + full test suites**

Run: `python -m pytest tests/test_factors_analysis.py tests/test_recommend_pool.py tests/test_ml_strategy_panel.py -v`
Expected: PASS

Run: `python -m pytest tests/ -q`
Expected: All 374+ tests pass

- [ ] **Step 4.7: Commit**

```bash
git add src/stockpool/cli.py src/stockpool/backtest_runner.py src/stockpool/recommend_pool.py tests/test_factors_analysis.py
git commit -m "$(cat <<'EOF'
feat(factors): inject sector_map at 4 entry points

cmd_factors_analyze / prepare_pool / compute_or_load_pool_b 在调任何
因子计算前调用 set_sector_map(industry_map). 这样 industry_relative_strength
+ WQ101 industry_neutral alphas 都能拿到 sector 上下文, 不再退化为
cross-sec demean. + 给 test_factors_analysis 加 industry_neutral 覆盖.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Run factor analysis + pick-by-ic (operational, no new code)

**Files:**
- Create (operational): `reports/factor_analysis/<date>.json`
- Create (operational): `reports/factor_analysis/<date>.html`
- Create (operational): `reports/factor_analysis/latest.html`
- Create (operational): `reports/selection.json`

- [ ] **Step 5.1: Verify fetch-universe cache is ready**

Run: `ls data/universe.parquet && python -c "import pandas as pd; df = pd.read_parquet('data/universe.parquet'); print(f'{len(df)} codes, mtime check OK')"`

Expected: Output like `4350 codes, mtime check OK`.

If file is missing or much older than ~1 month, run first:

```bash
python -m stockpool fetch-universe --workers 8
```

This is a ~1-minute fetch. Skip if cache already fresh.

- [ ] **Step 5.2: Verify sector_map cache (optional, will be lazily fetched)**

Run: `ls data/stock_industry_map.parquet 2>/dev/null && echo OK || echo "will fetch lazily"`

If "will fetch lazily" is printed, the first invocation of `factors analyze` will spend ~5-10s hitting baostock. No action needed.

- [ ] **Step 5.3: Run factors analyze on full universe**

Run:

```bash
python -m stockpool factors analyze \
    --universe all \
    --horizon 3 \
    --ic-window 252 \
    --output reports/factor_analysis
```

Expected: Log lines like:

```
INFO Analyzing 114 factors over ~4350 stocks (universe=all)
INFO Wrote reports/factor_analysis/<today>.json and reports/factor_analysis/<today>.html
```

Runtime: ~3-10 minutes depending on machine. Check the HTML opens correctly:

```bash
ls -la reports/factor_analysis/
```

Should show today's `.json` + `.html` + `latest.html`.

- [ ] **Step 5.4: Run pick-by-ic to generate selection.json**

Run:

```bash
TODAY=$(date +%Y-%m-%d)
python -m stockpool factors pick-by-ic \
    --input reports/factor_analysis/${TODAY}.json \
    --output reports/selection.json \
    --top-n 20 --max-corr 0.6 --min-ir 0.05
```

Expected: A printed list of top-20 factor names + `reports/selection.json` written.

- [ ] **Step 5.5: Inspect the selection**

Run:

```bash
python -c "import json; print(json.dumps(json.load(open('reports/selection.json')), indent=2, ensure_ascii=False))" | head -40
```

Sanity checks:
- 20 factors listed
- At least 1 custom factor (`industry_relative_strength_*` / `limit_up_count_*` / `turnover_zscore_*`) appears (if none do, that's a finding — note in §5.7)
- Mix of wq101 + builtin technical (not all from one source)
- `metadata.universe == "all"`, `metadata.top_n == 20`

- [ ] **Step 5.6: Save artefacts (uncommitted; decision deferred to Task 7)**

The files exist on disk but **do not commit yet**. Task 7 decides commit policy based on A/B verdict.

- [ ] **Step 5.7: Record observations**

Note in your local notes (will fold into commit message later):
- Total runtime
- Number of custom factors in top-20 (0/1/2/3)
- Top-5 factor names and their IC IR
- Any factors that errored / returned all NaN (look in stderr log)

No commit at this step.

---

## Task 6: Run A/B baseline vs candidate

**Files:**
- Create: `docs/ab_runs/p5_1_old8_vs_top20.yaml`
- Create (operational): `reports/ab/p5_1_old8_vs_top20_<date>.html`

- [ ] **Step 6.1: Create A/B config**

Create `docs/ab_runs/p5_1_old8_vs_top20.yaml`:

```yaml
# P5-1 A/B: 当前 8 个手挑因子 vs factor-analysis 选出的 top-20
# Spec: docs/superpowers/specs/2026-05-24-f1-plan-2-custom-factors-and-ranking-design.md
base_config: config.yaml

arms:
  baseline:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j

  candidate:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
```

- [ ] **Step 6.2: Smoke-test each arm independently (optional but recommended)**

Run baseline arm alone first to confirm it builds:

```bash
python -m stockpool ab --config docs/ab_runs/p5_1_old8_vs_top20.yaml --arm baseline
```

Expected: console-only output (per-stock metrics), no HTML. No traceback.

Then candidate:

```bash
python -m stockpool ab --config docs/ab_runs/p5_1_old8_vs_top20.yaml --arm candidate
```

Expected: same. If candidate fails to resolve `factors_file`, verify path is correct relative to repo root.

- [ ] **Step 6.3: Run full A/B**

Run:

```bash
python -m stockpool ab --config docs/ab_runs/p5_1_old8_vs_top20.yaml
```

Expected log lines:

```
INFO Running A/B with arms: ['baseline', 'candidate']
INFO Arm baseline: ... per-stock complete
INFO Arm candidate: ... per-stock complete
INFO Wrote reports/ab/p5_1_old8_vs_top20_<today>.html
```

Open the HTML and read the aggregate metrics table.

- [ ] **Step 6.4: Compute verdict per §3.2 decision gate**

From the aggregate table, record:
- `sharpe_baseline`, `sharpe_candidate`
- `annual_return_baseline`, `annual_return_candidate`
- Compute: `r = sharpe_candidate / sharpe_baseline`
- Compute: `Δreturn = annual_return_candidate - annual_return_baseline`

Apply the decision gate from the spec (§3.2):

| Condition | Verdict |
|---|---|
| `r ≥ 1.10` or `Δreturn ≥ +2pp` | ✅ pass target |
| `r ∈ [0.80, 1.10)` and `|Δreturn| < 2pp` | ⚠️ tied |
| `r < 0.80` or `Δreturn ≤ -2pp` | ❌ regression |

Edge cases:
- If `sharpe_baseline ≤ 0`: use absolute Δsharpe (`+0.10` = pass, `-0.10` = regression, middle = tied)
- If sharpe and return conflict: sharpe wins (sharpe is risk-adjusted)

Record the verdict (✅ / ⚠️ / ❌) and proceed to Task 7.

- [ ] **Step 6.5: No commit yet**

Task 7 handles commits conditionally on the verdict.

---

## Task 7: Conditional landing (depends on §6.4 verdict)

**Files (verdict-dependent):**
- `docs/ab_runs/p5_1_old8_vs_top20.yaml` (always commit)
- `reports/factor_analysis/<date>.{json,html}` (always commit)
- `reports/ab/p5_1_old8_vs_top20_<date>.html` (always commit)
- `reports/selection.json` (commit if ✅ or ⚠️; **omit** if ❌)
- `config.yaml` (modify only if ✅)
- `CLAUDE.md` (modify all 3 verdicts; content depends)
- `README.md` (modify if ✅)
- `docs/strategy_improvement_2026.md` (modify all 3 verdicts)
- `docs/ab_validation_results.md` (modify all 3 verdicts)

**Read the relevant subsection (7A / 7B / 7C) and execute only that one.**

### Task 7A: ✅ Pass target

- [ ] **Step 7A.1: Update config.yaml default**

Edit `config.yaml` strategy block. Find:

```yaml
    factors:
      - momentum_20
      - macd_hist
      - rsi_centered_14
      - ma_distance_20
      - vol_ratio_5
      - boll_position_20
      - ma_slope_20_5
      - kdj_j
```

Replace with:

```yaml
    # F1 plan-2 (2026-05-24): top-20 factors auto-selected by `factors analyze`
    # + `pick-by-ic` over the full A-share universe.
    # Override by switching back to inline `factors: [...]` if needed.
    factors_file: reports/selection.json
```

- [ ] **Step 7A.2: Update CLAUDE.md**

Edit `CLAUDE.md`. In the "模块地图" table, add a row for `src/stockpool/factors/context.py` and `src/stockpool/factors/custom.py`. In the "sweet spot" YAML block(if exists in CLAUDE.md), update the `factors:` line to reflect the new default.

Specifically find the "因子库" section and append a paragraph:

```markdown
**Custom 因子** (`factors/custom.py`):
- `industry_relative_strength_N` — N 日动量减去同行业中位动量(sector_map 通过 `factors/context.py` 注入)
- `limit_up_count_N` — 近 N 日触及涨停(>10%)次数,主板 0.099 阈值
- `turnover_zscore_N` — log(volume) 的 N 日时间序列 z-score,异常活跃度

**sector_map 共享**(`factors/context.py`):WQ101 的 `set_sector_map` 已经提升到这里,
4 个入口(`cmd_factors_analyze` / `prepare_pool` / `compute_or_load_pool_b`)在
build panel 前注入 industry_map。
```

- [ ] **Step 7A.3: Update README.md**

In the "因子分析" section, replace the `pick-by-ic` example output instructions with:

```markdown
- 已生成的 `reports/selection.json`(F1 plan-2 跑出,默认已启用)是 114 因子的 top-20。
  想重跑只需 `python -m stockpool factors analyze --universe all` + `factors pick-by-ic`
  即可覆盖。
```

- [ ] **Step 7A.4: Update docs/strategy_improvement_2026.md §6**

In the "✅ 已完成(经 A/B 验证)" table, append a new row:

```markdown
| **F1 plan-2** — Custom factors (industry_relative_strength / limit_up_count / turnover_zscore) + WQ101 ranking + A/B | spec `2026-05-24-f1-plan-2-...`;`reports/selection.json` checked-in,`config.yaml` 默认改 `factors_file` | **✅ pass target (P5-1)** — Δsharpe=<num>, Δreturn=<num> |
```

Update the "当前 sweet spot 默认" YAML block: change the `factors:` line to mention `factors_file: reports/selection.json`.

- [ ] **Step 7A.5: Update docs/ab_validation_results.md**

Append a "P5-1" section with the verdict table format used by existing entries (P3-2, P3-1, etc.). Include the 7-indicator table + verdict text.

- [ ] **Step 7A.6: Commit (single commit)**

```bash
git add config.yaml CLAUDE.md README.md \
        docs/strategy_improvement_2026.md docs/ab_validation_results.md \
        docs/ab_runs/p5_1_old8_vs_top20.yaml \
        reports/factor_analysis/ reports/selection.json reports/ab/

git commit -m "$(cat <<'EOF'
feat(factors): F1 plan-2 — promote top-20 factors_file as ml_factor default

A/B P5-1 ✅ pass target: candidate (selection.json top-20) vs baseline (老 8 个手挑因子)
Δsharpe=<num>, Δreturn=<num>. selection.json checked-in, config.yaml 默认改
factors_file: reports/selection.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `<num>` with the actual deltas from Step 6.4.)

### Task 7B: ⚠️ Tied

- [ ] **Step 7B.1: Do NOT modify config.yaml**

Default remains `factors: [...]`. Skip.

- [ ] **Step 7B.2: Update CLAUDE.md (lighter touch than 7A)**

Add the same "因子库" paragraph as 7A.2 (custom factors + sector_map), but do **not** change the sweet spot YAML block.

- [ ] **Step 7B.3: Update docs/strategy_improvement_2026.md §6**

Append to the "✅ 已完成(经 A/B 验证)" table:

```markdown
| **F1 plan-2** — Custom factors + WQ101 ranking + A/B | spec `2026-05-24-f1-plan-2-...`;selection.json checked-in,默认未改 | **⚠️ tied (P5-1)** — Δsharpe=<num>, Δreturn=<num>; 无害不必要 |
```

- [ ] **Step 7B.4: Update docs/ab_validation_results.md**

Same as 7A.5 — add P5-1 section with tied verdict explained.

- [ ] **Step 7B.5: Commit**

```bash
git add CLAUDE.md docs/strategy_improvement_2026.md docs/ab_validation_results.md \
        docs/ab_runs/p5_1_old8_vs_top20.yaml \
        reports/factor_analysis/ reports/selection.json reports/ab/

git commit -m "$(cat <<'EOF'
feat(factors): F1 plan-2 — custom factors + A/B tied, default unchanged

A/B P5-1 ⚠️ tied: candidate (selection.json top-20) vs baseline (老 8 因子)
Δsharpe=<num>, Δreturn=<num>. selection.json checked-in 作存档但默认仍是
老 8 因子, 沿用 P2-1 embargo 惯例.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7C: ❌ Regression

- [ ] **Step 7C.1: Do NOT commit selection.json**

Remove the local file to avoid accidental future commit:

```bash
rm -f reports/selection.json
```

(The factors and infrastructure code are still useful — they stay. Only the specific top-20 selection is discarded.)

- [ ] **Step 7C.2: Update CLAUDE.md (custom factors note only)**

Same as 7A.2 / 7B.2 — add the custom factors paragraph. Do not touch sweet spot.

- [ ] **Step 7C.3: Update docs/strategy_improvement_2026.md §6**

Move the F1 plan-2 row to the **"🧊 暂搁"** table:

```markdown
| F1 plan-2 top-20 as default | (a) 扩 cfg.stocks 池本身 (16 → 50+) OR (b) 配合 P4 LGB 重启 | P5-1 Δsharpe=<num> (×<r>), Δreturn=<num> |
```

The custom 因子实现本身保留(已经 commit 在 Task 3),只是 top-20 选择被搁置。

- [ ] **Step 7C.4: Update docs/ab_validation_results.md**

Add P5-1 section explaining regression: which factors were in top-20, hypothesized cause (small-pool sample size? cross-sec factor mismatch?), next steps.

- [ ] **Step 7C.5: Commit**

```bash
git add CLAUDE.md docs/strategy_improvement_2026.md docs/ab_validation_results.md \
        docs/ab_runs/p5_1_old8_vs_top20.yaml \
        reports/factor_analysis/ reports/ab/

# Note: NOT adding reports/selection.json (deleted) or config.yaml (unchanged)

git commit -m "$(cat <<'EOF'
docs(factors): F1 plan-2 — A/B regressed, top-20 selection shelved

A/B P5-1 ❌ regression: candidate (selection.json top-20) vs baseline (老 8 因子)
Δsharpe=<num> (×<r>), Δreturn=<num>. selection.json 未 commit 避免被误用为默认.
factors/context.py 提升 + 3 个 custom 因子实现保留 (factors/custom.py), 留作后续
P4 LGB 重启 / 扩股池后可能复用.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Step F.1: Full test suite**

Run: `python -m pytest tests/ -q`
Expected: All tests pass

- [ ] **Step F.2: Sanity check git log**

Run: `git log --oneline -10`

Expected: 4-5 commits from this plan (Task 1, 2, 3, 4 always; Task 7 single commit). No accidental WIP commits.

- [ ] **Step F.3: Sanity check workspace**

Run: `git status`

Expected: clean working tree, no untracked files (apart from anything that was already untracked before this plan started, e.g. the existing `ab.yaml` / `docs/ab_runs/p*.yaml` already in the gitignore-like state).

- [ ] **Step F.4: Final note (optional)**

If verdict was ✅ or ⚠️: Re-run `python -m stockpool run --config config.yaml` quickly to verify the new default config still produces a daily report. (Cheap sanity check, <30s.)

If verdict was ❌: No daily report sanity check needed (config.yaml unchanged).

---

## Self-Review Checklist

The plan author performs this check inline after writing — not a re-pass for the executor.

1. **Spec coverage**: every requirement from `docs/superpowers/specs/2026-05-24-f1-plan-2-custom-factors-and-ranking-design.md` has a task:
   - §2.1 shared context → Task 1 ✓
   - §2.2 custom factors → Tasks 2 + 3 ✓
   - §2.3 injection (4 entry points — spec lists 5 incl. `build_factor_panel` and `factors_analysis.py`; `build_factor_panel` is reached via `prepare_pool` injection, `factors_analysis.py` is reached via `cmd_factors_analyze` injection, so 3 actual code-edit sites cover all 5 spec sites + 1 docstring note. ACK and proceed as written) → Task 4 ✓
   - §2.4 selection.json format → produced by Task 5, not user-authored ✓
   - §3 A/B execution → Tasks 5 + 6 ✓
   - §3.2 decision gate → Step 6.4 ✓
   - §4 tests → Tasks 1.1, 2.1, 3.1, 4.1 ✓
   - §5 landing branches → Tasks 7A / 7B / 7C ✓
   - §6 self-review concerns → addressed by tests `test_wq101_set_sector_map_reexport` (Task 1.1) and the `*_look_ahead` tests (Tasks 2 / 3) ✓

2. **No placeholders**: Step 7A.6 / 7B.5 / 7C.5 contain `<num>` and `<r>` placeholders for verdict-specific values. These are **intentional and required**: the executor fills them after reading Step 6.4 output. Marked explicitly.

3. **Type consistency**: `set_sector_map` and `get_sector_map` signatures are identical across Tasks 1, 2, 4. `IndustryRelativeStrengthFactor` defined in Task 2 used in Task 4.1 test under name `industry_relative_strength_20`. `factor_panel` / `pool_data` interfaces unchanged.
