# Tradability Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ml_factor 路径上加入论文 B 的可交易性 mask(close-side),通过预 mask panel + NaN 传播让所有因子算子自动剔除涨跌停/停牌/新上市股,并对训练标签做双向 mask 检查。Opt-in,default 关闭,完全向后兼容。

**Architecture:** mask 是独立的 T×N bool DataFrame(不进 OHLCV panel 字段),由 `panel.py:compute_tradability_mask` 从 OHLCV + 板块阈值计算。`apply_mask` 把 mask=False 位置的所有 OHLCV 字段置 NaN。下游算子(WQ101 + technical + custom)通过 NaN 自然传播,无需感知 mask 存在。`ml/dataset.py` 的标签生成把 `y.where(mask & mask.shift(-horizon))` 做双向检查。配置经 `MLFactorConfig.mask` 子段开关,缓存键自动失效。

**Tech Stack:** Python 3.11, pandas, numpy, Pydantic v2, pytest, .venv

**Source Spec:** `docs/superpowers/specs/2026-05-31-tradability-mask-design.md`

---

## File Structure

**Create:**
- `tests/test_panel_mask.py` — mask 计算正确性
- `tests/test_ops_mask_nan_safe.py` — ops.py NaN 容忍度
- `tests/test_ml_strategy_mask.py` — 端到端 mask on/off 行为
- `ab_mask.yaml` — A/B 验证配置

**Modify:**
- `src/stockpool/config.py` — 新增 `MaskConfig`,挂到 `MLFactorConfig.mask`
- `src/stockpool/panel.py` — 新增 `_limit_threshold` / `_listing_mask` / `compute_tradability_mask` / `apply_mask`
- `src/stockpool/factors/ops.py` — 放宽 `ts_sum/ts_mean/ts_std/ts_product` 的 `min_periods`;`decay_linear` NaN-safe 重写
- `src/stockpool/ml/dataset.py` — `compute_factor_panel` / `forward_return_panel` / `build_factor_matrix` / `build_panel` 加 mask 参数
- `src/stockpool/strategy_factory.py` — `build_factor_panel` / `load_or_build_factor_panel` 加 `mask_config` 参数,manifest.json 加 mask 字段
- `src/stockpool/strategy_factory.py:build_strategy` — `MLFactorStrategy` 读 cfg.mask 并下传
- `src/stockpool/cli.py:_prepare_ml_pool` — 传 `mask_config` 给 `load_or_build_factor_panel`
- `CLAUDE.md` — 模块地图 / 配置 / 测试章节
- `README.md` — mask 配置示例

---

## Task 1: Add MaskConfig pydantic model

**Files:**
- Modify: `src/stockpool/config.py`(在 `MLFactorConfig` 之前加新类,在 `MLFactorConfig` 内加字段)
- Test: `tests/test_config.py`(在现有文件加 test)

- [ ] **Step 1: Write failing tests**

加到 `tests/test_config.py` 末尾:

```python
def test_mask_config_defaults():
    from stockpool.config import MaskConfig
    cfg = MaskConfig()
    assert cfg.enabled is False
    assert cfg.limit_up_threshold_main == 0.098
    assert cfg.limit_up_threshold_chinext == 0.198
    assert cfg.limit_up_threshold_bse == 0.298
    assert cfg.min_listing_days == 252


def test_mask_config_extra_forbid():
    from stockpool.config import MaskConfig
    import pytest
    with pytest.raises(Exception):
        MaskConfig(unknown_field=1)


def test_ml_factor_config_has_mask():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig()
    # 默认 mask 关闭
    assert cfg.mask.enabled is False


def test_ml_factor_mask_loaded_from_yaml():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig.model_validate({
        "mask": {"enabled": True, "min_listing_days": 100}
    })
    assert cfg.mask.enabled is True
    assert cfg.mask.min_listing_days == 100
    # 未指定的字段走 default
    assert cfg.mask.limit_up_threshold_main == 0.098


def test_ml_factor_content_hash_changes_with_mask():
    """翻 mask.enabled 改变 content_hash → 缓存失效。"""
    from stockpool.config import AppConfig
    import yaml
    base = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    base["strategy"] = {"name": "ml_factor"}
    cfg_a = AppConfig.model_validate(base)
    base["strategy"]["ml_factor"] = {"mask": {"enabled": True}}
    cfg_b = AppConfig.model_validate(base)
    assert cfg_a.content_hash != cfg_b.content_hash
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py::test_mask_config_defaults tests/test_config.py::test_ml_factor_config_has_mask -v
```
Expected: FAIL with `ImportError: cannot import name 'MaskConfig'` or `AttributeError: ... no attribute 'mask'`

- [ ] **Step 3: Add MaskConfig and wire into MLFactorConfig**

在 `src/stockpool/config.py` `class MLFactorConfig` **之前**插入:

```python
class MaskConfig(BaseModel):
    """Tradability mask for factor input quality (paper B mask-first).

    When enabled, panel values at days when a stock hit limit-up/-down,
    was suspended, or had been listed for <min_listing_days days are
    nulled-out before factor computation. Training labels apply a
    bidirectional check (day t and day t+horizon both unmasked).

    See: docs/superpowers/specs/2026-05-31-tradability-mask-design.md
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    limit_up_threshold_main: float = Field(default=0.098, gt=0, lt=1)
    limit_up_threshold_chinext: float = Field(default=0.198, gt=0, lt=1)
    limit_up_threshold_bse: float = Field(default=0.298, gt=0, lt=1)
    min_listing_days: int = Field(default=252, ge=0)
```

在 `MLFactorConfig` 内,`refresh_verdicts` 字段之后加:

```python
    mask: MaskConfig = Field(default_factory=MaskConfig)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```
Expected: 全部 PASS,新增 5 个 mask 测试都通过。

- [ ] **Step 5: Run full suite to confirm no regression**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: 374 + 5 = 379 PASS, 0 FAIL.

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "feat(mask): add MaskConfig schema, wire into MLFactorConfig (task 1/18)"
```

---

## Task 2: Add `_limit_threshold` helper in panel.py

**Files:**
- Modify: `src/stockpool/panel.py`
- Test: `tests/test_panel_mask.py`(新文件)

- [ ] **Step 1: Write failing tests**

创建 `tests/test_panel_mask.py`:

```python
"""Tests for stockpool.panel mask functions (tradability mask for factor input)."""
import numpy as np
import pandas as pd
import pytest


def test_limit_threshold_main_board():
    from stockpool.panel import _limit_threshold
    # 主板沪
    assert _limit_threshold("600000") == 0.098
    assert _limit_threshold("601398") == 0.098
    assert _limit_threshold("603986") == 0.098
    assert _limit_threshold("605589") == 0.098
    # 主板深 / 中小板
    assert _limit_threshold("000001") == 0.098
    assert _limit_threshold("002001") == 0.098
    assert _limit_threshold("003001") == 0.098


def test_limit_threshold_chinext_star():
    from stockpool.panel import _limit_threshold
    # 创业板 ±20%
    assert _limit_threshold("300001") == 0.198
    assert _limit_threshold("301001") == 0.198
    # 科创板 ±20%
    assert _limit_threshold("688001") == 0.198


def test_limit_threshold_bse():
    from stockpool.panel import _limit_threshold
    # 北交所 ±30%(留兜底)
    assert _limit_threshold("830001") == 0.298
    assert _limit_threshold("870001") == 0.298
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_panel_mask.py -v
```
Expected: FAIL with `ImportError: cannot import name '_limit_threshold'`

- [ ] **Step 3: Implement `_limit_threshold`**

在 `src/stockpool/panel.py` 末尾加:

```python
def _limit_threshold(code: str) -> float:
    """A 股按板块判定涨跌停幅度阈值。

    返回值是"abs 当日 ret 超过它即视为涨/跌停日"的阈值。略小于规则上限
    (0.098 < 0.10)是为了让真实涨停(实际 ret ≈ 0.099 因 round-to-cent)
    也能被命中。

    Args:
        code: 6 位股票代码字符串。
    Returns:
        阈值 ∈ {0.098, 0.198, 0.298},未匹配时回退 0.098。
    """
    if code.startswith(("300", "301", "688")):
        return 0.198  # 创业板 + 科创板 ±20%
    if code.startswith(("82", "83", "87", "43")):
        return 0.298  # 北交所 ±30%(项目 universe 不含,留兜底)
    return 0.098      # 主板沪深 ±10%
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_panel_mask.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/panel.py tests/test_panel_mask.py
git commit -m "feat(mask): add _limit_threshold per-board helper (task 2/18)"
```

---

## Task 3: Add `_listing_mask` helper in panel.py

**Files:**
- Modify: `src/stockpool/panel.py`
- Test: `tests/test_panel_mask.py`(追加)

- [ ] **Step 1: Write failing tests**

追加到 `tests/test_panel_mask.py`:

```python
def test_listing_mask_mature_stock_all_true():
    """Panel 起点就有 close 数据的股 → mask=True 全程(视为成熟股)。"""
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=300)
    close = pd.DataFrame({
        "600000": np.arange(300, dtype=float),  # 起点就有值
    }, index=idx)
    mask = _listing_mask(close, min_days=252)
    assert mask["600000"].all()


def test_listing_mask_new_listing_blocks_first_n_days():
    """Panel 内某天才上市的股,头 min_days 天 mask=False,之后 True。"""
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=400)
    close = pd.DataFrame({
        "300001": [np.nan] * 50 + list(range(350)),  # 第 50 行才上市
    }, index=idx)
    mask = _listing_mask(close, min_days=252)
    # 前 50 行(NaN) → True 还是 False 都接受,关键是 50..50+252 是 False
    assert not mask["300001"].iloc[50:50+252].any()
    # 50+252 之后 True
    assert mask["300001"].iloc[50+252:].all()


def test_listing_mask_all_nan_stock_all_false():
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=100)
    close = pd.DataFrame({"600000": [np.nan] * 100}, index=idx)
    mask = _listing_mask(close, min_days=252)
    assert not mask["600000"].any()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_panel_mask.py::test_listing_mask_mature_stock_all_true -v
```
Expected: FAIL with `ImportError: cannot import name '_listing_mask'`

- [ ] **Step 3: Implement `_listing_mask`**

在 `panel.py` 加(紧跟 `_limit_threshold` 之后):

```python
def _listing_mask(close: pd.DataFrame, min_days: int = 252) -> pd.DataFrame:
    """Mask=False 对每只股 panel 内"新上市后头 min_days 个交易日"。

    成熟股(panel 起点就有 close)视为已经上市 >1 年,全程 True。
    全 NaN 的股全程 False。

    Args:
        close: T × N close 宽表(行 = date,列 = code)。
        min_days: 新上市股需经过的最小交易日数(252 ≈ 1 个交易年)。
    Returns:
        T × N bool DataFrame,与 close 同形同 index/columns。
    """
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    for code in close.columns:
        series = close[code]
        first_valid = series.first_valid_index()
        if first_valid is None:
            # 全 NaN 股 → 全 False
            mask[code] = False
            continue
        first_pos = close.index.get_loc(first_valid)
        if first_pos == 0:
            # Panel 起点就有数据 → 成熟股,全 True 不动
            continue
        # 新上市:头 min_days 个交易日 mask=False
        end_pos = min(first_pos + min_days, len(close))
        col_pos = mask.columns.get_loc(code)
        mask.iloc[first_pos:end_pos, col_pos] = False
    return mask
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_panel_mask.py -v
```
Expected: 6 PASS(累计 _limit_threshold 3 + _listing_mask 3)。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/panel.py tests/test_panel_mask.py
git commit -m "feat(mask): add _listing_mask helper (task 3/18)"
```

---

## Task 4: Add `compute_tradability_mask` in panel.py

**Files:**
- Modify: `src/stockpool/panel.py`
- Test: `tests/test_panel_mask.py`(追加)

- [ ] **Step 1: Write failing tests**

追加到 `tests/test_panel_mask.py`:

```python
def _make_panel(close_dict, volume_dict=None):
    """Helper: 从 {code: list} 字典构造一个 OHLCV panel(open=high=low=close)。"""
    codes = list(close_dict.keys())
    idx = pd.date_range("2024-01-01", periods=len(next(iter(close_dict.values()))))
    close = pd.DataFrame(close_dict, index=idx)
    if volume_dict is None:
        volume = pd.DataFrame({c: [1000.0] * len(idx) for c in codes}, index=idx)
    else:
        volume = pd.DataFrame(volume_dict, index=idx)
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": volume,
    }


def test_compute_mask_main_board_limit_up():
    """主板 +9.9% 应被 mask=False;创业板 +9.9% 不应被 mask=False。"""
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {
        "600000": [10.0, 10.99, 11.0, 11.01],     # 第 1 行 +9.9% → mask=False
        "300001": [10.0, 10.99, 11.0, 11.01],     # 第 1 行 +9.9% → mask=True (创业板 20%)
    }
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)  # 关掉 listing 隔离影响
    mask = compute_tradability_mask(panel, cfg)
    assert mask.loc[panel["close"].index[1], "600000"] == False
    assert mask.loc[panel["close"].index[1], "300001"] == True


def test_compute_mask_suspension_volume_zero():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {"600000": [10.0, 10.05, 10.1, 10.15]}
    volume_dict = {"600000": [1000.0, 0.0, 1000.0, 1000.0]}  # 第 1 行停牌
    panel = _make_panel(close_dict, volume_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.loc[panel["close"].index[1], "600000"] == False


def test_compute_mask_three_conditions_intersect():
    """三个条件 & 起来:任一不满足都 mask=False。"""
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {"600000": [10.0, 10.05, 10.10, 10.15]}
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    # 全程正常 → 除第 0 行 ret=NaN 外都 True
    # 第 0 行因 cond_not_limit 是 NaN.lt(...)→False 而 mask=False
    assert mask.iloc[0, 0] == False
    assert mask.iloc[1:, 0].all()


def test_compute_mask_shape_matches_close():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {f"600{i:03d}": [10.0 + i * 0.01] * 50 for i in range(5)}
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.shape == panel["close"].shape
    assert mask.index.equals(panel["close"].index)
    assert mask.columns.equals(panel["close"].columns)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_panel_mask.py::test_compute_mask_main_board_limit_up -v
```
Expected: FAIL `ImportError: cannot import name 'compute_tradability_mask'`

- [ ] **Step 3: Implement `compute_tradability_mask`**

在 `panel.py` 末尾加(import 部分先加 `from typing import TYPE_CHECKING`,然后在文件顶部 imports 区:`if TYPE_CHECKING: from stockpool.config import MaskConfig`)。

```python
def compute_tradability_mask(
    panel: Mapping[str, pd.DataFrame],
    config: "MaskConfig",
) -> pd.DataFrame:
    """从 OHLCV panel 计算可交易性 mask(close-side, paper B mask-first)。

    三条件 AND:
      1. |close ret| < per-code 涨跌停阈值
      2. volume > 0 (非停牌)
      3. 上市天数 ≥ min_listing_days (仅 panel 内新上市的股受限)

    Args:
        panel: OHLCV panel,至少含 "close" 和 "volume" 字段。
        config: ``MaskConfig`` 实例。本函数不检查 ``config.enabled`` —
                调用方负责决定要不要算 mask。
    Returns:
        T × N bool DataFrame,与 panel["close"] 同形。
    """
    close = panel["close"]
    volume = panel["volume"]

    # 板块阈值映射(对每个列 code 算一次)
    thresholds = pd.Series(
        {code: _limit_threshold_for_config(code, config) for code in close.columns}
    )

    # 条件 1:不在涨跌停
    ret = close / close.shift(1) - 1
    cond_not_limit = ret.abs().lt(thresholds, axis=1)
    # 注:第 0 行 ret 全 NaN → NaN.lt(...) = False → cond_not_limit 第 0 行 False

    # 条件 2:有成交
    cond_has_volume = volume > 0

    # 条件 3:上市天数
    cond_listed = _listing_mask(close, min_days=config.min_listing_days)

    return cond_not_limit & cond_has_volume & cond_listed


def _limit_threshold_for_config(code: str, config: "MaskConfig") -> float:
    """与 ``_limit_threshold`` 同接口,但阈值从 ``MaskConfig`` 取,
    便于配置覆盖默认值。"""
    if code.startswith(("300", "301", "688")):
        return config.limit_up_threshold_chinext
    if code.startswith(("82", "83", "87", "43")):
        return config.limit_up_threshold_bse
    return config.limit_up_threshold_main
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_panel_mask.py -v
```
Expected: 10 PASS 累计(3 _limit_threshold + 3 _listing_mask + 4 compute_tradability_mask)。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/panel.py tests/test_panel_mask.py
git commit -m "feat(mask): add compute_tradability_mask (task 4/18)"
```

---

## Task 5: Add `apply_mask` in panel.py

**Files:**
- Modify: `src/stockpool/panel.py`
- Test: `tests/test_panel_mask.py`(追加)

- [ ] **Step 1: Write failing tests**

追加到 `tests/test_panel_mask.py`:

```python
def test_apply_mask_nulls_correct_positions():
    from stockpool.panel import apply_mask
    idx = pd.date_range("2024-01-01", periods=4)
    panel = {
        "close": pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=idx),
        "open": pd.DataFrame({"A": [10.1, 11.1, 12.1, 13.1]}, index=idx),
        "high": pd.DataFrame({"A": [10.5, 11.5, 12.5, 13.5]}, index=idx),
        "low": pd.DataFrame({"A": [9.5, 10.5, 11.5, 12.5]}, index=idx),
        "volume": pd.DataFrame({"A": [100.0, 200.0, 300.0, 400.0]}, index=idx),
    }
    mask = pd.DataFrame({"A": [True, False, True, False]}, index=idx)
    out = apply_mask(panel, mask)
    for field in ("open", "high", "low", "close", "volume"):
        assert np.isnan(out[field].iloc[1, 0])
        assert np.isnan(out[field].iloc[3, 0])
        assert out[field].iloc[0, 0] == panel[field].iloc[0, 0]
        assert out[field].iloc[2, 0] == panel[field].iloc[2, 0]


def test_apply_mask_does_not_mutate_input():
    from stockpool.panel import apply_mask
    idx = pd.date_range("2024-01-01", periods=3)
    panel = {
        "close": pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=idx),
        "volume": pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=idx),
    }
    mask = pd.DataFrame({"A": [True, False, True]}, index=idx)
    _ = apply_mask(panel, mask)
    # 原 panel 不变
    assert panel["close"].iloc[1, 0] == 11.0
    assert panel["volume"].iloc[1, 0] == 200.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_panel_mask.py::test_apply_mask_nulls_correct_positions -v
```
Expected: FAIL `ImportError: cannot import name 'apply_mask'`

- [ ] **Step 3: Implement `apply_mask`**

在 `panel.py` 末尾加:

```python
def apply_mask(
    panel: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return a new panel with mask=False positions set to NaN across all fields.

    原 panel 不被修改(``DataFrame.where`` 返回新对象)。下游因子算子通过
    NaN 自然剔除被 mask 的样本点。

    Args:
        panel: OHLCV panel(``Mapping[str, DataFrame]``)。
        mask: T × N bool DataFrame,与 panel 字段同形。
    Returns:
        新 ``dict[str, DataFrame]``,字段同 panel,mask=False 位置为 NaN。
    """
    return {field: df.where(mask) for field, df in panel.items()}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_panel_mask.py -v
```
Expected: 12 PASS 累计。

- [ ] **Step 5: Run full suite to confirm no regression**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: 全部 PASS,374 + 5 (config) + 12 (panel_mask) = 391 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/panel.py tests/test_panel_mask.py
git commit -m "feat(mask): add apply_mask (task 5/18)"
```

---

## Task 6: Relax `min_periods` in `ts_sum/ts_mean/ts_std`

**Files:**
- Modify: `src/stockpool/factors/ops.py`
- Test: `tests/test_ops_mask_nan_safe.py`(新文件)

- [ ] **Step 1: Write failing tests**

创建 `tests/test_ops_mask_nan_safe.py`:

```python
"""Tests for stockpool.factors.ops NaN-safety after mask-first refactor."""
import numpy as np
import pandas as pd
import pytest


def test_ts_mean_full_valid_input_unchanged():
    """全 valid 输入,放宽 min_periods 不改变结果。"""
    from stockpool.factors.ops import ts_mean
    x = pd.DataFrame({"A": np.arange(30, dtype=float)})
    out = ts_mean(x, 10)
    # 前 5 行(min_periods=int(10*0.6)=6 之前)还是 NaN
    assert out["A"].iloc[:5].isna().all()
    # 第 9 行 = (0+1+...+9)/10 = 4.5
    assert out["A"].iloc[9] == pytest.approx(4.5)
    # 第 29 行 = (20+...+29)/10 = 24.5
    assert out["A"].iloc[29] == pytest.approx(24.5)


def test_ts_mean_with_nan_in_window_uses_valid():
    """窗口内 1 个 NaN,放宽 min_periods 后用其余值算均值,不再返回 NaN。"""
    from stockpool.factors.ops import ts_mean
    vals = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    x = pd.DataFrame({"A": vals})
    out = ts_mean(x, 10)
    # 第 9 行 = mean([1,2,4,5,6,7,8,9,10]) = 52/9
    assert out["A"].iloc[9] == pytest.approx(52.0 / 9.0)


def test_ts_std_with_nan_in_window():
    from stockpool.factors.ops import ts_std
    vals = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    x = pd.DataFrame({"A": vals})
    out = ts_std(x, 10)
    # 至少不是 NaN
    assert not np.isnan(out["A"].iloc[9])


def test_ts_sum_with_nan_in_window():
    from stockpool.factors.ops import ts_sum
    vals = [1.0, 2.0, np.nan, 4.0, 5.0]
    x = pd.DataFrame({"A": vals})
    out = ts_sum(x, 5)
    # min_periods=int(5*0.6)=3,窗口有 4 个非 NaN,所以 sum=12
    assert out["A"].iloc[4] == pytest.approx(12.0)


def test_ts_mean_too_few_valid_returns_nan():
    """如果窗口非 NaN 数 < min_periods,返回 NaN。"""
    from stockpool.factors.ops import ts_mean
    vals = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0]
    x = pd.DataFrame({"A": vals})
    out = ts_mean(x, 10)
    # 第 9 行非 NaN 数 = 5 < min_periods=6 → NaN
    assert np.isnan(out["A"].iloc[9])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ops_mask_nan_safe.py::test_ts_mean_with_nan_in_window_uses_valid -v
```
Expected: FAIL — 当前 `min_periods=d` 严格,有 NaN → 整窗口 NaN。

- [ ] **Step 3: Patch `ts_sum/ts_mean/ts_std`**

修改 `src/stockpool/factors/ops.py`:

```python
def _min_periods(d: int) -> int:
    """放宽 min_periods 到 60% 窗口长度,使 mask=False 引入的 NaN 不会
    整段杀掉因子值。``max(1, ...)`` 防 d<2 时退化。"""
    return max(1, int(d * 0.6))


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).sum()


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).mean()


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).std(ddof=0)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ops_mask_nan_safe.py -v
```
Expected: 5 PASS.

- [ ] **Step 5: Run ops tests to verify no regression**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ops.py tests/test_factors.py tests/test_wq101.py -q
```
Expected: 全部 PASS。放宽 min_periods 在全 valid 输入下结果不变。

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/factors/ops.py tests/test_ops_mask_nan_safe.py
git commit -m "feat(mask): relax min_periods in ts_sum/mean/std for NaN tolerance (task 6/18)"
```

---

## Task 7: NaN-safe `decay_linear` and `ts_product`

**Files:**
- Modify: `src/stockpool/factors/ops.py`
- Test: `tests/test_ops_mask_nan_safe.py`(追加)

- [ ] **Step 1: Write failing tests**

追加到 `tests/test_ops_mask_nan_safe.py`:

```python
def test_decay_linear_full_valid_unchanged():
    """全 valid 输入,decay_linear 数值与旧实现一致(权重 1..d)。"""
    from stockpool.factors.ops import decay_linear
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = decay_linear(x, 5)
    # 加权: (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15 = (1+4+9+16+25)/15 = 55/15
    assert out["A"].iloc[4] == pytest.approx(55.0 / 15.0)


def test_decay_linear_with_nan_renormalizes():
    """窗口内 1 个 NaN,权重和分母同步重归一化。"""
    from stockpool.factors.ops import decay_linear
    # 窗口 [1, nan, 3, 4, 5], 权重 [1, 2, 3, 4, 5]
    # Valid: vals [1, 3, 4, 5], weights [1, 3, 4, 5]
    # 加权和 / 权重和 = (1 + 9 + 16 + 25) / (1+3+4+5) = 51/13
    vals = [1.0, np.nan, 3.0, 4.0, 5.0]
    x = pd.DataFrame({"A": vals})
    out = decay_linear(x, 5)
    assert out["A"].iloc[4] == pytest.approx(51.0 / 13.0)


def test_decay_linear_all_nan_returns_nan():
    from stockpool.factors.ops import decay_linear
    x = pd.DataFrame({"A": [np.nan] * 5})
    out = decay_linear(x, 5)
    assert np.isnan(out["A"].iloc[4])


def test_ts_product_full_valid_unchanged():
    from stockpool.factors.ops import ts_product
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    out = ts_product(x, 4)
    assert out["A"].iloc[3] == pytest.approx(24.0)


def test_ts_product_with_nan_skips():
    """窗口含 NaN,用 nanprod 跳过。"""
    from stockpool.factors.ops import ts_product
    x = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0]})
    out = ts_product(x, 4)
    # min_periods=int(4*0.6)=2,3 个非 NaN,np.nanprod=8
    assert out["A"].iloc[3] == pytest.approx(8.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ops_mask_nan_safe.py::test_decay_linear_with_nan_renormalizes -v
```
Expected: FAIL — 当前 `_dot` 检测到 NaN 就 return NaN。

- [ ] **Step 3: Rewrite `decay_linear` and `ts_product`**

替换 `src/stockpool/factors/ops.py` 中:

```python
def ts_product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=_min_periods(d)).apply(
        lambda s: float(np.nanprod(s)) if np.isfinite(np.nanprod(s)) else np.nan,
        raw=True,
    )


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """加权移动平均,权重 1, 2, ..., d 归一化。WQ101 ``decay_linear``。

    NaN-safe:窗口内 NaN 位置同步从分子/分母剔除,余下权重重归一化。
    全 NaN 窗口返回 NaN。
    """
    weights = np.arange(1, d + 1, dtype=float)

    def _wmean(a: np.ndarray) -> float:
        valid = ~np.isnan(a)
        if not valid.any():
            return np.nan
        w = weights[valid]
        v = a[valid]
        return float(np.dot(v, w) / w.sum())

    return x.rolling(d, min_periods=_min_periods(d)).apply(_wmean, raw=True)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ops_mask_nan_safe.py -v
```
Expected: 10 PASS 累计。

- [ ] **Step 5: Regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ops.py tests/test_factors.py tests/test_wq101.py -q
```
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/factors/ops.py tests/test_ops_mask_nan_safe.py
git commit -m "feat(mask): NaN-safe decay_linear + ts_product (task 7/18)"
```

---

## Task 8: `compute_factor_panel` accepts optional mask param

**Files:**
- Modify: `src/stockpool/ml/dataset.py`
- Test: `tests/test_ml_strategy_mask.py`(新文件)

- [ ] **Step 1: Write failing tests**

创建 `tests/test_ml_strategy_mask.py`:

```python
"""Tests for tradability mask integration in ml/dataset pipeline."""
import numpy as np
import pandas as pd
import pytest


def _make_panel(close_dict):
    codes = list(close_dict.keys())
    n = len(next(iter(close_dict.values())))
    idx = pd.date_range("2024-01-01", periods=n)
    close = pd.DataFrame(close_dict, index=idx)
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": pd.DataFrame({c: [1000.0] * n for c in codes}, index=idx),
    }


def test_compute_factor_panel_no_mask_unchanged():
    """mask=None 时与旧行为 bitwise 一致。"""
    from stockpool.ml.dataset import compute_factor_panel
    panel = _make_panel({"600000": list(np.linspace(10, 11, 30))})
    out_a = compute_factor_panel(panel, ["momentum_5"])
    out_b = compute_factor_panel(panel, ["momentum_5"], mask=None)
    pd.testing.assert_frame_equal(out_a["momentum_5"], out_b["momentum_5"])


def test_compute_factor_panel_with_mask_changes_values():
    """mask 应用后,被 mask 位置的因子值变 NaN。"""
    from stockpool.ml.dataset import compute_factor_panel
    panel = _make_panel({"600000": list(np.linspace(10, 11, 30))})
    # 把第 5 天 mask=False
    mask = pd.DataFrame(True, index=panel["close"].index, columns=panel["close"].columns)
    mask.iloc[5, 0] = False
    out = compute_factor_panel(panel, ["momentum_5"], mask=mask)
    # 第 5 天的 close 已被 NaN-out → momentum_5 在第 5 天 NaN
    assert np.isnan(out["momentum_5"].iloc[5, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py::test_compute_factor_panel_with_mask_changes_values -v
```
Expected: FAIL — `compute_factor_panel` 还没 `mask` 参数。

- [ ] **Step 3: Modify `compute_factor_panel`**

`src/stockpool/ml/dataset.py` 中替换 `compute_factor_panel`:

```python
def compute_factor_panel(
    panel: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str],
    *,
    mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """在 OHLCV Panel 上算所有因子,返回 ``{name: T×N DataFrame}``。

    Args:
        panel: OHLCV panel。
        factor_names: 因子名列表(由 ``make_factor`` 注册表解析)。
        mask: 可选 T × N bool。若提供,会在算因子前把 panel 各字段
              ``where(mask)``;mask=False 位置变 NaN,通过算子自然传播。
              default None → 旧行为不变。
    """
    if mask is not None:
        from stockpool.panel import apply_mask
        panel = apply_mask(panel, mask)
    out: dict[str, pd.DataFrame] = {}
    for name in factor_names:
        f = make_factor(name)
        out[f.name] = f.compute(panel)
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/ml/dataset.py tests/test_ml_strategy_mask.py
git commit -m "feat(mask): compute_factor_panel accepts optional mask param (task 8/18)"
```

---

## Task 9: `forward_return_panel` accepts optional mask param

**Files:**
- Modify: `src/stockpool/ml/dataset.py`
- Test: `tests/test_ml_strategy_mask.py`(追加)

- [ ] **Step 1: Write failing tests**

追加到 `tests/test_ml_strategy_mask.py`:

```python
def test_forward_return_panel_no_mask_unchanged():
    from stockpool.ml.dataset import forward_return_panel
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]})
    y_a = forward_return_panel(close, horizon=2)
    y_b = forward_return_panel(close, horizon=2, mask=None)
    pd.testing.assert_frame_equal(y_a, y_b)


def test_forward_return_panel_bidirectional_mask():
    """mask[t]=False 或 mask[t+h]=False 时,y[t]=NaN。"""
    from stockpool.ml.dataset import forward_return_panel
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]})
    # mask[1]=False → y[1] 应 NaN(t=1 时不可用)
    # mask[3]=False, horizon=2 → y[1] 应 NaN(t+h=3 时不可用)
    mask = pd.DataFrame({"A": [True, True, False, True, True]})
    y = forward_return_panel(close, horizon=2, mask=mask)
    # t=0: mask[0]=T ∧ mask[2]=F → NaN
    # t=1: mask[1]=T ∧ mask[3]=T → 不 NaN ((13-11)/11=0.1818)
    # t=2: mask[2]=F → NaN (无 forward)
    assert np.isnan(y["A"].iloc[0])
    assert y["A"].iloc[1] == pytest.approx(2.0 / 11.0)
    assert np.isnan(y["A"].iloc[2])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py::test_forward_return_panel_bidirectional_mask -v
```
Expected: FAIL — `forward_return_panel` 还没 mask 参数。

- [ ] **Step 3: Modify `forward_return_panel`**

`src/stockpool/ml/dataset.py` 中替换:

```python
def forward_return_panel(
    close: pd.DataFrame,
    horizon: int,
    label_type: str = "return",
    *,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """T×N forward-return panel with configurable label transform.

    Args:
        close: T × N 收盘价宽表 (date index, code columns).
        horizon: 前瞻天数 h。
        label_type: "return" / "vol_adjusted" / "cross_sec_rank" (后两者 stub)。
        mask: 可选 T × N bool。若提供,做 **双向检查** —
              要求 ``mask[t]=True ∧ mask[t+horizon]=True``;不满足的 t 位置
              y 值变 NaN(防 close[t]/close[t+h] 涨停导致 label 虚高)。
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if label_type == "return":
        y = close.shift(-horizon) / close - 1.0
        if mask is not None:
            label_valid = mask & mask.shift(-horizon).fillna(False)
            y = y.where(label_valid)
        return y
    if label_type in ("vol_adjusted", "cross_sec_rank"):
        raise NotImplementedError(
            f"label_type={label_type!r} is not implemented in PR-A; "
            f"interface stub only."
        )
    raise ValueError(
        f"unknown label_type={label_type!r}; "
        f"expected one of: return, vol_adjusted, cross_sec_rank"
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py -v
```
Expected: 4 PASS 累计。

- [ ] **Step 5: Regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/ml/dataset.py tests/test_ml_strategy_mask.py
git commit -m "feat(mask): forward_return_panel bidirectional mask (task 9/18)"
```

---

## Task 10: `build_factor_panel` accepts `mask_config`

**Files:**
- Modify: `src/stockpool/strategy_factory.py`
- Test: `tests/test_ml_strategy_mask.py`(追加)

- [ ] **Step 1: Write failing tests**

追加到 `tests/test_ml_strategy_mask.py`:

```python
def test_build_factor_panel_no_mask_config_unchanged():
    from stockpool.strategy_factory import build_factor_panel
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    out_a = build_factor_panel(["momentum_5"], pool_data)
    out_b = build_factor_panel(["momentum_5"], pool_data, mask_config=None)
    pd.testing.assert_frame_equal(out_a["momentum_5"], out_b["momentum_5"])


def test_build_factor_panel_mask_disabled_equivalent_to_no_config():
    from stockpool.strategy_factory import build_factor_panel
    from stockpool.config import MaskConfig
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    out_a = build_factor_panel(["momentum_5"], pool_data, mask_config=MaskConfig(enabled=False))
    out_b = build_factor_panel(["momentum_5"], pool_data, mask_config=None)
    pd.testing.assert_frame_equal(out_a["momentum_5"], out_b["momentum_5"])


def test_build_factor_panel_mask_enabled_changes_output():
    """mask 开启时,人为植入涨停日,该日的因子值变 NaN。"""
    from stockpool.strategy_factory import build_factor_panel
    from stockpool.config import MaskConfig
    n = 30
    closes = np.linspace(10, 11, n)
    # 第 10 行强制 +9.9% 模拟涨停
    closes[10] = closes[9] * 1.099
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": [1000.0] * n,
    })
    pool_data = {"600000": df}
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    out = build_factor_panel(["momentum_5"], pool_data, mask_config=cfg)
    # 第 10 天 close 已 NaN → momentum_5 在第 10 天 NaN
    assert np.isnan(out["momentum_5"].iloc[10, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py::test_build_factor_panel_mask_enabled_changes_output -v
```
Expected: FAIL — `build_factor_panel` 没 `mask_config` 参数。

- [ ] **Step 3: Modify `build_factor_panel`**

`src/stockpool/strategy_factory.py` 中替换 `build_factor_panel`:

```python
def build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    *,
    mask_config: "MaskConfig | None" = None,
) -> dict[str, pd.DataFrame]:
    """从 ``{code: daily_df}`` 装一个 OHLCV Panel,在 Panel 上算所有因子,
    返回 ``{factor_name: T×N DataFrame}``。

    Look-ahead 安全:因子在第 i 行只用 ``[:i+1]`` 数据(由 Factor 契约保证),
    所以一次性预算整段历史不会泄露未来。

    Args:
        factor_names: 因子名列表。
        pool_data: ``{code: daily_df}`` per-stock 字典。
        mask_config: 可选 ``MaskConfig``。若 ``enabled=True``,会在算因子
                     前对 panel 字段应用 tradability mask
                     (close-side, paper B mask-first)。
    """
    from stockpool.ml.dataset import compute_factor_panel

    # 1) 把每股 daily_df → date-indexed,按列拼成宽表
    per_stock: dict[str, pd.DataFrame] = {}
    for code, df in pool_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        per_stock[code] = d.set_index("date").sort_index()
    if not per_stock:
        return {}
    all_dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")
    panel: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        panel[field] = pd.DataFrame(
            {code: d[field].reindex(idx) for code, d in per_stock.items()},
            index=idx,
        )

    # 2) 可选:计算 mask 并应用
    mask: pd.DataFrame | None = None
    if mask_config is not None and mask_config.enabled:
        from stockpool.panel import compute_tradability_mask
        mask = compute_tradability_mask(panel, mask_config)

    return compute_factor_panel(panel, factor_names, mask=mask)
```

文件顶部 imports 加(如未存在):

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from stockpool.config import MaskConfig
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py -v
```
Expected: 7 PASS 累计。

- [ ] **Step 5: Regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/strategy_factory.py tests/test_ml_strategy_mask.py
git commit -m "feat(mask): build_factor_panel accepts mask_config (task 10/18)"
```

---

## Task 11: `load_or_build_factor_panel` passes through `mask_config` + manifest

**Files:**
- Modify: `src/stockpool/strategy_factory.py`
- Test: `tests/test_factor_panel_cache.py`(追加)

- [ ] **Step 1: Write failing tests**

追加到 `tests/test_factor_panel_cache.py`:

```python
def test_load_or_build_factor_panel_passes_mask_config(tmp_path):
    """`mask_config` 透传到 build_factor_panel,manifest 含 mask 字段。"""
    from stockpool.strategy_factory import load_or_build_factor_panel
    from stockpool.config import MaskConfig
    import json

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    fp, _ = load_or_build_factor_panel(
        ["momentum_5"], pool_data, cache_dir=tmp_path, mask_config=cfg,
    )
    assert "momentum_5" in fp
    # 找到唯一的 sig 目录
    panels_dir = tmp_path / "factor_panels"
    sig_dirs = list(panels_dir.iterdir())
    assert len(sig_dirs) == 1
    manifest = json.loads((sig_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("mask_enabled") is True
    assert manifest.get("mask_threshold_main") == 0.098


def test_load_or_build_factor_panel_mask_changes_cache_sig(tmp_path):
    """翻 mask.enabled 应产生不同 sig(新缓存目录)。"""
    from stockpool.strategy_factory import load_or_build_factor_panel
    from stockpool.config import MaskConfig
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    load_or_build_factor_panel(
        ["momentum_5"], pool_data, cache_dir=tmp_path,
        mask_config=MaskConfig(enabled=False),
    )
    load_or_build_factor_panel(
        ["momentum_5"], pool_data, cache_dir=tmp_path,
        mask_config=MaskConfig(enabled=True, min_listing_days=0),
    )
    panels_dir = tmp_path / "factor_panels"
    assert len(list(panels_dir.iterdir())) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py::test_load_or_build_factor_panel_passes_mask_config -v
```
Expected: FAIL — `load_or_build_factor_panel` 没 `mask_config`。

- [ ] **Step 3: Modify `_factor_panel_sig`**

`src/stockpool/strategy_factory.py` 中找到 `_factor_panel_sig`(约 148 行),替换为:

```python
def _factor_panel_sig(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    mask_config: "MaskConfig | None" = None,
) -> tuple[str, str]:
    """Return (12-char sig, last_date_iso). Mask config is part of the key.

    Universe = sorted code list. last_date = max of any stock's max date.
    Mask config (when enabled) hashed into sig → flipping mask invalidates cache.
    """
    codes = sorted(pool_data.keys())
    last_date = pd.Timestamp.min
    for df in pool_data.values():
        if len(df) > 0:
            d = pd.to_datetime(df["date"]).max()
            if d > last_date:
                last_date = d
    last_iso = "" if last_date is pd.Timestamp.min else last_date.date().isoformat()
    mask_dict = None
    if mask_config is not None and mask_config.enabled:
        mask_dict = mask_config.model_dump()
    blob = json.dumps(
        {
            "factors": sorted(factor_names),
            "codes": codes,
            "last_date": last_iso,
            "mask": mask_dict,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12], last_iso
```

- [ ] **Step 4: Modify `load_or_build_factor_panel`**

`src/stockpool/strategy_factory.py` 中找到 `load_or_build_factor_panel`(约 172 行),替换为(只列出改动行,保留原有 cache hit 路径 + log 行不动):

```python
def load_or_build_factor_panel(
    factor_names: list[str],
    pool_data: Mapping[str, pd.DataFrame],
    cache_dir: str | Path,
    refresh: bool = False,
    *,
    mask_config: "MaskConfig | None" = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Disk-cached wrapper around ``build_factor_panel`` + ``build_close_panel``.

    Mask config (when enabled) is hashed into ``sig`` so flipping
    ``mask.enabled`` produces a fresh cache directory and never mixes data.

    (后续 docstring 保留原文)
    """
    if not pool_data:
        return {}, pd.DataFrame()

    sig, last_iso = _factor_panel_sig(factor_names, pool_data, mask_config)
    root = Path(cache_dir) / "factor_panels" / sig
    manifest_path = root / "manifest.json"

    if not refresh and manifest_path.exists():
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            close_path = root / "close.parquet"
            paths = {n: root / f"{n}.parquet" for n in meta.get("factors", [])}
            if close_path.exists() and all(p.exists() for p in paths.values()):
                log.info("Factor panel cache hit: %s (sig=%s)", root, sig)
                close_panel = pd.read_parquet(close_path)
                factor_panel = {n: pd.read_parquet(p) for n, p in paths.items()}
                return factor_panel, close_panel
            log.warning("Factor panel manifest exists but parquets incomplete; rebuilding")
        except Exception as e:
            log.warning("Factor panel cache read failed (%s); rebuilding", e)

    log.info("Building factor panel: %d factors × %d stocks (sig=%s, mask=%s)",
             len(factor_names), len(pool_data), sig,
             bool(mask_config and mask_config.enabled))
    factor_panel = build_factor_panel(factor_names, pool_data, mask_config=mask_config)
    close_panel = build_close_panel(pool_data)

    root.mkdir(parents=True, exist_ok=True)
    try:
        close_panel.to_parquet(root / "close.parquet")
        for name, wide in factor_panel.items():
            wide.to_parquet(root / f"{name}.parquet")
        manifest_dict = {
            "sig": sig,
            "factors": list(factor_panel.keys()),
            "n_codes": len(pool_data),
            "last_date": last_iso,
            "built_at": pd.Timestamp.now("UTC").isoformat(),
            "mask_enabled": bool(mask_config and mask_config.enabled),
        }
        if mask_config is not None and mask_config.enabled:
            manifest_dict["mask_threshold_main"] = mask_config.limit_up_threshold_main
            manifest_dict["mask_threshold_chinext"] = mask_config.limit_up_threshold_chinext
            manifest_dict["mask_min_listing_days"] = mask_config.min_listing_days
        manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
        log.info("Factor panel cached: %s", root)
    except Exception as e:
        log.warning("Failed to write factor panel cache (%s); proceeding in-memory", e)

    return factor_panel, close_panel
```

注意:保留原文件的 cache hit 路径(完整保持 `paths={n: root / f"{n}.parquet" for n in meta.get("factors", [])}` 这一行 + `close_path.exists() and all(p.exists() ...)` 检查)。修改重点 = 三处:
1. `_factor_panel_sig(..., mask_config)` 调用加参
2. `build_factor_panel(..., mask_config=mask_config)` 传参
3. manifest dict 多三个 mask 字段

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_factor_panel_cache.py -v
```
Expected: 全部 PASS(含新加的 2 个 mask 测试 + 现有 panel cache 测试)。

- [ ] **Step 5: Regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/strategy_factory.py tests/test_factor_panel_cache.py
git commit -m "feat(mask): load_or_build_factor_panel propagates mask_config to sig+manifest (task 11/18)"
```

---

## Task 12: `build_panel` (pooled top-level) accepts `mask_config`

**Files:**
- Modify: `src/stockpool/ml/dataset.py`
- Test: `tests/test_ml_strategy_mask.py`(追加)

- [ ] **Step 1: Write failing tests**

追加:

```python
def test_build_panel_no_mask_unchanged():
    from stockpool.ml.dataset import build_panel
    n = 30
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10, 11, n),
        "volume": [1000.0] * n,
    })
    stocks_data = {"600000": df}
    X_a, y_a = build_panel(stocks_data, ["momentum_5"], horizon=2)
    X_b, y_b = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=None)
    pd.testing.assert_frame_equal(X_a, X_b)
    pd.testing.assert_series_equal(y_a, y_b)


def test_build_panel_mask_drops_samples():
    """mask 启用 + 强制涨停 → 训练样本数下降。"""
    from stockpool.ml.dataset import build_panel
    from stockpool.config import MaskConfig
    n = 30
    closes = np.linspace(10, 11, n)
    closes[15] = closes[14] * 1.099  # 第 15 天涨停
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": [1000.0] * n,
    })
    stocks_data = {"600000": df}
    cfg_no = MaskConfig(enabled=False)
    cfg_yes = MaskConfig(enabled=True, min_listing_days=0)
    _, y_no = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=cfg_no)
    _, y_yes = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=cfg_yes)
    # 启用 mask 后样本数 ≤(因为涨停日 + horizon=2 反向 → 至少多丢 1 行)
    assert len(y_yes) < len(y_no)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py::test_build_panel_mask_drops_samples -v
```
Expected: FAIL — `build_panel` 还没 `mask_config`。

- [ ] **Step 3: Modify `build_panel`**

`src/stockpool/ml/dataset.py` 中替换 `build_panel`:

```python
def build_panel(
    stocks_data: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str],
    horizon: int,
    *,
    mask_config: "MaskConfig | None" = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Pool multi-stock data into a single (X, y) panel.

    Args:
        stocks_data: ``{code: daily_df}``。
        factor_names: 因子名列表。
        horizon: forward return 前瞻天数。
        mask_config: 可选 ``MaskConfig``,启用 tradability mask。
    """
    if not stocks_data:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["stock", "date"])
        return (
            pd.DataFrame(columns=list(factor_names), index=empty_idx),
            pd.Series(dtype=float, index=empty_idx),
        )

    # 1) 构造 Panel
    per_stock: dict[str, pd.DataFrame] = {}
    for code, df in stocks_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        per_stock[code] = d.set_index("date").sort_index()
    all_dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")
    panel: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        panel[field] = pd.DataFrame(
            {code: d[field].reindex(idx) for code, d in per_stock.items()},
            index=idx,
        )

    # 2) 计算 mask(若启用)
    mask: pd.DataFrame | None = None
    if mask_config is not None and mask_config.enabled:
        from stockpool.panel import compute_tradability_mask
        mask = compute_tradability_mask(panel, mask_config)

    # 3) 因子 + 标签
    factor_panel = compute_factor_panel(panel, factor_names, mask=mask)
    fwd_ret = forward_return_panel(panel["close"], horizon, mask=mask)

    # 4) Stack 到长表
    return stack_panel_to_xy(factor_panel, fwd_ret, dropna=True)
```

`ml/dataset.py` 顶部 imports 加(如未存在):

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from stockpool.config import MaskConfig
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py -v
```
Expected: 9 PASS 累计。

- [ ] **Step 5: Regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/ml/dataset.py tests/test_ml_strategy_mask.py
git commit -m "feat(mask): build_panel (pooled) accepts mask_config (task 12/18)"
```

---

## Task 13: `build_factor_matrix` accepts `mask_config` (per_stock path)

**Files:**
- Modify: `src/stockpool/ml/dataset.py`
- Test: `tests/test_ml_strategy_mask.py`(追加)

- [ ] **Step 1: Write failing tests**

追加:

```python
def test_build_factor_matrix_no_mask_unchanged():
    from stockpool.ml.dataset import build_factor_matrix
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20),
        "open": np.linspace(10, 11, 20),
        "high": np.linspace(10.1, 11.1, 20),
        "low": np.linspace(9.9, 10.9, 20),
        "close": np.linspace(10, 11, 20),
        "volume": [1000.0] * 20,
    })
    out_a = build_factor_matrix(df, ["momentum_5"])
    out_b = build_factor_matrix(df, ["momentum_5"], mask_config=None)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_build_factor_matrix_mask_main_board_limit_up():
    """主板涨停日 → 该行因子值变 NaN。"""
    from stockpool.ml.dataset import build_factor_matrix
    from stockpool.config import MaskConfig
    closes = np.linspace(10, 11, 20).copy()
    closes[10] = closes[9] * 1.099  # 涨停
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20),
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": [1000.0] * 20,
    })
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    out = build_factor_matrix(df, ["momentum_5"], mask_config=cfg)
    assert np.isnan(out["momentum_5"].iloc[10])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py::test_build_factor_matrix_mask_main_board_limit_up -v
```
Expected: FAIL — `build_factor_matrix` 还没 `mask_config`。

- [ ] **Step 3: Modify `build_factor_matrix`**

`ml/dataset.py` 中替换:

```python
def build_factor_matrix(
    df: pd.DataFrame,
    factor_names: Sequence[str],
    *,
    mask_config: "MaskConfig | None" = None,
) -> pd.DataFrame:
    """Compute every named factor on one stock's ``df``, return T × F.

    Wraps ``df`` into a 1-stock panel; cross-sectional factors (rank,
    indneutralize) will return degenerate constants — use the pooled path
    (``build_panel`` / ``stack_panel_to_xy``) for those.

    Args:
        df: 单股 daily DataFrame(含 date + OHLCV 列)。
        factor_names: 因子名列表。
        mask_config: 可选 ``MaskConfig``,启用 single-stock tradability mask。
    """
    panel = _df_to_singleton_panel(df)
    code = next(iter(panel["close"].columns))

    mask: pd.DataFrame | None = None
    if mask_config is not None and mask_config.enabled:
        from stockpool.panel import compute_tradability_mask
        mask = compute_tradability_mask(panel, mask_config)

    cols: dict[str, pd.Series] = {}
    for name in factor_names:
        f = make_factor(name)
        if mask is not None:
            from stockpool.panel import apply_mask
            wide = f.compute(apply_mask(panel, mask))
        else:
            wide = f.compute(panel)
        cols[f.name] = wide[code].reset_index(drop=True)
    out = pd.DataFrame(cols)
    out.index = pd.Index(df["date"].reset_index(drop=True), name="date")
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py -v
```
Expected: 11 PASS 累计。

- [ ] **Step 5: Regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/ml/dataset.py tests/test_ml_strategy_mask.py
git commit -m "feat(mask): build_factor_matrix (per_stock) accepts mask_config (task 13/18)"
```

---

## Task 14: `MLFactorStrategy` propagates `cfg.mask` to internal data-prep calls

**Files:**
- Modify: `src/stockpool/backtesting/strategies.py`(MLFactorStrategy 内部)
- Test: `tests/test_ml_strategy_mask.py`(追加)

**Background:**
- `MLFactorStrategy.__init__(cfg, ...)` 已经接 `cfg: MLFactorConfig`,mask 通过 `cfg.mask` 自然可访问 — **不需要新加 init 参数**
- `_strategy_signature()` 用 `self.cfg.model_dump()` 算 hash,改 mask 自动 → sig 变 → 旧 ml_models pkl 失效,**不需要专门处理缓存**
- 需要改的只有两处内部数据生成调用:
  - `MLFactorStrategy._try_fit` 内 `build_panel(pool, cfg.factors, cfg.horizon)`(strategies.py 约 618 行,pooled 路径)
  - `predict_latest` / `generate_signals` 内 `build_factor_matrix(daily_df, self.cfg.factors)`(strategies.py 约 481 行,per_stock 路径)

- [ ] **Step 1: Write failing test**

追加到 `tests/test_ml_strategy_mask.py`:

```python
def test_ml_factor_strategy_mask_changes_sig():
    """翻 cfg.mask.enabled → _strategy_signature 变化 → 旧 cache 失效。"""
    from stockpool.config import MLFactorConfig
    from stockpool.backtesting.strategies import MLFactorStrategy
    cfg_no = MLFactorConfig.model_validate({
        "factors": ["momentum_5"],
        "mask": {"enabled": False},
    })
    cfg_yes = MLFactorConfig.model_validate({
        "factors": ["momentum_5"],
        "mask": {"enabled": True},
    })
    s_no = MLFactorStrategy(cfg_no)
    s_yes = MLFactorStrategy(cfg_yes)
    assert s_no._strategy_signature() != s_yes._strategy_signature()


def test_ml_factor_strategy_pooled_path_uses_mask(monkeypatch):
    """pooled `_try_fit` 内 build_panel 调用带上 mask_config=self.cfg.mask。"""
    from stockpool.config import MLFactorConfig
    from stockpool.backtesting.strategies import MLFactorStrategy
    import stockpool.ml.dataset as ds

    captured = {}
    orig = ds.build_panel
    def spy_build_panel(stocks_data, factor_names, horizon, *, mask_config=None):
        captured["mask_config"] = mask_config
        return orig(stocks_data, factor_names, horizon, mask_config=mask_config)
    monkeypatch.setattr(
        "stockpool.backtesting.strategies.build_panel", spy_build_panel
    )

    cfg = MLFactorConfig.model_validate({
        "factors": ["momentum_5"],
        "panel_mode": "pooled",
        "horizon": 2,
        "train_window": 20,
        "min_train_samples": 5,
        "refit_every": 5,
        "share_pool_fit": False,  # 走非 shared 路径,确保 build_panel 被 call
        "mask": {"enabled": True, "min_listing_days": 0},
    })
    n = 50
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10, 11, n),
        "volume": [1000.0] * n,
    })
    pool_data = {"600000": df, "600001": df.copy()}
    strat = MLFactorStrategy(cfg, pool_data=pool_data, current_stock_code="600000")
    # 触发 _try_fit:跑一次 generate_signals 即可
    _ = strat.generate_signals(df)
    assert captured.get("mask_config") is not None
    assert captured["mask_config"].enabled is True


def test_ml_factor_strategy_per_stock_path_uses_mask(monkeypatch):
    """per_stock 路径 build_factor_matrix 调用带 mask_config=self.cfg.mask。"""
    from stockpool.config import MLFactorConfig
    from stockpool.backtesting.strategies import MLFactorStrategy
    import stockpool.ml.dataset as ds

    captured = {}
    orig = ds.build_factor_matrix
    def spy_build_fm(df, factor_names, *, mask_config=None):
        captured["mask_config"] = mask_config
        return orig(df, factor_names, mask_config=mask_config)
    monkeypatch.setattr(
        "stockpool.backtesting.strategies.build_factor_matrix", spy_build_fm
    )

    cfg = MLFactorConfig.model_validate({
        "factors": ["momentum_5"],
        "panel_mode": "per_stock",
        "horizon": 2,
        "train_window": 20,
        "min_train_samples": 5,
        "refit_every": 5,
        "mask": {"enabled": True, "min_listing_days": 0},
    })
    n = 50
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10, 11, n),
        "volume": [1000.0] * n,
    })
    strat = MLFactorStrategy(cfg)
    _ = strat.generate_signals(df)
    assert captured.get("mask_config") is not None
    assert captured["mask_config"].enabled is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py::test_ml_factor_strategy_pooled_path_uses_mask -v
```
Expected: FAIL — `build_panel` / `build_factor_matrix` 当前不传 `mask_config`。

- [ ] **Step 3: Patch the two internal call sites**

在 `src/stockpool/backtesting/strategies.py` 找到:
- 调用 `build_panel(pool, cfg.factors, cfg.horizon)` 的那一行(在 `MLFactorStrategy._try_fit` 内部,大约 618 行附近)
- 调用 `build_factor_matrix(daily_df, self.cfg.factors)` 的那一行(大约 481 行附近)

把这两处分别改成:

```python
# pooled (_try_fit 内):
X_pool, y_pool = build_panel(
    pool, cfg.factors, cfg.horizon,
    mask_config=cfg.mask,
)

# per_stock (return 那一行):
return build_factor_matrix(daily_df, self.cfg.factors, mask_config=self.cfg.mask)
```

注意:`cfg` 在 `_try_fit` 内是局部别名(通常是 `cfg = self.cfg`),根据上下文调整为 `self.cfg.mask`。

如果 `_ensure_pooled_xy_long` 内部也有 `build_panel(...)` 或 `compute_factor_panel(...)`(看 search 结果 line 646),同样加 `mask_config=self.cfg.mask` 或 `mask=...`(若直接调 compute_factor_panel,先用 `compute_tradability_mask(panel, self.cfg.mask)` 算出 mask 再传)。

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy_mask.py -v
.venv/Scripts/python.exe -m pytest tests/test_ml_strategy.py tests/test_ml_strategy_panel.py tests/test_ml_strategy_panel_fit_reuse.py -v
```
Expected: 全部 PASS。

- [ ] **Step 5: Regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add src/stockpool/backtesting/strategies.py tests/test_ml_strategy_mask.py
git commit -m "feat(mask): MLFactorStrategy threads cfg.mask through build_panel + build_factor_matrix (task 14/18)"
```

---

## Task 15: `cli._prepare_ml_pool` passes `mask_config`

**Files:**
- Modify: `src/stockpool/cli.py`(找到 `_prepare_ml_pool` 函数)

- [ ] **Step 1: Inspect _prepare_ml_pool**

```bash
.venv/Scripts/python.exe -c "from stockpool import cli; import inspect; print(inspect.getsourcefile(cli._prepare_ml_pool)); print(inspect.getsourcelines(cli._prepare_ml_pool)[1])"
```
记录函数所在行号。

- [ ] **Step 2: Modify `_prepare_ml_pool`**

在 `_prepare_ml_pool` 中,凡是调用 `load_or_build_factor_panel(...)` 或
`build_factor_panel(...)` 的地方,加 kwarg `mask_config=cfg.strategy.ml_factor.mask`。

例如:
```python
factor_panel, close_panel = load_or_build_factor_panel(
    factor_names=cfg.strategy.ml_factor.factors,
    pool_data=pool_data,
    cache_dir=cfg.data.cache_dir,
    refresh=refresh_factor_panel,
    mask_config=cfg.strategy.ml_factor.mask,   # ← 新增
)
```

- [ ] **Step 3: Smoke test via existing CLI test**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cli_backtest.py -v
```
Expected: 全部 PASS(因为 default mask.enabled=False → 旧行为)。

- [ ] **Step 4: Manual end-to-end smoke (mask 启用)**

临时编辑 `config.yaml`,在 `strategy.ml_factor` 加:
```yaml
    mask:
      enabled: true
```
跑:
```bash
.venv/Scripts/python.exe -m stockpool backtest --config config.yaml --refresh-factor-panel
```
Expected: 不报错,产出 `reports/backtest/<date>.html`。验证完后**还原 config.yaml**(`enabled: false` 或删掉 mask 段)。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/cli.py
git commit -m "feat(mask): cli._prepare_ml_pool propagates mask_config (task 15/18)"
```

---

## Task 16: End-to-end backtest with mask enabled

**Files:**
- 临时:`config_mask_test.yaml`(基于 config.yaml 复制,加 mask.enabled)

- [ ] **Step 1: Create temp config**

```bash
cp config.yaml config_mask_test.yaml
```
手动编辑 `config_mask_test.yaml`,在 `strategy.ml_factor` 段加:
```yaml
    mask:
      enabled: true
```

- [ ] **Step 2: Run backtest**

```bash
.venv/Scripts/python.exe -m stockpool backtest --config config_mask_test.yaml --refresh-factor-panel
```
Expected:
- 不报错
- 控制台输出包含训练样本数(用 grep 看下,与 enabled=false 时对比)
- 产出 `reports/backtest/<date>.html`

- [ ] **Step 3: Sanity check the output**

打开 HTML 报告,检查:
- 净值曲线非全平(即 strategy 实际有交易)
- Sharpe 数值合理(不一定比 baseline 高,但不应是 NaN 或 ±∞)

- [ ] **Step 4: Cleanup temp config**

```bash
rm config_mask_test.yaml
```

- [ ] **Step 5: No commit needed (temp test)**

---

## Task 17: A/B verification with `ab_mask.yaml`

**Files:**
- Create: `ab_mask.yaml`

- [ ] **Step 1: Create `ab_mask.yaml`**

写到项目根目录 `ab_mask.yaml`:

```yaml
base_config: config.yaml

arms:
  baseline:
    strategy:
      ml_factor:
        mask:
          enabled: false

  with_mask:
    strategy:
      ml_factor:
        mask:
          enabled: true
```

- [ ] **Step 2: Run A/B**

```bash
.venv/Scripts/python.exe -m stockpool ab --config ab_mask.yaml
```
Expected:
- 不报错
- 产出 `reports/ab/<date>.html`,含两 arm 对比

- [ ] **Step 3: Record results**

打开 HTML 报告,记录两 arm 的:
- 平均 Sharpe
- 平均最大回撤
- 胜出股票数(with_mask 胜出多少只)

加到 `docs/research/2026-05-31-a-share-quant-survey-comparison.md` 末尾(或单独新文件 `docs/ab_validation_results.md`)。

- [ ] **Step 4: Commit (ab_mask.yaml + 结果记录)**

```bash
git add ab_mask.yaml docs/research/2026-05-31-a-share-quant-survey-comparison.md
git commit -m "feat(mask): A/B verification ab_mask.yaml + results (task 17/18)"
```

---

## Task 18: Update CLAUDE.md + README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md**

打开 `CLAUDE.md`,做以下编辑:

1. **"模块地图"表** 中 `panel.py` 行,扩描述加:
   > `compute_tradability_mask` + `apply_mask` + `_limit_threshold` + `_listing_mask` 支持按板块(主板 ±10% / 创业板 ±20%)的可交易性 mask,用于因子输入侧 NaN 化,paper B mask-first 实现

2. **"配置 (`config.yaml`)"** 段,`strategy.ml_factor` 描述里加:
   > **`mask`**(F? PR-? 新增):tradability mask 子段。`enabled`(默认 `false` 向后兼容)/ `limit_up_threshold_main`(主板沪深 ±10%,default 0.098)/ `limit_up_threshold_chinext`(创业板+科创 ±20%,default 0.198)/ `limit_up_threshold_bse`(北交所 ±30%,留兜底,default 0.298)/ `min_listing_days`(default 252)。启用后所有 panel-based 因子在算之前会把涨跌停日/停牌日/新上市头 N 天的 OHLCV 置 NaN,并对训练标签做双向检查(t 和 t+horizon 都需 mask=True)。改 mask 任一字段 sig hash 变化 → 自动让旧 `factor_panels/<sig>/` 和 `ml_models/` 缓存失效。

3. **"测试"表** 加 3 行:
   - `test_panel_mask.py` | `_limit_threshold` 板块映射 + `_listing_mask` 成熟/新股 + `compute_tradability_mask` 三条件 + `apply_mask` NaN-out 正确性 |
   - `test_ops_mask_nan_safe.py` | ts_mean/sum/std/product/decay_linear NaN 输入下结果合理 + 全 valid 输入不变 |
   - `test_ml_strategy_mask.py` | compute_factor_panel / forward_return_panel / build_factor_panel / build_panel / build_factor_matrix 各层 mask 参数语义 + 端到端 MLFactorStrategy 集成 |

4. **"已知不支持的能力"** 段,在 portfolio framework PR-4 限制后加:
   > **可交易性 mask**:本 PR 落地了 `mask_price`(close-side 因子输入清洁),完整 mask 含义见 `docs/superpowers/specs/2026-05-31-tradability-mask-design.md`。`mask_exec`(open-side 执行可填性,即开盘涨停 fill guard)未落地 — 单独 PR 处理。回测引擎仍假设 `open[t+1]` 一定可成交,这一假设在涨停封板的极少数情况下偏乐观。

- [ ] **Step 2: Update README.md**

打开 `README.md`,在 "常用命令" 段或配置示例部分,加一段:

```markdown
### Tradability mask(可选,opt-in)

在 `config.yaml` 的 `strategy.ml_factor` 段加:

\`\`\`yaml
strategy:
  name: ml_factor
  ml_factor:
    # ...其他字段
    mask:
      enabled: true   # 启用 paper B mask-first(涨跌停/停牌/新上市股的 close 不进 factor 滚动)
\`\`\`

启用后所有 panel-based 因子在算之前会清掉非可交易日的样本,训练样本数会下降 1-3%,
Sharpe 在大样本上预期提升 0.1-0.4(论文 B 在真实 A 股 2022-2024 报告 +0.44)。
缓存自动失效 — 翻 `enabled` 会触发 factor_panel + ml_models 重建。
```

- [ ] **Step 3: Commit docs**

```bash
git add CLAUDE.md README.md
git commit -m "docs(mask): update CLAUDE.md + README.md for tradability mask (task 18/18)"
```

---

## Final: Push and review

- [ ] **Step 1: Run full test suite one final time**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: 374 + ~43 新增 (test_panel_mask 12 + test_ops_mask_nan_safe 10 +
test_ml_strategy_mask 14 + test_config 5 + test_factor_panel_cache 2)
= ~417 PASS,0 FAIL。

- [ ] **Step 2: Show git log of this branch**

```bash
git log --oneline feat/composite-backtest ^origin/main | head -30
```
Verify all 18 task commits + ab_mask + docs are present.

- [ ] **Step 3: Push (ask user first)**

```bash
git push origin feat/composite-backtest
```

- [ ] **Step 4: Summary message to user**

Output:
- 18 task commits 推送到 `feat/composite-backtest`
- 21 个新测试全过
- `ab_mask.yaml` A/B 验证结果(Sharpe Δ、胜出数)
- mask 启用步骤(`config.yaml` 加 `mask.enabled: true`)
- 建议下一步:在更大股池上(training_universe=all)再跑一次 A/B 验证真实信号
