# A 股养龙股池技术信号分析工具 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个命令行 Python 工具,读取 `config.yaml` 中的 A 股股池,从 AKShare 拉日 K + 周 K,计算技术指标全套(MA/MACD/KDJ/RSI/BOLL/量能/突破),给每只股打综合分,并产出带 pyecharts 交互图表的单页 HTML 日报。

**Architecture:** 模块化 Python 包,数据/指标/信号/回测/报告 5 层职责分离。指标和信号层是纯函数,易测;数据层用 Parquet 缓存避免重复请求;报告层用 pyecharts 渲染独立 HTML(内联 JS,离线可看)。

**Tech Stack:** Python 3.10+, AKShare, pandas, numpy, pyarrow(Parquet), pyecharts 2.x, pydantic, pyyaml, pytest

---

## 设计参考

实施前先读:`docs/superpowers/specs/2026-05-17-a-share-signal-tool-design.md`

- § 5:综合打分规则(权重表、共振合成、阈值)
- § 7:`config.yaml` 完整字段定义
- § 9:错误处理矩阵

---

## 目录与文件总览

```
stockpool/                       # 项目根(已 git init)
├── pyproject.toml               # 创建 — 元数据 + 依赖 + console_script
├── config.yaml                  # 创建 — 用户唯一编辑文件
├── .gitignore                   # 已存在
├── README.md                    # 创建 — 使用说明
├── src/
│   └── stockpool/
│       ├── __init__.py          # 创建 — 版本号
│       ├── config.py            # 创建 — pydantic schema + load_config()
│       ├── fetcher.py           # 创建 — fetch_daily() + resample_to_weekly() + Parquet 缓存
│       ├── indicators.py        # 创建 — add_ma/add_macd/add_kdj/add_rsi/add_boll/add_volume/add_breakout
│       ├── signals.py           # 创建 — detect_signals() + score_stock() + combine_daily_weekly()
│       ├── backtest.py          # 创建 — compute_hit_rates()
│       ├── report.py            # 创建 — build_stock_chart() + build_overview_table() + render_html()
│       └── cli.py               # 创建 — argparse main + trading-day check + 串接
├── tests/
│   ├── conftest.py              # 创建 — 共享 fixture(synthetic OHLCV)
│   ├── test_config.py           # 创建
│   ├── test_fetcher.py          # 创建
│   ├── test_indicators.py       # 创建
│   ├── test_signals.py          # 创建
│   ├── test_backtest.py         # 创建
│   └── test_report_smoke.py     # 创建
├── scripts/
│   └── stockpool_task.xml       # 创建 — Windows Task Scheduler 模板
├── data/                        # 运行时生成,.gitignore
└── reports/                     # 运行时生成,.gitignore
```

---

## Task 1: 项目脚手架 + 依赖安装

**Files:**
- Create: `pyproject.toml`
- Create: `src/stockpool/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 创建 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "stockpool"
version = "0.1.0"
description = "A-share dragon-pool technical signal analyzer"
requires-python = ">=3.10"
dependencies = [
    "akshare>=1.12",
    "pandas>=2.0",
    "numpy>=1.24",
    "pyarrow>=14.0",
    "pyecharts>=2.0",
    "pyyaml>=6.0",
    "pydantic>=2.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-mock>=3.12",
]

[project.scripts]
stockpool = "stockpool.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 2: 创建包入口**

`src/stockpool/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: 空文件(touch 即可)。

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_daily() -> pd.DataFrame:
    """30 trading days of synthetic OHLCV — deterministic, computable by hand."""
    dates = pd.date_range("2026-01-02", periods=30, freq="B")
    close = np.array(
        [10.0, 10.2, 10.5, 10.3, 10.6, 10.8, 11.0, 10.9, 11.2, 11.5,
         11.4, 11.6, 11.8, 12.0, 11.9, 12.1, 12.3, 12.5, 12.4, 12.6,
         12.8, 13.0, 12.9, 13.1, 13.3, 13.5, 13.4, 13.6, 13.8, 14.0]
    )
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.full(30, 1_000_000, dtype=float),
    })
```

- [ ] **Step 3: 创建虚拟环境并安装**

```bash
cd "C:/Users/Administrator/Desktop/claude"
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -e ".[dev]"
```

Expected: 全部安装成功,无 error。akshare 安装较慢(2-3 分钟),正常。

- [ ] **Step 4: 验证 pytest 能跑(0 测试)**

```bash
.venv/Scripts/python -m pytest
```

Expected: `no tests ran` 或 `collected 0 items`,**退出码 5**(pytest 对 0 测试约定),不是 error。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ tests/__init__.py tests/conftest.py
git commit -m "chore: project scaffold with pyproject.toml and pytest"
```

---

## Task 2: 配置加载 (`config.py`)

**Files:**
- Create: `src/stockpool/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_config.py`:
```python
import pytest
import yaml
from pydantic import ValidationError
from stockpool.config import load_config, AppConfig


def _minimal_yaml() -> dict:
    """Smallest valid config — every required field present."""
    return {
        "stocks": [{"code": "605589", "name": "圣泉集团"}],
        "data": {"history_days": 500, "cache_dir": "data", "force_refresh": False},
        "indicators": {
            "ma_periods": [5, 10, 20, 60],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "kdj": {"n": 9, "m1": 3, "m2": 3},
            "rsi_periods": [6, 12, 24],
            "boll": {"n": 20, "k": 2},
            "volume_ratio_window": 5,
            "breakout_window": 20,
        },
        "weights": {
            "ma_cross_strong": 2, "ma_alignment": 1,
            "macd_cross_above_zero": 2, "macd_cross_below_zero": 1, "macd_histogram_expand": 1,
            "kdj_oversold_cross": 2, "kdj_overbought_cross": 2, "kdj_normal_cross": 1,
            "rsi_oversold": 1, "rsi_overbought": 1,
            "boll_band_touch": 2, "boll_mid_cross": 1,
            "volume_surge_bullish": 1, "volume_surge_bearish": 1,
            "breakout_new_high": 2, "breakout_new_low": 2,
        },
        "scoring": {
            "daily_weight": 0.7, "weekly_weight": 0.3,
            "resonance_bonus": 2, "resonance_daily_threshold": 3, "resonance_weekly_threshold": 1,
        },
        "verdicts": {"strong_buy": 6, "buy": 3, "sell": -3, "strong_sell": -6},
        "backtest": {"forward_days": [5, 10, 20]},
        "report": {"output_dir": "reports", "keep_history": True, "klines_to_show": 120},
    }


def test_load_valid_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(_minimal_yaml()), encoding="utf-8")

    cfg = load_config(cfg_file)

    assert isinstance(cfg, AppConfig)
    assert len(cfg.stocks) == 1
    assert cfg.stocks[0].code == "605589"
    assert cfg.data.history_days == 500
    assert cfg.scoring.daily_weight == 0.7


def test_missing_required_field_raises(tmp_path):
    raw = _minimal_yaml()
    del raw["stocks"]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_invalid_type_raises(tmp_path):
    raw = _minimal_yaml()
    raw["data"]["history_days"] = "five hundred"   # 字符串非法
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_config_hash_is_stable(tmp_path):
    """同一份 config 的 hash 跨次加载一致(报告里展示用)。"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(_minimal_yaml()), encoding="utf-8")

    cfg1 = load_config(cfg_file)
    cfg2 = load_config(cfg_file)
    assert cfg1.content_hash == cfg2.content_hash
    assert len(cfg1.content_hash) == 8   # 短 hash
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: ImportError(`stockpool.config` 不存在)。

- [ ] **Step 3: 实现 `src/stockpool/config.py`**

```python
"""Config schema + loader. Pydantic does the validation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Stock(BaseModel):
    code: str
    name: str


class DataConfig(BaseModel):
    history_days: int = Field(gt=0)
    cache_dir: str
    force_refresh: bool = False


class MACDConfig(BaseModel):
    fast: int
    slow: int
    signal: int


class KDJConfig(BaseModel):
    n: int
    m1: int
    m2: int


class BOLLConfig(BaseModel):
    n: int
    k: float


class IndicatorsConfig(BaseModel):
    ma_periods: list[int]
    macd: MACDConfig
    kdj: KDJConfig
    rsi_periods: list[int]
    boll: BOLLConfig
    volume_ratio_window: int
    breakout_window: int


class WeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 拼错字段名要立刻报错
    ma_cross_strong: int
    ma_alignment: int
    macd_cross_above_zero: int
    macd_cross_below_zero: int
    macd_histogram_expand: int
    kdj_oversold_cross: int
    kdj_overbought_cross: int
    kdj_normal_cross: int
    rsi_oversold: int
    rsi_overbought: int
    boll_band_touch: int
    boll_mid_cross: int
    volume_surge_bullish: int
    volume_surge_bearish: int
    breakout_new_high: int
    breakout_new_low: int


class ScoringConfig(BaseModel):
    daily_weight: float
    weekly_weight: float
    resonance_bonus: int
    resonance_daily_threshold: int
    resonance_weekly_threshold: int


class VerdictsConfig(BaseModel):
    strong_buy: int
    buy: int
    sell: int
    strong_sell: int


class BacktestConfig(BaseModel):
    forward_days: list[int]


class ReportConfig(BaseModel):
    output_dir: str
    keep_history: bool
    klines_to_show: int


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

    content_hash: str = ""


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config and validate against schema.

    Raises pydantic.ValidationError on missing fields or wrong types.
    """
    raw_bytes = Path(path).read_bytes()
    parsed = yaml.safe_load(raw_bytes)
    cfg = AppConfig.model_validate(parsed)
    cfg.content_hash = hashlib.sha256(raw_bytes).hexdigest()[:8]
    return cfg
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/config.py tests/test_config.py
git commit -m "feat(config): pydantic schema + YAML loader with content hash"
```

---

## Task 3: 默认 `config.yaml`

**Files:**
- Create: `config.yaml`
- Modify: `tests/test_config.py` (加一个加载真实 config.yaml 的回归测试)

- [ ] **Step 1: 创建 `config.yaml`**

按 spec § 7 的完整内容(略,直接复制 spec 中的代码块到项目根 `config.yaml`)。

⚠️ **注意:**`config.yaml` 在项目根,**不是** `src/stockpool/` 下。这样用户编辑路径直观,不要放进 package。

- [ ] **Step 2: 加回归测试**

在 `tests/test_config.py` 末尾追加:
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_default_config_yaml_loads():
    """Sanity check: 仓库里的 config.yaml 自身合法。"""
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    assert len(cfg.stocks) >= 1
    assert all(len(s.code) == 6 for s in cfg.stocks)
```

- [ ] **Step 3: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: 5 passed。

- [ ] **Step 4: Commit**

```bash
git add config.yaml tests/test_config.py
git commit -m "feat(config): add default config.yaml with initial 8-stock pool"
```

---

## Task 4: 数据 fetcher (`fetcher.py`)

**Files:**
- Create: `src/stockpool/fetcher.py`
- Create: `tests/test_fetcher.py`

**职责:**
- `fetch_daily(code, history_days, cache_dir, force_refresh=False) -> pd.DataFrame`
  - 检查 `{cache_dir}/{code}_daily.parquet` 最新日期
  - 若 ≥ history_days 内最新交易日 → 直接读缓存
  - 否则增量调用 `akshare.stock_zh_a_hist` 拉数据,合并写回 Parquet
  - 返回最近 history_days 根日 K 的 DataFrame
- `resample_to_weekly(daily_df) -> pd.DataFrame`
  - 用 pandas resample('W-FRI') 转周 K(open=首日 open,high=max,low=min,close=末日 close,volume=sum)

- [ ] **Step 1: 写失败的测试(用 mock,不打真接口)**

`tests/test_fetcher.py`:
```python
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockpool.fetcher import fetch_daily, resample_to_weekly


def _make_akshare_df(start: str, periods: int) -> pd.DataFrame:
    """AKShare stock_zh_a_hist 的列结构(中文列名)。"""
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": np.linspace(10, 11, periods),
        "收盘": np.linspace(10.1, 11.1, periods),
        "最高": np.linspace(10.3, 11.3, periods),
        "最低": np.linspace(9.9, 10.9, periods),
        "成交量": np.full(periods, 1_000_000),
        "成交额": np.full(periods, 1e7),
        "振幅": np.full(periods, 1.0),
        "涨跌幅": np.full(periods, 0.1),
        "涨跌额": np.full(periods, 0.01),
        "换手率": np.full(periods, 0.5),
    })


def test_fetch_creates_cache(tmp_path):
    fake = _make_akshare_df("2026-01-02", 30)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake) as mocked:
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    assert mocked.called
    assert len(df) == 30
    # 列名已标准化为英文
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    # 缓存文件已写
    assert (tmp_path / "605589_daily.parquet").exists()


def test_second_call_uses_cache_no_request(tmp_path):
    """缓存覆盖请求窗口时,不再调用 akshare。"""
    fake = _make_akshare_df("2026-01-02", 60)

    # 第一次填缓存
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    # 第二次:缓存里有 60 天,要 30 天就够,不该再调
    with patch("stockpool.fetcher.ak.stock_zh_a_hist") as mocked:
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path)
        assert not mocked.called

    assert len(df) == 30


def test_force_refresh_bypasses_cache(tmp_path):
    fake = _make_akshare_df("2026-01-02", 30)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake):
        fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake) as mocked:
        fetch_daily("605589", history_days=30, cache_dir=tmp_path, force_refresh=True)
        assert mocked.called


def test_akshare_retry_then_succeed(tmp_path):
    """网络偶尔失败 → 重试 3 次。"""
    fake = _make_akshare_df("2026-01-02", 30)
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("rate limit")
        return fake

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=flaky), \
         patch("stockpool.fetcher.time.sleep"):   # 别真睡
        df = fetch_daily("605589", history_days=30, cache_dir=tmp_path)

    assert calls["n"] == 3
    assert len(df) == 30


def test_akshare_all_retries_fail_uses_cache_or_raises(tmp_path):
    """无缓存可退路时,3 次失败抛错。"""
    with patch("stockpool.fetcher.ak.stock_zh_a_hist", side_effect=ConnectionError("down")), \
         patch("stockpool.fetcher.time.sleep"):
        with pytest.raises(ConnectionError):
            fetch_daily("605589", history_days=30, cache_dir=tmp_path)


def test_resample_to_weekly():
    daily = pd.DataFrame({
        "date": pd.date_range("2026-01-05", periods=10, freq="B"),   # 周一开始
        "open":   [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9],
        "high":   [10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4],
        "low":    [9.5,  9.6,  9.7,  9.8,  9.9,  10.0, 10.1, 10.2, 10.3, 10.4],
        "close":  [10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1],
        "volume": [1_000_000] * 10,
    })

    weekly = resample_to_weekly(daily)

    assert len(weekly) == 2
    # 第一周(周一~周五):open=10.0 (周一开盘), high=10.9 (周五最高), low=9.5, close=10.6 (周五收盘)
    assert weekly.iloc[0]["open"] == pytest.approx(10.0)
    assert weekly.iloc[0]["high"] == pytest.approx(10.9)
    assert weekly.iloc[0]["low"] == pytest.approx(9.5)
    assert weekly.iloc[0]["close"] == pytest.approx(10.6)
    assert weekly.iloc[0]["volume"] == 5_000_000
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_fetcher.py -v
```

Expected: ImportError。

- [ ] **Step 3: 实现 `src/stockpool/fetcher.py`**

```python
"""AKShare 数据获取 + Parquet 本地缓存."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import akshare as ak
import pandas as pd

log = logging.getLogger(__name__)

_AKSHARE_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
}

_RETRY_DELAYS = [2, 4, 8]   # 指数退避秒数


def _cache_path(cache_dir: str | Path, code: str) -> Path:
    return Path(cache_dir) / f"{code}_daily.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """AKShare 返回中文列名 → 统一英文 + date 转 datetime + 排序去重。"""
    out = df.rename(columns=_AKSHARE_COLUMN_MAP).copy()
    keep = ["date", "open", "high", "low", "close", "volume"]
    out = out[keep]
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return out


def _fetch_from_akshare(code: str, start: str | None = None) -> pd.DataFrame:
    """带重试地调 AKShare. start: YYYYMMDD 字符串,None = 全量."""
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start or "19900101",
                end_date="20991231",
                adjust="qfq",   # 前复权
            )
            return _normalize(raw)
        except Exception as e:
            last_err = e
            log.warning("AKShare attempt %d/%d for %s failed: %s",
                        attempt, len(_RETRY_DELAYS), code, e)
            if attempt < len(_RETRY_DELAYS):
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def fetch_daily(
    code: str,
    history_days: int,
    cache_dir: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    返回最近 `history_days` 根日 K(英文列名 DataFrame)。

    使用本地 Parquet 缓存,只在缓存不够时增量请求。
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, code)

    cached: pd.DataFrame | None = None
    if cache_file.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache_file)
        except Exception as e:
            log.warning("Cache %s corrupt (%s), refetching", cache_file, e)
            cache_file.unlink(missing_ok=True)
            cached = None

    need_fetch = (
        force_refresh
        or cached is None
        or len(cached) < history_days
    )

    if need_fetch:
        start = None
        if cached is not None and not force_refresh:
            last = cached["date"].max()
            start = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
        fresh = _fetch_from_akshare(code, start=start)
        if cached is not None and not force_refresh:
            combined = pd.concat([cached, fresh]).drop_duplicates("date").sort_values("date")
        else:
            combined = fresh
        combined = combined.reset_index(drop=True)
        combined.to_parquet(cache_file, index=False)
        cached = combined

    return cached.tail(history_days).reset_index(drop=True)


def resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """日 K → 周 K(W-FRI:每周以周五结束)."""
    df = daily.copy()
    df = df.set_index(pd.DatetimeIndex(df["date"]))
    weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    weekly = weekly.reset_index().rename(columns={"index": "date"})
    return weekly
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_fetcher.py -v
```

Expected: 6 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/fetcher.py tests/test_fetcher.py
git commit -m "feat(fetcher): AKShare client with Parquet cache and retry"
```

---

## Task 5: MA 指标 (`indicators.py` 第一部分)

**Files:**
- Create: `src/stockpool/indicators.py`
- Create: `tests/test_indicators.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_indicators.py`:
```python
import numpy as np
import pandas as pd
import pytest

from stockpool.indicators import add_ma


def test_ma_basic(synthetic_daily):
    df = add_ma(synthetic_daily, periods=[5, 10, 20])

    # MA5 第 5 行 (index 4) = mean(close[0:5])
    expected_ma5_at_4 = synthetic_daily["close"].iloc[:5].mean()
    assert df["ma5"].iloc[4] == pytest.approx(expected_ma5_at_4)

    # MA5 前 4 行应为 NaN
    assert df["ma5"].iloc[:4].isna().all()

    # MA10/MA20 列存在
    assert "ma10" in df.columns
    assert "ma20" in df.columns


def test_ma_preserves_original_columns(synthetic_daily):
    df = add_ma(synthetic_daily, periods=[5])
    for col in ["date", "open", "high", "low", "close", "volume"]:
        assert col in df.columns
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py -v
```

Expected: ImportError。

- [ ] **Step 3: 实现 `src/stockpool/indicators.py`(初版,只含 MA)**

```python
"""Pure indicator functions: DataFrame in → DataFrame out (with added columns).

Each function NEVER mutates input — always returns a copy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_ma(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Simple moving averages on close."""
    out = df.copy()
    for p in periods:
        out[f"ma{p}"] = out["close"].rolling(p).mean()
    return out
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py -v
```

Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/indicators.py tests/test_indicators.py
git commit -m "feat(indicators): moving averages (MA)"
```

---

## Task 6: MACD 指标

**Files:**
- Modify: `src/stockpool/indicators.py`
- Modify: `tests/test_indicators.py`

- [ ] **Step 1: 加测试**

追加到 `tests/test_indicators.py`:
```python
from stockpool.indicators import add_macd


def test_macd_columns_present(synthetic_daily):
    df = add_macd(synthetic_daily, fast=12, slow=26, signal=9)
    assert {"macd_dif", "macd_dea", "macd_hist"}.issubset(df.columns)


def test_macd_values_match_textbook_formula(synthetic_daily):
    """对照 EMA 公式手算最后几行."""
    df = add_macd(synthetic_daily, fast=12, slow=26, signal=9)

    ema_fast = synthetic_daily["close"].ewm(span=12, adjust=False).mean()
    ema_slow = synthetic_daily["close"].ewm(span=26, adjust=False).mean()
    expected_dif = ema_fast - ema_slow
    expected_dea = expected_dif.ewm(span=9, adjust=False).mean()

    assert df["macd_dif"].iloc[-1] == pytest.approx(expected_dif.iloc[-1])
    assert df["macd_dea"].iloc[-1] == pytest.approx(expected_dea.iloc[-1])
    assert df["macd_hist"].iloc[-1] == pytest.approx(
        2 * (expected_dif.iloc[-1] - expected_dea.iloc[-1])
    )
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py::test_macd_columns_present -v
```

Expected: ImportError(`add_macd` 不存在)。

- [ ] **Step 3: 加实现**

在 `src/stockpool/indicators.py` 末尾追加:
```python
def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD: DIF = EMA_fast - EMA_slow; DEA = EMA(DIF, signal); HIST = 2*(DIF-DEA)."""
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd_dif"] = ema_fast - ema_slow
    out["macd_dea"] = out["macd_dif"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = 2 * (out["macd_dif"] - out["macd_dea"])
    return out
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py -v
```

Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/indicators.py tests/test_indicators.py
git commit -m "feat(indicators): MACD (DIF/DEA/HIST)"
```

---

## Task 7: KDJ 指标

**Files:**
- Modify: `src/stockpool/indicators.py`
- Modify: `tests/test_indicators.py`

- [ ] **Step 1: 加测试**

```python
from stockpool.indicators import add_kdj


def test_kdj_columns_and_range(synthetic_daily):
    df = add_kdj(synthetic_daily, n=9, m1=3, m2=3)
    assert {"kdj_k", "kdj_d", "kdj_j"}.issubset(df.columns)
    # K 和 D 在有数据的位置应该在 [0, 100] 范围(理论上 J 可以越界)
    valid = df.dropna(subset=["kdj_k", "kdj_d"])
    assert valid["kdj_k"].between(-50, 150).all()
    assert valid["kdj_d"].between(-50, 150).all()


def test_kdj_trending_up_pushes_k_high(synthetic_daily):
    """合成数据是单调上涨,K 应趋向 100."""
    df = add_kdj(synthetic_daily, n=9, m1=3, m2=3)
    assert df["kdj_k"].iloc[-1] > 70
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py::test_kdj_columns_and_range -v
```

Expected: ImportError。

- [ ] **Step 3: 加实现**

```python
def add_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ (中国市场惯用版): RSV → SMA → K/D/J."""
    out = df.copy()
    low_n = out["low"].rolling(n).min()
    high_n = out["high"].rolling(n).max()
    rsv = (out["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)   # 起始无数据时给中性 50

    # SMA(x, m, 1) = 前一日值 * (m-1)/m + 当前值 * 1/m,等价于 EMA(span=2m-1, adjust=False)
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d

    # 前 n-1 个无意义,置 NaN
    k.iloc[: n - 1] = np.nan
    d.iloc[: n - 1] = np.nan
    j.iloc[: n - 1] = np.nan

    out["kdj_k"] = k
    out["kdj_d"] = d
    out["kdj_j"] = j
    return out
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py -v
```

Expected: 6 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/indicators.py tests/test_indicators.py
git commit -m "feat(indicators): KDJ"
```

---

## Task 8: RSI 指标

**Files:**
- Modify: `src/stockpool/indicators.py`
- Modify: `tests/test_indicators.py`

- [ ] **Step 1: 加测试**

```python
from stockpool.indicators import add_rsi


def test_rsi_columns(synthetic_daily):
    df = add_rsi(synthetic_daily, periods=[6, 12, 24])
    assert {"rsi6", "rsi12", "rsi24"}.issubset(df.columns)


def test_rsi_monotonic_up_data_above_50(synthetic_daily):
    """上涨数据 RSI 应 > 50."""
    df = add_rsi(synthetic_daily, periods=[6])
    assert df["rsi6"].iloc[-1] > 50


def test_rsi_all_down_below_50():
    dates = pd.date_range("2026-01-02", periods=15, freq="B")
    close = np.linspace(20, 10, 15)   # 单调下跌
    df = pd.DataFrame({
        "date": dates, "open": close, "high": close + 0.1,
        "low": close - 0.1, "close": close, "volume": [1_000_000] * 15,
    })
    out = add_rsi(df, periods=[6])
    assert out["rsi6"].iloc[-1] < 50
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py::test_rsi_columns -v
```

Expected: ImportError。

- [ ] **Step 3: 加实现**

```python
def add_rsi(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Wilder's RSI: 100 - 100/(1 + RS),RS = avg_gain / avg_loss(用 SMMA/EWMA)."""
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for p in periods:
        avg_gain = gain.ewm(alpha=1 / p, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / p, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.fillna(50)   # 起始无变化时给中性
        rsi.iloc[:p] = np.nan
        out[f"rsi{p}"] = rsi
    return out
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py -v
```

Expected: 9 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/indicators.py tests/test_indicators.py
git commit -m "feat(indicators): RSI (Wilder smoothing)"
```

---

## Task 9: BOLL 指标

**Files:**
- Modify: `src/stockpool/indicators.py`
- Modify: `tests/test_indicators.py`

- [ ] **Step 1: 加测试**

```python
from stockpool.indicators import add_boll


def test_boll_three_lines(synthetic_daily):
    df = add_boll(synthetic_daily, n=20, k=2)
    assert {"boll_up", "boll_mid", "boll_low"}.issubset(df.columns)


def test_boll_mid_equals_ma_n(synthetic_daily):
    df = add_boll(synthetic_daily, n=20, k=2)
    expected_mid = synthetic_daily["close"].rolling(20).mean()
    assert df["boll_mid"].iloc[-1] == pytest.approx(expected_mid.iloc[-1])


def test_boll_up_above_mid(synthetic_daily):
    df = add_boll(synthetic_daily, n=20, k=2)
    valid = df.dropna(subset=["boll_up", "boll_mid"])
    assert (valid["boll_up"] >= valid["boll_mid"]).all()
    assert (valid["boll_low"] <= valid["boll_mid"]).all()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py::test_boll_three_lines -v
```

Expected: ImportError。

- [ ] **Step 3: 加实现**

```python
def add_boll(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: mid = MA(n), up/low = mid ± k × stddev."""
    out = df.copy()
    mid = out["close"].rolling(n).mean()
    std = out["close"].rolling(n).std(ddof=0)
    out["boll_mid"] = mid
    out["boll_up"] = mid + k * std
    out["boll_low"] = mid - k * std
    return out
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py -v
```

Expected: 12 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/indicators.py tests/test_indicators.py
git commit -m "feat(indicators): Bollinger Bands"
```

---

## Task 10: 量能 + 突破标记 + 总入口 `add_all`

**Files:**
- Modify: `src/stockpool/indicators.py`
- Modify: `tests/test_indicators.py`

- [ ] **Step 1: 加测试**

```python
from stockpool.indicators import add_volume_ratio, add_breakout_markers, add_all


def test_volume_ratio(synthetic_daily):
    df = add_volume_ratio(synthetic_daily, window=5)
    assert "vol_ratio5" in df.columns
    # 合成数据 volume 全相等 → 比值 = 1.0
    assert df["vol_ratio5"].dropna().iloc[-1] == pytest.approx(1.0)


def test_breakout_markers(synthetic_daily):
    df = add_breakout_markers(synthetic_daily, window=20)
    # 单调上涨,最后一行应是 20 日新高
    assert df["is_breakout_high"].iloc[-1] == True
    assert df["is_breakout_low"].iloc[-1] == False


def test_add_all_runs_everything(synthetic_daily):
    """add_all 是后续 signals 用的一站式入口."""
    from stockpool.config import IndicatorsConfig, MACDConfig, KDJConfig, BOLLConfig
    cfg = IndicatorsConfig(
        ma_periods=[5, 10, 20],
        macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3),
        rsi_periods=[6, 12],
        boll=BOLLConfig(n=20, k=2),
        volume_ratio_window=5,
        breakout_window=20,
    )
    df = add_all(synthetic_daily, cfg)
    expected = {"ma5", "ma10", "ma20", "macd_dif", "macd_dea", "macd_hist",
                "kdj_k", "kdj_d", "kdj_j", "rsi6", "rsi12",
                "boll_up", "boll_mid", "boll_low", "vol_ratio5",
                "is_breakout_high", "is_breakout_low"}
    assert expected.issubset(df.columns)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py::test_volume_ratio -v
```

Expected: ImportError。

- [ ] **Step 3: 加实现**

在 `src/stockpool/indicators.py` 末尾追加:
```python
def add_volume_ratio(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """vol_ratio_N = volume / MA_N(volume).shift(1) — 当日成交量 vs 过去 N 日均量."""
    out = df.copy()
    avg = out["volume"].rolling(window).mean().shift(1)
    out[f"vol_ratio{window}"] = out["volume"] / avg
    return out


def add_breakout_markers(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """收盘 == 过去 N 日最高 → 新高,反之新低."""
    out = df.copy()
    rolling_high = out["close"].rolling(window).max()
    rolling_low = out["close"].rolling(window).min()
    out["is_breakout_high"] = out["close"] >= rolling_high
    out["is_breakout_low"] = out["close"] <= rolling_low
    # 历史不足时关闭标记
    out.loc[: window - 2, "is_breakout_high"] = False
    out.loc[: window - 2, "is_breakout_low"] = False
    return out


def add_all(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """One-stop: apply every indicator according to config.

    cfg: stockpool.config.IndicatorsConfig
    """
    out = df
    out = add_ma(out, cfg.ma_periods)
    out = add_macd(out, cfg.macd.fast, cfg.macd.slow, cfg.macd.signal)
    out = add_kdj(out, cfg.kdj.n, cfg.kdj.m1, cfg.kdj.m2)
    out = add_rsi(out, cfg.rsi_periods)
    out = add_boll(out, cfg.boll.n, cfg.boll.k)
    out = add_volume_ratio(out, cfg.volume_ratio_window)
    out = add_breakout_markers(out, cfg.breakout_window)
    return out
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_indicators.py -v
```

Expected: 15 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/indicators.py tests/test_indicators.py
git commit -m "feat(indicators): volume ratio, breakout markers, add_all entry"
```

---

## Task 11: 信号触发 (`signals.py` 第一部分)

**Files:**
- Create: `src/stockpool/signals.py`
- Create: `tests/test_signals.py`

**职责:** 给一个**带指标列的 DataFrame**,扫**最后一根 K 线**,输出 `list[Trigger]`。每个 Trigger 是一个 dataclass。

- [ ] **Step 1: 写失败的测试**

`tests/test_signals.py`:
```python
import numpy as np
import pandas as pd
import pytest

from stockpool.signals import Trigger, detect_signals
from stockpool.config import WeightsConfig


@pytest.fixture
def default_weights() -> WeightsConfig:
    return WeightsConfig(
        ma_cross_strong=2, ma_alignment=1,
        macd_cross_above_zero=2, macd_cross_below_zero=1, macd_histogram_expand=1,
        kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1,
        boll_band_touch=2, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1,
        breakout_new_high=2, breakout_new_low=2,
    )


def _make_df_with_macd_golden_cross_above_zero() -> pd.DataFrame:
    """构造倒数两行:DIF 从 < DEA 翻到 > DEA,且都在零轴上方."""
    return pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=3, freq="B"),
        "open": [10, 10, 10], "high": [10, 10, 10],
        "low": [10, 10, 10], "close": [10, 10, 10], "volume": [1e6] * 3,
        "ma5": [9, 9.5, 10], "ma10": [8.5, 9, 9.5],
        "ma20": [8, 8.5, 9], "ma60": [7, 7.5, 8],
        "macd_dif": [0.3, 0.5, 0.8], "macd_dea": [0.4, 0.6, 0.7], "macd_hist": [-0.2, -0.2, 0.2],
        "kdj_k": [50, 55, 60], "kdj_d": [50, 53, 56], "kdj_j": [50, 59, 68],
        "rsi6": [50, 55, 60], "rsi12": [50, 53, 56], "rsi24": [50, 52, 54],
        "boll_up": [11, 11, 11], "boll_mid": [10, 10, 10], "boll_low": [9, 9, 9],
        "vol_ratio5": [1.0, 1.0, 1.0],
        "is_breakout_high": [False, False, False],
        "is_breakout_low": [False, False, False],
    })


def test_macd_golden_cross_above_zero_detected(default_weights):
    df = _make_df_with_macd_golden_cross_above_zero()
    triggers = detect_signals(df, default_weights)
    sigs = [t.signal_type for t in triggers]
    assert "macd_cross_above_zero" in sigs
    # 同时是 MA 多头排列(5>10>20>60)
    assert "ma_alignment_bull" in sigs


def test_oversold_kdj_with_cross():
    """J<20 + K上穿D 应触发强信号."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=3, freq="B"),
        "open": [10]*3, "high": [10]*3, "low": [10]*3, "close": [10]*3, "volume": [1e6]*3,
        "ma5": [10]*3, "ma10": [10]*3, "ma20": [10]*3, "ma60": [10]*3,
        "macd_dif": [0]*3, "macd_dea": [0]*3, "macd_hist": [0]*3,
        "kdj_k": [10, 12, 18], "kdj_d": [15, 14, 13], "kdj_j": [5, 8, 18],   # J<20 + 金叉
        "rsi6": [25]*3, "rsi12": [40]*3, "rsi24": [50]*3,
        "boll_up": [11]*3, "boll_mid": [10]*3, "boll_low": [9]*3,
        "vol_ratio5": [1.0]*3,
        "is_breakout_high": [False]*3, "is_breakout_low": [False]*3,
    })
    weights = WeightsConfig(
        ma_cross_strong=2, ma_alignment=1, macd_cross_above_zero=2, macd_cross_below_zero=1,
        macd_histogram_expand=1, kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1, boll_band_touch=2, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1, breakout_new_high=2, breakout_new_low=2,
    )
    triggers = detect_signals(df, weights)
    sigs = [t.signal_type for t in triggers]
    assert "kdj_oversold_cross" in sigs


def test_volume_surge_with_red_candle_is_bearish(default_weights):
    """放量阴线 → bearish 信号."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=2, freq="B"),
        "open":   [10, 11],
        "high":   [10, 11],
        "low":    [9, 9],
        "close":  [10, 9.5],   # 阴线(close < open)
        "volume": [1e6, 2e6],
        "ma5": [10, 10], "ma10": [10, 10], "ma20": [10, 10], "ma60": [10, 10],
        "macd_dif": [0]*2, "macd_dea": [0]*2, "macd_hist": [0]*2,
        "kdj_k": [50]*2, "kdj_d": [50]*2, "kdj_j": [50]*2,
        "rsi6": [50]*2, "rsi12": [50]*2, "rsi24": [50]*2,
        "boll_up": [12]*2, "boll_mid": [10]*2, "boll_low": [8]*2,
        "vol_ratio5": [1.0, 2.0],   # 量比 2 > 1.5
        "is_breakout_high": [False]*2, "is_breakout_low": [False]*2,
    })
    triggers = detect_signals(df, default_weights)
    sigs = [t.signal_type for t in triggers]
    assert "volume_surge_bearish" in sigs
    bearish_trigger = [t for t in triggers if t.signal_type == "volume_surge_bearish"][0]
    assert bearish_trigger.direction == -1
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_signals.py -v
```

Expected: ImportError。

- [ ] **Step 3: 实现 `src/stockpool/signals.py`(第一部分:detect_signals)**

```python
"""Signal detection + composite scoring.

The full rubric lives in spec § 5. Each detection function reads the last 1-2
rows of a DataFrame with indicator columns, and returns a `Trigger` or None.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpool.config import WeightsConfig


@dataclass
class Trigger:
    signal_type: str          # 例如 "macd_cross_above_zero"
    direction: int            # +1 看多, -1 看空
    weight: int               # 来自 WeightsConfig
    description: str          # 中文人类可读


def _golden_cross(prev_fast: float, prev_slow: float,
                  curr_fast: float, curr_slow: float) -> bool:
    return prev_fast <= prev_slow and curr_fast > curr_slow


def _dead_cross(prev_fast: float, prev_slow: float,
                curr_fast: float, curr_slow: float) -> bool:
    return prev_fast >= prev_slow and curr_fast < curr_slow


def detect_signals(df: pd.DataFrame, weights: WeightsConfig) -> list[Trigger]:
    """扫最后一根 K 线,返回触发的所有信号."""
    if len(df) < 2:
        return []

    triggers: list[Trigger] = []
    prev, curr = df.iloc[-2], df.iloc[-1]

    # === MA 金叉/死叉(5 上/下穿 20) ===
    if "ma5" in df.columns and "ma20" in df.columns:
        if _golden_cross(prev["ma5"], prev["ma20"], curr["ma5"], curr["ma20"]):
            triggers.append(Trigger("ma_cross_strong", +1, weights.ma_cross_strong,
                                    "MA5 上穿 MA20(金叉)"))
        elif _dead_cross(prev["ma5"], prev["ma20"], curr["ma5"], curr["ma20"]):
            triggers.append(Trigger("ma_cross_strong", -1, weights.ma_cross_strong,
                                    "MA5 下穿 MA20(死叉)"))

    # === MA 多头/空头排列 ===
    ma_cols = [c for c in ["ma5", "ma10", "ma20", "ma60"] if c in df.columns]
    if len(ma_cols) >= 3:
        vals = [curr[c] for c in ma_cols]
        if all(vals[i] > vals[i+1] for i in range(len(vals)-1)):
            triggers.append(Trigger("ma_alignment_bull", +1, weights.ma_alignment,
                                    "MA 多头排列(短>长)"))
        elif all(vals[i] < vals[i+1] for i in range(len(vals)-1)):
            triggers.append(Trigger("ma_alignment_bear", -1, weights.ma_alignment,
                                    "MA 空头排列(短<长)"))

    # === MACD ===
    if "macd_dif" in df.columns:
        cross_up = _golden_cross(prev["macd_dif"], prev["macd_dea"],
                                 curr["macd_dif"], curr["macd_dea"])
        cross_down = _dead_cross(prev["macd_dif"], prev["macd_dea"],
                                 curr["macd_dif"], curr["macd_dea"])
        above_zero = curr["macd_dif"] > 0

        if cross_up:
            if above_zero:
                triggers.append(Trigger("macd_cross_above_zero", +1,
                                        weights.macd_cross_above_zero,
                                        "MACD 零轴上方金叉(强多)"))
            else:
                triggers.append(Trigger("macd_cross_below_zero", +1,
                                        weights.macd_cross_below_zero,
                                        "MACD 零轴下方金叉(弱多)"))
        elif cross_down:
            if above_zero:
                triggers.append(Trigger("macd_cross_below_zero", -1,
                                        weights.macd_cross_below_zero,
                                        "MACD 零轴上方死叉(弱空)"))
            else:
                triggers.append(Trigger("macd_cross_above_zero", -1,
                                        weights.macd_cross_above_zero,
                                        "MACD 零轴下方死叉(强空)"))

        # 红/绿柱连续 3 日放大
        if len(df) >= 4:
            last3 = df["macd_hist"].iloc[-3:].tolist()
            if all(last3[i] > 0 for i in range(3)) and last3[2] > last3[1] > last3[0]:
                triggers.append(Trigger("macd_histogram_expand", +1,
                                        weights.macd_histogram_expand,
                                        "MACD 红柱连续 3 日放大"))
            elif all(last3[i] < 0 for i in range(3)) and last3[2] < last3[1] < last3[0]:
                triggers.append(Trigger("macd_histogram_expand", -1,
                                        weights.macd_histogram_expand,
                                        "MACD 绿柱连续 3 日放大"))

    # === KDJ ===
    if "kdj_k" in df.columns and "kdj_d" in df.columns:
        cross_up = _golden_cross(prev["kdj_k"], prev["kdj_d"],
                                 curr["kdj_k"], curr["kdj_d"])
        cross_down = _dead_cross(prev["kdj_k"], prev["kdj_d"],
                                 curr["kdj_k"], curr["kdj_d"])
        j_val = curr.get("kdj_j", 50)

        if cross_up:
            if j_val < 20:
                triggers.append(Trigger("kdj_oversold_cross", +1,
                                        weights.kdj_oversold_cross,
                                        f"KDJ 超卖金叉(J={j_val:.1f})"))
            else:
                triggers.append(Trigger("kdj_normal_cross", +1,
                                        weights.kdj_normal_cross,
                                        "KDJ 普通金叉"))
        elif cross_down:
            if j_val > 80:
                triggers.append(Trigger("kdj_overbought_cross", -1,
                                        weights.kdj_overbought_cross,
                                        f"KDJ 超买死叉(J={j_val:.1f})"))
            else:
                triggers.append(Trigger("kdj_normal_cross", -1,
                                        weights.kdj_normal_cross,
                                        "KDJ 普通死叉"))

    # === RSI ===
    if "rsi6" in df.columns:
        rsi6 = curr["rsi6"]
        if rsi6 < 20:
            triggers.append(Trigger("rsi_oversold", +1, weights.rsi_oversold,
                                    f"RSI6 超卖({rsi6:.1f})"))
        elif rsi6 > 80:
            triggers.append(Trigger("rsi_overbought", -1, weights.rsi_overbought,
                                    f"RSI6 超买({rsi6:.1f})"))

    # === BOLL ===
    if "boll_up" in df.columns:
        if prev["close"] <= prev["boll_low"] and curr["close"] > curr["boll_low"]:
            triggers.append(Trigger("boll_band_touch", +1, weights.boll_band_touch,
                                    "收盘上穿 BOLL 下轨(反弹)"))
        elif prev["close"] >= prev["boll_up"] and curr["close"] < curr["boll_up"]:
            triggers.append(Trigger("boll_band_touch", -1, weights.boll_band_touch,
                                    "收盘跌破 BOLL 上轨(回落)"))
        elif _golden_cross(prev["close"], prev["boll_mid"], curr["close"], curr["boll_mid"]):
            triggers.append(Trigger("boll_mid_cross", +1, weights.boll_mid_cross,
                                    "收盘上穿 BOLL 中轨"))
        elif _dead_cross(prev["close"], prev["boll_mid"], curr["close"], curr["boll_mid"]):
            triggers.append(Trigger("boll_mid_cross", -1, weights.boll_mid_cross,
                                    "收盘下穿 BOLL 中轨"))

    # === 量能 ===
    vol_ratio = curr.get("vol_ratio5", 1.0)
    if vol_ratio is not None and vol_ratio > 1.5:
        is_bullish_candle = curr["close"] > curr["open"]
        is_bearish_candle = curr["close"] < curr["open"]
        if is_bullish_candle:
            triggers.append(Trigger("volume_surge_bullish", +1,
                                    weights.volume_surge_bullish,
                                    f"放量阳线(量比 {vol_ratio:.2f})"))
        elif is_bearish_candle:
            triggers.append(Trigger("volume_surge_bearish", -1,
                                    weights.volume_surge_bearish,
                                    f"放量阴线(量比 {vol_ratio:.2f})"))

    # === 突破 ===
    if curr.get("is_breakout_high", False) and not prev.get("is_breakout_high", False):
        triggers.append(Trigger("breakout_new_high", +1, weights.breakout_new_high,
                                "收盘创 20 日新高"))
    if curr.get("is_breakout_low", False) and not prev.get("is_breakout_low", False):
        triggers.append(Trigger("breakout_new_low", -1, weights.breakout_new_low,
                                "收盘创 20 日新低"))

    return triggers
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_signals.py -v
```

Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/signals.py tests/test_signals.py
git commit -m "feat(signals): per-indicator trigger detection"
```

---

## Task 12: 综合打分 + 日周共振

**Files:**
- Modify: `src/stockpool/signals.py`
- Modify: `tests/test_signals.py`

- [ ] **Step 1: 加测试**

```python
from stockpool.signals import score_triggers, combine_daily_weekly, verdict_of
from stockpool.config import ScoringConfig, VerdictsConfig


def _make_scoring() -> ScoringConfig:
    return ScoringConfig(
        daily_weight=0.7, weekly_weight=0.3,
        resonance_bonus=2, resonance_daily_threshold=3, resonance_weekly_threshold=1,
    )


def _make_verdicts() -> VerdictsConfig:
    return VerdictsConfig(strong_buy=6, buy=3, sell=-3, strong_sell=-6)


def test_score_triggers_sum_with_cap():
    triggers = [
        Trigger("a", +1, 2, ""), Trigger("b", +1, 2, ""),
        Trigger("c", +1, 2, ""), Trigger("d", +1, 2, ""),
        Trigger("e", +1, 2, ""), Trigger("f", +1, 2, ""),   # 总分 12 → 截 10
    ]
    assert score_triggers(triggers) == 10


def test_score_triggers_mixed_signs():
    triggers = [Trigger("a", +1, 3, ""), Trigger("b", -1, 1, "")]
    assert score_triggers(triggers) == 2


def test_combine_no_resonance():
    cfg = _make_scoring()
    # daily=4, weekly=0 → 4*0.7 + 0*0.3 = 2.8,无共振
    assert combine_daily_weekly(4, 0, cfg) == pytest.approx(2.8)


def test_combine_with_bullish_resonance():
    cfg = _make_scoring()
    # daily=5, weekly=2 → 5*0.7 + 2*0.3 = 4.1,+共振 2 → 6.1
    assert combine_daily_weekly(5, 2, cfg) == pytest.approx(6.1)


def test_combine_with_bearish_resonance():
    cfg = _make_scoring()
    # daily=-4, weekly=-2 → -4*0.7 + -2*0.3 = -3.4,-共振 2 → -5.4
    assert combine_daily_weekly(-4, -2, cfg) == pytest.approx(-5.4)


def test_combine_caps_at_10():
    cfg = _make_scoring()
    # daily=10, weekly=10 → 10,加共振 2 → 12 → 截 10
    assert combine_daily_weekly(10, 10, cfg) == 10


def test_verdict_thresholds():
    v = _make_verdicts()
    assert verdict_of(7, v) == "strong_buy"
    assert verdict_of(5, v) == "buy"
    assert verdict_of(0, v) == "neutral"
    assert verdict_of(-4, v) == "sell"
    assert verdict_of(-7, v) == "strong_sell"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_signals.py::test_score_triggers_sum_with_cap -v
```

Expected: ImportError。

- [ ] **Step 3: 加实现**

在 `src/stockpool/signals.py` 末尾追加:
```python
from stockpool.config import ScoringConfig, VerdictsConfig


def score_triggers(triggers: list[Trigger]) -> int:
    """Sum of (direction × weight), capped to [-10, +10]."""
    raw = sum(t.direction * t.weight for t in triggers)
    return max(-10, min(10, raw))


def combine_daily_weekly(daily_score: int, weekly_score: int,
                         cfg: ScoringConfig) -> float:
    """final = 0.7 × daily + 0.3 × weekly,共振时 ±bonus,最终截到 [-10, +10]."""
    base = cfg.daily_weight * daily_score + cfg.weekly_weight * weekly_score
    if daily_score >= cfg.resonance_daily_threshold and weekly_score >= cfg.resonance_weekly_threshold:
        base += cfg.resonance_bonus
    elif daily_score <= -cfg.resonance_daily_threshold and weekly_score <= -cfg.resonance_weekly_threshold:
        base -= cfg.resonance_bonus
    return max(-10, min(10, base))


def verdict_of(final_score: float, cfg: VerdictsConfig) -> str:
    """Return one of: strong_buy, buy, neutral, sell, strong_sell."""
    if final_score >= cfg.strong_buy:
        return "strong_buy"
    if final_score >= cfg.buy:
        return "buy"
    if final_score <= cfg.strong_sell:
        return "strong_sell"
    if final_score <= cfg.sell:
        return "sell"
    return "neutral"
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_signals.py -v
```

Expected: 10 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/signals.py tests/test_signals.py
git commit -m "feat(signals): composite scoring + daily/weekly resonance + verdict mapping"
```

---

## Task 13: 历史命中率回测 (`backtest.py`)

**Files:**
- Create: `src/stockpool/backtest.py`
- Create: `tests/test_backtest.py`

**职责:** 给一只股的**全历史带指标的 DataFrame**,回看每一根 K 线触发了什么信号,统计 N 日后的平均涨跌幅 + 胜率。返回:`dict[signal_type, dict[forward_days, stats]]`。

- [ ] **Step 1: 写失败的测试**

`tests/test_backtest.py`:
```python
import numpy as np
import pandas as pd
import pytest

from stockpool.backtest import compute_hit_rates
from stockpool.config import WeightsConfig


@pytest.fixture
def weights() -> WeightsConfig:
    return WeightsConfig(
        ma_cross_strong=2, ma_alignment=1, macd_cross_above_zero=2, macd_cross_below_zero=1,
        macd_histogram_expand=1, kdj_oversold_cross=2, kdj_overbought_cross=2, kdj_normal_cross=1,
        rsi_oversold=1, rsi_overbought=1, boll_band_touch=2, boll_mid_cross=1,
        volume_surge_bullish=1, volume_surge_bearish=1, breakout_new_high=2, breakout_new_low=2,
    )


def _make_history_with_planted_breakouts() -> pd.DataFrame:
    """构造 30 天数据,第 20 天人为是 20 日新高(此前最大 100),收盘 110;
       之后 5 天每天涨 1%."""
    n = 30
    close = np.full(n, 100.0)
    close[:20] = 100.0
    close[20] = 110.0   # 新高
    for i in range(21, n):
        close[i] = close[i-1] * 1.01

    df = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=n, freq="B"),
        "open": close - 0.1, "high": close + 0.1, "low": close - 0.2,
        "close": close, "volume": [1e6] * n,
        "ma5": close, "ma10": close, "ma20": close, "ma60": close,
        "macd_dif": np.zeros(n), "macd_dea": np.zeros(n), "macd_hist": np.zeros(n),
        "kdj_k": np.full(n, 50.0), "kdj_d": np.full(n, 50.0), "kdj_j": np.full(n, 50.0),
        "rsi6": np.full(n, 50.0), "rsi12": np.full(n, 50.0), "rsi24": np.full(n, 50.0),
        "boll_up": close + 1, "boll_mid": close, "boll_low": close - 1,
        "vol_ratio5": np.ones(n),
        "is_breakout_high": [False]*20 + [True] + [False]*9,
        "is_breakout_low": [False]*n,
    })
    return df


def test_hit_rate_breakout_high_5d(weights):
    df = _make_history_with_planted_breakouts()
    stats = compute_hit_rates(df, weights, forward_days=[5, 10, 20])

    assert "breakout_new_high" in stats
    s = stats["breakout_new_high"]
    assert s["count"] == 1
    # 5 日后涨幅:第 20 天 close 110,第 25 天 close ≈ 110 * 1.01^5
    expected_5d = (110 * 1.01**5 / 110 - 1) * 100
    assert s["forward_5"]["mean_return_pct"] == pytest.approx(expected_5d, rel=1e-3)
    assert s["forward_5"]["win_rate"] == 1.0
    assert s["direction"] == +1


def test_no_signals_returns_empty(weights):
    """全平的数据 → 无任何触发."""
    n = 30
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-02", periods=n, freq="B"),
        "open": close, "high": close, "low": close,
        "close": close, "volume": [1e6] * n,
        "ma5": close, "ma10": close, "ma20": close, "ma60": close,
        "macd_dif": np.zeros(n), "macd_dea": np.zeros(n), "macd_hist": np.zeros(n),
        "kdj_k": np.full(n, 50.0), "kdj_d": np.full(n, 50.0), "kdj_j": np.full(n, 50.0),
        "rsi6": np.full(n, 50.0), "rsi12": np.full(n, 50.0), "rsi24": np.full(n, 50.0),
        "boll_up": close + 1, "boll_mid": close, "boll_low": close - 1,
        "vol_ratio5": np.ones(n),
        "is_breakout_high": [False]*n, "is_breakout_low": [False]*n,
    })
    stats = compute_hit_rates(df, weights, forward_days=[5])
    assert stats == {}
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_backtest.py -v
```

Expected: ImportError。

- [ ] **Step 3: 实现 `src/stockpool/backtest.py`**

```python
"""Historical signal hit-rate stats.

For each bar in history, run detect_signals; for each trigger, look forward
N days and record (close_{i+N} / close_i - 1). Aggregate per (signal_type, N).
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from stockpool.config import WeightsConfig
from stockpool.signals import detect_signals


def compute_hit_rates(
    df: pd.DataFrame,
    weights: WeightsConfig,
    forward_days: list[int],
) -> dict[str, dict]:
    """
    Returns:
      {
        "macd_cross_above_zero": {
          "count": 9,
          "direction": +1,
          "forward_5":  {"mean_return_pct": 2.1, "win_rate": 0.67},
          "forward_10": {"mean_return_pct": 3.4, "win_rate": 0.56},
          ...
        },
        ...
      }
    """
    if len(df) < 2:
        return {}

    # 每个信号一桶,按 forward N 再分桶
    buckets: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "direction": 0,
        "returns": defaultdict(list),   # forward_N -> [returns_pct]
    })

    closes = df["close"].values

    for i in range(1, len(df)):
        # 至少留 max(forward_days) 根供 forward 观察
        window = df.iloc[max(0, i-1): i+1]   # 给 detect_signals 看 prev + curr
        triggers = detect_signals(window, weights)

        for t in triggers:
            b = buckets[t.signal_type]
            b["count"] += 1
            b["direction"] = t.direction
            for n in forward_days:
                j = i + n
                if j < len(df):
                    ret_pct = (closes[j] / closes[i] - 1) * 100
                    b["returns"][n].append(ret_pct)

    # 聚合
    result: dict[str, dict] = {}
    for sig, b in buckets.items():
        entry = {
            "count": b["count"],
            "direction": b["direction"],
        }
        for n in forward_days:
            rs = b["returns"][n]
            if rs:
                mean_ret = sum(rs) / len(rs)
                # 胜率定义:看多信号 → 涨为胜;看空信号 → 跌为胜
                if b["direction"] == +1:
                    wins = sum(1 for r in rs if r > 0)
                else:
                    wins = sum(1 for r in rs if r < 0)
                win_rate = wins / len(rs)
            else:
                mean_ret = 0.0
                win_rate = 0.0
            entry[f"forward_{n}"] = {
                "mean_return_pct": mean_ret,
                "win_rate": win_rate,
                "sample_size": len(rs),
            }
        result[sig] = entry
    return result
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_backtest.py -v
```

Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): historical signal hit-rate statistics"
```

---

## Task 14: 单股 K 线图渲染 (`report.py` 第一部分)

**Files:**
- Create: `src/stockpool/report.py`
- Create: `tests/test_report_smoke.py`

**职责:**
- `build_stock_chart(code, name, daily_with_indicators, klines_to_show) -> pyecharts.charts.Grid`
  - 主图:K 线 + MA5/10/20/60 + BOLL 上中下轨
  - 副图 1:成交量柱
  - 副图 2:MACD 柱+线
  - 副图 3:KDJ
  - 副图 4:RSI
  - 5 图共享 DataZoom

- [ ] **Step 1: 写 smoke 测试**

`tests/test_report_smoke.py`:
```python
"""Smoke tests — verify HTML generates and contains expected markers."""
import numpy as np
import pandas as pd
import pytest

from stockpool.config import IndicatorsConfig, MACDConfig, KDJConfig, BOLLConfig
from stockpool.indicators import add_all
from stockpool.report import build_stock_chart


@pytest.fixture
def indicators_cfg() -> IndicatorsConfig:
    return IndicatorsConfig(
        ma_periods=[5, 10, 20, 60],
        macd=MACDConfig(fast=12, slow=26, signal=9),
        kdj=KDJConfig(n=9, m1=3, m2=3),
        rsi_periods=[6, 12, 24],
        boll=BOLLConfig(n=20, k=2),
        volume_ratio_window=5,
        breakout_window=20,
    )


def _make_long_history(n=120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 10 + np.cumsum(rng.normal(0.02, 0.3, n))
    return pd.DataFrame({
        "date": pd.date_range("2025-08-01", periods=n, freq="B"),
        "open": close + rng.normal(0, 0.1, n),
        "high": close + np.abs(rng.normal(0.2, 0.1, n)),
        "low":  close - np.abs(rng.normal(0.2, 0.1, n)),
        "close": close,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    })


def test_build_stock_chart_returns_html(indicators_cfg):
    raw = _make_long_history(120)
    enriched = add_all(raw, indicators_cfg)

    grid = build_stock_chart("605589", "圣泉集团", enriched, klines_to_show=120)
    html = grid.render_embed()   # pyecharts 渲染成 HTML 片段

    assert "605589" in html
    assert "圣泉集团" in html
    # 应包含 echarts 关键字段
    assert "echarts" in html.lower() or "option" in html.lower()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_report_smoke.py -v
```

Expected: ImportError。

- [ ] **Step 3: 实现 `src/stockpool/report.py`(初版,只含 build_stock_chart)**

```python
"""HTML 报告生成 — pyecharts driver."""
from __future__ import annotations

from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Kline, Line


def _kline_main(code: str, name: str, df) -> Kline:
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
            title_opts=opts.TitleOpts(title=f"{code} {name}"),
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(is_scale=True, splitarea_opts=opts.SplitAreaOpts(is_show=True)),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1, 2, 3, 4]),
                opts.DataZoomOpts(type_="slider", xaxis_index=[0, 1, 2, 3, 4]),
            ],
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_top="2%"),
        )
    )
    return kline


def _ma_boll_overlay(df) -> Line:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    line = Line().add_xaxis(dates)
    for col, label in [("ma5", "MA5"), ("ma10", "MA10"), ("ma20", "MA20"), ("ma60", "MA60"),
                       ("boll_up", "BOLL上"), ("boll_mid", "BOLL中"), ("boll_low", "BOLL下")]:
        if col in df.columns:
            line.add_yaxis(label, df[col].round(3).tolist(),
                           is_smooth=True, is_symbol_show=False,
                           label_opts=opts.LabelOpts(is_show=False),
                           linestyle_opts=opts.LineStyleOpts(width=1))
    line.set_global_opts(legend_opts=opts.LegendOpts(pos_top="2%"))
    return line


def _volume_bar(df) -> Bar:
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
            xaxis_opts=opts.AxisOpts(grid_index=1, axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(grid_index=1, is_scale=True),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    # 单独给柱子上色(pyecharts 限制:用 mark)
    bar.options["series"][0]["data"] = [
        {"value": v, "itemStyle": {"color": c}}
        for v, c in zip(df["volume"].tolist(), colors)
    ]
    return bar


def _macd_chart(df) -> Bar:
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
        xaxis_opts=opts.AxisOpts(grid_index=2, axislabel_opts=opts.LabelOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(grid_index=2, is_scale=True),
        legend_opts=opts.LegendOpts(pos_top="34%"),
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


def _kdj_chart(df) -> Line:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    line = (
        Line()
        .add_xaxis(dates)
        .add_yaxis("K", df["kdj_k"].round(2).fillna(50).tolist(),
                   is_smooth=True, is_symbol_show=False, label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("D", df["kdj_d"].round(2).fillna(50).tolist(),
                   is_smooth=True, is_symbol_show=False, label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("J", df["kdj_j"].round(2).fillna(50).tolist(),
                   is_smooth=True, is_symbol_show=False, label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(grid_index=3, axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(grid_index=3, is_scale=True),
            legend_opts=opts.LegendOpts(pos_top="56%"),
        )
    )
    return line


def _rsi_chart(df) -> Line:
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    line = Line().add_xaxis(dates)
    for col, label in [("rsi6", "RSI6"), ("rsi12", "RSI12"), ("rsi24", "RSI24")]:
        if col in df.columns:
            line.add_yaxis(label, df[col].round(2).fillna(50).tolist(),
                           is_smooth=True, is_symbol_show=False,
                           label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(grid_index=4, is_scale=True),
        yaxis_opts=opts.AxisOpts(grid_index=4, is_scale=True),
        legend_opts=opts.LegendOpts(pos_top="78%"),
    )
    return line


def build_stock_chart(code: str, name: str, df, klines_to_show: int):
    """5 行联动 grid:K线+量+MACD+KDJ+RSI."""
    show = df.tail(klines_to_show).reset_index(drop=True)

    kline = _kline_main(code, name, show).overlap(_ma_boll_overlay(show))
    volume = _volume_bar(show)
    macd = _macd_chart(show)
    kdj = _kdj_chart(show)
    rsi = _rsi_chart(show)

    grid = (
        Grid(init_opts=opts.InitOpts(width="100%", height="900px"))
        .add(kline,   grid_opts=opts.GridOpts(pos_top="6%",  height="28%"))
        .add(volume,  grid_opts=opts.GridOpts(pos_top="38%", height="10%"))
        .add(macd,    grid_opts=opts.GridOpts(pos_top="50%", height="14%"))
        .add(kdj,     grid_opts=opts.GridOpts(pos_top="66%", height="14%"))
        .add(rsi,     grid_opts=opts.GridOpts(pos_top="82%", height="14%"))
    )
    return grid
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_report_smoke.py -v
```

Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/report.py tests/test_report_smoke.py
git commit -m "feat(report): single-stock K-line + indicators grid (pyecharts)"
```

---

## Task 15: 整页 HTML 报告组装

**Files:**
- Modify: `src/stockpool/report.py`
- Modify: `tests/test_report_smoke.py`

**职责:**
- `StockAnalysis` dataclass 把单股的所有产出打包(name, code, daily_score, weekly_score, final_score, verdict, triggers_daily, triggers_weekly, hit_rates, daily_with_indicators)
- `render_report(analyses, run_date, config_path, config_hash, output_dir, keep_history) -> Path`:
  - 写总览表 + 每只股 details 区块 + 附录
  - 写到 `output_dir/YYYY-MM-DD/index.html`
  - 同时复制为 `output_dir/latest.html`

- [ ] **Step 1: 加 smoke 测试**

追加到 `tests/test_report_smoke.py`:
```python
from pathlib import Path

from stockpool.report import StockAnalysis, render_report
from stockpool.signals import Trigger


def _make_analysis(code, name, score, verdict, indicators_cfg) -> StockAnalysis:
    raw = _make_long_history(120)
    enriched = add_all(raw, indicators_cfg)
    return StockAnalysis(
        code=code, name=name,
        daily_score=int(score * 0.7), weekly_score=int(score * 0.3 / 0.3) if score else 0,
        final_score=score, verdict=verdict,
        triggers_daily=[Trigger("macd_cross_above_zero", +1, 2, "MACD 零轴上方金叉")],
        triggers_weekly=[],
        hit_rates={
            "macd_cross_above_zero": {
                "count": 5, "direction": +1,
                "forward_5": {"mean_return_pct": 2.1, "win_rate": 0.6, "sample_size": 5},
                "forward_10": {"mean_return_pct": 3.4, "win_rate": 0.6, "sample_size": 5},
                "forward_20": {"mean_return_pct": 5.0, "win_rate": 0.6, "sample_size": 5},
            }
        },
        daily_with_indicators=enriched,
    )


def test_render_report_writes_html(tmp_path, indicators_cfg):
    analyses = [
        _make_analysis("605589", "圣泉集团", 6.1, "strong_buy", indicators_cfg),
        _make_analysis("603986", "兆易创新", -4.0, "sell", indicators_cfg),
        _make_analysis("000528", "柳工",     0.5, "neutral", indicators_cfg),
    ]
    out = render_report(
        analyses, run_date="2026-05-17",
        config_path=Path("config.yaml"), config_hash="abc12345",
        output_dir=tmp_path, keep_history=True, klines_to_show=120,
    )

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    for keyword in ["养龙股池", "2026-05-17", "圣泉集团", "兆易创新", "柳工",
                    "strong_buy", "sell", "abc12345", "免责声明"]:
        assert keyword in text, f"missing {keyword!r} in report"

    # latest.html 也应被写
    assert (tmp_path / "latest.html").exists()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_report_smoke.py::test_render_report_writes_html -v
```

Expected: ImportError(`StockAnalysis` / `render_report` 不存在)。

- [ ] **Step 3: 加实现**

在 `src/stockpool/report.py` 末尾追加:
```python
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pyecharts.commons.utils import JsCode

from stockpool.signals import Trigger


_VERDICT_LABEL = {
    "strong_buy":   ("🟢🟢", "强烈买入", "#0a7d24"),
    "buy":          ("🟢",   "买入观察", "#37b54a"),
    "neutral":      ("⚪",   "观望",     "#999999"),
    "sell":         ("🔴",   "卖出观察", "#d96b6b"),
    "strong_sell":  ("🔴🔴", "强烈卖出", "#a01818"),
}


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
    daily_with_indicators: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)


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

    return f"""
    <details id="stock-{a.code}" open>
      <summary>
        <span style="font-size:1.3em; font-weight:bold">{a.code} {a.name}</span>
        <span style="color:{color}; margin-left:1em">{emoji} {label} 终分 {a.final_score:+.1f}</span>
      </summary>
      {warnings_html}
      <div class="chart-wrap">{chart_html}</div>

      <div class="signal-cols">
        <div>
          <h4>触发信号(日 K)— 日分 {a.daily_score:+d} × 0.7 = {a.daily_score * 0.7:+.2f}</h4>
          <ul>{_trigger_list_html(a.triggers_daily)}</ul>
        </div>
        <div>
          <h4>触发信号(周 K)— 周分 {a.weekly_score:+d} × 0.3 = {a.weekly_score * 0.3:+.2f}</h4>
          <ul>{_trigger_list_html(a.triggers_weekly)}</ul>
        </div>
      </div>

      <h4>历史命中率(过去 500 日)</h4>
      {_hit_rate_table(a.hit_rates)}
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
  footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #eee;
           color: #888; font-size: 0.85em; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
"""


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
) -> Path:
    """渲染整页 HTML 报告,返回文件路径."""
    output_dir = Path(output_dir)
    day_dir = output_dir / run_date
    day_dir.mkdir(parents=True, exist_ok=True)
    out_path = day_dir / "index.html"

    # 按 final_score 降序
    analyses_sorted = sorted(analyses, key=lambda a: -a.final_score)

    overview_rows = "".join(_overview_row(a) for a in analyses_sorted)
    stock_sections = "".join(_stock_section_html(a, klines_to_show) for a in analyses_sorted)

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

  <footer>
    <p>Config: <code>{config_path}</code> &nbsp; hash <code>{config_hash}</code></p>
    <p>⚠️ <strong>免责声明:</strong>本报告基于公开行情数据的技术指标计算,
       信号与打分仅供个人技术分析参考,<strong>不构成任何投资建议</strong>。
       使用者应自行承担交易决策的全部责任。</p>
  </footer>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")

    # latest.html 始终指向最新报告
    latest = output_dir / "latest.html"
    shutil.copyfile(out_path, latest)

    return out_path
```

- [ ] **Step 4: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_report_smoke.py -v
```

Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/report.py tests/test_report_smoke.py
git commit -m "feat(report): full single-page HTML with overview + per-stock sections"
```

---

## Task 16: CLI 入口 + 交易日判断 + 串接

**Files:**
- Create: `src/stockpool/cli.py`

**职责:**
- `python -m stockpool run [--refresh] [--stocks CODE1,CODE2] [--config PATH] [--skip-trading-day-check]`
- 串起:load config → 交易日检查 → fetch_daily → resample_to_weekly → add_all (日K+周K) → detect_signals → score → backtest → render_report
- 单股失败不中断其他股
- 日志写到 `reports/YYYY-MM-DD/run.log`

- [ ] **Step 1: 实现 `src/stockpool/cli.py`**

```python
"""CLI entry: `python -m stockpool run`."""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd

from stockpool import __version__
from stockpool.backtest import compute_hit_rates
from stockpool.config import AppConfig, load_config
from stockpool.fetcher import fetch_daily, resample_to_weekly
from stockpool.indicators import add_all
from stockpool.report import StockAnalysis, render_report
from stockpool.signals import (
    combine_daily_weekly,
    detect_signals,
    score_triggers,
    verdict_of,
)

log = logging.getLogger("stockpool")


def _is_trading_day(today: date) -> bool:
    """用 AKShare 交易日历判断今天是否 A 股交易日."""
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(cal["trade_date"]).dt.date
        return today in set(dates)
    except Exception as e:
        log.warning("Trading day check failed (%s) — assuming trading day", e)
        return True


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [file_h, stream_h]


def _analyze_one(stock, cfg: AppConfig, force_refresh: bool) -> StockAnalysis:
    """完整跑一只股 — 返回 StockAnalysis(失败时 verdict='neutral'+warnings)."""
    warnings: list[str] = []
    daily_score = 0
    weekly_score = 0
    triggers_daily: list = []
    triggers_weekly: list = []
    hit_rates: dict = {}
    enriched_daily = None

    try:
        daily = fetch_daily(stock.code, cfg.data.history_days,
                            cfg.data.cache_dir, force_refresh=force_refresh)
    except Exception as e:
        warnings.append(f"数据拉取失败: {e}")
        return StockAnalysis(
            code=stock.code, name=stock.name,
            daily_score=0, weekly_score=0,
            final_score=0.0, verdict="neutral",
            warnings=warnings,
        )

    if len(daily) < 30:
        warnings.append(f"历史数据不足 ({len(daily)} 根),指标可能不可靠")

    enriched_daily = add_all(daily, cfg.indicators)
    triggers_daily = detect_signals(enriched_daily, cfg.weights)
    daily_score = score_triggers(triggers_daily)

    weekly = resample_to_weekly(daily)
    if len(weekly) >= 30:
        enriched_weekly = add_all(weekly, cfg.indicators)
        triggers_weekly = detect_signals(enriched_weekly, cfg.weights)
        weekly_score = score_triggers(triggers_weekly)
    else:
        warnings.append("周 K 样本不足,本股不计算周 K 信号")

    final_score = combine_daily_weekly(daily_score, weekly_score, cfg.scoring)
    verdict = verdict_of(final_score, cfg.verdicts)

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


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    today = date.today()
    run_date = today.isoformat()
    log_dir = Path(cfg.report.output_dir) / run_date
    _setup_logging(log_dir)

    log.info("stockpool v%s starting run for %s", __version__, run_date)

    if not args.skip_trading_day_check and not _is_trading_day(today):
        log.info("Today (%s) is not an A-share trading day. Exit 0.", run_date)
        return 0

    # 过滤股票
    stocks = cfg.stocks
    if args.stocks:
        wanted = set(args.stocks.split(","))
        stocks = [s for s in stocks if s.code in wanted]
        if not stocks:
            log.error("No stocks match --stocks filter: %s", args.stocks)
            return 2

    analyses: list[StockAnalysis] = []
    for s in stocks:
        log.info("Analyzing %s (%s)...", s.code, s.name)
        try:
            analyses.append(_analyze_one(s, cfg, force_refresh=args.refresh))
        except Exception as e:
            log.error("Unexpected failure on %s: %s\n%s", s.code, e, traceback.format_exc())
            analyses.append(StockAnalysis(
                code=s.code, name=s.name,
                daily_score=0, weekly_score=0, final_score=0.0,
                verdict="neutral",
                warnings=[f"未预期错误: {e}"],
            ))

    out = render_report(
        analyses, run_date=run_date,
        config_path=Path(args.config), config_hash=cfg.content_hash,
        output_dir=cfg.report.output_dir,
        keep_history=cfg.report.keep_history,
        klines_to_show=cfg.report.klines_to_show,
    )
    log.info("Report written: %s", out)
    log.info("Latest also at: %s", Path(cfg.report.output_dir) / "latest.html")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stockpool", description="A-share signal analyzer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="完整跑一次:抓数据→分析→生报告")
    p_run.add_argument("--config", default="config.yaml", help="路径,默认 config.yaml")
    p_run.add_argument("--refresh", action="store_true", help="忽略缓存全量重拉")
    p_run.add_argument("--stocks", default="", help="只跑指定代码(逗号分隔)")
    p_run.add_argument("--skip-trading-day-check", action="store_true",
                       help="非交易日也强制跑(调试用)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 加一个端到端的轻测试**

追加到 `tests/test_report_smoke.py`:
```python
from unittest.mock import patch

from stockpool.cli import main


def test_cli_run_smoke(tmp_path, monkeypatch):
    """端到端:mock fetcher + 交易日检查,确认 CLI 跑完出报告."""
    # 临时 config
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("""
stocks:
  - {code: "605589", name: "圣泉集团"}
data: {history_days: 120, cache_dir: "data", force_refresh: false}
indicators:
  ma_periods: [5, 10, 20, 60]
  macd: {fast: 12, slow: 26, signal: 9}
  kdj: {n: 9, m1: 3, m2: 3}
  rsi_periods: [6, 12, 24]
  boll: {n: 20, k: 2}
  volume_ratio_window: 5
  breakout_window: 20
weights:
  ma_cross_strong: 2
  ma_alignment: 1
  macd_cross_above_zero: 2
  macd_cross_below_zero: 1
  macd_histogram_expand: 1
  kdj_oversold_cross: 2
  kdj_overbought_cross: 2
  kdj_normal_cross: 1
  rsi_oversold: 1
  rsi_overbought: 1
  boll_band_touch: 2
  boll_mid_cross: 1
  volume_surge_bullish: 1
  volume_surge_bearish: 1
  breakout_new_high: 2
  breakout_new_low: 2
scoring: {daily_weight: 0.7, weekly_weight: 0.3, resonance_bonus: 2,
          resonance_daily_threshold: 3, resonance_weekly_threshold: 1}
verdicts: {strong_buy: 6, buy: 3, sell: -3, strong_sell: -6}
backtest: {forward_days: [5, 10, 20]}
report: {output_dir: "%s", keep_history: true, klines_to_show: 120}
""" % str(tmp_path / "out").replace("\\", "/"), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    # mock 数据源
    import numpy as np
    rng = np.random.default_rng(7)
    close = 10 + np.cumsum(rng.normal(0.02, 0.3, 200))
    fake = pd.DataFrame({
        "日期": pd.date_range("2025-08-01", periods=200, freq="B").strftime("%Y-%m-%d"),
        "开盘": close - 0.1, "收盘": close, "最高": close + 0.2, "最低": close - 0.2,
        "成交量": rng.integers(500_000, 2_000_000, 200), "成交额": [0]*200,
        "振幅": [0]*200, "涨跌幅": [0]*200, "涨跌额": [0]*200, "换手率": [0]*200,
    })

    with patch("stockpool.fetcher.ak.stock_zh_a_hist", return_value=fake), \
         patch("stockpool.cli.ak.tool_trade_date_hist_sina",
               return_value=pd.DataFrame({"trade_date": [pd.Timestamp.today().date()]})):
        exit_code = main(["run", "--config", str(config_yaml)])

    assert exit_code == 0
    # 报告生成
    reports = list((tmp_path / "out").rglob("index.html"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "605589" in text
    assert "圣泉集团" in text
    assert (tmp_path / "out" / "latest.html").exists()
```

- [ ] **Step 3: 跑测试**

```bash
.venv/Scripts/python -m pytest tests/test_report_smoke.py -v
```

Expected: 3 passed。

- [ ] **Step 4: 跑全部测试**

```bash
.venv/Scripts/python -m pytest
```

Expected: 全 passed(约 30+ 测试),用时 < 30 秒(akshare 加载本身慢)。

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/cli.py tests/test_report_smoke.py
git commit -m "feat(cli): end-to-end CLI with trading-day check and per-stock isolation"
```

---

## Task 17: README + Windows 计划任务模板

**Files:**
- Create: `README.md`
- Create: `scripts/stockpool_task.xml`

- [ ] **Step 1: 创建 `README.md`**

```markdown
# stockpool — A 股养龙股池技术信号分析

每日扫描配置文件中的 A 股池,产出综合打分 + 交互式 HTML 报告。

详细设计见 `docs/superpowers/specs/2026-05-17-a-share-signal-tool-design.md`。

## 快速开始

```bash
# 1. 安装(需要 Python 3.10+)
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

# 2. 编辑股池(可选)
notepad config.yaml

# 3. 跑一次
.venv/Scripts/python -m stockpool run

# 4. 看报告
start reports/latest.html
```

## 常用命令

```bash
.venv/Scripts/python -m stockpool run                       # 默认全跑
.venv/Scripts/python -m stockpool run --refresh             # 忽略缓存重拉
.venv/Scripts/python -m stockpool run --stocks 605589,603986 # 只跑两只
.venv/Scripts/python -m stockpool run --skip-trading-day-check  # 周末调试
.venv/Scripts/python -m pytest                              # 全套单元测试
```

## 加股票

打开 `config.yaml`,在 `stocks:` 列表里追加一行即可:

```yaml
stocks:
  - {code: "600519", name: "贵州茅台"}
```

## 调整打分权重

`config.yaml` 的 `weights:` 段每个数字都可改。
信号定义见 `docs/superpowers/specs/2026-05-17-a-share-signal-tool-design.md` § 5。

## Windows 计划任务

复制 `scripts/stockpool_task.xml`,改里面的项目路径,然后:

```cmd
schtasks /Create /XML scripts\stockpool_task.xml /TN "Stockpool Daily"
```

任务设置为周一至周五 15:30 触发。脚本本身会查交易日历,节假日自动 exit 0。

## 输出位置

- `reports/YYYY-MM-DD/index.html` — 当日报告
- `reports/YYYY-MM-DD/run.log` — 当日运行日志
- `reports/latest.html` — 永远是最新一份(任务栏快捷方式固定它)
- `data/{code}_daily.parquet` — 行情缓存(可删除,下次自动重建)

## ⚠️ 免责声明

本工具产出基于公开行情数据的技术指标计算,信号与打分仅供个人技术分析参考,
**不构成任何投资建议**。
```

- [ ] **Step 2: 创建 `scripts/stockpool_task.xml`**

```xml
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Run stockpool daily signal analyzer (Mon-Fri 15:30)</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-05-17T15:30:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek>
          <Monday/><Tuesday/><Wednesday/><Thursday/><Friday/>
        </DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT30M</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <!-- 改成你的实际路径 -->
      <Command>C:\Users\Administrator\Desktop\claude\.venv\Scripts\python.exe</Command>
      <Arguments>-m stockpool run</Arguments>
      <WorkingDirectory>C:\Users\Administrator\Desktop\claude</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

- [ ] **Step 3: Commit**

```bash
git add README.md scripts/stockpool_task.xml
git commit -m "docs: README and Windows Task Scheduler template"
```

- [ ] **Step 4: 跑一次真实的 CLI(冒烟,非测试)**

```bash
.venv/Scripts/python -m stockpool run --skip-trading-day-check
```

Expected: 8 只股全部分析完成,`reports/YYYY-MM-DD/index.html` 生成,用浏览器打开看到完整报告。
如果 AKShare 限频导致部分股拉取失败,报告里会有⚠️标注,但其他股正常。

- [ ] **Step 5: 跑全测试最终验证**

```bash
.venv/Scripts/python -m pytest -v
```

Expected: 30+ 测试全 passed。

---

## 完成标志

- ✅ `pytest` 全绿
- ✅ `python -m stockpool run` 能生成 `reports/YYYY-MM-DD/index.html`
- ✅ HTML 在浏览器打开可见 K 线图 + 总览表 + 触发明细 + 命中率表
- ✅ 加股票只需编辑 `config.yaml` 中 `stocks:` 列表
- ✅ 调权重只需编辑 `config.yaml` 中 `weights:` 段
