# Portfolio Framework Design

> **Spec date**: 2026-05-24
> **Status**: Draft, awaiting user review
> **Related**: `docs/strategy_improvement_2026.md` §6 (路线图 P1 替换),`docs/superpowers/specs/2026-05-24-ab-testing-design.md` (per-stock AB,本 spec 的对照参考)

---

## 1. 背景与动机

### 1.1 现状

`stockpool` 当前的回测和 A/B 框架都是 **per-stock** 级:

- `Strategy.generate_signals(daily_df)` 单股接口
- `BacktestEngine` / `MultiLotBacktestEngine` 单股 simulation
- `backtest_runner.backtest_stocks` 是外层循环,对 cfg.stocks 每只票独立 simulate
- `stockpool ab` 跨股聚合 mean/median/胜出计数

per-stock 框架的根本局限:**比较的是"信号在单股上的命中表现",不是"信号驱动的真实组合表现"**。

### 1.2 问题

1. **不能反映真实组合行为**:多股共享资金、行业分散、调仓 turnover 等组合层概念,per-stock 框架完全捕捉不到
2. **AB 样本量小且不独立**:8-30 只股聚合统计噪声大;不同股的成功/失败可能来自市值/行业/流动性 bias
3. **风控视角缺失**:max DD / 集中度限制在单股语境下要么退化(单股 cap = 单仓位 cap)要么无意义(单股没有"组合 DD")
4. **F3 路线图 PR-D**(单股 risk overlay)在设计时即承认"sector cap 在单股引擎里退化成单股最大仓位",做了也只是"看起来有风控"

### 1.3 目标

用户在 2026-05-24 决策:撤销 F3 PR-D,跳过单股风控,直接做 **portfolio 级回测和 A/B 框架**。理由:per-stock 回测的目的就是"比较哪个信号更好",加 sizing/DD 风控偏离这个目的;真正的组合层视角需要新抽象。

---

## 2. 目标与非目标

### 2.1 目标

- **G1**:新增 `PortfolioStrategy` ABC + `PortfolioEngine`,在每个 bar 看横截面,输出组合净值
- **G2**:**复用现有 strategies**(`MLFactorStrategy` / `CompositeVerdictStrategy`)不改一行,通过 score panel 预算适配到 portfolio 框架
- **G3**:支持 staggered ensemble(N 个 start offset 同跑,做 sensitivity + 真实 ensemble portfolio)
- **G4**:新增 `stockpool portfolio-backtest` + `stockpool portfolio-ab` CLI,跟现有 `backtest` / `ab` 平行
- **G5**:HTML 报告:portfolio backtest 单 arm 出净值 + 包络 + 持仓时间线;portfolio AB 出双 arm 对比 + per-stock 贡献分解
- **G6**:零回归 — 现有 per-stock 框架(`Strategy` / `BacktestEngine` / `MultiLotBacktestEngine` / `cmd_backtest` / `cmd_ab`)完全不动

### 2.2 非目标(本 spec 不解决,见 §12 详表)

- 生存偏差修正、历史 ST 状态、历史行业映射
- 持仓部分调整 / turnover 上限
- 滑点模型(冲击成本)、涨停打不进单过滤
- 空头、期货、期权
- AB > 2 arm
- AB 统计显著性 p 值(用包络替代)
- Score panel 跨 arm 共享缓存
- Staggered 并行化(第一版串行)
- panel-batch ML fit/predict 优化
- Score smoothing (EMA)
- 单股 risk overlay(F3 PR-D 已撤销)

---

## 3. 设计决策摘要

Brainstorm 过程中由用户确认的 6 个核心决策:

| # | 决策 | 选项 | 理由 |
|---|------|------|------|
| 1 | Strategy 输出语义 | **Score per stock**(引擎做 top-K 等权) | 跟"比较信号"目标契合;weighting scheme 留 followup |
| 2 | Universe 范围 | **动态 Pool B 风格**(逐 bar 流动性 + ST + 行业 cap 筛选) | 最贴近实战;复用 `recommend_pool` 漏斗的语义 |
| 3 | Rebalance 频率 | **Fixed period(每 N 个 bar)**,默认 N=5(weekly) | 跟 strategy 类型解耦;turnover 可控;ISO 周对齐受节假日扰动 |
| 4 | 与 per-stock 框架关系 | **共存,新老平行**(adapter 复用) | 零回归;两套框架职责清晰 |
| 5 | AB 输出 | **Per-stock 贡献分解 + staggered 包络/均值** | portfolio 独有视角;回答"为什么 A 赢 B" |
| 6 | 交付方式 | **一个 spec,4 个 PR 渐进交付** | 设计一致性 + 早期反馈 + 失败可回滚 |

额外修正:**Adapter 用 `predict_latest` 有 look-ahead bug**,改用 `generate_signals` + score panel 预算(walk-forward 在 strategy 内部完成)。

---

## 4. 架构概览

### 4.1 顶层组件

```
┌─────────────────────────────────────────────────────────────────┐
│  src/stockpool/portfolio/                                       │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  PortfolioStrategy (ABC)                               │    │
│  │  predict_scores(date_t, panel_data) -> dict[code, f]  │    │
│  └─────────────────────▲──────────────────────────────────┘    │
│                        │ implements                             │
│  ┌─────────────────────┴──────────────────────────────────┐    │
│  │  PrecomputedScoreStrategy                              │    │
│  │   wraps a pre-built T×N score panel                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  precompute_scores_from_legacy(legacy, panel_data)    │    │
│  │   helper: 对每只票调 legacy.generate_signals(),         │    │
│  │   拼出 (T × N) 分数面板                                 │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  EligibilityFilter                                     │    │
│  │  eligible(date_t, panel_data) -> set[code]            │    │
│  │   - 流动性: rolling avg_amount_20d ≥ threshold        │    │
│  │   - ST: name 不含 "ST"                                 │    │
│  │   - min_history_bars                                   │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  PortfolioEngine                                       │    │
│  │   维护 Portfolio(cash + positions)                     │    │
│  │   每个 rebalance bar:                                  │    │
│  │     1. scores = strategy.predict_scores(...)          │    │
│  │     2. eligible = filter.eligible(...)                │    │
│  │     3. top-K + 行业 cap 贪心                            │    │
│  │     4. diff vs 当前持仓 → 卖/买 → 等权 rebalance        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  StaggeredRunner(engine_factory)                       │    │
│  │   .run(panel_data, n_offsets) -> EnsembleResult       │    │
│  │   N 个 offset 各跑一遍;输出包络 + ensemble 均值        │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  src/stockpool/portfolio_ab/                                    │
│                                                                 │
│  config.py:    PortfolioABConfig + PortfolioArmOverride         │
│  runner.py:    run_portfolio_ab + ArmResult + ABResult          │
│  report.py:    HTML 双 arm 对比报告                              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 与现有代码的关系

| 现有 | 新增 | 关系 |
|---|---|---|
| `Strategy` ABC (per-stock) | `PortfolioStrategy` ABC | **平行,无继承** |
| `BacktestEngine` / `MultiLotBacktestEngine` | `PortfolioEngine` | 平行 |
| `BacktestResult` | `PortfolioBacktestResult` | 平行,字段不同 |
| `CompositeVerdictStrategy` / `MLFactorStrategy` | `PrecomputedScoreStrategy` + `precompute_scores_from_legacy()` | **包装复用,不改原类** |
| `stockpool backtest` | `stockpool portfolio-backtest` | 新 CLI,独立 |
| `stockpool ab` | `stockpool portfolio-ab` | 新 CLI,独立 |
| `ab/config.py`, `ab/runner.py`, `ab/report.py` | `portfolio_ab/config.py`, `runner.py`, `report.py` | 平行子包 |

### 4.3 关键设计取舍

- **Strategy 无继承关系**:`PortfolioStrategy` 不继承自 `Strategy`,两者 I/O 语义完全不同(一个单股、一个横截面),强行继承会出脏接口
- **Adapter 复用而非 fork**:`MLFactorStrategy` / `CompositeVerdictStrategy` 不动一行;`precompute_scores_from_legacy()` 对每只票调一次 `generate_signals()`(walk-forward 在 strategy 内部完成,无 look-ahead),拼成 T×N score panel
- **Eligibility 与 ranking 分离**:filter 只决定"能不能买",ranking 只决定"该买谁",行业 cap 是 portfolio 状态依赖逻辑放 engine
- **现有 per-stock CLI 不动**:`stockpool backtest` 和 `stockpool ab` 完全保留,零回归

---

## 5. 核心抽象签名

### 5.1 `PortfolioStrategy` ABC

```python
# src/stockpool/portfolio/strategy.py
from abc import ABC, abstractmethod
import pandas as pd

class PortfolioStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def predict_scores(
        self,
        date_t: pd.Timestamp,
        panel_data: dict[str, pd.DataFrame],   # {code: daily_df bars ≤ date_t}
    ) -> dict[str, float]:
        """返回每只股的连续分数。
        - 高分 = 更想持有
        - 不在 panel_data 的 code 跳过
        - 返回 NaN/None 的会被引擎当作不可投资
        Look-ahead 契约: 只能用 daily_df.loc[date ≤ date_t]
        """
```

### 5.2 `PrecomputedScoreStrategy`

```python
# src/stockpool/portfolio/strategy.py (同文件)
class PrecomputedScoreStrategy(PortfolioStrategy):
    """包装一个预算好的 (T × N) score panel,
    把 PortfolioEngine 跟 legacy strategy 训练逻辑解耦。"""

    def __init__(self, score_panel: pd.DataFrame, name: str = "precomputed"):
        # score_panel: index=date, columns=code, values=score (float, NaN 允许)
        self._panel = score_panel
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def predict_scores(self, date_t, panel_data):
        if date_t not in self._panel.index:
            return {}
        row = self._panel.loc[date_t].dropna()
        return {code: float(v) for code, v in row.items() if code in panel_data}
```

### 5.3 `precompute_scores_from_legacy()` helper

```python
# src/stockpool/portfolio/scoring.py
def precompute_scores_from_legacy(
    legacy_strategy,                  # MLFactorStrategy or CompositeVerdictStrategy
    panel_data: dict[str, pd.DataFrame],
    score_field: str = "final_score",
) -> pd.DataFrame:
    """对池里每只票调 legacy.generate_signals(),拼成 T×N 分数面板。
    Walk-forward 训练在 generate_signals 内部完成,无 look-ahead。

    Failures isolated: 任一只票生成异常,跳过该 code,不阻断 panel 构建。

    Returns:
        T×N DataFrame, index=date, columns=code, values=score
    """
    series_by_code = {}
    for code, daily in panel_data.items():
        try:
            sig = legacy_strategy.generate_signals(daily)
            if score_field in sig.columns:
                series_by_code[code] = sig.set_index("date")[score_field]
        except Exception:
            continue   # log warning + skip
    return pd.DataFrame(series_by_code)
```

### 5.4 `EligibilityFilter`

```python
# src/stockpool/portfolio/eligibility.py
from dataclasses import dataclass

@dataclass
class EligibilityConfig:
    min_avg_amount_20d: float = 5e7    # 5000 万元
    exclude_st: bool = True            # 名称含 "ST" 排除
    min_history_bars: int = 60         # 至少 60 bar 历史

class EligibilityFilter:
    def __init__(self, cfg: EligibilityConfig, name_map: dict[str, str]):
        self.cfg = cfg
        self.name_map = name_map       # {code: 当前 name}

    def eligible(
        self, date_t: pd.Timestamp, panel_data: dict[str, pd.DataFrame],
    ) -> set[str]:
        out = set()
        for code, daily in panel_data.items():
            df = daily[daily["date"] <= date_t]
            if len(df) < self.cfg.min_history_bars:
                continue
            if self.cfg.exclude_st and "ST" in self.name_map.get(code, ""):
                continue
            recent = df.tail(20)
            avg_amount = (recent["close"] * recent["volume"] * 100).mean()
            if avg_amount < self.cfg.min_avg_amount_20d:
                continue
            out.add(code)
        return out
```

行业 cap **不放这里**,因为它依赖"已经持有的股的行业分布",是 engine 层状态。

### 5.5 `PortfolioEngine`

```python
# src/stockpool/portfolio/engine.py
from typing import Literal

@dataclass
class PortfolioConfig:
    top_k: int = 20
    rebalance_n_days: int = 5
    max_per_industry: int | None = 5
    initial_cash: float = 1.0

class PortfolioEngine:
    def __init__(
        self,
        strategy: PortfolioStrategy,
        eligibility: EligibilityFilter,
        sector_map: dict[str, str],
        portfolio_cfg: PortfolioConfig,
        costs: TradeCosts = TradeCosts(),
        risk_free_rate: float = 0.02,
    ): ...

    def run(
        self,
        panel_data: dict[str, pd.DataFrame],
        start_offset: int = 0,            # staggered ensemble 用
    ) -> "PortfolioBacktestResult": ...
```

### 5.6 `PortfolioBacktestResult` / `PortfolioTrade`

```python
@dataclass(frozen=True)
class PortfolioTrade:
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    weight_at_entry: float
    ret: float
    days_held: int
    exit_reason: Literal["rebalance_drop", "no_longer_eligible", "end_of_backtest"]

@dataclass
class PortfolioBacktestResult:
    curve: pd.DataFrame              # date / equity / num_positions / cash_ratio
    trades: list[PortfolioTrade]
    rebalance_log: pd.DataFrame      # 每个 rebalance bar 的 top-K 持仓快照
    metrics: dict                    # sharpe / max_dd / total_return / turnover / win_rate
    strategy_name: str
    config_hash: str
```

---

## 6. 端到端数据流

### 6.1 `stockpool portfolio-backtest` 流程

```
1. CLI: cli.cmd_portfolio_backtest
   - load_config("config.yaml") → AppConfig
   - 检查 cfg.portfolio_backtest.enabled

2. 准备 universe 数据
   - panel_data = load_universe_cache(cache_dir)
   - sector_map = load_or_build_industry_map(cache_dir, source="auto")
   - name_map = {code: name from universe.parquet}
   - set_sector_map(sector_map)   # 给 WQ101 industry_neutral 因子用

3. 构造 legacy strategy(复用 strategy_factory)
   legacy = build_strategy(
       cfg,
       pool_data=panel_data,
       factor_panel=build_factor_panel(cfg.strategy.ml_factor.factors, panel_data),
       shared_cache={},
   )

4. 预算 score panel(可缓存)
   cache_key = f"scorepanel_{cfg.content_hash}.parquet"
   path = Path(cfg.portfolio_backtest.score_cache_dir) / cache_key
   if path.exists() and not refresh:
       score_panel = pd.read_parquet(path)
   else:
       score_panel = precompute_scores_from_legacy(legacy, panel_data)
       score_panel.to_parquet(path)

5. 装配 portfolio components
   portfolio_strategy = PrecomputedScoreStrategy(score_panel)
   eligibility = EligibilityFilter(
       cfg=EligibilityConfig(**cfg.portfolio_backtest.eligibility.dict()),
       name_map=name_map,
   )
   engine = PortfolioEngine(
       strategy=portfolio_strategy,
       eligibility=eligibility,
       sector_map=sector_map,
       portfolio_cfg=PortfolioConfig(**cfg.portfolio_backtest.portfolio.dict()),
       costs=cfg.backtest.costs,
       risk_free_rate=cfg.backtest.risk_free_rate,
   )

6. 跑 backtest
   n = cfg.portfolio_backtest.staggered_starts
   if n > 1:
       runner = StaggeredRunner(engine_factory=...)
       result = runner.run(panel_data, n_offsets=n)
   else:
       result = engine.run(panel_data, start_offset=0)

7. 渲染报告
   render_portfolio_report(result, output_dir=...)
   # reports/portfolio/<YYYY-MM-DD>.html + latest.html
```

### 6.2 Engine 内部 bar-by-bar 循环(伪代码)

```python
def run(self, panel_data, start_offset=0):
    dates = sorted(union of all dates in panel_data)
    portfolio = Portfolio(cash=self.cfg.initial_cash)
    rebalance_bars = self._compute_rebalance_bars(dates, start_offset)
    # rebalance_bars = {start_offset, start_offset + n, start_offset + 2n, ...}

    for t_idx, date_t in enumerate(dates):
        portfolio.mark_to_market(date_t, panel_data)

        if t_idx not in rebalance_bars:
            portfolio.snapshot(date_t)
            continue

        scores = self.strategy.predict_scores(date_t, panel_data)
        eligible = self.eligibility.eligible(date_t, panel_data)
        ranked = sorted(
            [(c, s) for c, s in scores.items() if c in eligible],
            key=lambda x: -x[1],
        )
        target = self._select_with_industry_cap(
            ranked, self.cfg.top_k, self.sector_map,
        )

        # T+1: 决策在 bar t,成交在 bar t+1 open
        if t_idx + 1 < len(dates):
            portfolio.rebalance_to(
                target_codes=target,
                exec_bar_idx=t_idx + 1,
                panel_data=panel_data,
                costs=self.costs,
            )

    return PortfolioBacktestResult(...)
```

### 6.3 关键时序细节(T+1)

- 决策 bar 是 `date_t`(看到 `close[t]`),成交在 `open[t+1]` 集合竞价
- mark-to-market 用 `close[t]`
- 与现有 `MultiLotBacktestEngine` T+1 语义一致

### 6.4 Rebalance bar 计算

`rebalance_n_days=5, start_offset=k` 下,rebalance bar 索引集合 = `{k, k+5, k+10, k+15, ...}`(以 bar index 而非日历日为单位,避开节假日扰动)。

### 6.5 Score panel 缓存策略

- 路径:`data/portfolio_scores/<cfg.content_hash>.parquet`(`score_cache_dir` 可配)
- 缓存键 = `cfg.content_hash`,覆盖整份 yaml(已有字段)
- 失效:改 yaml 任一字段(包括 portfolio_backtest 段的 top_k 等)都会触发重算
  - **已知 suboptimal**:改 top_k 理论上不影响 score panel,但仍会失效缓存。第一版接受这一冗余;真正想做"只在 strategy 段变时失效"的话需要新增"partial hash"机制,见 §12 followup
- AB 场景:两个 arm 各算各的,**不共享**(各 arm content_hash 不同)
- 复用率:同一份 yaml 跑两次 → 第二次秒级完成(只需 engine simulation)

### 6.6 Composite_verdict 在 portfolio 框架下的特殊性

`CompositeVerdictStrategy.generate_signals()` 不走 ML 训练,但仍输出 `final_score` 列。`precompute_scores_from_legacy()` 对它和 ML 走同一路径,没有训练成本。这统一了 ML 和规则两种 strategy 的接入。

---

## 7. YAML 配置

### 7.1 新增顶层段

```yaml
# config.yaml 新增
portfolio_backtest:
  enabled: false                  # 默认关,opt-in

  portfolio:
    top_k: 20
    rebalance_n_days: 5
    max_per_industry: 5           # null = 不限
    initial_cash: 1.0

  eligibility:
    min_avg_amount_20d: 50_000_000
    exclude_st: true
    min_history_bars: 60

  staggered_starts: 1             # >1 启用 ensemble

  score_cache_dir: data/portfolio_scores
```

### 7.2 Pydantic schema 增量

```python
# src/stockpool/config.py
class PortfolioConfig(BaseModel):
    top_k: int = Field(default=20, ge=1, le=200)
    rebalance_n_days: int = Field(default=5, ge=1)
    max_per_industry: int | None = Field(default=5, ge=1)
    initial_cash: float = Field(default=1.0, gt=0.0)

class EligibilityConfig(BaseModel):
    min_avg_amount_20d: float = Field(default=5e7, ge=0.0)
    exclude_st: bool = True
    min_history_bars: int = Field(default=60, ge=1)

class PortfolioBacktestConfig(BaseModel):
    enabled: bool = False
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    staggered_starts: int = Field(default=1, ge=1, le=20)
    score_cache_dir: str = "data/portfolio_scores"

    model_config = ConfigDict(extra="forbid")

class AppConfig(BaseModel):
    # ... existing fields ...
    portfolio_backtest: PortfolioBacktestConfig = Field(
        default_factory=PortfolioBacktestConfig,
    )
```

### 7.3 与现有字段的复用关系

| 现有字段 | 在 portfolio_backtest 里如何用 |
|---|---|
| `backtest.costs.{buy_cost, sell_cost}` | 复用(`PortfolioEngine` 接同一 `TradeCosts`) |
| `backtest.risk_free_rate` | 复用(metrics 里 sharpe 用) |
| `backtest.sizing` | **不复用**(portfolio 是等权 top-K) |
| `backtest.engine` | **不复用**(`single`/`multi_lot` 是 per-stock 引擎) |
| `data.*` | 复用 |
| `strategy.*` | 复用(legacy 怎么训怎么训) |
| `stocks` | **不用**(universe 来自 `load_universe_cache`) |
| `recommend_pool.*` | **不用**(eligibility 单独配,Pool B 是应用层概念) |

### 7.4 CLI 调用

```bash
python -m stockpool portfolio-backtest --config config.yaml
python -m stockpool portfolio-backtest --config config.yaml --refresh-scores
```

**Config 字段不接受 CLI 覆盖** — `top_k` / `staggered_starts` / `rebalance_n_days` 等全部走 yaml,跟项目 yaml-first 风格一致。

**操作性 flag** 允许:
- `--refresh-scores`:强制重算 score panel(忽略 `data/portfolio_scores/` 缓存),不改变 config 语义

---

## 8. Staggered Ensemble

### 8.1 Offset 错位机制

`PortfolioEngine.run(panel_data, start_offset=k)`:rebalance bar 索引整体右移 k 个 bar。

例(`rebalance_n_days=5, staggered_starts=5`):

```
bar index: 0  1  2  3  4  5  6  7  8  9  10 11 12 ...
offset=0:  R           R           R           R   ...    (bars 0,5,10,...)
offset=1:     R           R           R           R...    (bars 1,6,11,...)
offset=2:        R           R           R           R    (bars 2,7,12,...)
offset=3:           R           R           R          R  (bars 3,8,13,...)
offset=4:              R           R           R          (bars 4,9,14,...)
```

每 offset 跑完整回测,得各自 `PortfolioBacktestResult`。

### 8.2 `StaggeredRunner` 实现

```python
# src/stockpool/portfolio/ensemble.py
@dataclass
class EnsembleResult:
    individual_results: list[PortfolioBacktestResult]
    ensemble_curve: pd.DataFrame                # 等权均值净值
    envelope: pd.DataFrame                      # date / min / p25 / median / p75 / max
    aggregated_metrics: dict

class StaggeredRunner:
    def __init__(self, engine_factory):
        """engine_factory: callable returning new PortfolioEngine
        (每次 run 需要 fresh portfolio 状态)."""
        self._engine_factory = engine_factory

    def run(self, panel_data, n_offsets: int) -> EnsembleResult:
        # 串行(第一版,并行 followup)
        results = [
            self._engine_factory().run(panel_data, start_offset=k)
            for k in range(n_offsets)
        ]
        return self._aggregate(results)
```

### 8.3 报告呈现

`staggered_starts > 1` 时:

- 指标表多列:`| ensemble | median | min | max |`
- 净值图:灰色 min-max band + 黑色 median 线 + 红色 ensemble(均值)线 + 蓝色 B&H baseline
- 每 offset 卡片(折叠):各自 sharpe/return/dd

`staggered_starts = 1` 时:band/median/ensemble 退化成同一条;同模板,无需写两套渲染逻辑。

### 8.4 关键设计决策

- **Ensemble 等权均值 = 真实可执行 portfolio**(等价于"周一开仓 1/5 资金、周二 1/5、...、周五 1/5"的实战部署),不是事后凑的统计 artifact
- **同时报 median 和 ensemble**:median 反映"典型 offset",ensemble 反映"实际部署";通常接近但不一样
- **`sharpe_range`、`max_dd_worst` 给 sensitivity 视角**:跨度大 → 策略对开仓日期敏感 → 不稳健
- **不报 p 值 / t-test**:用包络宽窄替代显著性(包络窄 + 不重叠 = 鲁棒差异;包络重叠多 = 不可靠)

---

## 9. Portfolio AB 框架

### 9.1 子包结构(平行于现有 `ab/`)

```
src/stockpool/portfolio_ab/
├── __init__.py
├── config.py          # PortfolioABConfig + PortfolioArmOverride
├── runner.py          # run_portfolio_ab + ArmResult + ABResult
└── report.py          # HTML 对比报告
```

不复用 `ab/`,因为允许覆盖的字段不同(portfolio AB 多 `portfolio_backtest` 段、少 `stocks_filter`)。

### 9.2 `portfolio_ab.yaml` 示例

```yaml
base_config: config.yaml

arms:
  arm_lasso_ic:
    strategy:
      name: ml_factor
      ml_factor:
        selector: {type: lasso}
        weighter: {type: ic}
        factors_file: reports/selection.json

  arm_lgb_lgb:
    strategy:
      name: ml_factor
      ml_factor:
        selector: {type: lightgbm}
        weighter: {type: lightgbm}
        factors_file: reports/selection.json
    portfolio_backtest:
      portfolio:
        top_k: 15
```

### 9.3 `PortfolioABConfig` schema

```python
# src/stockpool/portfolio_ab/config.py
class PortfolioArmOverride(BaseModel):
    strategy: dict | None = None              # 整段替换
    portfolio_backtest: dict | None = None    # 字段级合并
    model_config = ConfigDict(extra="forbid")

class PortfolioABConfig(BaseModel):
    base_config: str
    arms: dict[str, PortfolioArmOverride] = Field(..., min_length=2, max_length=2)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_arms_count(self):
        if len(self.arms) != 2:
            raise ValueError(f"portfolio AB 必须恰好 2 个 arm,当前 {len(self.arms)}")
        return self
```

### 9.4 Runner 流程

```python
@dataclass
class ArmResult:
    name: str
    effective_cfg: AppConfig
    individual: list[PortfolioBacktestResult]
    ensemble: EnsembleResult | None
    failed: bool
    error: str | None = None

@dataclass
class ABResult:
    arms: dict[str, ArmResult]

def run_portfolio_ab(ab_cfg: PortfolioABConfig) -> ABResult:
    base = load_config(ab_cfg.base_config)
    arms_results = {}
    for arm_name, override in ab_cfg.arms.items():
        effective = build_effective_cfg(base, override)
        try:
            arm_result = run_single_arm(arm_name, effective)
        except Exception as e:
            arm_result = ArmResult(
                name=arm_name, effective_cfg=effective,
                individual=[], ensemble=None,
                failed=True, error=str(e),
            )
        arms_results[arm_name] = arm_result
    return ABResult(arms=arms_results)
```

### 9.5 报告布局

```
[Banner] arm_a vs arm_b,strategy / selector / weighter / top_k diff 高亮
[Aggregated 指标对比表] 两列 + Δ + Δ%(staggered 时多 ensemble/median 两栏)
[净值对比图] 两条 ensemble 曲线 + 各自 min/max band + B&H baseline
[Per-stock 贡献分解]
  - Top 15 贡献股 in arm_a / in arm_b(表)
  - 散点图: x=arm_a 贡献%, y=arm_b 贡献%
  - 集合分析: 仅 a 持 / 仅 b 持 / 都持(Venn 风格)
[失败提示] 一边失败时红色 banner
```

### 9.6 CLI

```bash
python -m stockpool portfolio-ab --config portfolio_ab.yaml

# 单 arm 调试(stdout 指标,不出 HTML)
python -m stockpool portfolio-ab --config portfolio_ab.yaml --arm arm_lasso_ic
```

### 9.7 关键决策

- **新建 `portfolio_ab/`,不复用 `ab/`**:ArmOverride 字段不同;报告布局完全不同;演化解耦更干净
- **每 arm 独立算 score panel**:承接 §6.5 决策;同 hash 共享优化留 followup
- **`--arm` 调试模式**:复现现有 per-stock AB 同名 flag
- **失败处理**:一 arm 失败仍出半张 HTML(per per-stock AB 一致行为)
- **散点坐标用 normalized contribution %**:跨 arm 更可比

---

## 10. PR 拆分

### 10.1 总览

```
PR-1 (核心骨架,MVP) ──┬──> PR-2 (universe + eligibility + 行业 cap)
                       │
                       ├──> PR-3 (staggered ensemble)
                       │
                       └──> PR-4 (portfolio AB,依赖 PR-2 + PR-3)
```

PR-2 / PR-3 互相独立,可并行;PR-4 在 PR-2 + PR-3 合并后启动。

### 10.2 PR-1:Portfolio backtest 核心骨架(MVP)

**新文件**:
- `src/stockpool/portfolio/__init__.py`
- `src/stockpool/portfolio/strategy.py` — `PortfolioStrategy` + `PrecomputedScoreStrategy`
- `src/stockpool/portfolio/scoring.py` — `precompute_scores_from_legacy()`
- `src/stockpool/portfolio/engine.py` — `PortfolioEngine` + `PortfolioConfig` + `Portfolio` 内部类
- `src/stockpool/portfolio/result.py` — `PortfolioBacktestResult` + `PortfolioTrade`
- `src/stockpool/portfolio/report.py` — 单 arm HTML(指标表 + 净值 + 持仓时间线,**无包络**)

**修改**:
- `src/stockpool/config.py` 加完整 `PortfolioBacktestConfig`(含 eligibility schema,**但 PR-1 的 engine 暂不读 eligibility 字段;universe 直接用 cfg.stocks**)。完整 schema 提前进 PR-1 是为了避免 PR-2 改动 schema 触发用户 yaml 迁移
- `src/stockpool/cli.py` 加 `cmd_portfolio_backtest`

**新测试**:
- `tests/test_portfolio_strategy.py`
- `tests/test_portfolio_engine.py`
- `tests/test_portfolio_scoring.py`
- `tests/test_cli_portfolio_backtest.py`

**退化策略**(PR-1 独有):
- `universe = cfg.stocks` 全部
- 无 eligibility 过滤
- 无行业 cap(top_k 取分数前 K)

**验收**:
```bash
python -m stockpool portfolio-backtest --config config.yaml
# 期望: reports/portfolio/<date>.html 生成,有 1 条净值曲线
```

**估算**:8-10 个 task。

### 10.3 PR-2:Universe + Eligibility + 行业 cap

**新文件**:
- `src/stockpool/portfolio/eligibility.py` — `EligibilityFilter` + `EligibilityConfig`

**修改**:
- `src/stockpool/portfolio/engine.py` 加 `_select_with_industry_cap()` 方法
- `src/stockpool/config.py` 加 `EligibilityConfig` 段
- `cli.cmd_portfolio_backtest` 切到 `load_universe_cache` + `sector_map` + `name_map`

**新测试**:
- `tests/test_portfolio_eligibility.py`
- `tests/test_portfolio_industry_cap.py`
- `tests/test_cli_portfolio_backtest_universe.py`

**前置**:`fetch-universe` 已跑过

**验收**:
```bash
python -m stockpool portfolio-backtest --config config.yaml
# 期望: 持仓覆盖到 cfg.stocks 之外;同一行业不超过 5 只
```

**估算**:5-7 个 task。

### 10.4 PR-3:Staggered ensemble

**新文件**:
- `src/stockpool/portfolio/ensemble.py` — `StaggeredRunner` + `EnsembleResult`

**修改**:
- `src/stockpool/portfolio/engine.py:run()` 加 `start_offset` 参数
- `src/stockpool/portfolio/report.py` 加包络图 + per-offset 卡片
- `cli.cmd_portfolio_backtest` 检测 `staggered_starts > 1` 走 ensemble 路径

**新测试**:
- `tests/test_portfolio_ensemble.py`
- `tests/test_portfolio_report_ensemble.py`

**串行实现**(并行留 followup)。

**验收**:`staggered_starts: 5` 跑出 5 条线 + median + ensemble + 包络。

**估算**:4-6 个 task。

### 10.5 PR-4:Portfolio AB

**新子包**:`src/stockpool/portfolio_ab/`(`config.py` / `runner.py` / `report.py`)

**修改**:
- `cli.py` 加 `cmd_portfolio_ab`

**新测试**:
- `tests/test_portfolio_ab_config.py`
- `tests/test_portfolio_ab_runner.py`
- `tests/test_portfolio_ab_report.py`
- `tests/test_cli_portfolio_ab.py`

**新例子**:`portfolio_ab.yaml.example`

**验收**:
```bash
python -m stockpool portfolio-ab --config portfolio_ab.yaml
# 期望: reports/portfolio_ab/<date>.html 出对比报告
```

**估算**:7-9 个 task。

### 10.6 文档维护(per CLAUDE.md 规则)

每个 PR 同步更新:
- `CLAUDE.md` 模块地图 / 配置 / 测试段
- `README.md` 常用命令 / 端到端示例
- `docs/strategy_improvement_2026.md` §6 路线图(打勾)

### 10.7 跨 PR 共通约束

- 不引入新依赖(portfolio 框架纯 pandas + numpy + pyecharts)
- 每个 PR 走 `docs/superpowers/specs/` + `docs/superpowers/plans/` 流程
- 每个 PR 末附一份 A/B 对照报告(同一段历史,新 config vs 旧 config 7 个指标)

---

## 11. 测试策略

### 11.1 测试金字塔

```
              ┌─────────────────────────────┐
              │  CLI smoke (4 files)         │
              └─────────────────────────────┘
            ┌────────────────────────────────┐
            │  Integration (5 files)          │
            └────────────────────────────────┘
       ┌──────────────────────────────────────┐
       │  Unit (8 files)                       │
       └──────────────────────────────────────┘
```

新增 ~17 个测试文件,跨 4 个 PR。

### 11.2 测试模式(沿用现有项目惯例)

- **合成 OHLCV**(参考 `tests/test_cli_backtest.py`):不依赖真实 fetcher
- **monkeypatch 外部依赖**:`load_universe_cache`、`load_or_build_industry_map`、`precompute_scores_from_legacy`
- **Strategy stub**:测 engine 用 `StubPortfolioStrategy(scores_by_date)`,避免 ML 训练耗时

### 11.3 必须覆盖的核心契约

**Engine**:
- T+1 契约(bar t 决策、bar t+1 open 成交)
- Cash 守恒(`cash + Σ position_value == total_equity`,浮点误差 < 1e-9)
- Rebalance diff 逻辑(从 {A,B,C,D,E} 转到 {B,C,D,F,G} → 卖 A E,买 F G,B C D 重等权)
- 确定性(同 panel + 同 N,跑两次结果一致)

**Score panel(关键!)**:
- **No look-ahead**:对同一只票造两组 panel,A 截到 bar T,B 截到 bar T+10。比较 `score[0..T]` 应一致
- **回归测试**:`MLFactorStrategy` + 已知 seed,断言 score panel 哈希稳定

**Eligibility**:
- 流动性阈值边界(4999 万剔、5001 万留)
- ST 排除
- min_history_bars 阈值
- 空池 fallback(全部过滤掉时全持现金,不 crash)

**行业 cap**:
- 贪心算法正确性(scores=[1.0,0.9,0.8,0.7,0.6], 行业=[A,A,A,B,B], k=3, cap=2 → [1.0,0.9,0.7])
- "未知"行业 fallback(全 unknown 跳过 cap,部分 unknown 正常计)— 跟 Pool B 一致

**Ensemble**:
- N=1 退化等价于单跑
- Offset 不重叠(N=5, n_days=5 → 5 个 offset 的 rebalance bar 集合不相交)
- Ensemble = 等权均值(数学等价)

**AB**:
- `extra=forbid`(`PortfolioArmOverride` 不允许 `stocks_filter` / `indicators` 等)
- Arms count(< 2 或 > 2 报错)
- Build_effective_cfg(strategy 段整段替换、portfolio_backtest 字段级合并)
- 失败隔离(一 arm 挂,另一 arm 仍生成,报告红色提示)

### 11.4 性能预期(informational,非 hard gate)

- PR-2 完成后:4350 只全市场 + 1000 bar + `composite_verdict` 应 < 60s
- PR-1 完成后:8 只 cfg.stocks + 500 bar + `composite_verdict` 应 < 5s

慢于预期提 followup,不挡 PR。

### 11.5 测试套件影响

| 阶段 | 测试数 |
|---|---|
| 现状 | 374 |
| PR-1 后 | ≈ 380 |
| PR-2 后 | ≈ 383 |
| PR-3 后 | ≈ 385 |
| PR-4 后 | ≈ 391 |

`pytest tests/ -q` 目标 < 60s(现在 ~20s)。

---

## 12. 范围外 + 已知局限

| 项 | 为什么不做 | 何时再考虑 |
|---|---|---|
| 生存偏差修正 | `universe.parquet` 是今天的快照,历史退市股不在 | 接历史 universe 数据源 |
| 历史 ST 状态 | `name_map` 用当前 name | 接历史 ST 标记数据 |
| 历史行业映射 | `sector_map` 用当前快照 | 接历史 industry data |
| 持仓部分调整 / turnover 上限 | 第一版全量 rebalance 到等权 | 加 `turnover_cap` |
| 滑点模型 | 第一版只有固定比例 costs | 加 `slippage_model` |
| 涨停打不进单过滤 | 第一版假设次日 open 总能成交 | 加 `daily_limit_check` |
| 空头 / 期货 | 项目定位 long-only | 不打算做 |
| AB > 2 arm | schema 强制 2 | 真有用户需求时扩 |
| AB 统计显著性 p 值 | 用包络替代 | 长样本(>20 年)时考虑 |
| Score panel 跨 arm 共享缓存 | 各 arm hash 不同,已决策不共享 | AB 跑慢成瓶颈时 |
| Score panel partial-hash 失效 | 缓存键用全 content_hash,改 top_k 也会冗余重算 | 真有人观察到 portfolio_backtest 字段频繁调而重算成本高时 |
| Staggered 并行化 | 第一版串行 | 加 `ProcessPoolExecutor` |
| PortfolioMLFactorStrategy(panel-batch) | precompute 已够用 | score panel 成瓶颈时 |
| 风控 overlay(max DD / 动态 sector cap) | F3 PR-D 已撤销 | 真需求时单独 spec |
| Score smoothing (EMA) | 列为 portfolio 框架后的 P3 | portfolio AB 跑通后观察 churn |

---

## 13. 路线图整合(`docs/strategy_improvement_2026.md` §6)

写本 spec 时**同步更新** §6 路线图:

### 13.1 待做 P1 段替换

```diff
- ~~PR-D Risk overlay~~     # 撤销:单股语境下风控退化
- ~~PR-E Score smoothing~~  # 推后到 portfolio 框架完成后

+ Portfolio framework (4 PRs, 见本 spec §10)
+   PR-1 核心骨架(MVP)
+   PR-2 Universe + Eligibility + 行业 cap
+   PR-3 Staggered ensemble
+   PR-4 Portfolio AB
```

### 13.2 顺位调整

- 原 P2(F1 plan-2 自定义因子 + WQ101 ranking)→ 保持 P2,跟 portfolio 独立
- 新增 P3:Score smoothing(原 F3 PR-E,推后)
- 原 P3 工具改进表格 → 平移到 P4

### 13.3 暂搁段新增

- **Per-stock risk overlay(原 F3 PR-D)** — 触发条件:per-stock backtest 真的需要 DD 熔断作为信号过滤;理由:单股语境下 sector cap 退化,DD 熔断与"比较信号"目的脱节

### 13.4 跨阶段约束(继承自 §6)

- 任何改默认值的 PR 附 A/B 报告作证据
- 不引入新依赖(本 spec 满足)
- 每个 PR 走 specs/ + plans/ 流程

---

## 14. 验收标准(整体)

本 spec 4 个 PR 全部合入后,应能:

1. **跑通 portfolio backtest**:
   ```bash
   python -m stockpool portfolio-backtest --config config.yaml
   ```
   全市场 universe + dynamic eligibility + top-K + 行业 cap + 可选 staggered ensemble,输出 HTML 报告。

2. **跑通 portfolio AB**:
   ```bash
   python -m stockpool portfolio-ab --config portfolio_ab.yaml
   ```
   对比两个 strategy 的组合表现,出双 arm HTML(指标 + 净值 + per-stock 贡献分解)。

3. **零回归**:
   - `python -m stockpool backtest --config config.yaml` 不动
   - `python -m stockpool ab --config ab.yaml` 不动
   - `pytest tests/ -q` 全绿(374 → ~391 个测试)

4. **`docs/strategy_improvement_2026.md` §6 路线图反映新状态**:4 个 portfolio PR 都标记 ✅,每个挂 A/B 验证链接。

---

**文档维护**:
- 本 spec 完成对应的 plan 后,在 §10 各 PR 段加 plan 链接
- 路线图发生变化(顺序、范围、放弃)时,在本文末加 changelog 段
