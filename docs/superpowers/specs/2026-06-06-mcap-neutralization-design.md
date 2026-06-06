# Market-Cap 中性化 — Design Spec

**Date:** 2026-06-06
**Status:** Draft, pending implementation
**Phase:** Phase 2 (factor preprocessing 续作,接续 Phase 1 / 1.5)
**Owner:** stockpool / ml.preprocess
**Tracking branch:** TBD (建议 `feat/mcap-neutralize`)

---

## 1. 问题陈述

小盘 vs 大盘的风格暴露在当前 ML pipeline 里完全没剥离。具体症状:

- `book_to_price` 在小盘股堆积 — 小盘股结构性 PB 偏低,不是真 alpha
- `momentum` 在大盘股堆积 — 大盘动量更平滑,不是真 alpha
- 任何价值/质量/动量类因子的截面差异都被 **size β** 污染

Phase 1 的 `cs_zscore` 把每日 μ/σ 归一了,但 z-score 不剥离**与其他变量(尤其 log market cap)线性相关**的部分;`industry_neutralize` 当前是纯组内 demean,也没碰 mcap。

**目标:**

- 加 `log(market_cap)` 作为可选的中性化协变量
- `industry_neutralize_panel` 升级到能跑联合 OLS(industry dummies + log_mcap)
- 新增独立开关 `PreprocessConfig.mcap_neutralize: bool = False`,可单独剥 mcap 不剥 industry
- 用 A/B 验证 mcap 中性化是否在 4000+ 票训练池上带来 Sharpe / annualized return 提升

**非目标:**

- 不引入 `mcap` 之外的连续协变量(beta、流动性、波动率 ...)。如有需要后续 PR
- 不改 `industry_neutralize=True` **不带 mcap** 时的数值行为(向后兼容)
- 不动 Phase 1 的 `winsorize` / `cs_zscore` 实现

---

## 2. 决策汇总(brainstorming 阶段定稿)

| Q | 选择 | 理由 |
|---|---|---|
| `industry_neutralize` 和 `mcap_neutralize` 关系 | **两个独立开关,内部按需合并成单次 OLS** | AB 能干净隔离 mcap 贡献;符合用户研究问题 |
| fundamental 因子要不要剥 mcap | **剥(细粒度版)**:ROE/margin/yoy 剥,PE/PB 不剥(分子含 close × shares 强共线) | 保留剥 size β 的研究价值,避免 PE/PB 数值病态 |
| `market_cap` 是否注册成因子 | **是,顺便注册 `market_cap` + `log_market_cap`** | 让用户能选进 selection.json 独立测;cost ~30 行 |
| mcap_panel 是否落盘 cache | **否,每次 build_factor_panel 现 build** | ~3000 stocks × 250 days 几十毫秒;避免引入新 cache 文件 |
| AB 默认对照组 | **`preprocess_only` vs `preprocess_plus_mcap`**(industry 都关) | 与产线默认对齐(industry_neutralize=false);先验单变量效应,联合那组之后再做 |
| OLS 后端 | **numpy `lstsq` 逐日** | 不引 statsmodels / sklearn 新依赖 |
| 单成员 industry 处理 | **当日 drop 那只股出回归**(非合并到 `_unknown_`) | 避免污染 unknown 残差;沿用 Phase 1.5 教训 |

---

## 3. 数据通路

### 3.1 mcap_panel 构造

新增 helper `build_log_mcap_panel`,放在 `src/stockpool/ml/mcap.py`(新文件,薄薄一个模块,避免 `strategy_factory.py` 继续膨胀):

```python
# src/stockpool/ml/mcap.py
from __future__ import annotations
import numpy as np
import pandas as pd

def build_log_mcap_panel(
    panel: dict[str, pd.DataFrame], cache_dir
) -> pd.DataFrame:
    """build T×N log(market_cap) panel, PIT-aligned by pubDate.

    mcap = close × totalShare;mcap ≤ 0 → NaN(让 per-day OLS dropna 处理)。
    """
    from stockpool.fundamentals_loader import load_or_build_fundamentals
    from stockpool.factors.fundamentals import _pit_align
    balance = load_or_build_fundamentals("balance", cache_dir=cache_dir)
    shares_panel = _pit_align(balance, "totalShare", panel["close"])
    mcap = panel["close"] * shares_panel
    return np.log(mcap.where(mcap > 0))
```

- `_pit_align` 是 `factors/fundamentals.py` 私有 helper(下划线前缀),但同包内复用是 OK 的模式;不另暴露成公开 API,避免下游误用
- `cache_dir` 由 `build_factor_panel` 顶层传入(跟它写入 `factor_panels/` 的根目录一致)
- `balance` 表来自 baostock,30 天 parquet 缓存,无新 fetch
- 调用点:`strategy_factory.build_factor_panel(cfg, ...)` 在 `cfg.preprocess.mcap_neutralize=True` 时调用并把 `log_mcap` panel 透传给 `apply_preprocess_pipeline`;否则传 `None`

### 3.2 market_cap / log_market_cap 作为可选因子

在 `factors/fundamentals.py` 增 2 个 class:

```python
@register(
    "market_cap",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "size"),
    description="总市值 (close × 总股本)。规模因子,小盘溢价 / 大盘稳定的常用代理。",
)
class MarketCapFactor(Factor):
    def compute(self, panel):
        balance = load_or_build_fundamentals("balance", cache_dir=_default_cache_dir())
        shares_panel = _pit_align(balance, "totalShare", panel["close"])
        return panel["close"] * shares_panel  # NaN where shares missing

@register(
    "log_market_cap",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "size"),
    description="log(总市值)。剥离市值 β 时常用;线性回归更稳定。",
)
class LogMarketCapFactor(Factor):
    def compute(self, panel):
        mcap = MarketCapFactor().compute(panel)
        return np.log(mcap.where(mcap > 0))
```

新 type tag `"size"` 加进允许列表(`factors/registry.py` 已经支持任意 tag,不需要改 schema)。

---

## 4. OLS 数学层

### 4.1 公开函数签名(`ml/preprocess.py`)

```python
def mcap_neutralize_panel(
    df: pd.DataFrame, log_mcap: pd.DataFrame,
) -> pd.DataFrame:
    """Per-day 残差化:Y_t = α_t + β_t · log_mcap_t + ε_t,返回 ε_t。

    Args:
        df: T × N factor 宽表。
        log_mcap: T × N,与 df 同对齐(index 同,columns 子集即可)。

    Returns:
        Same shape as df。NaN cells 保持 NaN;退化日见 §4.3 fallback。
    """

def industry_neutralize_panel(
    df: pd.DataFrame,
    sector_map: Mapping[str, str],
    log_mcap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-day 残差化。

    - log_mcap=None: 当前行为(快速分组 demean,bit-for-bit 不变)
    - log_mcap=given: OLS Y ~ industry_dummies(drop first) + log_mcap,返残差。
    """
```

### 4.2 逐日 OLS 实现

`_per_day_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray`(模块私有):

1. 当日 cross-section 取 `y_valid = y[~isnan(y) & ~isnan(X).any(axis=1)]`
2. 单成员 industry → drop 那个 code(detect:industry_dummies 列和 == 1)
3. 校验 `len(y_valid) >= 10`(硬下限,数值上低于 10 个观测对 K-1 + 1 维回归严重 unstable) 且 `X_valid.shape[0] > X_valid.shape[1]`(超定);否则当日 fallback:
   - `industry_neutralize` 走 demean,`mcap_neutralize` 不修改(原值返回)
4. `coef, *_ = np.linalg.lstsq(X_valid, y_valid, rcond=None)`
5. `resid_valid = y_valid - X_valid @ coef`
6. 写回 `out[date]` 对应位置;NaN 位置不动

**Multi-factor 优化(向量化机会):**同一天 N 个 factor 用同一份 X(industry_dummies + log_mcap 跟 factor 无关),可以把 N 个 y 拼成 `Y_t (codes × n_factors)` 一次 lstsq:`coef_block = lstsq(X, Y)`,残差矩阵一次性算完。每日省 N-1 次 SVD。先实现简单版(逐 factor),profiler 看到 hot spot 后再 vectorize。

### 4.3 退化场景与日志

| 场景 | 行为 |
|---|---|
| 当日 valid 数 < 10 | fallback(demean / no-op),计数器 +1 |
| X 满秩检查 fail(全同 industry 等) | fallback,计数器 +1 |
| log_mcap 全 NaN 的日子 | mcap 不参与;industry 路径正常 demean |
| 单成员 industry 的列 | 把那只股从当日回归 drop,不丢回 `_unknown_` 桶 |
| `n_codes < min_pool_size`(P1.5 panel-level guard) | 跳过整个 pipeline,行为已有 |

`apply_preprocess_pipeline` 退出前聚合 warning:

```
log.warning(
    "OLS neutralize: fallback on %d / %d days (degenerate cross-section). "
    "Threshold: valid >= 10 codes, design matrix full-rank.",
    fallback_days, total_days,
)
```

---

## 5. Config 层

### 5.1 PreprocessConfig 扩展(`src/stockpool/config.py`)

```python
class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winsorize: tuple[float, float] | None = None
    zscore: bool = False
    industry_neutralize: bool = False
    mcap_neutralize: bool = False        # ← 新增
    min_pool_size: int = Field(default=200, ge=0)

    @field_validator("winsorize")
    ...
```

`extra="forbid"` 保证拼错字段直接 raise。

### 5.2 `_is_all_off` 跟进

```python
def _is_all_off(cfg: "PreprocessConfig") -> bool:
    return (
        cfg.winsorize is None
        and cfg.zscore is False
        and cfg.industry_neutralize is False
        and cfg.mcap_neutralize is False     # ← 新增
    )
```

### 5.3 `apply_preprocess_pipeline` 签名

```python
def apply_preprocess_pipeline(
    factor_panel: dict[str, pd.DataFrame],
    cfg: "PreprocessConfig",
    sector_map: Mapping[str, str] | None = None,
    factor_types: Mapping[str, tuple[str, ...]] | None = None,
    n_codes: int | None = None,
    log_mcap_panel: pd.DataFrame | None = None,   # ← 新增
) -> dict[str, pd.DataFrame]:
    ...
```

### 5.4 调用矩阵(每个 factor 独立判定 skip)

| `industry` | `mcap` | log_mcap 可用 | factor tag | 实际运行 |
|---|---|---|---|---|
| F | F | — | — | (无 neutralize) |
| T | F | — | non-fundamental | `industry_neutralize_panel(df, sector_map)` |
| T | F | — | fundamental | skip(legacy) |
| F | T | yes | non-fundamental / fundamental-non-pe-pb | `mcap_neutralize_panel(df, log_mcap)` |
| F | T | yes | `contains_mcap`(PE/PB) | skip mcap |
| T | T | yes | non-fundamental | `industry_neutralize_panel(df, sector_map, log_mcap)` 联合 OLS |
| T | T | yes | fundamental-non-pe-pb | `mcap_neutralize_panel(df, log_mcap)` only(industry legacy skip) |
| T | T | yes | `contains_mcap`(PE/PB) | skip 两步 |
| any | T | **None** | any | log warning "mcap_neutralize=True but log_mcap_panel missing" + skip mcap;industry 照常 |

### 5.5 PE/PB 因子加 tag

在 `factors/fundamentals.py` 给 `PEFactor` / `PBFactor` 的 `@register` 类型元组追加 `"contains_mcap"`:

```python
@register(
    "pe",
    sources=("custom",),
    types=("fundamental", "cross_sectional", "contains_mcap"),  # ← 加
    description=...,
)
class PEFactor(Factor): ...
```

preprocess 读 `factor_types[name]` 看到 `"contains_mcap"` 就 skip mcap 步骤。

---

## 6. Cache 失效

### 6.1 factor_panels/<sig>/

`PreprocessConfig` 的所有字段已经 baked 进 `<sig>` 的 sha256 hash(P1.5 落地时已完成)。加 `mcap_neutralize` 字段自动改 sig — 旧 panel cache 不复用,自动重算。旧 cache 文件保留不删(磁盘占用是已知 follow-up)。

### 6.2 ml_models/<sig>_*.pkl

同样,`PreprocessConfig` 在 `MLFactorConfig.content_hash` 里 — 改 mcap_neutralize 后旧训练 pkl 自动失效。

### 6.3 mcap_panel 不单独缓存

每次 `build_factor_panel(mcap_neutralize=True)` 现 build:
- `balance` 表本身是 parquet 缓存(30 天)
- `_pit_align` + 乘 close + log,4000 票 × 250 天 < 100ms 量级
- 加单独 cache 文件徒增复杂度

---

## 7. A/B 验证方案

### 7.1 `ab_mcap.yaml`(新增,平行于 `ab_preprocess.yaml`)

```yaml
base_config: config.yaml
arms:
  preprocess_only:           # baseline,同 ab_preprocess.yaml 的 with_preprocess
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: all
        selector: &id001
          type: lasso
          lasso: {alpha: 0.001, max_iter: 1000, tol: 1.0e-06}
        weighter: &id002
          type: ic
          ic: {use_rank: true, min_abs_ic: 0.0}
        thresholds: &id003
          strong_buy: 0.9
          buy: 0.7
          sell: 0.3
          strong_sell: 0.1
        buy_verdicts: &id004 [buy, strong_buy]
        sell_verdicts: &id005 [sell, strong_sell]
        refresh_verdicts: &id006 [strong_buy]
        mask: {enabled: false}
        preprocess:
          winsorize: [0.01, 0.99]
          zscore: true
          industry_neutralize: false
          mcap_neutralize: false
    backtest:
      equity_curve_holding_days: [10]

  preprocess_plus_mcap:      # treatment
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: all
        selector: *id001
        weighter: *id002
        thresholds: *id003
        buy_verdicts: *id004
        sell_verdicts: *id005
        refresh_verdicts: *id006
        mask: {enabled: false}
        preprocess:
          winsorize: [0.01, 0.99]
          zscore: true
          industry_neutralize: false
          mcap_neutralize: true       # ← 唯一差异
    backtest:
      equity_curve_holding_days: [10]
```

### 7.2 跑法

```bash
# 1. 先 dry-run 单 arm 检查指标合理 + OLS warning 计数
python -m stockpool ab --config ab_mcap.yaml --arm preprocess_plus_mcap

# 2. 完整双 arm
python -m stockpool ab --config ab_mcap.yaml
```

### 7.3 验收门槛(参考 `docs/ab_validation_results.md` P4-1b 口径)

| Verdict | 判定 |
|---|---|
| **PASS** | `Δ Sharpe ≥ +0.10` **AND** `Δ ann_return ≥ +1%` |
| **HOLD** | `|Δ Sharpe| < 0.05` → 中性,保留开关但不进 default |
| **REJECT** | `Δ Sharpe ≤ -0.10` 或 `Δ ann_return ≤ -1%` |

结果追加段写到 `docs/ab_validation_results.md`,**不改 yaml default**(mcap_neutralize 默认 False 不变)。

### 7.4 (可选)第二组对照

如果第一组 PASS,可以加 `ab_mcap_with_industry.yaml`:
- baseline: `industry_neutralize=true, mcap_neutralize=false`
- treatment: `industry_neutralize=true, mcap_neutralize=true`

验证联合 OLS 是否进一步加分。**不在本 spec 第一轮提交范围内**。

---

## 8. 测试矩阵

新增测试文件 `tests/test_ml_preprocess_mcap.py`(参考 `test_ml_preprocess.py` 的组织方式):

### 8.1 单元测试 `mcap_neutralize_panel`

1. **happy path** — 2 day × 5 code 合成数据,log_mcap 是 known x,y = 2*x + ε,验证残差 ≈ ε
2. **NaN 安全** — y 含 NaN cell,X 含 NaN cell,正确 dropna 但形状保留
3. **退化日 fallback** — valid 数 < 10 → 当日返原值,计数器 +1
4. **log_mcap 全 NaN 的日** — 跳过 mcap 步骤,其他日子正常
5. **日间独立** — 改一日 log_mcap 不影响其他日

### 8.2 单元测试 `industry_neutralize_panel(..., log_mcap=...)`

1. **legacy 行为不变** — `log_mcap=None` 时 bit-for-bit 等于旧实现
2. **联合 OLS** — 2 industry × 5 code,验证残差正交于 industry dummy 和 log_mcap
3. **单成员 industry drop** — 一只股自成一组 + 多股一组,验证该股**不**被 demean 到 0(而是从回归中排除,保留原值)
4. **`_unknown_` 桶非合并** — sector_map 缺一只股,不混进单成员组里污染

### 8.3 集成测试 `apply_preprocess_pipeline`

1. **`mcap_neutralize=True, log_mcap_panel=None`** → warning + skip mcap,industry 正常
2. **`mcap_neutralize=True, fundamental factor with contains_mcap`** → PE/PB 跳 mcap,ROE 剥 mcap
3. **PreprocessConfig sig 变化** — 切 `mcap_neutralize=True` 触发新 `<sig>`,旧 cache 不复用

### 8.4 配置层测试 `tests/test_config.py`

1. `mcap_neutralize: bool = False` 默认 OK
2. `extra="forbid"` 仍然拒绝拼错字段
3. content_hash 含 `mcap_neutralize`(改值 hash 变)

### 8.5 工厂层测试 `tests/test_strategy_factory.py` 或新文件

1. `build_factor_panel(cfg.preprocess.mcap_neutralize=True)` 调用了 `_build_log_mcap_panel`
2. `mcap_neutralize=False` 时不构建(profiler / monkeypatch spy)

### 8.6 因子层测试 `tests/test_factors_fundamentals.py`

1. `make_factor("market_cap")` / `make_factor("log_market_cap")` 注册成功
2. compute 后形状对、PIT 对齐、shares 缺数时 NaN

### 8.7 AB smoke `tests/test_ab.py` 增量

`build_effective_cfg` 把 `mcap_neutralize` 字段正确合并到 effective_cfg,content_hash 重算。

---

## 9. 实施分解(suggested PR sequence)

### PR-1: 数据 + math + config
- `build_log_mcap_panel` helper 落到新文件 `src/stockpool/ml/mcap.py`,在 `strategy_factory.build_factor_panel` 入口调用
- `PreprocessConfig.mcap_neutralize` 字段
- `mcap_neutralize_panel` + `industry_neutralize_panel(..., log_mcap=...)`
- `apply_preprocess_pipeline` 接 `log_mcap_panel` 参数 + 调用矩阵
- 注册 `market_cap` / `log_market_cap` 因子
- PE/PB 加 `contains_mcap` tag
- 单元 + 集成 + 配置层测试(§8.1 - §8.6)
- 所有现有测试不破(向后兼容)

### PR-2: AB yaml + 跑结果
- `ab_mcap.yaml` 入 repo
- `--arm preprocess_plus_mcap` dry run 通过
- 全量双 arm 跑通,HTML 报告 inspection
- `docs/ab_validation_results.md` 追加 P5-mcap 段(PASS / HOLD / REJECT)
- CLAUDE.md / README.md 更新 PreprocessConfig 段落

### (可选)PR-3: 第二组对照
仅在 PR-2 PASS 且联合效应有研究价值时启动。

---

## 10. 文档更新(每个 PR 必更)

- `CLAUDE.md`
  - 配置段:`preprocess.mcap_neutralize` 加进字段表
  - 模块地图:`src/stockpool/ml/preprocess.py` 一行补 `+ mcap_neutralize`
  - 测试表:加 `test_ml_preprocess_mcap.py`
  - "已知不支持" 段:如有限制(例如尚未测 industry+mcap 联合)记一行
- `README.md`
  - 配置示例段如有 `preprocess:` block,加 `mcap_neutralize` 行 + 一句解释

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `balance` 表缺数据,大量 mcap NaN | per-day dropna + degenerate-day fallback;聚合 warning 输出 NaN 比例 |
| OLS 数值不稳(rcond / 几乎共线) | `np.linalg.lstsq(rcond=None)` 已带 SVD-based 截断;额外 `n>>k` 校验 |
| 全市场 mcap 分布偏态(头部超大) | log 变换;winsorize 由 pipeline 上游处理 |
| 性能回归 | 4000 × 250 天 × N factor 的 lstsq:估算 < 30s;profile 实测后决定是否需 vectorize |
| 用户改 mcap_neutralize 没清旧 ml_models pkl | sig 变化自动失效,无需手动清 |

---

## 12. Follow-ups(超出本 spec 范围)

- 联合 industry+mcap AB 对照
- 加 `beta` / `liquidity` 等更多连续协变量(变成 risk-model neutralization)
- mcap_panel 单独缓存(若 build 成本占比超 5%)
- size 因子组(`market_cap` / `log_market_cap` / `mcap_rank` / `mcap_quartile`)的因子分析报告

---

## 附录 A — 与论文 B (arXiv 2507.07107) 的对照

论文 B 在 risk-model 神经化阶段用 `industry + size + beta + liquidity` 四维。本 spec **只做 industry + size** 一步;beta/liquidity 列为 follow-up。与论文 B 的 mask-first finding 类似,我们也是先做单变量增量验证再扩展协变量集。
