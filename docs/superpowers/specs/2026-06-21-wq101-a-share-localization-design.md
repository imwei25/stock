# WQ101 因子的 A 股参数本土化 — 设计

**日期**:2026-06-21
**状态**:Draft(等用户审阅)
**前置依赖**:Phase 0(IC 诊断管道升级)可与本 spec 同 PR 也可独立先行;若已修复破损因子(`docs/.../broken-factors-fix-design.md`,尚未写)可选合并

---

## 1. 背景与动机

`reports/factor_analysis/2026-06-20.json`(4357 股 × 500 日 × 167 因子 × horizon=3)显示:

| 家族 | n | 平均 `abs_ic` |
|---|---|---|
| `single_stock_vol`(ATR/Parkinson/GK) | 4 | **0.1529** |
| `vwap_deviation` | 2 | **0.1366** |
| `fundamentals`(去除 3 个破损) | 6 | 0.1248 |
| `ewma` | 4 | 0.1248 |
| `builtin_or_other`(MA/MACD/KDJ…) | 25 | 0.0989 |
| **`wq101`** | **101** | **0.0796**(垫底,且 worst-50 中占 43 个) |

WQ101 整体 IC 不到本土家族一半。**主导假设**:WQ101 公式里的窗口常数(10 / 20 / 60 日)是美股换手率/波动率 calibrate 出来的;A 股相对美股有

- 换手率 3-5×
- 日波动 1.5-2×
- 散户主导 → 短期反转更强、中期动量更弱
- T+1 结算(已通过引擎处理,本 spec 不涉及)
- 涨跌停 / ST 制度(已通过 mask 层处理,本 spec 不涉及)

预期通过**只改窗口、不改公式**,WQ101 的有效因子能拿回 30-50% 的 |IC|。

---

## 2. 范围

| 改 | 不改 |
|---|---|
| WQ101 `ts_*` / `delay` / `correlation` 的窗口常数 | 公式本身(不重写 alpha) |
| `factors_analysis.analyze_factors` 加 IC 退化日诊断 + winsorize | horizon(独立轴,留下次) |
| 注册新变体,**保留原版**(可 AB) | 新增数据源(北上/龙虎榜) |

**YAGNI 否决**:不动 Lasso/IC 权重、不动 horizon、不动行业中性化粒度、不引新数据源。

---

## 3. Phase 0:IC 诊断管道升级(前置)

`factors_analysis.py:analyze_factors` 目前直接吃 raw factor panel 算 Spearman IC,既不 winsorize 也不检测横截面退化日。Agent B 已经验证 `alpha_096` 的 `abs_ic_mean=0.4773` 是因为 `ts_argmax(...)` 输出 0-12 离散整数,某些日子横截面 nunique=2,Spearman 在并列秩上吐出 ±1.0 噪声,占总日数 40.8%。**没修这个,本 spec 任何 IC 比较都不可信**。

### 3.0.1 改动

在 `analyze_factors` 签名追加两个可选 kwarg(默认开启,向后兼容只是它会让数字变化):

```python
def analyze_factors(
    panel, factor_names, horizon=3, ic_window=252,
    regime_index_close=None, method="spearman",
    winsorize: tuple[float, float] | None = (0.01, 0.99),   # 新增
    degenerate_day_unique_ratio_threshold: float = 0.01,    # 新增
):
```

实现要点:

- **winsorize**:每天对横截面值做 1%/99% 分位 clip(per-day,不是全局)。复用 `ml/preprocess.py:winsorize_panel`,行为与训练管道一致
- **健康检查**:每天计算 `nunique / n_valid`;低于阈值时该日 `daily_ic[t] = NaN`,不进 `mean_ic` / `abs_ic_mean` / `ic_ir`。同时记录 `degenerate_day_ratio` 进每个因子的输出字段
- 报告 HTML(`factors_analysis_report.py`)加一列 `退化日比例`,> 20% 标红警告

### 3.0.2 验收

跑一次 `python -m stockpool factors analyze` 后,要求:

- `alpha_096` 的 `abs_ic_mean` 从 0.4773 降到 ≤ 0.10
- `alpha_096` 的 `degenerate_day_ratio` ≥ 30%
- 健康因子(如 `ewma_vol_hl10`)`abs_ic_mean` 变化 ≤ ±0.02(winsorize 只 trim 1%/99%,不应大幅扰动健康因子)
- 全因子 `degenerate_day_ratio` 分布的 p95 < 10%(否则需调阈值)

### 3.0.3 旁通

Phase 0 完成后,`reports/factor_analysis/<新日期>.json` 是后续所有 IC 比较的**新基线**。**不复用** `2026-06-20.json` 的 IC 值,因为那是未 winsorize / 未健康检查的版本。

---

## 4. Phase 1:WQ101 窗口盘点(0.5 h)

### 4.1 输入

`src/stockpool/factors/wq101.py` 全部 101 个 alpha 类。

### 4.2 产出

`reports/wq101_window_inventory.csv`,列:

| 列 | 含义 |
|---|---|
| `alpha_id` | 如 `alpha_003` |
| `op` | `ts_sum` / `ts_rank` / `correlation` / `delay` / … |
| `window` | 当前数值 |
| `count_in_alpha` | 该窗口在 alpha 内出现次数 |
| `category` | `short`(≤10)/ `medium`(11-30)/ `long`(≥60)/ `other` |

### 4.3 方法

- 手段是 AST 扫描或正则 + 单元测试反查,任选;不强制工具
- 不调用任何因子 `.compute()`,纯静态扫描

### 4.4 验收

CSV 行数 ≥ 200(101 个 alpha 平均每个 ~2-3 个 window 常数,WQ101 多窗口比例高)。

---

## 5. Phase 2:变体生成器(2 h)

### 5.1 新模块

`src/stockpool/factors/wq101_variants.py`,**不动** `wq101.py` 原 alpha 类。

### 5.2 三种变换规则

| 名称 | 变换 | 假设 |
|---|---|---|
| `_compress` | 所有窗口 `N → ceil(N × 0.5)`,最小 2 | A 股价格发现快,短窗信号更新鲜 |
| `_rev_short` | 短窗(≤10)`N → ceil(N × 0.5)`,中长窗保持 | 散户超调反转,短期信号需更快 |
| `_expand_long` | 短窗保持,长窗(≥60)`N → ceil(N × 1.5)` | 基本面 re-pricing 在 A 股比美股慢 |

`ceil` 是为了避免出现窗口 = 0(`compress` 把 N=2 压到 1 是合法的,但小于 2 没意义)。

### 5.3 命名

`alpha_003_compress` / `alpha_003_rev_short` / `alpha_003_expand_long`。注册时同时给:

- `sources = ("wq101", "wq101_localized")` — 加新 source tag,方便 picker 筛选
- `types` 继承原 alpha

### 5.4 实现策略

工厂函数:

```python
def make_variant(original_cls, rule: str) -> type[Factor]:
    # 通过 monkey-patch 或子类化覆盖原 alpha 的 .compute(),
    # 在 ops.* 调用入口替换窗口参数
    ...
```

具体可以靠 `inspect.signature` + AST 重写,或者更简单:每个 alpha 类把窗口提到 class attribute,变体子类只覆盖 attribute。**Phase 1 的 inventory 决定**用哪种(如果窗口集中在少数共享常量,attribute 方案;如果分散在 compute 方法体里,AST 方案)。

### 5.5 Round 1 范围(top-30)

从 Phase 0 重新生成的 baseline IC 排名里,挑 **WQ101 内 |IC| 前 30** 注册三个变体 = 90 新因子。**`alpha_096` 不进 Round 1**(已废,会在 Phase 0 自动被诊断标黑)。

### 5.6 验收

- `factors list --source wq101_localized` 输出 90 行
- 任挑 5 个变体调 `.compute()` 不抛错、不出 `inf`、不全 NaN
- 单元测试:对一个已知 alpha(如 `alpha_003`),手工算 `compress` 变体的窗口数应等于工厂自动生成的值

---

## 6. Phase 3:评估(1 h compute)

### 6.1 输入

`reports/wq101_round1_factors.json`:30 baseline(top-30 wq101)+ 90 变体 = 120 个因子。

### 6.2 命令

```bash
python -m stockpool factors analyze \
  --factors-file reports/wq101_round1_factors.json \
  --output reports/factor_analysis/wq101_round1
```

(可能需要给 `factors analyze` CLI 加 `--factors-file` 参数,如果当前 CLI 还不支持。Phase 1 走 inventory 时顺便确认 CLI 接口。)

### 6.3 walk-forward 拆分

把 500 个交易日拆成前 250 / 后 250,**两半段各算一份 IC 报告**。本 phase 输出两份 JSON,Phase 4 用。

### 6.4 验收

- 两份 JSON 都有 120 个因子
- 两份的 `degenerate_day_ratio` 分布 p95 < 10%(说明 Phase 0 阈值合理)

---

## 7. Phase 4:Winner 选优(0.5 h)

### 7.1 脚本

`scripts/pick_wq101_winners.py`:

输入:Phase 3 的两份 JSON。

逻辑:对每个 baseline alpha,枚举其三个变体,只保留**同时**满足以下条件的变体:

- 前 250 日:`Δ abs_ic ≥ 0.02` 且 `Δ |ir| ≥ 0.1` vs baseline
- 后 250 日:`Δ abs_ic ≥ 0.02` 且 `Δ |ir| ≥ 0.1` vs baseline
- `degenerate_day_ratio ≤ 10%` 在两半段
- 同一 alpha 多个变体都过线时,选 `abs_ic` 在两半段的**最小值**最大的那个(保守口径)

输出:

- `reports/wq101_round1_winners.csv` — 列:baseline_alpha / chosen_variant / Δ abs_ic h1 / Δ abs_ic h2 / Δ ir h1 / Δ ir h2
- `reports/selection_wq101_localized.json` — `factors_file` 兼容格式,把现 selection 里的 top-30 WQ101 替换成 winners(对应 baseline 没 winner 时保留原版)

### 7.2 验收

- winners 数量 ≥ 6(即至少 20% 的 top-30 wq101 有改进)。否则假设证伪,本 spec 失败,Round 2 取消
- winners CSV 中 winners 在两半段都为正提升

---

## 8. Phase 5:AB 终验(1 h compute)

### 8.1 配置

`ab/wq101_localized.yaml`:

```yaml
base_config: config.yaml
use_ab_pool: true
arms:
  baseline:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json   # 当前
  localized:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection_wq101_localized.json
```

### 8.2 命令

```bash
python -m stockpool ab --config ab/wq101_localized.yaml
```

### 8.3 验收

- per-stock 中位 `Δ Sharpe ≥ +0.10` → 本 spec 成功,merge `selection_wq101_localized.json` 为新默认
- 中位 `Δ Sharpe ∈ [-0.05, +0.10)` → "neutral",归档但不合并,转入 Round 2 看是否能加分
- 中位 `Δ Sharpe < -0.05` → 失败,本 spec 终止,winners 全部回滚

---

## 9. Phase 6:Round 2(bottom-30,可选)

仅当 Round 1 合并后才启动。流程与 Round 1 一致,只把目标群换成 bottom-30 wq101(Phase 0 修后的新基线里 |IC| 最低的 30 个,排除已 winsorize/健康检查后仍 ≤ 0.02 的"真垃圾")。

**Round 2 验收阈值放宽**:per-stock 中位 `Δ Sharpe ≥ +0.05` 即可合并(因为这群因子本来就差,小改进也有意义)。

---

## 10. Phase 7:文档与提交(0.5 h)

- `CLAUDE.md` 因子库小节加一行:"WQ101 本土化变体(`wq101_localized` source)"
- `README.md` 不动(用户层无新命令)
- 本 spec 标 "Implemented" + 引用合并 PR 号
- 失败时(任何 Phase 验收 fail),把失败原因填到 spec 末尾"复盘"小节,不删除

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 变体跟原版数学高度相关 → 选优重复计数 | walk-forward 双半段验证 + winners 强制 `Δ abs_ic ≥ 0.02`(噪声水平外) |
| factor_panel cache sig 变化 → 全量重算 ~1 h | Round 1 只引 120 因子,sig 改一次,接受 |
| 过拟合 2024-2026 行情(熊牛切换) | 双半段(2024H2 vs 2025H2)各自独立验收 |
| AB 池 ~100 票太少,Δ Sharpe 不稳 | 复用 `ab_pool.parquet`,后续可扩到 Pool B(几百票);本 spec 不扩 |
| Phase 0 winsorize 改变所有因子的 IC → 跟历史报告不可直接比较 | Phase 0 输出附 "before/after" 表;后续 spec 默认引用新基线 |
| Phase 2 AST 改写复杂度爆炸 | 优先用 class-attribute 方案;若 inventory 显示窗口分散在表达式里,降级到只本土化"窗口集中"的 alpha(可能从 30 降到 20),其他保持原版 |

---

## 12. 成本估算

| Phase | 工作类型 | 估时 |
|---|---|---|
| 0 — 诊断升级 | 代码 | 2 h |
| 1 — 盘点 | 代码 + 静态分析 | 0.5 h |
| 2 — 变体生成器 | 代码 | 2 h |
| 3 — 评估 | 计算 | 1 h |
| 4 — 选优 | 代码 + 数据 | 0.5 h |
| 5 — AB | 计算 | 1 h |
| 6 — Round 2 | 重跑 1-5 | 4 h(条件触发) |
| 7 — 文档 | 文档 | 0.5 h |
| **Round 1 合计** | | **~8 h(其中计算 2 h)** |

---

## 13. 待跟进 / 未来工作

- horizon 本土化(独立 axis,本 spec 不涉及)
- 引入北上资金 / 龙虎榜 / 融资融券数据源(独立 spec)
- 细网格扫描(0.25 / 0.75 / 2.0 等),如果 Round 1 显示某 baseline 在两个粗变体之间都有提升
- 行业中性化粒度切换到申万二级(需先有二级 sector_map)
- WQ101 输出层的 winsorize 是否作为 factor-side 的 default(本 spec 只在 IC 诊断侧加;ML 训练管道已有)

---

## 14. 复盘(实施后填)

待 Phase 7 完成时填入。
