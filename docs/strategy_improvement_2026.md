# 策略改进方向 — 2026 Q2

> 目标读者:本项目的后续维护者(人或 Claude 会话)。本文聚焦如何把当前 `ml_factor` 策略从"已能跑通端到端 walk-forward 回测"推进到"可用的、可诊断的 alpha 因子驱动量化系统"。

---

## 1. 执行摘要

当前 `ml_factor` 策略骨架已具备(见 `src/stockpool/backtesting/strategies.py:MLFactorStrategy` 与 `src/stockpool/ml/pipeline.py`):

- 两步法 pipeline:Lasso 筛因子 → IC/IR/equal 加权
- pooled 全市场训练(`training_universe=all`, ~4350 只 A 股)
- 月度 share-pool fit 缓存,跨股复用
- T+1 次日开盘成交、multi_lot 引擎、固定 `position_size=0.1`
- 训练集 quantile 把连续分数映射到 `verdict ∈ {strong_sell, sell, hold, buy, strong_buy}`

**但**:8 个手挑技术因子在用,101 个 WQ alpha 闲置;Lasso 是线性的,Walk-forward CV 没有 embargo;`position_size` 是死值,缺乏组合层风险控制。本文给出 **F1 因子库扩展 / F2 模型升级 / F3 组合构建** 三个深度展开方向 + 三条其他方向的简述,以及对应路线图。

**贯穿性设计原则(本项目偏好)**:任何新增组件都做成 config 驱动的可选 type,默认指向新方案、但保留旧实现可切回。这条原则在 F1/F2/F3 的设计里都会体现。

---

## 2. 方向总览(catalog)

### F1. 因子库扩展(深度展开见 §3)
**痛点**:`MLFactorConfig.factors` 默认只有 6-8 个手挑通用因子(`momentum_20`、`macd_hist`、`rsi_centered_14` 等),而 `src/stockpool/factors/wq101.py` 实现了 101 个 WorldQuant alpha,基本闲置。代码侧的 `factors_file` 入口已经准备好(支持从 HTML picker 导出的 JSON 装载),所缺的是"哪些因子值得用"的客观决策依据。

### F2. 模型升级(深度展开见 §4)
**痛点**:
1. 当前 `selector` 只有 Lasso,`SelectorConfig.type: Literal["lasso"]` 是单元素;Lasso 假设线性,WQ101 中很多 alpha 的预测力来自非线性交互。
2. 训练—测试切分没有 embargo,`horizon=3` 意味着训练集末尾 3 天的 label 和测试集首样本有信息重叠。
3. IC weighter 假设各因子独立线性加和,在因子相关性强时会过度押注同一信息源。

### F3. 组合构建与风险控制(深度展开见 §5)
**痛点**:
1. `position_size=0.1` 是配置写死的常数,不随股票波动率调整;高 vol 股和低 vol 股开同样大小的仓。
2. 没有组合层风险控制——没有 max DD 熔断、没有 sector 集中度上限、`max_concurrent_lots: null`(只靠现金封顶)。
3. `quantile → verdict` 离散化丢弃了连续分数信息;一只票 0.91 分位和 0.99 分位都映射成 `strong_buy`,但后者应该拿更大仓。

### F4. 验证体系(简述)
**痛点**:回测主要输出净值曲线 + sharpe/max DD,缺乏:
- 走样式中每个 refit 窗口的 IC、命中率、贡献分解
- 不同 regime(牛/熊/震荡)下的策略表现切片
- 因子稳定性诊断(IC 时间序列、半衰期)
- 成本敏感性扫描(费率从 0 → 真实区间)

**建议**:新建 `analyze_*.py` 系列脚本输出 HTML 诊断报告。属于 F1/F2 完成后的自然延伸,落到独立 spec 处理。

### F5. 执行真实性(简述)
**痛点**:
- 涨停打不进单的过滤目前没做;
- 停牌日填充策略不清晰(缓存里 volume=0 的 bar 是否参与训练?);
- T+1 在 long-only 已隐含,但分红除权日的价格跳变没有特殊处理。

**建议**:小修补,落 followup ticket。回测口径影响小,实盘前必做。

### F6. composite_verdict 策略的去留(简述)
**痛点**:默认 `strategy.name` 已切到 `ml_factor`,但 `composite_verdict` 仍在 codebase 里(CompositeVerdictStrategy + 大量阈值配置 + 16 个硬编码 trigger)。

**建议**:**保留**作为 baseline。理由:(1) 在 ML 模型失效时是 fallback 选择;(2) 提供"规则系统 vs ML 系统"的对照基线;(3) 配置驱动设计原则——用户应能通过 `strategy.name: composite_verdict` 一键切回。如果维护成本上升再考虑提取为"baseline 子包"。

---

## 3. F1 — 因子库扩展

### 3.1 现状与痛点

| 维度 | 现状 | 痛点 |
|------|------|------|
| 已实现因子 | builtin technical 10 个 + WQ101 共 ~111 个 | 大多闲置 |
| 默认 config 因子数 | 6-8 个 | 利用率 <10% |
| 因子选择机制 | YAML `factors: [...]` 或 `factors_file` 指向 JSON | 已可配置,但缺乏**客观选因子的依据** |
| 因子诊断工具 | 无 | 用户/Claude 凭直觉挑因子 |
| 池化 cross-sec 因子 | 支持(`build_factor_panel` 注入 `factor_panel`) | 已通,但没人用 WQ101 验证过实际增益 |

核心矛盾:**因子库已建好,但选哪些因子用、按什么标准选,是空白**。直接把 111 个全塞给 Lasso 不是好主意——alpha=0.001 下 Lasso 留下太多相关因子,IC weighter 会过度押注同一信息源;且训练慢,缓存命中率低(`_strategy_signature` 对 factors 列表敏感)。

### 3.2 候选方案

#### 方案 A:静态全量 + 提高 Lasso alpha
- **做法**:`factors: <全部 111 个>`,把 `selector.alpha` 从 0.001 调到 0.01-0.05,让 Lasso 自己做更狠的稀疏化。
- **优点**:零新增代码。
- **缺点**:(1) Lasso 在相关特征间随机选一个,选择不稳定;(2) alpha 是超参,缺乏调优依据;(3) 每次 refit 还要算 111 个因子;(4) 不可解释——用户不知道为什么留下 alpha_017 没留 alpha_034。

#### 方案 B(推荐):IC 分析脚本 + selection.json + 多策略保留
- **做法**:
  1. 新建 `scripts/analyze_factors.py`(或 `src/stockpool/factors_analysis.py` 模块 + CLI 子命令 `python -m stockpool factors analyze`),在 pooled 全市场样本上,对每个注册因子算:
     - 滚动 IC(rank, 252d 窗口)、均值 IC、IC IR(IC 均值 / IC 标准差)
     - IC 半衰期(自相关 ACF 衰减到 0.5 的滞后数)
     - 因子间 IC 相关矩阵
     - 不同 regime(用 cfg.context.indices[0] 切牛/熊/震荡)下的 IC 切片
  2. 输出 `reports/factor_analysis/<YYYY-MM-DD>.html`(因子排名表 + IC 时序图 + 相关性 heatmap)+ `reports/factor_analysis/<YYYY-MM-DD>.json`
  3. 提供 `pick` 子命令:在分析结果上做"按 IC IR 排序 + 相关性贪心剔除"的自动选 top-N,输出 `selection.json`
  4. 用户可:
     - 用脚本输出的 selection.json 直接喂给 `factors_file`
     - 或者打开 HTML picker(已有功能)手动微调
- **优点**:(1) 客观可解释;(2) 一次分析,长期复用;(3) 与现有 HTML picker 串联,符合"配置驱动"原则;(4) 减少因子数 → 训练加速 → 缓存命中率高。
- **缺点**:需要新增 ~200 行分析代码 + HTML 模板。

#### 方案 C:动态因子选择(在线学习)
- **做法**:在 `MLFactorStrategy._refit` 中,先对池化样本算每个候选因子的 IC,丢掉 |IC| < 阈值的,再喂给 Lasso。
- **优点**:自适应 regime 变化。
- **缺点**:(1) 每次 refit 都要算所有候选因子的 IC,训练时间从 O(k) 变 O(k_pool);(2) 增加 walk-forward 中的"选因子也是模型一部分"的可重复性负担;(3) 与已有 selector 抽象重复(本质上是另一种 selector)。

#### 方案 D(配套):新增高质量自定义因子
独立于上面三个方案。提议补几个 WQ101 没覆盖、A 股专属的因子:
- `northbound_flow`(北向资金净买入 / 总成交)—— 需新数据源,放 followup
- `industry_relative_strength_20`(相对所在板块的 20 日超额收益)—— 项目已有板块 K 线
- `limit_up_count_20`(20 日涨停次数)—— OHLC 派生即可
- `turnover_zscore_60`(60 日换手率 z-score,反映异常活跃度)

### 3.3 推荐路径(方案 B + 方案 D 部分)

**配置层(`config.py:MLFactorConfig`)** — 无变化,沿用 `factors`/`factors_file` 二选一。

**新增代码**:
- `src/stockpool/factors_analysis.py` — 新模块,核心函数:
  ```python
  def analyze_factors(
      universe: list[str],
      factor_names: list[str],
      panel_data: dict[str, pd.DataFrame],
      horizon: int = 3,
      ic_window: int = 252,
      regime_index: str = "sh000001",
  ) -> FactorAnalysisResult: ...

  def pick_top_factors(
      result: FactorAnalysisResult,
      top_n: int = 20,
      max_correlation: float = 0.6,
      min_ir: float = 0.05,
  ) -> list[str]: ...
  ```
- `src/stockpool/factors_analysis_report.py` — pyecharts HTML 渲染(复用 `report.py` 的样式)
- `src/stockpool/factors/custom.py`(可选)— 方案 D 的几个新因子
- `cli.py` 加 `factors analyze` 和 `factors pick-by-ic` 子命令

**配置驱动原则的体现**:
- `pick_top_factors` 的参数(`top_n`、`max_correlation`、`min_ir`)全部走 CLI 参数,而非函数内常量;
- 分析参数(IC 窗口、horizon、regime 指数)同样;
- 用户始终可以**绕过自动选择**——直接在 `factors_file` 写人工选择的列表。

**测试覆盖**:
- `test_factors_analysis.py`:合成 5 个因子(1 个强 + 2 个相关 + 2 个噪声),验证 IC 排名、相关性剔除、regime 切片正确
- `test_cli_factors_analyze.py`:CLI 烟雾 + 输出文件存在性

### 3.4 验证标准

走完 F1 后,我们应该能回答:
1. **客观因子排名**:对 111 个注册因子,我们有滚动 IC、IR、半衰期、相关性数据,落地为 HTML/JSON 报告
2. **A/B 对比**:同一段历史(比如 2022-01 到 2025-12),`factors=[6 默认]` vs `factors_file=selection.json(top 20)` 的回测,分两档:
   - **最低 gate(必过)**:新因子集 sharpe ≥ 旧因子集 sharpe × 80%。允许小幅退化,代价换长期可扩展性。
   - **成功目标(期望达成)**:新因子集 sharpe ≥ 旧 × 1.1,或年化收益绝对值改善 ≥ 2pp。
3. **可复现**:同一 cache_dir 跑两次 `factors analyze` 输出 hash 一致

---

## 4. F2 — 模型升级

### 4.1 现状与痛点

`MLFactorStrategy` 的 walk-forward 训练流程:

```
for each refit_bar t:
    train_X, train_y = build_factor_matrix on bars [t - train_window, t - horizon)
    pipeline = TwoStepPipeline(LassoSelector, ICWeighter)
    pipeline.fit(train_X, train_y)
    quantiles = compute on pipeline.predict(train_X)
    # use until next refit_bar
```

| 维度 | 现状 | 痛点 |
|------|------|------|
| Selector | 只有 LassoSelector (`SelectorConfig.type: Literal["lasso"]`) | 线性 |
| Weighter | IC / IR / equal,均为标准化后线性加权 | 仍线性 |
| CV / 切分 | 训练集结束 = `t - horizon`,测试始于 t | **无 embargo gap**;`horizon=3` 时,t-2、t-1 的因子和 t 的 label 信息有重叠 |
| 标签 | `(close[t+h] - close[t]) / close[t]` | 绝对收益,未做 vol/cross-sec demean |
| 量纲处理 | 因子 z-score 标准化(每次 refit 重算) | 截尾/异常值未处理(WQ101 有些 alpha 长尾) |
| 超参选择 | 全部 YAML 写死(`alpha=0.001`、`n_chunks=6` 等) | 没有 hyperparameter tuning |

### 4.2 候选方案

#### 方案 A:不动 selector,加 embargo + 标签规范化
- **做法**:
  1. 在 `_refit` 里把训练集结束截到 `t - horizon - embargo_days`,默认 `embargo_days = horizon`
  2. 标签从绝对收益 → cross-sectional rank(pooled mode 下,每天对所有 host 标准化到 [-0.5, 0.5])或 vol-adjusted return
- **优点**:最小改动,泄露问题立竿见影
- **缺点**:不解决线性瓶颈

#### 方案 B(推荐):方案 A(embargo + 标签规范化)全做 + 新增 LightGBM selector / weighter,保留 Lasso/IC 全栈
- **做法**:**包含方案 A 的全部内容**,在其之上加:
  1. **Selector 扩展**:
     - 新增 `LightGBMSelector` —— 拟合 LGB,按 `feature_importance(gain)` 排序,留 top-K(K 可配)或重要性 > 阈值的因子。
     - `SelectorConfig.type: Literal["lasso", "lightgbm"]`,默认 **"lightgbm"**(但用户可切 "lasso")。
     - 新增 `LightGBMSelectorConfig`(num_leaves、min_data_in_leaf、learning_rate、num_iterations、top_k_factors、importance_threshold)。
  2. **Weighter 扩展**:
     - 新增 `LightGBMWeighter` —— 用 LGB 直接预测,`predict` 返回 LGB 输出而非 z-score 加权和。
     - `WeighterConfig.type: Literal["ic", "ir", "equal", "lightgbm"]`,默认 **"lightgbm"**。
     - 注意:LightGBM 不需要 standardisation,但要在 `_apply_standardiser` 路径里跳过标准化(条件分支)。
  3. **依赖**:`pyproject.toml` 加 `lightgbm` 到 `dependencies`(或 `optional-dependencies: ml`);未装 lightgbm 时,选 `type: "lightgbm"` 报清晰的 ImportError。
- **优点**:
  - 兼容现有 pipeline 抽象(selector → weighter)
  - LGB 处理非线性 + 交互项 + 缺失值原生
  - 老用户可一行 config 切回 lasso+ic 做对照
- **缺点**:新增依赖;LGB 训练比 Lasso 慢 3-10×(单次 refit 几秒到几十秒,可接受)

#### 方案 C:端到端 LightGBM(放弃两步法骨架)
- **做法**:整体替换 `TwoStepPipeline`,直接 LGB 拟合 X→y。
- **优点**:更简洁,无 selector→weighter 的人为切分
- **缺点**:丢失 IC 诊断(F4 验证体系会受影响);打破现有 `FitInfo` / `contributions()` API;不符合配置驱动原则——切回 lasso 需要切 strategy 而非切组件

### 4.3 推荐路径(方案 B,拆成 PR-A embargo + PR-B lightgbm 两步落地)

**配置层(`config.py`)修改**:

```python
class LassoConfig(BaseModel):
    alpha: float = Field(default=0.001, ge=0.0)
    max_iter: int = 1000
    tol: float = 1e-6

class LightGBMSelectorConfig(BaseModel):
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    learning_rate: float = 0.05
    num_iterations: int = 200
    top_k_factors: int | None = None         # None = 用 importance_threshold
    importance_threshold: float = 0.0        # 累计 importance > x 之后截断
    random_state: int = 42

class SelectorConfig(BaseModel):
    type: Literal["lasso", "lightgbm"] = "lightgbm"          # ← 默认改 lightgbm
    lasso: LassoConfig = Field(default_factory=LassoConfig)
    lightgbm: LightGBMSelectorConfig = Field(default_factory=LightGBMSelectorConfig)

class LightGBMWeighterConfig(BaseModel):
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    learning_rate: float = 0.05
    num_iterations: int = 300
    random_state: int = 42

class WeighterConfig(BaseModel):
    type: Literal["ic", "ir", "equal", "lightgbm"] = "lightgbm"   # ← 默认改 lightgbm
    use_rank: bool = True
    min_abs_ic: float = 0.0
    n_chunks: int = 6
    min_abs_ir: float = 0.0
    lightgbm: LightGBMWeighterConfig = Field(default_factory=LightGBMWeighterConfig)
```

**注意配置兼容性**:旧 YAML 的 `selector: {type: lasso, alpha: 0.001}` 需要继续工作。两种实现路径:
1. **嵌套优先 + 向后兼容字段**:保留顶层 `alpha`、`max_iter`、`tol` 字段(deprecated 但不报错),Pydantic `model_validator` 把它们迁到 `selector.lasso.*`;新写法用 `selector.lasso: {alpha: ...}` 子段。
2. **fail-loud**:旧字段直接拒绝,要求用户改 YAML。

**推荐 (1)**——三个月内打 deprecation warning,然后切 (2)。在 spec 写入时把这条迁移路径明确写出来。

**ml 模块修改**:
- `selectors.py` 新增 `LightGBMSelector`,实现 `FactorSelector` 抽象
- `weighters.py` 新增 `LightGBMWeighter`,实现 `FactorWeighter` 抽象(`predict` 走 lgb model,`weights()` 返回 feature_importance 归一化后的 Series 做兼容)
- `strategies.py:_build_weighter` 加 `if cfg.type == "lightgbm"` 分支
- `strategies.py` 加 `_build_selector(cfg) -> FactorSelector` 函数(目前 `LassoSelector` 是直接 import + 实例化,集中到工厂)
- `TwoStepPipeline.contributions()` 在 LGB weighter 下返回 SHAP 值(或 `lgb_model.predict(pred_contrib=True)`)

**embargo(PR-A 独立交付)**:
- `MLFactorConfig` 加 `embargo_days: int | None = None`(None = 自动用 `horizon`)
- `_refit` 中 `train_end = bar_t - horizon - embargo`(当前是 `bar_t - horizon`)
- 同步把 `selector`/`weighter` 的字段子段化(`selector.lasso.*`)但默认行为不变——为 PR-B 切到 lightgbm 让路
- 标签规范化(方案 A 第 2 条)也归入此 PR:加 `MLFactorConfig.label_type: Literal["return", "vol_adjusted", "cross_sec_rank"] = "return"`,默认 "return" 保持现行行为

**测试覆盖**:
- `test_ml_selector_lightgbm.py`:合成数据(线性 + 非线性 + 交互项),验证 LGB selector 在非线性 case 下 selected_factors 优于 lasso
- `test_ml_weighter_lightgbm.py`:LGB weighter 的 fit/predict、weights() 兼容、`contributions()` 行为
- `test_ml_pipeline.py` 现有用例需补 `LightGBMSelector + LightGBMWeighter` 组合
- `test_ml_strategy.py` 加 walk-forward 不泄露(embargo 生效)的契约测试
- `test_config.py` 加 selector/weighter 双模式 + 老 YAML 兼容性测试

**`_strategy_signature` 影响**:加 `embargo_days`、`selector.lightgbm.*`、`weighter.lightgbm.*` 后哈希会变,旧 `ml_models/*.pkl` 自动失效——符合预期,首次跑会全量重训。

### 4.4 验证标准

1. **泄露消除**:`test_ml_strategy.py` 中加 `test_embargo_blocks_label_leak`——构造一个 horizon 范围内 forward returns 强相关的合成数据集,验证不加 embargo 时 train IC 远高于 test IC,加 embargo 后差距缩小
2. **非线性增益**:相同因子集 + 相同 walk-forward 窗口,lightgbm 路径 vs lasso+ic 路径,在 OOS IC 上至少不差
3. **可切换性**:用 `selector.type: lasso, weighter.type: ic` 跑回测,行为与上一版完全一致(回归测试)
4. **依赖隔离**:不装 lightgbm 时,默认 config(lightgbm)启动报清晰错误,且切到 lasso 仍可跑

---

## 5. F3 — 组合构建与风险控制

### 5.1 现状与痛点

```yaml
# 当前 config.yaml 中相关字段
backtest:
  engine: multi_lot
  position_size: 0.1
  max_concurrent_lots: null
strategy:
  ml_factor:
    thresholds:
      strong_buy: 0.90
      buy: 0.70
      sell: 0.30
      strong_sell: 0.10
    buy_verdicts: ["buy", "strong_buy"]
    sell_verdicts: ["sell", "strong_sell"]
    refresh_verdicts: ["strong_buy"]
```

| 维度 | 现状 | 痛点 |
|------|------|------|
| 单笔仓位大小 | 固定 `position_size: 0.1`(10% 现金) | 不随波动率调整 |
| 最大并发 lots | `null`(只靠现金封顶) | 极端行情可能全部满仓 → 集中度风险 |
| 组合层风控 | 无 | 没有 max DD 熔断 |
| 板块/集中度 | 无 | 8 只票里 2 只化工 + 2 只通用机械,可能同向亏 |
| 信号离散化 | quantile → 5 档 verdict,死阈值 | 0.91 分位和 0.99 分位同 `strong_buy` |
| 信号噪声平滑 | 无 | 一次 refit 后,分数小幅波动可能引起翻仓 |

### 5.2 候选方案

#### 方案 A:不动引擎,只加可选 sizing & 风控 overlay
- **做法**:
  1. `BacktestConfig` 加 `sizing` 子段(`type: fixed | vol_target`,默认 `vol_target`);
  2. `BacktestConfig` 加 `risk_overlay` 子段(max_dd_stop、sector_cap、min_holding_after_stop)
  3. 引擎在每个 buy 前查 overlay,被拒就不开仓
- **优点**:增量式;`engine: multi_lot` 不变;可对单一方向独立验证
- **缺点**:风险 overlay 在 long-only 单仓基础上做,组合优化没真正引入

#### 方案 B(推荐):方案 A 全做 + 信号平滑
- **做法**:方案 A 之外加:
  1. **Verdict EMA 平滑**:对 ml_factor 预测的连续分数做 EMA(span 可配),映射 verdict 时用平滑后值。可减少 churn
  2. **连续仓位映射(可选 type)**:`MLFactorConfig` 加 `position_mapping: Literal["verdict", "continuous"]`,`continuous` 时 `entry_size = sigmoid((score - score_buy_threshold) / temperature)`,这条作为可选高级路径,默认仍 `verdict`(配置驱动原则)
- **优点**:把连续信号的信息完整用上;减少抖动
- **缺点**:多了 2 个超参,需要回测调

#### 方案 C:换引擎到 portfolio rebalance(top-K)
- **做法**:每 K 天对池中所有票按预测分数排序,持有 top-K(K 比如 5)等权,clip 到现金;不再有 multi_lot 一笔笔进出。
- **优点**:符合主流学术口径;cross-sectional 一致性更好
- **缺点**:`backtesting/framework.py` 单仓位 ABC 不支持,要扩 ABC 或新建 `PortfolioStrategy`;改动面大;失去 N 天平仓和 refresh_timer 这套已经验证的逻辑

### 5.3 推荐路径(方案 B,方案 C 列为 followup)

**配置层(`config.py`)修改**:

```python
class FixedSizingConfig(BaseModel):
    size: float = Field(default=0.1, gt=0.0, le=1.0)

class VolTargetSizingConfig(BaseModel):
    target_vol_annual: float = Field(default=0.20, gt=0.0)   # 目标 20% 年化波动
    vol_window: int = 20                                     # 计算最近股票 vol 用的天数
    min_size: float = Field(default=0.03, gt=0.0)
    max_size: float = Field(default=0.20, gt=0.0)
    fallback_to: Literal["fixed", "skip"] = "fixed"          # vol 算不出时

class SizingConfig(BaseModel):
    type: Literal["fixed", "vol_target"] = "vol_target"      # ← 默认 vol_target
    fixed: FixedSizingConfig = Field(default_factory=FixedSizingConfig)
    vol_target: VolTargetSizingConfig = Field(default_factory=VolTargetSizingConfig)

class RiskOverlayConfig(BaseModel):
    enabled: bool = True
    max_drawdown: float = Field(default=0.15, gt=0.0, lt=1.0)    # 组合 DD 超 15% 熔断
    drawdown_cooldown_bars: int = 5                              # 熔断后 N bar 不开新仓
    sector_concentration_cap: float | None = Field(             # 单 sector 持仓比上限
        default=0.40, gt=0.0, le=1.0,
    )

class BacktestConfig(BaseModel):
    # ... existing fields ...
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    risk_overlay: RiskOverlayConfig = Field(default_factory=RiskOverlayConfig)
    # position_size 字段保留为 deprecated alias of sizing.fixed.size,
    # 1 个版本周期后移除;model_validator 自动迁移
    position_size: float | None = None
```

**引擎层(`backtesting/framework.py`)修改**:

`MultiLotBacktestEngine` 在 buy 触发点加两个钩子调用:
1. `_compute_lot_size(bar_idx, signal_row, recent_returns) -> float` — 走 sizing 配置
2. `_risk_overlay_allows_buy(equity_curve_so_far, current_positions) -> bool` — 走 risk_overlay 配置

两者都是引擎内部方法,在 `__init__` 时根据配置 wire 好实现(`fixed` → 返回常量,`vol_target` → 算 vol;`risk_overlay.enabled=False` → 永远 True)。这样老的单仓位测试和单一 fixed sizing 测试零回归。

**关于 sector 数据**:`risk_overlay.sector_concentration_cap` 需要 `cfg.stocks[].sector` 字段(已有)。在 multi-lot 引擎里维护一个 `{sector → current_exposure}` 累计计数,buy 前查上限。**注意**:engine 当前是单股 simulation(`engine.simulate(daily_df, strategy)`),`{sector}` 维度的组合视图需要在更上层(`strategy_factory.simulate` 或 `backtest_composite.simulate_equity_curve` 的 multi-stock 聚合)落地。这条要决策:
- (a) 简化版:每个 simulate 是单股的,sector cap 退化成"单股最大仓位 = cap"(其实就是 `max_size`)
- (b) 真正版:在聚合层做 sector 累计,需要让 engine 接受外部状态注入

**推荐**:F3 第一版做 (a),把 (b) 列为 F3.5 followup。这样 F3 可以纯在引擎内闭环。

**信号平滑(MLFactorStrategy)**:
```python
class MLFactorConfig(BaseModel):
    # ... existing fields ...
    score_smoothing: Literal["none", "ema"] = "none"   # ← 默认 none(保守)
    score_smoothing_span: int = 5                      # EMA span
```
**注意**:默认 `none` 而不是 `ema`,因为信号平滑的最优 span 因股而异,且会改变信号语义——这是少数我违反"新方案做 default"原则的地方,**理由是**:平滑会延后信号 ~span/2 bar,对趋势策略可能扩大滑点,先让用户主动启用、观察效果。在文档里把这个例外明确标出。

**连续仓位映射**(`position_mapping: continuous`)列为 F3 可选高级路径,**不在第一版实现**——先把 sizing + risk_overlay 跑通验证。

**测试覆盖**:
- `test_sizing_vol_target.py`:合成两只票(高/低 vol),验证 lot size 反比于 recent vol;clip 边界
- `test_risk_overlay_max_dd.py`:构造一段单调下跌行情,验证熔断后 cooldown_bars 内不开新仓
- `test_engine_overlay_compat.py`:`risk_overlay.enabled=False` + `sizing.type=fixed` 时,行为与上版完全一致(回归)
- `test_config_sizing_migration.py`:`backtest.position_size: 0.1`(老 YAML)→ 等价于新 `sizing.fixed.size: 0.1`

### 5.4 验证标准

1. **vol-target 有效**:同一段历史,fixed sizing 的 max DD vs vol_target sizing 的 max DD —— 后者应当显著小(>20% 改善)且夏普至少持平
2. **熔断真熔断**:max_dd_stop=0.10 跑一段 2015-Q2 或 2022-Q1 这种 DD 容易超 10% 的窗口,确认 trades 数量在 cooldown 期内归零
3. **零回归**:`risk_overlay.enabled=false, sizing.type=fixed, sizing.fixed.size=0.1` 跑完整测试套件,所有 `test_multi_lot_engine.py` / `test_backtest_composite.py` 通过
4. **配置可逆**:用户改一行 `sizing.type: fixed` 就能完全回到旧行为,无需改代码

---

## 6. 路线图(2026-05-24 更新)

> 本节是**唯一**反映项目当前状态的清单。任何 follow-up / 暂搁项都应该汇总在这里,而不是散落在各个 spec 的"Non-Goals"/"范围外"段。
>
> A/B 验证全部结果见 `docs/ab_validation_results.md`;运行手册见 `docs/ab_validation_runbook.md`。

### ✅ 已完成(经 A/B 验证)

| 项目 | 关键 commit / spec | A/B verdict |
|------|--------------------|-------------|
| **F1 plan-1** — factor analysis 模块(`factors_analysis.py` + CLI `factors analyze` / `pick-by-ic` + pyecharts 报告) | spec `2026-05-23-factor-analysis-module.md` | — (无 A/B,工具性 PR) |
| **F2 PR-A** — embargo + label_type 接口 + Lasso 子段化 | spec `2026-05-23-f2-pr-a-...`;`embargo_days` 默认 None=auto=horizon | **⚠️ tied (P2-1)** — Δsharpe=-0.034,无害,**默认保留** |
| **F2 PR-B1** — LightGBMSelector + `selector.lightgbm.*` 子段 | spec `2026-05-23-f2-pr-b1-...` | **⚠️ tied (P1-1)** — LGB selector 单独 Δsharpe=-0.027 |
| **F2 PR-B2** — LightGBMWeighter + WeighterConfig 子段化 + `contributions()` 多态 | spec `2026-05-23-f2-pr-b2-...` | **❌ regression (P1-2)** — Δreturn=-12.72%,LGB weighter 过拟合主因 |
| **F2 默认回退**(commit `28a1584`,2026-05-24) | `selector.type` `lightgbm`→`lasso`,`weighter.type` `lightgbm`→`ic` | 由 P0-2 触发 |
| **T1 A/B testing tool** — `stockpool ab` 子命令 + `ab/{config,runner,report}.py` + `backtest_runner.py` | spec `2026-05-24-ab-testing-design.md` | — (工具性 PR) |
| **T1 fix** — `_decide_pool_sharing` 按 arm 需要 gate universe 注入(commit `3648d50`) | — | 由 P3-2 第一次跑出 Δ=0 触发 |
| **P3-2 验证 + verdict** — `training_universe=all` 在 16 股上 Δsharpe=-0.243 / Δreturn=-14.32%,**保持 `pool` 默认** | `ab_validation_results.md` §3.7 | **❌ regression** |
| **F3 PR-C** — Sizing 子段化 + vol-target 仓位 | spec `2026-05-24-f3-pr-c-...` | **✅ pass gate** — Δsharpe=-0.007, Δmax_dd_ratio=-27.8%, Δreturn=-4.21pp. DD 收窄 27.8% 超过 20% 门槛,Sharpe 几乎持平(-0.007),**vol_target 为默认** |

**当前 sweet spot 默认**(8 个 A/B 对照之上):

```yaml
strategy:
  name: ml_factor
  ml_factor:
    panel_mode: pooled          # ✅ P3-1 真收益 (+0.23 sharpe)
    training_universe: pool     # ❌ "all" 倒退 (-0.24 sharpe)
    embargo_days: null          # ✅ auto=horizon,P2-1 tied 无害
    selector: {type: lasso}     # ❌ lightgbm 倒退 (P0-2/P1-2)
    weighter: {type: ic}        # ❌ lightgbm 倒退 (P0-2/P1-2)
    share_pool_fit: true        # 与 pooled+pool 配合工作良好
backtest:
  sizing: {type: vol_target}    # ✅ vol-target won validation (DD -27.8%, sharpe flat)
```

### 🚧 待做(按建议优先级)

#### P1: F3 — 组合构建与风险控制(原 §5)

跨度比 F2 大,建议拆 3 个子 PR:

| 子 PR | 交付物 | 估算 | A/B 验证 |
|------|--------|------|---------|
| **PR-D** — Risk overlay | `BacktestConfig.risk_overlay`(max DD 熔断 + cooldown + sector cap 单股版本)| 6-8 task | A/B:overlay on/off,看 max DD 与 trade count 变化 |
| **PR-E** — Score smoothing | `MLFactorConfig.score_smoothing: none | ema`,默认 `none`(显式违反"新方案做 default"原则,因为延迟成本不确定) | 4-6 task | A/B:span 不同值,看 trade churn |

依赖:不依赖任何待做项。可以独立启动。

#### P2: F1 plan-2 — Custom factors + WQ101 ranking + 实跑 A/B

F1 plan-1 末尾 deferred 的部分:

- `factors/custom.py`(`industry_relative_strength_20`、`limit_up_count_20`、`turnover_zscore_60`)
- 用 `factors analyze --universe all` 跑一次 WQ101 全量排名,选 top-20 写 `selection.json`
- A/B 对照"老 6 因子" vs "新 selection.json",决定 ml_factor 默认 factors 列表是否扩

**前置**:`fetch-universe` 已跑过(✅ 满足)

**估算**:3-4 task(`custom.py` 因子实现 + 跑 ranking + 跑 A/B + 决定默认更新)

#### P3: 工具 / 基础设施改进

| 项目 | 描述 | 来自哪里 |
|------|------|----------|
| **A/B portfolio-level** | 当前 A/B 是 per-stock 各跑各 + 跨股聚合;portfolio-level 要求"每 strategy 一条组合净值",需要新 `PortfolioStrategy` ABC + portfolio engine | A/B spec §Non-Goals |
| **A/B 统计显著性** | paired t-test / Wilcoxon / bootstrap CI。8-30 股样本太小,Pool B 联动扩到几百只再启 | A/B spec §Non-Goals |
| **`composite_verdict` 参数子段化** | 把 `indicators` / `weights` / `verdicts` / `scoring` 从 `AppConfig` 顶层下沉到 `strategy.composite_verdict.*`。完成后 A/B 自动获得"对比两个 composite_verdict 配置"能力(目前 ArmOverride 拒绝顶层字段覆盖) | A/B spec §Non-Goals |
| **A/B multi-arm (>2)** | 当前 schema 强制 `exactly 2 arms`。多 arm 需要重新设计 diff table / scatter / histogram | A/B spec §Non-Goals |
| **LGB holdout + early stopping** | 当前 LGB walk-forward 无 holdout,依赖"refit 频率"做正则。引入 holdout 可能让 LGB 真正可用 | F2 PR-B1/B2 spec §范围外 |
| **LGB selector/weighter 共享 booster** | 当前同时 LGB+LGB 时训 2 次。可共享 importance 通道避免重复训练 | F2 PR-B2 spec §3.12 |

#### P4: F2 LGB 重启(待 P2/P3 之后)

P0-2 判定 LGB 倒退,但**有条件可以重启**:

- (a) 调严超参:`num_leaves=7-10`,`min_data_in_leaf=50+`,`num_iterations=100`
- (b) 扩训练样本:`training_universe=all` 配合 LGB(单独切 universe 也倒退 -0.24,但配合调严的 LGB 可能展现非线性优势,需另跑 A/B 验证)
- (c) 扩 cfg.stocks 池本身(16 → 50+),让 selection bias 不那么主导
- (d) 引入 LGB holdout + early stopping(见 P3)

启动 P4 的前提:P2 (F1 plan-2) 跑通 + 上述至少两条同时启用 + A/B 验证 sharpe ≥ baseline。**否则不要再尝试默认切 LGB**。

### 🧊 暂搁(已验证倒退,需明确触发条件才重启)

| 项目 | 触发条件 | 验证证据 |
|------|---------|---------|
| LGB+LGB 作为默认 | 见 P4 | P0-2 Δsharpe=-0.20,P1-2 Δreturn=-12.72% |
| `training_universe=all` 作为默认 | (a) 配合 P2 扩股池 + 调参 LGB 后单独 A/B 验证 OR (b) 用户明确说"我的股池有强 selection bias" | P3-2 重跑 Δsharpe=-0.243 |

### 跨阶段约束(2026-05-24 新增)

**所有未来 PR 必须通过的 gate**:

1. **任何改默认值的 PR** 必须随附一份用 `stockpool ab` 跑的 A/B 报告作为证据,verdict 至少不是 ❌ regression
2. **任何新增组件**(新 selector / weighter / engine / sizing / overlay)必须在 spec 里写明"如何 A/B 验证它带来的收益"
3. **A/B 工具自身的 follow-up**(见 P3 表格)优先级取决于实际堵塞:
   - 真有人想跑 composite_verdict A/B → 解锁 composite_verdict 子段化
   - 真有人想看显著性 → 加 t-test
   - 真有人想跑 5 个 arm → 改 schema
   - **没需求就不动**(避免过度工程)
4. **`docs/improvement_ideas.md` 已被本路线图取代**;那个文件保留作为历史背景,不再更新

### 跨阶段维持事项

- 每个 PR 走 `docs/superpowers/specs/` + `docs/superpowers/plans/` 流程(per 项目惯例)
- 不引入新依赖时优先(F3、F1 plan-2 都不需要新依赖)
- 每个 PR 末附一份 A/B 对照报告(同一段历史,旧 config vs 新 config 的 7 个指标 + 净值散点)

---

## 7. 附录

### 7.1 `factor_analysis` 脚本草图

```python
# scripts/analyze_factors.py
import argparse
from stockpool.config import load_config
from stockpool.fetcher import load_universe_cache
from stockpool.panel import build_panel_from_cache
from stockpool.factors.registry import list_specs
from stockpool.factors_analysis import analyze_factors, pick_top_factors
from stockpool.factors_analysis_report import render_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--universe", choices=["pool", "all"], default="all")
    ap.add_argument("--factors", nargs="*", default=None,
                    help="缺省 = 全部已注册因子")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--ic-window", type=int, default=252)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--max-corr", type=float, default=0.6)
    ap.add_argument("--min-ir", type=float, default=0.05)
    ap.add_argument("--output", default="reports/factor_analysis")
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = build_panel_from_cache(cfg.data.cache_dir, args.universe, cfg)
    factor_names = args.factors or [s.name for s in list_specs()]

    result = analyze_factors(
        universe=list(panel["close"].columns),
        factor_names=factor_names,
        panel_data=panel,
        horizon=args.horizon,
        ic_window=args.ic_window,
    )

    selection = pick_top_factors(
        result, top_n=args.top_n,
        max_correlation=args.max_corr, min_ir=args.min_ir,
    )

    render_report(result, selection, output_dir=args.output)
    print(f"Wrote {args.output}/<date>.html")
    print(f"Top-{args.top_n} factors: {selection}")

if __name__ == "__main__":
    main()
```

### 7.2 A/B 对照实验设计模板

每个 PR 末附的"前后对比":

| 指标 | 旧 config | 新 config | 差值 | 显著? |
|------|-----------|-----------|------|-------|
| 总收益 (5 年) | | | | |
| 年化收益 | | | | |
| 年化波动 | | | | |
| Sharpe | | | | |
| Max DD | | | | |
| 平均持仓天数 | | | | |
| 总交易笔数 | | | | |
| 平均单笔收益(扣费后) | | | | |

加一张 `reports/comparison/<PR_name>.html` 的双线净值图(同一时间轴,旧/新)。

### 7.3 已知 unknowns / 不打算碰的范围

文档不覆盖的方向(写在这里避免后续讨论时撞):

- **做空/期货/期权** —— 项目定位 long-only A 股
- **盘中/tick 级数据** —— 缓存粒度是日线
- **大规模超参搜索/AutoML** —— 当前规模(8-50 只股 × 100 因子)不需要 Bayesian opt;若未来扩到 500+ 股池再说
- **特征工程自动化(autoencoder、PCA 因子)** —— 与 WQ101 手工因子的可解释性方向相悖,暂不考虑
- **强化学习交易** —— 数据量、稳定性都不够;不在路线图内
- **多策略组合(meta-strategy)** —— 先把单策略做扎实,再谈组合

---

**文档维护**:
- F1/F2/F3 各自的 detailed spec 完成后,把对应章节链接放到本文 §6 路线图对应行
- 路线图发生变化(顺序、范围、放弃某方向)时,在本文末加 changelog 段
