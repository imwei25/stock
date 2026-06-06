# 因子预处理与正交化:产业界做法 + 与本项目对比

**日期**: 2026-06-06
**范围**: 量化因子在进入选择器/合成器之前的"标准化流水线",以及"去相关"和"因子选择"两个动作如何衔接
**起点问题**: 本项目当前怎么处理相关因子?业界更通用的做法是什么?差距在哪?

---

## 0. TL;DR

| 维度 | 业界共识做法 | 本项目现状 | 差距 |
|---|---|---|---|
| 缺失填充 | 截面中位数 / 行业中位数 | 训练阶段 `dropna`,predict 阶段 fill 0(整列) | 中等 — predict 路径凑合可用,训练损失样本 |
| 去极值 | 每日截面 winsorize 1%/99% 或 ±3·MAD | **完全没有** | 大 — 异常股(资产重组、连续涨停)会拖坏权重估计 |
| 标准化 | 每日截面 z-score | 训练样本**整体**(全 T·N 行)z-score | 大 — 等于把跨时间漂移当成因子信号 |
| 中性化(行业/市值) | OLS 残差化(默认开启) | `ops.indneutralize` 可用,**默认未启用**,需手动注入 sector_map;无市值中性 | 大 |
| 多重共线性 | (a) 预处理阶段正交化 (b) 选择阶段去相关 (c) 模型自带正则,**三选一或叠加** | (b) `pick_top_factors` 贪心去相关 + (c) Lasso L1 / LGB gain;**无 (a)** | 中等 — 只用 (b)+(c),正交化路径完全缺失 |
| 因子选择 vs 预处理 | 业界主流:**预处理彻底解掉相关性后,选择只评单因子效力** | 选择器接到的是**未中性化、未截面标准化**的原始因子 | 整条流水线相位错位 |

核心结论:**项目当前的"去相关"完全压在 `pick_top_factors` + `LassoSelector` 这两个阶段后置处理上,前置的截面预处理流水线几乎是空白**。这是回测 sharpe 难以稳定提升的潜在天花板之一。

---

## 1. 业界标准预处理流水线(5 步)

主流量化机构(WorldQuant / AQR / Two Sigma / 国内中泰、天风、广发等卖方研报)的因子流水线高度趋同,顺序是:

```
原始因子 (T × N)
   │
   ├─① 缺失填充 (NaN handling)
   │       └─ 按截面中位数 / 行业中位数 / 整体均值
   │
   ├─② 去极值 (winsorize / clip)
   │       └─ 每日截面 ±3σ、±5×MAD,或分位 1%/99%
   │
   ├─③ 标准化 (cross-sectional standardize)
   │       └─ 每日截面 z-score (x - μ_t) / σ_t
   │
   ├─④ 中性化 (neutralize)
   │       └─ OLS 残差:f_neut = f - β·industry_dummies - γ·log(market_cap)
   │
   └─⑤ (可选) 正交化 (orthogonalize)
           ├─ Schmidt 顺序正交(给定优先级)
           ├─ Symmetric 对称正交(无序、保持原信息最大)
           └─ PCA 主成分(降维)
   │
   ▼
合成 / 选择阶段(IC weighting / Lasso / 树模型)
```

每一步的目的、做法和踩坑点:

### 1.1 缺失填充

**为什么**: 因子产生 NaN 的原因很多 — 新股 warmup、停牌断带、财务数据未披露、计算溢出。直接 dropna 会损失大量训练样本(尤其 cross-sectional 因子)。

**做法**:
- **首选**: 截面同期中位数填充。逻辑:NaN 通常代表"未知",用同截面群体的中位水平作 prior 最保守。
- **次选**: 行业内中位数填充。当因子有显著行业差异(如基本面 ROE)时更合理。
- **最差**: 直接填 0。会在 z-score 后代表"中性",但若 z-score 在填充**之后**做,等于把这个股拉到截面均值;若在**之前**做,等于异常向均值靠拢。本项目 predict 路径就是后者。

**项目现状**:
- 训练:`stack_panel_to_xy(dropna=True)` 直接丢含 NaN 的行 — 长 rolling 因子(`alpha_037`、200 日 corr 等)warmup 期会损失 30%+ 样本
- predict:`MLFactorStrategy._build_x_full` 对 NaN 列 fill 0(原 commit 注释 "归一化后 0 是中性值"),仅在**整行 NaN** 时返回 neutral

**潜在改进**: 训练前做截面中位数填充,可显著扩大 ML 训练样本。

### 1.2 去极值 (Winsorize)

**为什么**: 单只票当日 30% 涨停 + 转融通卖空异动会让 `momentum_5` 因子值飘出 10·σ。OLS / Lasso / IC 计算对这种 outlier 极其敏感,一个样本能把单日 IC 从 +0.05 翻成 -0.10。

**做法**:
- **分位截断**: 每日截面取 1%/99% 分位,超出的 clip 回去。最常用,稳健。
- **MAD 截断**: `clip(x, μ - k·MAD, μ + k·MAD)`,k=5。对极厚尾分布更鲁棒。
- **±3σ**: 假设正态,工业界已较少用(因子分布通常厚尾)。

**关键细节**:
- 必须**每日截面**做,不能 pooled。pooled winsorize 会把整段牛市的高动量股全部 clip 掉。
- 在 z-score **之前**做,否则 outlier 已经污染了 μ 和 σ。

**项目现状**: 完全没有。只在某些因子内部局部 `clip` 防 inf。

### 1.3 标准化 (Cross-Sectional Z-Score)

**为什么**: 不同因子量纲完全不同(momentum 在 ±0.3,volume_zscore 在 ±5,turnover 在 0~0.5)。IC 加权前必须统一量纲,否则 weights 没法相加。

**做法**:
- `(x_t - μ_t) / σ_t`,**每日**重算 μ 和 σ。
- 数学上 IC = corr(rank_x, rank_y) 本身是 rank-based,不依赖标准化;但**线性合成**(`weighted sum`)必须标准化。

**关键细节**:
- 业界绝大多数都做**截面**(cross-sectional)z-score,而非时间序列 z-score。原因:截面才反映"今天这只票相对同行的位置",而时间序列 z-score 会把"市场整体波动率上升"算成因子信号。
- 项目当前做的是**整个训练样本(T·N 行)一次性 z-score**。等价于 pooled standardize,既不是截面也不是时间序列,数学上偏向时间序列 z-score 但又混进了截面均值。这一步是隐藏的设计 bug。

**项目现状**: `_StandardisingMixin._fit_standardiser` 对全部训练样本调 `standardize_fit` → 单一 (μ, σ) 应用到所有 (stock, date) 行。

### 1.4 中性化 (Neutralization)

**为什么**: 因子常常和行业、市值高度相关。`momentum_60` 在 2020-2021 高度集中于新能源板块;`book_to_price` 在银行股堆积。如果不剥离这些"风格暴露",IC 看起来很高,但回测中策略实际是在押注行业/市值轮动,而非因子本身。

**做法**:
- 标准做法:每日截面跑 OLS,`f ~ industry_dummies + log_mcap`,取残差。
- 简化版:仅行业中性 — 每个行业内做 `f - industry_mean`(等价于 dummy-only 残差化)。
- WQ101 中的 `IndNeutralize(x, IndClass.subindustry)` 就是这一步,但 WQ 是按子行业,本项目退化到行业。

**关键细节**:
- 中性化是有代价的 — 一个本身就有行业逻辑的因子(如"新能源行业内的高 ROE 股"),做完行业中性后信息所剩无几。
- 业界一般会区分:"alpha 因子"做中性化,"风格因子"(SMB / HML)本身就是 portfolio,不需要。

**项目现状**:
- `ops.indneutralize` 存在(`factors/ops.py`),但只有少数 WQ101 alpha 内部调用
- `factors.context.set_sector_map(...)` 是 opt-in 注入点
- **市值中性完全没有** — 没有 market_cap 数据通路
- `IndustryRelativeStrengthFactor` 在 `get_sector_map()` 为空时直接 raise(算改进),但只覆盖一个因子

### 1.5 正交化 (Orthogonalize) — 可选

**为什么**: 即使做了 1-4,因子之间仍可能高度相关(如 `momentum_5` 和 `momentum_10`)。正交化是从数学上**强制**因子线性无关。

**三种主流做法**:

#### (a) Gram-Schmidt 顺序正交

```
f̃_1 = f_1
f̃_2 = f_2 - proj(f_2, f̃_1)
f̃_3 = f_3 - proj(f_3, f̃_1) - proj(f_3, f̃_2)
...
```

- **优点**: 简单,可控顺序(把"最重要"的因子排第一,保留它的全部信息;后面的因子只贡献增量)。
- **缺点**: 顺序敏感 — 谁排前面谁吃到肉,后面的因子被砍剩残差,可能砍掉它原本的核心信号。
- **适用**: 你对因子重要性有 strong prior 时。

#### (b) Symmetric (对称) 正交化

```
F̃ = F · (FᵀF)^(-1/2)
```

- **优点**: 无序,且数学上保证"正交后矩阵和原矩阵 Frobenius 距离最小"— 信息损失最小。
- **缺点**: 数值上要做特征值分解,因子相关性极高时(λ→0)不稳定;计算量大。
- **适用**: 你认为所有因子地位平等,不愿意人为排序。**业界买方机构(高频做市、统计套利)的默认选择**。

#### (c) PCA 主成分降维

```
F = U·Σ·Vᵀ → 取前 k 个主成分
```

- **优点**: 强制降维,把 100 个因子压成 10 个独立轴。
- **缺点**: 可解释性骤降 — 第 1 个主成分是"什么因子的线性组合",通常没业务含义。
- **适用**: 因子库膨胀到几百个、需要可视化或快速建模时。

**项目现状**: 三种都没实现。

---

## 2. 因子选择与去相关的 5 种范式

"预处理"和"选择"是两个动作,它们处理"因子相关性"的方式可以有不同组合:

### 范式 A: 预处理彻底正交化,选择只看单因子效力

```
原始因子 → ①-⑤ 全流水线(含正交化) → 选择(按 IC/IR 取 top-N) → 等权合成
```

- **典型**: AQR、Two Sigma 内部组合
- **优势**: 选择阶段简单,IC 排序直接用;权重稳定
- **代价**: 正交化代价大(信息损失 + 计算量),小因子库不划算

### 范式 B: 不预处理相关性,选择阶段贪心去相关

```
原始因子 → ①-④(不含正交化) → 选择(按 IC 排序 + pairwise corr < threshold 贪心剔除) → IC 加权
```

- **典型**: 国内卖方研报常见做法,本项目 `pick_top_factors` 就是这条路
- **优势**: 实现简单,可解释("我选了 alpha_017 是因为 IC 高 + 和其他选中因子相关性 < 0.6")
- **代价**: 阈值是超参,0.5 和 0.7 选出来的因子集合差异巨大,缺乏理论指导

### 范式 C: 不预处理,选择交给 L1/Tree 自动稀疏化

```
原始因子 → ①-④ → 全部喂 LassoSelector / LightGBMSelector → 模型自动决定哪些权重压到 0
```

- **典型**: Kaggle 风、ML-first 量化团队
- **优势**: 不需要人为定 threshold,模型自己学
- **代价**: (1) Lasso 在相关特征间随机选一个,跨 refit 不稳;(2) 树模型 importance 对相关特征只挑一个,另一个 importance 直接 0,看起来"被剔除"实际是被吸收
- **本项目现状**: `LassoSelector`(默认)和 `LightGBMSelector` 都是这条路。但**没做 ①-④ 预处理**,所以 Lasso 在量纲不一的输入上做 L1 等于胡来 — 部分被 `_StandardisingMixin` 抢救回来,但截面标准化、winsorize、中性化全缺。

### 范式 D: Residualization(残差化)

```
按重要性排好 N 个因子 → 第 i 个因子对已选的 1..i-1 跑 OLS,取残差作为新的因子值 → 残差化后的因子做合成
```

- **典型**: 多因子 Barra 风格模型,行业研究"贝塔中性化"
- **优势**: 保留所有 N 个因子,只剥离共同信号
- **代价**: 顺序敏感(同 Gram-Schmidt),且每次 refit 都要重跑残差化
- **本项目现状**: 没有

### 范式 E: Symmetric Orthogonalization + IC 加权(纯 quant 主流)

```
预处理后的因子 → symmetric orthogonalize(F·(FᵀF)^(-1/2))→ 在正交因子上算 IC → IC 加权合成
```

- **典型**: 国内顶级私募(明汯、九坤)的标准做法之一
- **优势**: 数学上最优,信息损失最小,无序、可重现
- **代价**: 实现复杂(数值稳定性),且**只有当原因子已经做了 ①-④ 预处理**才有意义 — 直接对未标准化的原始因子做对称正交,等于把量纲噪声当信息

---

## 3. 业界推荐的"预处理 + 选择"组合

以"目标稳定 IR > 0.5"的中频(日频~周频)横截面策略为例,主流组合是:

| 阶段 | 动作 | 备注 |
|---|---|---|
| 1 | 截面中位数填充 NaN | 行业内中位数更好,但要有 sector_map |
| 2 | 截面 winsorize 1%/99% | clip 而非 drop,保留行数 |
| 3 | 截面 z-score | 每日单独算 μ/σ |
| 4 | 行业 + 市值中性化 (OLS 残差) | log(market_cap) 必备 |
| 5 | 滚动 IC 评估 + 半衰期 + IR | 同本项目 `factors_analysis.py` |
| 6 | **预选**:IC 阈值过滤 + 贪心去相关 | 同 `pick_top_factors`,top 30-50 |
| 7 | **可选**:Symmetric orthogonalize 选中的因子 | 主流做法,本项目缺失 |
| 8 | IC / IR 加权合成 | 同 `ICWeighter` / `IRWeighter` |

**关键观察**:
- 步骤 6 (`pick_top_factors`) 和步骤 7 (`symmetric orthogonalize`) 是**互补**而非互斥 — 6 解决"factor library 太大需要降维",7 解决"选中的因子仍然 30% 相关需要彻底解耦"
- 步骤 1-4 是**必经预处理**,跳过它们直接做选择/合成等于在脏数据上建模
- 步骤 7 是 nice-to-have,小因子库(<20 个)收益有限;大因子库(>50)价值显著

---

## 4. 本项目当前状态 vs 业界做法 — 详细对比

| 阶段 | 业界做法 | 本项目实现 | 文件位置 | 评估 |
|---|---|---|---|---|
| NaN 填充 | 截面 / 行业中位数 | 训练 dropna + predict fill 0 | `stack_panel_to_xy`, `MLFactorStrategy._build_x_full` | ⚠️ 训练损失样本,predict 凑合 |
| Winsorize | 每日截面 1%/99% | **无** | — | ❌ 完全缺失 |
| 截面 z-score | 每日单独 | 训练样本 pooled z-score (整体一次) | `_StandardisingMixin._fit_standardiser` | ❌ 错位 — pooled ≠ cross-sectional |
| 行业中性 | 默认开启 OLS 残差 | `ops.indneutralize` 存在,opt-in,仅 WQ101 部分 alpha 使用 | `factors/ops.py`, `factors.context.set_sector_map` | ⚠️ 工具齐但默认不用 |
| 市值中性 | 标准做法 | **无** — 项目无 market_cap 数据通路 | — | ❌ |
| 多重共线性处理 | 正交化 OR 选择阶段去相关 | 贪心去相关 (`pick_top_factors`) + Lasso L1 + LGB gain | `factors_analysis.py:pick_top_factors`, `ml/selectors.py` | ✅ 选择阶段有 |
| Symmetric / Schmidt 正交化 | 主流私募默认 | **无** | — | ❌ |
| PCA 降维 | 大因子库可选 | **无** | — | ❌ |
| 因子选择 | IC + 去相关 + L1 / Tree | `pick-by-ic` CLI + LassoSelector + LightGBMSelector | `cli.py:cmd_factors_pick_by_ic`, `ml/selectors.py` | ✅ 三条路径齐备 |
| IC / IR 加权 | 标准 | `ICWeighter` / `IRWeighter` / `EqualWeighter` / `LightGBMWeighter` | `ml/weighters.py` | ✅ |
| Walk-forward 训练 | 标准 + embargo | 已有 + auto embargo = horizon | `MLFactorStrategy`, F2 PR-A | ✅ |
| 标签 mask(涨跌停/新股) | 标准 | 已落地(mask 只作用标签) | `panel.py`, F2 PR-B | ✅ |

**结论**: 本项目"后端"(选择、合成、训练框架)已经接近业界水准;真正的洼地在"前端"(截面预处理流水线)。

---

## 5. 推荐改进路径(分阶段,小步快跑)

### Phase 1 — 截面预处理 MVP(预期 +0.05~0.15 Sharpe,改动小,无风险)

新增 `src/stockpool/ml/preprocess.py`,加 3 个纯函数:

```python
def cross_sec_winsorize(panel: dict[str, pd.DataFrame], lower=0.01, upper=0.99) -> dict:
    """每日截面 clip 到 [lower 分位, upper 分位]"""

def cross_sec_zscore(panel: dict[str, pd.DataFrame]) -> dict:
    """每日截面 (x - mean) / std"""

def cross_sec_neutralize(panel, sector_map: dict[str, str], demean_only=True) -> dict:
    """每日截面按行业 demean(简化版,无市值)"""
```

在 `MLFactorConfig` 加:

```yaml
strategy:
  ml_factor:
    preprocess:
      winsorize: [0.01, 0.99]      # null = 关闭
      zscore: true                  # 截面 z-score(替换 pooled z-score)
      industry_neutralize: true     # 需要 sector_map
```

`compute_factor_panel` 之后、`stack_panel_to_xy` 之前插一道:

```python
fp = compute_factor_panel(panel, factor_names)
if cfg.preprocess.winsorize:
    fp = {n: cross_sec_winsorize(df, *cfg.preprocess.winsorize) for n, df in fp.items()}
if cfg.preprocess.zscore:
    fp = {n: cross_sec_zscore(df) for n, df in fp.items()}
if cfg.preprocess.industry_neutralize:
    fp = {n: cross_sec_neutralize(df, sector_map) for n, df in fp.items()}
```

同时把 `_StandardisingMixin` 改成 no-op(或仅做 sanity check)— 因为输入已经截面 z-score 过。

**A/B 测试**: 跑 `ab.yaml`,arm_A = 当前流水线,arm_B = 加 Phase 1 预处理。预期 IC IR ↑,sharpe ↑ 5-15%,回撤 ↓。

### Phase 2 — 行业 + 市值双重中性(预期 +0.05 Sharpe,需要新数据通路)

- 接入 baostock 或 akshare 的 `total_market_cap` 数据(可基于 close × shares_outstanding 自算)
- `cross_sec_neutralize` 升级成 OLS 残差化(industry_dummies + log_mcap)
- A/B 验证,只对 alpha 因子开,不对基本面因子开(防止双重中性化抵消)

### Phase 3 — Symmetric Orthogonalization(预期 +0.03~0.10 Sharpe,主要降低权重不稳)

仅在选择**之后**对幸存的 top-N 因子做对称正交:

```python
# selectors.py 加一个后处理
def symmetric_orthogonalize(X: pd.DataFrame) -> pd.DataFrame:
    """F·(FᵀF)^(-1/2),保留列名,数值正交"""
    from scipy.linalg import sqrtm
    cov = X.T @ X / len(X)
    transform = np.linalg.inv(sqrtm(cov + 1e-8 * np.eye(len(cov))).real)
    return pd.DataFrame(X.values @ transform, index=X.index, columns=X.columns)
```

只在 `IRWeighter` / `ICWeighter` 之前调,LGB 不需要(树模型对相关特征鲁棒)。

### Phase 4 — pick_top_factors 算法升级(可选)

当前贪心是"按 IR 排序 + pairwise corr cap";可升级为"最大覆盖问题"近似算法 — 选 k 个因子使得**信息熵覆盖率**最大,而非简单的两两去相关。
学术参考:Tibshirani 的 "minimum redundancy maximum relevance" (mRMR) 算法。但工程性价比不高,建议 Phase 1-3 跑完后再评估。

---

## 6. 关键参考资料

| 来源 | 内容 |
|---|---|
| Barra USE3/USE4 模型手册 | 行业 + 风格中性化的工业级标准 |
| WorldQuant 101 Formulaic Alphas (Kakushadze 2015) | `IndNeutralize` 和 `scale` 的语义 |
| AQR 多篇 papers(尤其 "Buffett's Alpha") | 因子中性化在长期股票多头策略中的必要性 |
| 中泰证券《多因子模型系列报告》 | 国内市场实证:截面 z-score + 行业市值中性 的边际贡献 |
| arXiv 2507.07107 (tradability mask 论文) | mask 与因子预处理的耦合点,本项目 `docs/handoff/2026-05-31-mask-ab-investigation.md` 已采纳一半 |
| 本项目 `docs/research/2026-05-31-a-share-quant-survey-comparison.md` §3.4 | 已经识别出 winsorize / 截面 zscore / 行业中性这三个缺口,但没展开 |

---

## 7. 与本项目其他 doc 的关系

- `docs/strategy_improvement_2026.md` 第 32 行已经识别:"IC weighter 假设各因子独立线性加和,在因子相关性强时会过度押注同一信息源" — 本文给出完整解法
- `docs/research/2026-05-31-a-share-quant-survey-comparison.md` §3.4 列出 "因子预处理 ⭐⭐⭐" 改进项 — 本文是该项的展开
- `docs/handoff/2026-05-31-mask-ab-investigation.md` 已经做了 mask-first 改造,但只覆盖标签层 — 本文的 Phase 1-3 是因子输入层的对应工作

**优先级建议**: Phase 1 是最高 ROI 的下一步工作,代码量 < 200 行(含测试),预期带来 +0.05~0.15 sharpe,且改动可通过 A/B 完全验证。

---

## Phase 1 Outcome (2026-06-06)

Validated via `ab_preprocess.yaml` (16 票, training_universe=pool, lasso+ic, holding_days=10).
**Verdict: ⚠️ INDECISIVE** (|Δsharpe| = 0.013, threshold +0.05).
See `docs/ab_validation_results.md` P4-1 for full metrics.

**核心发现**:
1. 大幅 drawdown 改善(-8.63pp)— winsorize 收紧极端值有效
2. 但 5/16 只票零交易 — 怀疑 verdict thresholds(`strong_buy: 0.9`)未适配 z-score 后的 score 分布
3. Mean sharpe / total return 持平偏负 — alpha 没明显提升

Phase 2 (市值中性) 不应直接上;先做 Phase 1.5:阈值校准 + per-step ablation。
默认 `preprocess` 段保持全关。
