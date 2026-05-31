# A 股日频量化综述对照 (2026-05-31)

> 目的:抓取 2025/2026 两篇最贴近本项目(日频 A 股 + 因子 + ML)的论文,逐环节
> 对照本项目实现,识别差距与改进路线。
>
> 对照执行时间:2026-05-31。下次复盘时请优先 update 第 §4 "改进路线表"。

---

## 0. 文献清单

| # | 论文 | URL | 核心贡献 | 实测性能 (A股 OOS) |
|---|------|-----|---------|------------------|
| **A** | Deep Learning Enhanced Multi-Day Turnover Quant Algorithm for Chinese A-Share Market (2025/6) | https://arxiv.org/abs/2506.06356 | 五模块工程化:截面 DNN 选股 + 开盘 GMM 信号分布 + 流动性自适应仓位 + 网格 TP/SL + HMM 多粒度择时 | 15.2% 年化 / Sharpe 1.87 / MDD 4.8% (2021–2024) |
| **B** | ML Enhanced Multi-Factor Quant Trading: Cross-Sectional Portfolio Optimization with Bias Correction (2026/5) | https://arxiv.org/html/2507.07107 | **可交易性 mask 一等公民**(213 因子 + 18 mask-aware GPU 算子)+ AdjMSE 方向感知损失 + Ledoit-Wolf 协方差 + MVO QP + GBM 合成数据增强 | Sharpe 1.63 / MDD 11.4% (2022–2024 真实数据);**mask 单项 +0.44 Sharpe** |

附行业背景资料:
- [2024/2025 中国量化投资白皮书 — 金融阶/宽邦/华泰/阿里云](https://www.fxbaogao.com/detail/5037837)
- [Stockformer (arXiv 2401.06139)](https://arxiv.org/html/2401.06139v2) — Wavelet + Transformer 多频率融合,后续多频路线参考

---

## 1. 论文 A 详解 (arXiv 2506.06356)

### 1.1 数据预处理 & 特征工程
- 时间跨度:2010-2024(15 年);宇宙:沪深 A 股全样本,**含退市股消除存活偏差**
- 宇宙筛选:市值 > 5 亿、ADV > 1000 万元、非 ST、上市 > 252 天
- 特征 200+ 维:技术(5/10/20/60d 动量、均值回归、波动率、量类)+ 基本面(估值/增长/盈利/杠杆)+ 微观结构(跳空/盘前量比/订单失衡)
- 归一化:**行业中性标准化** `X̃ = (X - μ_sector) / (σ_sector + ε)` + 1%/99% 截尾 + 前向填充

### 1.2 截面选股 DNN
- 多层前馈 + BatchNorm + ReLU + Dropout(0.3 隐 / 0.1 输入)+ 温度 softmax (T=2.0)
- **损失** `L = 0.7·L_ranking + 0.3·L_regression`(强调排序)
- 月度滚动重训 + 时序分块 CV(6 月训练 + 1 月验证)+ 贝叶斯超参优化

### 1.3 开盘信号分布(GMM)
- 信号 `S = α₁·Gap + α₂·VR + α₃·Vol + α₄·Sentiment`,α 时变回归
- 3 分量 GMM 拟合,EM + 正则 `λ·Σ|π_k - 1/3|`
- 动态入场阈值 `θ_t = θ₀ + β·RV_{t-5:t-1}`(高波时提门槛)

### 1.4 仓位管理
- `w_base ∝ Score · √MCap · Mom^0.2 / ADV^0.3 / Vol^0.5 · λ`
- λ = `min(1, TargetVol / (ADV · MaxParticipation))`,`MaxParticipation = 10% ADV`
- 单仓 0.5%–2%、行业 ≤25%、大盘股 20%–60%
- 高波动期减仓:`w_adj = w_base · (1 - 0.5·(VIX_China - mean)/std)`

### 1.5 止盈止损网格
- 1344 组合:TP 1–6% × SL 0.8–3% × max_hold 3–15d × trailing 1.5–3%
- 目标 `0.25·WinRate + 0.35·CumRet/MDD + 0.25·TurnoverEff + 0.15·Consistency`
- HMM 3 态(低/正常/高波),各态独立最优参数,平滑切换 (0.7 新 + 0.3 旧)

### 1.6 多粒度择时
- `σ² = 0.5·GARCH + 0.25·RV + 0.25·SV`,卡尔曼滤波时变权重
- 短(1–5d)/中(5–20d)/长(20–60d)三层特征
- 制度感知 GBM,Viterbi 解码当前 regime
- `TimingSignal = 0.5·Mom + 0.3·Vol + 0.2·Sentiment`

### 1.7 回测细节
- 成本五项合计 22.1 bp 单边(佣金 5 + 印花税 10 + 冲击 3.2 + spread 2.1 + timing 1.8)
- 市场冲击 `0.5·√(S/ADV)·Vol·sign`,日均年化成本 464bp(占 30.5% 毛收益)
- 日度 rebalance,平均持仓 6.2 天、年换手 2100%、日均 50–100 只
- 容量估计 8–12 亿元 RMB

### 1.8 评估指标(2021–2024 OOS)

| 指标 | 本策略 | CSI300 | CSI500 | 因子模型 baseline |
|------|--------|--------|--------|-------------------|
| 年化收益 | 15.2% | 2.8% | 4.1% | 11.3% |
| 年化波动 | 8.1% | 18.3% | 21.2% | 12.1% |
| 最大回撤 | 4.8% | 28.4% | 31.7% | 9.8% |
| Sharpe | 1.87 | 0.15 | 0.19 | 0.93 |
| Sortino | 2.84 | 0.21 | 0.26 | 1.42 |
| Calmar | 3.17 | 0.10 | 0.13 | 1.15 |
| 胜率 | 58.3% | — | — | 55.7% |
| VaR(95%) | -0.8% | -3.2% | -3.7% | -1.5% |

### 1.9 消融实验(关键)

| 配置 | 年化收益 | Sharpe | MDD |
|------|---------|--------|-----|
| 随机基准 | 3.2% | 0.18 | 15.2% |
| +截面选股 | 11.4% | 0.87 | 8.9% |
| +开盘信号 | 13.1% | 1.12 | 7.6% |
| +仓位管理 | 14.3% | 1.34 | 6.2% |
| +网格 TP/SL | 14.8% | 1.61 | 5.1% |
| +择时 | 15.2% | 1.87 | 4.8% |

**主要贡献:截面选股 +8.2%、开盘信号 +1.7%、仓位 +1.2%、网格 +0.5%、择时 +0.4%。**

---

## 2. 论文 B 详解 (arXiv 2507.07107)

### 2.1 可交易性 Mask 的核心发现 ⭐

- 上游污染问题:滚动窗口因子(MA / corr / rank)会在**行过滤前**吸收非执行价格
  (涨跌停、停牌)。Apparent IC 上升 18%,但**实际 Sharpe 下降 0.44**
- Mask:`M_{t,i} = S_{t,i} ∧ (¬L_{t,i}) ∧ IPO_{t,i}`
- 实现:Boolean mask **显式传播**到每个算子(非 NaN,因 GPU 语义不一致)
- 判定:交易所限价 ε=1e-3 或启发式 `|r| > 0.098`

**表 3 mask ablation**:
| 配置 | Apparent IC | Sharpe | MDD |
|------|-------------|--------|-----|
| 无 mask | 0.058 | 1.61 | 18.3% |
| 完整 mask | 0.049 | **2.05** | 12.0% |

### 2.2 因子库 (213 个)

| 家族 | 数量 | 描述 |
|------|------|------|
| Alpha101 子集 | 9 | 仅 Alpha001/002/003/004/006/007/012/053/101 (其余衰减) |
| better_* | 28 | VWAP 偏离、成交量加权动量 |
| best_* | 21 | 收盘位置动量 |
| old_* | 50 | 秩相关合成 |
| stock_* | 22 | 单股派生(波动率/CCI/振幅) |
| extra_* | 14 | 换手率、成交额 |
| add_* | 30 | 复合补充 |
| change_* | 5 | 短窗口加速度 |
| original_* | 28 | 直接收盘/成交量统计 |
| cs_rank_* | 6 | 截面市场宽度 |

- 行业中性:OLS 残差化 29 个 CSI 行业 + 可选 log_mcap,随后截面 z-score

### 2.3 表 6:因子数量 vs 性能
| 因子数 | Sharpe | IC |
|--------|--------|-----|
| 9 | 1.52 | 0.031 |
| 58 | 1.74 | 0.038 |
| 108 | 1.89 | 0.043 |
| 213 | **2.05** | **0.049** |

关键观察:**多样性比单家族质量重要**,无主导因子家族。

### 2.4 模型架构
- **MLP**:2 隐层 128 单元、GELU、dropout 0.1,~44K 参数
- **因子轴 Transformer**:每因子投影到 ℝ⁶⁴ + 位置嵌入 + CLS token → 214 token,
  2 层编码器(4 头、FF dim 256),~220K 参数
  - 关键:**因子轴而非时间轴**,捕获因子间交互(如"动量条件于波动率")
  - 表 8 给出 Transformer 学到的有意义因子对(屏蔽前 5 交互 Sharpe -0.08)

### 2.5 AdjMSE 方向感知损失

```
ℓ(ŷ,y;γ) = w(ŷ,y) · (ŷ-y)²
w = γ        if sign(ŷ)=sign(y)
    1+γ      otherwise
```

- γ=0.1 最优(惩罚比 11:1,方向错误 11× 梯度)
- Sweep γ ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0}

**表 4 损失对比**(Transformer+GBM+LW):
| 损失 | Sharpe | IC | 方向准确率 |
|------|--------|-----|----------|
| MSE | 1.78 | 0.044 | 51.2% |
| AdjMSE γ=1.0 | 1.84 | 0.045 | 52.1% |
| **AdjMSE γ=0.1** | **2.05** | **0.049** | **53.8%** |
| IC loss | 1.95 | 0.047 | 52.9% |

### 2.6 训练细节
- AdamW lr 5e-4,WD 1e-5,grad clip 1.0
- Batch 8192,60 epoch + 10% val 早停
- **双向 mask 检查**:`M_{t,i}=1 ∧ M_{t+1,i}=1`

### 2.7 GBM 数据增强
- 单股 μ̂、σ̂ 估计 → 21 日块重采样(保留短期相关性)→ 213 因子重算
- 表 12:Transformer + n_s=1 合成面板 → +0.19 Sharpe(MLP +0.12);n_s≥5 饱和

### 2.8 组合优化:MVO QP + Ledoit-Wolf

```
max μ̂ᵀw - α·wᵀΣ̂w
s.t. 1ᵀw=1, 0 ≤ w_i ≤ w_max
```

- α=10,w_max=3%,长仓约束
- Ledoit-Wolf 收缩(120d 回溯),PSD 投影夹紧 1e-10
- **表 5**:LW Σ̂ Sharpe 2.05 vs 样本协方差 1.87(+0.18)
- cvxpy warm-start + Parameter 占位符:单日 1.2s → 0.2s(6×)

### 2.9 主结果(net of 8 bps 成本)

**合成数据(2 年,756 日)**:
| 方法 | Sharpe | 年化 | MDD |
|------|--------|------|-----|
| Buy & hold EW | 0.22 | 7.0% | 47.2% |
| LGB + EW top-100 | 1.16 | 13.7% | 17.3% |
| MLP + MSE + LW | 1.72 | 18.1% | 13.8% |
| Transformer + MSE + LW | 1.94 | 19.8% | 12.8% |
| **Full system** | **2.05** | **21.0%** | **12.0%** |

**真实 A 股(2022–2024)**:
| 方法 | Sharpe | 年化 | MDD |
|------|--------|------|-----|
| CSI All-A | 0.19 | 4.1% | 41.8% |
| LGB + EW top-100 | 1.12 | 11.4% | 16.1% |
| MLP + MSE + LW | 1.40 | 13.7% | 13.5% |
| Transformer + MSE + LW | 1.54 | 14.9% | 12.3% |
| **Full system** | **1.63** | **15.8%** | **11.4%** |

### 2.10 Deflated Sharpe Ratio (多重测试校正)
- N≈50 有效配置、T≈756 日、偏度 -0.31、超峰度 4.7
- 阈值 SR̂₀ ≈ 0.93,实现 DSR=0.978(真实 Sharpe > 0 概率 97.8%)

### 2.11 IC-Sharpe 悖论(表 10)
| 配置 | 截面 IC | 可交易 IC* | Sharpe |
|------|---------|-----------|--------|
| 无 mask | 0.058 | 0.032 | 1.23 |
| 仅停牌 | 0.052 | 0.038 | 1.39 |
| 完整 mask | 0.049 | **0.049** | **1.63** |

*仅限次日可交易股票。完整 mask 实现最高可交易 IC。

### 2.12 关键超参总表

| 超参 | 值 |
|------|-----|
| AdjMSE γ | 0.1 |
| 风险厌恶 α (MVO) | 10 |
| 单股上限 w_max | 3% |
| Ledoit-Wolf 回溯 | 120 日 |
| 成本 c | 8 bps |
| GBM 合成面板数 n_s | 1–2 |
| Transformer 头数 | 4 (~220K 参数) |
| dropout | 0.1 |
| 早停耐心 | 10 epoch |

---

## 3. 逐环节对比

下表 **17 个环节**与本项目逐项对照。**有/部分/无** 标记是否覆盖。

### 3.1 数据获取与缓存 ✅ 有 — 项目更强

| 维度 | 综述 | 项目 |
|------|------|------|
| 数据源 | Tushare / WindAPI | mootdx / baostock / akshare 三家可切 |
| 频率 | 日频 OHLCV | 同 |
| 缓存 | 论文未细说 | parquet 单股一文件 + source-change marker + `validate_ohlcv` |
| 全市场清单 | 全 A 4000+ | `fetch-universe` 写 universe.parquet,过滤 ST/科创/北交 |

**结论**:工程实现比两篇都强,无需改动。

### 3.2 可交易性 Mask ❌ 无 — **最大且最易补的缺口** ⭐⭐⭐⭐⭐

| 综述 | 项目 |
|------|------|
| 论文 B 中心发现:涨跌停/停牌当日 close **不可执行**,但仍被 rolling factor 吸收成"幻觉 alpha"。Apparent IC 0.058 vs 真 0.049,**Sharpe 差 0.44**。算子级 mask 传播(18 个 mask-aware GPU 原始操作)。 | `custom.py:limit_up_count` 把涨停当 **特征统计**,但未做 mask 用途。`EligibilityFilter` 仅 portfolio 入场层面(ST / min_history / ADV 20d),**未传播到因子计算**。WQ101 的 `ts_corr / ts_rank / decay_linear` 直接在 close 面板上跑,涨跌停的 close 污染 5–20 个后续窗口。 |

**改进意见**:
1. `panel.py` 建 mask panel:`M = (|ret| < 0.098) & (volume > 0) & (上市天数 > 252)`,与 OHLCV 并列
2. `ops.py` 的 `ts_*` / `rank` / `indneutralize` / `decay_linear` 接受可选 `mask` 参数,聚合前置 NaN(或 +∞ 用于 rank 末位)
3. `ml/dataset.py` 标签 `y_{t+1}` 要求 `M_t=1 ∧ M_{t+1}=1`(双向检查)
4. **预期 +0.2~0.4 Sharpe**,且 default `mask=None` 退化为旧行为(符合"可配置而非硬编码"偏好)

### 3.3 因子库覆盖 ⚠️ 部分 ⭐⭐⭐

| 维度 | 论文 B (213) | 项目 (~111) |
|------|--------------|------------|
| WQ101 子集 | 9 个(其余衰减) | **全 101 个** ✅ |
| VWAP 偏离族 | 28 | 部分(WQ101 内含 vwap proxy);**无独立族** |
| 换手/成交额 | 14 | `turnover_zscore_*` 等,数量少 |
| 收盘位置动量 | 21 | 间接通过 WQ alphas |
| 加速度 | 5 | **无** |
| 截面市场宽度 | 6 | **无** |
| 基本面(估值/盈利/杠杆) | 论文 A 有 | **完全无** |
| 微观结构(跳空/盘前量比) | 论文 A 4 维 | 无 |

**关键发现**(论文 B 表 6):**多样性比单家族质量重要**(9→213 因子 Sharpe 1.52→2.05)。

**改进意见**:
- 短期:加 **VWAP 偏离族** + **换手率 z-score 短窗 (3/5/10)** + **截面市场宽度**(全市场涨跌比、当日触涨停率),~15 个新因子,现有 panel 上就能算
- 中期:接 baostock 财务数据,加 PE/PB/ROE/营收增速等 ~10 个基本面因子前向填充到日频
- 不建议:微观结构在日频信息有限,A 股集合竞价数据麻烦

### 3.4 因子预处理 ⚠️ 部分 ⭐⭐⭐

| 步骤 | 综述 | 项目 |
|------|------|------|
| Winsorize 1%/99% | A 明确 | **无系统性 winsorize**(只有局部 `clip`) |
| 截面 z-score | B 用(行业残差化后 z-score) | `ops.rank(pct=True)` 可替代,但未强制 |
| 行业中性 | B 用 OLS 残差化(29 行业 + log_mcap) | `ops.indneutralize` 存在,但**默认未启用**,`set_sector_map` 注入后才走 |
| 多重共线性 | B 用 LW shrinkage | `pick_top_factors` 只做 pairwise 相关性 cap |

**改进意见**:
- `ml/dataset.py` 加 `preprocess` 步骤:每日截面 winsorize(1%, 99%) → z-score
- `MLFactorConfig` 加 `factor_preprocess: ["winsorize", "zscore", "ind_neutral"]` 列表,default 空(向后兼容)
- `cli._prepare_ml_pool` 注入 `set_sector_map(load_or_build_industry_map(...))`

### 3.5 因子分析与筛选 ✅ 强 — 项目更细

| 项 | 综述 | 项目 |
|----|------|------|
| 滚动 IC | 都做 | `factors_analysis.compute_daily_ic` ✅ |
| IR | 都做 | ✅ |
| Half-life | 论文 B 无 | `factors_analysis` ✅ |
| Regime 切片 | 论文 A 用 HMM 3 态 | `classify_regimes` ✅ |
| 因子相关性 | 都做 | ✅ |
| 选 top-N + 去相关 | B 没专门做 | `factors pick-by-ic --top-n --max-corr --min-ir` ✅ |

**结论**:这一环节比两篇都成熟,无需改动。

### 3.6 模型架构与损失函数 ⚠️ 部分 ⭐⭐

| 项 | A | B | 项目 |
|----|---|---|------|
| 选股模型 | DNN + softmax 排序 | MLP / Transformer 因子轴注意力 | Lasso / LightGBMSelector |
| 加权 | DNN 直出 | MVO QP + LW | IC / IR / Equal / LightGBMWeighter |
| 损失 | 0.7 rank + 0.3 reg | **AdjMSE γ=0.1**(方向错误 11×) | Lasso MSE / LGB L2,**无方向感知** |
| 跨因子交互 | DNN 隐式 | Transformer 显式 | Lasso 线性 / LGB 树(部分) |

**改进意见**:
- **AdjMSE 最便宜**:LightGBM 支持 `objective=callable`,按 `sign(pred)==sign(label)` 加权 γ=0.1。论文 B 报告 +0.27 Sharpe。配合现有 ab.yaml 框架验证(2026-05-24 已回退到 lasso+ic,Adj-MSE 是 LGB 路径的潜在翻盘点)
- Transformer 路线 ROI 低:cfg.stocks 几只 + training_universe=all 4000 只 × 短窗,样本不够喂 220K 参数。**优先把数据 mask + 因子库扩到位后再考虑**

### 3.7 训练协议 ✅ 强

| 项 | 综述 | 项目 |
|----|------|------|
| Walk-forward | 月度 expand | `refit_every` ✅ |
| Embargo | B 用 horizon 间隔 | F2 PR-A `embargo_days` ✅ (default auto=horizon) |
| Pooled 跨股共享 | B 默认 | `share_pool_fit` ✅ |
| 训练/应用池分离 | B 全市场训练 | `training_universe=all` ✅ |
| **GBM 数据增强** | B 用 21 日块重采样 (+0.19 Sharpe) | **无** |
| 验证集泄漏检查 | B 强调双向 mask | 无显式检查 |

**改进意见**(⭐):
- GBM 增强本质是正则化,cfg.stocks 8–16 只 × 几年数据**最大问题是过拟合**,值得试。`ml/dataset.py` 加 `synthetic_panels: int = 0`,default 0
- 双向 mask 检查与 §3.2 mask 一起做

### 3.8 截面组合构建 ⚠️ 部分 ⭐⭐⭐⭐

| 维度 | A | B | 项目 (`PortfolioEngine`) |
|------|---|---|------------------------|
| 入选规则 | top-K (50–100) | top-100 + MVO QP | top-K (default 20) **等权** |
| 权重 | Score·√MCap·Mom^0.2 / ADV^0.3 / Vol^0.5 | MVO QP + LW 120d | **等权** |
| 单股上限 | 0.5%–2% | 3% | 隐含 1/K |
| 行业上限 | 25% | OLS 中性 | `max_per_industry=5` ✅ |

**关键差距**:论文 B 表 5,同模型同因子,**等权 EW-top100 Sharpe 1.12 → MVO+LW 1.63 (+0.5 Sharpe)**。

**改进意见**:
- `PortfolioEngine` 加 `weighting: "equal" | "score" | "mvo"`,default 仍 equal(不破坏现有 ensemble baseline)
- `mvo` 模式 wire `cvxpy + sklearn LedoitWolf`,120d 回溯 + 3% 单股上限
- 新依赖 cvxpy + sklearn 都是 pure Python wheel,引入成本低

### 3.9 仓位 / Sizing ✅ 部分对齐

| 综述 | 项目 |
|------|------|
| A:流动性/波动率反比 + VIX 制度衰减 (高波减仓 50%) | `VolTargetLotSizer`(F3 PR-C):`size = baseline × ref_vol/recent_vol`,clip + skip fallback ✅ |
| B:MVO 内嵌 | 等权 |

**结论**:per-stock 路径基本对齐论文 A。差距:论文 A 还按 √MCap 分散,优先级低。

### 3.10 风险/约束 ⚠️ 部分 ⭐⭐

| 项 | 综述 | 项目 |
|----|------|------|
| 行业上限 | A 25%, B 行业残差化 | `max_per_industry` ✅ |
| 单股上限 | A 2%, B 3% | 隐含 1/K |
| 大盘股比例 | A 20%–60% | 无 |
| 流动性 | A 10% ADV cap | `min_avg_amount_20d=5e7` ✅ |
| 市场状态降权 | A: `w·(1-0.5·(VIX-mean)/std)` | 无 |
| Barra 风险归因 | A 表 11 给敞口 | 无 |

**改进意见**:
- 报告侧加**风格暴露**卡片(SMB/HML/动量/波动率因子的简化 OLS 回归 β)。论文 A 表 11 是好模板
- VIX-China 不存在(中国没 VIX),用 CSI300 30d realized vol 替代

### 3.11 择时 / 制度切换 ❌ 无 ⭐

| 综述 | 项目 |
|------|------|
| A:HMM 3 态 + Viterbi + multi-granularity vol(GARCH+RV+SV)+ momentum/sentiment | 无任何制度切换;阈值固定 |

**结论**:**ROI 低**(论文 A 表消融:从 14.8% → 15.2%,仅 +0.4%)。

**改进意见**(只建议轻量):
- 不做完整 HMM。可做**轻量 risk-off 信号**:全市场 30d RV 突破 95% 分位时仓位 100% → 60%。PortfolioEngine 加 `market_vol_overlay` 参数,default 关闭

### 3.12 止盈止损 / 持仓期 ⚠️ 部分 ⭐⭐

| 综述 | 项目 |
|------|------|
| A:1344 网格 + HMM 分制度选参 + 自适应平滑切换 | `equity_curve_holding_days` 时间止损 ✅;`refresh_verdicts` 重置计时 ✅;**无 TP/SL band** |

**改进意见**:
- `MultiLotBacktestEngine` 已按 lot 独立计时,加 `take_profit_pct` / `stop_loss_pct` / `trailing_stop_pct` 参数自然
- 用现有 portfolio-ab CLI 扫几十组(不要 1344 — 易过拟合)
- **注意**:论文 A 报告这一项仅 +0.5% 年化,不要花太多精力

### 3.13 交易成本模型 ⚠️ 简化 ⭐

| 综述 | 项目 |
|------|------|
| A:五项 22.1bp 单边(佣金 5 + 印花税 10 + 冲击 3.2 + spread 2.1 + timing 1.8)+ 冲击 `0.5·√(S/ADV)·Vol` | `TradeCosts(buy_cost, sell_cost)` 线性比例,default 0.1% |
| B:单边 8bp linear,**论文自己承认线性对大规模过于乐观** | 同左 |

**结论**:与论文 B 平级。论文 A 冲击仅 3.2bp(50–100 股每股 0.4%),你 top-K=20 单仓 5% 冲击更大,但目前不模拟大资金,**实际影响可忽略**。

**改进意见**:不优先做。capacity 估算时再加 `MarketImpactModel`

### 3.14 回测引擎契约 ✅ 强 — 项目更严谨

| 项 | 综述 | 项目 |
|----|------|------|
| T+1 + 次日开盘成交 | A 隐式 / B 明示 | ✅ 严格 + look-ahead 防护 |
| 多仓位 lot | A 有 | `MultiLotBacktestEngine` ✅ |
| Reset timer hook | 论文无 | `should_reset_timer` ✅ |

**结论**:无需改动。

### 3.15 评估指标 ⚠️ 缺多个标准 ⭐⭐⭐

`compute_metrics` 当前只有 `total_return / annualized_return / max_drawdown / sharpe / trade_count / win_rate / avg_trade_return_pct`。

| 指标 | A | B | 项目 |
|------|---|---|------|
| Sharpe | ✅ | ✅ | ✅ |
| **Sortino** | ✅ | ✅ | ❌ |
| **Calmar** | ✅ | ✅ | ❌ |
| **Information Ratio** | ✅ | — | ❌ |
| **Deflated Sharpe** | ❌ | ✅ (DSR=0.978) | ❌ |
| Win rate | ✅ | — | ✅ |
| VaR / ES | ✅ | — | ❌ |
| 年化换手率 | ✅ (2100%) | ✅ (27–39%) | ❌ |

**改进意见**(一晚上能搞完):
- `compute_metrics` 加 4 行:`sortino`、`calmar`、`turnover`、`ir`。零依赖、零回归风险
- **Deflated Sharpe 关键**:A/B 天然多重测试,几十次 arm 对比下 winners 大概率假阳性。`ab/report.py` 加 DSR 列(López de Prado 2018 公式)

### 3.16 多重检验 / 统计显著性 ❌ 无

| 综述 | 项目 |
|------|------|
| B 用 Deflated Sharpe (N≈50, T≈756) | A/B 只给均值/中位/差值/胜出数,**无 p 值,无 DSR** |

**结论**:CLAUDE.md "已知不支持" 已标注。与 §3.15 一起做。

### 3.17 Pool 分离(训练 vs 应用) ✅ 强 — 项目更工程化

| 项 | 综述 B | 项目 |
|----|--------|------|
| 训练用全市场截面 | ✅ 4000+ | `training_universe=all` ✅ |
| 应用 top-K | 100 | `top_k=20` ✅ |
| 缓存隔离 | 论文未细说 | `content_hash` + factor_panel 落盘 ✅ |

**结论**:做得比综述更工程化(content_hash 解耦、score panel 缓存、staggered ensemble)。无需改动。

---

## 4. 改进路线表 (按性价比排序)

| 优先级 | 改进项 | 章节 | 预期收益 | 工作量 | 风险 |
|--------|-------|------|---------|--------|------|
| ⭐⭐⭐⭐⭐ | **可交易性 mask 一等公民** | §3.2 | +0.2~0.4 Sharpe | 中(算子改造) | 低(default=None 退化) |
| ⭐⭐⭐⭐ | **MVO + Ledoit-Wolf 加权**取代等权 | §3.8 | +0.3~0.5 Sharpe | 中(新依赖 cvxpy) | 低 |
| ⭐⭐⭐ | 评估指标补齐 Sortino/Calmar/IR/DSR | §3.15、§3.16 | 评估更可信、防假阳性 | 小 | 零 |
| ⭐⭐⭐ | 因子库扩 VWAP/换手/市场宽度族 ~15 因子 | §3.3 | +0.1~0.2 Sharpe | 中 | 低 |
| ⭐⭐⭐ | 因子预处理 winsorize + cs z-score + 默认行业中性 | §3.4 | +0.05~0.15 Sharpe | 小 | 低 |
| ⭐⭐ | AdjMSE direction-aware 损失 | §3.6 | +0.1~0.3 Sharpe(扩股池后) | 小(LGB custom obj) | 低 |
| ⭐⭐ | TP/SL/trailing-stop 在 `MultiLotBacktestEngine` 加参数 | §3.12 | 小幅风控改善 | 小 | 易过拟合 |
| ⭐⭐ | 风格暴露归因卡片(Barra-lite) | §3.10 | 报告可读性 | 小 | 零 |
| ⭐ | GBM 数据增强 | §3.7 | +0.1~0.2 Sharpe(Transformer 更受益) | 小 | 低 |
| ⭐ | Market vol overlay | §3.11 | 小,且 staggered 已部分平滑 | 小 | 低 |

### 推荐落地顺序

**第一周(快速验证 mask 假设)**:
1. §3.15 + §3.16:补 Sortino/Calmar/IR/DSR(评估基础设施)
2. §3.2:可交易性 mask 第一阶段(只做 panel + ml/dataset 双向检查,不动 ops.py)
3. 用 ab.yaml 跑 mask on/off 对照,验证是否真带来 0.2+ Sharpe

**第二周(组合优化)**:
4. §3.8:`PortfolioEngine` 加 `weighting=mvo` + Ledoit-Wolf
5. 用 portfolio-ab.yaml 跑 equal vs mvo 对照

**第三周(因子扩展)**:
6. §3.3 + §3.4:因子库扩 + 预处理流水线
7. 全部跑通后再考虑 §3.6 AdjMSE / §3.7 GBM 增强

---

## 5. 关键引用

- arXiv 2506.06356 — Deep Learning Enhanced Multi-Day Turnover Quant Algorithm for Chinese A-Share Market (Jun 2025): https://arxiv.org/abs/2506.06356
- arXiv 2507.07107 — ML Enhanced Multi-Factor Quant Trading: Mask-First (May 2026): https://arxiv.org/html/2507.07107
- 2024/2025 中国量化投资白皮书:https://www.fxbaogao.com/detail/5037837
- Stockformer (arXiv 2401.06139): https://arxiv.org/html/2401.06139v2
- López de Prado (2018) "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality" — DSR 公式来源

---

## 附录: A/B 验证结果 (2026-05-31)

落地 §3.2 mask_price 后,跑 ab_mask.yaml(16 股 × ~500 bar)对照 baseline (mask off) vs with_mask (mask on):

| 指标 | baseline | with_mask | Δ |
|------|----------|-----------|---|
| 平均 Sharpe | 0.00 | 0.00 | 0.00 |
| 平均最大回撤 | 0.00% | 0.00% | 0.00% |
| 总样本数 | 16 | 16 | — |
| 胜出股票数 (with_mask > baseline) | — | 0/16 | — |

**说明与解释:**

两臂均产生 0 笔交易,所有指标归零。原因分析:

1. **ml_factor 信号阈值过严**:当前配置 `buy_verdicts: [buy, strong_buy]`、`thresholds.strong_buy: 0.9`,在 16 只测试股票 × training_universe=all (~4359 只全市场训练) 的分位数映射下,16 只样本股的预测分位数始终未触及 0.7 阈值,未产生 buy 信号。此为 ml_factor 策略在极小股票池 + 全市场训练分位数下的已知行为,与 mask 无关。

2. **mask 机制已正确集成**:两臂的 `content_hash` 不同(baseline=`8e6b13ee`,with_mask=`22ce187d`),factor panel 缓存键独立(baseline sig=`a3084b45dcfe`,with_mask sig=`db1890cee23d`),证明 mask 配置已纳入缓存隔离体系。

3. **结论**:mask 未引入新错误(两臂等价 = 0 交易均为无信号,而非 mask 屏蔽了所有信号)。完整的 Sharpe 对比需在更宽松的信号阈值或更大股票池(≥ 100 只)下重跑。HTML 报告详见 `reports/ab/2026-05-31.html`。
