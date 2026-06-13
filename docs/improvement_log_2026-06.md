# 策略改进日志(2026-06,v2 干净面板时代)

> 目标:在 correlation inf 修复(panel_version=2)后的干净数据上,通过
> 逐轮"设计 → 单股 A/B 回测验证 → 记录 → commit"持续提升策略 return 与
> Sharpe,直至连续两轮无显著增益(|Δsharpe| < ~0.05 且胜出数无一致优势)。
>
> **评估基准**:`config_eval48.yaml`(48 只 = 原 16 只 + 流动性 top32,
> 行业 cap=3)——原 16 只池 10/16 是半导体,行业集中度会把"策略增益"和
> "板块 beta"混在一起;48 只池同时提高统计功效(A/B 无显著性检验,靠
> 胜出计数,N=48 比 N=16 噪声小得多)。生产 `config.yaml` 仍 16 只不动。
> 历史可比基线(16 只、v2 面板):见 `docs/ab_validation_results.md` 顶表。

---

## 轮 1(2026-06-13):v2 干净面板重新选因子 — ✅ 阴性结果,选因子鲁棒,零改动

**假设**:`reports/selection.json` 的 top-20 是在 v1(inf 中毒)面板上
`pick-by-ic` 选出的;alpha_045 的 IC 算在含 63k inf 的数据上,修复后排名
可能变化,重选可能带来增益。

**做法**:v2 面板上重跑 `factors analyze --universe all --end-date 2024-05-20`
(156 因子 × 4599 股,与 v1 同窗口同参数)→ `pick-by-ic --top-n 20
--max-corr 0.6 --min-ir 0.05` → `reports/selection_v2.json`。

**结果**:**v2 选出的 20 因子集合与 v1 完全一致**(仅 alpha_045/069、
alpha_063/067 等相邻名次微调)。逐因子 IC 对照:除 alpha_045 本身
(IC 0.0221→0.0247,IR 0.423→0.387)外,其余 19 个因子 IC/IR 变化都在
第三位小数。A/B 两臂相同,跳过回测。

**结论**:rank IC + 截面去相关的选因子流程对 ~1.7%/日的 inf 极值污染
**鲁棒**(rank 统计量对极值天然钝感;中毒行又被 mcap 残差化转 NaN 丢弃,
没有进入 IC 样本)。`selection.json` 维持现状;`selection_v2.json` 留档
(reports/,不入库)。**本轮零改动,不构成"无增益轮"计数**(没有做出
改变默认值的尝试)。

---

## 轮 2(2026-06-13):training_universe pool vs all,生产配置 48 股复测 — ⚠️ tied,维持 all,旧 P3-2 结论不可外推

**假设**:master 表最大表观缺口是 P3-2(pool 比 all 好 +0.17 sharpe,
16 股池)——若在生产口径下复现,切回 pool 是免费大增益。

**做法**:`configs/ab/ab_eval48_pool_vs_all.yaml`,两臂仅 training_universe
不同,其余 = 生产默认(selection top-20 + winsorize/zscore/mcap + lasso+ic
+ embargo auto)。48 股池。注意 pool 臂 48 码 < min_pool_size=200 → 截面
预处理被 size guard 跳过(小池训练固有约束)。

**结果**(N=10,48 只):

| 指标 | train_pool | train_all | Δ(all−pool) | wins(pool:all) |
|---|---|---|---|---|
| Total return | **17.92%** | 14.10% | −3.82% | 25:23 |
| Sharpe | 0.224 | **0.241** | +0.017 | 22:26 |
| Max drawdown | 17.08% | **15.02%** | −2.05% | 18:30 |
| Win rate | 54.0% | **64.1%** | +10.1% | 10:36 |
| Avg trade ret | 1.70% | **2.35%** | +0.65% | 14:32 |
| Trade count/股 | 150 | 100 | −50 | — |

**结论**:生产口径 + 行业分散池下 pool 的优势**消失**(sharpe/return
互有胜负;质量指标——胜率/单笔/回撤——一致且明显偏 all)。旧 P3-2 的
"pool 显著优于 all"是 **old8 因子 + 无 preprocess + embargo=0 + 16 股
半导体集中池**的产物(pool 训练≈直接在评估板块上学板块内动量),不可
外推到生产。**training_universe=all 维持**(且它是 Pool B 全市场推荐的
前提)。本轮无默认值变更 —— 计为"无增益轮" #1(连续 2 轮无增益则停)。

---

## 轮 3(2026-06-13):horizon / thresholds / train_window 单旋钮扫描 — ❌ 三项全无增益,生产默认是局部最优

**做法**:三组单旋钮 A/B 串行(`configs/ab/ab_eval48_{horizon5,thresholds_tight,window500}.yaml`),
A 臂统一为生产默认(h3 / 0.90-0.70 / w250,training_universe=all),48 股池,N=10。

| 旋钮 | 变体 | Δsharpe(B−A) | Δreturn | base wins(sharpe) | 判定 |
|---|---|---|---|---|---|
| horizon 3→5 | h5 | −0.002 | +0.00% | 21:25 | ⚠️ 完全打平 |
| thresholds 0.90/0.70→0.95/0.80 | th_tight | **−0.088** | −3.68% | **31:15** | ❌ 倒退 |
| train_window 250→500 | w500 | −0.012 | −0.10% | 30:16 | ⚠️ 微负 |

**机制注脚**:收紧阈值把交易数砍掉 26%(100→73/股)但单笔收益不变
(2.35%→2.36%)——0.90 分位以上没有"更纯"的子集,信号在 0.90/0.70 处
已经被充分切分;再收紧只是均匀地扔掉交易、亏掉收益基数(回撤略降不足
以补偿)。horizon 与 window 则是平坦的:标签窗 3→5 和训练窗 1→2 年都
不改变可学到的截面信号。

**结论**:生产默认(h3 / 0.90-0.70 / w250)在这三个方向上均为局部最优,
零改动。**无增益轮 #2**。剩余值得一试的旋钮:weighter ic→ir、持有期
N=5(信号 horizon=3,N=10 可能稀释信号)→ 轮 4 合并验证后决定是否收敛。

---

## 轮 4(2026-06-13):weighter ic vs ir — ❌ ir 明确更差,IC 加权维持

**做法**:`configs/ab/ab_eval48_weighter_ir.yaml`,仅 weighter.type 不同
(ir 用 n_chunks=6 时间块 IR = mean(日IC)/std(日IC)),48 股池,N=10。

| 指标 | ic(生产) | ir | Δ(ir−ic) | wins(ic:ir) |
|---|---|---|---|---|
| Sharpe | **0.241** | 0.201 | −0.040 | 28:18 |
| Total return | **14.10%** | 11.26% | −2.84% | 30:16 |
| Avg trade ret | **2.35%** | 1.70% | −0.65% | 31:15 |
| Win rate | **64.1%** | 62.3% | −1.8% | 30:16 |

**机制注脚**:IR 把"低波动稳定小 IC"因子加权抬高、"高均值高波动 IC"
因子压低 —— 在本因子组(top-20 已按 NW 修正 ic_ir 选过一轮)上属于
二次惩罚,把真信号的权重摊薄(单笔收益 −28%)。**ic 维持,无增益轮计数
不变**(ir 是探索性新旋钮,非生产默认的回归)。

---

## 轮 5(2026-06-13):sizing vol_target vs fixed — ✅ **翻转默认:fixed 同时赢 return 与 sharpe**

**假设**:v2 重跑的 sizing 组(16 股/旧因子)里 fixed 同时赢 sharpe
(0.623 vs 0.567,11:5)与 return(66.9% vs 47.8%,12:4),但旧结论
"保持 vol_target 观察"——与 return+sharpe 优先的目标矛盾,值得在生产
口径复测。

**做法**:`configs/ab/ab_eval48_sizing_fixed.yaml`,两臂策略完全相同
(生产默认),仅 backtest.sizing.type 不同。48 股池,N=10。

**结果**:

| 指标 | vol_target | fixed | Δ(fixed−vt) | wins(vt:fixed) |
|---|---|---|---|---|
| Total return | 14.10% | **19.54%** | **+5.44pp** | 14:32 |
| Sharpe | 0.241 | **0.292** | **+0.051** | 16:30 |
| Max drawdown | **15.02%** | 17.28% | +2.25pp | 36:10 |
| Win rate | 64.1% | 63.8% | −0.3pp | 30:16 |
| Avg trade ret | 2.35% | 2.26% | −0.09pp | 24:22 |

**机制**:vol_target 按近期波动倒数缩放仓位 —— 但本策略的收益恰恰集中
在高波动名字(单笔质量不变、胜率不变,唯独仓位被系统性砍小),vol_target
等于在最赚钱的交易上低配。fixed 的代价是回撤 +2.3pp,但 return/dd 比
反而从 0.94 升到 1.13。

**决策**:**生产 `config.yaml` 与评估基线 `config_eval48.yaml` 的
backtest.sizing.type 都切到 fixed**(2026-06-13)。后续轮次的基线 =
fixed(sharpe 0.292 / return 19.54%)。求更低回撤的使用者可自行切回
vol_target。无增益轮计数清零。

---

## 评估池扩充(2026-06-13,基础设施)

`config_eval48.yaml`:48 只评估池。原 16 只原样保留(与历史 A/B 表可比),
追加 32 只 = 全市场近 250 日均成交额 top,行业 cap=3,剔 ST/退市/历史
< 790 bars。覆盖:通信设备、电池、证券、保险、汽车、医药、有色、白酒、
数据中心、畜牧等 ~20 个行业。改进轮的所有 A/B 在该池上跑;
`training_universe=all` 时面板与 16 只池共享缓存(codes=全市场),
新增成本仅每臂 ~3 分钟回测。
