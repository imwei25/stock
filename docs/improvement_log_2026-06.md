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

## 轮 6(2026-06-13):refit_every 与因子数 — refit10 ❌ 打平 / **top30 ✅ 采纳**

基线已是轮5 的 fixed sizing(sharpe 0.292 / return 19.54%)。两组单旋钮:

**G1 refit_every 20→10**(`ab_eval48_refit10.yaml`):

| 指标 | refit20(base) | refit10 | Δ |
|---|---|---|---|
| Sharpe | 0.292 | 0.288 | −0.003 |
| Total return | 19.54% | 19.31% | −0.23% |

完全打平(交易数都没动,refit 周期对 pooled 全市场月度缓存几乎无影响)。

**G2 因子 top-20→top-30**(`ab_eval48_top30.yaml`,同口径 pick-by-ic 放宽,
新增 alpha_018/027/006/072/036/077/032、mom_vol_interact_10、hl_range_20、
corr_high_low_20):

| 指标 | top20(base) | top30 | Δ | wins(20:30) |
|---|---|---|---|---|
| Sharpe | 0.292 | **0.353** | **+0.062** | 21:25 |
| Sharpe(median) | 0.279 | **0.349** | **+0.070** | — |
| Total return | 19.54% | **20.01%** | +0.47% | 24:22 |
| Max drawdown | 17.28% | **16.72%** | −0.55% | 23:23 |
| Avg trade ret | 2.26% | **2.52%** | +0.26% | 19:27 |

**决策**:**采纳 top30 为生产默认因子清单**。sharpe(mean +0.062 / median
+0.070)、回撤、单笔三维度一致改善;新增 10 因子在 ≤2024-05-20 窗口选取
(评估期 2024-05-21 起,无 in-sample 泄漏),Lasso 在更宽候选上自行稀疏,
不是塞满 30 个。胜出计数偏弱(sharpe 25:21、return base 略赢 24:22),
故标注为**边际增益**,机制上靠新增的量价交互/高低价区间因子补充了
原 20 因子未覆盖的截面信息。

`reports/selection.json` 已更新为 top30(旧 top20 备份
`reports/selection_top20_v2.json`);`selection.json` 纳入版本控制
(.gitignore 加例外,它是配置输入非报告产物)。**新生产基线:
fixed + top30,sharpe 0.353 / return 20.01% / dd 16.72%。**

---

## 轮 7(2026-06-13):lasso alpha 扫描 + min_abs_ic 过滤 — ❌ 三组均未达阈值,无增益轮 #1

基线 = fixed + top30(sharpe 0.353 / return 20.01%)。三组单旋钮:

| 旋钮 | 变体 | 变体 sharpe | Δsharpe | Δreturn | sharpe wins(base:var) | 判定 |
|---|---|---|---|---|---|---|
| lasso alpha 0.001→0.0005 | 更松 | 0.322 | **−0.032** | −1.45% | 25:21 | ❌ 更差 |
| lasso alpha 0.001→0.002 | 更紧 | 0.369 | +0.016 | +1.17% | 22:24 | 微正,噪声级 |
| min_abs_ic 0.0→0.01 | 过滤弱因子 | 0.363 | +0.009 | +0.83% | 21:25 | 微正,噪声级 |

**观察**:alpha 调小(选更多因子)明确变差 → 当前 0.001 已偏稀疏侧最优;
alpha 调大(更稀疏)与 min_abs_ic 过滤弱因子**方向一致都微正**(+0.009~
+0.016),暗示"略保守一点"有极小好处,但单独都不破 +0.05 阈值,胜出
计数也接近五五开。**无增益轮 #1**。下一轮(轮8)验证两个弱正信号叠加
是否到显著,并试 winsorize 收紧与 horizon=2 两个未触方向。

---

## 轮 8(2026-06-13):收敛确认 — ❌ 三组全无显著增益,无增益轮 #2 → **收敛**

基线 = fixed + top30(sharpe 0.353)。三组:

| 方向 | 变体 | 变体 sharpe | Δsharpe | Δreturn | 判定 |
|---|---|---|---|---|---|
| combo:alpha 0.002 + min_abs_ic 0.01 | 弱信号叠加 | 0.359 | +0.006 | +0.87% | 噪声(叠加未放大,反比单独小)|
| winsorize 0.01/0.99→0.02/0.98 | 更激进去极值 | 0.306 | **−0.047** | −1.64% | ❌ 更差 |
| horizon 3→2 | 更短预测窗 | 0.327 | −0.027 | +0.31% | ❌ 略差 |

轮7 两个微正信号(alpha 0.002 +0.016、min_abs_ic 0.01 +0.009)**叠加后
只剩 +0.006**,证明是采样噪声而非可叠加的真实效应。winsorize 收紧与
更短 horizon 均无益。**无增益轮 #2,连续两轮达到预设停止条件,收敛。**

---

# 收敛总结(2026-06-13)

**8 轮扫描、~16 个旋钮,确认当前配置为局部最优。** 起点(修复后的旧默认
vol_target + top20)→ 终点(生产新默认 fixed + top30):

| | 起点 | 终点 | 净增益 |
|---|---|---|---|
| Sharpe | 0.241 | **0.353** | **+0.112(+47%)** |
| Total return | 14.10% | **20.01%** | **+5.91pp** |
| Max drawdown | 15.02% | 16.72% | +1.70pp(代价)|

**三个起作用的改动**(按贡献):
1. **correlation inf 修复**(前置,非调参):平盘日 ±inf 治本,让 P4-1 从
   零交易死表起死回生,并保证后续全部 A/B 在干净面板上跑 —— 真正的地基。
2. **sizing vol_target → fixed**(轮5,+0.051 sharpe / +5.4pp return):
   vol_target 对高波动股缩仓,恰好低配了策略最赚钱的名字。
3. **因子 top-20 → top-30**(轮6,+0.062 sharpe):新增量价交互 / 高低价
   区间因子补充截面信息,Lasso 在更宽候选上自行稀疏。

**确认为局部最优(无显著增益)的旋钮**:training_universe(pool/all 打平)、
horizon(3 最优,2/5 都差)、thresholds(0.90/0.70 最优,收紧伤收益)、
train_window(250 最优)、weighter(ic 优于 ir)、refit_every(20 即可)、
lasso alpha(0.001 最优,两侧都差)、min_abs_ic(0 即可)、winsorize 边界
(0.01/0.99 最优)。

**进一步提升需"重新设计"而非"调参"**(超出本次超参扫描范围,留作后续):
- 新因子研发(当前 top-30 的 IC 上限就这么高;另类数据 / 基本面深加工)。
- **组合层优化**:`training_universe=all` 的真正主战场是 Pool B 的 top-K
  全市场选股 + 周期 rebalance,而非 48 只固定池的单股回测;组合层的行业
  cap / rebalance 频率 / staggered ensemble 还没系统调过。
- 动态标的池(本次评估池是流动性 top,换成因子打分 top-K 动态池会不同)。

---

## 评估池扩充(2026-06-13,基础设施)

`config_eval48.yaml`:48 只评估池。原 16 只原样保留(与历史 A/B 表可比),
追加 32 只 = 全市场近 250 日均成交额 top,行业 cap=3,剔 ST/退市/历史
< 790 bars。覆盖:通信设备、电池、证券、保险、汽车、医药、有色、白酒、
数据中心、畜牧等 ~20 个行业。改进轮的所有 A/B 在该池上跑;
`training_universe=all` 时面板与 16 只池共享缓存(codes=全市场),
新增成本仅每臂 ~3 分钟回测。
