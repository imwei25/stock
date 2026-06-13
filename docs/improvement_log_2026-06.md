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

## 评估池扩充(2026-06-13,基础设施)

`config_eval48.yaml`:48 只评估池。原 16 只原样保留(与历史 A/B 表可比),
追加 32 只 = 全市场近 250 日均成交额 top,行业 cap=3,剔 ST/退市/历史
< 790 bars。覆盖:通信设备、电池、证券、保险、汽车、医药、有色、白酒、
数据中心、畜牧等 ~20 个行业。改进轮的所有 A/B 在该池上跑;
`training_universe=all` 时面板与 16 只池共享缓存(codes=全市场),
新增成本仅每臂 ~3 分钟回测。
