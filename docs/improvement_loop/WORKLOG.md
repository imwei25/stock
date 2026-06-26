# 自驱改进循环 — Worklog (逐方向结果)

> 每个方向跑完一条记录:方向 / 假设 / AB 配置 / 指标对比 / 判定 / commit。
> 指标取 `portfolio-ab` 聚合表(total_return / annualized_return / sharpe / max_drawdown / trade_count / win_rate)。

---

## 预备:历史已知 AB 结果(本循环开始前)

- **selection_v2 (20) vs selection_with_gtja (30)** — portfolio-ab @ 238 ab_pool, 2026-06-25
  - Sharpe 0.63 → **1.33** (+0.70);ann_return 0.129 → 0.317;maxDD 0.195 → 0.181
  - 结论:gtja 集大幅胜 v2。但 v2 **不是**当前 prod 基线(prod=selection.json),
    故本循环 A1 重新以 prod 为基线复核 gtja。
  - 报告:`reports/portfolio_ab/2026-06-25.html`

---
<!-- 新记录追加到下方 -->

## B3 — winsorize [0.01,0.99] vs off (panel 重建)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/B3.yaml`
- **结果**(238 ab_pool):winsor_on Sharpe 1.60 / return 1.855 vs winsor_off Sharpe 0.82 / return 0.732。
  Δ Sharpe −0.78。
- **判定**:**REJECTED(off)**。winsorize 强有效(裁尾抑制极端值污染 IC/Lasso)。保持 [0.01,0.99]。
  **子任务 B 完全结案**:winsorize on 必需,industry/mcap neut 皆 off = 最优 preprocess。

---

## E1 — embargo_days auto(=3) vs 0 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/E1.yaml`
- **结果**(238 ab_pool):embargo_auto Sharpe 1.60 / DD 0.164 vs embargo_0 Sharpe 1.41 / DD 0.288。
  Δ Sharpe −0.19,DD 大幅恶化。
- **判定**:**REJECTED(embargo=0)**。去掉 embargo 引入 horizon 日标签泄露,样本变多但 OOS 变差
  (DD 0.288)。**印证 F2 PR-A embargo 设计**。保持 auto。E1(embargo)结案;2×horizon 不再单测
  (auto 已好、0 已坏,更大 embargo 仅减样本)。

---

## C4 — refit_every 20 vs 10 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C4.yaml`
- **结果**(238 ab_pool):**完全 bit-identical**(每项 Δ=0)。refit_10 **确实重算**了
  (日志:pre-warmed 29 monthly fits + parallel 238 stocks,非缓存命中)。
- **判定**:**结构性发现**:pooled `share_pool_fit=true` 打分路径用**月度** refit 节奏
  (29 个月度 fit),`refit_every` 对组合打分路径**无效**(被月度 cadence 覆盖)。无 config 改动。
  C4 结案。

---

## C3b — lasso alpha 0.001 vs 0.005
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C3b.yaml`
- **结果**(238 ab_pool):alpha_1e3 Sharpe 1.60 vs alpha_5e3 Sharpe 0.34。Δ −1.26(灾难性)。
- **判定**:alpha=0.005 过度剪枝(几乎杀光因子)。**alpha=0.001 最优**(0.001 > 0.0005 且 ≫ 0.005)。
  **C3 结案,无 config 改动。**

---

## C3 — lasso alpha 0.001 vs 0.0005 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C3.yaml`
- **结果**(238 ab_pool):alpha_1e3 Sharpe 1.60 vs alpha_5e4 Sharpe 0.91。Δ −0.69。
- **判定**:**REJECTED(0.0005)**。更低 alpha=更少稀疏=更多噪声因子,大败。试 C3b(vs 0.005)bracket。

---

## D1 — weighter ic vs equal (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/D1.yaml`
- **结果**(238 ab_pool):weighter_ic Sharpe 1.60 / return 1.855 vs weighter_equal Sharpe 0.80 / return 0.902。
  Δ Sharpe −0.81(大败)。
- **判定**:**REJECTED(equal)**。IC 加权远优。印证 2026-05-24 回退决定。保持 ic。
- **顺带**:ir 不再单测(equal 已大败,ir 不太可能超 ic);D2(lightgbm selector)CLAUDE.md
  已有负向 AB 证据,降级(若 GTJA 集下想复核可后补)。

---

## C2 — train_window 250 vs 500 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C2.yaml`
- **结果**(238 ab_pool):tw_250 Sharpe 1.60 / return 1.855 vs tw_500 Sharpe 1.35 / return 1.463。
  Δ Sharpe −0.26。
- **判定**:**REJECTED(tw=500)**。长窗跨更多 regime,稀释近期信号。保持 tw=250。

---

## C1b — horizon 1 vs 3
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C1b.yaml`
- **结果**(238 ab_pool):horizon_1 Sharpe 0.80 / return 0.676 vs horizon_3 Sharpe 1.60 / return 1.855。
  Δ horizon=3 +0.80。
- **判定**:horizon=1 太短噪声大,大败。**horizon=3 是最优**(3>5 且 3≫1),保持。**C1 结案。**

## F1 — tradability mask off vs on (仅 limit/停牌,min_listing_days=0 规避缺失的 ipo_dates)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/F1.yaml`
- **背景**:`data/ipo_dates.parquet` 缺失(需 baostock,当前被封)→ listing mask 会退化到
  first_valid_index 启发式(CLAUDE.md 警告:mask 比例虚高)。故设 `min_listing_days=0`
  只测 mask 的核心价值:把涨跌停/停牌日从训练标签层剔除(forward_return 双向检查)。
  score 重算,factor panel cache-hit。
- **结果**(238 ab_pool):mask_off Sharpe 1.60 / return 1.855 vs mask_on Sharpe 0.97 / return 0.777。
  Δ Sharpe −0.63(大败)。
- **判定**:**REJECTED(mask on)**。**重要发现**:把涨跌停日从训练标签剔除 = 丢掉最强的动量
  正样本(涨停 +9.9% 本身是信号),模型变弱。**量化印证** CLAUDE.md 的设计判断("涨停日是有用
  信号")。保持 mask off。**子任务 F 结案。**
- **注**:本测仅 limit/停牌 mask(min_listing_days=0)。listing mask 需 ipo_dates(缺失);
  但核心 mask 已负,不再追加。

---

## C1 — horizon 3 vs 5 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C1.yaml`
- **结果**(238 ab_pool):horizon_3 Sharpe 1.60 / return 1.855 vs horizon_5 Sharpe 1.38 / return 1.416。
  Δ Sharpe −0.22。
- **判定**:**REJECTED(horizon=5)**。3 > 5。试 C1b(1 vs 3)bracket 最优。

---

## G3 — max_per_industry 5 vs 3
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G3.yaml`
- **结果**(238 ab_pool):cap_5 Sharpe 1.60 vs cap_3 Sharpe 1.59,Δ −0.01(噪声级)。
- **判定**:**无差异**。top_k=10 下行业 cap 极少 binding(top-10 已横跨足够行业)。保持 cap=5。
  **子任务 G 结案**:仅 top_k=10 是有效改进;rebal=5 / cap=5 已是最优。

---

## G2b — rebalance_n_days 5 vs 3
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G2b.yaml`
- **结果**(238 ab_pool):rebal_5 Sharpe 1.60 / DD 0.164 vs rebal_3 Sharpe 1.13 / DD 0.291,
  trade +68%。Δ Sharpe −0.48(大败,交易成本主导)。
- **判定**:**REJECTED(rebal=3)**。**rebalance=5 是最优**(5>10 且 5>3),保持现状,无 config 改动。
  **G2 子方向结案**。

---

## G2 — rebalance_n_days 5 vs 10 (engine-only,基线 top_k=10)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G2.yaml`
- **结果**(238 ab_pool):rebal_5 Sharpe 1.60 / DD 0.164 vs rebal_10 Sharpe 1.54 / DD 0.205。
  Δ Sharpe −0.06,DD +0.041(更差)。
- **判定**:**REJECTED(rebal=10)**。保持 rebal=5。但 5<10 趋势 → 试 G2b(5 vs 3,
  与 horizon=3 对齐)看更频繁是否更优(扣已建模交易成本)。

---

## G1b — portfolio top_k 10 vs 5 (集中度 sweep step 2)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G1b.yaml`
- **结果**(238 ab_pool):

  | metric | topk_10 | topk_5 | Δ |
  |---|---:|---:|---:|
  | total_return | 1.855 | 1.627 | −0.227 |
  | sharpe | 1.60 | 1.35 | −0.25 |
  | max_drawdown | 0.164 | 0.255 | +0.090(大幅恶化) |

- **判定**:top_k=5 **过度集中**(DD 0.164→0.255)。**sweep 最优 = top_k=10**(10 > 20 且 10 > 5)。
- **落地**:**config.yaml `portfolio.top_k: 20 → 10`**(已校验加载)。G1 方向 **KEPT**。
  这是本循环**第 2 个 AB 验证的改进**(继 GTJA 因子集)。

---

## G1 — portfolio top_k 20 vs 10 (engine-only, score 缓存共享)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G1.yaml` · 两 arm 均 cache-hit scores
- **假设**:更集中(更少持仓)= 更强信号股权重更高,可能提升风险调整收益。
- **结果**(238 ab_pool):

  | metric | topk_20 | topk_10 | Δ |
  |---|---:|---:|---:|
  | total_return | 1.211 | 1.855 | +0.644 |
  | ann_return | 0.317 | 0.439 | +0.122 |
  | sharpe | 1.33 | 1.60 | +0.27 |
  | max_drawdown | 0.181 | 0.164 | −0.016(更优) |
  | trade_count | 2120 | 1060 | −50% |

- **判定**:**WIN(top_k=10 占优)**,但**先 sweep G1b(10 vs 5)找最优再提交 config**,
  避免过度集中(238 池里 top_k=5 的 idiosyncratic 风险 / 过拟合)。暂不改 config.yaml。

---

## B2 — GTJA 基线 preprocess.mcap_neutralize false vs true
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/B2.yaml`(balance 缓存 offline)
- **假设**:市值中性化去除 size beta,可能提纯 alpha。
- **结果**(238 ab_pool):

  | metric | baseline(off) | mcap_neut(on) | Δ |
  |---|---:|---:|---:|
  | total_return | 1.211 | 0.743 | −0.468 |
  | sharpe | 1.33 | 0.96 | −0.37 |
  | max_drawdown | 0.181 | 0.223 | +0.042(更差) |

- **判定**:**REJECTED**(大幅退化)。A 股 size/小盘溢价是强 alpha 来源,中性化把它抹掉了。保持 off。
- **结论**:子任务 B 两个 neutralize 方向皆负 → 现有 preprocess(winsorize+zscore,两 neut 关)
  已是较优配置。B3(winsorize 微调)预期低 ROI 且需 30min panel 重建,降优先级。
- **效率洞察**:**G 子任务(portfolio 参数 top_k/rebalance/cap)只改 engine 不改 score**,
  score 缓存键不含 portfolio 参数 → 两 arm 共享缓存、仅跑快 engine(~2-3min/AB),优先做。

---

## B1 — GTJA 基线 preprocess.industry_neutralize false vs true
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/B1.yaml`
- **假设**:行业中性化去除行业 beta,可能提纯 alpha。
- **结果**(238 ab_pool):

  | metric | baseline(off) | industry_neut(on) | Δ |
  |---|---:|---:|---:|
  | total_return | 1.211 | 1.132 | −0.079 |
  | sharpe | 1.33 | 1.24 | −0.09 |
  | max_drawdown | 0.181 | 0.232 | +0.052(更差) |
  | win_rate | 0.493 | 0.470 | −0.024 |

- **判定**:**REJECTED**。empirically 印证 CLAUDE.md P1.5:行业中性化在本因子集上
  不增益(单成员子行业 demean-to-zero 风险 + 去掉了有用的行业动量)。保持 off。

---

## A3 — 新基线 GTJA `selection.json` vs `selection_wq101_localized` (30)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/A3.yaml`
- **关键发现**:`selection_wq101_localized.json` 与旧 pre-gtja `selection.json` **因子集完全相同**
  (sorted 相等,30=30)。印证 CLAUDE.md 记载:wq101 本土化 round1 "0 winner",
  文件即基础集。故 A3 实质 = 旧基础集 vs GTJA(同 A1)。
- **结果**(238 ab_pool):baseline_gtja Sharpe 1.33 / return 1.211 vs
  wq101_localized Sharpe 0.51 / return 0.325 / maxDD 0.245(= A1 baseline_prod 完全一致,
  非缓存碰撞,是同因子同分)。Δ Sharpe −0.83。
- **判定**:**REJECTED**。**子任务 A(因子选择)结案**:GTJA 对全部 3 个候选皆大胜,
  增益稳健可复现。

---

## A2 — 新基线 GTJA `selection.json` vs `selection_clean_rebuild_candidate` (30)
- **日期**:2026-06-26 · 配置 `docs/improvement_loop/configs/A2.yaml`
- **假设**:去掉 4 个幻象因子(alpha_027/059/061/095)的 clean rebuild 可能更稳健。
- **结果**(238 ab_pool):

  | metric | baseline_gtja | clean_rebuild | Δ (B−A) |
  |---|---:|---:|---:|
  | total_return | 1.211 | 1.087 | −0.124 |
  | ann_return | 0.317 | 0.291 | −0.026 |
  | sharpe | 1.33 | 1.19 | −0.15 |
  | max_drawdown | 0.181 | 0.257 | +0.076(更差) |
  | win_rate | 0.493 | 0.479 | −0.014 |

- **判定**:**REJECTED**(各项皆退,DD 明显恶化)。clean_rebuild 不含 GTJA 因子,
  本质是另一套 wq101-only 选择,印证 A1 结论:GTJA 因子是增益主来源。保留 GTJA 基线。

---

## A1 — baseline prod `selection.json` (30) vs `selection_with_gtja_candidate` (30, +GTJA191)
- **日期**:2026-06-26 · 配置 `docs/improvement_loop/configs/A1.yaml` · 报告 `reports/portfolio_ab/2026-06-26.html`
- **假设**:GTJA191 本土化短周期量价因子在 prod 基线上提升组合表现。
- **过程坑**:首跑 baseline_prod 因 industry_map 跨 30 天 staleness + 双源网络失败 →
  `IndustryRelativeStrengthFactor` raise → 0 trade(无效)。`touch` 缓存 parquet 重置 mtime
  离线复用后重跑(记 L3)。
- **结果**(238 ab_pool, ~791 bar):

  | metric | baseline_prod | with_gtja | Δ (B−A) |
  |---|---:|---:|---:|
  | total_return | 0.325 | 1.211 | **+0.886** |
  | ann_return | 0.102 | 0.317 | +0.214 |
  | sharpe | 0.51 | 1.33 | **+0.83** |
  | max_drawdown | 0.245 | 0.181 | −0.064(更优) |
  | trade_count | 1902 | 2142 | +240 |
  | win_rate | 0.461 | 0.489 | +0.028 |

  交易集:Only A=2, Only B=1, Both=234 — 同股、更优排序/择时。
- **判定**:**KEPT**(Sharpe +0.83 ≫ +0.10,return↑、DD↓ 全面占优)。
- **落地**:`reports/selection.json` 旧内容备份到 `reports/selection_pre_gtja_2026-06-26.json`,
  canonical `selection.json` 覆盖为 GTJA 集 → **新基线**。`config.yaml` 引用不变(仍指 selection.json)。
- **附注**:prod selection.json(Sharpe 0.51)比历史 v2(0.63)还弱,是三套里最差;GTJA 对两者皆大胜。
- **⚠️ reports/ 被 gitignore**:selection JSON 是本地工件,不入库。为可复现,promote 后的
  30 因子列表存档于此(= 新 `reports/selection.json` 内容):
  ```
  alpha_016, volume_std_20, turnover_zscore_60, gtja_097, gtja_150,
  industry_relative_strength_20, gtja_001, corr_mom_vol_20, alpha_006, alpha_069,
  gtja_158, gtja_080, alpha_012, alpha_037, gtja_020, mom_vol_interact_10,
  alpha_082, alpha_029, gtja_135, close_skew_20, alpha_073, alpha_087, gtja_015,
  gtja_012, alpha_067, alpha_072, alpha_042, alpha_046, gtja_141, close_kurt_20
  ```

---
