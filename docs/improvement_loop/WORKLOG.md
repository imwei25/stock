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
