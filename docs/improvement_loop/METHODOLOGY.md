# 改进循环 — 方法学审查与硬化 (2026-06-27)

## 审查结论:PASS-WITH-CAVEATS

独立 agent 对前 24 个 AB 方向的方法学做了对抗式审查。结论:

**机制层面正确(审查方努力证伪但未果)**:
- ✅ score 缓存隔离正确(`scoring.py:score_cache_key` 把 factors_file 内容+所有 strategy 维度入哈希);
  之前"GTJA cache hit 复用早前 run"是合法复用(同因子同分),非污染。
- ✅ 无前视泄露:embargo(auto=horizon)、open-to-open 标签、per-day 截面 winsorize/zscore 均 leak-free;
  E1 实测 embargo=0 退化(DD 0.181→0.288)反证 embargo 确实在去泄露。
- ✅ **GTJA 因子集胜出(ΔSharpe +0.83)稳健** — 足以扛过多重比较与噪声检验。

**统计层面欠功率(必须硬化后再做更多优化)**:
- ⚠️ 全部在**单池(238)单段(~791bar)**上选参+验证,无 OOS、无显著性检验、无子段稳健性。
- ⚠️ 单 arm 年化 Sharpe 抽样标准误 ≈ **0.57**;**top_k=10 的 ΔSharpe +0.27 落在噪声带内**。
- ⚠️ ab_pool 是 top-mcap/top-liq 分层 → 幸存者偏差,且与训练池(全市场)耦合;
  "neutralization 抹掉 alpha"(B1/B2)部分可能是高 mcap 评测池的产物。

## 硬化措施(已落地)

**M2/M3/M4 — 噪声感知判定器** `analysis/ab_significance.py`:
- M2 paired circular-block bootstrap → ΔSharpe 95% CI(保 pairing + 自相关,block≈√T)。
- M3 子段(halves/thirds)ΔSharpe 符号一致性。
- M4 arm-validity guard(trade_count>0 + 覆盖≥50%),自动抓 A1 那类 0-trade 退化 arm。
- **新判定准则**:`点 ΔSharpe ≥ +0.10` **且** bootstrap 95% CI 排除 0 **且** 各子段符号一致 **且** 两 arm valid。

**M1 — 第二个 disjoint 评测池**(`ab_pool_v2.parquet`):非 top-mcap/liq 分层、与原池低重叠,
纯离线(从 universe.parquet + 缓存日线算 20d 流动性,跨流动性分位随机分层;不依赖被封的 baostock)。
确认的 win 必须在两池同符号、量级≥半。

## 既有 2 个 win 的重新定级(用硬化判定器)

| Win | 旧结论 | 硬化后 |
|---|---|---|
| **GTJA 因子集** | Sharpe 0.51→1.33 | **CONFIRMED**(+0.83 ≫ 噪声;审查认定稳健)。保留。 |
| **top_k=10** | Sharpe 1.33→1.60 | **NOT CONFIRMED@95%**(ΔSharpe +0.27,CI [−0.13,+0.62] 含 0,单侧 p≈0.09;
  但各子段符号皆正)。**定级降为 provisional**:方向性偏正、DD 更低、经济上合理,**暂保留**,
  待 M1 第二池 + bootstrap 复核。 |

## 后续优化的工作流(硬化版)

每个新方向:portfolio-ab → **两池**(原 ab_pool + ab_pool_v2)各跑 → `ab_significance.py` 出
bootstrap CI + 子段 + validity → **仅 CONFIRMED(两池一致)才 promote 到 config.yaml**;
borderline(单侧 p<0.15 但 CI 含 0)记为 provisional,不动 config。
