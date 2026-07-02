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

## ★★ 第二池复核结果(M1 落地后,2026-06-27)— 两个 win 均不泛化

用 `ab_pool_v2`(240 票,流动性分位分层,与原池 **0 重叠**)+ bootstrap 复核:

| 方向 | 原池(top-mcap/liq) | 第二池(liq 分层) | 结论 |
|---|---|---|---|
| **top_k 20→10** | ΔSharpe +0.27(CI 含 0) | ΔSharpe **−0.06**(符号翻转,CI 含 0) | **撤销**,config 恢复 top_k=20 |
| **GTJA vs pre-GTJA** | ΔSharpe +0.83 | ΔSharpe **−0.26**(反转;pre_gtja 在 v2 反而更好 1.125 vs 0.868) | **universe-conditional,非普适** |

**关键洞察**:因子集的 edge **依赖投资universe**——GTJA191 短周期量价因子利好**大/流动性**名,
wq101 基础集利好**中等流动性**名。前 24 方向循环的结论是**池特定的**,不跨池泛化。
绝对 Sharpe 在 v2(0.87/1.13)远低于原池(1.33/0.51 各异),印证原池大盘幸存者偏差抬高了绝对数。

**这是比 audit 预期更深的问题**:不是"top_k 边际不显著",而是**连 GTJA 这个最强 win 也不跨池成立**。
→ 在确定**目标投资 universe**(及是否要求 win 跨池一致)之前,**不应继续在单池上做更多优化**
(否则重蹈覆辙)。已暂停新优化循环,待用户决策目标 universe。

config.yaml 处置:top_k 已恢复 20;GTJA selection.json **暂保留**(在设计池/大盘上确有强 edge,
反转可能是真实的 universe 效应而非纯过拟合),但**降级为 universe-conditional**,待用户定目标池后定夺。

## 2026-07-02 — harness 三盲区修复(外部评审驱动),oracle 口径第二次变更

前述硬化(bootstrap/子段/两池)解决的是**统计功率与多重比较**;本次修的是 harness 自身
测不到的**样本构建与执行现实性**盲区。四项修复,全部默认向后兼容、研究路径显式 opt-in:

1. **幸存者偏差**:训练/评估 universe 只含当前上市股(窗口内退市 258 只全部缺席),
   且 `top1000_liquid.parquet` 按**期末**流动性选成员 → 双重前视。修复:退市股回填
   (`scripts/fetch_delisted_universe.py`,244/253)+ as-of PIT 流动性成员
   (`eligibility.top_n_liquidity` / layer_b `--asof-top-n`)。实测静态池每天只覆盖
   真实 as-of top-1000 的中位数 326 只;15 年 union ≈ 全市场。
2. **Layer B oracle 口径**:close-to-close 前向收益含拿不到的隔夜段;不查可交易性。
   修复:默认 open-basis + 入场一字涨停剔除。**新旧 Layer B 数字不可直接比较**
   (旧口径 `--fwd-basis close --tradability none` 可逐位复现)。
3. **执行现实性**:全清仓 rebalance 的虚拟往返成本把换手钉死在 ~100%/rebalance,
   **所有换手敏感 AB(rebalance 周期 / horizon 换手论证 / MVO 换手优势)在旧模型下
   均失焦**;组合引擎无涨停拒单。修复:`hold_survivors` + `limit_guard`。
4. **时间维度 selection bias**:因子池在完整评估期上重选,"两池验证"只解决 universe 轴
   不解决时间轴。修复:`holdout_reselect.py`(截断 ≤cutoff 重选)+ layer_b
   `--date-start`,因子池类结论必须加时间 hold-out 检验。

**重验协议(新方向自此适用)**:Layer B 用新默认 oracle + `--asof-top-n` + 评估池含
退市回填;因子池变更加时间 hold-out;Layer D 引用换手/成本相关结论时必须
`hold_survivors: true` + `limit_guard: true`。既有结论的重验结果见 WORKLOG「RV」条目。

## 后续优化的工作流(硬化版)

每个新方向:portfolio-ab → **两池**(原 ab_pool + ab_pool_v2)各跑 → `ab_significance.py` 出
bootstrap CI + 子段 + validity → **仅 CONFIRMED(两池一致)才 promote 到 config.yaml**;
borderline(单侧 p<0.15 但 CI 含 0)记为 provisional,不动 config。
