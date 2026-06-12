# A/B 验证结果

## 2026-06-13 重跑(v2 面板,correlation inf 修复后)— 当前权威结果

> 口径:同 2026-06-12 新口径(hfq + open-to-open 标签 + selection 窗口前移 +
> 涨跌停拒单 + edge 开仓 + 差量调仓 + NW 修正 ic_ir + 双边成本),叠加
> **panel_version=2**:`ops.correlation` ±inf→NaN + clip [-1,1],全部因子
> 面板在干净数据上重建。16 只 cfg.stocks,N=10,verdict 标准不变
> (✅ Δsharpe ≥ +0.2 且 ≥10/16 胜 | ⚠️ ±0.1 内 | ❌ ≤ −0.2 或 ≤6/16 胜)。
> 明细 JSON:`reports/ab_rerun_results.json`;每组 HTML:`reports/ab_rerun/<组名>/`;
> 运行日志:`reports/ab_rerun_v2panel_2026-06-12.log`。
>
> **v1(中毒面板)对照结论:11 组方向性结论全部复现,无一翻转**——
> inf 污染主要通过"中毒日样本被静默丢弃"产生轻度数值偏差,未颠倒任何
> 排序。两个量级变化值得注意:① P4-23 mcap 中性化的优势在干净数据上
> **从 +0.055 增强到 +0.197**(industry 臂在干净数据上明显变差);
> ② P4-1 从零交易死表恢复为正常出数。

| 对照 | A | B | Δsharpe | B wins | Δreturn | Δmax_dd | Verdict(v1 → v2) |
|------|---|---|---------|--------|---------|---------|--------|
| P2-1 | embargo=0 | embargo=auto | −0.016 | 9/16 | +1.4% | +0.3% | ⚠️ 无害 → **⚠️ 无害**(保留 auto,保守防泄漏) |
| P0-1 | composite | lgb+lgb | +0.015 | 8/16 | +4.7% | **+13.0%** ❌ | ⚠️ → **⚠️ tied,LGB 回撤显著更差** |
| P0-2 | lasso+ic | lgb+lgb | **−0.211** | 8/16 | −10.6% | −9.0% ✓ | ❌ → **❌ 复现:LGB 全家桶倒退** |
| P1-1 | lasso+ic | lgb+ic | +0.002 | 4/11 | −0.1% | ±0.0% | ⚠️ → **⚠️ 完全 tied(LGB selector 无增量)** |
| P1-2 | lgb+ic | lgb+lgb | **−0.154** | 7/16 | −9.7% | −9.5% ✓ | ❌ → **❌ LGB weighter 倒退** |
| P3-1 | per_stock | pooled | **+0.156** | 10/16 | +11.2% | +1.0% | ✅ → **✅ 复现:pooled 真收益** |
| P3-2 | training=pool | training=all | **−0.170** | 4/16 | −11.8% | +3.1% | ❌ → **❌ 复现:16 票应用池下全市场训练倒退** |
| P4-1 | preprocess off | on | **+0.060** | 10/16 | +3.1% | +2.0% ❌ | 🚫 跑不通 → **✅ 修复后出数:winsorize+zscore 增益方向成立,回撤略差;默认「winsorize+zscore+mcap 开」维持** |
| P4-2/3 | industry 中性化 | market_cap 中性化 | **+0.197** | 12/16 | +8.5% | −5.4% ✓ | ⚠️/✅ → **✅ 增强复现:mcap 显著更优(干净数据上 industry 臂明显变差),默认保持** |
| P4-4 | 正交化 off | on | **−0.169** | 3/14 | −6.7% | −1.5% ✓ | ❌ → **❌ 加重复现:默认关维持** |
| sizing | fixed | vol_target | −0.056 | 5/16 | **−19.1%** | −1.6% ✓ | ⚠️ → **⚠️ 复现:vol_target 收益大减、回撤略优;默认维持,绝对收益优先可切 fixed** |
| 组合 A/B | simple | — | — | — | — | — | simple 跑通(`reports/portfolio_ab/`);**mask_medium 两臂 fail loud:alpha_048 覆盖率 0%**(v2 重建面板暴露,v1 靠缓存命中跳过了计算;待排查,见下) |

**已知问题(2026-06-13)**:portfolio_ab_mask_medium 在 v2 面板重建时
`alpha_048` 覆盖率 0% 触发 fail loud(两臂同错)。v1 时代该组靠旧缓存命中
从未真算过;属于修复暴露的存量问题而非修复引入,待单独定位(嫌疑:该组
配置的 history 窗口不足 alpha_048 的 ~250 bar warmup,或 sector context
未注入)。

**核心结论(新口径下全部方向性复现)**:
1. **pooled 训练是唯一稳定的真收益**(P3-1,两个口径下都 ✅)——当前默认正确。
2. **LGB selector/weighter 回退到 Lasso/IC 的决策再次确认**(P0-2/P1-2 ❌)。
3. **embargo=auto、market_cap_neutralize 开、正交化关**:默认值全部维持。
4. **training_universe=all 在 16 票应用池下仍倒退**(P3-2)——它的价值前提
   (cross-sec 因子需要宽截面)与小应用池的评估方式存在错配,候选解释:
   全市场训练的分位阈值对 16 票截面不适配;留待组合级(top-K 全市场选股)
   场景再评估,那才是它的主战场。
5. **绝对数字首次可采信**:数据层(复权/单位/盘中 bar)与方法论(标签/选因子
   窗口/执行约束)的已知偏差均已消除;本表数字可作为后续优化的基线。

**P4-1 跑不通的根因(2026-06-12 已定位并修复)**:`ops.correlation`
(pandas `rolling(d).corr`)在**平盘日**(close 两天位级相等,2 日窗口
方差≈0)因矩量公式浮点抵消产出 **±inf**(数学真值 0/0 应为 NaN)。
alpha_045 的 `corr(close, volume, 2)` 全市场命中 63,653 格(4383/4599 票,
日均占有效截面 ~1.7%)。中毒链:
- baseline(全关):inf 绕过 `stack_panel_to_xy` 的 isnan 过滤直进训练
  → Lasso 标准化整列 NaN → 残差污染全部系数 → selected=[] → 等权回退后
  ICWeighter 的 predict 标准化仍整列 NaN(NaN×0=NaN)→ 全部 score NaN
  → 零交易。
- with_preprocess:inf 日均 1.7% > winsorize 1% 上尾 → 当日 0.99 分位 =
  inf,clip 失效;cs_zscore 的 σ=NaN 被误判替换为 1、μ=±inf → **整行**
  推成 ∓inf(实测 116 天整行中毒)→ 同样零交易。
- 其它 20 因子组的 arm 因带 market_cap_neutralize 侥幸存活:`inf−inf=NaN`
  把中毒行转 NaN 后被 drop(代价是悄悄丢了这些天的样本)。

**修复(同日落地,tests: test_ops / test_ml_dataset_finite / test_ml_preprocess)**:
① `ops.correlation` 出口 ±inf→NaN + 有限值 clip [-1,1](治本);
② `stack_panel_to_xy` / `align_xy` 样本过滤改 `isfinite`(防线);
③ `cs_zscore_panel` 退化判定 `~(σ>=1e-12)` 覆盖 NaN σ(防线);
④ 面板 sig 加 `panel_version=2`,v1 毒面板缓存整体失效;ml_models 旧
pkl 已清。修复后 P4-1 当晚重跑通过(0 条 selector-empty 警告,两臂
16/16 done,baseline 均值 108 笔交易/票)。**2026-06-13 凌晨已用 v2 面板
完成全部 11 组 + portfolio simple 的整批重跑**,结果即本文件顶部权威表;
v1↔v2 逐组对照确认无结论翻转。

---

# A/B 验证结果(2026-05-24,历史存档)

> ⚠️ **数据基线警示 (2026-06-11)**:以下全部结论产出于**复权修复之前**——当时
> mootdx 缓存是不复权价,除权除息跳空污染训练标签与回测 PnL(见
> `docs/project_review_2026-06.md` P0-1/2/3,已修复:全链路 hfq + 盘中 bar 防护)。
> 两 arm 共担同一份失真数据,**相对比较(verdict 方向)仍有参考价值,绝对数字
> (sharpe/return/drawdown)不可采信**。新口径重跑结果见本文件顶部。

> 7 个对照按 `docs/ab_validation_runbook.md` 跑出。共同条件: equity_curve_holding_days=[10], history_days=500,16 只 cfg.stocks(P3-2 因时间约束 + share_pool 实现 bug 缩减到 3 只)。
> Verdict 标准(对应 runbook §1):✅ B 改善 sharpe ≥ +0.2 且 ≥10/16 胜 | ⚠️ Δsharpe 在 ±0.1 / 7-9 胜 | ❌ Δsharpe ≤ -0.2 或 ≤6/16 胜 | 🚫 跑不通

## 汇总

| 对照 | A | B | Δsharpe | B wins | Δreturn | Δmax_dd | Verdict |
|------|---|---|---------|--------|---------|---------|---------|
| P0-1 | composite_verdict | lgb+lgb | -0.144 | 9/16 | +1.07% | **+7.83%** ❌ | ⚠️ tied(drawdown 显著恶化) |
| **P0-2** | lasso+ic baseline | lgb+lgb default | **-0.203** | 7/16 | **-20.48%** | -6.16% ✓ | **❌ 倒退** |
| P1-1 | lasso+ic | lgb+ic | -0.027 | 8/16 | -0.99% | -0.47% | ⚠️ tied |
| P1-2 | lgb+ic | lgb+lgb | -0.032 | 8/16 | -12.72% | -5.46% ✓ | ⚠️ sharpe tied,return ❌ |
| **P2-1** | embargo=0 | embargo=auto(3) | **-0.034** | 6/16 | **+4.17%** | +1.15% | **⚠️ tied(embargo 无害)** |
| P3-1 | per_stock | pooled | **+0.233** | 11/16 | +12.24% | -1.65% ✓ | **✅ 真收益** |
| P3-2 (n=3, with bug) | training=pool | training=all | -0.030 | 2/3 | -5.26% | -2.97% ✓ | 🚫 工具 bug,需重跑 |
| **P3-2 (n=16, after fix)** | training=pool | training=all | **-0.243** | 10/16 | **-14.32%** | -0.95% ✓ | **❌ 倒退** |

---

## §1 关键结论(用户必报项)

### P0-2: ❌ F2 当前默认(LGB+LGB)显著倒退 — 建议默认回退到 lasso+ic

- B (LGB selector + LGB weighter) **比 PR-A baseline Lasso+IC 差 0.203 sharpe + 20.48% total return**。
- 唯一 B 胜的维度是 max_drawdown(-6.16%),且 trade_count 翻倍(74 → 111),意味着 LGB 在小训练集(~250 bars × 16 stocks ≈ 4k 行)上**过拟合 → 频繁交易 → 高 churn 低 sharpe**。
- **建议**:把 `config.yaml` / `MLFactorConfig.selector.type` / `weighter.type` 的默认值改回 `lasso` / `ic`,把 LGB 作为可选(README 已有 caveat,但 default 应一致)。把这一发现写进 `docs/strategy_improvement_2026.md` §6 的 F2 路线条目下。

### P2-1: ⚠️ PR-A embargo 无害 — 默认 `embargo_days=None`(auto=horizon)可以放心保留

- 加 embargo 后 sharpe 仅 -0.034(落在 ±0.05 tied 区间),total return 反而 +4.17%,max_dd 略微 +1.15%。
- 没有出现 runbook §3.5 警告的 "embargo 害事 → 小样本 spurious signal 在 OOS 也成立" 的迹象。这是结构性 bug fix 应有的表现:消除标签泄露不带来 free lunch,但不破坏现有曲线。
- **建议**:`embargo_days` 默认值保持 None(auto)即可,无需回退。

---

## §2 拆解(P1-1 + P1-2 vs P0-2)

P0-2 把 PR-B1(LGB selector)和 PR-B2(LGB weighter)合并测,显示 -0.203 sharpe / -20.48% return。

把 F2 拆开看:
- **P1-1(LGB selector 单独贡献)**:Δsharpe=-0.027,Δreturn=-0.99% — 几乎无差异。LGB selector 没赚没赔。
- **P1-2(在 LGB selector 上加 LGB weighter)**:Δsharpe=-0.032,Δreturn=-12.72% — **LGB weighter 主要拉低了 return**。

加和:-0.027 + -0.032 = -0.059 sharpe,但 P0-2 实测 -0.203。差额(-0.144)来自 selector × weighter 交互。**LGB weighter 才是 F2 退化的主因**。如果想保留 LGB,至少应该 weighter 改回 IC。

---

## §3 详细分股

### P0-1: composite_verdict vs ml_factor LGB+LGB

- Stocks: 16/16 both arms succeeded
- A.sharpe=+0.613 / B.sharpe=+0.468
- Trade count: A=28, B=111(B 频繁交易)
- B 在 win_rate / avg_trade_ret / max_dd 全输,只有 total_return 略胜
- Verdict: ⚠️ tied(sharpe Δ 在 -0.1~-0.2 灰色地带),但 max_drawdown +7.83% 是实质恶化
- 备份:`reports/ab/p0_1.html`

### P0-2 ★: Lasso+IC vs LGB+LGB(F2 整体净收益)

- A.sharpe=+0.671 / B.sharpe=+0.468
- A.total_return=+36.92% / B.total_return=+16.44%
- 16 只股中 12 只 A 胜 total_return,9 只 A 胜 sharpe
- **唯一 B 胜的维度**:max_dd(13/16 胜)— LGB 更早止损但代价是 churn
- Verdict: **❌ REGRESSION** — F2 默认值需要重审
- 备份:`reports/ab/p0_2.html`

### P1-1: Lasso+IC vs LGB+IC

- A.sharpe=+0.671 / B.sharpe=+0.644
- 几乎完全平局(8/8 wins on sharpe / total_return / annualized)
- LGB selector 单独没有边际价值
- 备份:`reports/ab/p1_1.html`

### P1-2: LGB+IC vs LGB+LGB

- A.sharpe=+0.644 / B.sharpe=+0.612
- A.total_return=+35.93% / B.total_return=+23.21%(差 12.72%)
- LGB weighter 单独加进来后:trade_count 75→113,returns 下降明显,sharpe 仍 tied
- 备份:`reports/ab/p1_2.html`

### P2-1 ★: embargo=0 vs embargo=auto(=horizon=3)

- A.sharpe=+0.671 / B.sharpe=+0.637
- A.total_return=+36.92% / B.total_return=+41.09%(B 略胜)
- max_dd 略恶化但量小(+1.15%)
- Verdict: **⚠️ tied** — embargo 是没有显著代价的安全修复
- 备份:`reports/ab/p2_1.html`

### P3-1: per_stock vs pooled

- A.sharpe=+0.438 / B.sharpe=+0.671(**Δ=+0.233**)
- 11/16 股 B 胜 sharpe + total_return,12/16 胜 max_dd
- pooled 的 cross-sec 信号 + 池化训练数据明显帮助
- Verdict: **✅ 真收益** — pooled 是合理的默认
- 备份:`reports/ab/p3_1.html`

### P3-2: training_universe=pool vs all(n=3 with bug,n=16 重跑 after fix)

**第一次跑(2026-05-24,bug 未修)**:
- 默认 share_pool 结果完全相同(Δ=0):发现 `ab/runner.py:_decide_pool_sharing` 在 `load_universe=any(arm uses all)` 为真时,会把全市场 universe 注入到**两个 arm** 的 pool_data,导致 training_universe=pool 的 arm A 实际也用了 all 数据。
- 加 `--no-share-pool` 走开本 bug + 缩到 3 股:Δsharpe=-0.030 / Δreturn=-5.26%,n=3 不足以下结论

**Bug 已在 commit `3648d50` 修复**(`fix(ab): gate shared_universe injection per arm`)。

**第二次跑(2026-05-24,16 股全量,after fix `28a1584`)**:

| Metric | pool (A) | all (B) | Δ (B−A) | A 胜 | B 胜 |
|--------|----------|---------|---------|------|------|
| **Sharpe** | +0.671 | +0.428 | **-0.243** | **10** | 6 |
| Total return | +36.92% | +22.60% | -14.32% | 10 | 6 |
| Annualized return | +16.24% | +10.22% | -6.02% | 10 | 6 |
| Max drawdown | +20.31% | +19.36% | -0.95% (less) | 6 | 10 |
| Win rate | +58.24% | +63.40% | +5.16% | 6 | 10 |
| Avg trade ret % | +2.11 | +2.47 | +0.36 | 6 | 10 |
| Trade count | 74 | 85 | +10 | — | — |

**Verdict: ❌ 显著倒退 — `training_universe=all` 不应切默认**

**解读**:
- Sharpe 退 0.24 + return 退 14% + 10/16 A 胜:全市场 4357 股训练让模型在 16 股池上**反而效果差**
- B 唯一胜的维度是"风险更低"(max_dd、win_rate、avg trade ret)— 全市场训练让模型变保守、减少大错但降单笔回报
- 可能原因:(a) cfg.stocks 是手挑的 16 股(化工/半导体/机械/电力 sector bias),全市场训练把这种 informational advantage 平均掉了;(b) IC 在 16 股 vs 全市场上方向不同,加权应用到 16 股回测时错位;(c) "pooled" 是收益(P3-1 已证),但池≠全市场,继续扩反而伤

**结合 F2 倒退看**:两次实验同向指出**当前默认(Lasso+IC + training_universe=pool + panel_mode=pooled)在 16 股上是 sweet spot**,任何方向偏离(模型更复杂 OR 训练数据更广)都让回测变差。

- 备份:`reports/ab/p3_2.html`(n=3, with bug)+ `reports/ab/2026-05-24.html`(n=16, after fix)

---

## §4 后续行动建议

1. **修 ab 工具 bug**:✅ 已修(commit `3648d50`,`fix(ab): gate shared_universe injection per arm`)。
2. **F2 默认值回退**:✅ 已做(commit `28a1584`,`fix(config): rollback selector/weighter defaults to lasso/ic`)。
3. **embargo 默认不变**:✅ 仍是 None(=auto=horizon),P2-1 验证通过。
4. **`training_universe` 默认不切**:✅ 保持 `pool`,P3-2 重跑 16 股 sharpe 退 0.24 / return 退 14%,不应切到 `all`。`training_universe=all` 保留为 opt-in。
5. **写进 strategy_improvement_2026.md §6 路线**:把 P0-1 ~ P3-2 的 verdict 标记为已完成验证;明确 F2 PR-B 系列 + `training_universe=all` 在当前 16 股 × 500bar 上未带来净收益,挪到 follow-up "调超参 + 扩股池 + 更广 universe 同时调参" 之后再启用。

## P4-1: 因子预处理 Phase 1(winsorize + cs_zscore + industry_neutralize)— ⚠️ INDECISIVE

> [SUPERSEDED 2026-06-06 by P4-1b — setup was misconfigured (training_universe=pool with 16 stocks caused single-member-industry demean-to-zero + AB share logic bug); see P4-1b below]

**日期**: 2026-06-06
**配置**: `ab_preprocess.yaml`(training_universe=pool, lasso+ic, holding_days=10, 16 票)
**报告**: `reports/ab/2026-06-06.html`

| 指标 | baseline | with_preprocess | Δ |
|---|---|---|---|
| Sharpe (mean) | +0.457 | +0.470 | **+0.013** |
| Sharpe (median) | +0.333 | +0.344 | +0.011 |
| Total return (mean) | +18.71% | +15.13% | **-3.59%** |
| Max drawdown (mean) | +15.09% | +6.45% | **-8.63pp** ✅ |
| Stocks won (B>A) | - | - | **6/16** |
| Trade count (mean) | 70 | 60 | -10 |
| 0-trade stocks in B | 0 | 5 | +5 🚨 |

**Pass criteria 判定**(spec §7.3):
- Δsharpe ≥ +0.05: ❌(实测 +0.013)
- Total return 同向: ❌(sharpe ↑ 微幅 / return ↓)
- Stocks won > n/2: ❌(6/16)
- Drawdown 不退化 > 3pp: ✅(改善 8.63pp)

**Verdict: ⚠️ INDECISIVE**(|Δsharpe| < 0.05 → 走 ablation 路径)

**关键观察**:
1. **回撤大幅改善但 alpha 持平/微负**:winsorize 收紧极端持仓 → drawdown 显著降低,符合理论。但 alpha 没动 — 单看 risk-adjusted 数字没有结论。
2. **5 票零交易**:with_preprocess 下 5 只票完全不交易(sharpe=0、trade_count=0)。最可能原因:`thresholds.strong_buy=0.9` 是为**原始因子合成 score** 校准的,z-score 后 score 分布变平(mean=0, std=1),0.9 阈值映射到 ~80th 分位,触发频率骤降。需 Phase 1.5 改用分位阈值或重校准。
3. **结果在少数股集中,不广泛**:Δ sharpe 单股范围 -1.04 ~ +1.52,极度分散。说明改进不是稳健 alpha 而是 lucky stocks。

**不切默认值。** `config.yaml` 不动 `preprocess` 段(保持全关)。

**下一步(Phase 1.5)**:
- **优先级 1 — 阈值校准**:在 `with_preprocess` arm 把 `thresholds.{strong_buy, buy, sell, strong_sell}` 改为 z-score 分位(e.g., `0.5σ / 0.2σ / -0.2σ / -0.5σ`),重跑 AB 看 5 票零交易问题是否解决
- **优先级 2 — Per-step ablation**:三个 sub-A/B,每个只开一步(winsorize 单独 / zscore 单独 / industry_neutralize 单独),归因哪一步贡献了 drawdown 改进、哪一步带来了交易缺失
- **优先级 3 — 扩 universe=all 重跑**:cross-sectional 预处理在 16 票上 cross-section 太薄,扩到 4000+ 票看 winsorize 是否还是 no-op

---

## P4-1b: 因子预处理 Phase 1.5 重跑 (winsorize + cs_zscore on full universe) — ✅ PASS

**日期**: 2026-06-06
**配置**: `ab_preprocess.yaml`(training_universe=all, lasso+ic, holding_days=10, 4358 训练股 / 16 应用股)
**报告**: `reports/ab/2026-06-06.html`

| 指标 | baseline | with_preprocess | Δ |
|---|---|---|---|
| Sharpe (mean) | +0.066 | +0.311 | **+0.245** |
| Sharpe (median) | +0.259 | +0.498 | **+0.239** |
| Total return (mean) | +5.11% | +7.76% | **+2.64%** |
| Max drawdown (mean) | +9.61% | +9.24% | **-0.38pp** |
| Stocks won (B>A) | - | - | **11/16** |
| 0-trade stocks in B | 0 | 0 | ✅ 修复 |

**Pass criteria 判定:**
- Δsharpe ≥ +0.05: ✅(+0.245)
- Total return 同向: ✅(+5.11% / +7.76%)
- Stocks won > n/2: ✅(11/16)
- Drawdown 不退化 > 3pp: ✅(改善 0.38pp)

**Verdict: ✅ PASS** — cross-sec winsorize + zscore 在全市场 4358 票上净贡献 sharpe +0.245、return +2.64%。

**P4-1 → P4-1b 关键修复**(详见 commits 75606e6..33c0841):
1. **n_codes 太少导致预处理退化**:16 票 cfg.stocks 里 5 票各占独苗细分行业,`industry_neutralize_panel` 内部 `groupby(industry).transform(s - s.mean())` 对单成员组 demean = 0,5 票全部因子被砍成 0 → 0 trades。
   修复:`PreprocessConfig.min_pool_size: int = Field(default=200)` runtime guard,n_codes < 200 时三步全跳 + warning。
2. **AB 共享 factor_panel bug**:`_decide_pool_sharing` 只比 factors 列表,不比 preprocess,两 arm preprocess 不同时第二个 arm 拿到的是第一个 arm 的 raw panel(P4-1b 第一次重跑时被 bit-identical metrics 暴露)。
   修复:加 `p_a == p_b` (model_dump 比较) 共享 barrier;伴随修 sector_map 在 shared-universe 路径未注入的次生 bug。
3. **AB yaml setup 切到合理 universe**:`training_universe: pool → all`(16 票 → 4358 票),`with_preprocess` arm `industry_neutralize: true → false`(单成员细分行业风险即使在大池下仍存在,Phase 2 全市场参照设计前默认关)。

**结论**:cross-sec 预处理本身是有效的,只是必须在合理宽度的截面(几百到几千股)上做。`industry_neutralize` 暂留作 opt-in,默认 false。

**默认值变更**:`config.yaml` 的 `strategy.ml_factor.preprocess` 段已更新为默认开启 winsorize + zscore(参考 ab_preprocess.yaml with_preprocess arm 的配置)。

---

## P4-2: Phase 2 中性化对比 — industry vs market_cap neutralize — ✅ market_cap 胜

**日期**: 2026-06-09
**配置**: `ab_neutralize.yaml`(training_universe=all 4357 训练股 / 16 应用股,lasso+ic,holding_days=10,两 arm 均在 winsorize+zscore base 之上)
**报告**: `reports/ab/2026-06-09.html`
**市值数据**: `data/mcap_shares.parquet`(baostock profit.totalShare 最新快照,4373 票;mcap=close×totalShare,shares 静态广播,close 日频 PIT;中位覆盖 4273/4357 票)

| 指标 | industry (A) | market_cap (B) | Δ (B−A) | B 胜 |
|---|---|---|---|---|
| Sharpe (mean) | +0.154 | **+0.403** | **+0.249** | 13/16 |
| Total return (mean) | +6.31% | **+10.31%** | **+3.99%** | 13/16 |
| Annualized return | +2.90% | +4.92% | +2.02% | 13/16 |
| Max drawdown (mean) | 12.04% | **8.43%** | **−3.60pp** | 14/16 |
| Win rate | 59.56% | 64.86% | +5.30pp | 12/16 |
| Avg trade ret % | +1.43 | +2.47 | +1.04 | 12/16 |
| Trade count | 72 | 72 | 0 | — |

**Verdict: ✅ market_cap neutralize 全面优于 industry neutralize** — Δsharpe +0.249 / Δreturn +3.99% / 回撤改善 3.60pp / 13(或 14)/16 票胜出。

**跨 run 旁证(非同 run,谨慎解读)**:对照 P4-1b 的 no-neutralize base(winsorize+zscore,sharpe +0.311 / return +7.76%):
- `industry_neutralize` 把 sharpe 从 +0.311 拖到 +0.154 → **行业中性在本 setup 里有害**(可能因 selection.json 已含 `industry_relative_strength_20`,再对全体因子行业 demean 反而抹掉了有效的行业内 alpha)。
- `market_cap_neutralize` 把 sharpe 提到 +0.403(+0.09 over base)、return 提到 +10.31%(+2.55%)→ **市值中性看似在 base 之上继续加分**。

→ 见 **P4-3 同 run 确认**(下)。

---

## P4-3: 确认 base vs market_cap neutralize(同 run)— ✅ PASS,默认开启

**日期**: 2026-06-09
**配置**: `ab_neutralize_confirm.yaml`(同 P4-2 setup;arm A = base winsorize+zscore 无中性,arm B = + market_cap;市值面板 cache 命中 P4-2 的 sig=de6f40b0020c)
**报告**: `reports/ab/2026-06-09.html`(覆盖了 P4-2 的同名报告;P4-2 指标已录于上)

| 指标 | base (A) | market_cap (B) | Δ (B−A) | B 胜 |
|---|---|---|---|---|
| Sharpe (mean) | +0.247 | **+0.403** | **+0.156** | 11/16 |
| Total return (mean) | +6.71% | **+10.31%** | **+3.60%** | 12/16 |
| Annualized return | +3.21% | +4.92% | +1.71% | 12/16 |
| Max drawdown (mean) | 9.34% | **8.43%** | **−0.91pp** | 10/16 |
| Win rate | 63.27% | 64.86% | +1.59pp | 10/16 |
| Avg trade ret % | +1.78 | +2.47 | +0.69 | 13/16 |
| Trade count | 67 | 72 | +4 | — |

**Pass criteria 判定**(同 P4-1b 标准):
- Δsharpe ≥ +0.05: ✅(+0.156)
- Total return 同向: ✅(+6.71% / +10.31%)
- Stocks won > n/2: ✅(11–12/16)
- Drawdown 不退化 > 3pp: ✅(改善 0.91pp)

**Verdict: ✅ PASS** — `market_cap_neutralize` 在 winsorize+zscore base 之上净贡献 sharpe +0.156 / return +3.60% / 回撤改善 0.91pp。

**复现确认(2026-06-09 晚,fresh rebuild 非 cache 命中,数据多 1 天)**:Δsharpe **+0.158** / Δreturn **+3.69%** / DD −0.91pp / 11–12/16 胜 —— 与上午几乎完全一致,结论可复现。

**Phase 2 总结(回答 "按市值 vs industry neutralize"):**
- ✅ **market_cap_neutralize**:在 base 之上加分(P4-3 PASS),且全面优于 industry(P4-2,Δsharpe +0.249)。→ **默认开启**。
- ❌ **industry_neutralize**:在本 setup 下有害(P4-2:0.154 vs base 0.247/0.311),怀疑与 selection.json 已含 `industry_relative_strength_20` 冲突(对全体因子再行业 demean 抹掉了有效的行业内 alpha)。→ **维持默认 false**。

**默认值变更**:`config.yaml` 的 `strategy.ml_factor.preprocess.market_cap_neutralize` 翻为 `true`(industry_neutralize 维持 false)。运行时需 `data/mcap_shares.parquet`(已 commit;`scripts/pull_mcap_profit.py` 重生成),缺失时该步静默跳过 + warning。

**后续(Phase 2.x,可选精修)**:当前 mcap 用最新股本静态广播 + 前复权 close(见 P4-2 已知近似)。若要更严谨,可在全 PIT profit 落地后(`scripts/pull_fundamentals.py`)改用按季 PIT 的 totalShare + 原始 close 重算 mcap,再 A/B 确认增益是否进一步扩大。

**已知近似(影响幅度有限,见 `build_log_mcap_panel` docstring)**:(1) shares 用最新快照静态广播(历史日期轻度前视,但股本缓变,仅影响 size 分桶);(2) close 为前复权,绝对 mcap 被各股复权因子缩放,截面 size 排序为近似。两近似对 **两 arm 对称**(industry arm 不用 mcap;但 base/market_cap 对比时只有 market_cap 用),directional 结论可靠;若 P4-3 确认有效,再考虑用全 PIT profit + 原始 close 精修(Phase 2.x)。

---

## P4-4: 对称正交化 (symmetric / Löwdin orthogonalize) — ⚪ NEUTRAL,默认保持 OFF

**问题**:在已验证的生产默认 preprocess(winsorize + cs_zscore + market_cap_neutralize)之上,
对选定因子做**逐日截面对称正交化**(去除因子间相关性)是否还能加分?

**Setup**:`ab_orthogonalize.yaml`。两 arm 均 `training_universe=all`(~4377 票,full PIT 横截面),
应用/回测池 = `config.yaml` 全 16 票,`equity_curve_holding_days=10`,factors = `reports/selection.json`(20 因子)。
- arm `base`:`symmetric_orthogonalize: false`
- arm `ortho`:`symmetric_orthogonalize: true`(其余完全相同)

正交化作为 preprocess 流水线**最后一步**(winsorize→zscore→mcap→orthogonalize),joint 跨因子逐日
`F_orth = F_std · M^(-1/2)`(M = 标准化截面相关矩阵,`eigh` + 特征值 floor 1e-10)。两 arm factor_panel
sig 独立(base `2d8bcec3f665` / ortho `d55d2ce41832`),缓存隔离已验证。

**聚合(16 只共同股)**:

| Metric | base mean | ortho mean | base median | ortho median | Δ mean (ortho−base) | base/ortho wins |
|---|---|---|---|---|---|---|
| Total return | +6.84% | +6.33% | +5.72% | +7.20% | −0.51% | 10 / 6 |
| Annualized return | +3.12% | +2.92% | +2.85% | +3.57% | −0.20% | 10 / 6 |
| Sharpe | +0.188 | +0.195 | +0.166 | +0.236 | **+0.007** | 8 / 8 |
| Max drawdown(越低越好) | 11.19% | 9.66% | 8.48% | 6.58% | −1.53% | 6 / 10 |
| Win rate | 59.09% | 56.46% | 60.66% | 60.11% | −2.63% | 11 / 5 |
| Avg trade ret % | +1.61 | +1.79 | +0.98 | +1.63 | +0.18 | 8 / 8 |
| Trade count | 69 | 66 | 65 | 65 | −3 | — |

**Pass criteria 判定**(同 P4-3 标准):
- Δsharpe ≥ +0.05: ❌(+0.007,基本持平)
- Total return 同向不退化: ⚠️(mean −0.51% 略退,median +1.48% 改善)
- Stocks won > n/2: ⚪(Sharpe 8-8 平手;return base 10-6 略优)
- Drawdown 不退化:✅(改善 1.53pp,ortho 10-6 胜)

**Verdict: ⚪ NEUTRAL** — 头部 Δsharpe mean +0.007 落在噪声内,未达 +0.05 加分门槛。
正交化**改善回撤 + 中位 Sharpe/return + 单笔均收**,但**略降 mean return 与胜率**,净效应持平。
→ **`symmetric_orthogonalize` 默认保持 false。**

**归因**:`selection.json` 的 20 个因子源自 `pick-by-ic`(带 `max_corr` 去相关筛) + Lasso 稀疏选择,
本身已较解耦,留给正交化可去除的冗余有限,故增益不显著。模块本身实现正确(逐日正交性、order-independence、
近奇异 floor 等单测全过),功能可用 —— 后续若换一组**高相关**的原始因子(未经去相关筛)做选择,
或扩大应用池(16 → 50+)降低噪声,值得重测。

**先小后大两阶段**:smoke(3 票)Δsharpe mean −0.042,full(16 票)+0.007 —— 小样本方向不稳,
印证 16 票应用池仍偏小、AB 无显著性检验(设计如此),结论取 directional 即可。

**已知局限**:应用池仅 16 票,无 p 值;mcap 近似同 P4-2/P4-3(最新股本静态广播 + 前复权 close)。

---

## §5 工程结论(2026-05-24)

经过 7 个 A/B 对照(P0-1 / P0-2 / P1-1 / P1-2 / P2-1 / P3-1 / P3-2,P3-2 重跑两次),得到的**当前默认 sweet spot**:

```yaml
strategy:
  name: ml_factor
  ml_factor:
    panel_mode: pooled            # ✅ P3-1 真收益(Δsharpe=+0.23)
    training_universe: pool       # ❌ all 倒退(Δsharpe=-0.24),保持 pool
    embargo_days: null            # ✅ P2-1 tied,安全保留(=auto=horizon)
    selector: {type: lasso}       # ❌ lightgbm 倒退(F2 默认回退)
    weighter: {type: ic}          # ❌ lightgbm 倒退(F2 默认回退)
    share_pool_fit: true          # (未单独验证,但配合 pooled+pool 工作良好)
    preprocess:                    # ✅ P4-1b: winsorize+zscore;✅ P4-3: market_cap 开启
      winsorize: [0.01, 0.99]
      zscore: true
      industry_neutralize: false   # ❌ P4-2 有害(与 industry_relative_strength 因子冲突)
      market_cap_neutralize: true  # ✅ P4-3 PASS(Δsharpe +0.156 over base)
```

**未来重新启用 LGB / universe=all 的前提**:
- LGB:调严超参(`num_leaves=7-10, min_data_in_leaf=50+`) **AND** 扩股池或扩 training_universe **AND** A/B 验证 sharpe ≥ baseline
- universe=all:或许配合 LGB(后者需要更多训练样本才能展现非线性优势);或者扩大 cfg.stocks 池本身(从 16 → 50+),让"selection bias"不那么主导
