# A/B 验证结果(2026-05-24)

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
    preprocess: { ... }            # ⚠️ Phase 1 indecisive (P4-1),保持全关
```

**未来重新启用 LGB / universe=all 的前提**:
- LGB:调严超参(`num_leaves=7-10, min_data_in_leaf=50+`) **AND** 扩股池或扩 training_universe **AND** A/B 验证 sharpe ≥ baseline
- universe=all:或许配合 LGB(后者需要更多训练样本才能展现非线性优势);或者扩大 cfg.stocks 池本身(从 16 → 50+),让"selection bias"不那么主导
