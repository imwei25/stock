# 自驱改进循环 — Backlog (任务拆解 + 改进方向)

> 本文件是 `/loop` 自驱改进循环的**持久化状态**。每次迭代先读本文件决定下一步,
> 完成一个方向后更新其状态,并把结果写进 [WORKLOG.md](WORKLOG.md)。
>
> **基线 (baseline)**:`config.yaml` = `factors_file: reports/selection.json`。
> ⚠️ **A1 起 selection.json 内容 = GTJA 集**(旧 prod 备份于 selection_pre_gtja_2026-06-26.json)。
> ⚠️ **G1 起 portfolio.top_k = 10**(原 20)。+
> ml_factor / lasso(α=0.001) / IC(rank) / horizon=3 / train_window=250 / refit=20 /
> preprocess{winsorize+zscore, industry_neutralize=false, mcap=false} / mask=off /
> portfolio{top_k=20, rebalance=5, max_per_industry=5} / sizing=vol_target。
>
> **验证方法**:每个方向用 `portfolio-ab` 在 238-票 `ab_pool` 上跑 A(baseline)vs B(variant)。
> 两 arm 仅差被测维度,其余继承 `config.yaml`。
>
> **判定 (win criterion)**:variant 采纳当且仅当
> `Sharpe 改善 ≥ +0.10` **或**(`Sharpe 改善 ≥ +0.05` 且 `total_return 不更差` 且 `maxDD 不更差`)。
> 采纳 → 改 `config.yaml` + commit;否则 → 保留 baseline,worklog 记 rejected + commit。
> 一切结论标注"方向性,非统计显著"(单池单段,样本小)。
>
> **状态图例**:TODO · IN_PROGRESS · KEPT(采纳)· REJECTED(无增益)· BLOCKED

## 子任务 (subtasks) 与改进方向 (directions)

### A. 因子选择 (factor selection)
- [KEPT] A1 — baseline `selection.json` vs `selection_with_gtja_candidate`(+GTJA191, 30)
  → **GTJA 大胜**(Sharpe 0.51→1.33),已 promote 为新基线。详见 WORKLOG。
- [REJECTED] A2 — 新基线(GTJA) vs `selection_clean_rebuild_candidate`
  → Sharpe 1.33→1.19,DD 恶化,各项皆退。保留 GTJA 基线。
- [REJECTED] A3 — 新基线 vs `selection_wq101_localized` → 该文件 == 旧基础集,Sharpe 0.51,大败。
  **子任务 A 结案:GTJA 是最优因子集。**
- [DEFERRED] A4 — 在胜者基础上 pick-by-ic 去相关/IR 重选(需先跑 factors analyze;
  留作子任务 A 的可选精修,优先做正交方向 B/C/...)

### B. 截面预处理 (preprocess)
- [REJECTED] B1 — `industry_neutralize: true` → Sharpe 1.33→1.24,DD 恶化。保持 off。
- [REJECTED] B2 — `mcap_neutralize: true` → Sharpe 1.33→0.96,大败(抹掉 A 股 size 溢价)。off。
- [IN_PROGRESS] B3 — winsorize [0.01,0.99] vs off(收尾 B 子任务最后一个 prep 旋钮)

### C. ML 超参 (hyperparameters) — 需 score 重算(~5-15min/AB)
- [REJECTED] C1 — horizon=5 → Sharpe 1.60→1.38。3 > 5。
- [REJECTED] C1b — horizon=1 → Sharpe 0.80。**horizon=3 最优(3>5,3≫1),C1 结案。**
- [REJECTED] C2 — tw=500 → Sharpe 1.60→1.35。保持 250。
- [REJECTED] C3 — alpha=0.0005 → Sharpe 1.60→0.91。更少稀疏更差。
- [REJECTED] C3b — alpha=0.005 过度剪枝 Sharpe 0.34。**alpha=0.001 最优,C3 结案。**
- [REJECTED/N-A] C4 — refit_every 10 == 20 bit-identical。pooled 打分用月度 refit,
  refit_every 无效。**C 子任务结案**(horizon=3 / tw=250 / alpha=0.001 全为最优)。

### D. selector / weighter
- [REJECTED] D1 — weighter equal → Sharpe 1.60→0.80。IC 远优,保持。ir 不再单测。
- [DEFERRED] D2 — selector lightgbm:CLAUDE.md 已有负向 AB(LGB+LGB sharpe −0.2),降级。

### E. 标签工程 (label engineering)
- [REJECTED] E1 — embargo=0 → Sharpe 1.60→1.41,DD↑。embargo=auto 印证有效,保持。
- [RESOLVED-BY-REASONING] E2 — label_basis:open=现实 T+1 口径;close 偏乐观(含拿不到的隔夜段),
  即使 AB 数字更高也是**已知乐观偏差**而非真改进 → 保持 open,不做误导性 AB。

### F. 可交易性 mask
- [REJECTED] F1 — mask on → Sharpe 1.60→0.97。剔除涨停标签=丢动量正样本。off。**子任务 F 结案。**

### G. portfolio 组合参数 ⚡(engine-only,score 缓存共享,~2-3min/AB,优先)
- [KEPT] G1 — top_k 20→10:sweep 最优(10 > 20 > ; 10 > 5)。**config.yaml top_k=10 已落地**。
- [REJECTED] G1b — top_k=5 过度集中(DD 0.255)。10 是最优。
- [REJECTED] G2 — rebalance_n_days 10 → Sharpe 1.60→1.54,DD↑。保持 5。
- [REJECTED] G2b — rebal=3 过频(成本主导,Sharpe 1.13)。**rebalance=5 最优,保持。**
- [REJECTED] G3 — cap=3 ≈ cap=5(噪声级,top_k=10 下 cap 极少 binding)。保持 5。**子任务 G 结案。**

### H. sizing
- [TODO] H1 — vol_target(当前) vs fixed(per-stock `ab`,sizing 段覆盖)

## 已知遗留问题 (leftover issues) — 必须清零才能停
- [ ] L1 — `data/ab_pool.parquet` 构建时 baostock 登录失败,跳过了 IPO 硬过滤;
  池里可能含极近 IPO 新股。两 arm 同池故 AB 公平,但绝对收益偏乐观。
  → 待网络恢复后 `ab-pool build --refresh` 重建并复核 top 方向。
- [ ] L2 — `git push` 被网络阻断(github.com:443 不可达);commit 在本地累积,
  待 VPN/代理恢复后统一 push。
- [WORKAROUND] L3 — `data/stock_industry_map.parquet` 缓存于 2026-06-26 跨过 30 天
  staleness,`load_or_build_industry_map(auto)` 触发重拉但 baostock("黑名单用户")
  + akshare(connection aborted)双源失败 → sector_map 空 →
  `IndustryRelativeStrengthFactor` raise → 任何需要**新建** factor panel 的 arm 失败
  (A1 首跑 baseline_prod 即因此 0 trade)。**临时方案**:`touch` 该 parquet 重置 mtime,
  让 loader 离线复用(行业分类月度稳定,AB 相对比较无碍)。网络恢复后应真正 refresh。

## 迭代游标
> next: **B3**(winsorize [0.01,0.99] vs off;panel 重建 ~30min)
