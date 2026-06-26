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
- [TODO-LATER] B3 — winsorize 分位 sweep(低 ROI + 需 30min 重建,排到便宜方向之后)

### C. ML 超参 (hyperparameters) — 需 score 重算(~5-15min/AB)
- [REJECTED] C1 — horizon=5 → Sharpe 1.60→1.38。3 > 5。
- [REJECTED] C1b — horizon=1 → Sharpe 0.80。**horizon=3 最优(3>5,3≫1),C1 结案。**
- [REJECTED] C2 — tw=500 → Sharpe 1.60→1.35。保持 250。
- [TODO] C3 — lasso alpha 0.001 vs 0.0005 vs 0.005
- [TODO] C4 — refit_every 20 vs 10 vs 40

### D. selector / weighter
- [IN_PROGRESS] D1 — weighter ic vs equal
- [TODO] D2 — selector lasso vs lightgbm(需超参,先小心)

### E. 标签工程 (label engineering)
- [TODO] E1 — embargo_days auto(=horizon) vs 0 vs 2×horizon
- [TODO] E2 — label_basis open(当前) vs close(确认 open 更优 / 不退化)

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
> next: **D1**(weighter ic vs equal;score 重算)
