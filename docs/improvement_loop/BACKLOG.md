# 自驱改进循环 — Backlog (任务拆解 + 改进方向)

> 本文件是 `/loop` 自驱改进循环的**持久化状态**。每次迭代先读本文件决定下一步,
> 完成一个方向后更新其状态,并把结果写进 [WORKLOG.md](WORKLOG.md)。
>
> **基线 (baseline)**:`config.yaml` = `factors_file: reports/selection.json`。
> ⚠️ **A1 起 selection.json 内容 = GTJA 集**(旧 prod 备份于 selection_pre_gtja_2026-06-26.json)。+
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
- [IN_PROGRESS] A3 — 新基线 vs `selection_wq101_localized`(WQ101 本土化窗口变体, 30)
- [TODO] A4 — 在 A1-A3 胜者基础上做去相关/IR 重选(pick-by-ic 调 max-corr / min-ir)

### B. 截面预处理 (preprocess)
- [TODO] B1 — `industry_neutralize: true`(当前 false)
- [TODO] B2 — `mcap_neutralize: true`(需 baostock balance 缓存;若拉不到则 BLOCKED)
- [TODO] B3 — winsorize 分位 sweep([0.01,0.99] vs [0.025,0.975] vs off)

### C. ML 超参 (hyperparameters)
- [TODO] C1 — horizon 3 vs 5 vs 1
- [TODO] C2 — train_window 250 vs 500
- [TODO] C3 — lasso alpha 0.001 vs 0.0005 vs 0.005
- [TODO] C4 — refit_every 20 vs 10 vs 40

### D. selector / weighter
- [TODO] D1 — weighter ic vs ir vs equal
- [TODO] D2 — selector lasso vs lightgbm(需超参,先小心)

### E. 标签工程 (label engineering)
- [TODO] E1 — embargo_days auto(=horizon) vs 0 vs 2×horizon
- [TODO] E2 — label_basis open(当前) vs close(确认 open 更优 / 不退化)

### F. 可交易性 mask
- [TODO] F1 — `mask.enabled: true`(涨跌停/停牌/新股标签层屏蔽)

### G. portfolio 组合参数
- [TODO] G1 — top_k 20 vs 10 vs 30
- [TODO] G2 — rebalance_n_days 5 vs 10
- [TODO] G3 — max_per_industry 5 vs 3 vs 8

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
> next: **A3**(基线=GTJA selection.json vs wq101_localized)
