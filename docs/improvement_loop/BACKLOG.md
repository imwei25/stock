# 自驱改进循环 — Backlog (任务拆解 + 改进方向)

> 本文件是 `/loop` 自驱改进循环的**持久化状态**。每次迭代先读本文件决定下一步,
> 完成一个方向后更新其状态,并把结果写进 [WORKLOG.md](WORKLOG.md)。
>
> **基线 (baseline)**:`config.yaml` = `factors_file: reports/selection.json`。
> ⚠️ **2026-06-30 起 selection.json = 15yr-IR 重选池**(top-1000 × 15yr IR 重选;Layer B CONFIRMED
>   ΔIC +18.5% / Layer D Sharpe 0.23→0.47;旧 GTJA 集备份 selection_pre_15yr_2026-06-30.json)。见 WORKLOG「P0-3」。
>   revert: `cp reports/selection_pre_15yr_2026-06-30.json reports/selection.json`。
> ⚠️ (历史)A1 起 selection.json 曾 = GTJA 集(备份 selection_pre_gtja_2026-06-26.json)。
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

## ★ 自驱循环当前状态 (cron 0715a0ec, 每 30min, 2026-06-30 起)

> 每个 cron fire:① 读本节;② PowerShell `Get-Process python` 查在跑实验,有(GB 级)就读其 log 看进度、结束本 cycle;
> ③ 若上一实验**已完成**(python 退出且 log 有 VERDICT)→ 按硬化判定记 WORKLOG、更新本节、必要时 promote;
> ④ 空闲 → 起队列下一个;⑤ 跑法:PowerShell Start-Process + `python -u` + scratchpad log + Get-Process 判活。
> log 全在 scratchpad `loop_*.log`(stdout, 有 VERDICT)+ `loop_*.err`(stderr, 有 precompute 进度)。

- **L_alpha = ✅ PROMOTED**(2026-07-01):Layer B CONFIRMED(ΔIC+15.3% t=8.1 全7regime正,OOS反驳过拟合)
  + Layer D not-worse(+0.051)→ **config.yaml alpha 0.001→0.0005**(备份 config_pre_alpha5e4_2026-07-01.yaml)。
  **新基线 alpha=0.0005**。见 WORKLOG「L_alpha」。
- **L_poolsize = ❌ NOT CONFIRMED**:pool44 vs 30 ΔIC +0.0008 持平。保留 30。
- **L_horizon = ❌ NOT CONFIRMED**:h3 vs h5 ΔSharpe −0.003 持平,全市场"h5优"未复现。保留 horizon=3。
- **L_topk = ❌ NOT CONFIRMED**:k10 vs k20 ΔSharpe +0.072 但 CI 含 0、子段不稳、cov 掉 0.89(过度集中)。保留 top_k=20。同 G1。
- **模式**:两真 win(因子集/alpha)皆打分层 Layer B;所有组合参数(top_k/horizon/MVO)持平不硬化 → **转打分层找增益**。
- **L_selector = ❌ NOT CONFIRMED**:lgbm vs lasso ΔIC +0.0007 持平(不再灾难,但不超越)。保留 lasso。
- **L_trainwin = tw=250 确认最优**:tw500 显著更差(ΔIC−0.0011 CI 排除 0)。保留 250。**★ cheap 旧猜想全部重验完毕。**
- **L_ridge = ❌ NOT CONFIRMED(borderline 正)**;**L_halflife = ❌ NOT CONFIRMED**(flat)。→ **weighter 空间穷尽,ic 最优**。
- **L_maxcorr = ❌ NOT CONFIRMED(最大新 lead)**:corr045/22 vs corr06/30 ΔIC +0.0031(+4.7%)IR+7.2% t=2.67,但 CI 勉强含 0+子段不稳。保留。
- **L_rebal = ❌ NOT CONFIRMED**(rebal_10 略差)。保留 rebalance=5。**组合参数空间全重验完毕**。
- **P1-2 mask = ❌ HURTS(项目 prior 验证)**:mask 因子输入 ΔIC −0.0057(−7%)t=−2.8 子段一致负。**归档,不实现生产 mask。**
  剩余最大杠杆 settle 为负。见 WORKLOG「P1-2」。
- **L_combo = ❌ NOT CONFIRMED(lead 是噪声)**:combo +0.0012 弱于任一单 lead → 不叠加 = 多重检验噪声。保留 baseline。
- **★ 可信改进空间穷尽**。转硬化阶段。
- **⚠️ F1 死锁在 1500 票复现**:pool2(1500 票)workers=2 时 precompute 卡在 1499/1500 + workers 不退(Pool teardown hang)。
  **验证 handoff「F1 fix 未在 scale 验证」的担忧:1000 票 OK,1500 票 hang。** 规避:workers=1(串行,无 Pool)。
  → 后续 >1000 票实验一律 workers=1,或修 F1(scoring.py Pool teardown)。
- **✅ 因子集 win = CROSS-POOL CONFIRMED**:pool2(disjoint mid-liq)ΔIC +0.0048 t=4.5 CI 排除 0 halves/thirds 一致。
  **非 universe-conditional,跨池稳健**(不同于旧 GTJA)。见 WORKLOG「P1-2b」。
- **✅✅ 两 win 均 CROSS-POOL CONFIRMED**:因子集(pool2 t=4.5)+ alpha(pool2 t=10.8,全子段正)。**跨池稳健真 alpha。** 见 WORKLOG「两池硬化结论」。
- **★ 硬化阶段完成。** 两个改进经最严两池验证,非过拟合。
- **h10 analyze = 无独立 alpha**:h=10 top-IR 因子同 h=3(动量/技术),基本面仍不入;非新信号源。见 WORKLOG「h10-analyze」。
- **L_h10d = ❌ NOT CONFIRMED(h=10 更差)**:Sharpe 0.533→0.422 ΔSharpe −0.111;换手未降(rebalance 控换手非 horizon)。**horizon 方向彻底结案,h=3 最优。**
- **★★ 可信改进空间彻底穷尽。** 2 cross-pool win;mask 负;params/weighter/pool/horizon 全 flat/负;borderline lead 是噪声。
- **✅ L_bottomline = 累计 Sharpe ~2.2×**:old→new 整体基线 Sharpe 0.247→0.533(ΔSharpe +0.286,p=0.035,粗子段一致)。
  本 session 工作显著提升组合 Sharpe(Layer B 两池 CONFIRMED 的下游兑现)。见 WORKLOG「L_bottomline」。
- **★★★ 循环状态 = IDLE-PENDING-USER-DIRECTION(2026-07-02)**:
  **可信、不过拟合的 cheap 改进空间彻底穷尽**(~18 方向全测)。2 cross-pool 硬化 win 落地(IC +34%,Sharpe ~2.2×)。
  **剩余方向都是下一 tier,需用户决策**:① 新数据源(微观结构/资金流/舆情)② 模型架构(截面 DNN/Transformer+AdjMSE)
  ③ 接受现状。**cron 存活但不再自动 launch 实验**(避免 mine 噪声违背"可信不过拟合")。
  每次 fire:确认无在跑实验 + 无新可信方向 → idle 汇报,等用户在 ①②③ 选向。用户选向后恢复 launch。
- **pool2 bottomline = 放弃**:机器磁盘 IO thrashing(35s/stock,17× 正常,ETA >14h/arm),不可行。且冗余
  (pool2 IC 已 CONFIRMED,cumulative Sharpe top-1000 已 +0.286)→ 证据已足。**机器当前太慢,任何实验暂不可行。**
- **循环终态**:credible 空间穷尽 + 机器 thrashing → 纯 IDLE。等用户 ①②③/停,或机器恢复后再做收尾验证。
- **NEW 方向队列**:
  1-2. ✅ ridge / halflife(均未采纳)。
  3. `L_maxcorr`(进行中)。
  4. **P1-2 算子级 mask**(高EV 冲突 prior)——若 cheap 方向续判负,考虑投入实现 或 向用户报告 cheap 空间穷尽。
- **⚠️ 趋势**:两大 win(因子集/alpha)后,后续所有 cheap 方向(参数/weighter/池组成)**持平不硬化** → 基线是强局部最优。
  真增益或只剩 P1-2 mask(高成本)或全新信号源。若续判负,下一 idle cycle 考虑向用户汇报状态。
  - **P1-2 算子级 mask**(高EV 打分层):⚠️ 与项目现有强 prior 冲突(CLAUDE.md:涨停 close 是有用信号,不 mask 因子输入;
    F1 标签层 mask LOST −0.63)。算子已 NaN-safe → 可不改 ops.py,直接对 pool_data close NaN-out 涨停/停牌格再 build factor panel 来测。
    高成本/不确定,settle 项目-vs-survey 冲突。
  - regime-robust 重选(用 analyze JSON 的 regime_ic 选跨 regime 一致因子)→ 但 robustness 增益体现在 Sharpe(低功率)非 IC。
  - P1-3 基本面(低 EV,h=3 弱)。
- **判负小结**:poolsize/horizon/topk/selector 全持平不硬化;参数空间已穷,**增益只来自打分层信号**(已落地 因子集+alpha)。
- **累计**:session 起 IC 0.0496→0.0665(+34%):① 因子集 15yr 重选 ② alpha 0.001→0.0005,均双层硬化 promoted。
- **队列(未验证,按序)**:
  1. `L_horizon` — horizon 3 vs 5(全市场曾反转favor 5),新池。注意 Layer B 需统一 eval horizon,或改 Layer D。
  2. `L_selector` — lasso vs lightgbm selector(238 灾难 vs 全市场持平),新池 top-1000 Layer B。
  3. `P1-2` — 算子级 mask 传播(综述 +0.44;需改 ops.py 加 mask 参数;与循环里 LOST 的标签层 mask 不同)。Layer B 先筛。
  4. `P1-3` — 基本面因子入 selection(需 baostock fundamentals 缓存;L1 黑名单要先确认缓存在不在)。
  5. (穷尽后)基于 docs/research survey 提新方向:AdjMSE 损失 / GBM 数据增强 / 第二流动性池硬化 P0-3。
- **基线**:selection.json=15yr-IR 池(P0-3 promoted);weighting=equal;其余见下方 baseline 注。
- **判定门槛**:点 ΔIC/ΔSharpe≥阈值 AND bootstrap 95% CI 排除 0 AND 子段符号一致 AND 两 arm valid。

## 子任务 (subtasks) 与改进方向 (directions)

### A. 因子选择 (factor selection)
- [KEPT] A1 — baseline `selection.json` vs `selection_with_gtja_candidate`(+GTJA191, 30)
  → **GTJA 大胜**(Sharpe 0.51→1.33),已 promote 为新基线。详见 WORKLOG。
- [REJECTED] A2 — 新基线(GTJA) vs `selection_clean_rebuild_candidate`
  → Sharpe 1.33→1.19,DD 恶化,各项皆退。保留 GTJA 基线。
- [REJECTED] A3 — 新基线 vs `selection_wq101_localized` → 该文件 == 旧基础集,Sharpe 0.51,大败。
  **子任务 A 结案:GTJA 是最优因子集。**
- [REJECTED] A4 — pick-by-ic 25(无 gtja)→ Sharpe 1.60→1.37,DD↑。**子任务 A 彻底结案,GTJA 最优。**
- [REJECTED] A5 — GTJA-inclusive pick-by-ic 25(含 6 gtja)→ Sharpe 1.60→1.32。
  **factor 空间彻底穷尽,GTJA 手工集对 5 类候选全胜。**

### B. 截面预处理 (preprocess)
- [REJECTED] B1 — `industry_neutralize: true` → Sharpe 1.33→1.24,DD 恶化。保持 off。
- [REJECTED] B2 — `mcap_neutralize: true` → Sharpe 1.33→0.96,大败(抹掉 A 股 size 溢价)。off。
- [REJECTED] B3 — winsorize off → Sharpe 1.60→0.82。winsorize 强有效,保持。**子任务 B 结案。**

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
- [REJECTED] D2 — lightgbm selector → Sharpe 1.60→0.19(灾难,过拟合)。lasso 远优。**子任务 D 结案。**

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

## ★ 遗留问题最终状态 (2026-06-27 network-watcher 收尾)
- **L2 — ✅ CLEARED**:github 恢复可达,`git push origin main` 成功(0261bb2..c738141,25 commits 全推)。
- **L1 — ⚠️ 需用户处理(非循环可清)**:`ab_pool` IPO 硬过滤依赖 baostock `query_stock_basic`,
  但 baostock 账号被**黑名单封锁**(error 10001011「黑名单用户,请与管理员联系」),网络恢复后**仍封**。
  → 非网络问题,polling 无效。需用户:解除 baostock 黑名单,或为 ipo_dates 增加 akshare 数据源路径(代码改动)。
  影响:绝对收益偏乐观;**相对 AB 结论不受影响(两 arm 同池)**。
- **L3 — workaround 生效**:industry_map 缓存 touch 续命,离线复用功能正常(分类月度稳定)。
  baostock 黑名单 + 早前 akshare 连接中断 → 暂不强刷;blacklist 解除后可真正 refresh。

## 2026-06-28 — Sharpe weighter 验证 + 数据扩展过程中的 follow-ups

> 上下文:试图把 weighter 从 `ic` 切到 `sharpe`。Layer D 显示 ΔSharpe +0.117 但
> bootstrap CI 含 0 + 子段反转(NOT CONFIRMED)。扩 cache 到 15-yr,Layer B 在
> top1000 上 3533 日 IC 配对检验也 NOT CONFIRMED(且方向反过来 IC 略胜,详见 WORKLOG
> "D3-Layer-B")。期间发现 + 留下若干技术债。

### follow-ups 处理状态(2026-06-28 收尾)

- [VERIFIED — 根因=内存] **F1 — portfolio_ab runner 15-yr 死锁**
  - **2026-06-30 实测根因**:本机 **31.7GB RAM**。15-yr × 4601 票 × workers≥3 的内存
    footprint(factor panel ~5GB + pooled_xy + 每 worker pickle 整 strategy ~6GB)超 32GB
    → worker 在 prewarm 就被 **OS OOM-kill**(无 traceback)。原 "workers idle / RSS 不释放 /
    hang" 症状 = worker 被 OOM-kill / thrash → Pool imap/teardown 卡死。**这是 F3(内存)
    问题,不是 teardown API bug**。
  - **teardown 修复(close/join 替 forcible terminate)在 15-yr 深度 T=3769 已验证可用**:
    内存安全变体(500 训练池 + 250 组合 universe + workers=3)实跑 `run_single_arm`
    **1064s 干净返回**(14000 trades,引擎跑完);faulthandler 栈确认真正走了 3-worker
    `multiprocessing.Pool` + close/join teardown,无 hang。详见 handoff 文档。
  - **能否本机跑全 15-yr × 4601?不能**(超 RAM,与修复无关)。需更大内存机 / worker 共享
    panel(F3 长期修法)/ 用 ≤1000 票子集。
  - 改动:`portfolio/scoring.py` 显式 close/join + terminate fallback;`runner.py` DBG 探针。
    `pytest tests/test_portfolio_*` + 全套 1119 passed,无回归。

- [DONE] **F2 — `.data_source` marker 并发 race condition**
  - `update_source_marker` 改为 **idempotent**(同 source 跳过写)+ **atomic**
    (temp + `os.replace`);Windows 并发 replace 的 `ERROR_ACCESS_DENIED` 当良性吞掉
    (所有 racer 写同值)。`check_source_change` 把空/空白内容当"无变化"。
  - 测试:`tests/test_fetcher.py` 加 4 个(空 marker / idempotent no-rewrite / 无残留
    .tmp / 8 线程并发无虚报)。全过。

- [MITIGATED] **F3 — 15-yr × full universe ~1% OOM skip**
  - F1 的 close/join 收尾让 worker RSS 跑完即释放;`precompute_scores_from_legacy`
    auto 默认已 `min(3, cpu-1)` workers。文档化于 CLAUDE.md「15-yr scale 已知问题」:
    降 `--workers 3` 规避。长期 panel 分块 / 共享仍未做(ROI 低,非阻塞)。

### 方法学 / 研究 follow-ups

- [DRAFTED] **F4 — Sharpe weighter 作为 regime-conditional alpha**
  - 设计草案落地:`docs/superpowers/specs/2026-06-28-regime-conditional-weighter-design.md`
    (expanding 分位 regime detector → 高波动用 IC / 低波动用 Sharpe;两层评估 + 过拟合护栏)。
  - **未实装**(新研究方向,非 weighter 替换)。default 仍 `ic`。Layer B 先行验证,
    很可能仍 NOT CONFIRMED → 那就归档草案不进生产。

- [DONE] **F5 — `layer_b_direct.py` 工具化**
  - 不 promote 到 `src/`(循环专用分析脚本,非公开 API)。改为**文档化**于 CLAUDE.md
    新增「改进循环分析工具」节(layer_b_direct.py + ab_significance.py + 15-yr 已知问题)。

## 已知遗留问题 (leftover issues) — 原始登记
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
> **★ 循环已收敛并停止(2026-06-27)。** 8 子任务 × 24 方向全部 AB 验证;2 改进落地
> (GTJA 因子集 + top_k=10),累计 portfolio Sharpe 0.51→1.60。其余 22 方向 REJECT,
> 基线为强局部最优。详见 WORKLOG 最终收敛总结。
> 遗留:L2(push)✅ 已于 network-watcher 恢复后清零;L1(baostock 黑名单)需用户处理,
> 非循环可清(polling 无效,blacklist 网络恢复后仍封);L3 workaround 生效。
> **循环正式收尾停止。**
