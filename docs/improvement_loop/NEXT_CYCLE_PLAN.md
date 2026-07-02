# 下一轮改进循环 — 地基计划 (2026-06-29)

> **目标 universe(用户已定 2026-06-29)**:**全市场 top-K**(全 A 训练 + 全市场流动性过滤后 top-K 持仓)。
> 所有 AB 结论 universe-conditional;本计划所有验证都以全市场 top-K 为准绳。
>
> **背景**:前 24 方向单池循环已结案,但 `FULL_MARKET_RESULTS.md` 全市场复核证明绝大多数"胜利"
> 是 238-池过拟合产物(GTJA 头条 +0.83→+0.10 持平,多个方向符号反转,唯一稳健显著的是已默认的
> IC 加权)。`METHODOLOGY.md` 明确:**在修好验证 harness + 定 universe 之前不应再做单池优化**。
> 本计划先修地基(P0),再在地基上跑新机制(P1/P2)。

---

## 0. 为什么不直接开 /loop

`/loop` 要三件套:清晰 backlog + **便宜可信的 oracle** + 明确 win 准则。现在第二件是坏的——
单池 portfolio-Sharpe AB(~790 bar,Sharpe SE≈0.57)已被证明会骗人。**先开 loop = 自动量产
跨不了池的伪信号。** 故 P0 先把 oracle 修成可信,P1/P2 才开 loop。

---

## Phase P0 — 地基:让"测得准"成为可能 (gating,必须先做)

### P0-1 验证 15-yr runner 死锁修复(F1)— 几乎零成本,先做
- **现状**:15-yr cache 已建(`data/*_daily.parquet` 4398 files,`history_days: 3750`);F1 已
  hardening(`scoring.py` Pool 改 close()+join();`runner.py` 加 DBG 探针)但**未在真数据验证**。
- **动作**:按 handoff `2026-06-28-portfolio-ab-15yr-deadlock.md` §六跑:
  ```bash
  python -m stockpool portfolio-ab --config docs/improvement_loop/configs/D3b_sharpe_full.yaml --workers 3
  ```
  - **跑通** → F1 CLEARED,Layer D(组合 Sharpe AB)在全市场可用,删 handoff 文档。
  - **仍 hang** → 看最后一条 `DBG:` 探针 + `py-spy dump --pid <PID>` 拿栈,按 handoff §三继续修。
- **产出**:全市场 Layer D 可用 / 不可用的明确结论。

### P0-2 固化"全市场 top-K"评测配置
- 写一个 canonical 评测 config(`docs/improvement_loop/configs/_FULLMKT_BASE.yaml`):
  `training_universe: all` + portfolio universe = 全市场流动性过滤后 top-K(不是 ab_pool 子池)。
- 这是之后**所有** AB 的 base,两 arm 只差被测维度。
- **同时保留 Layer B 路径**(`analysis/layer_b_direct.py`):daily 截面 IC,T=数千日,功率远高于
  组合 Sharpe,且即使 P0-1 失败也能用。**新方向先过 Layer B(高功率筛),通过的再上 Layer D。**

### P0-3 锁死硬化判定标准(写进每个方向的执行)
采纳一个方向当且仅当(`METHODOLOGY.md` 已落地的判定器):
1. 点 ΔSharpe ≥ **+0.10**(或 Layer B 点 ΔIC 对应量级),**且**
2. paired block-bootstrap **95% CI 排除 0**,**且**
3. 子段(halves/thirds)符号一致,**且**
4. 两 arm 都 valid(trade_count>0,覆盖≥50%)。
- **全市场单池**已是最强 oracle(top-K 全市场无幸存者偏差);若仍想要跨池稳健,加 `ab_pool_v2` 复核。

**P0 退出条件**:P0-1 给出 runner 结论 + P0-2 canonical config 就位 + P0-3 判定器接好。
**只有 P0 完成,才进 P1。**

---

## Phase P1 — 新机制(循环从未碰过的结构性杠杆,ROI 最高)

> 这些**不是**因子重排(那个空间已穷尽),而是新机制。每个都走 P0-3 判定器,全市场验证。

### P1-1 MVO + Ledoit-Wolf 组合权重(替代等权)⭐ 组合侧最大单一杠杆
- **依据**:综述论文 B 表 5,同模型同因子,等权 EW-top100 Sharpe 1.12 → MVO+LW **1.63(+0.5)**。
  G 子任务只调过 top_k/rebalance/cap,**从没动过 weighting 机制**。
- **改动**:`PortfolioEngine` 加 `weighting: "equal" | "mvo"`(default equal,不破坏 ensemble baseline);
  `mvo` wire `cvxpy + sklearn.covariance.LedoitWolf`,120d 回溯,单股上限 3%,long-only。
  新依赖均 pure-Python wheel。
- **验证**:全市场 top-K,equal vs mvo,P0-3 判定。

### P1-2 算子级 mask 传播(非 F1 测过的粗糙版)⭐
- **关键区分**:循环里 F1 测的是"标签层剔除涨停"——LOST(丢动量正样本)。综述论文 B 说的是
  在 `ts_corr / ts_rank / decay_linear` **算子内部**屏蔽不可执行价(涨跌停/停牌),防止它们污染
  后续 5-20 个 rolling 窗口。**两件不同的事**;论文 B 报 +0.44 Sharpe,本项目**未实装**。
- **改动**:`ops.py` 的 `ts_*` / `rank` / `decay_linear` 接可选 `mask`,聚合前置 NaN;default
  `mask=None` 退化旧行为(`docs/superpowers/specs/2026-05-31-tradability-mask-design.md` 的
  `mask_price` 算子侧,循环未做完整版)。
- **验证**:Layer B 先筛(IC 层面 mask 效果直接),通过再 Layer D。

### P1-3 基本面因子进 selection
- **依据**:PE/PB/ROE/size 已实装(`factors/fundamentals.py`),但 A1–A5 因子选择**全是技术/量价**,
  **从没把基本面放进 selection 测过**。综述论文 B 表 6:多样性 > 单家族质量(9→213 因子 +0.5 Sharpe)。
- **前置**:需 baostock 财务缓存(`fundamentals_*.parquet`);baostock 账号当前被黑名单(L1),
  确认缓存是否已存在 / 是否需用户解封或加 akshare 路径。
- **改动**:`factors analyze`(含 fundamentals)→ `pick-by-ic` 重选 → AB vs 现 selection。
- **验证**:全市场 top-K,P0-3 判定。

---

## Phase P2 — 次级机制(P1 有产出后再排)

- **P2-1 AdjMSE 方向感知损失**(LGB 路径,γ=0.1):论文 B +0.27 Sharpe。LGB 因过拟合被弃,
  AdjMSE 是潜在翻盘点(`objective=callable`,sign 错误 11× 梯度)。
- **P2-2 GBM 数据增强**(21 日块重采样):本质正则化,小股池过拟合的解药,论文 B +0.19 Sharpe。
- **P2-3 风格暴露报告卡片**(SMB/HML/动量/波动率 β):不提收益,但让"alpha 还是 beta"可见,
  防止下一轮再被 size/行业 beta 当 alpha 误导(这正是 B1/B2 中性化结论可疑的根源)。

---

## 进入 P1/P2 后才用的 /loop prompt(模板)

> P0 完成后,P1/P2 每个方向是独立的"改一处 → 全市场 AB → 判定 → 采纳/驳回 → 记 WORKLOG"。
> 那时才适合自驱循环。prompt 如下(self-paced,无固定 interval):

```
/loop 按 docs/improvement_loop/NEXT_CYCLE_PLAN.md 的 P1/P2 backlog 逐方向自驱改进。每轮:
1. 读 BACKLOG.md + NEXT_CYCLE_PLAN.md,选下一个 TODO 方向(P1 优先于 P2)。
2. 在 _FULLMKT_BASE.yaml 上建两 arm(只差被测维度),先跑 Layer B(layer_b_direct.py)高功率筛;
   通过再跑 Layer D(portfolio-ab 全市场)。
3. 跑 analysis/ab_significance.py --full-market 出 bootstrap CI + 子段 + validity。
4. 按 P0-3 判定:点 ΔSharpe≥+0.10 且 CI 排除 0 且子段符号一致 且两 arm valid → 采纳(改 config.yaml
   + commit),否则驳回(保留 baseline + commit)。
5. 把方向/假设/配置/指标/判定/commit 写进 WORKLOG.md,更新 BACKLOG.md 状态。
6. 所有 P1/P2 方向跑完或连续 N 个驳回且无新假设 → 停止循环并总结。
每个结论标注"全市场 top-K,硬化判定";不在单 ab_pool 上做选参(已证过拟合)。
```

**注意**:若 P0-1 runner 仍 hang 未修,Layer D 不可用,prompt 第 2 步只能停在 Layer B——
那 loop 只能做 IC 层面的方向(weighter / 因子集 / mask),做不了组合 Sharpe 方向(MVO 权重)。
**所以 P0-1 是 P1-1(MVO)的硬前置。**

---

## 一页纸总结:从哪入手

1. **先 P0-1**(验证 15-yr runner,cache 已建,近零成本)→ 决定 Layer D 是否可用。
2. **P0-2/P0-3**(canonical 全市场 config + 判定器)→ 让 oracle 可信。
3. **再 P1-1 MVO+LW 权重**(组合侧 +0.5 Sharpe 的最大未碰杠杆)。
4. P1-2 算子级 mask、P1-3 基本面入 selection 并行候选。
5. 一切走全市场 top-K + 两池/bootstrap 硬化判定;**不再单池扫参**。
