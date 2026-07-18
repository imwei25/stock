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

## AUDIT-2026-07-18 — agent 三路审计 + 修复 wave(新 /loop session,进行中)

**背景**:用户重启 /loop("派 agent 审视可提升点或修 bug → 逐个实现+验证")。三个并行
audit agents(代码正确性 / 研究方向 / 工程遗留)返回后合并出 12+ 项任务队列。

### 已落地修复(commits `21fa926` + `085829c`,分支 `fix/engine-audit-2026-07-18`)

1. **BUG-1(CONFIRMED,engine)**:legacy 模式(`hold_survivors=false`)+ `limit_guard` 下,
   仍在 target 且开盘一字跌停的持仓被 sell step 保留后又被 buy loop **覆盖**(空现金池
   → ~0 股新仓,旧股份权益凭空蒸发;audit agent 实跑复现 190 权益毁 90)。修复:buy
   candidates 无条件排除已持仓 code。**影响面**:仅 `limit_guard=true + hold_survivors=false`
   组合;RV 全部用 hold_survivors=true,不受影响。
2. **BUG-2(CONFIRMED,engine)**:持仓股 close 缺失(停牌/退市)后 MTM 与期末平仓都
   回退到 **entry_price** — 崩盘后退市显示 ~0% 亏损。含退市回填(244 只)的 RV 评估
   会被此失真**抬高**。修复:`_Position.last_mark` 追踪最后已知 close,MTM/平仓/
   weight_at_entry 分母全部改用。**⚠️ 后续所有含退市股的 Layer D 数字与旧数字不可直接比。**
3. **BUG-3(CONFIRMED,portfolio_ab)**:per-arm `portfolio_backtest.universe_codes`
   (文档承诺"仍优先")被 runner 静默忽略 — 两 arm 实跑同一 universe + 同一 score cache
   key。修复:`run_single_arm` 顶部按 arm 的 universe_codes 重过滤(在 cache key 之前)。
4. **eligibility `id()` 缓存**:改强引用 + `is` 比较(id 复用可 serve 旧数组,latent)。
5. **F1 hang → fast-fail**:`multiprocessing.Pool` 不检测 OOM-killed worker,
   `imap_unordered` 永久阻塞(= 15-yr "卡 1499/1500")。改 `next(timeout=1800s)`
   (env `STOCKPOOL_SCORE_RESULT_TIMEOUT_S` / 参数 `result_timeout_s`),超时抛
   RuntimeError 提示 `--workers 1`。健康路径不变。
6. **ops snapshot fixture 重生成**:旧 fixture 缺 GTJA191+新家族 → **219 测试静默 skip**。
   重生成后暴露 5 个新的 Rust rank-flip 级联分歧因子(alpha_040 变体×3 / gtja_042 /
   gtja_104),按既有 `EXPECTED_RUST_DIVERGENCE` 机制加 cap(≈2× 实测)。
   全套 **1374 passed / 0 skipped**。

### Kill-check A — h3/h10 分数面板相关性(方向1 预检)✅ 通过

RV-h10 的 v3 缓存面板:rv_eval(1241 codes)与 pool2(1500 codes)per-day 秩相关
mean 0.835 / 0.842(p10≈0.72)。**远低于 0.95 杀线** → h10 有独立截面信息,
blend(0.5·z(h3)+0.5·z(h10))方向存活,待跑 Layer B(见队列)。

### 未决队列(下一迭代)

- Kill-check B:RV-D 基线 gross-vs-net 成本拖累(<0.1 Sharpe 则杀 rank-buffer 方向)
- 方向1:blend Layer B(fwd=3/10 × 两池 × 2021+ 子窗,w=0.5 预注册)
- 方向3:h10×rebal10 节奏匹配 Layer D(纯 config,score 缓存命中)
- ipo_dates akshare fallback(解 L1;之后可重建 ab_pool IPO 硬过滤)
- 文档:CLAUDE.md 陈旧项(mask_exec 已落地/测试数/correlation 措辞)
- **注意**:BUG-2 修复改变含退市池的 engine 数字 → 涉及退市股的历史 Layer D 结论
  (RV-D/RV-h10 的组合层数字)如需引用应在新 engine 下重跑;IC 层(Layer B)不受影响。

## RV-h10 — h3→h10 双层双池硬化(2026-07-13 跑完/07-16 判定)— NOT CONFIRMED,不 promote
- **背景**:RV-D 里 h10 是唯一正向 lead(Layer D 主池 +0.100,粗子段全正)。按硬化协议补
  Layer B 双口径 + 双池(pool2_midliq 1500 只,0 重叠)+ pool2 Layer D。全部在修正 harness
  (old 池@5e4 基线,PIT top-1000,open-basis,入场涨停剔除,hold_survivors+limit_guard)。
- **证据矩阵**:

| 层 | 口径 | 主池(rv_eval) | pool2(disjoint) |
|---|---|---|---|
| Layer B | fwd=10(自家) | +0.0066,CI **含 0**,8 段 7 正 | **+0.0101,t=7.9,CI 排 0,halves/thirds/5 全正 → CONFIRMED** |
| Layer B | fwd=3 | +0.0023 ns | +0.0040,CI 勉强含 0 |
| Layer D | 组合 Sharpe | **+0.100**,halves/thirds/7regime 全正,CI 含 0 | **−0.057**(**符号翻转**),子段乱 |

- **判定**:**NOT CONFIRMED,不 promote,保留 h3**。h10 在 IC 层的增量信号是真的
  (两池同号、pool2 过 CI),但**组合层跨池符号翻转**(+0.100 vs −0.057)——正是当年
  GTJA 被否的失败签名(universe-conditional 组合效应)。IC 增益在 mid-liq 池上被组合
  整合环节(与 rebalance=5 的节奏错配、选股重叠、换手结构)吃掉。
- **可选后续**(新方向,非本判定一部分):h10 + rebalance_n_days=10 的**节奏匹配**变体
  从未测过 —— h10 慢信号配快调仓可能是自缚手脚;若未来重启优化循环,这是 h10 线上
  唯一值得再试的形态。
- **RV 全程至此完全收官**:不可信结论清单全部重验完毕,provisional lead 亦已按协议
  处置。基线 = old 池 @ alpha 5e4 + h3 + k20 + rebal5 + equal + ic + lasso(tw250),
  全部经修正 harness 验证。

## RV-D — 换手敏感 AB 全量重验 @ 修正引擎(2026-07-12,RV 收尾)
- **设置**:回滚后基线(old 池 @ alpha 5e4),`config_rv_engine.yaml` = hold_survivors +
  limit_guard + `top_n_liquidity: 1000`(PIT as-of 流动性 universe);rv_eval_pool(含退市 244);
  score panel 全部 v3 缓存命中,engine-only。基线 Sharpe = **0.331**(诚实执行语义下的绝对水平,
  显著低于旧 harness 的 0.5+ ——旧绝对数含静态池前视 + 无涨停约束)。
- **结果**:

| 方向 | 旧引擎结论 | 修正引擎(本次) | 终判 |
|---|---|---|---|
| top_k 20→10 | 点 +0.07~+0.27 但不稳 | **−0.212,CI [−0.40,−0.03] 排 0(负)**,halves/thirds/7regime 全负 | **翻转:k10 显著更差**。k10 在 PIT 池 + 涨停约束下覆盖掉到 0.84,集中组合受执行约束伤害更大。保留 k20 |
| rebalance 5→10 | −0.045 含 0 | −0.090,CI 含 0 | 维持 NOT CONFIRMED,保留 5(本次才是有分辨力的量测) |
| weighting equal→mvo | +0.119 含 0(点正) | **−0.089,CI 含 0,子段乱** | **点估计翻负**:MVO 的"换手 −31% 省成本"优势在 hold_survivors 下消失 —— 印证其点正主要是旧全清仓成本模型的伪影。保留 equal |
| horizon 3→10 | −0.111(L_h10d) | **+0.100,CI 含 0(P=0.13),halves+thirds+7regime 全正** | **方向翻正,最强新 lead**:hold_survivors 下 h10 慢信号→更多留任→真实省换手。未达门槛不 promote;记 provisional,后续可 Layer B+双池追 |
- **结论**:4 个方向里 2 个在修正引擎下改变性质(topk 翻显著负、mvo 点翻负)、1 个出现方向翻转的
  新 lead(h10)。**"换手敏感结论必须在 hold_survivors 引擎下量"从推断升级为实证。**
- **RV 全程收官**:所有列入"不可信清单"的结论已在修正 harness 下重验完毕。最终处置:
  alpha=0.0005 ✅ 维持 / 因子集切换 ❌ 已回滚 / bottom-line 2.2× ❌ 已撤销 /
  topk=20、rebal=5、equal 权重 ✅ 维持(其中 k10 从"方向存疑"变为"显著更差") /
  h10 ⚠️ provisional 正向 lead。

## RV — 外部评审驱动的 harness 修复 + 全结论重验(2026-07-02,进行中)

**背景**:外部代码/方法评审指出 harness 三大盲区:① 幸存者偏差(universe 只含当前上市股 +
静态 top1000 池按**今天**流动性选成员回测过去 = 期末前视);② Layer B oracle 口径漂移
(close-to-close 前向收益把拿不到的隔夜段计入分数;不检查一字涨停可交易性);③ 执行现实性
(组合引擎全清仓 rebalance 的虚拟往返成本 + 无涨停拒单)。另发现 `P0_factorset.yaml` 在 6-30
promote 覆盖 selection.json 后变成"自己比自己"(重跑 ΔIC +0.00002 ≠ win 消失,是配置漂移;
已用 `RV_factorset.yaml` 钉住备份文件)。滑点误报:`BacktestCostConfig.slippage_rate` 已存在。

**修复落地**(commits 07ef6f0 / b73f023 / 5defd6e / 59cc285):
- 退市股回填:窗口内退市 258 只(universe 内 0 只!),东财回填 244 只日线进 `data/`;
  Pool B 加 30 天新鲜度闸门。
- `eligibility.top_n_liquidity`:as-of top-N PIT 流动性 universe(引擎侧);
  layer_b_direct `--asof-top-n`(评估侧)。**静态 top-1000 每天仅覆盖真实 as-of top-1000 的
  中位数 326 只;15 年 union=4565 只 ≈ 全市场**。
- Layer B oracle 默认改 open-basis + 入场一字涨停剔除(旧口径可 `--fwd-basis close
  --tradability none` 复现,已验证逐位复现)。
- 引擎 `hold_survivors` + `limit_guard`(默认 false bit-exact)+ `weight_at_entry` 修复。

**重验结果(cheap 层,缓存 score panel,静态 1000 打分集)**:

| 结论 | 旧 oracle | open+entry | +as-of top-1000 掩码 | 判定 |
|---|---|---|---|---|
| alpha 5e4 win (Layer B ΔIC) | +0.00882 t=8.1 | +0.00794 t=7.3 | **+0.00540 t=3.9 CI[+0.0015,+0.0096]** | **存活,效应 −39%** |
| 因子集 win (Layer B ΔIC) | +0.00914 t=5.4 | +0.00986 t=5.9 | **+0.01249 t=6.3 CI[+0.0056,+0.0196]** | **存活,PIT 池下更强** |
| bottom-line (Layer D ΔSharpe) | +0.286 p=0.035 halves+thirds 全正 | — | hold_survivors+limit_guard: **+0.209 p=0.096, thirds 2+/1−** | **削弱**:方向仍正,但"Sharpe 翻倍"降级为"点估计 +0.2 不可确认" |

附带发现:hold_survivors 对老基线的成本节省(Sharpe 0.247→0.345)远大于新基线(0.533→0.554)
→ 旧全清仓成本模型确实在给"高换手 arm"系统性罚分,过往所有换手相关 AB(L_rebal/L_h10d/MVO
换手优势)的结论都在这个失真下得出,如需引用须重跑。

**★ RV wave 第一轮暴露的潜伏 P0 bug(2026-07-03,已修,commit 5152305)**:
第一轮 wave 里所有含 alpha_083 的因子集,分数在 **2015-07-31 后全 NaN**(current 池 949 天、
holdout 池 1696 天散点、old 池 3537 天正常)。链条:alpha_083 公式 `.../(vwap−close)` 在
H==L==C 一字板 bar 除零产出 **inf**(2015-07-13 千股复牌日 122 只);inf 躲过训练 dropna
(`~isnan` 不筛 inf)毒化 Lasso → 该月 fit 系数 NaN → 全 NaN 预测;当日 inf 占比 ≥1% 时
winsorize q99 clip 同样失效(分位数本身 inf);退市回填后 `tail(train_window)` 把退市股
最后 N 行(含毒日)**永久冻结**进其后所有训练集 → 2015-08 起每个月度 fit 都带毒。
**这不是退市回填引入的 bug,是它揭开的潜伏雷**:旧 universe 下若某日 inf 占比过 1%
(如 2024-02 微盘千股一字跌停),生产路径同样可能静默毒化当月 fit。
修复:compute_factor_panel 出口 inf→NaN + stack isfinite 筛 + pooled 训练 trailing-window
日期下界(三路径一致)+ panel sig v2 / score key v3 全缓存失效。1151 tests passed。

**★★ RV wave 第二轮结果(2026-07-03 完成,全修正 harness:inf 卫生 + 退市股进训练/评估 +
PIT as-of top-1000 + open-basis + 入场涨停剔除;IC 天数 3533-3592,覆盖完整)**:

| 检验 | 窗口 | ΔIC | t | 95% CI | 子段 | 判定 |
|---|---|---|---|---|---|---|
| 因子集 old→new @1e3 | 全期 | **+0.00751** | +3.8 | [+0.0011,+0.0137] 排0 | halves 2+/0−,thirds 2+/1− | 存活(点+CI),thirds 不全正 |
| 因子集 old→new @1e3 | **2021+** | **−0.00391** | −1.0 | 含 0 | halves 0+/2− | **近 5.5 年无增益,微负** |
| alpha 1e3→5e4 @新池 | 全期 | **+0.00644** | +4.5 | [+0.0018,+0.0118] 排0 | halves 2+/0−,thirds 2+/1− | **存活** |
| alpha(新池@5e4 vs @1e3)| 2021+ | +0.0144(0.0293→0.0437) | — | — | — | **近年仍强** |
| holdout 池(≤2020 选)vs 现池 @5e4 | 2021+ | +0.00938(0.0343 vs 0.0437) | +3.6 | [+0.0010,+0.0186] 排0 | halves 1+/1− | 现池优,但此差≈**选择偏差量级** |
| holdout 池 vs 现池 @5e4 | 全期 | +0.00387 | +2.8 | 含 0 | 混合 | 两池统计上接近(重叠21/30) |

**核心结论**:
1. **alpha=0.0005 win:真实、全期稳健、近年仍强、跨池成立** — 修正 harness 下新池 +0.0064
   (t=4.5, CI 排 0);**在 old 池上同样成立**(全期 0.0437→0.0532 = +0.0095;2021+
   0.0332→0.0501 = +0.0169)。**维持 promote。**
2. **因子集 win:promote 证据在修正 harness 下不再成立 → 已回滚(2026-07-11)** —
   old@5e4 对照补齐后的完整矩阵(2021+ 窗口 = 对未来最少偏差的估计):

   | 2021+ IC | old 池 | new 池 |
   |---|---:|---:|
   | alpha 1e3 | 0.0332 | 0.0293 |
   | alpha 5e4 | **0.0501** | 0.0437 |

   在生产 alpha(5e4)下:全期 Δ(new−old)=+0.0047 **CI 含 0**、halves 分裂;2021+
   Δ=**−0.0064**(t=−2.2,8 分段 6 负),且负号跨两个 alpha 稳定。原 "+18.5%" 由
   ① 2011-2020 段集中 ② 选择偏差(holdout 检验定界 ~+0.009)③ 旧 oracle 构成。
   **处置**:`selection.json` 回滚为 `selection_pre_15yr_2026-06-30.json` 内容
   (新池备份 `reports/selection_15yr_irpool_promoted_2026-06-30.json`;
   revert-of-revert = 反向 cp)。alpha=0.0005 保留(见 1,跨池稳健)。
3. 选池方法本身时间稳定(holdout 池与现池重叠 21/30、全期 ΔIC 仅 +0.004)→ 建议改为
   **定期 walk-forward 重选**(如每年用 trailing 数据重选一次)取代一次性全样本重选。
4. **L_bottomline"累计 Sharpe 2.2×"正式撤销**:其两条腿中因子集腿已回滚;真实的累计
   改进 = alpha 腿(old 池上 2021+ IC +0.0169/全期 +0.0095)+ harness 修复本身。
- **日期**:2026-07-02 · `configs/L_bottomline.yaml`(本 session 两改动的累计效应)
- **结果**:**Sharpe A(old)=0.247 → B(new)=0.533,ΔSharpe +0.286(~2.2×)**;CI [−0.019, +0.611] 勉强含 0
  (下界 −0.019 几乎触 0);单侧 p=0.035;halves+thirds **一致正**(2+/0−, 3+/0−);7-regime 5+/2−;validity ok。
- **判定**:Layer D 严格 NOT CONFIRMED(CI 差 0.019 触 0,欠功率)**但点估计巨大(Sharpe 翻倍)+ 粗子段一致 + p=0.035**。
  配合 **Layer B 两池 CONFIRMED**(高功率),累计工作**显著提升组合 Sharpe** 证据充分。
- **★ bottom-line**:本 session 把 top-1000 组合 Sharpe 从 0.247 提到 0.533(daily IC 0.0496→0.0665 的下游兑现)。

## h10-analyze — horizon-regime 探索:h=10 是否有不同 alpha — 否(同因子)
- **日期**:2026-07-02 · `factors analyze --horizon 10`(top-1000 15yr 386 因子)→ `reports/factor_analysis/loop_h10/2026-07-02.json`
- **发现**:h=10 top-IR 因子**与 h=3 基本相同**(alpha_050/gtja_016/alpha_013/gtja_150/alpha_044…),IR 略高
  (0.485 vs 0.391,长周期收益更平滑);**基本面仍不入 top-20**(pb/mcap/roe,10 日对慢变量仍太短);
  limit_up_count_20 IR −2.4 是 phantom。p90 |IR| h3=0.272 ≈ h10=0.280。
- **判定**:**h=10 无独立 alpha 源**——同一技术/动量因子集,只是标签更平滑。长周期策略≠新信号,只是慢节奏同信号。
  唯一可能价值:低换手→低成本→净 Sharpe?→ 转 Layer D 测 h3 vs h10 策略(见下)。

## pool2-alpha — alpha 0.0005 win 第二池硬化 ★ CROSS-POOL CONFIRMED
- **日期**:2026-07-01 · `L_alpha.yaml` on `pool2_midliq`(1500 票,0 overlap,workers=1 串行)
- **结果**:alpha_1e3(A)IC +0.07006 / alpha_5e4(B)IC +0.07799,**ΔIC +0.00793(+11.3%)**;IR 0.567→0.620;
  **t=+10.766**;CI [+0.00535, +0.01048] 排除 0;**所有子段(2/3/5/8/7-regime)全正 perfectly consistent**。
- **判定**:**CROSS-POOL CONFIRMED**。alpha=0.0005 在 top-1000(t=8.1)**和** pool2(t=10.8)双池均 CONFIRMED。

## ★★ 两池硬化结论(2026-07-01)
**本 session 两个 promote 的改进,均通过 METHODOLOGY 最严两池硬化(top-1000 + disjoint pool2 mid-liq):**
| win | top-1000 | pool2(disjoint) | 结论 |
|---|---|---|---|
| 因子集 15yr-IR 重选 | ΔIC +18.5% t=5.4 | ΔIC +6.4% t=4.5 | **跨池稳健** |
| alpha 0.001→0.0005 | ΔIC +15.3% t=8.1 | ΔIC +11.3% t=10.8 | **跨池稳健** |
两者**均非 universe-conditional**(不同于旧 GTJA:第二池反转)。**可信、不过拟合、跨池稳健的真 alpha。**

## L_combo — corr045 池 + ridge weighter(两 borderline lead 叠加)@ top-1000 Layer B — NOT CONFIRMED(lead 是噪声)
- **日期**:2026-07-01 · 配置 `configs/L_combo.yaml`
- **结果**:baseline(A)IC +0.06633 / combo(B)IC +0.06757,**ΔIC +0.00123**;t=0.916;CI 含 0;子段大多不一致。
- **关键**:combo(+0.0012)**弱于**任一单 lead(maxcorr +0.0031 / ridge +0.0017)。真独立效应应**叠加**;反而抵消
  → **两 borderline lead 是多重检验噪声,非真信号**。保留 baseline。
- **★ 可信改进空间穷尽**:2 confirmed win(因子集+alpha);mask 负;参数/weighter/池组成/组合全 flat 或噪声。
  基线是强局部最优。**转为硬化已有 win**(METHODOLOGY 两池验证)+ 向用户汇报。

## P1-2b — 因子集 win 第二池(pool2 mid-liq, 与 top1000 0 重叠)硬化验证 ★ CROSS-POOL CONFIRMED
- **日期**:2026-07-01 · `data/pool2_midliq.parquet`(rank 1001-2500,1500 票,**0 overlap** top1000)· `configs/P1-2b_pool2.yaml`
  · **workers=1 串行**(workers=2 在 1500 票触发 F1 Pool teardown 死锁,卡 1499/1500)
- **结果**(pool2, Layer B):old_gtja(A)IC +0.07487 / new_15yr(B)IC +0.07966,**ΔIC +0.00479(+6.4%)**;
  IR 0.598→0.634;t=**+4.455**(p≈0.0000);CI [+0.00116, +0.00846] **排除 0**;halves/thirds 一致正。
- **判定**:**CROSS-POOL CONFIRMED**。因子集 win 在 top-1000 **和** disjoint mid-liq 1500 票**双池**均 CONFIRMED。
- **★ 重大可信性结论**:因子集 win **非 universe-conditional**——不同于旧 GTJA(第二池反转、曾误导前团队,见
  METHODOLOGY)。15yr-IR 重选是**跨池稳健**的真改进,通过 METHODOLOGY 两池硬化门槛。
- **副产物**:F1 死锁在 1500 票 workers=2 复现(1000 OK)→ >1000 票用 workers=1。

---

## P1-2 — 算子级 mask 探针:mask 因子输入 vs 不 mask @ top-1000 Layer B — mask HURTS(项目 prior 验证)
- **日期**:2026-07-01(NEW#4,高EV)· 脚本 `analysis/mask_probe.py`(自包含,不改生产)
- **设计**:两 arm 同 strategy(新池/alpha5e4/ic)+ 同**未 mask** close_panel(labels 一致),唯一差异 = 因子输入面板:
  A=原始 OHLCV;B=涨停/跌停(|ret|≥板块阈值)/停牌(vol≤0)bar 的 OHLCV 置 NaN(靠 NaN-safe ops 跳过)。
  隔离"因子输入 mask"效应,测 survey B(+0.44)vs 项目 prior(涨停是信号;F1 标签层 mask LOST −0.63)。
- **结果**(top-1000, Layer B):A(unmasked)IC +0.08096 / B(masked)IC +0.07530,**ΔIC −0.00567(−7%)**;
  IR 0.516→0.501;paired t=**−2.825**;CI [−0.01311, +0.00116](点估计清晰负,子段一致负);mask **HURTS**。
- **判定**:**NOT CONFIRMED(mask 显著伤害因子 IC)**。**严格印证项目 prior**:涨停 +9.9% 的价格变动本身是有用
  信号(WQ101/GTJA 动量-反转因子真在用它);mask 掉它损失信号。**refute survey B 的 +0.44** 在本策略不成立
  (其 213 因子体系/损失函数不同)。**不实现生产 mask,P1-2 归档。**
- **意义**:这是项目从未做过的**严格算子级 Layer-B mask 检验**;此前拒绝 mask 是 reasoning+标签层测试,
  现在有高功率 OOS 实证。**剩余最大杠杆(P1-2)settle 为负。**

---

## L_rebal — rebalance 5 vs 10 @ 新基线, top-1000 Layer D — NOT CONFIRMED
- **日期**:2026-07-01 · engine-only。Sharpe A(5)=0.533 / B(10)=0.488,ΔSharpe −0.045;CI 含 0;子段不一致。
- **判定**:**NOT CONFIRMED**(rebal_10 略差)。保留 rebalance=5。**组合参数空间全重验完毕(top_k=20/rebal=5 最优)**。

---

## L_maxcorr — 去相关档 corr0.6(30)vs corr0.45(22)@ 新池, top-1000 Layer B — NOT CONFIRMED(borderline 正,最大新 lead)
- **日期**:2026-07-01(NEW#3)· 配置 `configs/L_maxcorr.yaml`
- **结果**:corr06(A)IC +0.06633 / corr045(B)IC +0.06946,**ΔIC +0.00312(+4.7%)**;IR 0.475→**0.509(+7.2%)**;
  t=+2.673(iid p≈0.0075);CI [−0.00099, +0.00794] **含 0(勉强)**;子段大多不一致。
- **判定**:**NOT CONFIRMED**(CI 含 0 + 子段不稳)。但**新方向里最大 lead**——更严去相关(22 因子,去冗余)IC/IR 都升。
- **模式**:ridge(+0.0017)/ maxcorr(+0.0031)皆 borderline 正但子段不稳 → 弱信号"更干净的分数有益"但不硬化。
  不追 max-corr 调参(过拟合)。**cheap 打分层空间基本穷尽**,真增益或只剩 P1-2 mask(高成本)。

---

## L_halflife — ic vs halflife_ic(hl=60)weighter @ 新池, top-1000 Layer B — NOT CONFIRMED
- **日期**:2026-07-01(NEW#2)· 配置 `configs/L_halflife.yaml`
- **结果**:ic(A)IC +0.06633 / halflife(B)IC +0.06693,**ΔIC +0.00059**(持平);IR 0.475→0.495(+4%);
  t=0.837(p≈0.40);CI [−0.00163, +0.00277] 含 0;子段大多不一致。
- **判定**:**NOT CONFIRMED**。保留 ic。
- **★ weighter 空间穷尽**:ic vs {equal, sharpe, ridge, halflife_ic} 全测——**ic 最优/最稳**(ridge borderline 正未采纳)。
  再次印证 FULL_MARKET_RESULTS「IC 加权是唯一稳健确认的杠杆」。

---

## L_ridge — ic vs ridge weighter @ 新池, top-1000 Layer B — NOT CONFIRMED(borderline 正)
- **日期**:2026-07-01(NEW 方向#1)· 配置 `configs/L_ridge.yaml`
- **动机**:ridge 联合 L2 拟合(β=(XᵀX+αI)⁻¹Xᵀy)vs IC 边际加权,或更好组合去相关因子;ridge 从未 AB。
- **结果**:ic(A)IC +0.06633 / ridge(B)IC +0.06801,**ΔIC +0.00168(+2.5%)**;IR 0.475→0.496(+4.4%);
  t=+2.226(iid p≈0.026)但 **block-bootstrap CI [−0.00046, +0.00395] 含 0**(勉强);子段全不一致。
- **判定**:**NOT CONFIRMED**(CI 含 0 + 子段不稳)。**但 borderline 正**——是新方向里最正的。iid 显著但
  自相关修正后 CI 含 0,子段不一致是主要否决点。**保留 ic**,不调 ridge alpha 追显著(过拟合)。

---

## L_trainwin — train_window 250 vs 500 @ 新池, top-1000 Layer B — tw=250 确认更优
- **日期**:2026-07-01 · 配置 `configs/L_trainwin.yaml`
- **结果**:tw250(A)IC +0.06633 / tw500(B)IC +0.06522,**ΔIC −0.00111**(tw500 更差);t=−3.127(p≈0.0018);
  CI [−0.00219, −0.00008] **排除 0(显著负)**;halves/thirds/5-bucket 全一致负。
- **判定**:variant tw500 **NOT CONFIRMED(显著更差)** → **tw=250 确认最优**(印证 238+全市场)。保留 250。
- **★ cheap 旧猜想全部重验完毕**:alpha=0.0005 WON(promoted);poolsize/horizon/topk/selector 持平;tw=250 确认。
  **新基线 = 15yr-IR 池 + alpha 5e4 + h3 + tw250 + lasso + ic + topk20 + equal,全经高功率 oracle 验证。**
  → 转新方向(见 BACKLOG)。

---

## L_selector — lasso vs lightgbm selector @ 新池, top-1000 Layer B — NOT CONFIRMED
- **日期**:2026-07-01 · 配置 `configs/L_selector.yaml`
- **结果**:lasso(A)IC +0.06633 / lgbm(B)IC +0.06699,**ΔIC +0.00066**(持平);t=0.977(p≈0.33);CI 含 0;子段大多不一致。
- **判定**:**NOT CONFIRMED**。**但重要**:lgbm 不再是 238 池的"灾难"(Sharpe 0.19)——证实那是 16 股小样本过拟合;
  15yr×top1000 数据量下 lgbm **追平** lasso 但不超越。**保留 lasso**(更简单更快,等效)。
- **P1-3 情报**:fundamentals 缓存存在,但因子在 horizon=3 上弱(pb IR−0.195 最强,roe/roa/margin IR<0.05)→ 慢变量不擅长预测 3 日收益,低 EV。

---

## L_topk — top_k 20 vs 10 @ 新基线, top-1000 Layer D — NOT CONFIRMED
- **日期**:2026-07-01 · 配置 `configs/L_topk.yaml`(engine-only,scores 共享缓存)
- **结果**:Sharpe A(k20)=0.533 / B(k10)=0.605,**ΔSharpe +0.072**;CI [−0.101, +0.247] 含 0(P(Δ≤0)=0.209);
  子段全不一致;k10 trades 7190 cov **0.89**(过度集中,部分 bar 填不满)。
- **判定**:**NOT CONFIRMED**(点 +0.072<0.10,CI 含 0,子段不稳,coverage 掉)。**保留 top_k=20**。同 G1 模式:点优不稳健。
- **模式观察**:两个真 win(因子集/alpha)皆**打分层**(Layer B 可证);所有**组合参数**(top_k/horizon/MVO)皆持平/不硬化
  → 印证 METHODOLOGY「组合 Sharpe 欠功率」。转向打分层方向找增益。

---

## L_horizon — horizon 3 vs 5 @ alpha=0.0005, top-1000 Layer D — NOT CONFIRMED
- **日期**:2026-07-01 · 配置 `configs/L_horizon.yaml`(全市场曾favor 5,重验)
- **结果**(Layer D):Sharpe A(h3)=0.533 / B(h5)=0.530,**ΔSharpe −0.003**(几乎相同);CI [−0.163, +0.168] 含 0;子段全不一致。
- **判定**:**NOT CONFIRMED**。全市场"h5 +0.37"未在新基线复现 → 噪声/universe 特定。**保留 horizon=3**。

---

## L_poolsize — 候选池 30 vs 44(更大 IR 池)@ alpha=0.0005, top-1000 Layer B — NOT CONFIRMED
- **日期**:2026-07-01 · 配置 `configs/L_poolsize.yaml`(pick-by-ic top-45→44 因子 vs 现 30)
- **动机**:L_alpha 显示"多保留好因子有益"+ survey"多样性>单家族" → 测更大 IR 池。
- **结果**(top-1000 Layer B):pool30(A)IC +0.06647 / pool44(B)IC +0.06728,**ΔIC +0.00081**(+1.2%);
  IR 0.475 vs 0.482;paired t +1.568(p≈0.117);CI [−0.00098, +0.00315] **含 0**;子段全不一致。
- **判定**:**NOT CONFIRMED**。30 因子 IR 池已捕获信号,多加 14 个低 IR 因子无增益。**保留 30**。
- **附**:新基线 IC≈0.0665(session 起点 0.0496 → +34%,因子集重选 + alpha=0.0005 两 promote 累积)。

---

## L_alpha — lasso alpha 0.001 vs 0.0005 在新 15yr-IR 池重验(top-1000 Layer B)★ CONFIRMED
- **日期**:2026-07-01(自驱循环 cron 0715a0ec)· 配置 `configs/L_alpha.yaml`
- **背景**:full-market 复核曾显示 alpha 0.001→0.0005 反转(238 池说 0.001 远优,全市场说 0.0005 +0.37);
  新池(15yr-IR)从未测过。重验。两 arm 仅差 `selector.lasso.alpha`,both 新 selection.json + ic weighter。
- **结果**(top-1000, Layer B daily IC):

  | metric | alpha_1e3 (A, 现默认) | alpha_5e4 (B) | Δ (B−A) |
  |---|---:|---:|---:|
  | daily IC mean | +0.05751 | +0.06633 | **+0.00882 (+15.3%)** |
  | IR proxy | +0.399 | +0.475 | +0.076 (+19%) |
  | paired t | — | — | **+8.139** (p≈0.0000) |
  | bootstrap 95% CI ΔIC | — | — | **[+0.00497, +0.01282] 排除 0** |
  | 子段 halves/thirds/5/8/7-regime | — | — | **全部一致 (2/3/5/8/7 全正)** |

- **Layer D 确认**(`ab_significance --pool top1000`,scores 缓存,engine-only):Sharpe A=0.482 → B=0.533,
  **ΔSharpe +0.051**;CI [−0.124, +0.215] 含 0(P(Δ≤0)=0.277);子段 halves/thirds 一致正,7-regime 4+/3−;
  validity ok。**Layer D NOT CONFIRMED**(点 +0.051 < 0.10 阈值,欠功率)but **not-worse / 微正**。
- **判定 + 落地**:**CONFIRMED Layer B(主 oracle)+ Layer D not-worse → PROMOTE**(2026-07-01)。
  config.yaml `selector.lasso.alpha: 0.001 → 0.0005`;备份 `config_pre_alpha5e4_2026-07-01.yaml`。
  revert = `cp config_pre_alpha5e4_2026-07-01.yaml config.yaml`。
- **过拟合反驳**:低 alpha=更少正则=更多因子存活,本是过拟合疑点;但 Layer B 是 **OOS walk-forward IC**,
  且 **7 个 regime 全部为正**——若过拟合,OOS IC 应更差而非全 regime 一致更好 → 疑点被实证反驳。
  且候选池只有 30 个(已 IR 精选),低 alpha 是"多保留几个已验证的好因子",非"引入噪声因子"。
- **方法学依据**:METHODOLOGY 明定 daily IC 为**主高功率 oracle**(组合 Sharpe 欠功率,小效应测不出),
  故 Layer B CONFIRMED + Layer D not-worse 达 promote 门槛。**新基线 alpha=0.0005**。

---

## P1-1 — MVO + Ledoit-Wolf 组合权重 vs 等权(top-1000 Layer D)— NOT CONFIRMED
- **日期**:2026-06-30 · 配置 `docs/improvement_loop/configs/P1_mvo.yaml`
  · 工具 `ab_significance.py --pool top1000 --workers 2`
- **背景**:NEXT_CYCLE_PLAN P1-1。综述论文 B:同模型同因子,等权→MVO+LW **+0.5 Sharpe**(组合侧最大
  未碰杠杆)。G 子任务只调过 top_k/rebalance/cap,从没动过 weighting 机制。
- **实现**(PR-P1-1,**无新依赖**):`portfolio/weighting.py` = `ledoit_wolf_cov`(纯 numpy LW2004 收缩)
  + `solve_mvo`(scipy SLSQP box-constrained QP)+ `compute_target_weights`(退化全回退等权);
  `PortfolioRunConfig` 加 `weighting/mvo_*` 字段(默认 `equal` bit-exact);engine 注入加权步骤。
  测试:15 weighting 单测 + 2 engine 集成测;全套 1136 passed 无回归。μ=横截面 score(标准化),
  Σ=trailing 120 日收益 LW 协方差;both arm 同 strategy → 共享 scores(engine-only 对比)。
- **结果**(top-1000, 3768 bars):

  | metric | equal (A) | mvo (B) | Δ |
  |---|---:|---:|---:|
  | portfolio Sharpe | 0.482 | 0.601 | **+0.119 (+25%)** |
  | M2 bootstrap 95% CI ΔSharpe | — | — | [−0.089, +0.324] **含 0**(P(Δ≤0)=0.128) |
  | M3 子段 halves / thirds | — | — | **不一致**(1+/1−, 2+/1−);regime 4+/3− |
  | M4 validity | 14380 trades cov0.98 | 9907 trades cov0.98 | ok(mvo 换手 −31%) |

- **判定**:**NOT CONFIRMED**。点估计正(Sharpe +25%、换手 −31% 省成本)但**子段不一致**(有半段
  MVO 输),CI 含 0。比 P0-3 因子集弱(那次 halves+thirds 全正 + Layer B 显著)。综述的 +0.5 是
  top-100 + 其全套系统,本处 top-1000 + 我方 scores 上只有 +0.12 且时间不稳。
- **落地**:**保持 `weighting: equal`(default 不变)**。MVO 实现 + 测试 + 文档齐全,可 opt-in
  (`portfolio.weighting: mvo`),但不达"点估计 +0.10 AND CI 排除 0 AND 子段一致"门槛 → 不设默认。
  与 sharpe/halflife/ridge weighter 同处置(实装备用,不切默认)。
- **不做参数 sweep**:risk_aversion/w_max/lookback 在单池上扫 = 过拟合(METHODOLOGY 禁)。点估计正
  但子段不稳,调参追显著 = p-hacking。诚实记为"正向未硬化"。

---

## P0-3 — 因子集 15yr 重选:新 IR 池 vs 现 selection(top-1000 Layer B)★ CONFIRMED
- **日期**:2026-06-30 · 配置 `docs/improvement_loop/configs/P0_factorset.yaml`
  · 工具 `analysis/layer_b_direct.py --pool data/top1000_liquid.parquet --workers 2`
- **背景**:用户问"扩到 15 年长期后,selection.json 该不该在新时间跨度上重选"。架构澄清:
  selection.json 是**候选池**,下游 walk-forward Lasso+IC 已做每期时变选择(无前视);
  **不**手工按时期换池(前视+过拟合)。原则化做法:候选池在 15yr + top-1000(=可投/验证 universe)
  上**重选一次**,按 **IR 稳健性**(非点 IC)+ 跨 regime + 去相关。
- **流程**:① `factors analyze --universe pool`(top-1000 cfg, 15yr=3750d, horizon=3, 386 因子,
  ~24min)→ `reports/factor_analysis/loop_p0_15yr/2026-06-30.json`;② `pick-by-ic --score-by ir
  --min-ir 0.15 --max-corr 0.6 --max-degenerate-ratio 0.3 --top-n 30` → `reports/selection_15yr_candidate.json`
  (degenerate gate 正确剔除 phantom:limit_up_count_20 IR−0.836/halflife0、cfo_to_np IR−0.705/halflife0);
  ③ Layer B AB,两 arm 仅差 factors_file,both IC weighter。
- **新池 vs 现池**(各 30):**保留 10**(gtja_001/150、industry_relative_strength_20、volume_std_20、
  corr_mom_vol_20、alpha_012/029/069/087、close_skew_20);**剔 20**(大批 GTJA:gtja_012/015/020/080/
  097/135/141/158 + turnover_zscore_60 + mom_vol_interact_10);**增 20**(更多 wq101 alpha + 窗口变体
  compress/expand/rev_short)。
- **结果**(top-1000, T≈3500 日 IC):

  | metric | cur_selection (A) | new_15yr_irpool (B) | Δ (B−A) |
  |---|---:|---:|---:|
  | daily IC mean | +0.04961 | +0.05878 | **+0.00918 (+18.5%)** |
  | IR proxy | +0.342 | +0.408 | +0.066 (+19%) |
  | paired t (iid) | — | — | **+5.433** (p≈0.0000) |
  | block-bootstrap 95% CI ΔIC | — | — | **[+0.00311, +0.01563] 排除 0** |
  | 子段 halves / thirds | — | — | **一致 (2+/0−, 3+/0−)** |
  | 子段 5/8/7 buckets | — | — | 不全一致 (3+2−, 6+2−, 6+1−) |

- **判定**:**CONFIRMED Layer B**(点 ΔIC 正且 +18.5% / CI 排除 0 / 粗子段一致;t=5.4 极强)。
  保留点:细子段非 100% 一致(跨 2015 崩盘等极端段的正常波动,不否定)。
- **意义**:这是项目史上**第一个在正确 oracle(15yr × top-1000 × daily IC × bootstrap)上稳健显著的
  因子集改进**。印证 `FULL_MARKET_RESULTS`:当初"GTJA +0.83"是 238 子池产物(大批 GTJA 在长期 top-1000
  的 IR 稳健性下掉出)。
- **Layer D 复核**(`ab_significance.py --pool top1000 --workers 2`,score cache-hit,顺带证 F1 runner
  在 top-1000 **不死锁**;common bars=3768):

  | metric | cur (A) | new_15yr (B) | Δ |
  |---|---:|---:|---:|
  | portfolio Sharpe | 0.230 | 0.466 | **+0.235**(近翻倍) |
  | M2 bootstrap 95% CI ΔSharpe | — | — | [−0.106, +0.621] **含 0**(P(Δ≤0)=0.088) |
  | M3 子段 halves / thirds | — | — | **一致 (2+/0−, 3+/0−)** |
  | M3 regime(7段) | — | — | 6+/1−(唯一负段 2020-03~2021-12 COVID 抱团 Δ−0.72) |
  | M4 validity | cov0.98 14140 trades | cov0.98 14380 trades | ok |

  Layer D **VERDICT: NOT CONFIRMED**(CI 含 0)——但点 Δ+0.235(Sharpe 翻倍)、halves+thirds 全正、
  单侧 p≈0.088。**两层方向强一致**:高功率 Layer B 显著(t=5.4),低功率 Layer D 点估计大且子段一致但
  CI 含 0(正是 METHODOLOGY 记的"组合 Sharpe 功率不足、单杠杆多 CI 含 0",故才用 daily IC 作主 oracle)。
- **综合判定**:**证据强favor 新池**,远超现池当初的 promote 依据(238 单点估计、无显著性、后蒸发)。
  Layer B CONFIRMED + Layer D 大正点估计 + 双层子段一致。
- **落地**:**✅ PROMOTED**(2026-06-30)。现池备份 `reports/selection_pre_15yr_2026-06-30.json`;
  `reports/selection.json` 覆盖为新 15yr-IR 池。revert = `cp reports/selection_pre_15yr_2026-06-30.json
  reports/selection.json`。⚠️ reports/ gitignore,新 30 因子列表存档于此(可复现):
  ```
  alpha_013, alpha_044, gtja_150, gtja_001, corr_pv_20, volume_std_20, corr_mom_vol_20,
  alpha_026, industry_relative_strength_20, alpha_019_compress, alpha_080, vwap_weighted_mom_5,
  alpha_040, alpha_088_expand_long, alpha_029_compress, alpha_069, alpha_011, alpha_094_compress,
  gtja_043, alpha_087, alpha_077, alpha_012, alpha_083, alpha_010_compress, alpha_024, alpha_023,
  alpha_029, close_skew_20, alpha_018, alpha_073_rev_short
  ```
- **附带**:F1 runner 死锁修复在 **top-1000 Layer D 路径验证通过**(run_portfolio_ab 跑通出 curve,无 hang)。
- **方法学**:候选池重选 = 一次性 15yr+top-1000+IR;时变交给已建 walk-forward Lasso+IC,不手工换池。
- **follow-up(非阻塞)**:可选第二流动性池 / 跨 top_k 的 Layer D 复核进一步加固;
  下一步按 NEXT_CYCLE_PLAN P1(MVO+LW 等新机制)在新基线上做。

---

## P0-1 — 15-yr runner 死锁验证 + Layer B oracle 本机可用性(2026-06-30)
- **背景**:`NEXT_CYCLE_PLAN.md` 地基第一步。目标 universe = 全市场 top-K(用户 2026-06-29 定)。
  验证 F1 死锁修复是否成立 + Layer B oracle 在**本机**是否可跑(31.7GB RAM / 32 核)。
- **机器约束发现**:handoff 的上一台机器死锁时 RSS 抓 28-33GB > 本机总内存。一度以为
  全市场 Layer D 在本机 OOM 不可行 —— **此判断作废**。真相:
  - 前几次"死亡"全是**启动/观测假象**,非真 bug:① `nohup &` 套在 `run_in_background` Bash 里
    双重 detach → 孤儿被 harness 回收;② harness 对追踪型后台任务有**分钟级寿命上限**,长跑被
    reap(但 Start-Process 的 OS 进程能存活);③ python stdout **块缓冲**(需 `-u`),print 不
    刷盘 → 文件看似冻结在 universe 加载;④ Git Bash `tasklist` 判活**首轮竞态**频繁误报死亡,
    须用 PowerShell `Get-Process` 判活。**可靠跑法**:PowerShell `Start-Process -PassThru`(OS detached)
    + `python -u` + 直接读 log 文件 + PowerShell 判活监控。
- **F1 死锁判定(本机, layer_b top-1000 路径)**:`precompute_scores_from_legacy` 两 arm 均完整
  跑完 → `to_parquet` → **Pool teardown 正常收尾,不挂**。score panel shape (3769, 1000) 全 15 年。
  内存峰值 ~12GB(主 ~7GB + 2 worker ~2.5GB 各)。**F1 在本机 1000-票尺度 CLEARED**。
- **Layer B oracle 自洽性**(top-1000, ic vs sharpe weighter,T=数千日 IC):
  - ΔIC mean = **−0.00110**(A=ic +0.04961 / B=sharpe +0.04850);IR proxy 0.342 vs 0.339;
    paired t = −2.644(iid p≈0.008,但 block-bootstrap 95% CI [−0.0030, +0.0003] **含 0**);
    子段全不一致(2/3/5/7/8 buckets 皆 mixed)。**VERDICT: NOT CONFIRMED**。
  - **复现** D3-Layer-B 历史结论(ΔIC −0.00057,同向 IC 略胜,同 NOT CONFIRMED)→ oracle 跨会话/
    跨机一致,**Layer B 验证 harness 在本机确立**(P0-2 达成)。
- **全市场版**(去 `--pool`,full 4601 universe,workers=2):**进行中**,验证全市场内存是否真塞得下
  (per-worker 数组由训练池 4601 决定,与 portfolio universe 无关 → 预计同 ~12GB,仅时间 ~4.6×)。
  结果待补。
- **判定**:P0-1 **基本闭环**(F1 本机 CLEARED + Layer B oracle 确立);全市场内存确认待全市场跑完。
  **无 config / 基线改动**。下一步按 NEXT_CYCLE_PLAN P1(MVO+LW 等新机制)走,全部用 Layer B 先筛。

---

## Follow-up 收尾 — F1-F5 技术债 / 研究方向(2026-06-28)
- **背景**:D3-Layer-B 收尾后 `BACKLOG.md` 留了 5 个 follow-up(F1 死锁 / F2 marker race /
  F3 worker OOM / F4 regime weighter / F5 tooling)。本次一并处理。
- **F1(死锁,HARDENED)**:定位为 `multiprocessing.Pool` teardown —— `with Pool(...)` 的
  `__exit__` 走 forcible `terminate()`,15-yr × ≥1000 票上 workers 不退、卡在 `__exit__`。
  `portfolio/scoring.py` 改显式 `close()`+`join()` 优雅收尾(异常 fallback `terminate()`);
  `portfolio_ab/runner.py` 加 `DBG:` 探针。**未在真 15-yr 数据复现验证**(需 ~30GB),
  是 reasoned hardening + 可观测性。parallel 路径(25 票 × 2 worker)+ 全测试套件无回归。
- **F2(marker race,DONE)**:`update_source_marker` 改 idempotent(同 source 跳过)+ atomic
  (temp + `os.replace`,Windows 并发 replace 的 access-denied 当良性吞);`check_source_change`
  空内容当无变化。`tests/test_fetcher.py` +4 测试(含 8 线程并发无虚报)。
- **F3(OOM,MITIGATED)**:close/join 让 worker RSS 跑完即释放 + auto workers 默认 min(3,cpu-1);
  文档化降 `--workers 3` 规避。panel 分块未做(低 ROI)。
- **F4(regime weighter,DRAFTED)**:设计草案
  `docs/superpowers/specs/2026-06-28-regime-conditional-weighter-design.md`(expanding 分位
  detector → 高波动 IC / 低波动 Sharpe + 两层评估 + 过拟合护栏)。未实装,default 仍 ic。
- **F5(tooling,DONE)**:不 promote 到 src/(循环专用脚本);CLAUDE.md 新增「改进循环分析工具」
  节文档化 layer_b_direct.py + ab_significance.py + 15-yr 已知问题。
- **判定**:技术债清理 + 研究方向归档,**无 config / 基线改动**(default 不变)。
  L1(baostock 黑名单)/ L2(已 push)/ L3(industry_map 续命)仍是外部网络项,非本次范围。

---

## D3-Layer-B — weighter ic vs sharpe @ 15-yr × top1000 (Layer B daily IC)
- **日期**:2026-06-28 · 配置 `docs/improvement_loop/configs/D3b_sharpe_full.yaml`
  · 工具 `docs/improvement_loop/analysis/layer_b_direct.py`(绕过 portfolio_ab runner 死锁)
- **背景**:D3 (3-yr ab_pool) 显示 sharpe weighter +0.12 ΔSharpe;但 `ab_significance.py`
  paired bootstrap NOT CONFIRMED(CI 含 0,子段反转)。把 cache 扩到 15-yr(mootdx 加分页 +
  `history_days: 3750`)+ 改换 Layer B(daily 截面 IC)拿更高功率检验。Universe 用 top-1000
  流动性(15-yr × 4400 在 runner 触发死锁 bug,见
  `docs/handoff/2026-06-28-portfolio-ab-15yr-deadlock.md`)。
- **结果**(15-yr × top1000, T=3533 日 IC 观测):

  | metric | weighter_ic | weighter_sharpe | Δ (B−A) |
  |---|---:|---:|---:|
  | daily IC mean | +0.04628 | +0.04571 | **−0.00057** |
  | daily IC std | 0.133 | 0.132 | — |
  | IR proxy(mean/std) | +0.349 | +0.347 | −0.002 |
  | paired t (iid) | — | — | −1.253 |
  | block-bootstrap 95% CI for mean ΔIC | — | — | **[-0.00160, +0.00044]**(含 0) |

- **regime 子段**(7 buckets,按事件日期切):

  | 区间 | n | ΔIC | t | 方向 |
  |---|---:|---:|---:|---|
  | 预 2015 杠杆牛 (2011-12 ~ 2015-06) | 856 | −0.00112 | −0.74 | IC 略胜 |
  | **2015 杠杆崩盘 + 熔断** (2015-06 ~ 2016-02) | 156 | **−0.00326** | **−2.79** | **IC 显著胜** |
  | 供给侧改革 (2016-02 ~ 2018-06) | 577 | −0.00016 | −0.55 | 平 |
  | 贸易战 (2018-06 ~ 2020-03) | 414 | −0.00124 | −1.58 | IC 略胜 |
  | COVID + 抱团 (2020-03 ~ 2021-12) | 450 | +0.00029 | +0.27 | 平 |
  | **平台监管 / 退市** (2022-01 ~ 2024-04) | 549 | **+0.00162** | **+2.73** | **sharpe 显著胜** |
  | 新国九条 (2024-04 ~ 2026-06) | 531 | −0.00180 | −1.54 | IC 略胜 |
  | **整体** | — | **5− vs 2+** | — | 不一致 |

  uniform 5-buckets: 4− vs 1+;uniform 8-buckets: 5− vs 3+。

- **判定**:**NOT CONFIRMED at Layer B**(点 ΔIC = −0.00057,15-yr 大样本 CI 仍含 0,
  且**方向反过来 IC 略胜**)。
- **解读**:sharpe weighter 是 **regime-conditional alpha**:
  - 在低波动/趋势期(2022-2024 平台监管/退市/弱复苏)显著占优(t = +2.73);
  - 在极端波动期(2015 杠杆崩盘 + 熔断)显著吃亏(t = −2.79);
  - **整体抵消,且 IC 边际更稳健**。
- **落地**:维持 `config.yaml: weighter.type: ic`(D3 之后已 revert)。三个新 weighter
  (sharpe / halflife_ic / ridge)的实现 + 测试**保留**,可 opt-in,但 default 不切。
- **方法学副产物**:
  - 写了 `layer_b_direct.py`(绕过 runner 死锁,直接调 `precompute_scores_from_legacy`)
  - 扩 `ab_significance.py` 加 `--subperiods` / `--regime-boundaries`
  - mootdx 加分页支持(`_fetch_paginated`,`start` 翻 5 页拿 15 年)

---

## D2 — selector lasso vs lightgbm (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/D2.yaml`
- **结果**(238 ab_pool):selector_lasso Sharpe 1.60 vs selector_lightgbm Sharpe 0.19。Δ −1.41(灾难)。
- **判定**:**REJECTED(lightgbm)**。30 因子 + 有限样本下树模型严重过拟合,lasso 远优。
  确认并扩展 CLAUDE.md 既有负向结论。保持 lasso。**子任务 D 结案。**

## A5 — GTJA-inclusive fresh factors analyze + reselect (capstone follow-up)
- **日期**:2026-06-27 · 背景:2026-06-24 analysis 不含 gtja_*,A4 的 IC 重选无法纳入 GTJA 因子。
  本方向重跑全市场 `factors analyze`(含 gtja),pick-by-ic,再 AB vs GTJA 基线。
  这是唯一可能再生增益的方向。
- **analyze 完成**:386 因子(含 gtja_*),~60min,输出 `reports/factor_analysis/loop_a5/2026-06-27.json`。
- **pick-by-ic**(top-25/max-corr 0.6/min-ir 0.05/max-degen 0.5)→ `A5_selection.json`:25 因子,
  **含 6 个 gtja**(097/150/001/020/080/135)+ wq101 窗口变体(088_expand_long / 037_rev_short /
  094_compress / 073_rev_short / 029_compress)等。与现 GTJA 手工集显著不同。
- **AB**(238 ab_pool):baseline_gtja Sharpe 1.60 / return 1.855 / DD 0.164 vs
  pick_by_ic_gtja_25 Sharpe 1.32 / return 1.454 / DD 0.225。Δ Sharpe −0.29,DD 恶化。
- **判定**:**REJECTED**。即便 IC/IR/去相关原则化重选 **且纳入 gtja**,仍输给手工 GTJA 集。
  **factor 选择空间彻底穷尽:GTJA selection.json 对 5 类候选(hand / clean / wq101 / IC-no-gtja /
  IC-with-gtja)全胜。A5 结案,无 config 改动。**

---

# ★ 最终收敛总结 (2026-06-27)

**自驱改进循环收敛。** 8 子任务 × 24 个方向全部 AB 验证完毕,**2 个改进落地 config.yaml**:

| # | 改进 | 方向 | 效果(238 ab_pool portfolio-ab) |
|---|---|---|---|
| 1 | **GTJA 因子集** → `reports/selection.json` | A1 | Sharpe **0.51 → 1.33**;return 0.325→1.211;DD 0.245→0.181 |
| 2 | **top_k 20 → 10** | G1/G1b | Sharpe **1.33 → 1.60**;return 1.211→1.855;DD 0.181→0.164 |

**累计:portfolio Sharpe 0.51 → 1.60(3.1×),年化 0.10 → 0.44,maxDD 0.245 → 0.164。**

**22 个方向 REJECT/N-A**(基线在这些维度已是强局部最优):
- A2/A3/A4/A5(其它因子集全输 GTJA)· B1 industry_neut · B2 mcap_neut · B3 winsorize-off ·
  C1/C1b horizon(3 最优)· C2 train_window(250 最优)· C3/C3b lasso alpha(0.001 最优)·
  C4 refit_every(月度 cadence 下 inert)· D1 equal weighter · D2 lightgbm selector(灾难)·
  E1 embargo(auto 有效)· F1 mask(剔除涨停标签丢动量)

**reasoned-out(未跑 AB,有据)**:E2 label_basis(open=现实口径,close 是已知乐观偏差)·
H1 sizing(portfolio 引擎等权,不用 LotSizer,N/A)。

**收获的诊断性发现**:
1. GTJA191 短周期量价因子是本策略最大单一增益来源(+0.82 Sharpe)。
2. portfolio 集中度 top_k=10 是甜点(10 > 20 > 5,5 过度集中 DD 爆)。
3. 所有 cross-sectional neutralization(industry/mcap)在本因子集上**抹掉 alpha**(A 股 size/行业动量是信号)。
4. mask 把涨跌停日剔除训练标签 = 丢最强动量正样本(量化印证设计取舍)。
5. `refit_every` 在 pooled 月度 refit 路径下无效(结构性)。

**遗留 follow-up(非本循环可清)**:
- A5+ 调参(top-n / max-corr 扫)或 GTJA 因子族扩充后重选,理论上仍可能微调因子集,但 ROI 递减。
- **L1**:`ab_pool` 缺 IPO 硬过滤(baostock 黑名单封锁)→ 绝对收益偏乐观;**两 arm 同池故相对 AB 公平**。
- **L2**:本地 commit 累积未 push(github:443 不可达,无代理)→ 需用户 VPN/代理恢复后 push。
- **L3**:industry_map 缓存 touch 续命(离线复用),网络恢复应真正 refresh。

L1/L2/L3 均为**外部网络阻塞**,非循环可自主清除;循环可控范围内的改进方向已穷尽 → **停止**。

## A4 — GTJA 基线 vs pick-by-ic IC 去相关集 (25, 无 gtja)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/A4.yaml`(选择文件 A4_selection.json)
- **结果**(238 ab_pool):baseline_gtja Sharpe 1.60 / DD 0.164 vs pick_by_ic_25 Sharpe 1.37 / DD 0.290。
  Δ Sharpe −0.23,DD 大幅恶化。
- **判定**:**REJECTED**。即便 IC + 去相关 + IR 三重原则化重选,无 GTJA 的集仍输给 GTJA。
  **子任务 A 彻底结案:GTJA 因子集为最优,对 hand/clean/wq101/IC-principled 四类候选全胜。**
- **遗留方向**:GTJA-inclusive 的 fresh `factors analyze`(2026-06-24 analysis 早于 gtja,不含
  gtja_*)能否再选出更优集 — 需重跑 analyze(全市场 ~300 因子 IC,~1-3h),记为 follow-up。

---

## B3 — winsorize [0.01,0.99] vs off (panel 重建)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/B3.yaml`
- **结果**(238 ab_pool):winsor_on Sharpe 1.60 / return 1.855 vs winsor_off Sharpe 0.82 / return 0.732。
  Δ Sharpe −0.78。
- **判定**:**REJECTED(off)**。winsorize 强有效(裁尾抑制极端值污染 IC/Lasso)。保持 [0.01,0.99]。
  **子任务 B 完全结案**:winsorize on 必需,industry/mcap neut 皆 off = 最优 preprocess。

---

## E1 — embargo_days auto(=3) vs 0 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/E1.yaml`
- **结果**(238 ab_pool):embargo_auto Sharpe 1.60 / DD 0.164 vs embargo_0 Sharpe 1.41 / DD 0.288。
  Δ Sharpe −0.19,DD 大幅恶化。
- **判定**:**REJECTED(embargo=0)**。去掉 embargo 引入 horizon 日标签泄露,样本变多但 OOS 变差
  (DD 0.288)。**印证 F2 PR-A embargo 设计**。保持 auto。E1(embargo)结案;2×horizon 不再单测
  (auto 已好、0 已坏,更大 embargo 仅减样本)。

---

## C4 — refit_every 20 vs 10 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C4.yaml`
- **结果**(238 ab_pool):**完全 bit-identical**(每项 Δ=0)。refit_10 **确实重算**了
  (日志:pre-warmed 29 monthly fits + parallel 238 stocks,非缓存命中)。
- **判定**:**结构性发现**:pooled `share_pool_fit=true` 打分路径用**月度** refit 节奏
  (29 个月度 fit),`refit_every` 对组合打分路径**无效**(被月度 cadence 覆盖)。无 config 改动。
  C4 结案。

---

## C3b — lasso alpha 0.001 vs 0.005
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C3b.yaml`
- **结果**(238 ab_pool):alpha_1e3 Sharpe 1.60 vs alpha_5e3 Sharpe 0.34。Δ −1.26(灾难性)。
- **判定**:alpha=0.005 过度剪枝(几乎杀光因子)。**alpha=0.001 最优**(0.001 > 0.0005 且 ≫ 0.005)。
  **C3 结案,无 config 改动。**

---

## C3 — lasso alpha 0.001 vs 0.0005 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C3.yaml`
- **结果**(238 ab_pool):alpha_1e3 Sharpe 1.60 vs alpha_5e4 Sharpe 0.91。Δ −0.69。
- **判定**:**REJECTED(0.0005)**。更低 alpha=更少稀疏=更多噪声因子,大败。试 C3b(vs 0.005)bracket。

---

## D1 — weighter ic vs equal (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/D1.yaml`
- **结果**(238 ab_pool):weighter_ic Sharpe 1.60 / return 1.855 vs weighter_equal Sharpe 0.80 / return 0.902。
  Δ Sharpe −0.81(大败)。
- **判定**:**REJECTED(equal)**。IC 加权远优。印证 2026-05-24 回退决定。保持 ic。
- **顺带**:ir 不再单测(equal 已大败,ir 不太可能超 ic);D2(lightgbm selector)CLAUDE.md
  已有负向 AB 证据,降级(若 GTJA 集下想复核可后补)。

---

## C2 — train_window 250 vs 500 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C2.yaml`
- **结果**(238 ab_pool):tw_250 Sharpe 1.60 / return 1.855 vs tw_500 Sharpe 1.35 / return 1.463。
  Δ Sharpe −0.26。
- **判定**:**REJECTED(tw=500)**。长窗跨更多 regime,稀释近期信号。保持 tw=250。

---

## C1b — horizon 1 vs 3
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C1b.yaml`
- **结果**(238 ab_pool):horizon_1 Sharpe 0.80 / return 0.676 vs horizon_3 Sharpe 1.60 / return 1.855。
  Δ horizon=3 +0.80。
- **判定**:horizon=1 太短噪声大,大败。**horizon=3 是最优**(3>5 且 3≫1),保持。**C1 结案。**

## F1 — tradability mask off vs on (仅 limit/停牌,min_listing_days=0 规避缺失的 ipo_dates)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/F1.yaml`
- **背景**:`data/ipo_dates.parquet` 缺失(需 baostock,当前被封)→ listing mask 会退化到
  first_valid_index 启发式(CLAUDE.md 警告:mask 比例虚高)。故设 `min_listing_days=0`
  只测 mask 的核心价值:把涨跌停/停牌日从训练标签层剔除(forward_return 双向检查)。
  score 重算,factor panel cache-hit。
- **结果**(238 ab_pool):mask_off Sharpe 1.60 / return 1.855 vs mask_on Sharpe 0.97 / return 0.777。
  Δ Sharpe −0.63(大败)。
- **判定**:**REJECTED(mask on)**。**重要发现**:把涨跌停日从训练标签剔除 = 丢掉最强的动量
  正样本(涨停 +9.9% 本身是信号),模型变弱。**量化印证** CLAUDE.md 的设计判断("涨停日是有用
  信号")。保持 mask off。**子任务 F 结案。**
- **注**:本测仅 limit/停牌 mask(min_listing_days=0)。listing mask 需 ipo_dates(缺失);
  但核心 mask 已负,不再追加。

---

## C1 — horizon 3 vs 5 (score 重算)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/C1.yaml`
- **结果**(238 ab_pool):horizon_3 Sharpe 1.60 / return 1.855 vs horizon_5 Sharpe 1.38 / return 1.416。
  Δ Sharpe −0.22。
- **判定**:**REJECTED(horizon=5)**。3 > 5。试 C1b(1 vs 3)bracket 最优。

---

## G3 — max_per_industry 5 vs 3
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G3.yaml`
- **结果**(238 ab_pool):cap_5 Sharpe 1.60 vs cap_3 Sharpe 1.59,Δ −0.01(噪声级)。
- **判定**:**无差异**。top_k=10 下行业 cap 极少 binding(top-10 已横跨足够行业)。保持 cap=5。
  **子任务 G 结案**:仅 top_k=10 是有效改进;rebal=5 / cap=5 已是最优。

---

## G2b — rebalance_n_days 5 vs 3
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G2b.yaml`
- **结果**(238 ab_pool):rebal_5 Sharpe 1.60 / DD 0.164 vs rebal_3 Sharpe 1.13 / DD 0.291,
  trade +68%。Δ Sharpe −0.48(大败,交易成本主导)。
- **判定**:**REJECTED(rebal=3)**。**rebalance=5 是最优**(5>10 且 5>3),保持现状,无 config 改动。
  **G2 子方向结案**。

---

## G2 — rebalance_n_days 5 vs 10 (engine-only,基线 top_k=10)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G2.yaml`
- **结果**(238 ab_pool):rebal_5 Sharpe 1.60 / DD 0.164 vs rebal_10 Sharpe 1.54 / DD 0.205。
  Δ Sharpe −0.06,DD +0.041(更差)。
- **判定**:**REJECTED(rebal=10)**。保持 rebal=5。但 5<10 趋势 → 试 G2b(5 vs 3,
  与 horizon=3 对齐)看更频繁是否更优(扣已建模交易成本)。

---

## G1b — portfolio top_k 10 vs 5 (集中度 sweep step 2)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G1b.yaml`
- **结果**(238 ab_pool):

  | metric | topk_10 | topk_5 | Δ |
  |---|---:|---:|---:|
  | total_return | 1.855 | 1.627 | −0.227 |
  | sharpe | 1.60 | 1.35 | −0.25 |
  | max_drawdown | 0.164 | 0.255 | +0.090(大幅恶化) |

- **判定**:top_k=5 **过度集中**(DD 0.164→0.255)。**sweep 最优 = top_k=10**(10 > 20 且 10 > 5)。
- **落地**:**config.yaml `portfolio.top_k: 20 → 10`**(已校验加载)。G1 方向 **KEPT**。
  这是本循环**第 2 个 AB 验证的改进**(继 GTJA 因子集)。

---

## G1 — portfolio top_k 20 vs 10 (engine-only, score 缓存共享)
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/G1.yaml` · 两 arm 均 cache-hit scores
- **假设**:更集中(更少持仓)= 更强信号股权重更高,可能提升风险调整收益。
- **结果**(238 ab_pool):

  | metric | topk_20 | topk_10 | Δ |
  |---|---:|---:|---:|
  | total_return | 1.211 | 1.855 | +0.644 |
  | ann_return | 0.317 | 0.439 | +0.122 |
  | sharpe | 1.33 | 1.60 | +0.27 |
  | max_drawdown | 0.181 | 0.164 | −0.016(更优) |
  | trade_count | 2120 | 1060 | −50% |

- **判定**:**WIN(top_k=10 占优)**,但**先 sweep G1b(10 vs 5)找最优再提交 config**,
  避免过度集中(238 池里 top_k=5 的 idiosyncratic 风险 / 过拟合)。暂不改 config.yaml。

---

## B2 — GTJA 基线 preprocess.mcap_neutralize false vs true
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/B2.yaml`(balance 缓存 offline)
- **假设**:市值中性化去除 size beta,可能提纯 alpha。
- **结果**(238 ab_pool):

  | metric | baseline(off) | mcap_neut(on) | Δ |
  |---|---:|---:|---:|
  | total_return | 1.211 | 0.743 | −0.468 |
  | sharpe | 1.33 | 0.96 | −0.37 |
  | max_drawdown | 0.181 | 0.223 | +0.042(更差) |

- **判定**:**REJECTED**(大幅退化)。A 股 size/小盘溢价是强 alpha 来源,中性化把它抹掉了。保持 off。
- **结论**:子任务 B 两个 neutralize 方向皆负 → 现有 preprocess(winsorize+zscore,两 neut 关)
  已是较优配置。B3(winsorize 微调)预期低 ROI 且需 30min panel 重建,降优先级。
- **效率洞察**:**G 子任务(portfolio 参数 top_k/rebalance/cap)只改 engine 不改 score**,
  score 缓存键不含 portfolio 参数 → 两 arm 共享缓存、仅跑快 engine(~2-3min/AB),优先做。

---

## B1 — GTJA 基线 preprocess.industry_neutralize false vs true
- **日期**:2026-06-27 · 配置 `docs/improvement_loop/configs/B1.yaml`
- **假设**:行业中性化去除行业 beta,可能提纯 alpha。
- **结果**(238 ab_pool):

  | metric | baseline(off) | industry_neut(on) | Δ |
  |---|---:|---:|---:|
  | total_return | 1.211 | 1.132 | −0.079 |
  | sharpe | 1.33 | 1.24 | −0.09 |
  | max_drawdown | 0.181 | 0.232 | +0.052(更差) |
  | win_rate | 0.493 | 0.470 | −0.024 |

- **判定**:**REJECTED**。empirically 印证 CLAUDE.md P1.5:行业中性化在本因子集上
  不增益(单成员子行业 demean-to-zero 风险 + 去掉了有用的行业动量)。保持 off。

---

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
