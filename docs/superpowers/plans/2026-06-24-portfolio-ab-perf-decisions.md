# Portfolio-AB 性能优化 — 调查 + 决策日志

> 任务:用户反馈 `portfolio-ab`(以及底层 portfolio backtest)跑得太慢,要求定位瓶颈、
> 优化并加速,**每次优化都要 minimal example 验证优化前后只有精度以内的改变**。
> 用户授权:决策按推荐执行,记录到本文档汇总。
>
> 环境:`.venv`(pandas 3.0.x),本地 `data/` 全市场缓存 4599 只日线(无需联网)。
>
> 注:本工作首次于 2026-06-24 完成;因 working tree 被重置丢失,2026-06-27 在更新后的
> `main` 上原样 reapply(`generate_signals` / `eligibility.py` / indicators / signals /
> resample 全部 byte-identical,bit-exact 结论不变;`scoring.py` 的 `_shared_cache` 崩溃
> 此时已被上游用 `getattr(...) is not None` 守护修复,故 #3 不再需要重做)。

---

## 0. 瓶颈定位(profiling)

`portfolio-ab` 每个 arm 的流程:
1. `load_or_build_factor_panel`(仅 ml_factor,落盘缓存)
2. `build_strategy`
3. **`precompute_scores_from_legacy`** — 对 portfolio universe 每只票调
   `legacy.generate_signals(daily)`,拼成 T×N score 面板(落盘缓存,key=content_hash)
4. `PortfolioEngine.run`(× staggered offsets)

冷缓存下 **(3) precompute 是绝对大头**。在 100 票池子上用 `composite_verdict`
(无 ML 成本)测得每票 `generate_signals` ~17-19s。单票 cProfile + 分段计时定位到
**weekly 段**:

| 段 | 耗时(单票 1600 bar) |
|---|---|
| daily 循环(`detect_signals` per bar) | 0.27 s |
| **weekly 循环** | **16.6 s** |
| — 其中 per-bar `resample_to_weekly` | 5.2 s |
| — 其中 per-bar `add_all`(周线指标) | 11.5 s |

根因:`CompositeVerdictStrategy.generate_signals` 是 **O(T²)** —— 每个 daily bar 都
重新 `resample_to_weekly(daily[:i+1])`(O(i))并对整条周线重算 `add_all`(O(i/5))。
关键发现:**`add_all` 的耗时 ~7.4 ms/call 与帧大小几乎无关**(30 行 vs 320 行都是 ~7.4ms),
是固定 per-call 开销。所以瓶颈是 **调用次数(~1571 次/票)**,不是单次大小。

随后引擎也 profile 出第二个瓶颈(见 #2)。

---

## #1 — `composite_verdict` weekly 计算 O(T²) → O(T)(已落地)

- **文件**:新增 `src/stockpool/backtesting/composite_weekly.py`;改 `strategies.py:CompositeVerdictStrategy.generate_signals`;新增 `tests/test_composite_weekly_fast.py`
- **做法**:把"逐 bar resample+add_all"换成"**一次** resample 全量周线 + `add_all` 一次得到
  所有**已完成周**的精确指标,再对每个 daily bar 只增量算**当前部分周(partial week)**那一行,
  复用真实 `detect_signals`(4 行尾帧 `[完整周 k-3..k-1, partial 行]`,因为 detect_signals 只读
  最后 2-3 行 + `len>=4` gate)。
- **为什么 bit-exact**(逐指标论证 + 实测):
  - 所有 EMA 用 `adjust=False`(纯递归 `ema[t]=a·x[t]+(1-a)·ema[t-1]`)→ partial 行的
    MACD/KDJ/RSI 可由"已完成周 k-1 的递归状态 + 一步外推"得到,**实测 bit-exact**。
  - `rolling.min/max`(KDJ 窗口、breakout)顺序无关 → 用 `min/max(完整窗口, partial)` 精确。
  - ⚠️ **`rolling.mean` / `rolling.std` 在 pandas 里是 online(Kahan/Welford)累加器,
    history-dependent**:同一个 20 窗口,放在 70 行尾巴 vs 全序列尾部,最后一格的值都不一样
    (实测 std 差 ~4e-13,mean 差 ~1.8e-15)。所以 **MA / BOLL / vol_ratio 不能用闭式**,
    必须对**完整 as-of 周线序列**跑 pandas rolling(每 bar 6 次 rolling,但远比每 bar 一次
    `add_all` 便宜)。
  - 之前尝试的闭式(prefix-sum MA、`sqrt(E[x²]-E[x]²)` std、手写 Welford 复刻)都在
    1e-12~1e-15 量级与 pandas 不一致,会在 `ma_alignment`(ma10==ma20 的 1e-15 平局)等
    离散触发点**翻转 verdict**,不满足验收 → 故采用 as-of rolling。
- **minimal example 验证(优化前后)**:
  - `scripts/verify_portfolio_ab_perf.py` 对照"慢路径 per-bar"逐 bar 比较 weekly_score:
    30 票真实 A 股(~47000 bar)+ 8 票 + 3 合成种子 + 短历史(<30 周全 0):**全 0 mismatch**。
  - 锁死测试 `tests/test_composite_weekly_fast.py`(6 case)+ 原有
    `tests/test_backtest_composite.py`(walk-forward == live pipeline)全绿。
- **加速实测(单票 weekly 段)**:**16.98 s → 1.52 s ≈ 11×**;整 `generate_signals`
  约 17s → ~1.8-2s。100 票 precompute(串行)从 ~32min → ~3min。
- **副作用**:`generate_signals` 不再逐 bar 调 `resample_to_weekly`;`predict_latest`(单点
  日报路径)**未改**(本来就是单点,不慢)。`resample_to_weekly` import 仍保留(predict_latest 用)。

---

## #2 — `PortfolioEngine` eligibility O(rebalances×N×T) → 一次性预算(已落地)

- **文件**:重写 `src/stockpool/portfolio/eligibility.py`;新增 `tests/test_portfolio_eligibility_equiv.py`
- **定位**:`generate_signals` 提速后再 profile 100 票 `PortfolioEngine.run`,发现 **38.2s**,
  cProfile 显示 ~98% 在 `EligibilityFilter.eligible`。根因:engine 每个 rebalance bar
  (~376 次)对**每只票**重新 `pd.to_datetime(daily["date"])` + 全表过滤 + tail(20) 聚合。
  37600 次 `to_datetime`,pandas `should_cache` 遍历 6M 个 datetime。**这段不进任何缓存,
  每次 portfolio-backtest / portfolio-ab 调用都重跑**(即使 score panel 命中磁盘缓存)。
- **做法**:eligibility 在 `date_t` 的判定是每只票自身历史的 as-of 纯函数 →
  **每个 panel 预算一次**,得到每票 `(sorted_date_ns, eligible_bool_per_bar)`,之后每个
  rebalance bar 用 `searchsorted` O(N log T) 查表。cache 按 `id(panel_data)` 失效
  (engine 整个 run 复用同一个 dict)。
- **为什么 bit-exact**:
  - ST / 缺列 / `min_history_bars` / 阈值=0 跳过 等分支语义逐条照搬。
  - 流动性用 `tail(20).mean()` 的**新鲜窗口均值**(不是 `rolling().mean()` 的 online 累加器),
    用 `sliding_window_view(amount,20).mean(axis=1)`(full window)+ 前 19 个 partial 直接均值,
    **实测与 `Series(window).mean()` bit-exact**(避免 #1 里踩过的 online-rolling 漂移坑)。
  - 防御性按 date 排序(与 engine 的 `_build_wide_pivots` 一致;真实/测试数据本就有序)。
- **minimal example 验证**:
  - `scripts/verify_portfolio_ab_perf.py` + 单独脚本在 100 票 × 122 个 rebalance date 对照旧逻辑:**0 mismatch**。
  - 锁死测试 `tests/test_portfolio_eligibility_equiv.py`(10 case,3 seed × 3 config + 缺列)+
    原有 `tests/test_portfolio_eligibility.py`(13 case)全绿。
- **加速实测**:
  - eligibility 段(100 票,122 date):**11.74s → 0.09s ≈ 130×**
  - `PortfolioEngine.run`(100 票全程):**38.2s → 0.45s ≈ 85×**
  - (12 票小池子时 eligibility 不是瓶颈,加速 ~1×;收益随 universe×rebalance 规模放大)
- **副作用**:`EligibilityFilter` 多了 `_cache` / `_cache_key` 实例状态;接口 `eligible(date_t, panel_data)`
  不变。**universal win** — 所有策略(composite_verdict / ml_factor)的 portfolio engine 都受益。

---

## #3 — pre-existing bug:composite_verdict 过 portfolio precompute 崩溃(上游已修)

- **现象**:end-to-end 跑 `portfolio-ab`(≥20 票,composite_verdict)时 arm 报
  `AttributeError: 'CompositeVerdictStrategy' object has no attribute '_shared_cache'`。
- **根因**:`precompute_scores_from_legacy` 在 `n_workers>1 && prewarm` 分支里无条件访问
  `legacy_strategy._shared_cache`(只有 ml_factor 有该属性)。
- **状态**:首次工作(2026-06-24)曾用 1 行 `getattr` 守护修复;**2026-06-27 reapply 时发现
  上游 `scoring.py` 已用 `and getattr(legacy_strategy, "_shared_cache", None) is not None`
  把整个 prewarm 块守护掉(更干净)**,故本次 **不再重做** #3。

---

## 总体效果 & 验收

- 单票 `composite_verdict.generate_signals`:**~17s → ~1.8-2s(~9×)**。
- 100 票 `PortfolioEngine.run`:**38.2s → 0.45s(~85×)**;eligibility 段 130×。
- 全直接相关测试套件绿:`test_composite_weekly_fast`(6)、`test_portfolio_eligibility(_equiv)`(13+10)、
  `test_portfolio_engine` / `industry_cap` / `scoring` / `portfolio_ab_*` / `cli_portfolio_*` /
  `ml_strategy` / `report_smoke` / `backtest_composite` / `timer_reset`。
- 已知 pre-existing 失败(与本次无关):`test_ops_snapshot.py`(pandas 3.0 `dropna` API 变更)、
  `test_cli_backtest.py` 部分(离线无网 baostock/akshare + 空 sector_map)。

## #A — score panel 跨 arm 共享(已落地;核心由上游实现 + 本次补 refresh 缺口)

- **日期**:2026-06-27
- **背景**:portfolio-ab 两 arm 若只差 portfolio 参数(`top_k` / `rebalance` / eligibility),
  **per-stock score 完全相同**,旧 cache key = `content_hash`(含 `portfolio_backtest`)→ 两 arm
  各自重算一遍。
- **核心(上游已实现)**:`scoring.py:score_cache_key(cfg, universe_codes)` —— 把 config
  `model_dump` 后剔除 `_NON_SCORING_CFG_KEYS`(`portfolio_backtest` / `report` / `recommend_pool` /
  `ab_pool` / `content_hash`)+ 拼上 universe codes + 版本 tag `v2`,sha256[:16]。
  "conservative by construction:只丢列出的非打分段,任何打分相关字段不同都强制换 key,**绝不会错误共享**"。
  `run_single_arm` 用它做磁盘 parquet 文件名。已有测试 `tests/test_portfolio_score_cache.py`
  覆盖:同 scoring 不同 portfolio 参数 → 同 key;universe 不同 → 不同 key(且顺序无关);strategy 不同 → 不同 key。
  **实测**:`portfolio-ab --config portfolio_ab_simple.yaml`(2 arm 只差 top_k)第二个 arm 命中
  `cache hit`,只写 1 个 score 文件(不是 2)。
- **本次补的缺口(in-memory memo)**:旧逻辑在 `--refresh-scores` 下两 arm 仍各重算(磁盘 cache 被绕过)。
  改 `run_single_arm` 加可选 `score_memo` 参数,`run_portfolio_ab` 串行路径建一个共享 `{key: panel}` dict:
  - 三级复用(由廉到贵):**in-memory memo**(跨 arm,且覆盖 refresh)→ 磁盘 parquet(跨 run,refresh 时跳过)→ 重算。
  - **额外收益**:命中 memo/磁盘时**跳过 `load_or_build_factor_panel` + `build_strategy`**(对 ml_factor 省一次因子面板加载)。
  - 并行 arm 路径(`parallel_arms=True`,子进程)无法共享内存 memo,仍走磁盘 cache(不变)。
  - **实测**:`portfolio-ab --refresh-scores` 现在第二个 arm 打 `reusing in-memory score panel`,
    只 precompute 一次(之前 refresh 下会算两次)。
- **bit-exact / 正确性**:复用的是同一个 score panel 对象/同一份 parquet,数值完全一致(非近似)。
  新测试 `tests/test_portfolio_ab_runner.py::test_score_memo_shares_panel_across_arms`:
  monkeypatch `precompute_scores_from_legacy` 计数,`refresh_scores=True` 下两 arm 只调 1 次。
- **文件**:`portfolio_ab/runner.py`(`run_single_arm` 重构三级复用 + `score_memo` 参数;`run_portfolio_ab` 串行建 memo);
  `tests/test_portfolio_ab_runner.py`(+1 test)。`scoring.py:score_cache_key` 上游已在,未改。
- **副作用**:`run_single_arm` 把 factor_panel/`build_strategy` 移进"需要重算"分支(失败隔离不变,仍在 try 内)。

## #B — precompute 并行 `--workers`(已落地)

- **日期**:2026-06-27
- **背景**:`precompute_scores_from_legacy(n_workers=...)` 早已用 `multiprocessing.Pool` 实现并行,
  且 `portfolio-ab` 的 CLI `--workers` 早已端到端接好。**唯一缺口**:`portfolio-backtest` 子命令
  没暴露 `--workers`,且 `cmd_portfolio_backtest` 调 precompute 时没传 `n_workers`(用函数默认 auto ≤3)。
- **做法**:
  - `cli.py` 给 `portfolio-backtest` 子解析器加 `--workers`(type=int,default=None,help 与 portfolio-ab 对齐)。
  - `cmd_portfolio_backtest` 的 `precompute_scores_from_legacy(legacy, portfolio_pool_data)` 改为传
    `n_workers=args.workers`。
  - `portfolio-ab` 无需改(已就绪)。
- **验证**:
  - `tests/test_cli_portfolio_backtest.py::test_portfolio_backtest_workers_flag`(`--workers 1` smoke)+ 全套 CLI 测试绿。
  - 端到端 `portfolio-ab --workers 2 --refresh-scores`(46 票 data_small):日志出
    `precompute_scores: parallel mode (n_workers=2) over 46 stocks`,与 #A 的 in-memory memo 并存
    (arm2 仍 `reusing in-memory score panel`),出报告 EXIT 0。
- **收益/代价**:大池子(几百~4000+ 票)拿 ~cores× 加速(并行打分);Windows spawn 每 worker pickle 整个
  strategy(含 pool_data + factor_panel)~hundreds MB–6GB,内存换时间;`1` 强制串行省内存;小池子
  (<20 票)precompute 内部 tiny-workload 短路强制串行,该 flag 无副作用。
- **文件**:`cli.py`(portfolio-backtest 子解析器 + 调用点);`tests/test_cli_portfolio_backtest.py`(+1);
  `CLAUDE.md` + `README.md`(命令 + 说明)。

## 其余后续优化(未实现)

- **#C ml_factor generate_signals**:**评估后不做**(低性价比)。实测(25 票 × 1600 bar × 8 因子):
  因子面板 0.28s(磁盘缓存)、首票 0.95s(付**共享**月度 Lasso fits)、第 2..N 票 **0.257s/票**(复用 fits,
  纯预测循环)。cProfile 显示 per-stock loop 热点全是 **pandas 逐 bar 索引开销**(`isinstance`/arrow
  `__getitem__`/`.iloc[i]`),`pipeline.predict` 连前 12 都进不去(batch-predict 已把预测摊薄)。即:
  算法结构(月度共享 fit + batch predict + 面板缓存)已最优,剩下只有 **~1.5-2× 常数因子**(把标量
  `.iloc[i]` 换 numpy),且要动全项目最 intricate、背 serial/parallel/prewarm bit-exact 不变量的代码;
  又是一次性、已被 `score_cache_key` + memo 缓存的成本。ROI 远低于 #1/#2/#A/#B。真要榨 ml_factor,
  优先用 #B 的 `--workers` 并行,而非动这段循环。
