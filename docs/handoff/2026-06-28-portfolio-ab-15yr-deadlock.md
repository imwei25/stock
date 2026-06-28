# Handoff: portfolio_ab runner 在 15-year scale 上死锁

> 写于 2026-06-28。前一个 claude code 进程在做 sharpe weighter 显著性验证时,把 cache 扩展到
> 15 年数据,触发了 portfolio_ab runner 的一个死锁 bug。已绕过(写了直接调
> `precompute_scores_from_legacy` 的脚本)拿到结果,但**没修这个 bug**,留给后续进程。

---

## 一、症状

在 `--config docs/improvement_loop/configs/D3b_sharpe_full.yaml` 上跑
`ab_significance.py --full-market --workers {3,6}` 后,**100% 复现**以下死锁:

1. `precompute_scores_from_legacy` 100% 完成(进度条到顶)。
2. 之后 main 进程 0% CPU,worker 子进程也 0% CPU,**但 RSS 不释放**(28-33 GB 抓着)。
3. 没有任何 disk I/O(`io_counters` 增量为 0)。
4. 没有 `score_panel.to_parquet` 的输出文件被创建。
5. 持续半小时不动 → 必须 kill。

**复现条件**:
- ✅ 15-yr × full 4400 universe + workers=6
- ✅ 15-yr × full 4400 universe + workers=3
- ✅ 15-yr × top 1000 universe + workers=6
- ❓ 3-yr 数据下从未触发过(D3b 上午成功跑过)

**规模因素**:不完全是规模问题(top 1000 也卡)。但 3-yr 数据 + 4400 stocks 历史上 OK,
说明问题跟"每股的时间序列长度"或"score panel 行数"相关,而不是股票数。
score panel shape:3-yr × 1000 = (763, 1000) 工作 ✓;15-yr × 1000 = (3769, 1000) 死锁 ✗。

---

## 二、绕过方案(已落地)

`docs/improvement_loop/analysis/layer_b_direct.py` 直接调
`precompute_scores_from_legacy`,跳过 `run_portfolio_ab` 的 simulate 路径,**工作正常**:

```python
# 关键调用,跑通了 15-yr × 1000 × 2 arms,2x ~14 min/arm
sp = precompute_scores_from_legacy(legacy, portfolio_pool_data, n_workers=6)
sp.to_parquet(score_path)  # ← 这步在直接调用下成功;在 run_single_arm 里卡死
```

所以 bug **不在** `precompute_scores_from_legacy` 本身,也**不在** `score_panel.to_parquet`
当独立调用时。某个东西发生在 `run_single_arm` 的特定代码顺序里。

---

## 三、可能的根因(未验证)

仔细看 `src/stockpool/portfolio_ab/runner.py:run_single_arm` 在 precompute 之后的几步:

```python
score_panel.to_parquet(score_path)                     # ← 怀疑这步在 runner 里阻塞
if score_memo is not None:
    score_memo[cache_key] = score_panel                # ← 给 score_memo 引用全 panel
portfolio_strat = PrecomputedScoreStrategy(score_panel, ...)  # ← 又一个引用
eligibility = EligibilityFilter(...)
costs = TradeCosts(...)
def _factory(): return PortfolioEngine(...)
# ...
single = _factory().run(portfolio_pool_data, start_offset=0)  # ← 或这步
```

**候选假设**(按可能性):

1. **multiprocessing.Pool 资源清理 + 大 DataFrame 同时持有的死锁**
   - 6 workers 各持有 ~5 GB pickled state,Pool.close() 后 join 时 GC 触发跨进程引用问题
   - 大 panel 在 main 和 worker 进程间通过 pickled __reduce__ 同时被引用,触发 fork-on-write 阻塞

2. **`score_panel.to_parquet` 自身被 pyarrow / fsspec 阻塞**
   - 但单跑直接版本不阻塞 → 跟 Pool teardown 时序相关?

3. **PrecomputedScoreStrategy 构造时对 score_panel 做了高成本检查**
   - 排序、validate、deep-copy 之类。15-yr × 4400 上慢但应该不是 hang。

4. **engine.run() 在 simulate 里有 O(N²) 操作**
   - 例如 sector_map 查询、rebalance loop 内不必要的全集 reindex。

### 建议的排查路径

1. **加日志确认死锁位置**(最直接):
   ```python
   # 在 runner.py 每两行加一个 log.info 探针:
   log.info("[%s] DBG: pre to_parquet", arm_name)
   score_panel.to_parquet(score_path)
   log.info("[%s] DBG: post to_parquet", arm_name)
   if score_memo is not None: ...
   log.info("[%s] DBG: post memo", arm_name)
   portfolio_strat = PrecomputedScoreStrategy(score_panel, ...)
   log.info("[%s] DBG: post strategy ctor", arm_name)
   # ... etc
   ```
   重跑 D3b 全 universe AB,看最后一条 log 在哪。预期会停在某条 DBG 后。

2. **多进程栈追踪**:卡住后,用 `py-spy dump --pid <PID>` 拿 main 和 children 的当前 Python 栈,
   能直接看是哪一行阻塞。安装:`pip install py-spy`。

3. **怀疑 Pool 是元凶**:在 `precompute_scores_from_legacy` 里 Pool 退出前后加 log,
   确认 Pool.close + join 是否完整退出。可能 worker 残留导致 main 等不到 Pool 完全 teardown。

---

## 四、影响范围

- ❌ `python -m stockpool portfolio-ab --config <yaml>` 在 15-yr 数据上**无法完成**
- ❌ `python -m stockpool portfolio-backtest --config config.yaml` 同样路径,**未测但大概率受影响**
- ❌ `docs/improvement_loop/analysis/ab_significance.py --full-market`(它内部调 run_portfolio_ab)
- ✅ `docs/improvement_loop/analysis/layer_b_direct.py` — 是这次写的绕过工具,只算 Layer B IC,
  够做 weighter 显著性研究,不够做 portfolio Sharpe AB。

**所以当前 15-yr 数据下只能做 Layer B 研究,Layer D(组合 Sharpe AB)还不可用**。

---

## 五、当前 git 状态(handoff 时)

- **分支**:`perf/portfolio-ab-speedup`
- **已 pull** `origin/perf/portfolio-ab-speedup`(包含 `FULL_MARKET_RESULTS.md` 等新文件)
- **未 commit** 的本地改动:
  - `src/stockpool/data_sources/mootdx_backend.py` — 加分页(`_fetch_paginated` + `_pages_for_total`)
  - `src/stockpool/fetcher.py` — `fetch_daily` 在 force_refresh 时计算 start 日期触发分页;
    3 个 dispatch 函数默认 source 从 `"akshare"` → `"mootdx"`
  - `src/stockpool/config.py` — 加 `SharpeWeighterConfig` / `HalfLifeICWeighterConfig` /
    `RidgeWeighterConfig`,`WeighterConfig.type` Literal 加 5 个新选项
  - `src/stockpool/ml/weighters.py` — 加 3 个新 weighter 类
  - `src/stockpool/backtesting/strategies.py` — `_build_weighter` 分发到新 type
  - `config.yaml` — `history_days: 500→3750`,`weighter.type` 仍为 `ic`(2026-06-28 试切 sharpe 后 NOT CONFIRMED 回退)
  - `CLAUDE.md` — 新 weighter 文档
  - `tests/test_fetcher.py` — 5 个 akshare 测试加 `source="akshare"`
  - `tests/test_ml_weighter_sharpe.py` / `_halflife_ic.py` / `_ridge.py` — 新增测试
  - `docs/improvement_loop/configs/D3*.yaml` / `D4-D7.yaml` — 新 AB 配置
  - `docs/improvement_loop/analysis/ab_significance.py` — 加 `--subperiods` / `--regime-boundaries`
  - `docs/improvement_loop/analysis/layer_b_direct.py` — 绕过 runner 的 Layer B 工具
- **未 commit** 的 cache(不要 commit):
  - `data/*_daily.parquet` (4398 files, ~340MB) — 15-yr cache 重建
  - `data/factor_panels/52f602173324/` — 15-yr factor panel
  - `data/portfolio_scores/*.parquet` — 15-yr score panels(ic / sharpe arms 都有,15-yr × 1000 stocks)
  - `data/top1000_liquid.parquet` — 自建 top1000 liquidity pool
  - `data/.data_source` = `mootdx`

---

## 六、修完后建议验证

1. 跑 `python -m stockpool portfolio-ab --config docs/improvement_loop/configs/D3b_sharpe_full.yaml --workers 3`
   应该完整跑完(15-yr × 4400 stocks × 2 arms),输出 `reports/portfolio_ab/<date>.html`。
2. 跑 `python docs/improvement_loop/analysis/ab_significance.py --config docs/improvement_loop/configs/D3b_sharpe_full.yaml --full-market --workers 3 --subperiods 5,8 --regime-boundaries 2015-06-15,2016-02-01,2018-06-15,2020-03-01,2022-01-01,2024-04-12`
   应该输出完整 verdict + 多粒度子段。
3. 跑 `pytest tests/test_portfolio_ab_runner.py -q`,所有 portfolio_ab 测试通过。

如果上述都过,这个 handoff 文档可以删掉。
