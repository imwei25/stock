# Tier 1 性能优化 — backtest / portfolio-backtest / AB 路径

> 状态:设计稿,等用户 review
> 适用路径:`python -m stockpool backtest` / `portfolio-backtest` / `ab` / `portfolio-ab`
> 不影响:`run`(日报)/ `fetch-universe`(I/O 路径)/ `factors analyze`

## 1. Context

`feat/rust-ops` 分支已经把 6 个最热的算子(`rank` / `ts_std` / `ts_argmax` /
`ts_argmin` / `ts_rank` / `correlation`)移到 Rust。`precompute_scores_from_legacy`
已经并行化,`_ensure_pooled_xy_long` 已经预 stack 缓存。**剩下三条路径里仍有
四块明显短板**,本 spec 一次性消除:

| 短板 | 当前现状 | 影响路径 |
|---|---|---|
| Rust ops 还差 `decay_linear` + `indneutralize` 未落地 | spec `2026-06-20-rust-ops-acceleration-design.md` PR-4 deferred | factor_panel build (B / C / D) |
| `portfolio-ab` 两 arm 串行 | `portfolio_ab/runner.py:229` `for arm_name in ab_cfg.arms` | D |
| `StaggeredRunner` N 个 offset 串行 | `portfolio/ensemble.py:71` `for k in range(n_offsets)` | C / D |
| `industry_neutralize_panel(log_mcap=...)` 联合 OLS per-day Python loop | `ml/preprocess.py:255` `for date in df.index` | B / C / D(每次 build factor_panel 都过一遍) |

四块互相独立,可以分别 PR,出问题独立回滚。

## 2. Goals

- **不改任何用户可见的语义、配置 schema、输出文件结构**。所有改动对调用者透明。
- **每块改动都有 minimal example 校验**:同一份输入,优化前后输出在指定容差内一致。
- **每块改动都可独立回滚**(env 开关 / CLI flag / 代码 revert)。
- 累计预算:**3.5-4 天工作量**;总加速预期(满足 Tier 1 假设的工作负载):
  factor_panel build -25~40%;portfolio-ab 总时间 ÷ 1.5~1.8;staggered 跑
  ÷ 3~5;preprocess(industry+log_mcap)单步 ~50× 提速。

## 3. Non-goals

- 不改 `MLFactorStrategy.generate_signals` 的 per-bar predict 调用(Tier 2 候选)。
- 不动 `precompute_scores` 的 worker 间 ML fit 共享(Tier 2 候选)。
- 不改 `score_panel` / `pool_data` 的内存布局(Tier 3 候选)。
- 不改 cache 文件 schema(`factor_panels/<sig>/manifest.json` 不变)。
- 不引入新的依赖(只用现有的 `multiprocessing.Pool` / `ProcessPoolExecutor`)。

---

## 4. 改动 #1 — 完成 Rust ops PR-4:`decay_linear` + `indneutralize`

### 4.1 设计

执行 `docs/superpowers/specs/2026-06-20-rust-ops-acceleration-design.md` PR-4
deferred 的部分。无新设计,新增 2 个 `#[pyfunction]`:

- `decay_linear(x: PyReadonlyArray2<f64>, d: usize) -> PyArray2<f64>`
  - NaN-aware 加权和 + 部分窗口权重尾对齐;权重 `1..=d`,window 内 NaN
    cell 从分子分母双重剔除。
  - `min_periods = max(1, int(d * 0.6))`;全 NaN 窗口 → NaN。
- `indneutralize(x: PyReadonlyArray2<f64>, sector_ids: PyReadonlyArray1<i32>) -> PyArray2<f64>`
  - Python wrapper 把 `{code: sector_str}` 编码成 `np.int32` sector_id 数组(连续 1..K)。
  - Rust 端:per-day 按 sector_id groupby mean,然后 broadcast 减去。
  - 不在 group_map 里的 code → 自成一组(self - self = 0,与 pandas oracle 一致)。
  - NaN cell 从 group mean 剔除,NaN cell 输出仍 NaN。

`ops.py` 中现有的 `dispatch` 函数照搬已有 6 个 op 的模式 — try import,
失败 fallback `_ops_py`,`STOCKPOOL_USE_PYTHON_OPS=1` 强制走 oracle。

### 4.2 影响范围

WQ101 中使用 `decay_linear` 的 alpha:#004, #030, #042, #060, #075, #098…
使用 `indneutralize` 的 alpha:#008, #016, #019, #020, #038, #040…

预期单一 factor_panel build 上 WQ101 部分加速 25-40%(剩余瓶颈在 `delta` /
`delay` / `ts_sum` 等轻量 op,但那是 Tier 3 候选,不纳入本 spec)。

### 4.3 校验(minimal example)

无需新工具,沿用已有 fixture:

1. **Layer A 单测** — 新增 `tests/test_ops_rust_equivalence.py::test_decay_linear_*`
   和 `::test_indneutralize_*`。每 op 至少 5 个 case:
   - happy path(T=50, N=20, d=5)
   - 全 NaN 窗口
   - 部分 NaN(mid-partial)
   - constant 输入(避免 div0 测试)
   - 单成员 group / 缺 group_map 的 code(只对 indneutralize)
   - 容差:`np.allclose(rust_out, pandas_out, atol=1e-9, rtol=1e-7, equal_nan=True)`

2. **Layer B snapshot 测** — 现有 `tests/test_ops_snapshot.py` +
   `tests/fixtures/ops_snapshot.parquet`,跑 167 个因子的中等规模 panel,逐
   factor `equal_nan=True` 比对。**这块当前已经在 CI 里跑,只需保证新 PR 不
   破坏它**。

3. **回滚开关** — `STOCKPOOL_USE_PYTHON_OPS=1` 强制全部走 pandas oracle,Rust
   产物仍编译但永不调用。

### 4.4 不变量

- `ops.py` 公开 API 签名完全不变;调用方(WQ101 + 自定义因子)零修改。
- 现有 `Cargo.toml` workspace 不动,只加 2 个 mod。

---

## 5. 改动 #2 — `portfolio_ab` 两 arm 并行

### 5.1 设计

当前 `portfolio_ab/runner.py:229`:

```python
for arm_name, override in ab_cfg.arms.items():
    arms[arm_name] = run_single_arm(arm_name, effective, ...)
```

两 arm 完全独立(arm B 不依赖 arm A 的任何状态)。改成可选并行:

```python
# run_portfolio_ab 新增参数:parallel_arms: bool = False
if parallel_arms:
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        # 把每个 arm 的所有可 pickle 参数打包,subprocess 内重跑 build_effective_cfg
        # 和 run_single_arm。pool_data / sector_map / name_map 通过 pickle 复制到
        # 子进程(代价是单 arm 内存 × 2)。
        futures = {
            arm_name: ex.submit(
                _run_single_arm_in_subprocess,
                arm_name, ab_cfg, base_cfg, override,
                pool_data, sector_map, name_map,
                refresh_scores, portfolio_pool_data, n_workers,
            )
            for arm_name, override in ab_cfg.arms.items()
        }
        arms = {n: f.result() for n, f in futures.items()}
else:
    # 旧串行路径保留,完全不变
    for arm_name, override in ab_cfg.arms.items():
        ...
```

`_run_single_arm_in_subprocess` 是模块顶层 def(可 pickle),内部调用
`build_effective_cfg` + `run_single_arm` — 即把现在的 loop body 抽成
一个顶层函数。

**CLI 入口**:`python -m stockpool portfolio-ab --parallel-arms`(默认 False,
opt-in)。`ab` 命令同样加 flag。

**内存权衡**:并行时每个 subprocess pickle 拿到独立的 `pool_data` /
`factor_panel`,内存 ~翻倍。warning 在 stdout 提示:

```
[parallel-arms] both arms running concurrently;
peak memory ~= 2× single-arm. Disable with --no-parallel-arms.
```

### 5.2 校验

**Minimal example 脚本**(临时,放在 `scripts/verify_parallel_arms.py`,验证完
即可删)。**直接调用 Python 函数,不走 subprocess**,避免文件 I/O / stdout 解析
误差:

```python
# 同一份小型 ab.yaml (2 arms, ~30 stocks, 1 年历史)
from stockpool.portfolio_ab.config import load_portfolio_ab_config
from stockpool.portfolio_ab.runner import run_portfolio_ab
from stockpool.config import load_config
from stockpool.fetcher import load_universe_cache
# ... 加载 pool_data / sector_map / name_map 一次,两边共用 ...

ab_serial   = run_portfolio_ab(ab_cfg, base_cfg, pool_data, sector_map, name_map,
                                parallel_arms=False)
ab_parallel = run_portfolio_ab(ab_cfg, base_cfg, pool_data, sector_map, name_map,
                                parallel_arms=True, refresh_scores=False)

for arm in ab_cfg.arms:
    m_s = ab_serial.arms[arm].primary_metrics
    m_p = ab_parallel.arms[arm].primary_metrics
    assert set(m_s) == set(m_p), f"{arm}: keys mismatch"
    for k in m_s:
        # engine 确定性 + score_panel 从 parquet 读 → 应 bit-exact
        assert m_s[k] == m_p[k] or (np.isnan(m_s[k]) and np.isnan(m_p[k])), (arm, k, m_s[k], m_p[k])
    # equity curve 全段比对
    pd.testing.assert_series_equal(
        ab_serial.arms[arm].primary_curve["equity"].reset_index(drop=True),
        ab_parallel.arms[arm].primary_curve["equity"].reset_index(drop=True),
        check_exact=True,
    )
```

**前提**:第二次跑前 score_panel parquet 已经被第一次跑写入磁盘
(`refresh_scores=False` 命中),所以两次都从 parquet 读同一份 score panel
→ 输入完全一致 → 输出 bit-exact。

### 5.3 不变量

- 串行路径(默认)零改动,所有现有测试不受影响。
- `ABResult` 数据结构不变;HTML 报告渲染不变。
- 并行失败(subprocess 崩)不污染另一 arm:已有 `per-arm failure isolation`
  契约由 `run_single_arm` 内 `try/except` 保证;并行模式只是把这个 try 移到
  subprocess 内。

---

## 6. 改动 #3 — `StaggeredRunner` 并行化 N 个 offset

### 6.1 设计

当前 `portfolio/ensemble.py:71`:

```python
for k in range(n_offsets):
    engine = self._engine_factory()
    results.append(engine.run(panel_data, start_offset=k))
```

N 个 offset 完全独立(spec §12 已点名 followup)。改:

```python
# 新参数:parallel: bool = False(可由调用者传入)
if parallel and n_offsets > 1:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(n_offsets, os.cpu_count() or 1),
    ) as ex:
        futures = [ex.submit(_run_one_offset, self._engine_factory, panel_data, k)
                   for k in range(n_offsets)]
        results = [f.result() for f in futures]
else:
    for k in range(n_offsets):
        engine = self._engine_factory()
        results.append(engine.run(panel_data, start_offset=k))
```

**配置入口**:在 `cfg.portfolio_backtest` 新增 **可选** `parallel_staggered: bool = False`
(opt-in,与 #2 思路一致)。CLI 不加 flag — 通过 YAML 控制。

**Pickle 注意**:`engine_factory` 在 `portfolio_ab/runner.py:173` 是 `run_single_arm`
内部的 nested def,**不可直接 pickle 传给 subprocess**。处理方式:并行模式
里**不传 factory**,改传可 pickle 的"构件包" `(strategy, portfolio_cfg, costs,
risk_free_rate, eligibility, sector_map)`,在 subprocess 顶层 helper
`_run_one_offset(components, panel_data, k)` 里**在子进程内重新构造 engine**。
StaggeredRunner 接收 `parallel: bool` 时,如果 `engine_factory` 不能拆出
构件(将来有人传 lambda),catch `TypeError` → log warning + fallback 串行。

**API 变化**:`StaggeredRunner.__init__` 新增可选参数
`components: tuple | None = None`,与 `engine_factory` 二选一。
`portfolio_ab/runner.py` 调用点同时传两者(`engine_factory` 给串行,
`components` 给并行)。`portfolio_backtest` 单跑入口同样改造。

### 6.2 校验

**Minimal example**:

1. 跑 `portfolio-backtest` 配 `staggered_starts: 3, parallel_staggered: false` →
   保存 `EnsembleResult.ensemble_curve` 到 parquet。
2. 同一 yaml 改 `parallel_staggered: true` → 同样保存。
3. 比对:

   ```python
   np.array_equal(
       a["ensemble_curve"]["equity"].values,
       b["ensemble_curve"]["equity"].values,
       equal_nan=True,
   )  # 应为 True — engine 是确定性的,数学相同
   ```

4. `aggregated_metrics` dict 严格相等。

### 6.3 不变量

- 默认 `parallel_staggered: false`,所有现有 `test_portfolio_ensemble.py`
  测试不受影响。
- `EnsembleResult` 结构不变;`individual_results` 顺序按 `k=0..N-1` 排列
  (并行模式收集后按 k 重新排序)。

---

## 7. 改动 #4 — `industry_neutralize_panel(log_mcap=...)` 批量 lstsq

### 7.1 设计

当前 `ml/preprocess.py:255`:

```python
for date in df.index:
    y = df.loc[date]
    m = log_mcap_aligned.loc[date]
    X = dummies.copy()        # ← 每天复制一次 dummies
    X["intercept"] = 1.0
    X["log_mcap"] = m.values
    resid, used_ols = _per_day_ols_residual(y, X)
    ...
```

T=1000 天 → 1000 次 `np.linalg.lstsq`,每次 ~20-50 行业 dummy 列。

**改造思路**:行业 dummies 在所有日期之间相同(只是 log_mcap 列在变)。把它
拆成两段:

1. **常量部分**:`X_const = [industry_dummies(drop_first) | intercept]` shape
   `(N, K+1)`,在所有日子里相同。预先算 `(X_const.T @ X_const)` 和 QR
   分解,只算一次。
2. **变量部分**:`log_mcap` 列每日变化。但每行的 OLS 输入仍是
   `[dummies, intercept, log_mcap]`,需要每日重 lstsq。

这里有 2 种可行路径,**默认采用 (a)**:

**路径 (a) 批量 normal equation**:
- 把 X 拼成 `(T, N, K+2)` 三维,用 `np.einsum` 批量做
  `XᵀX` (T, K+2, K+2) 和 `XᵀY` (T, K+2)。
- 用 `np.linalg.solve(XtX, XtY)` 一次性解 T 个 (K+2,) 系数。
- NaN 处理:在每个 t 上做 valid_mask,用 `np.where(valid, X, 0)` 把
  invalid 行零化(对 normal equation 等价于剔除);再单独算 `n_valid[t]`
  做 fallback 判断(< 10 或行列降秩 → fallback 到 legacy group demean)。
- 这与 `mcap_neutralize_panel` 已经向量化的做法是同一范式。

**路径 (b) 保留 lstsq 但 batch**:
- 把 T 天的 (y_t, X_t) stack 成 block-diagonal `(T·N, T·(K+2))` 矩阵 lstsq
  解一次。理论上和 (a) 等价但矩阵超大,反而慢。**放弃**。

**Fallback 路径**:
- 与现有逻辑一致 — 任何天 `< 10 有效行` 或 `rank deficient` → 用 `legacy_fallback`
  (group demean) 该天的结果。
- `legacy_fallback` 已经预先算好一份 group-demean panel,直接 `out.loc[date] = legacy_fallback.loc[date]`。

### 7.2 校验

**Minimal example**(放 `tests/test_ml_preprocess_mcap.py::test_industry_log_mcap_batch_matches_legacy`):

```python
# 合成 panel: T=50, N=200, 5 行业, log_mcap 服从 N(15, 2)
# 故意制造若干天 < 10 有效行(NaN-fill 80% codes)→ 触发 fallback
df = synthetic_panel(T=50, N=200, n_industries=5)
log_mcap = synthetic_log_mcap(T=50, N=200)
sector_map = build_sector_map(df.columns)

# 旧路径(per-day lstsq):mock 出来,临时保留旧函数
out_old = _industry_neutralize_per_day_loop(df, sector_map, log_mcap)
# 新路径(批量 normal equation)
out_new = industry_neutralize_panel(df, sector_map, log_mcap)

assert np.allclose(out_old.values, out_new.values, rtol=1e-9, atol=1e-12, equal_nan=True)
```

并且现有 fixture 测试 `test_ml_preprocess_mcap.py` 全部不变,确保覆盖
真实使用 case。

### 7.3 不变量

- 函数签名 `industry_neutralize_panel(df, sector_map, log_mcap=None)` 不变。
- `log_mcap is None` 分支(group demean)零改动。
- WARNING log 行 "OLS fallback on %d / %d days" 仍保留,fallback 计数语义
  一致。

---

## 8. PR 顺序 / 依赖关系

四个改动**相互独立**,但建议顺序:

1. **PR-T1.1** — 改动 #4(preprocess 向量化):risk 最低,纯数学;先做能给后续 PR 的 perf 测量提供更干净的 baseline。
2. **PR-T1.2** — 改动 #1(Rust ops PR-4):build 流程已经走通,有 oracle 兜底。
3. **PR-T1.3** — 改动 #3(staggered ensemble 并行):pickle 测试需要小心,但实现简单。
4. **PR-T1.4** — 改动 #2(portfolio_ab 并行):依赖 #3 验证好 pickle 流程,降低踩坑概率。

每个 PR 落地后跑一次 `tests/ -q`,确保不破坏现有 615 个测试。

## 9. 风险 / 回滚

| 风险 | 概率 | 缓解 |
|---|---|---|
| Rust `decay_linear` 部分窗口语义偏差 | 中 | Layer A 测覆盖 d=20 / len(a)=12 这种 mid-partial 边界;`STOCKPOOL_USE_PYTHON_OPS=1` 一键回滚 |
| Rust `indneutralize` sector_id 编码 off-by-one | 中 | Layer A 测覆盖空 sector / 单成员 sector / 缺 map 的 code |
| pickle 闭包失败(改动 #2、#3) | 中 | 异常 catch → log warning + 回退串行;不 raise |
| 批量 normal equation 数值不稳(改动 #4) | 低 | 与 mcap_neutralize_panel 已有的相同范式;有 fallback 兜底 |
| Windows spawn 内存爆掉(改动 #2 并行 arm) | 中 | `--parallel-arms` 默认 off;stdout warning 提示 ~2× 内存 |

每个 PR 都可独立 revert,不形成相互依赖。

## 10. 文档更新

每个 PR 同步更新:

- `CLAUDE.md` — 在对应小节(算子 / 配置 / 测试)加一行
- `README.md` — 如有用户可见的 flag(如 `--parallel-arms`)添加示例

## 11. Out of scope(已收录,留待后续 spec)

- T2.1 `precompute_scores` 主进程 ML fit 共享
- T2.2 `industry_neutralize_panel` legacy group demean 路径的 Rust 化
- T3.1 轻量 ops 批量上 Rust(`ts_sum/ts_mean/ts_min/ts_max/delta/delay`)
- T3.2 `stack_panel_to_xy` Rust 化(需先 profile 确认仍是热点)
- T4.1 `MLFactorStrategy.generate_signals` 批量 predict
- 内存优化方向(score_panel dtype 收紧 / shared_memory)
