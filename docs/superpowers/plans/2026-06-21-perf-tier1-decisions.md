# Tier 1 性能优化 — 决策日志

> 跟踪 implementer 在执行 `2026-06-21-perf-tier1.md` 时遇到的 spec 偏差,以及对应的处理决策。
> 每条记录:**日期 / Task / spec 原意 / 实际选择 / 理由**。

---

## #1 — PR-T1.2 Task 8:`correlation` 不走 Rust 调度

- **日期**:2026-06-21
- **Task**:PR-T1.2 Task 8(commit `576f8cf`)
- **Spec 原意**:在 `ops.py` 加 3 个 dispatcher(`correlation` / `decay_linear` / `indneutralize`),全部 try Rust 然后 fallback pandas。
- **实际选择**:`correlation` 改为 **无条件** 走 pandas oracle(`_py_ops.correlation`),不进 Rust 路径。
- **理由**:
  - Rust 端 `correlation` 采用 Welford 累加器,与 pandas `rolling.corr()` 在接近 ±1 相关的小窗口(d=3-8)上存在 **opposite FP overflow** — 一个落 +∞ 一个落 NaN,无法用后处理对齐。
  - 直接 dispatch 导致 `test_ops_snapshot.py` 中 21 个因子失败,`alpha_015` max_diff 飙到 2.0 / 18705 cells 不一致。
  - 单 PR 内无法 hit-fix Rust 实现;留 pandas 路径让本 PR 闭环,`correlation` 加速作为后续 spec 跟进。
- **副作用**:`ops.py` 多了 4 个文件(spec 说 2 个)— `cs.rs`(Kahan 修正)、`test_ops_snapshot.py`(alpha_076 白名单)、`CLAUDE.md`、`ops.py`。
- **审计**:用户已显式接受(对话中明确说"A — 接受所有 4 项偏差")。

---

## #2 — PR-T1.2 Task 8:`indneutralize` 用 Kahan 累加器

- **日期**:2026-06-21
- **Task**:PR-T1.2 Task 8(commit `576f8cf`,同上)
- **Spec 原意**:Task 6 落地的 Rust `indneutralize` 用 naive `+=` 累加 per-sector mean。
- **实际选择**:Task 8 顺带把 Rust 端切到 **Kahan compensated summation**(Neumaier 变种)。
- **理由**:
  - Naive `+=` 与 pandas `groupby().transform('mean')` 有 ~7e-15 ULP 漂移。
  - 该 ULP 漂移经下游 `rank()` 放大,3 个 alpha 在 snapshot 测试中失败。
  - Kahan 补偿后与 pandas bit-for-bit 一致,无后续级联误差。
- **副作用**:无 — Layer A 6 个等价性测试仍通过;snapshot 测试 167/167 通过。

---

## #3 — PR-T1.2 Task 8:`alpha_076` 加入 snapshot 白名单

- **日期**:2026-06-21
- **Task**:PR-T1.2 Task 8(commit `576f8cf`,同上)
- **Spec 原意**:不放松测试容差,任何 snapshot 失败要 raise BLOCKED。
- **实际选择**:在 `EXPECTED_RUST_DIVERGENCE` 加 `alpha_076: {atol:0.06, max_mismatches:30}`。
- **理由**:
  - `decay_linear` Rust 端 sequential `+=` 与 pandas `np.dot` 在 `ts_rank → decay_linear → ts_rank` 链上有 1-ULP 差,经第二次 `ts_rank(d=19)` 放大成 1/19=0.0526 的 rank step。
  - 影响 15 个 cell,与已有 4 个白名单条目(`e5f9259` 已加)的**机制完全相同**,不属于"silently loosening"。
  - 替代方案:把 Rust `decay_linear` 也切 Kahan / 浮点 sort + sum。代价过高(Layer A 已通过,实际下游因子是 cascade 效应)。
- **副作用**:白名单 4 → 5 entry。

---

## #4 — PR-T1.3 Task 11:并行执行器从 ProcessPool 切到 ProcessPool + relax 测试容差

- **日期**:2026-06-21
- **Task**:PR-T1.3 Task 11(commit `e38b6f3` — **将被修正**)
- **Spec 原意**:`StaggeredRunner.run(parallel=True)` 用 `ProcessPoolExecutor` 拿真并行;测试 `test_parallel_matches_serial` 用 `assert_array_equal`(bit-exact)。
- **实际选择**:**修正**为坚持 `ProcessPoolExecutor`,把测试断言从 `assert_array_equal` 换成 `assert_allclose(rtol=1e-12, atol=0)`。
- **理由**:
  - 第一版 implementer 切到 `ThreadPoolExecutor` 拿 bit-exact;但 `PortfolioEngine.run` 是 Python bar-loop(dict / pandas `.iloc[]` / Python sort),GIL-bound,**threads 拿不到真并行**,PR 等于裸跑。
  - 跨进程 `spawn` 后 BLAS 线程数 / 归约顺序可能不同 → 子 ULP 漂移(~3.3e-16)。这在金融回测里**显著小于**滑点 / 手续费 / 信号噪声,业界标准是用 `assert_allclose(rtol=1e-12)`。
  - 用户授权:对话中明确说"B — switch to ProcessPool + relax test tolerance"。
- **副作用**:本来 commit `e38b6f3` 用了 thread executor;follow-up 改 source + 测试容差,产生 1 个新 commit。

---

---

## #5 — Staggered 并行实测加速比 vs spec 估算

- **日期**:2026-06-21
- **Task**:PR-T1.3 验收(`scripts/bench_staggered_parallel.py`)
- **Spec 原意**:`parallel_staggered: true` 预期 ~3-5× wall-time 加速。
- **实测**:Windows spawn,N=5 offsets,合成 panel,n_offsets 都走 ProcessPoolExecutor:

| Panel size (codes × bars) | Single engine wall | Serial 5-offset | Parallel 5-offset | Speedup |
|---|---|---|---|---|
| 200 × 500 | ~0.16 s | 0.82 s | 1.44 s | **0.57×(并行更慢)** |
| 500 × 1500 | ~0.67 s | 3.35 s | 2.35 s | **1.42×** |
| 1000 × 2500 | ~1.73 s | 8.66 s | 4.48 s | **1.93×** |

  FP drift across processes:`max_abs ∈ [3.77e-15, 1.02e-14]`,远小于 `rtol=1e-12` 容差。

- **结论 & 调整建议**:
  - **当 single engine wall ≪ 1s 时,并行反而更慢**(subprocess spawn + pickle 全 panel 的开销 > 计算节省)。Windows spawn 比 Linux fork 更贵,Linux 上拐点可能更小。
  - 1000+ codes × 2000+ bars 时才看得到 ~2× 加速;远低于 spec 估的 3-5×(那个估算假设 fork-style 进程,且没算 IPC 开销)。
  - **真实生产场景预期**:大部分用户 `cfg.stocks` 在 16-200 之间,score panel cache 命中后 engine 单跑 1-3s,**并行加速 1-2×**。对于真正的全市场 4000+ 票场景,加速可达 2-3×。
- **行动**:在 CLAUDE.md `parallel_staggered` 说明里加一句务实的预期管理 — "实际加速比 1.5-2×,小 panel 上可能反而变慢";不撤销 PR-T1.3,在合适负载下仍有正收益。

---

## 决策原则备忘

- **战术偏差**(FP / 库选型 / 容差):recommender 选好就执行,写入本日志即可。
- **范围 / 接口 / 用户可见**改变(CLI flag / config schema / 删功能):仍找用户。
- 写日志时引用相关 commit SHA + 任何关键测试输出。
