# WQ101 A 股本土化 — 自主执行工作日志 & 决策记录(2026-06-24)

> 用户授权:夜间自主推进 `wq101-localization` 分支的 plan
> (`docs/superpowers/plans/2026-06-21-wq101-a-share-localization.md`),
> 后续决策按 Claude 推荐执行,所有决策记录于此,晨起汇总。

## 背景

- 切换前所在分支 `feat/composite-backtest`(轮15/16 WIP)→ `git stash push -u`
  暂存于 `stash@{0}`(本地,未丢失)。
- 切到本地新分支 `wq101-localization`(track `origin/wq101-localization`)。
- Plan 共 10 个 Task;入场时 Task 1–5 已由原作者提交(winsorize / 退化日检测 /
  CLI flags / Phase1 窗口盘点 / Phase2 变体生成器)。本次从 Task 6 起继续。

---

## 决策记录

### D1 — 未提交的轮15/16 WIP 处理方式 = stash(用户指定)
切分支前 `feat/composite-backtest` 有未提交改动(sign_align + monotone weighter +
文档)。**用户选择 stash**(非提交、非丢弃)。已 `git stash push -u -m "轮15/16
sign_align+monotone WIP (composite-backtest)"`。恢复方式:
`git switch feat/composite-backtest && git stash pop`。

### D2 — `reports/selection.json` 缺失修复 = 从 composite-backtest 写回磁盘(Claude 决策)
**问题**:切到 wq101 后 `factors analyze` 加载 `config.yaml` 崩溃,因 config 的
`factors_file: reports/selection.json` 指向的文件不存在。根因:该文件在
`feat/composite-backtest` 上是 .gitignore 例外被版本控制,在 wq101/main 上是
**gitignored 且未提交**;切分支时 git 删除了这个"源分支追踪、目标分支未追踪"的文件。
**决策**:`git show feat/composite-backtest:reports/selection.json > reports/selection.json`
写回磁盘(不 staging)。理由:wq101 分支本就 ignore 它,原作者也是本地放着用;
这是生产 selection,Task 8/9 的 baseline arm 需要它。**不污染分支**(git status 干净)。

---

## 执行进度

| Task | 状态 | 备注 |
|---|---|---|
| 6 自动导入变体 | ✅ 完成 | commit `6d7ae8a`;guarded import,167 因子,26 测试绿 |
| 7 Phase0 重跑基线 + Phase3 walk-forward | ⏳ 进行中 | Step1 全市场 analyze 后台跑 |
| 8 选优 (Phase 4) | ⏸ 待 | |
| 9 AB 验证 (Phase 5) | ⏸ 待 | ⚠️ 依赖 ab_pool(本分支未合,需决策回退) |
| 10 文档收尾 (Phase 7) | ⏸ 待 | |

### 预备工作(趁基线 analyze 后台跑时完成)
- 预写并验证 4 个 verbatim 脚本/测试(不依赖 analyze 产出):
  - `scripts/build_round1_factor_list.py`(Task 7 Step 4)
  - `scripts/run_walkforward_analyze.py`(Task 7 Step 5)
  - `scripts/pick_wq101_winners.py`(Task 8 Step 3)+ `tests/test_pick_wq101_winners.py`(2 测试 **绿**)
- 核实依赖 API 均存在:`build_panel_from_cache` / `analyze_factors` /
  `render_factor_analysis_report` / `FactorAnalysisResult.to_json` /
  `set_sector_map` / `load_or_build_industry_map`;`generate_wq101_variants.py`
  的 `--baseline/--top-n/--output` 匹配。
- 核实 `analyze_factors` 的 `winsorize=(0.01,0.99)` + `degenerate_threshold=0.01`
  为**默认开** → walkforward 两半段与基线 IC 口径一致(可比)。

---

## ⚠️ 阻塞点 B1 — Task 7 Step 2 基线 sanity gate 失败(调查中)

基线 analyze 完成(`reports/factor_analysis/2026-06-24.json`,4597 股 × 167 因子,~55 min)。
Step 2 验收(spec §3.0.2):

| 检查 | 期望 | 实测 | 结论 |
|---|---|---|---|
| `alpha_096` abs_ic | ≤ 0.10(原 0.4773) | **0.4049** | ❌ FAIL |
| `alpha_096` degenerate_ratio | ≥ 0.30 | **0.0** | ❌ FAIL |
| `ewma_vol_hl10` abs_ic | [0.155, 0.195] | 0.1694 | ✅ PASS |

**现象**:Phase 0 退化日检测对 alpha_096 **一天都没标记**(ratio=0.0),abs_ic 仍虚高 0.40
(全因子第 1 名,真实因子上限 ~0.17)。`abs_ic_mean` = 每日 |IC| 的均值,故 nunique 极低
日的 ±1 噪声会抬高它。plan 规定此时 **STOP and investigate**。

**初步定位**:`alpha_096 = -max(a,b)`,a/b 均为 `ts_rank` 包裹的离散分数(值域[0,1]),
不是 spec 设想的纯 `ts_argmax` 0-12 整数 → 最终输出 nunique 比预期高,
`nunique/n_valid ≤ 0.01` 的阈值**太严没罩住**。spec 的 gate 期望(degenerate_ratio≥0.30
@thr=0.01)很可能是按 Agent B 对**内层 argmax**(nunique=2)的分析定标,而非最终因子。

**为何必须修而非跳过**:alpha_096 当前是 abs_ic 全场第 1,若不修,Task 7 选 top-30 baseline
时它会被当成头号"待本土化"因子 → 整个 Round1 被幻象因子污染。

**调查中**(后台 `bjuf6sf00`):实测 alpha_096 每日 nunique/n_valid 分布 + 不同阈值下
退化日占比与 deg/clean 两组 abs_ic 对比,定标能罩住 alpha_096 phantom IC、又不误伤
ewma_vol_hl10 真因子的阈值。决策待数据回填。

### 调查结论(后台 `bjuf6sf00`)—— 根因 = 稀疏覆盖,非并列

实测 alpha_096 在 4597 股全市场:**`n_valid 每日 min=0 / 中位 3 / 最多 19`**。
即每天只有个位数股票有有效值。`nunique/n_valid` = 2/3 = 0.667(>0.01,看着"健康"),
故 ratio 检测永不触发;但在 ~3 只票上算 Spearman rank-IC ≈ ±1 纯噪声 → abs_ic 虚高
0.405。对比 `ewma_vol_hl10`:n_valid 中位 4536(满覆盖),abs_ic 0.169 真实。

**spec 的 Phase 0 诊断指标选错**:`nunique/n_valid` 只抓"并列"(离散因子),抓不到
"横截面太小"(稀疏因子)。根因:alpha_096 嵌套 adv60 + 多重 correlation,深层 NaN
传播吃掉几乎所有股票覆盖。

### 决策 D3 — 给退化日检测加"覆盖度比例"下限(Claude 决策,已实现+提交 `0e6cd62`)

**做法**:`analyze_factors` 新增 `min_coverage_frac`(默认 0.05):某日因子覆盖
< 当日**可投横截面**(有有效 close 的股票数)的该比例 → 当日 IC 置 NaN 并计入
`degenerate_day_ratio`。CLI `--min-coverage-frac`。新增回归测试
`test_analyze_factors_min_coverage_frac_flags_sparse_factor`。

**关键设计抉择 —— 用"比例"而非"绝对数"**:初版用绝对阈值 `min_valid_count=30`,
但它会把 `--universe pool`(config.stocks ~16 只)整池打死(16<30 → 全标记 →
pick-by-ic 空选,实测 break `test_factors_pick_by_ic_writes_selection`)。改为
**因子覆盖 / 当日可投票数**的比例后:
- `all`(4597):alpha_096 覆盖 ~0.1% << 5% → 全标记 → abs_ic=NaN → 自动排除出
  top-N(build/pick 脚本 filter NaN)→ Task7/8 不会去"本土化"一个幻象因子;
- `pool`(16):因子覆盖 16/16=100% → 不标记,pool 模式恢复正常(测试复绿);
- 满覆盖真因子(ewma_vol_hl10 等 0.98)完全不受影响。

**为何必须修而非跳过**:alpha_096 修前是 abs_ic 全场第 1,不修则 Task7 选 top-30
baseline 时它被当头号"待本土化"因子 → 整个 Round1 被污染。

**测试**:`tests/test_factors_analysis.py` 31 passed;之前 break 的 pool 测试复绿;
触及套件(analysis/cli/factors/wq101/picker)68 passed。两个微型 universe 测试
显式 `min_coverage_frac=0.0` 退出(它们的 16/20 股池是有意为之)。

### B1 解决中

重跑全市场基线(后台 `b76zcfdsn`,带覆盖修复,覆盖旧 buggy json)。完成后重验
Step 2 gate(预期 alpha_096 → degenerate_ratio≈1.0 / abs_ic=NaN→排除;
ewma_vol_hl10 ≈0.169 不变)。重跑产出的 `degenerate_day_ratio` 同时给出"哪些因子
被判稀疏"的全量范围(故砍掉了冗余的 coverage_audit 后台任务 `b21m0qlv0`)。

### B1 已解决 + Task 7 推进(commit `1a446a6`)

重跑基线(带 D3 修复)后 **Step 2 gate 通过**:
- `alpha_096`:abs_ic=**NaN**(排除)、degen_ratio=**1.0** ✓;`ewma_vol_hl10`=0.1694 ✓。
- top abs_ic 现由真因子领衔(ewma_vol/park_vol/gk_vol/vwap_dev)。
- **范围**:37 因子 degen_ratio≥0.30;**23/101 wq101 alpha 变 NaN(幻象,排除)**——
  深层嵌套+长窗口在 A 股吃光覆盖,本身佐证本土化命题。

Task 7 Step 3-4 完成:
- 变体生成:top-30 存活 wq101 alpha × 3 规则 = **90 变体**(0 不可转换),总因子 257。
- round1 因子表:**120**(30 baseline + 90 变体)。

### 决策 D4 — round1_factors.json 用 `git add -f` 强制纳入(Claude 决策)
`reports/` 在本分支整体 gitignore,且 git 无法对被忽略父目录内文件用 `!` 例外。
plan 指定 `reports/wq101_round1_factors.json`(及后续 winners.csv /
selection_wq101_localized.json)应提交以便复现/交接。采用 `git add -f` 外科式强制纳入
这几个 plan 指定的小产出,不改动整体 reports/ 规则(避免影响全仓 reports 处理)。

### 进行中
Task 7 Step 5:walk-forward 两半段 analyze(后台 `bhiif8hak`,120 因子 ×2,缓存热
~数分钟)。完成后 → Task 8 选优(picker 已预写测试绿),应用 spec §7.1 双半段判据
(Δabs_ic≥0.02 且 Δ|ir|≥0.1 且 degen≤0.10),spec §7.2 gate:winners≥6;若不足按
plan 记录"命题部分证伪"并继续(自主模式:我会记录决策而非停等)。

### Task 8 结果 — 0 winner(真·null,已提交 `fb560f8`)
walk-forward 两半段 picker:**0 winner**(spec §7.2 要求 ≥6)。诊断(`diag_winners.py`):
top-30 × 2 半段 = 60 instance,**无一变体 abs_ic 提升 ≥0.01**(最佳 alpha_088 +0.009);
多数规则对该 alpha 是 no-op,少数边际为负。`selection_wq101_localized.json` == 原版(0 swap)。
**判定:规则化窗口缩放对 top-30 wq101 alpha 无实质增益。**

### 决策 D5 — Task 9 AB 跳过(Claude 决策)
localized == baseline(0 swap),AB 两臂完全相同 → 无信息;且本分支无 ab_pool。
**跳过 Phase 5 AB**,不浪费 30-90 min。不修改生产 `selection.json`(spec §8.4:用户决策)。

### 决策 D6 — 不启动生产 selection 重建/AB,仅产出候选 + 上报(Claude 决策)
**衍生发现**:现役 `selection.json`(修复前 buggy IC 选)含 **4 纯幻象(alpha_027/059/061/095)
+ 7 部分退化 = ~11/30(37%)覆盖受损**。clean 基线重跑 pick-by-ic(参数假设 top-30/
max-corr0.6)得候选 `reports/selection_clean_rebuild_candidate.json`,与现役差 **15/30**。
**为何只产候选不替换**:① 替换生产选择是用户决策(spec §8.4);② 原始 pick-by-ic 参数未记录,
需用户确认;③ AB 需 ab_pool(本分支无)。→ 留候选 + 强烈建议用户用确认参数重建+AB。

---

# 晨间汇总(给用户)

## 一句话
WQ101 窗口本土化(Round 1)**证伪**(0/60 变体达标);但过程中**发现并修复了一个影响生产的真
bug**(因子分析的覆盖度盲点),这才是本次的真正收获。

## 干了什么(都已提交,分支 `wq101-localization`)
| commit | 内容 |
|---|---|
| `6d7ae8a` | Task6 变体 guarded auto-import |
| `0e6cd62` | **核心修复**:analyze 加 `min_coverage_frac` 覆盖度下限(+回归测试) |
| `1a446a6` | Task7 变体生成(90 个)+ walk-forward 驱动 + round1 因子表(120) |
| `fb560f8` | Task8 winner picker(0 winner 的 null 结果 + 证据) |
| (本次) | Task10 文档:CLAUDE.md + spec §14 + 本日志 |

## 两个发现
1. **Bug(已修)**:`analyze_factors` 退化检测只看"并列",漏掉"稀疏"。`alpha_096` 全市场每天
   仅 ~3 只有效股票,±1 噪声 IC 虚高 abs_ic 到 0.40(全场第1)。加覆盖度比例下限后,
   **23/101 wq101 alpha 被识别为幻象**。旧 IC 数字整体不可信。
2. **生产隐患(待你决策)**:现役 `selection.json` 用旧 buggy IC 选,**~11/30 因子覆盖受损
   (4 纯噪声 + 7 部分)**。

## 建议你做的决策(按优先级)
1. **【高价值】用 clean 基线重建 `selection.json` + AB 验证** —— 这是最可能提升生产的动作。
   我已留候选 `reports/selection_clean_rebuild_candidate.json`(参数为假设,请用你原始
   pick-by-ic 参数重跑确认)。需要 ab_pool 或固定池(如 config_eval48)做 AB。
2. **WQ101 本土化是否继续** —— Round 1 干净 null,**我建议停掉窗口规则方向**;Round 2
   (bottom-30)价值更低,不建议投入。功能代码已保留可 opt-in。
3. **分支合并** —— 覆盖度修复(`0e6cd62`)是独立的真 bug fix,值得**单独 cherry-pick 合回
   main**,不必等 wq101 整体结论。
4. 还有件事:本地 `feat/composite-backtest` 的轮15/16 WIP 仍在 `stash@{0}`,记得择机处理。

## 遗留/已知
- 期间有**外部进程反复 `git stash` 走我的工作树编辑**(疑似编辑器/并发工具的 format-on-save
  或 auto-stash),表现为"文件被回退";已逐次重新应用并 commit 锁定。所有交付都在 HEAD。
- **两个 stash,都未动(决策 D7:保守不 drop)**:
  - `stash@{1}` = **你的轮15/16 WIP**(sign_align+monotone,来自 feat/composite-backtest)。**保留**,
    恢复:`git switch feat/composite-backtest && git stash pop stash@{1}`。
  - `stash@{0}` = 外部进程自动创建的**垃圾回退快照**:含我已被 HEAD 取代的旧 min_valid_count 版,
    外加把 `strategies.py`/`portfolio/eligibility.py` 两处**性能优化反向撤销**(HEAD 里是正确优化版)。
    **无任何独有想要的内容,可安全 `git stash drop stash@{0}`**;我没动它以防万一。
- 全部测试在触及套件绿(analysis/cli/factors/wq101/picker 共 68+ passed);工作树干净。
