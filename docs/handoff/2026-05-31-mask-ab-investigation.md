# Handoff: mask 落地后 A/B 验证 0 trade 问题调查

> 写于 2026-05-31。前一个 claude code 进程在 mask 工作全部 commit 但**未 push**
> 时,开始调查 A/B 测试 0 trade 问题。本文记录当前进度,新进程接手用。

---

## 一、当前 Git 状态

- 分支:`feat/composite-backtest`
- HEAD: `f35db4b` (`docs(mask): update CLAUDE.md + README.md for tradability mask (task 18/18)`)
- **本地 21 个 commit 未 push** 到 `origin/feat/composite-backtest`
- 工作区未跟踪/已修改:`config.yaml`、`data_small/`、`portfolio_ab_simple*.yaml`(均与本工作无关,保留不动)
- 528 个测试全过 (`.venv/Scripts/python.exe -m pytest tests/ -q`)

## 二、Mask 工作交付清单(已完成)

按 spec `docs/superpowers/specs/2026-05-31-tradability-mask-design.md` 和 plan
`docs/superpowers/plans/2026-05-31-tradability-mask-plan.md` 完成 18 个 task:

| 模块 | 变更 |
|------|------|
| `src/stockpool/config.py` | 新增 `MaskConfig` Pydantic 类,挂到 `MLFactorConfig.mask` |
| `src/stockpool/panel.py` | 4 个 helper:`_limit_threshold` / `_listing_mask` / `compute_tradability_mask` / `apply_mask` |
| `src/stockpool/factors/ops.py` | `ts_sum/mean/std/product` `min_periods` 放宽到 60%;`decay_linear` NaN-safe 重写 |
| `src/stockpool/ml/dataset.py` | `compute_factor_panel/forward_return_panel/build_panel/build_factor_matrix` 加 `mask`/`mask_config` 参数 |
| `src/stockpool/strategy_factory.py` | `build_factor_panel/load_or_build_factor_panel` 加 `mask_config` + sig hash 包含 mask + manifest 新字段 |
| `src/stockpool/backtesting/strategies.py` | `MLFactorStrategy` 内 `build_panel` 和 `build_factor_matrix` 调用传 `mask_config=self.cfg.mask` |
| `src/stockpool/backtest_runner.py` | `prepare_pool` 传 `mask_config=cfg.strategy.ml_factor.mask` |
| `src/stockpool/cli.py` | `cmd_portfolio_backtest` 传 mask_config |
| `src/stockpool/ab/runner.py` | `_prepare_arm_pool` 传 mask_config + `_decide_pool_sharing` 拒绝跨 mask config 共享 |
| `src/stockpool/portfolio_ab/runner.py` | `run_single_arm` 传 mask_config |
| 新测试 | `tests/test_panel_mask.py` (12) + `test_ops_mask_nan_safe.py` (10) + `test_ml_strategy_mask.py` (14) + 5 in `test_config.py` + 2 in `test_factor_panel_cache.py` |
| `ab_mask.yaml` | A/B 验证配置(baseline mask=false vs with_mask mask=true) |
| `CLAUDE.md` + `README.md` | 文档更新 |

## 三、Mask 基础设施验证状态

✅ **工作正常**:
- `compute_tradability_mask` 按板块阈值(主板 ±10% / 创业板+科创 ±20%)正确计算
- `apply_mask` NaN-out 验证通过
- 算子 NaN 传播在 528 测试覆盖
- factor_panel cache sig 包含 mask_config(翻 enabled 产生不同 sig 目录)
- A/B 框架 `_decide_pool_sharing` 修了跨 mask config 共享 bug
- 真实跑 `ab_mask.yaml`:两 arm 各自 fresh build 各自 factor_panel(`a3084b45dcfe` baseline vs `db1890cee23d` with_mask)

❌ **未验证**:Sharpe Δ 是否复现论文 +0.2 假设 — **被下面的 0 trade 问题阻塞**

## 四、核心问题:A/B 跑 0 trade

### 4.1 现象

`ab_mask.yaml` 跑完两 arm 全部 16 票 `total_ret=+0.000 sharpe=+0.00`,**没有任何交易**。

### 4.2 已查明的根本原因 1:`industry_relative_strength_20` 全 NaN

**`src/stockpool/factors/custom.py:40-41`**:
```python
sector_map = get_sector_map()
if not sector_map:
    return pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
```

如果 `sector_map` 是空字典,这个 factor 整片 NaN。

**Bug 链**:
1. `factor_panels/<sig>/` cache 的 sig hash **不包含 sector_map 状态**
2. 历史上某次 build factor_panel 时 `set_sector_map` 没被调(可能那次是 `composite_verdict` 路径,或 unit test 路径)→ `industry_relative_strength_20` 算成全 NaN 落盘
3. 之后 ml_factor A/B 跑,`prepare_pool` 虽然调了 `set_sector_map(load_or_build_industry_map(...))`,但是 cache hit → 拿到**坏 panel**
4. ml_factor predict 路径(`strategies.py:527`)要求 X 行 `notna().all(axis=1)` 才预测,任何 1 个因子 NaN → 拒绝预测 → 输出 `signal=neutral, score=NaN`
5. 20 因子里有 1 个 100% NaN → 100% 的 bar 不预测 → 0 trade

**验证**:用 `refresh=True` 强制重建 factor_panel(set_sector_map 已调好后),`industry_relative_strength_20` NaN 率从 **100% → 12.7%**。

### 4.3 已查明的根本原因 2:`alpha_037` 36% warmup NaN

**`src/stockpool/factors/wq101.py:468-471`** 的 Alpha37:
```python
def compute(self, panel):
    c, o = panel["close"], panel["open"]
    return ops.rank(ops.correlation(ops.delay(o - c, 1), c, 200)) + ops.rank(o - c)
```

`correlation(..., 200)` 是 **200 日 rolling 窗口**。我们 `history_days: 500`,
**前 200 bar 必然 NaN warmup**,无法消除。这一项独立贡献 36% NaN。

### 4.4 NaN 全景(强制 refresh 后,605589 单股)

| 因子 | NaN 率 |
|------|--------|
| industry_relative_strength_20 | 12.7%(原 100%) |
| alpha_037 | **45.5%(含 36% warmup)** |
| turnover_zscore_60 | 19.8%(60 日 z-score warmup) |
| alpha_069 / alpha_082 | 14.7% |
| alpha_067 | 13.5% |
| 其他 | 9-13% |
| **任一因子 NaN 的行比例** | **45.5%** |
| **完全 valid 的行比例** | **54.5%**(原 0%) |

即便修了 cache bug,**仍有 45% 的 bar 不预测**,因为 alpha_037 200 日 warmup
+ turnover_zscore_60 + 其他因子的零散 NaN。

## 五、当前状态(进程重启前的最后操作)

- 已 `rm -rf data/factor_panels/a3084b45dcfe data/factor_panels/db1890cee23d`(stale cache)
- 已尝试重跑 `python -m stockpool ab --config ab_mask.yaml`(in background)
- bg task 报 `completed exit 0`,但输出文件只有 3 行(只到 "Building factor panel: 20 factors × 4359 stocks (sig=a3084b45dcfe)")
- **`data/factor_panels/` 仍然只有 `74cd7ff32264`(原有的)** — 新 fresh build 没落盘?
- **`reports/ab/` 没新文件**(最新还是 2026-05-24 的)

**可能原因**:
- (a) bg 跑得太慢,在 tail 捕获之前还在 build,但 exit code 0 又矛盾
- (b) 跑到一半 OOM 或 crash 但 swallow 了 exception(`_one` 函数有 try/except)
- (c) tail -120 把中间所有 log 都丢了,只剩 3 行截断

## 六、下一步建议(给新进程)

按优先级:

### Step 1:确认 A/B 是否真在跑 + 输出去向

```bash
# 不要 tail 截断,直接 redirect 到日志文件
.venv/Scripts/python.exe -m stockpool ab --config ab_mask.yaml > /tmp/ab_run.log 2>&1 &
# 然后 watch -n 5 ls data/factor_panels/  看新目录出现
```

预期:跑 2-3 分钟应该看到新 factor_panels/{两个 sig 目录} 出现 + `reports/ab/2026-05-31.html` 出现。

### Step 2:即使 fresh build 后仍 0 trade,定向修两个上游 bug

**Bug A — 修 factor_panel sig 包含 sector_map**(`src/stockpool/strategy_factory.py:_factor_panel_sig`):

把 `set_sector_map` 后 `get_sector_map()` 的 hash(或 sorted keys 长度)塞进 sig
blob。这样 sector_map 变化或缺失自动 invalidate cache。**有备选方案**:在
`IndustryRelativeStrengthFactor.compute` 里 sector_map 为空时 **raise** 而不是
silently 返回 NaN,这样 cache 不会被污染落盘(直接 build 失败让用户 fix)。

**Bug B — 修 ml_factor predict 容忍部分 NaN**(`strategies.py:527`):

当前:
```python
if bool(xi_row.notna().all(axis=1).iloc[0]):  # 任一 NaN 直接跳
    pred = ...
```

改为:
- impute NaN 为 0(z-score 后的中性值)再 predict;**或**
- 容忍少量 NaN(比如 ≤ 50% NaN 才跳过),**或**
- 在 selection.json 移除 alpha_037 等长 warmup 因子,扩 `history_days` 到 800+
  让 warmup 不占太大比例

推荐:先用最保守的 impute=0,看 0 trade → 有 trade 是不是真的。然后单独 PR
调整。

### Step 3:验证 mask Sharpe Δ

把 Bug A、B 修了之后再跑 `ab_mask.yaml`,目标看到 baseline 有 trade、with_mask
有 trade、Δ Sharpe ≥ 0.05(小样本上)。

### Step 4:push 决定

mask 工作本身**不依赖** Bug A、B 的修复 — mask 在测试里全跑通,functional 正确。

两个选择:
- **A**:先 push mask 工作(21 commit 上去),Bug A/B 单独 PR
- **B**:Bug A/B 修了再一起 push

如果走 A,push 后开 issue/spec 记录 Bug A、B 跟进。我倾向 **A**(mask 本身完整且
有清晰边界,Bug 是 pre-existing 不应 block mask merge)。

## 七、关键文件和命令速查

```bash
# 当前未 push 的 commits
git log --oneline a9bf126..HEAD

# 全测试
.venv/Scripts/python.exe -m pytest tests/ -q

# A/B(0 trade 现象)
.venv/Scripts/python.exe -m stockpool ab --config ab_mask.yaml

# 强制重建 factor_panel
.venv/Scripts/python.exe -m stockpool backtest --config config.yaml --refresh-factor-panel

# 看 cached quantile(在 ml_models 里)
.venv/Scripts/python.exe -c "
import pickle, pathlib
for p in pathlib.Path('data/ml_models').glob('*_shared.pkl'):
    d = pickle.load(open(p, 'rb'))
    print(p.name, d.get('quantiles'))
"

# 看 605589 因子 NaN 率
.venv/Scripts/python.exe -c "
import warnings; warnings.filterwarnings('ignore')
from stockpool.config import load_config
from stockpool.fetcher import load_universe_cache
from stockpool.strategy_factory import load_or_build_factor_panel
from stockpool.ml.dataset import slice_stock_factor_matrix
from stockpool.factors.context import set_sector_map
from stockpool.industry_map import load_or_build_industry_map

cfg = load_config('config.yaml')
pool = load_universe_cache(cfg.data.cache_dir, history_days=cfg.data.history_days)
set_sector_map(load_or_build_industry_map(cfg.data.cache_dir, source='auto'))
fp, _ = load_or_build_factor_panel(
    cfg.strategy.ml_factor.factors, pool, cache_dir=cfg.data.cache_dir,
    refresh=False,  # 改 True 强制重建
    mask_config=cfg.strategy.ml_factor.mask,
)
X = slice_stock_factor_matrix(fp, '605589')
print('any-NaN row ratio:', X.isna().any(axis=1).mean())
for f in X.columns:
    print(f'  {f}: {X[f].isna().mean():.1%}')
"
```

## 八、相关文件路径

- Spec: `docs/superpowers/specs/2026-05-31-tradability-mask-design.md`
- Plan: `docs/superpowers/plans/2026-05-31-tradability-mask-plan.md`
- 综述对照: `docs/research/2026-05-31-a-share-quant-survey-comparison.md`
- A/B config: `ab_mask.yaml`(项目根)
- 选定因子: `reports/selection.json`(20 个因子,含 alpha_037 等长 warmup)

## 九、记忆

新进程开起来读 `MEMORY.md`,关键 feedback:
- 配置改造默认保留旧实现为可选 type,新方案做 default 但可切回
- 用 `.venv/Scripts/python.exe`,不用全局 anaconda
