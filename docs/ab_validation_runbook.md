# A/B 验证 Runbook(2026-05-24)

> 给另一个 Claude Code 进程读。本文自包含:列出 F2 / PR-A 工作完成后亟需验证的 A/B 对照、每个对照的 `ab.yaml` 全文、跑法、验收标准。

## 1. 上下文

过去几次 PR 落地了 strategy_improvement_2026.md 中的 F2 全部内容,但**所有改进都没有用真实数据 A/B 验证**:

- **F2 PR-A**(`8a59c40` 系列)— walk-forward embargo + label_type 接口 + Lasso 子段化。默认 `embargo_days=None` 自动 = horizon,理论上消除标签泄露,但没在真实回测中量化过收益。
- **F2 PR-B1**(`2029bb4` 系列)— LightGBMSelector + 默认 `selector.type="lightgbm"`。非线性选因子。同样没实跑验证。
- **F2 PR-B2**(`1b72f70` 系列)— LightGBMWeighter + 默认 `weighter.type="lightgbm"` + `contributions()` 多态。默认现在是 LGB+LGB 全非线性。同样没实跑验证。

用户当前 `config.yaml` 显式 pin 在 `selector.type: lasso` + `weighter.type: ic`,即 PR-A 末态(Lasso+IC linear baseline)。

A/B 工具(`stockpool ab` 子命令)已经构建并通过 327 个测试。本 runbook 利用它把上面三轮 PR 的实际收益跑出来。

**目标**:每个对照得出 4 类结论之一:
- ✅ **真收益**(B 比 A 改善 sharpe ≥ 10% 或 max_dd 收窄 ≥ 10%,且 ≥ 60% 股票上 B 胜)
- ⚠️ **持平**(差异在 ±10% 内)
- ❌ **倒退**(B 比 A 显著差)
- 🚫 **跑不通**(数据/缓存/配置问题)

---

## 2. A/B 工具速查

完整 spec 在 `docs/superpowers/specs/2026-05-24-ab-testing-design.md`,这里给最常用的部分。

### 2.1 CLI

```bash
# 跑一次 A/B
python -m stockpool ab --config <ab_yaml_path>

# 单 arm 调试(只跑 arm_a,stdout 打印 per-stock metrics)
python -m stockpool ab --config <ab_yaml_path> --arm <arm_name>

# 强制每 arm 各跑各的,不共享 universe / factor panel
python -m stockpool ab --config <ab_yaml_path> --no-share-pool

# 强制重新拉数据(bypass cache)
python -m stockpool ab --config <ab_yaml_path> --refresh
```

退出码:
- `0` — 成功(包括 arm 内部 per-stock 失败,只要至少一只股成)
- `1` — 两个 arm 都 0 只股成功
- `2` — 配置错误(`--arm` 未知名 / `ab.yaml` 校验失败 / base_config 不存在)

### 2.2 `ab.yaml` schema(对照 spec §3.4)

```yaml
base_config: config.yaml              # 必填,base AppConfig,相对 ab.yaml 目录
stocks_filter: ["605589", "603986"]   # 可选,subset of base.stocks。空 = 全部
arms:                                  # 必填,**恰好 2 个** key
  arm_name_a:                          # arm 名字 free-form,在报告里原样显示
    strategy: ...                      # 必填,整个替换 base.strategy
    backtest:                          # 必填
      equity_curve_holding_days: [N]   # 必填,长度严格 == 1
      # 其他 backtest 字段(engine/costs/position_size/...) 不写则继承
  arm_name_b:
    strategy: ...
    backtest:
      equity_curve_holding_days: [N]
```

**关键约束**:
- `arms` 字典必须 **exactly 2 个 key**(Pydantic 强制)
- `equity_curve_holding_days` 必须 **length-1**(`[10]`,不能 `[5, 10]`)
- `arm.strategy` **整体替换**,不做字段级合并(避免半残配置)
- `arm.backtest` **部分继承**:未写的字段从 `base.backtest` 继承
- `stocks_filter` 必须是 `base.stocks` 的**子集**(只能减,不能加)
- 顶层字段(`data` / `indicators` / `weights` / `report` / `recommend_pool` 等)**不能被 arm 覆盖**

### 2.3 输出

```
reports/ab/
  2026-05-24/run.log              # 日志
  2026-05-24.html                 # 报告
  latest.html                     # 上面这个文件的拷贝
```

报告内容(自上而下):
1. **Metadata banner** — arm 名 / 各 arm 的 override / base config 哈希 / 股票数 / 成败计数
2. **Aggregate diff 表** — 每个 metric 一行,列: A 均值 / B 均值 / Δ / A 胜 / B 胜(仅 both-success 的股)
3. **Sharpe scatter** — x=A.sharpe, y=B.sharpe, 每只票一个点,y=x 参考线;点在线上方 = B 胜
4. **Sharpe diff histogram** — `B.sharpe - A.sharpe` 的分布
5. **Per-stock cards**(前 3 个默认展开,其余折叠)— 每股 equity curve(A/B/B&H 三条)+ side-by-side metrics 表
6. **失败列表 + 完整 effective config dump**(都折叠)

### 2.4 关键解读注意

- **比较 metrics 用 both-success 的股**(每只票在 A 和 B 都跑通才进 aggregate)
- **max_drawdown 是越小越好**(Δ 颜色逻辑已反向)
- **sharpe 是越大越好**;**回测样本相关性强,不要把 Δsharpe>0 直接当统计显著**,看 16 只票中**多少只胜**更稳健(scatter / wins 数)
- **trade_count** 显示但不计胜负(只是个观察指标,不是优化目标)

### 2.5 必备前置条件

每次跑 A/B 前确保:

1. **每只 `cfg.stocks` 的 daily parquet 在 `data/` 下存在**
   - 验证:`ls data/*_daily.parquet | wc -l` ≥ 16
   - 缺则:`python -m stockpool fetch-universe` 或 `python -m stockpool run --skip-trading-day-check` 拉一次

2. **`training_universe=all` 的对照** 还需要 `data/universe.parquet`
   - 验证:`ls data/universe.parquet`
   - 缺则:`python -m stockpool fetch-universe`

3. **指数 K 线**(`idx_sh000001.parquet` 等)用于 LGB 训练时不需要,但 `composite_verdict` 计算"市场背景"会用,缺会 warning 但不影响

---

## 3. 待验证的 A/B 对照清单

按优先级(P0 最高)。每个对照独立可跑;建议按 P0 → P3 顺序,前面跑通了再跑后面。

### P0-1: composite_verdict vs ml_factor LGB+LGB(规则 vs 完全非线性 ML)

**假设**:F2 PR-B2 默认的 LGB+LGB 比 PR-A 之前的纯规则策略 `composite_verdict` 表现更好。

**`docs/ab_runs/p0_1_composite_vs_lgb.yaml`**:

```yaml
base_config: ../../config.yaml
arms:
  composite_verdict:
    strategy:
      name: composite_verdict
    backtest:
      equity_curve_holding_days: [10]
  ml_factor_lgb_lgb:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        selector:
          type: lightgbm
        weighter:
          type: lightgbm
    backtest:
      equity_curve_holding_days: [10]
```

**注意**:为了让 ml 端公平,这里用 `panel_mode: pooled, training_universe: pool` —— 还是 cfg.stocks 内部 pool,不需要 fetch-universe。

**跑法**:
```bash
mkdir -p docs/ab_runs
# (复制上面 yaml 内容到 docs/ab_runs/p0_1_composite_vs_lgb.yaml)
python -m stockpool ab --config docs/ab_runs/p0_1_composite_vs_lgb.yaml
```

**验收**(看 `reports/ab/latest.html`):
- ✅ **真收益**:`ml_factor_lgb_lgb` 在 ≥10/16 只股的 sharpe 上胜过 `composite_verdict`,且 aggregate sharpe Δ ≥ +0.2
- ⚠️ **持平**:7-9/16 胜,或 sharpe Δ 在 ±0.1
- ❌ **倒退**:≤6/16 胜,或 sharpe Δ ≤ -0.2

记录:`aggregate sharpe`、`B wins (sharpe)` count、`max_dd` 是否同向。

---

### P0-2: Lasso+IC vs LGB+LGB(F2 整体收益)

**假设**:整个 F2(PR-A baseline → PR-B2 默认)的累积收益。

**`docs/ab_runs/p0_2_lasso_ic_vs_lgb_lgb.yaml`**:

```yaml
base_config: ../../config.yaml
arms:
  lasso_ic_baseline:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        embargo_days: 0          # 关键:还原 PR-A 之前的旧行为
        selector:
          type: lasso
          lasso: {alpha: 0.001, max_iter: 1000, tol: 1.0e-6}
        weighter:
          type: ic
          ic: {use_rank: true, min_abs_ic: 0.0}
    backtest:
      equity_curve_holding_days: [10]
  lgb_lgb_default:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        # embargo_days 默认 None(auto = horizon)
        selector:
          type: lightgbm
        weighter:
          type: lightgbm
    backtest:
      equity_curve_holding_days: [10]
```

**跑法**:
```bash
python -m stockpool ab --config docs/ab_runs/p0_2_lasso_ic_vs_lgb_lgb.yaml
```

**验收**(同 P0-1 verdict 标准):
- 这是 F2 整体增量的"end-to-end 视图"
- 即使倒退也有价值(说明 F2 在这股票池/这数据窗下没带来收益,需要重新看默认值)

---

### P1-1: Lasso+IC vs LGB+IC(F2 PR-B1 单独 LGB selector 的增量)

**假设**:LGB selector 单独(还配 IC 加权)比纯 Lasso+IC 好。

**`docs/ab_runs/p1_1_lasso_vs_lgb_selector.yaml`**:

```yaml
base_config: ../../config.yaml
arms:
  lasso_ic:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        embargo_days: 0
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
  lgb_ic:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        embargo_days: 0
        selector: {type: lightgbm}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
```

**验收**:LGB selector 的边际价值。如果这一步已经达到 P0-2 的差异,P1-2 就是"锦上添花";如果这一步差异不大,说明 LGB selector 单独贡献小,价值在 LGB weighter。

---

### P1-2: LGB+IC vs LGB+LGB(F2 PR-B2 单独 LGB weighter 的增量)

**假设**:在 LGB selector 已经选好因子的基础上,LGB weighter 替代 IC 线性加权能进一步改善。

**`docs/ab_runs/p1_2_lgb_ic_vs_lgb_lgb.yaml`**:

```yaml
base_config: ../../config.yaml
arms:
  lgb_ic:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        embargo_days: 0
        selector: {type: lightgbm}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
  lgb_lgb:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        embargo_days: 0
        selector: {type: lightgbm}
        weighter: {type: lightgbm}
    backtest:
      equity_curve_holding_days: [10]
```

**验收**:LGB weighter 单独的边际价值。结合 P1-1 + P1-2 = P0-2,可以拆解 F2 整体收益的来源。

---

### P2-1: embargo_days=0 vs embargo_days=null(PR-A walk-forward 标签泄露修复)

**假设**:`embargo_days=None`(auto=horizon)消除标签泄露,真实回测中体现为更稳定的 OOS sharpe(不一定更高,但应该不显著恶化;如果纯 IC 路径下 embargo 反而让 sharpe 显著降低,说明"泄露"在小样本上反而是"特征")。

**`docs/ab_runs/p2_1_embargo_off_vs_on.yaml`**:

```yaml
base_config: ../../config.yaml
arms:
  no_embargo:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        embargo_days: 0          # 关:旧行为
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
  with_embargo:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        # embargo_days: None 默认 = horizon
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
```

**验收**:
- ✅ **embargo 帮助**:`with_embargo` sharpe Δ ≥ -0.05 且 max_dd 收窄(消除了泄露依赖)
- ⚠️ **持平**:Δsharpe 在 ±0.05 内
- ❌ **embargo 害事**:Δsharpe ≤ -0.1(说明这窗口下"泄露"是 spurious 信号被消除,但策略其他依赖那点未来信息;需要单独研究)

注:这是结构性 bug fix,不一定带来 OOS sharpe 提升;但如果 sharpe 大幅下跌,说明小样本里"假信号"恰好在 OOS 也成立 —— 是不稳定的福利。

---

### P3-1: panel_mode per_stock vs pooled(池化训练价值)

**假设**:pooled 模式让 cross-sectional 因子有意义,且每个 refit 训练样本更多 → 更稳定。

**`docs/ab_runs/p3_1_per_stock_vs_pooled.yaml`**:

```yaml
base_config: ../../config.yaml
arms:
  per_stock:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: per_stock
        embargo_days: 0
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
  pooled:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        embargo_days: 0
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
```

**验收**:pooled vs per_stock 的相对优势,文献上通常 pooled 有优势但波动也大。

---

### P3-2: training_universe pool vs all(全市场训练的价值)— **需要 fetch-universe 已跑过**

**假设**:用全市场 ~4350 只票训练,模型见过更多市场状态,OOS 比 pool(只 16 只票)训练更好。

**前置**:
```bash
ls data/universe.parquet           # 必须存在
ls data/*_daily.parquet | wc -l    # 应该 ≥ 4000
```

**`docs/ab_runs/p3_2_pool_vs_all_universe.yaml`**:

```yaml
base_config: ../../config.yaml
arms:
  train_universe_pool:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: pool
        share_pool_fit: true
        embargo_days: 0
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
  train_universe_all:
    strategy:
      name: ml_factor
      ml_factor:
        factors:
          - momentum_20
          - macd_hist
          - rsi_centered_14
          - ma_distance_20
          - vol_ratio_5
          - boll_position_20
          - ma_slope_20_5
          - kdj_j
        horizon: 3
        train_window: 250
        min_train_samples: 60
        refit_every: 20
        panel_mode: pooled
        training_universe: all        # 关键
        share_pool_fit: true
        embargo_days: 0
        selector: {type: lasso}
        weighter: {type: ic}
    backtest:
      equity_curve_holding_days: [10]
```

**注意**:`training_universe: all` 会加载所有 4350 只缓存股票一次性进内存,LGB 训练会显著变慢(单次 refit 几十秒到一两分钟)。**估计跑完整个 A/B 需要 10-20 分钟**(对照 pool 路径几秒一次 refit)。如果时间不允许,可以先把 stocks_filter 减到 3 只(`stocks_filter: ["605589", "603986", "000528"]`),减少 backtest 次数到 6 个 stock-arm 组合。

**验收**:训练规模的价值。如果全市场训练显著胜出,意味着默认 `training_universe` 应该切到 `all`(strategy doc 也这么建议)。

---

## 4. 推荐跑法顺序

```bash
# 1. 确认前置条件
ls data/*_daily.parquet | wc -l    # ≥ 16
ls data/universe.parquet           # 存在(为 P3-2)

# 2. 创建 ab_runs 目录,把 §3 的 6 个 yaml 都保存进去
mkdir -p docs/ab_runs
# (粘贴 P0-1 到 P3-2 的 6 个 yaml 文件)

# 3. 按优先级跑(每次跑完看一下 reports/ab/latest.html,记录结论)
python -m stockpool ab --config docs/ab_runs/p0_1_composite_vs_lgb.yaml
# (备份:cp reports/ab/latest.html reports/ab/p0_1.html)
python -m stockpool ab --config docs/ab_runs/p0_2_lasso_ic_vs_lgb_lgb.yaml
# (备份:cp reports/ab/latest.html reports/ab/p0_2.html)
python -m stockpool ab --config docs/ab_runs/p1_1_lasso_vs_lgb_selector.yaml
# cp reports/ab/latest.html reports/ab/p1_1.html
python -m stockpool ab --config docs/ab_runs/p1_2_lgb_ic_vs_lgb_lgb.yaml
# cp reports/ab/latest.html reports/ab/p1_2.html
python -m stockpool ab --config docs/ab_runs/p2_1_embargo_off_vs_on.yaml
# cp reports/ab/latest.html reports/ab/p2_1.html
python -m stockpool ab --config docs/ab_runs/p3_1_per_stock_vs_pooled.yaml
# cp reports/ab/latest.html reports/ab/p3_1.html
python -m stockpool ab --config docs/ab_runs/p3_2_pool_vs_all_universe.yaml
# cp reports/ab/latest.html reports/ab/p3_2.html
```

**每次跑完写一个一段话总结**到 `docs/ab_validation_results.md`(新建),格式:

```markdown
## P0-1: composite_verdict vs ml_factor LGB+LGB
- Date: 2026-05-24
- Stocks: 16 (all base.stocks)
- A: composite_verdict
- B: ml_factor LGB+LGB (pool + builtin 8 factors)
- aggregate sharpe Δ: <Δ value>
- B wins: <n/16>
- max_dd Δ (B - A, less is better): <value>
- Verdict: ✅ / ⚠️ / ❌
- Notes: <anything anomalous in per-stock cards>
```

跑完 6 个对照,docs/ab_validation_results.md 应该长这样:

```
P0-1: <verdict>
P0-2: <verdict>
P1-1: <verdict>
P1-2: <verdict>
P2-1: <verdict>
P3-1: <verdict>
P3-2: <verdict>
```

---

## 5. 常见 pitfall

### 5.1 `FileNotFoundError: reports\selection.json`

如果用了 `factors_file: reports/selection.json` 但还没生成那个文件,load `ab.yaml` 即崩。本 runbook 所有 yaml 都用显式 `factors: [...]`,不依赖 selection.json,所以应该不会撞这个。

### 5.2 `FileNotFoundError: data/<code>_daily.parquet`

某只 cfg.stocks 的缓存还没拉。两种修法:
- 跑 `python -m stockpool run --skip-trading-day-check` 拉一次(会把所有 cfg.stocks 缓存补齐)
- 或者在 ab.yaml 加 `stocks_filter` 跳过那只票

### 5.3 训练样本不足导致 arm 全失败

如果 `cfg.stocks` 历史短 + `min_train_samples=60` + `panel_mode=per_stock` + `embargo_days=null`(auto=3),实际可用样本 = `history_days - factor_warmup(20) - 2*horizon(6) ≈ 474`,而 train_window=250 + min_train_samples=60 都能满足,所以一般不会出问题。但如果跑出 "Arm X: 0 done, N failed",到 `reports/ab/<date>/run.log` 看 stack trace。

### 5.4 第一次跑很慢,第二次秒回

`ml_models/<sig>_<code>.pkl` 缓存了 walk-forward 月度训练,第一次跑要从头训,第二次同 config 同股直接读缓存。如果你想验证"真实从头训练的耗时"用 `--refresh`。

### 5.5 Sharpe Δ 单点决定结论 — **不要**

A/B 报告里看到 `Δsharpe = +0.3` 别立刻下定论。看:
1. **多少只股 B 胜** — `B wins` count(16 只里至少 10 只)
2. **histogram 形状** — 应该有体感:多数股 Δ 集中在某一侧
3. **scatter plot** — 是不是有少数离群点拉飞了均值

### 5.6 PR-B1 / PR-B2 默认值"激进切到 LGB"未必赢

LGB 在 ~250 bars × N 股的小训练集上易过拟合(README 已经写明这条 caveat)。如果 P0-1 / P0-2 显示 LGB+LGB 倒退,这是已知风险,不是工具或代码 bug。下一步该回去看 `num_leaves` / `min_data_in_leaf` / `num_iterations` 是不是要调更保守,或者把默认切回 Lasso+IC。

---

## 6. 完成后

把 `docs/ab_validation_results.md` 的 6 个 verdict 汇报给用户。根据结果:

- 若 P0-2 显示 LGB+LGB 倒退 → **建议** F2 默认值需要重新审视(可能要回到 `selector.type=lasso, weighter.type=ic` 作为默认)
- 若 P2-1 显示 embargo 显著伤 sharpe → **建议** 看 backtest 时间窗,可能是回测窗口太短 / 太特殊
- 若 P3-2 显示全市场训练显著胜出 → **建议** 把 `training_universe` 默认切到 `all`

无论结果如何,这些对照结果**应该被加进** `docs/strategy_improvement_2026.md` 的 §6 路线图作为已完成验证的标记,推动 F3 / F1 plan-2 的优先级排序。
