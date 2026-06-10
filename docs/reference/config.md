# 配置参考 (`config.yaml`)

> 所有字段由 `config.py:AppConfig` 校验。**新增/改字段必须同步更新 `config.py` 和本文件。**

## 顶层结构

- `stocks` — 股票池,每条含 `code` / `name` / `sector`(sector 可填行业中文名或 6 位 TDX 88xxxx 代码)
- `data` — `history_days` / `cache_dir` / `force_refresh` / **`source`**(`mootdx` 默认 / `baostock` / `akshare`)
- `indicators` — MA 周期、MACD/KDJ/BOLL 参数
- `weights` — 各信号触发的得分
- `scoring` — `daily_weight` / `weekly_weight` / `resonance_bonus`(共振奖励)
- `verdicts` — `strong_buy` / `buy` / `sell` / `strong_sell` 分数阈值
- `backtest` — 见 [backtest 段](#backtest-段)
- `context` — `indices`(大盘指数列表,默认上证/深证成指)
- `report` — `output_dir` / `keep_history` / `klines_to_show`
- `strategy` — 见 [strategy 段](#strategy-段)
- `portfolio_backtest` — 见 [portfolio_backtest 段](#portfolio_backtest-段)
- `recommend_pool` — 见 [recommend_pool 段](#recommend_pool-段)

## data 段

切换 `source` **自动 force_refresh**:`cache_dir/.data_source` 记录上次 source,`fetch_daily` / `fetch_index_daily` / `fetch_sector_daily` / `fetch_universe` 入口比对,不一致直接丢弃旧 parquet 重拉(volume 单位 mootdx=手 / baostock=股,混源会污染相对成交量指标)。

行业板块仅在 `source=akshare` 时走东财,其他两种 source 都用 mootdx 的通达信行业指数 (88xxxx);名字→代码映射见 `mootdx_backend._TDX_INDUSTRY_CODES`,也可在 `stocks[].sector` 直接填 6 位 TDX 代码。

## backtest 段

- `forward_days` / `equity_curve_holding_days` / `risk_free_rate` / `costs`
- **`engine`**(`multi_lot` 默认 / `single`)
- **`sizing`** — `type: fixed | vol_target`(默认 `vol_target`);`fixed.size` 是 vol_target 公式的 baseline 锚点。详见 [conventions.md](conventions.md) Sizing 段
- **~~`position_size`~~** — deprecated alias of `sizing.fixed.size`(自动迁移 + DeprecationWarning)
- **`max_concurrent_lots`**

## strategy 段

`name`(`composite_verdict` 默认 / `ml_factor`)+ `ml_factor` 子配置。

### ml_factor 子配置

| 字段 | 默认 | 说明 |
|---|---|---|
| `factors` / `factors_file` | — | 因子列表 或 HTML picker 导出的 JSON,二选一 |
| `horizon` | — | 前向收益天数 |
| `train_window` / `refit_every` | — | walk-forward 窗口 |
| `panel_mode` | — | `per_stock` / `pooled`(WQ101 cross-sec 必须 pooled) |
| `training_universe` | `pool` | `pool`(只用 cfg.stocks)/ `all`(全市场 cache,需先 `fetch-universe`;仅 `panel_mode=pooled` 生效) |
| `share_pool_fit` | `true` | 仅 `pooled` 生效:跨股共享 fit,缓存键 `(sig, year, month)`;训练集不剔除 host,host 以 ~1/N 权重进入自己训练 |
| `embargo_days` | `null`=auto=`horizon` | walk-forward 训练/测试间隔,消除 horizon 日前向收益标签泄露;设 `0` 回到 pre-PR-A 行为 |
| `label_type` | `"return"` | 训练标签变换;`"vol_adjusted"` / `"cross_sec_rank"` 占位 raise `NotImplementedError` |
| `selector.{lasso\|lightgbm}` | `type: "lasso"` | `lasso.{alpha,max_iter,tol}` 或 `lightgbm.{num_leaves,...,top_k_factors,min_importance_ratio}` 子段二选一,顶层扁平字段被拒 |
| `weighter.{ic\|ir\|equal\|lightgbm}` | `type: "ic"` | `ic.{use_rank,min_abs_ic}` / `ir.{n_chunks,use_rank,min_abs_ir}` / `equal` / `lightgbm.{...}` 四选一 |
| `thresholds` / `*_verdicts` | — | 分位数→判定映射 |
| `preprocess` | — | 见 [preprocess 子段](#preprocess-子段) |
| `mask` | — | 见 [mask 子段](#mask-子段) |

**LGB 默认回退说明**:`selector.type` / `weighter.type` 默认 2026-05-24 从 `"lightgbm"` 回退到 `"lasso"` / `"ic"`(LGB+LGB 在 16 股 × 500bar baseline 上 sharpe 退 0.2 / return 退 20%,见 `docs/ab_validation_results.md`)。LGB 仍可 opt-in,但需先调超参或扩股池验证。

**缓存失效**:改 factors/horizon/selector/weighter/thresholds/embargo/label_type/training_universe/share_pool_fit 任一项 → `<sig>` 变化 → 旧 ml_models pkl 自动失效。

### preprocess 子段

截面预处理流水线 `preprocess.{winsorize, zscore, industry_neutralize, market_cap_neutralize, symmetric_orthogonalize, min_pool_size}`。baked 进 `factor_panels/<sig>/` cache(preprocess 入 hash),改 preprocess → 自动新 sig 重算。

| 字段 | 默认 | 说明 + A/B 结论 |
|---|---|---|
| `winsorize` | `null` | 每日截面去极值(如 `[0.01, 0.99]`)。P4-1b 验证默认开 |
| `zscore` | `false` | 每日截面 z-score。P4-1b 验证默认开(+0.245 sharpe / +2.64% return on 4358 票) |
| `industry_neutralize` | `false` | 行业内 demean。**实测有害**(与 `industry_relative_strength` 因子冲突 + 单成员细分行业 silent demean-to-zero),默认关 |
| `market_cap_neutralize` | `true` | 每日截面对 log(总市值) OLS 残差化。**P4-3 PASS**(Δsharpe +0.156),默认开 |
| `symmetric_orthogonalize` | `false` | 流水线最后一步,逐日截面对称(Löwdin)正交化去相关。**P4-4 NEUTRAL**(Δsharpe +0.007),默认关 |
| `min_pool_size` | `200` | runtime guard,n_codes < 此值时四步全跳 + warning |

**market_cap_neutralize 实现**:mcap = `close × totalShare`(totalShare 取 `data/mcap_shares.parquet` 最新快照,静态广播,`scripts/pull_mcap_profit.py` 拉)。需 caller 注入 log_mcap 面板(prepare_pool/cli/ab.runner 已 wire);**无 `data/mcap_shares.parquet` 时静默跳过 + warning**。与 industry_neutralize 同属中性化步,均跳过 types 含 "fundamental" 的因子。

**symmetric_orthogonalize 实现**:`F_std · M^(-1/2)`(M = z-score 截面相关矩阵,`eigh` + 特征值 floor 1e-10),order-independent;退化日(N_valid < K)passthrough,近奇异靠 floor 不爆 NaN,fundamental 因子跳过。stateless per-day → look-ahead safe。之后 selector/weighter 照常跑在去相关后的因子上。归因:`selection.json` 因子已 `pick-by-ic`(max_corr)+ Lasso 筛过,冗余有限 → 换高相关原始因子或扩大应用池值得重测。

A/B 配置:`ab_neutralize.yaml`(industry vs market_cap)/ `ab_neutralize_confirm.yaml`(base vs market_cap)/ `ab_orthogonalize_small.yaml`(smoke)/ `ab_orthogonalize.yaml`(full)。详见 `docs/ab_validation_results.md` P4-1b/P4-2/P4-3/P4-4。

### mask 子段

tradability mask(default 关闭,opt-in)。`enabled`(default `false`)/ `limit_up_threshold_main`(主板 0.098)/ `limit_up_threshold_chinext`(创业板+科创 0.198)/ `limit_up_threshold_bse`(北交所 0.298)/ `min_listing_days`(252)。

**应用范围(2026-05-31 重构)**:mask **仅** 作用于训练标签层(`forward_return_panel` 双向检查 `mask[t] ∧ mask[t+horizon]`,涨跌停/停牌/新股头 N 天产生 NaN 标签,经 `stack_panel_to_xy` dropna 剔除)。**不破坏因子输入** — 时间序列因子看到原始 close。`factor_panels/<sig>/` 缓存与 mask 状态**解耦**;`ml_models/<sig>_*.pkl` 含 mask 在 cfg hash 里。`mask.enabled=true` 时 `MLFactorStrategy` 自动加载 IPO 日期传给 `_listing_mask`。源自论文 B(arXiv 2507.07107)mask-first finding,详见 `docs/handoff/2026-05-31-mask-ab-investigation.md`。

## portfolio_backtest 段

Portfolio 级回测(PR-1 新增,默认 `enabled: false`)。CLI `portfolio-backtest` 在 `enabled=false` 时退出 2。

```yaml
portfolio_backtest:
  enabled: false                    # opt-in
  portfolio:
    top_k: 20                       # 每次取分数前 K 等权
    rebalance_n_days: 5             # 每 N bar 调仓
    max_per_industry: 5             # 同行业最多持仓(PR-2 起生效;null 关闭)
    initial_cash: 1.0
  eligibility:                      # 逐 bar 漏斗(PR-2 起 engine 实际读取)
    min_avg_amount_20d: 5e7         # 最近 20 bar 均成交额下限 (close*volume*100)
    exclude_st: true
    min_history_bars: 60
  staggered_starts: 1               # >1 自动走 ensemble(PR-3)
  score_cache_dir: data/portfolio_scores  # 缓存键 = cfg.content_hash
  universe_codes: null              # 见下
```

**PR-2 行为**:训练池 universe 自动从 `data/universe.parquet` + `load_universe_cache` 装 4000+ 票(无该文件回退 cfg.stocks 并 log warning);`industry_map` 经 `load_or_build_industry_map(source="auto")` 装载;name_map 来自 universe.parquet `name` 列;cfg.stocks 自动 merge 到 pool(保证应用池可投资)。

**PR-3 行为**:`staggered_starts > 1` 自动跑 N 个 offset(`{0..N-1}`)各一份回测,聚合成 envelope(min/p25/median/p75/max)+ ensemble mean → `reports/portfolio/<date>.html` 出包络图 + per-offset 折叠卡。

**`universe_codes`(2026-05-31)**:解耦训练池与 portfolio universe。`None`(默认)= portfolio universe 同训练池(全 pool_data,旧行为);设为显式 code 列表则 `precompute_scores_from_legacy` + `PortfolioEngine` 仅在该子集运行(`factor_panel` build + ml_factor training_universe=all 仍用全 pool_data)。典型用法:训练池 4358 票 + portfolio universe 几十到几百票,避免 `precompute_scores_from_legacy` 在 4358 票全跑时 segfault。

## recommend_pool 段

Pool B(全市场量化推荐池)。详见 [conventions.md](conventions.md) Pool B 段。

```yaml
recommend_pool:
  enabled: true                  # 默认开启
  top_n: 30
  min_avg_amount_20d: 5e7        # 5000 万元;mootdx vol*close*100
  max_per_industry: 5            # "未知" 桶在全未映射时跳过 cap,否则正常计
  refresh: weekly                # weekly 默认 / always / never
  cache_dir: data/recommend_pool
  industry_map_max_age_days: 30
  industry_source: auto          # auto = baostock→akshare / baostock / akshare
```

**前置条件**:必须先跑 `fetch-universe`;首次运行自动拉行业映射。**缓存键**含 `cfg.content_hash`,改 yaml 任一字段都失效。

## A/B 测试配置(独立文件)

### per-stock `ab.yaml`

```yaml
base_config: config.yaml          # 共享 stocks/data/indicators 来源
stocks_filter: [...]              # 可选,只能减
arms:                             # 恰好 2 个
  <name>:
    strategy: {...}               # 整段替换
    backtest: {equity_curve_holding_days: [N], sizing: {...}}  # 字段级合并
```

每个 arm 只能覆盖 `strategy` 段(整段替换)和 `backtest` 段(字段级合并,None 字段继承 base),其他顶层字段继承 base。`equity_curve_holding_days` 强制单元素列表。ML 缓存通过 `effective_cfg.content_hash` 自动隔离。`backtest.sizing` 是 F3 PR-C 起支持的覆盖字段。完整 schema 见 `docs/superpowers/specs/2026-05-24-ab-testing-design.md`。

### portfolio `portfolio_ab.yaml`

arm 仅允许覆盖 `strategy`(整段替换)和 `portfolio_backtest`(字段级合并)。其他顶层字段继承 base。两 arm 各算各的 score panel(per-arm content_hash 隔离)。模板见 `portfolio_ab.yaml.example`。
