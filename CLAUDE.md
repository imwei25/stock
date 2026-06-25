# stockpool — Claude Code 项目指南

> 本文为后续 Claude Code 会话的快速上手指南。**不需要通读整个项目**,按需查阅对应模块即可。
> 改动项目时请遵守文末的["更新本文件"](#改动后更新本文件)规则。

## 项目一句话

A 股技术信号分析工具:可插拔后端拉数据(mootdx/baostock/akshare) → 计算多指标 → 综合评级 → 输出 HTML 日报和走样式回测报告。

## 快速命令

```bash
# 日报 (拉数据 → 分析 → 渲染 HTML 到 reports/)
python -m stockpool run --config config.yaml

# 回测 (走样式综合策略 + 多 N 净值曲线)
python -m stockpool backtest --config config.yaml [--stocks 605589] [--refresh-factor-panel]

# 拉全市场 A 股缓存 (训练用,剔除 ST/科创/北交; 8 线程, mootdx 全量 ~1 分钟)
# 默认按 cfg.data.source 拉每只票;--source 临时覆盖
# (清单本身只有 mootdx 实现,所以"列清单"永远走 mootdx)
python -m stockpool fetch-universe [--workers 8] [--limit 100] [--refresh] [--source baostock]

# 测试
python -m pytest tests/ -q

# 因子管理 (浏览 / 筛选 / 选择)
python -m stockpool factors list                          # 列全部 (~165 个 base, 含变体 ~280-320)
python -m stockpool factors list --source wq101           # 按来源筛
python -m stockpool factors list --type cross_sectional   # 按类型筛
python -m stockpool factors show alpha_001                # 看单个因子的元数据
python -m stockpool factors pick                          # 打开 HTML 选择器

# 因子分析 (跑一次输出 HTML + JSON 报告)
python -m stockpool factors analyze --universe all --output reports/factor_analysis

# 从分析 JSON 自动选 top-N 去相关因子,写成 ml_factor.factors_file 兼容的 selection.json
python -m stockpool factors pick-by-ic \
  --input reports/factor_analysis/<日期>.json \
  --output reports/selection.json \
  --top-n 20 --max-corr 0.6 --min-ir 0.05

# A/B testing — 比较两个策略在同一池/同段历史下的优劣
python -m stockpool ab --config ab.yaml
# 调试某一边 (只跑单边 + 打印指标到 stdout, 不生成 HTML)
python -m stockpool ab --config ab.yaml --arm <arm_name>
# 强制每边独立 load universe / factor panel (escape hatch)
python -m stockpool ab --config ab.yaml --no-share-pool

# Portfolio backtest — 横截面 top-K 等权 + 周期性 rebalance + ensemble
# 需先 portfolio_backtest.enabled: true
python -m stockpool portfolio-backtest --config config.yaml [--refresh-scores]

# Portfolio AB — 对比两套 portfolio 策略 (与 per-stock `ab` 平行)
python -m stockpool portfolio-ab --config portfolio_ab.yaml
# 单 arm 调试(stdout 只打印指标,不出 HTML)
python -m stockpool portfolio-ab --config portfolio_ab.yaml --arm <arm_name>
# 两 arm 并行(subprocess; opt-in,peak 内存 ~2× single-arm)
python -m stockpool portfolio-ab --config portfolio_ab.yaml --parallel-arms

# AB candidate pool — stratified ~100-stock pool for AB tests (static, manual rebuild)
python -m stockpool ab-pool build [--refresh]
python -m stockpool ab-pool show    # 渲染 reports/ab_pool.html + 浏览器打开
```

## 模块地图

| 文件 | 职责 |
|---|---|
| `src/stockpool/cli.py` | 入口,定义 `run` / `backtest` / `fetch-universe` / `factors *` / `ab` 子命令 |
| `src/stockpool/backtest_runner.py` | 共享给 `cli.cmd_backtest` 和 `ab.runner` 的两个 helper:`prepare_pool` (ml_factor 池 + factor_panel 预算) 和 `backtest_stocks` (per-stock 回测循环,失败隔离,返回 `(成功, 失败)` 元组) |
| `src/stockpool/ab/` | A/B 测试子包: `config.py` (`ABConfig` / `ArmOverride` / `load_ab_config` / `build_effective_cfg`) + `runner.py` (`run_ab` / `run_single_arm` / `_decide_pool_sharing` + `ArmResult`/`ABResult`) + `report.py` (HTML 对比报告:banner / 聚合表 / Sharpe 散点 / Sharpe 差直方图 / per-stock 卡片) |
| `src/stockpool/portfolio_ab/` | **Portfolio AB 子包** (PR-4): `config.py` (`PortfolioABConfig` / `PortfolioArmOverride` 仅允许覆盖 `strategy` 整段替换 + `portfolio_backtest` 字段级合并 / `load_portfolio_ab_config` / `build_effective_cfg` — content_hash 重算保证 score panel 缓存隔离) + `runner.py` (`ArmResult` / `ABResult` / `run_single_arm` / `run_portfolio_ab` — 共享 universe + sector_map + name_map,per-arm failure 隔离) + `report.py` (`render_portfolio_ab_report` — banner / 聚合指标 + Δ 表 / 双 arm 净值 overlay / per-stock 贡献分解 top 15 / 已交易 code 集合分析 only-A/only-B/both / 失败 arm 红 banner) |
| `src/stockpool/portfolio/` | **Portfolio 框架** (PR-1/PR-2/PR-3): `strategy.py` (`PortfolioStrategy` ABC + `PrecomputedScoreStrategy`) + `scoring.py` (`precompute_scores_from_legacy` — 把 per-stock `Strategy.generate_signals` 拼成 T×N score 面板) + `engine.py` (`PortfolioEngine` — top-K 等权 + 周期 rebalance + T+1 + 行业 cap 贪心 + eligibility 过滤) + `eligibility.py` (`EligibilityFilter` — 流动性/ST/min_history_bars) + `ensemble.py` (`StaggeredRunner` + `EnsembleResult` — N 个 `start_offset` 串行跑,聚合 min/p25/median/p75/max envelope + 等权均值 ensemble curve) + `result.py` (`PortfolioBacktestResult`/`PortfolioTrade`) + `report.py` (单 arm HTML + ensemble HTML;`render_portfolio_report` / `render_ensemble_report`)。PR-2 起 universe = `load_universe_cache`(`universe.parquet` 不存在时回退 cfg.stocks),`sector_map` 来自 `load_or_build_industry_map`,行业 cap "未知" 全空时跳过 cap、否则 Unknown 桶正常计。PR-3 起 CLI `staggered_starts > 1` 自动走 `StaggeredRunner`(串行,engine_factory 闭包保证每 offset fresh state);N=1 退化回单 arm 路径。无 portfolio AB (PR-4) |
| `src/stockpool/config.py` | Pydantic schema + YAML 加载;**配置变更必须更新这里** |
| `src/stockpool/fetcher.py` | 公开 API + Parquet 缓存 + OHLCV 校验;按 `cfg.data.source` 派发后端;`fetch_universe`/`list_universe`/`load_universe_cache` 提供全市场批拉与读盘;`check_source_change`/`update_source_marker` 维护 `data/.data_source` 标记,源变化时自动 force_refresh |
| `src/stockpool/data_sources/mootdx_backend.py` | 通达信 TCP 后端(默认)。股票/指数/**行业板块(88xxxx)**;含当日盘中,TDX 占位 bar 会被丢弃 |
| `src/stockpool/data_sources/baostock_backend.py` | baostock 后端(无 token,收盘后约 18:00 更新);不支持板块,板块会自动走 mootdx |
| `src/stockpool/indicators.py` | MA / MACD / KDJ / RSI / BOLL / Volume / Breakout |
| `src/stockpool/signals.py` | `detect_signals` + `score_triggers` + `combine_daily_weekly` + `verdict_of` |
| `src/stockpool/factors/` | **连续因子库**(Factor ABC + 注册表 + 内置技术因子 + **WQ101**) |
| `src/stockpool/factors/ops.py` | **WQ 算子库** (ts_rank / rank / decay_linear / indneutralize / ...). Rust 加速状态:rank / ts_std / ts_argmax / ts_argmin / ts_rank (PR-T1.1) + decay_linear / indneutralize (PR-T1.2) + **ts_min / ts_max (PR-B2)** — 共 9 个 op dispatch 到 Rust,装好 maturin + `maturin develop --release --manifest-path rust/stockpool_ops/Cargo.toml` 后自动启用;`STOCKPOOL_USE_PYTHON_OPS=1` 强制回 pandas oracle (`_ops_py.py`)。**延迟 dispatch**(保留 pandas oracle 路径):correlation(Rust Welford 与 pandas rolling.corr 在近±1 相关窗口 FP 路径不同,级联 rank 差异超出 snapshot 容差) + ts_sum / ts_mean(Rust per-window 求和与 pandas Cython 内部路径相差 ~1 ULP ≈ 2e-13,在下游 rank() 中可翻转几乎相等的两个股票顺序,导致 5 个 alpha 超出 snapshot 容差);这两类均已有 Rust 实现并通过 Layer A 等价测试(rtol=1e-7),等待 snapshot 重新生成后可切换。 |
| `src/stockpool/factors/wq101.py` | **WorldQuant 101 Formulaic Alphas** 全套实现 (Alpha001..Alpha101) |
| `src/stockpool/factors/original_stats.py` | rolling 直接统计量族(close_std/skew/kurt、volume_skew/kurt、range_std、volume_std),~25 变体 |
| `src/stockpool/factors/ewma.py` | EWMA 平滑动量/波动/换手 族(halflife 参数化),~15 变体 |
| `src/stockpool/factors/vwap_deviation.py` | VWAP 偏离族 (close 相对 vwap proxy),~20 变体 |
| `src/stockpool/factors/close_position.py` | 收盘位置动量族 ((C-L)/(H-L)),~15 变体 |
| `src/stockpool/factors/turnover_extra.py` | 短窗换手 z-score / 量比,补 custom.turnover_zscore_60 长窗,~12 变体 |
| `src/stockpool/factors/acceleration.py` | 动量/换手二阶差分,捕获趋势变速,~9 变体 |
| `src/stockpool/factors/single_stock_vol.py` | ATR / CCI / 振幅 / Parkinson / Garman-Klass 波动率,~20 变体 |
| `src/stockpool/factors/composite.py` | rank/decay/scale 复合算子拼装,~12 变体 |
| `src/stockpool/factors/rank_correlation.py` | 价格秩 × 成交量秩滚动相关 系列,~20 变体 |
| `src/stockpool/factors/cross_sec_breadth.py` | 全市场宽度因子(>MA20 占比/涨停占比/横截面 std),~7 变体。⚠️ 涨停股算入分子,与 mask config 无关 |
| `src/stockpool/factors/fundamentals.py` | 基本面因子(PE/PB/ROE/ROA/毛利率/净利率/营收 YOY)+ **size 因子**(market_cap / log_market_cap),baostock 5 张季度表,严格 PIT。PE/PB 打 `contains_mcap` tag 防 mcap 神经化共线。 |
| `src/stockpool/fundamentals_loader.py` | baostock 5 张季度财务表 (profit/growth/balance/cash_flow/dupont) PIT 长期缓存,parquet 30 天 staleness |
| `src/stockpool/factors_picker.py` | **HTML 因子选择器** + `factors` CLI 子命令 |
| `src/stockpool/industry_map.py` | `code → 行业` 映射;多源(`auto` / `baostock` / `akshare`);缓存到 `data/stock_industry_map.parquet`,>30 天过期自动重拉。**mootdx 路径无效**:TDX 服务器对 `block_hy.dat` 返回 0 字节 |
| `src/stockpool/ipo_dates.py` | `code → IPO 日期` 映射(baostock `query_stock_basic`,5500+ 股一次性,~3-5 秒);缓存 `data/ipo_dates.parquet` 30 天有效。当 `cfg.strategy.ml_factor.mask.enabled=true` 且 `cache_dir` 可用时,`MLFactorStrategy._get_ipo_dates` 自动加载并传给 `_listing_mask`,避免后者 `first_valid_index` 启发式把缓存窗口短的成熟股错认成新上市(panel union 早于该股缓存起点时会触发 bug,可使 mask=False 比例虚高到 50%)|
| `src/stockpool/recommend_pool.py` | **Pool B**(全市场量化推荐池):漏斗 + 排序 + 周缓存;`compute_or_load_pool_b` 顶层 API。接收 caller 预算好的 `pool_data` / `factor_panel` / `close_panel`,沿 `_compute_pool_b` → `_score_universe` → `build_strategy` 一路透传,避免每股重跑 `build_close_panel(4000+ 股)` (~3s/股 × 4000 ≈ 3 小时 → 32 秒)。日志会按 200 股粒度打印 `[TIME] Pool B i/total ... build_avg=Xms predict_avg=Yms ETA=Zs` 便于观察长循环进度 |
| `src/stockpool/ab_pool.py` | **AB 候选池**:行业分层 top-2 mcap + top-2 liq(行内 union 合并 source_tag)+ akshare 流通市值快照 + ipo/流动性硬过滤;`build_ab_pool` 生成 `data/ab_pool.parquet`,`load_ab_pool` 读盘。静态、手动重建(`--refresh`)|
| `src/stockpool/ab_pool_report.py` | AB 池 HTML 渲染器(client-side JS 筛选 + 排序,无 HTTP server,无 jinja)|
| `src/stockpool/factors_analysis.py` | **因子分析**: 滚动 IC / IR / half-life / 相关性 / regime 切片;`analyze_factors` + `pick_top_factors` |
| `src/stockpool/factors_analysis_report.py` | pyecharts HTML 报告: 排名表 + IC 时序 + 相关性 heatmap + regime 拆分 |
| `src/stockpool/panel.py` | **Panel** 数据结构 (T×N 宽表 dict) + `build_panel_from_cache`。**+** `compute_tradability_mask` / `apply_mask` / `_limit_threshold` / `_listing_mask` 支持按板块(主板 ±10% / 创业板+科创 ±20%)的可交易性 mask。`_listing_mask` 接 `ipo_dates: Mapping[str, Timestamp] | None`,有真实 IPO 日期时按 IPO + 366 自然日(≈252 交易日)cutoff 屏蔽新股;无 ipo_dates 时退化到 first_valid_index 启发式(打 warning)。**注意(2026-05-31 重构)**:mask **不** 应用到因子输入面板 — 时间序列因子需要看真实 close(涨停日 +9.9% 本身是有用信号)。Mask 只在 `forward_return_panel` 的双向标签检查(`mask[t] ∧ mask[t+horizon]`)和训练样本 dropna 上生效。`apply_mask` 仍是公开工具,供有特殊需求的因子 `compute` 方法按需调用。详见 `docs/handoff/2026-05-31-mask-ab-investigation.md` 的演变记录 |
| `src/stockpool/ml/` | **两步法 ML 组合**(dataset / Lasso 或 LightGBM selector / IC&IR&Equal&LightGBM weighter / TwoStepPipeline) |
| `src/stockpool/ml/preprocess.py` | **截面预处理流水线** (winsorize / cs_zscore / industry_neutralize + mcap_neutralize) — Phase 1 + Phase 2。stateless 函数 + `apply_preprocess_pipeline` 驱动 + `_is_all_off` short-circuit。baked 进 `build_factor_panel` cache,所有 callers 共享。**`ml/dataset.py:build_panel`** 也接 `preprocess_cfg` + `cache_dir`(2026-06-20 Bug C 修复),legacy fallback 与 fast path 行为对齐。**PR-T1.1 性能**:`industry_neutralize_panel(log_mcap=...)` per-day OLS loop 改造为批量 normal equation(与 mcap_neutralize_panel 同范式),T=1000 时 ~50× 加速。`_industry_neutralize_per_day_loop` 保留为 test oracle。 |
| `src/stockpool/ml/mcap.py` | **log(market_cap) 面板构造**(Phase 2):`build_log_mcap_panel(panel, cache_dir)` 用 baostock `balance.totalShare × close` PIT-aligned。不入 cache,每次现 build(< 100ms on 4000 stocks × 250 days)。 |
| `src/stockpool/strategy_factory.py` | 按 `cfg.strategy.name` 工厂构造策略 + ML 通用 simulate;ml_factor 注入 `cache_dir` 以启用日报路径的月度训练缓存;`build_factor_panel` + `build_close_panel` 顶层助手用于 CLI 预算(close_panel 用于 `_try_fit` 跳过每 refit 重算因子);`load_or_build_factor_panel` 落盘缓存,key = (sorted factors + sorted codes + last_date) sha256[:12],写 `data/factor_panels/<sig>/{manifest.json, close.parquet, <factor>.parquet × N}`;input 任一变化生成新 sig 重算。**PR-3 性能**:`MLFactorStrategy._ensure_pooled_xy_long` 把全历史 stack 提到 `shared_cache["__pooled_xy_long__", sig]`,后续 refit 只做 `X.loc[:label_cutoff_ts]`(按 `label_end-1` 日期切,避免 horizon 行未来 label 泄露)+ per-stock tail,杜绝每次 refit 重 stack;`_try_fit` 用 `searchsorted` 早退跳过那些训练集明显太小的 cutoff,避免无意义的 slice+groupby。`stack_panel_to_xy` 内部改 numpy ravel + column_stack 替代逐因子 `.stack()`,~35× 加速 |
| `src/stockpool/report.py` | 日报 HTML(含市场背景、板块上下文);`_optimize_html` 做 echarts lib 去重 + `<details>` 默认折叠 + 图表懒加载,降低首屏开销 |
| `src/stockpool/backtest.py` | 单信号前瞻命中率 |
| `src/stockpool/backtesting/` | **回测框架**(策略 ABC + 引擎),见下 |
| `src/stockpool/backtesting/sizing.py` | **LotSizer Protocol** + `FixedLotSizer` / `VolTargetLotSizer` + `build_lot_sizer(SizingConfig)` 工厂 (F3 PR-C);独立模块,无 config 运行时依赖 (TYPE_CHECKING 引用) |
| `src/stockpool/backtest_composite.py` | 综合策略回测的旧 API 适配层,委托给框架 |
| `src/stockpool/backtest_report.py` | 回测 HTML 渲染 |

## 回测框架 (`stockpool.backtesting`)

**核心抽象**:

- `Strategy` ABC — 子类必须实现 `name` / `generate_signals` / `should_enter` / `should_exit`;
  `generate_signals` 输出至少含 `date / open / close / signal`(`open` 用于次日开盘成交;省略时引擎按 `close.shift(1)` 兜底);
  可选 `should_reset_timer`(返回 True 重置 N 天计时器)、`predict_latest(daily_df) -> dict`(给日报路径用,返回最后一根 bar 的 `{signal, final_score, ...}`;默认实现是跑全程 `generate_signals` 取末行,子类可重写做单点优化或加缓存)。
- `BacktestEngine` — 单仓位、long-only、T+1、单笔进出。
- `MultiLotBacktestEngine` — 每个 buy 开一个独立的 lot(仓位由 `LotSizer` 决定,见下方 Sizing 段),各自计 N、各自记账。
- `BacktestResult` — `signals` / `curve` / `trades` / `metrics` / `max_holding_days` / `strategy_name`。
- `TradeCosts(buy_cost, sell_cost)` — 比例(`0.001` = 0.1%)。
- `compute_metrics` 纯函数 — total/ann return、max DD、Sharpe、win rate、avg trade ret。
- `buy_and_hold_baseline` — 不扣手续费的全仓基准。

**内置策略** (`stockpool.backtesting.strategies`):

- `CompositeVerdictStrategy` — 项目主策略,综合评级 + 日周共振。
- `MLFactorStrategy` — 两步法(Lasso 筛因子 → IC/IR/equal 加权 → 分位数映射 verdict),walk-forward 重训;支持 `per_stock` 和 `pooled` 两种数据组织。
- `VerdictExecution` — 仅执行规则(配 `engine.run_on_signals` 用)。
- `SMACrossStrategy` — 扩展示范。

**verdict-based 策略默认参数**:

| 参数 | 默认 | 行为 |
|---|---|---|
| `buy_verdicts` | `("buy","strong_buy")` | 触发 `should_enter` |
| `sell_verdicts` | `("sell","strong_sell")` | 触发 `should_exit` |
| `refresh_verdicts` | `("strong_buy",)` | 持仓时触发 `should_reset_timer`(刷新 N 天计时);传 `()` 关闭 |

**CLI 默认引擎是 `multi_lot`**(见 `config.yaml:backtest.engine`),`sizing.type` 默认 `vol_target`(F3 PR-C 起;切回老的固定仓位行为:`sizing.type: fixed`,`sizing.fixed.size: 0.1`)。
要切回老的单仓位行为:`engine: single`。

完整 API 参见 `docs/backtesting_framework.md`。

## 引擎约定(重要)

- **T+1 + 次日开盘成交**:信号在第 `t-1` 根 bar 收盘后产生,**在第 `t` 根 bar 的 `open[t]` 成交**(A 股集合竞价价)。除非次日开盘直接涨停打不进单,否则视为可成交。
- **进场当日敞口** = `open[t] → close[t]`,扣 `buy_cost` 之后乘 `close[t]/open[t]`;后续持仓日仍按 `close[t-1] → close[t]` 累计。
- **出场当日敞口** = `close[t-1] → open[t]`,然后扣 `sell_cost`,当日剩余时间空仓。
- **`Trade.entry_idx` / `exit_idx` 指向执行 bar `t`**(而非决策 bar `t-1`);`entry_price` / `exit_price` 即 `open[t]`。
- **signal 帧必须带 `open` 列**;若缺(老 cache、手搓 fixture),引擎回退到 `open[t]=close[t-1]`,在此假设下新口径退化为旧的 close-to-close 行为,所有旧测试算术保留。
- **Look-ahead 安全契约**:`generate_signals` 第 `i` 行只能用 `daily_df.iloc[:i+1]`。
- **单仓位不加仓**;多仓位 lot 同信号入场。
- **`should_reset_timer` 胜出**:同时为真时优先于 `time_exit` 与 `should_exit`。
- **B&H 基准不扣手续费**(让策略对比更保守),锚定 `open[0]`:`equity[t] = close[t]/open[0]`。

## Sizing(F3 PR-C 起)

`MultiLotBacktestEngine` 不再硬编码 `position_size`,改由 `LotSizer` 注入:

- `FixedLotSizer(size)` — 老行为,每单恒定 `size` 比例
- `VolTargetLotSizer(baseline, ref_vol, window, min, max, fallback)` — 按个股最近 `window` bars 的滚动 std 反比调仓:`size = baseline × (ref_vol / recent_vol)`,clip 到 `[min, max]`
  - 冷启动(< window+1 bar)/ NaN / vol=0 → 走 `fallback`:`"fixed"` 退回 baseline,`"skip"` 返 0(本次不开仓)
  - 公式锚点 `baseline = cfg.backtest.sizing.fixed.size`:fixed 和 vol_target 之间切换时,锚点不变,差异纯来自 vol-adjust
- 工厂 `build_lot_sizer(cfg.backtest.sizing)` 是顶层 wiring(cli / backtest_runner / backtest_composite / strategy_factory / ab/config 全部走它)
- `Trade.lot_size` 记录每笔成交的实际仓位,A/B 报告可用其做归因

## 配置 (`config.yaml`)

所有字段由 `config.py:AppConfig` 校验。结构概览:

- `stocks` — 股票池,每条含 `code` / `name` / `sector`
- `data` — `history_days` / **`warmup_days`** / `cache_dir` / `force_refresh` / **`source`** (`mootdx` 默认 / `baostock` / `akshare`)。注意:(1) **`warmup_days`** (2026-06-06 Phase 2,默认 0,`config.yaml` 显式设 200):额外向前多拉的 bars 仅作为长 rolling 因子(200 日 corr / momentum_120 等)的 warmup。**进入** factor 计算 + 训练样本(都 look-ahead 安全);**不进入** backtest bar 迭代和绩效统计(`backtest_runner.backtest_stocks` 和 `cli._analyze_one` 在 `simulate_*` 之前自动 trim 掉)。`fetch_daily`/`fetch_index_daily`/`fetch_sector_daily`/`fetch_universe`/`load_universe_cache` 都接收 `warmup_days=0` 默认参数,实际值由 caller 从 `cfg.data.warmup_days` 透传。(2) 切换 source **自动** force_refresh:`cache_dir/.data_source` 记录上次 source,`fetch_daily` / `fetch_index_daily` / `fetch_sector_daily` / `fetch_universe` 入口比对,不一致直接丢弃旧 parquet 重拉(volume 单位 mootdx=手 / baostock=股,混源会污染相对成交量指标);(3) 行业板块仅在 `source=akshare` 时走东财,其他两种 source 都用 **mootdx 的通达信行业指数 (88xxxx)**;名字→代码映射见 `mootdx_backend._TDX_INDUSTRY_CODES`,也可在 `stocks[].sector` 直接填 6 位 TDX 代码
- `indicators` — MA 周期、MACD/KDJ/BOLL 参数
- `weights` — 各信号触发的得分
- `scoring` — `daily_weight` / `weekly_weight` / `resonance_bonus`(共振奖励)
- `verdicts` — `strong_buy` / `buy` / `sell` / `strong_sell` 分数阈值
- `backtest` — `forward_days` / `equity_curve_holding_days` / `risk_free_rate` / `costs` / **`engine`** / **`sizing`**(`type: fixed | vol_target`, 默认 `vol_target`;`fixed.size` 是 vol_target 公式的 baseline 锚点) / **~~`position_size`~~**(deprecated alias of `sizing.fixed.size`,自动迁移 + DeprecationWarning) / **`max_concurrent_lots`**
- `context` — `indices`(大盘指数列表,默认上证/深证成指)
- `report` — `output_dir` / `keep_history` / `klines_to_show`
- **`strategy`** — `name` (`composite_verdict` 默认 / `ml_factor`) + `ml_factor` 子配置(`factors` 或 **`factors_file`** / `horizon` / `train_window` / `refit_every` / `panel_mode` / **`training_universe`** / **`share_pool_fit`** / **`embargo_days`** / **`label_type`** / `selector.{lasso|lightgbm}` / `weighter` / `thresholds` / `*_verdicts`)。`factors_file` 指向 HTML picker 导出的 JSON,与 `factors` 列表二选一。**`training_universe`**: `pool`(默认,只用 cfg.stocks)/ `all`(全市场 cache,需先 `fetch-universe`;仅在 `panel_mode=pooled` 时生效)。**`share_pool_fit`**(默认 `true`,仅 `panel_mode=pooled` 生效):跨股共享 fit,缓存键 `(sig, year, month)`,同月内所有股、所有 refit_bar 复用同一 pipeline;训练集不再剔除 host,host 自己以 ~1/N 权重进入自己的训练。`false` 时 host 完全从训练集踢出 — fast path (`_build_pooled_xy_from_panel`) 与 legacy fallback (`_build_truncated_pool`) 都 drop host,两路径在非共享下输出一致(2026-06-20 修复 Bug B)。**`embargo_days`**(默认 `null` = auto = `horizon`,F2 PR-A 新增):walk-forward 训练集与测试集之间的额外间隔,消除 horizon 日前向收益的标签泄露;设 `0` 回到 pre-PR-A 行为。**`label_type`**(默认 `"return"`,F2 PR-A 接口位):训练标签变换 — `"return"` 已实装,`"vol_adjusted"` / `"cross_sec_rank"` 是占位 raise `NotImplementedError`,后续 PR 实装。**`label_basis`**(默认 **`"open"`**,审查 P2-3):训练标签价格基准 — `"open"` = `open[t+1+h]/open[t+1]−1`,与 T+1 次日开盘成交对齐(不含决策 bar 拿不到的 `close[t]→open[t+1]` 隔夜段);`"close"` = legacy `close[t+h]/close[t]−1`(偏乐观)。open 基准的标签多看 1 根 bar,embargo/截断数学(`_label_lag`)自动 +1;可交易性 mask 双向检查也移到实际进出场 bar(`mask[t+1] ∧ mask[t+1+h]`)。改 `label_basis` 后旧 ml_models pkl / sig 自动失效。`factors analyze`(CLI 传 `cfg.strategy.ml_factor.label_basis`)与 A/B score IC(各 arm 按其 `label_basis`,`ab/score_ic.py` + `_arm_label_basis`)同口径。**`selector.{lasso|lightgbm}`**(F2 PR-A 子段化 + PR-B1 加 LGB):`type` 默认 **`"lasso"`**(2026-05-24 从 `"lightgbm"` 回退,见 `docs/ab_validation_results.md`:LGB+LGB 在 16 股 × 500bar baseline 上 sharpe 退 0.2 / return 退 20%),`lasso.{alpha,max_iter,tol}` 或 `lightgbm.{num_leaves,min_data_in_leaf,learning_rate,num_iterations,max_depth,random_state,top_k_factors,min_importance_ratio}` 子段二选一,顶层扁平字段被 Pydantic 拒绝。改 `selector` 任一字段后旧 ml_models pkl 自动失效。切到 `all` 或翻 `share_pool_fit`、改 `embargo_days` / `label_type` / `selector` 任一项后旧的 ml_models pkl 会因 sig 变化自动失效。**`weighter.{ic|ir|equal|lightgbm}`**(F2 PR-B2 子段化):`type` 默认 **`"ic"`**(同 2026-05-24 回退,见上),`ic.{use_rank,min_abs_ic}` / `ir.{n_chunks,use_rank,min_abs_ir}` / `equal` (无参) / `lightgbm.{num_leaves,min_data_in_leaf,learning_rate,num_iterations,max_depth,random_state}` 子段四选一,顶层扁平字段被 Pydantic 拒绝。LGB 仍可 opt-in,但需先调超参或扩股池验证。
- **`preprocess.{winsorize, zscore, industry_neutralize, mcap_neutralize, min_pool_size}`**(2026-06-06, Phase 1+1.5+2):截面预处理流水线。Phase 1 三步 + Phase 2 `mcap_neutralize`(per-day OLS Y ~ log(market_cap),或与 industry 联合 OLS)。`mcap_neutralize` default `false`,需 baostock `balance.totalShare` parquet 缓存(`fundamentals_loader` 30 天),验证见 `docs/ab_validation_results.md` P5-mcap 段(待 AB 完成填入)。`min_pool_size: int = 200` runtime guard 同时守 mcap 步。PE/PB 因子打 `contains_mcap` tag,自动跳过 mcap 神经化(避免与 close × shares 强共线)。`mcap_neutralize=True` 时 `build_factor_panel` 接 `cache_dir` 参数后调 `ml.mcap.build_log_mcap_panel` 现 build,不入 panel cache。baked 进 `factor_panels/<sig>/` cache → 改 preprocess 自动新 sig 重算。
- **`mask`** 子段(default 关闭,opt-in):tradability mask 配置。`enabled`(bool,default `false` 完全向后兼容)/ `limit_up_threshold_main`(主板沪深,default 0.098)/ `limit_up_threshold_chinext`(创业板+科创,default 0.198)/ `limit_up_threshold_bse`(北交所兜底,default 0.298)/ `min_listing_days`(default 252)。**应用范围(2026-05-31 重构)**:mask **仅** 作用于训练标签层(`forward_return_panel` 双向检查 `mask[t] ∧ mask[t+horizon]`,涨跌停/停牌/新股头 N 天产生 NaN 标签),通过 `stack_panel_to_xy` 的 dropna 自然剔除这些样本。**不破坏因子输入** — 时间序列因子(`ts_corr`/`ts_rank`/argmin 等)看到的是原始 close 包括涨停日的 +9.9%(那本身是有用信号)。`factor_panels/<sig>/` 缓存与 mask 状态**解耦**(同因子列表+股池+last_date → 同 sig),`ml_models/<sig>_*.pkl` 仍含 mask 在 cfg hash 里。源自论文 B (arXiv 2507.07107) 的 mask-first finding,但放弃了他们 "value+mask 双通道传递" 的全套重构(对 ~111 因子 ROI 不合算,见 `docs/handoff/2026-05-31-mask-ab-investigation.md` 的设计讨论)。附于 `strategy.ml_factor` 段下。
- **`portfolio_backtest`** — Portfolio 级回测(PR-1 新增,默认 `enabled: false` 关闭)。`portfolio.{top_k=20, rebalance_n_days=5, max_per_industry=5, initial_cash=1.0}` / `eligibility.{min_avg_amount_20d=5e7, exclude_st=true, min_history_bars=60}`(PR-2 起 engine 实际读取) / `staggered_starts=1`(PR-3 起 `>1` 自动走 ensemble)/ `parallel_staggered: bool = False`(PR-T1.3,opt-in):N 个 staggered offset 并行跑(ProcessPoolExecutor)。要求 engine components 全部可 pickle;失败 fallback 串行 + warning。**实测加速**:single engine wall < 1s 的小 panel 反而变慢(Windows spawn + pickle 开销);1000 codes × 2500 bars 量级 ~1.9×;全市场 4000+ 票场景预期 2-3×。详见 `docs/superpowers/plans/2026-06-21-perf-tier1-decisions.md` #5。/ `score_cache_dir=data/portfolio_scores`(2026-06-25 起缓存键 = **scoring 签名** `score_cache_key`:仅打分相关配置 strategy+data + 打分 universe 的哈希,**不含** `portfolio_backtest`(top_k/rebalance/cap)→ 只调组合参数的 arm/重跑直接命中缓存、不重算昂贵的 walk-forward 打分;带 `v2` 版本前缀,旧缓存自动失效)/ **`universe_codes: list[str] | None = None`**(2026-05-31 新增,**解耦训练池与 portfolio universe**)。CLI `portfolio-backtest` 在 `enabled=false` 时退出 2。PR-2 起:**训练池** universe 自动从 `data/universe.parquet` + `load_universe_cache` 装 4000+ 票(无该文件回退 cfg.stocks 并 log warning);`industry_map` 通过 `load_or_build_industry_map(source="auto")` 装载(首次跑 baostock 慢);name_map 来自 universe.parquet 的 `name` 列;cfg.stocks 自动 merge 到 pool(保证应用池始终可投资)。PR-3 起 `staggered_starts > 1` 自动跑 N 个 offset(`{0, 1, ..., N-1}`)各一份回测,聚合成 envelope + ensemble mean → `reports/portfolio/<date>.html` 出包络图 + per-offset 折叠卡。**`universe_codes`**:`None`(默认)= portfolio universe 同训练池(即全 pool_data,旧行为);设为显式 code 列表则 `precompute_scores_from_legacy` + `PortfolioEngine` 仅在该子集上运行(`factor_panel` build + ml_factor training_universe=all 仍用 **全** pool_data)。典型用法:训练池 4358 票 + portfolio universe 几十到几百票,避免 `precompute_scores_from_legacy` 在 4358 票全跑时 segfault(C 层 bug 不易排查)。两 arm 解耦后跑 ab_mask 实测 mask 提升 Δ Sharpe +0.04(详见 `docs/research/2026-05-31-a-share-quant-survey-comparison.md` 附录)
- **`recommend_pool`** — Pool B(全市场量化推荐池)。`enabled`(默认 `true`)/ `top_n`(30)/ `min_avg_amount_20d`(5e7 元;mootdx `vol*close*100`)/ `max_per_industry`(5;"未知" 桶在**所有股都未映射时**自动跳过 cap,否则正常计)/ `refresh`(`weekly`默认/`always`/`never`)/ `cache_dir`(`data/recommend_pool`)/ `industry_map_max_age_days`(30)/ **`industry_source`**(`auto` 默认 = baostock→akshare 链 / `baostock` / `akshare`)。**前置条件**:必须先跑 `python -m stockpool fetch-universe`;首次运行自动从所选 industry_source 拉映射(baostock ~5-10s,akshare ~1-2min)。**缓存键**含 `cfg.content_hash`,改 yaml 任一字段都失效
- **`ab_pool`** — AB 候选池构建参数。`cache_path`(`data/ab_pool.parquet`)/ `industry_source`(`auto`/`baostock`/`akshare`)/ `min_listing_days`(252)/ `min_avg_amount_20d`(5e7)/ `per_industry_top_mcap`(2)/ `per_industry_top_liq`(2)/ `exclude_st`(true)/ `include_unknown_industry`(true)。整段可选,默认值复现 spec 28 行业 × 4 配方(~100 票)。前置:必须先 `fetch-universe`。生成命令:`python -m stockpool ab-pool build`
- **AB 池开关**(独立配置文件 `ab.yaml` / `portfolio_ab.yaml`):新增顶层 `use_ab_pool: bool`(默认 false)。`ab.yaml` 设 true 时,per-stock AB 用 ab_pool 替换 `cfg.stocks` 迭代(`stocks_filter` 仍生效,作为子集过滤)。`portfolio_ab.yaml` 设 true 时,把 ab_pool codes 注入到每个 arm 的 `portfolio_backtest.universe_codes`(per-arm 显式 `universe_codes` 仍优先)。训练池(`training_universe`)**不**受影响 — AB 池是"对比所用样本",训练池是"模型学习的横截面",两者解耦。
- **A/B 测试**(独立配置文件 `ab.yaml`,主 `AppConfig` 不变):见 `docs/superpowers/specs/2026-05-24-ab-testing-design.md`。结构 `base_config: <path>` + 可选 `stocks_filter: [...]` (只能减) + `arms: {<name>: {strategy: {...}, backtest: {equity_curve_holding_days: [N], ...}}}` (恰好 2 个)。每个 arm 只能覆盖 `strategy` 段(整段替换)和 `backtest` 段(字段级合并,None 字段继承 base),其他顶层字段(`indicators`/`weights`/`verdicts`/`scoring`/`data`/`stocks`)继承 base。`equity_curve_holding_days` 强制单元素列表。ML 缓存通过 `effective_cfg.content_hash` 自动隔离,arm 间互不污染。`backtest.sizing` 是 F3 PR-C 起新支持的覆盖字段(整段替换,与 strategy 段语义对齐),用于比较 fixed vs vol_target sizing。

## 数据流

```
{mootdx | baostock | akshare} → fetcher (cache parquet) → indicators (add_all)
       → signals (detect_signals → score → verdict_of) + strategy.predict_latest()
       → report.render_report  /  backtest_composite.simulate_equity_curve
       → HTML
```

**日报路径里的 verdict 来自 `cfg.strategy.name` 指定的策略**(`cli._analyze_one`):
- 仍然计算综合评级的 triggers/scores/hit_rates 作为展示补充
- 最终 `verdict` 和 `final_score` 由 `strategy.predict_latest(daily)` 给出
- `composite_verdict`:直接算最后一根 bar(快,等价于老逻辑)
- `ml_factor`:从 `<cache_dir>/ml_models/<sig>_<code>.pkl` 加载已训练 pipeline+quantiles,**每个自然月最多重训一次**(同月内 predict-only,跨月自动重训并覆写缓存);`<sig>` = 8 位 MLFactorConfig 哈希,改 factors/horizon/selector/weighter/thresholds 等任一项即失效
- `pooled` 模式下 `cmd_run` 会预加载整池 `pool_data` 喂给 `build_strategy`,保证 cross-sec 因子有真实横截面值
- `cmd_run` 在 per-stock loop 之前**一次性预算** `sector_context_cache: {sector_name: ContextSignal}`(对 `cfg.stocks` 里出现的所有 unique sector 各调一次 `fetch_sector_daily` + `_compute_verdict`),`_analyze_one` 收到 cache 后直接复用,失败时 cache 里存的是错误 str,append warning 到对应股 — 避免多个股共享同板块时重复拉取
- **顶层各阶段计时**:`cmd_run` 会按 `[TIME] setup+config / market_index_context / _prepare_ml_pool / sector_context prefetch / per_stock_loop / pool_b / render_report / TOTAL cmd_run` 打印 stdout,便于排查 `python -m stockpool run` 速度回归。Pool B 内部按 200 股粒度也会打 `[TIME] Pool B i/total ... build_avg= predict_avg= ETA=` 进度
- **Partial NaN 容忍**(`generate_signals` 和 `predict_latest`):predict 路径不再要求 X 行所有因子都非 NaN — 用 fill 0 对 NaN 列做 impute(归一化后 0 是中性值),仅在**整行** NaN 时才返 `signal=neutral, score=NaN`。原因:`selection.json` 含 alpha_037 这种 200 日 rolling correlation 因子,warmup 必然 36% NaN 比例,严格 `notna().all()` 会让模型 100% 拒绝预测 → 0 trade。
- **`generate_signals` 同时输出 `score` 和 `final_score`**(两列数值相同),便于 portfolio 的 `precompute_scores_from_legacy`(默认读 `final_score`)直接对接,无需特殊代码路径
- **Portfolio 打分切片预建面板(2026-06-25 修正)**:`precompute_scores_from_legacy` 现在每股调 `scoring._set_stock_context(legacy, code)` 设 `_current_stock_code`,使 `generate_signals` **切片预建的横截面因子面板**(与训练一致、与日报逐股路径 `cli._analyze_one` **逐位一致**),而非旧的**逐股 `build_factor_matrix` 重算**——后者对截面因子(rank/industry_relative/rank-corr)在单股上得到退化的错值(实测 100% 单元格不同)。**⚠️ 这是正确性修正:所有 ml_factor 组合回测/AB 的分数会变(变正确);旧的 `data/portfolio_scores` 缓存已由 v2 key 自动失效。** 兼带 ~1.5× 提速(省去重算)+ `_strategy_signature` memoize(原每 bar 重算配置哈希)。code 不在面板时回退重算(off-panel 股仍可打分)。
- **`IndustryRelativeStrengthFactor`** 在 `get_sector_map()` 为空时**raise**(不再 silent 返全 NaN),防止 factor_panel cache 中毒 — sector_map 必须在 build factor_panel 前由 caller 通过 `factors.context.set_sector_map(...)` 注入(`backtest_runner.prepare_pool` / `cli.cmd_portfolio_*` / `recommend_pool` 都已做)

**Pool B(全市场量化推荐池)**在 `render_report` 之前由 `recommend_pool.compute_or_load_pool_b` 计算:
- 始终用 `load_universe_cache(cfg.data.cache_dir)` 作为**应用池**(独立于 strategy 训练池)
- 漏斗:**流动性**(近 20 日 `vol*close*100` ≥ `min_avg_amount_20d`) → **ST 二次防御**(name 含 ST) → 调当前 strategy 的 `predict_latest` 打分 → **行业上限贪心**(score 降序扫描,每行业 ≤ `max_per_industry`,收满 `top_n` 即停)
- 复用 `cli._prepare_ml_pool` 给 strategy 的 `pool_data` / `factor_panel` / **`close_panel`**(ml_factor + training_universe=all 时跨 4000 票 cross-sec 真实横截面),不重复加载。`close_panel` 必须透传 — 否则 `build_strategy` 会在每股调用里重跑 `build_close_panel(4000+ 股)`(每股 ~3s,4007 股 → ~3 小时;透传后 build_avg 降到 ~3ms)
- 缓存键 `poolb_<content_hash>_<isoyear>w<isoweek>.parquet`;同周 + 同 yaml 直接读盘,跨周或改 yaml 自动重算
- 失败隔离:per-stock predict 异常只 log warning 跳过该股;Pool B 整体失败不影响 Pool A 日报

缓存(`data/`):
- `<code>_daily.parquet` — 股票
- `idx_<symbol>.parquet` — 指数(`stock_zh_index_daily` 全量替换)
- `sector_<name>.parquet` — 行业板块
- `ml_models/<sig>_<code>.pkl` — ml_factor 日报路径的月度训练缓存(`share_pool_fit=false` 时)
- `ml_models/<sig>_shared.pkl` — `share_pool_fit=true` 时,所有股共享一份月度 pickle
- `universe.parquet` — `fetch-universe` 写入的全 A 股清单 (code/name/market)
- `stock_industry_map.parquet` — Pool B 用的 `code → 行业` 映射(akshare 东财板块,30 天有效期)
- `ipo_dates.parquet` — baostock `query_stock_basic` 的 `code → IPO 日期` 映射(5500+ 行,~30 KB);30 天有效期。`mask.enabled=true` 时 `MLFactorStrategy` 自动加载,传给 `_listing_mask` 替代有 bug 的 `first_valid_index` 启发式
- `fundamentals_profit.parquet` / `fundamentals_growth.parquet` / `fundamentals_balance.parquet` / `fundamentals_cash_flow.parquet` / `fundamentals_dupont.parquet` — baostock 季度财务长期缓存,含 `code / pubDate / statDate / <fields...>` 列,30 天有效期
- `recommend_pool/poolb_<content_hash>_<isoyear>w<NN>.parquet` — Pool B 本周排名缓存
- `ab_pool.parquet` — AB 候选池(`code/name/industry/circ_mv/avg_amount_20d/source_tag/build_date`),`ab-pool build` 生成,静态不变除非 `--refresh`
- `factor_panels/<sig>/{manifest.json, close.parquet, <factor>.parquet × N}` — ml_factor pooled mode 的因子面板 + close 宽表落盘缓存 (PR-2);sig hash 包含 factors / sorted codes / last_date,任一变化自动失效。回测命令加 `--refresh-factor-panel` 旁路
- `.data_source` — 单行文本,记录上次写入该 cache_dir 的 source(`mootdx`/`baostock`/`akshare`);任何 `fetch_*` 启动时与 cfg.data.source 比对,不一致触发 force_refresh + 覆写

报告:
- 日报:`reports/<YYYY-MM-DD>.html` + `reports/latest.html`
- 回测:`reports/backtest/<YYYY-MM-DD>.html` + `reports/backtest/latest.html`

## 测试

615 个,`pytest tests/ -q` 一次跑完。按域分布:

| 文件 | 覆盖 |
|---|---|
| `test_backtesting_framework.py` | 引擎契约、T+1、成本、扫 N、Strategy ABC |
| `test_multi_lot_engine.py` | 多仓位 lot 独立计时、现金约束、reset hook;`lot_sizer` 注入 + `Trade.lot_size` 透传 + skip-fallback 不开仓 |
| `test_timer_reset.py` | strong_buy 刷新计时;reset 与 exit 同时为真时 reset 胜出 |
| `test_backtest_composite.py` | 适配层、综合策略 walk-forward 等价性 |
| `test_backtest.py` | 单信号命中率 |
| `test_cli_backtest.py` | CLI 烟雾测试 + 中途单股失败不阻断的回归 |
| `test_ab.py` | ab/config 校验(arms==2、length-1 N、`extra=forbid`、`stocks_filter` 子集检查)+ `build_effective_cfg`(strategy 整段替换 / backtest 字段级合并 / content_hash 重算 / 非破坏)+ `_decide_pool_sharing` 5 种组合 + `run_ab` / `run_single_arm` 集成 + per-stock 失败隔离 + 报告 smoke (含一边空、common-stocks intersection) |
| `test_cli_ab.py` | `python -m stockpool ab` CLI smoke:happy path 出 HTML、`--arm` 调试模式只 stdout 不出 HTML、未知 arm 退出 2、`--no-share-pool` 短路验证(monkeypatch `_decide_pool_sharing` 断言调用次数 0) |
| `test_fetcher.py` | 缓存 + 增量更新 + `validate_ohlcv` + source-change marker 自动 force_refresh |
| `test_cli_fetch_universe.py` | `python -m stockpool fetch-universe` 默认按 `cfg.data.source`、`--source` 覆盖、source 变更自动 force_refresh |
| `test_indicators.py` | 数值正确性 |
| `test_signals.py` | 信号触发条件 |
| `test_factors.py` | 因子注册表 + 后缀参数解析 + 无 look-ahead + 数值正确 |
| `test_ml_pipeline.py` | Lasso 选稀疏 + IC/IR/equal weighter + TwoStepPipeline |
| `test_ml_selector_lightgbm.py` | LightGBMSelector: 非线性选 / top_k / min_importance_ratio / 确定性 / 退化输入 / TwoStepPipeline 集成 |
| `test_ml_weighter_lightgbm.py` | LightGBMWeighter: fit→predict 通 / mean&#124;SHAP&#124; weights / SHAP contributions 行和接近 predict / 确定性 / 退化输入 / TwoStepPipeline 集成 |
| `test_ml_strategy.py` | MLFactorStrategy walk-forward、per_stock/pooled、引擎集成 |
| `test_ops.py` | WQ 算子库:时间序列/横截面/indneutralize/look-ahead |
| `test_wq101.py` | 101 alpha 注册 + 元数据 + 计算无异常 + look-ahead 截断不变 |
| `test_panel.py` | Panel 构造 + 截尾 + 缺失 / 错位对齐 |
| `test_ml_strategy_panel.py` | factor_panel 注入 + with_stock 传播 + cross-sec 不退化 |
| `test_ml_strategy_panel_fit_reuse.py` | 注入 close_panel 后 `_try_fit` 走快路径(不调 `build_panel`),快/慢路径 (X, y) 数值等价;未注入则自动回退;`with_stock` 传播 close_panel;**PR-3** pre-stack cache 与 per-call stack 在多个 cutoff 上 bitwise 等价 + 触发后 `stack_panel_to_xy` 不再被调用 |
| `test_factor_panel_cache.py` | `load_or_build_factor_panel` 落盘缓存:首次写 manifest + parquets / 二次命中不调 build_factor_panel / 改因子或股池触发新 sig / `refresh=True` 旁路 / 缓存内容与新建一致 / 空 pool 返回空 |
| `test_config.py` | Pydantic 校验(含 `strategy` 段) |
| `test_report_smoke.py` | 全链路 `cmd_run` 烟雾 |
| `test_industry_map.py` | baostock + akshare 双源 mock,auto-fallback 链,parquet 缓存 / 过期 / failure-isolation |
| `test_ipo_dates.py` | baostock `query_stock_basic` mock,cache hit / stale / force_refresh / failure-fallback / NaT 过滤 |
| `test_recommend_pool.py` | Pool B 漏斗(流动性/ST/行业上限)+ ISO 周缓存 + content_hash 失效 + 失败隔离 |
| `test_factors_analysis.py` | FactorAnalysisResult / compute_daily_ic / classify_regimes / half-life / analyze_factors / pick_top_factors |
| `test_factors_analysis_report.py` | HTML 渲染烟雾 + 空 regime 处理 |
| `test_cli_factors_analyze.py` | `factors analyze` 与 `factors pick-by-ic` CLI 烟雾 |
| `test_ml_dataset_labels.py` | forward_return / forward_return_panel 的 label_type 接口(只 "return" 已实装) |
| `test_label_basis.py` | open-to-open 标签(`label_basis`):forward_return(_panel) open 基准数学 + mask 进出场 bar 检查 + embargo `_label_lag` +1 + build_panel open 标签 |
| `test_limit_rejection.py` | 涨跌停一字板拒单(`backtesting/limits.py`):两引擎涨停拒买/跌停顺延 + `infer_limit_pct` 板块阈值 |
| `test_ml_strategy_embargo.py` | walk-forward embargo: 默认 auto=horizon,explicit 0 恢复旧行为,泄露 bar 被排除 |
| `test_sizing.py` | FixedLotSizer / VolTargetLotSizer 数学 + fallback + build_lot_sizer 工厂 |
| `test_portfolio_strategy.py` | `PrecomputedScoreStrategy` 语义:已知日期 / 缺失日期 / NaN 丢弃 / 限定 panel_data codes / 未排序面板 / ABC |
| `test_portfolio_scoring.py` | `precompute_scores_from_legacy`:happy / per-stock 失败隔离 / 全失败返回空 / 缺 score_field 跳过 / 不截断 daily history |
| `test_portfolio_engine.py` | PortfolioEngine:空面板 / 零成本恒价等权不变 / 现金守恒 / T+1 fill 在 open[t+1] / start_offset / 确定性 / rebalance diff / 末 bar 不执行 / 未知 code 过滤 / initial_cash 缩放 |
| `test_portfolio_eligibility.py` | EligibilityFilter:min_history_bars / 流动性边界 / ST 排除 / 缺 name_map 不当 ST / date 截断 / 缺 volume 排除 / 阈值 0 跳过 / `_is_st` parametrize |
| `test_portfolio_industry_cap.py` | `_select_top_k` 行业 cap:贪心正确性 / cap=None 不限 / sector_map 空时不限 / 全 unknown 跳过 cap / 部分 unknown 走 Unknown 桶 / 缺 open[t+1] 不计 cap / engine 集成 |
| `test_cli_portfolio_backtest.py` | `python -m stockpool portfolio-backtest`:smoke (composite_verdict 出 HTML) + `enabled=false` 退出 2 + `--refresh-scores` 旁路缓存 + `universe.parquet` 自动扩 universe + `staggered_starts=3` 出 ensemble HTML |
| `test_portfolio_ensemble.py` | `StaggeredRunner`:N=1 与单跑等价 / N=n_days 时 rebalance bar 集合 pairwise disjoint / ensemble = 等权均值数学等价 / envelope 列序 + 分位序 / aggregated_metrics 结构 / n_offsets<1 raises |
| `test_portfolio_report_ensemble.py` | `render_ensemble_report`:HTML smoke (含 "Per-offset metrics" + "Ensemble net asset value") + 空 ensemble 渲 Empty 占位 |
| `test_portfolio_ab_config.py` | PortfolioABConfig:arms != 2 拒 / extra=forbid / build_effective_cfg 整段替换 strategy + 字段合并 portfolio_backtest + 不动 base / 重算 content_hash 且可重现 |
| `test_portfolio_ab_runner.py` | run_portfolio_ab happy + per-arm failure isolation + ArmResult empty-failed 不 crash |
| `test_portfolio_ab_report.py` | render_portfolio_ab_report HTML smoke (含贡献分解 + 集合分析) + 失败 arm 红 banner + 0 arm 错误占位 |
| `test_cli_portfolio_ab.py` | `python -m stockpool portfolio-ab`:happy / unknown arm 退 2 / `--arm` 调试模式只 stdout 不出 HTML / base `enabled=false` 退 2 |
| `test_ab_pool.py` | AB 候选池: AbPoolConfig + 硬过滤(ST/IPO/流动性/NaN) + 行业分层 top-2+2 with overlap merge + akshare snapshot mock + 20d liquidity calc + build orchestration(idempotent guard, refresh, empty buckets)+ HTML 渲染 smoke + CLI build/show smoke + ab.yaml & portfolio_ab.yaml `use_ab_pool` 集成 |
| `test_panel_mask.py` | `_limit_threshold` 板块映射 + `_listing_mask` 成熟/新股 + `compute_tradability_mask` 三条件 + `apply_mask` NaN-out 正确性 + IPO 日期路径(成熟股短缓存不误屏蔽 / panel 内新 IPO 正确屏蔽 / 缺 IPO 默认成熟) |
| `test_ops_mask_nan_safe.py` | `ts_mean/sum/std/product` 放宽 `min_periods` + `decay_linear` NaN-safe 重归一化;全 valid 输入下数值不变 |
| `test_ml_strategy_mask.py` | `compute_factor_panel` / `forward_return_panel` / `build_factor_panel` / `build_panel` / `build_factor_matrix` 各层 mask 参数语义 + MLFactorStrategy sig 变化 + pooled/per_stock spy 验证 |
| `test_ml_preprocess.py` | 截面预处理 — 3 函数(winsorize / cs_zscore / industry_neutralize)+ `apply_preprocess_pipeline` 流水线 + `_is_all_off` short-circuit,20 个 case |
| `test_ml_preprocess_mcap.py` | `mcap_neutralize_panel` OLS 残差化 / `industry_neutralize_panel(log_mcap=...)` 联合 OLS / `apply_preprocess_pipeline` 路由与 skip 规则 / 单成员 industry 不被零化 / fallback 计数 |
| `test_ml_mcap.py` | `build_log_mcap_panel`:close × totalShare、PIT 对齐、缺数据 NaN、空 balance 全 NaN 输出 |
| `test_strategy_factory_mcap.py` | `build_factor_panel(cache_dir=...)` 在 `mcap_neutralize=True` 时调 `build_log_mcap_panel` 并传给 pipeline / disabled 时不 fetch balance / `cache_dir=None` 时 warning + skip |
| `test_factors_original_stats.py` | rolling 直接统计因子注册 + 数值 + look-ahead |
| `test_factors_ewma.py` | EWMA 平滑因子 halflife 解析 + 公式对照 |
| `test_factors_vwap_deviation.py` | VWAP 偏离族注册 + 单调性 + 无 look-ahead |
| `test_factors_close_position.py` | 收盘位置 ∈ [0,1]、涨停封板 range=0 NaN 守护 |
| `test_factors_turnover_extra.py` | 短窗换手族、停牌日 volume=0 NaN 守护 |
| `test_factors_acceleration.py` | 二阶差分公式对照 + 无 look-ahead |
| `test_factors_single_stock_vol.py` | ATR/CCI/振幅/Parkinson 正性 + 无 look-ahead |
| `test_factors_composite.py` | 复合算子拼装的注册 + 无 look-ahead |
| `test_factors_rank_correlation.py` | 秩相关 ∈ [-1,1] + 无 look-ahead |
| `test_factors_cross_sec_breadth.py` | 全市场标量广播 + 涨停股算入宽度分子(spec §6.1.2) |
| `test_factors_fundamentals.py` | 关键 PIT 测试:pubDate 之前 NaN、pubDate 后 ffill、亏损 PE NaN |
| `test_fundamentals_loader.py` | baostock mock + cache hit / stale / force_refresh / failure-fallback |
| `test_cli_refresh_fundamentals.py` | `--refresh-fundamentals` argparse wiring on run/backtest/portfolio-backtest |

写测试时:**用合成 OHLCV、`monkeypatch` 掉 AKShare 和 `_today`**(`test_cli_backtest.py` 是参考)。

## 因子库 (`stockpool.factors`)

**Factor ABC 是 panel-in / panel-out**(2026-05 重构):
```python
class Factor(ABC):
    sources: tuple[str, ...] = ("builtin",)   # 来源标签
    types: tuple[str, ...] = ()               # 类型多标签
    description: str
    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame: ...
```

Panel = `{"open"|"high"|"low"|"close"|"volume": T×N DataFrame}`,行 = date,列 = code。

**注册表**支持双轴元数据:
- `sources`: `builtin` / `wq101` / `custom`
- `types`: `momentum` / `reversal` / `trend` / `volatility` / `volume` /
           `time_series` / `cross_sectional` / `industry_neutral` / `fundamental`

**API**: `list_specs()` / `filter_specs(sources=, types=, match='any')` /
       `all_sources()` / `all_types()` / `make_factor(name)`.

**算子库** `factors/ops.py`(WQ101 必需):
- 时间序列: `ts_sum` `ts_mean` `ts_min/max` `ts_argmin/argmax` `ts_rank`
  `ts_std` `ts_product` `delta` `delay` `decay_linear` `correlation` `covariance`
- 横截面: `rank`(axis=1, pct=True) `scale`(L1 norm) `signedpower`
- 行业中性: `indneutralize(x, group_map)` —— 按 group 分组 demean
- 工具: `safe_div` `vwap`(`(H+L+C)/3` proxy) `adv(volume, d)`

**WQ101**(`factors/wq101.py`):全 101 个 alpha,名字 `alpha_001` .. `alpha_101`。
- 注入 `set_sector_map({code: sector})` 后,所有 `IndNeutralize` 退化的 alpha 走分组 demean;未注入则退化为整体 demean
- `IndClass.subindustry` 一律 fall back 到 sector(项目无 subindustry 数据)
- `Alpha056` 需要 `cap`(总市值),目前返回全 NaN
- **预算因子面板**:`strategy_factory.build_strategy` 在 `panel_mode=pooled` 且有 `pool_data` 时,会调 `build_factor_panel`(顶层助手)预算一份 `{factor_name: T×N}` 并注入 `MLFactorStrategy(factor_panel=...)`。`generate_signals` 通过 `_build_x_full` 切出本股的 X(`slice_stock_factor_matrix`),所以 cross-sec 因子在 predict 阶段也用真实横截面值,与训练一致。不注入时(``per_stock`` 模式或单股测试)fall back 到 `build_factor_matrix` 单股退化。CLI (`cli._prepare_ml_pool`) 在 8 只票循环外**预算 panel 一次**,避免每股重算 4000+ 列。
- **训练/应用池分离**:`training_universe=all` 时,`cli._prepare_ml_pool` 用 `load_universe_cache(data/)` 装全市场 ~4350 只票作 pool_data(应用池 cfg.stocks 仍 merge 进去保证存在);predict 仍只对 cfg.stocks 跑。意味着 cross-sec 因子和 IC 加权拿到的是全市场横截面,而日报/回测的标的仍是 cfg.stocks 那几只。
- **WQ101 本土化变体(`wq101_variants.py`,2026-06-24)**:`scripts/generate_wq101_variants.py` 从 IC 基线对 top-N wq101 alpha 应用 3 种窗口规则(`_compress`/`_rev_short`/`_expand_long`),生成 `factors/wq101_variants.py`(`factors/__init__` guarded import,文件不存在时无害跳过)。原版与变体共存,经 `selection.json` 切换。**Round 1 实测:窗口本土化对 top-30 存活 alpha 无实质 IC 增益(0 winner),默认未启用**;详见 `docs/superpowers/specs/2026-06-21-wq101-a-share-localization-design.md` §14 + `docs/wq101_localization_worklog_2026-06-24.md`。
- **`factors analyze` 退化日诊断(2026-06-24 起两道闸)**:`analyze_factors` 默认 `winsorize=(0.01,0.99)` + `degenerate_day_unique_ratio_threshold=0.01`(抓并列/离散因子)+ **`min_coverage_frac=0.05`**(抓**稀疏**因子:某日因子覆盖 < 当日可投横截面的 5% → 当日 IC 置 NaN)。后者修了一个真 bug:深层嵌套长窗口 alpha(如 `alpha_096`,全市场每天仅 ~3 只有效)的 ±1 噪声 IC 此前虚高 abs_ic 到 0.40(全场第 1)、ratio 检测抓不到。CLI `--min-coverage-frac`。⚠️ **旧 analyze 报告的 IC 数字与新版不可直接比较;用旧 buggy IC 选出的 `selection.json` 含 ~4 个幻象因子(alpha_027/059/061/095),建议在 clean 基线上重跑 pick-by-ic 重建并 AB 验证**(候选见 `reports/selection_clean_rebuild_candidate.json`)。
- **`pick-by-ic` 覆盖率 gate(2026-06-24)**:`pick_top_factors` 新增 `max_degenerate_ratio`(默认 0.5,CLI `--max-degenerate-ratio`):剔除 `degenerate_day_ratio` > 阈值的因子(IC 建在不足半数交易日上 = 噪声)。与 analyze 侧覆盖度下限互补——前者(全 NaN 的幻象)靠 NaN-score 自动排除,本 gate 专抓**部分退化**因子(如 alpha_048 51% 噪声日仍有非 NaN IC 会溜进 selection)。
- **GTJA191 因子族(`gtja191.py`,2026-06-24,验证子集 25 个)**:国泰君安 191 alpha = WQ101 的 A 股本土化对应物(短周期量价,窗口为 A 股设计),`sources=("gtja191",)`,名 `gtja_NNN`。**经 A 股 correlation 研究决策落地**:与其修 WQ101 的 correlation 算子(常数输入→NaN 数学上正确,业界也不在算子层改),不如换一套为 A 股设计的因子集。当前仅收录能用现有算子 + 新增 `ops.sma`(=`ewm(alpha=m/n)`)忠实移植的 25 个,跳过 WMA/REGBETA/REGRESI/SEQUENCE 等未实现/歧义算子;后续补齐算子可扩展。`factors/__init__` 自动导入。须经 `factors analyze`/`pick-by-ic`(覆盖率 gate 把关)选入并 A/B 验证后才进生产。

### 2026-05-31 因子扩展(11 家族 + 基本面)

114 → ~165 个 base 因子(变体计含 ~280-320),按 11 家族扩展:

- **VWAP 偏离 / 收盘位置 / 秩相关 / 单股波动 / 短窗换手 / 复合 / 加速度 / 直接统计 / 截面宽度 / EWMA / 基本面**

命名语义化(`vwap_dev_5` / `close_pos_10` / `roe` 等)。Mask 行为完全沿用现状:因子看 raw close,mask 只在标签层。

基本面族 PIT 设计:按 `pubDate`(公告日)前向填充,**不**用 `statDate`,防 ~1 个月未来泄露。首次拉 baostock 5 张季度表约 30-60 分钟,30 天缓存到 `data/fundamentals_*.parquet`。`--refresh-fundamentals` 强制重拉。

`factor_panels/<sig>/manifest.json` 现含 `fundamentals_snapshot_date` 字段:若 fundamentals parquet 在 panel 缓存之后被刷新,panel 缓存自动失效重建(spec §5.6)。

### HTML 选择器

```bash
python -m stockpool factors pick                          # 默认 server 模式
python -m stockpool factors pick --output my_sel.json     # 指定写入路径
python -m stockpool factors pick --port 18765             # 固定端口
python -m stockpool factors pick --static                 # 老的静态文件模式
```

**默认 server 模式** (推荐):起一个 `127.0.0.1` 本地 HTTP 服务(stdlib `http.server`,无新增依赖),浏览器打开页面。顶栏 **"应用"** 按钮 POST 到 `/save` 由服务端直接写 `reports/selection.json`(或 `--output` 指定路径)。Ctrl-C 退出服务。页面打开时也会 GET `/selection.json` 把现有选择载回来(以服务端文件为权威源,覆盖 localStorage)。

**`--static` 模式**:回退到生成静态 HTML 文件 (`file://`),"应用" 按钮在无服务端时自动降级为"下载"。适合归档 HTML 或 server 模式被防火墙挡住。

左侧双轴筛选(来源 × 类型)+ 任一/全部模式;右侧卡片勾选。顶栏按钮:**应用** / 下载 selection.json / 复制 YAML / 勾选当前可见 / 清空。

服务端路由(见 `factors_picker._make_handler`):
- `GET /` → HTML 页面
- `GET /selection.json` → 当前文件内容(不存在时返回 `{"factors": []}`)
- `POST /save` → 写文件,返回 `{"ok": true, "path": "..."}`

`config.yaml` 引用:
```yaml
strategy:
  ml_factor:
    factors_file: reports/selection.json   # 与 factors: [...] 二选一
```

## 已知不支持的能力

- 做空、多标的组合、盘中数据、部分成交、资金成本(融资融券)
- 仓位管理仅"满仓单笔"或"固定额度多笔"两种;无 Kelly / 比例追加
- 个股 → 板块的**自动**映射:`cfg.stocks[].sector` 仍需手填(中文名或 6 位 88xxxx 代码);**Pool B 的 code→行业映射独立**走 baostock/akshare(见 `industry_map.py`)。mootdx 路径不可用 —— 实测 TDX 服务器对 `block_hy.dat` 返回 0 字节,触发 tdxpy 的空 bytearray bug
- **A/B 测试不能覆盖顶层 `indicators` / `weights` / `verdicts` / `scoring`**:`composite_verdict` 的参数还散在 `AppConfig` 顶层(历史遗留;`ml_factor` 已规整到 `strategy.ml_factor.*` 子段),A/B arm 暂时只允许覆盖 `strategy:` 和 `backtest:`。两个 `composite_verdict` arm 想比不同 `weights` → 当前只能改主 cfg 跑两次。Follow-up:把 `composite_verdict` 的参数下沉到 `strategy.composite_verdict.*` 子段,A/B 工具会自动获益(`ab/` 代码不动一行)
- **A/B 不做 portfolio-level 回测**:每个 arm 仍是 per-stock 独立回测 + 跨股聚合统计(均值/中位/胜出数)。真正的 portfolio-level(策略在每根 bar 看到整个股池横截面,产出一条组合净值)需要新的 `PortfolioStrategy` ABC + portfolio engine,是独立 spec
- **A/B 报告无统计显著性**:8-30 只股票样本太小,p 值不稳;聚合表只给均值/中位/差值/胜出计数。Pool B 联动扩到几百只股后再考虑加 paired t-test / Wilcoxon
- **Portfolio framework PR-4 限制**:四个 PR 全部落地。`portfolio_backtest.enabled` 默认 false,需手动 opt-in。Staggered 第一版串行(没并行化,N=10 大约 N=1 的 10 倍耗时);portfolio AB 的 ArmOverride 仍只允许 `strategy:` 和 `portfolio_backtest:` 两段(其他顶层字段如 `data` / `weights` / `indicators` 想跨 arm 改还得改 base cfg 跑两次);两 arm 各自算各自的 score panel(per-arm content_hash 隔离),没做"同 hash 共享"优化。并行化与跨 arm 缓存共享都列在 spec §12 follow-up
- **可交易性 mask 已落地 mask_price(因子输入侧)**:完整 mask 含义见 `docs/superpowers/specs/2026-05-31-tradability-mask-design.md`。`mask_exec`(open-side 执行可填性,即开盘涨停 fill guard)未落地 — 单独 PR 处理。回测引擎仍假设 `open[t+1]` 一定可成交,这一假设在涨停封板的极少数情况下偏乐观。

## 改动后更新文档(CLAUDE.md + README.md)

**新增 / 修改任何面向用户的功能或设计时,必须同时更新 `CLAUDE.md` 和 `README.md`。
两份文档面向不同读者,缺一不可:**

- `CLAUDE.md` — 给后续 Claude 会话的内部地图(模块职责、API 契约、设计权衡)。
  原则:**后续 Claude 不读源码也能正确帮用户做事**。
- `README.md` — 给项目用户(包括人和 AI)的入口文档(快速开始、常用命令、配置示例)。
  原则:**新用户从 README 能跑通核心场景**。

### 何时必须更新

只要满足下列任一项,就要在**同一次改动**里把两份文档一并改完:

- 新增 / 删除 / 重命名顶层模块或公开 API
- `Strategy` ABC、`BacktestEngine` / `MultiLotBacktestEngine` 的公开签名变化
- `config.py` 中 schema 字段新增 / 删除 / 默认值改变 / 语义改变
- CLI 子命令的增减、参数变化、命令行用法示例变化
- 默认行为(尤其是 `engine`、`refresh_verdicts`、`sizing`、数据源行业板块路由)的切换
- 测试目录新增的"按域覆盖"文件 → CLAUDE.md 的测试表加一行
- 数据流 / 缓存路径 / 报告路径 / 数据源 行为变化
- 新增的因子来源/类型标签(影响 HTML picker 和 `factors list --source/--type` 用例)

### 怎么改

- CLAUDE.md:定位对应小节(模块地图 / 配置 / 测试 / 因子库 / 数据流 / 已知不支持),把改动点写进去
- README.md:更新 **快速开始** / **常用命令** 段;如果新增了用户级流程(如 `factors pick` HTML 选择器),加一段端到端示例
- 两边的命令示例要保持一致(同一份 `python -m stockpool ...` 命令不要在两份文档里写不同的参数)

**不需要每个 commit 都更新**。只在上述变化发生时跟一次,把对应小节改掉即可。
但**只更新 CLAUDE.md 不更新 README,或反过来,算未完成**——改动评审时按"两份都到位"判定。
