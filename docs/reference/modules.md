# 模块地图 (详细)

> CLAUDE.md 的模块表只列一句话职责;本文是完整版,含 API 契约与设计细节。

## 入口 / CLI

| 文件 | 职责 |
|---|---|
| `src/stockpool/cli.py` | 入口,定义 `run` / `backtest` / `fetch-universe` / `factors *` / `ab` / `portfolio-backtest` / `portfolio-ab` 子命令 |
| `src/stockpool/config.py` | Pydantic schema + YAML 加载;**配置变更必须更新这里**。详见 [config.md](config.md) |
| `src/stockpool/strategy_factory.py` | 按 `cfg.strategy.name` 工厂构造策略 + ML 通用 simulate;`build_factor_panel` / `build_close_panel` / `load_or_build_factor_panel` 顶层助手;市值面板注入。详见下方 [strategy_factory 细节](#strategy_factory-细节) |

## 数据获取

| 文件 | 职责 |
|---|---|
| `src/stockpool/fetcher.py` | 公开 API + Parquet 缓存 + OHLCV 校验;按 `cfg.data.source` 派发后端;价格统一**后复权 (hfq)**,akshare 走 `adjust="hfq"`;`_drop_in_progress_bar` 在 15:05 前丢当日半根 bar;增量从缓存最后一天(含)重叠拉取,`_reconcile_increment` 接缝校验(close >0.1% / volume >1% 偏差 → 全量重拉)+ mootdx 段锚定;`fetch_universe`/`list_universe`/`load_universe_cache` 全市场批拉与读盘;`check_source_change`/`update_source_marker` 维护 `data/.data_source` 标记(`<source>:hfq`),源或复权口径变化时自动 force_refresh |
| `src/stockpool/data_sources/mootdx_backend.py` | 通达信 TCP 后端(默认)。股票/指数/**行业板块(88xxxx)**;含当日盘中,TDX 占位 bar 会被丢弃。`_fetch_xdxr` + `_apply_hfq` 做段内锚定后复权(段首因子=1,事件因子 `P_prev/P_ex`,volume 不复权)。行业名→代码映射 `_TDX_INDUSTRY_CODES` |
| `src/stockpool/data_sources/baostock_backend.py` | baostock 后端(无 token,收盘后约 18:00 更新);`adjustflag="1"` 后复权;不支持板块,板块自动走 mootdx |
| `src/stockpool/fundamentals_loader.py` | baostock 5 张季度财务表 (profit/growth/balance/cash_flow/dupont) PIT 长期缓存,parquet 30 天 staleness |
| `src/stockpool/industry_map.py` | `code → 行业` 映射;多源(`auto` / `baostock` / `akshare`);缓存 `data/stock_industry_map.parquet`,>30 天过期自动重拉。**mootdx 路径无效**:TDX 服务器对 `block_hy.dat` 返回 0 字节 |
| `src/stockpool/ipo_dates.py` | `code → IPO 日期` 映射(baostock `query_stock_basic`,5500+ 股一次性 ~3-5 秒);缓存 `data/ipo_dates.parquet` 30 天有效。`mask.enabled=true` 时 `MLFactorStrategy._get_ipo_dates` 自动加载,传给 `_listing_mask`,替代有 bug 的 `first_valid_index` 启发式 |

## 信号 / 指标

| 文件 | 职责 |
|---|---|
| `src/stockpool/indicators.py` | MA / MACD / KDJ / RSI / BOLL / Volume / Breakout |
| `src/stockpool/signals.py` | `detect_signals` + `score_triggers` + `combine_daily_weekly` + `verdict_of` |

## 因子库

见 [factors.md](factors.md) 的完整说明。模块:

| 文件 | 职责 |
|---|---|
| `src/stockpool/factors/` | 连续因子库(Factor ABC + 注册表 + 内置技术因子 + WQ101) |
| `src/stockpool/factors/ops.py` | WQ 算子库 (ts_rank / rank / decay_linear / indneutralize / ...) |
| `src/stockpool/factors/wq101.py` | WorldQuant 101 Formulaic Alphas 全套 (Alpha001..Alpha101) |
| `src/stockpool/factors/original_stats.py` | rolling 直接统计量族,~25 变体 |
| `src/stockpool/factors/ewma.py` | EWMA 平滑动量/波动/换手族(halflife 参数化),~15 变体 |
| `src/stockpool/factors/vwap_deviation.py` | VWAP 偏离族,~20 变体 |
| `src/stockpool/factors/close_position.py` | 收盘位置动量族 ((C-L)/(H-L)),~15 变体 |
| `src/stockpool/factors/turnover_extra.py` | 短窗换手 z-score / 量比,~12 变体 |
| `src/stockpool/factors/acceleration.py` | 动量/换手二阶差分,~9 变体 |
| `src/stockpool/factors/single_stock_vol.py` | ATR / CCI / 振幅 / Parkinson / Garman-Klass 波动率,~20 变体 |
| `src/stockpool/factors/composite.py` | rank/decay/scale 复合算子拼装,~12 变体 |
| `src/stockpool/factors/rank_correlation.py` | 价格秩 × 成交量秩滚动相关,~20 变体 |
| `src/stockpool/factors/cross_sec_breadth.py` | 全市场宽度因子,~7 变体。⚠️ 涨停股算入分子 |
| `src/stockpool/factors/fundamentals.py` | 基本面因子(PE/PB/ROE/ROA/毛利率/净利率/营收 YOY),baostock 5 张季度表,严格 PIT |
| `src/stockpool/factors_picker.py` | HTML 因子选择器 + `factors` CLI 子命令 |
| `src/stockpool/factors_analysis.py` | 因子分析:滚动 IC / IR / half-life / 相关性 / regime 切片;`analyze_factors` + `pick_top_factors` |
| `src/stockpool/factors_analysis_report.py` | pyecharts HTML 报告:排名表 + IC 时序 + 相关性 heatmap + regime 拆分 |

## Panel / ML

| 文件 | 职责 |
|---|---|
| `src/stockpool/panel.py` | Panel 数据结构 (T×N 宽表 dict) + `build_panel_from_cache`。**+** `compute_tradability_mask` / `apply_mask` / `_limit_threshold` / `_listing_mask`(板块可交易性 mask)。详见 [tradability mask](#tradability-mask-细节) |
| `src/stockpool/ml/` | 两步法 ML 组合(dataset / Lasso 或 LightGBM selector / IC&IR&Equal&LightGBM weighter / TwoStepPipeline) |
| `src/stockpool/ml/preprocess.py` | 截面预处理流水线 (winsorize / cs_zscore / industry_neutralize / market_cap_neutralize / symmetric_orthogonalize)。详见 [config.md](config.md) preprocess 段 |

## 回测框架

见 [../backtesting_framework.md](../backtesting_framework.md) 的完整 API,以及 [conventions.md](conventions.md) 的引擎约定。模块:

| 文件 | 职责 |
|---|---|
| `src/stockpool/backtesting/` | 回测框架(策略 ABC + 引擎) |
| `src/stockpool/backtesting/sizing.py` | `LotSizer` Protocol + `FixedLotSizer` / `VolTargetLotSizer` + `build_lot_sizer` 工厂 |
| `src/stockpool/backtest_runner.py` | 共享给 `cli.cmd_backtest` 和 `ab.runner`:`prepare_pool`(ml_factor 池 + factor_panel 预算)和 `backtest_stocks`(per-stock 回测循环,失败隔离) |
| `src/stockpool/backtest_composite.py` | 综合策略回测旧 API 适配层,委托给框架 |
| `src/stockpool/backtest.py` | 单信号前瞻命中率 |
| `src/stockpool/backtest_report.py` | 回测 HTML 渲染 |

## Portfolio 框架

PR-1/2/3/4 全部落地。见 [config.md](config.md) 的 `portfolio_backtest` 段。模块:

| 文件 | 职责 |
|---|---|
| `src/stockpool/portfolio/strategy.py` | `PortfolioStrategy` ABC + `PrecomputedScoreStrategy` |
| `src/stockpool/portfolio/scoring.py` | `precompute_scores_from_legacy` — 把 per-stock `Strategy.generate_signals` 拼成 T×N score 面板 |
| `src/stockpool/portfolio/engine.py` | `PortfolioEngine` — top-K 等权 + 周期 rebalance + T+1 + 行业 cap 贪心 + eligibility 过滤 |
| `src/stockpool/portfolio/eligibility.py` | `EligibilityFilter` — 流动性/ST/min_history_bars |
| `src/stockpool/portfolio/ensemble.py` | `StaggeredRunner` + `EnsembleResult` — N 个 `start_offset` 串行跑,聚合 envelope + 等权均值 ensemble curve |
| `src/stockpool/portfolio/result.py` | `PortfolioBacktestResult` / `PortfolioTrade` |
| `src/stockpool/portfolio/report.py` | 单 arm HTML + ensemble HTML(`render_portfolio_report` / `render_ensemble_report`) |

**PR-2/3 行为要点**:universe = `load_universe_cache`(`universe.parquet` 不存在时回退 cfg.stocks),`sector_map` 来自 `load_or_build_industry_map`,行业 cap "未知" 全空时跳过 cap、否则 Unknown 桶正常计;`staggered_starts > 1` 自动走 `StaggeredRunner`(串行,engine_factory 闭包保证每 offset fresh state),N=1 退化回单 arm 路径。

## A/B 测试子包

| 文件 | 职责 |
|---|---|
| `src/stockpool/ab/` | per-stock A/B:`config.py`(`ABConfig` / `ArmOverride` / `load_ab_config` / `build_effective_cfg`)+ `runner.py`(`run_ab` / `run_single_arm` / `_decide_pool_sharing` + `ArmResult`/`ABResult`)+ `report.py`(HTML 对比报告) |
| `src/stockpool/portfolio_ab/` | Portfolio AB(PR-4):`config.py`(arm 仅允许覆盖 `strategy` 整段替换 + `portfolio_backtest` 字段级合并)+ `runner.py`(共享 universe + sector_map + name_map,per-arm failure 隔离)+ `report.py`(`render_portfolio_ab_report`) |

## 报告 / 推荐池

| 文件 | 职责 |
|---|---|
| `src/stockpool/report.py` | 日报 HTML(市场背景、板块上下文);`_optimize_html` 做 echarts lib 去重 + `<details>` 折叠 + 懒加载 |
| `src/stockpool/recommend_pool.py` | **Pool B**(全市场量化推荐池):漏斗 + 排序 + 周缓存;`compute_or_load_pool_b` 顶层 API。接收 caller 预算好的 `pool_data` / `factor_panel` / `close_panel` 一路透传,避免每股重跑 `build_close_panel`。详见 [conventions.md](conventions.md) Pool B 段 |

---

## strategy_factory 细节

- `build_strategy` 在 `panel_mode=pooled` 且有 `pool_data` 时,调 `build_factor_panel` 预算 `{factor_name: T×N}` 并注入 `MLFactorStrategy(factor_panel=...)`。不注入时 fall back 到 `build_factor_matrix` 单股退化。
- `load_or_build_factor_panel` 落盘缓存,key = (sorted factors + sorted codes + last_date + preprocess) 的 sha256[:12],写 `data/factor_panels/<sig>/{manifest.json, close.parquet, <factor>.parquet × N}`;input 任一变化生成新 sig 重算。
- ml_factor 注入 `cache_dir` 以启用日报路径的月度训练缓存。

**Phase 2 市值中性**:
- `build_log_mcap_panel(pool_data, cache_dir)` 从 `data/mcap_shares.parquet`(最新 totalShare 快照,`scripts/pull_mcap_profit.py` 拉)算 `log(close × totalShare)`(shares 静态广播,close 仍日频 PIT)。
- `maybe_inject_mcap_panel(preprocess_cfg, pool_data, cache_dir)` 在 `market_cap_neutralize=True` 时把该面板经 `factors.context.set_mcap_panel` 注入(prepare_pool / cli / ab.runner 三处 caller 已 wire,与 set_sector_map 并列);`build_factor_panel` 经 `get_mcap_panel()` 取回喂给 `apply_preprocess_pipeline`。

**PR-3 性能**:
- `MLFactorStrategy._ensure_pooled_xy_long` 把全历史 stack 提到 `shared_cache["__pooled_xy_long__", sig]`,后续 refit 只做 `X.loc[:label_cutoff_ts]`(按 `label_end-1` 切,避免 horizon 行未来 label 泄露)+ per-stock tail,杜绝每次 refit 重 stack。
- `_try_fit` 用 `searchsorted` 早退跳过训练集明显太小的 cutoff。
- `stack_panel_to_xy` 内部改 numpy ravel + column_stack 替代逐因子 `.stack()`,~35× 加速。

## tradability mask 细节

`panel.py` 的 mask 支持按板块(主板 ±10% / 创业板+科创 ±20%)的可交易性 mask。`_listing_mask` 接 `ipo_dates: Mapping[str, Timestamp] | None`,有真实 IPO 日期时按 IPO + 366 自然日(≈252 交易日)cutoff 屏蔽新股;无 ipo_dates 时退化到 first_valid_index 启发式(打 warning)。

**注意(2026-05-31 重构)**:mask **不** 应用到因子输入面板 — 时间序列因子需要看真实 close(涨停日 +9.9% 本身是有用信号)。Mask 只在 `forward_return_panel` 的双向标签检查(`mask[t] ∧ mask[t+horizon]`)和训练样本 dropna 上生效。`apply_mask` 仍是公开工具,供有特殊需求的因子 `compute` 方法按需调用。详见 `docs/handoff/2026-05-31-mask-ab-investigation.md` 与 `docs/superpowers/specs/2026-05-31-tradability-mask-design.md`。
