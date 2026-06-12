# 测试参考

~857 个测试,`python -m pytest tests/ -q` 一次跑完(约 3 分钟)。

**写测试时**:用合成 OHLCV、`monkeypatch` 掉 AKShare 和 `_today`(`test_cli_backtest.py` 是参考)。**新增按域覆盖文件时,在本表加一行。**

## 按域覆盖

| 文件 | 覆盖 |
|---|---|
| `test_consistency_contract.py` | **predict/backtest 一致性契约**(composite+ml)+ 端到端黄金值锚点(精确 equity/metrics) |
| `test_fetcher_adjust.py` | hfq 复权(段内锚定/接缝校验/盘中 bar/marker 迁移) |
| `test_volume_unit.py` / `test_mootdx_pagination.py` | volume 手→股统一;800 根分页 |
| `test_baostock_backend.py` | 线程锁、tradestatus 停牌行、增量空结果 |
| `test_fundamentals_fix.py` | PE/PB/ROA 字段修复、TTM、PIT 细节、覆盖率补拉、--refresh-fundamentals |
| `test_listing_pit.py` | stock_basics PIT 名单、ST 保留/标记、ST ±5% mask |
| `test_trade_calendar.py` | 交易日历、NO_PROXY 环境变量 |
| `test_label_basis.py` | open-to-open 标签数学、mask 进出场 bar、embargo +1 |
| `test_weighters_daily_ic.py` | 逐日截面 IC、IR 时间分块、行序不变性 |
| `test_factors_analysis_alignment.py` | analyze 口径对齐、end-date 截断、Newey-West |
| `test_preprocess_interlock.py` | broadcast×zscore 互锁、覆盖率总闸、build_strategy 回退 |
| `test_predict_impute.py` | fill_missing fit 均值、mcap 缺失 NaN、Lasso y 标准化 |
| `test_factor_panel_sig_completeness.py` | sig 含 first_date、aux mtime 失效 |
| `test_limit_rejection.py` | 涨跌停拒单(单仓/多 lot)、infer_limit_pct |
| `test_portfolio_engine_realism.py` | 差量调仓、turnover、退市核销、无 opens_next 窥视、min_commission |
| `test_entry_mode_edge.py` | edge 开仓、sizer 历史切片 |
| `test_config_guards.py` | verdicts/scoring/indicators 校验、extra=forbid |
| `test_scoring_with_stock.py` | 组合 scoring 逐股 with_stock 绑定 |
| `test_mcap_pit_shares.py` | mcap PIT 股本 + 快照回退 |
| `test_backtesting_framework.py` | 引擎契约、T+1、成本、扫 N、Strategy ABC |
| `test_multi_lot_engine.py` | 多仓位 lot 独立计时、现金约束、reset hook;`lot_sizer` 注入 + `Trade.lot_size` 透传 + skip-fallback |
| `test_timer_reset.py` | strong_buy 刷新计时;reset 与 exit 同时为真时 reset 胜出 |
| `test_backtest_composite.py` | 适配层、综合策略 walk-forward 等价性 |
| `test_backtest.py` | 单信号命中率 |
| `test_cli_backtest.py` | CLI 烟雾 + 中途单股失败不阻断回归 |
| `test_ab.py` | ab/config 校验 + `build_effective_cfg` + `_decide_pool_sharing` + `run_ab`/`run_single_arm` 集成 + 失败隔离 + 报告 smoke |
| `test_cli_ab.py` | `stockpool ab` CLI smoke:happy / `--arm` 只 stdout / 未知 arm 退 2 / `--no-share-pool` 短路 |
| `test_fetcher.py` | 缓存 + 增量更新 + `validate_ohlcv` + source-change marker |
| `test_cli_fetch_universe.py` | `fetch-universe` 默认按 source、`--source` 覆盖、source 变更 force_refresh |
| `test_indicators.py` | 数值正确性 |
| `test_signals.py` | 信号触发条件 |
| `test_factors.py` | 因子注册表 + 后缀参数解析 + 无 look-ahead + 数值 |
| `test_ml_pipeline.py` | Lasso 选稀疏 + IC/IR/equal weighter + TwoStepPipeline |
| `test_ml_selector_lightgbm.py` | LightGBMSelector:非线性选 / top_k / min_importance_ratio / 确定性 / 退化 / 集成 |
| `test_ml_weighter_lightgbm.py` | LightGBMWeighter:fit→predict / SHAP weights / 行和接近 predict / 确定性 / 退化 / 集成 |
| `test_ml_strategy.py` | MLFactorStrategy walk-forward、per_stock/pooled、引擎集成 |
| `test_ops.py` | WQ 算子库:时间序列/横截面/indneutralize/look-ahead + `correlation` 平盘日 ±inf→NaN / clip [-1,1] |
| `test_ml_dataset_finite.py` | 训练矩阵非有限值防线:`stack_panel_to_xy` / `align_xy` 对 ±inf 样本行(因子侧与标签侧)整行剔除 |
| `test_wq101.py` | 101 alpha 注册 + 元数据 + 计算无异常 + look-ahead 截断不变 |
| `test_panel.py` | Panel 构造 + 截尾 + 缺失/错位对齐 |
| `test_ml_strategy_panel.py` | factor_panel 注入 + with_stock 传播 + cross-sec 不退化 |
| `test_ml_strategy_panel_fit_reuse.py` | 注入 close_panel 后 `_try_fit` 快路径、快/慢 (X,y) 等价、PR-3 pre-stack cache bitwise 等价 |
| `test_factor_panel_cache.py` | `load_or_build_factor_panel` 落盘缓存 + preprocess sig 隔离 + `build_log_mcap_panel` 静态广播 + `maybe_inject_mcap_panel` |
| `test_config.py` | Pydantic 校验(含 strategy 段) |
| `test_report_smoke.py` | 全链路 `cmd_run` 烟雾 |
| `test_industry_map.py` | baostock + akshare 双源 mock,auto-fallback 链,缓存/过期/failure-isolation |
| `test_ipo_dates.py` | baostock mock,cache hit/stale/force_refresh/failure-fallback/NaT 过滤 |
| `test_recommend_pool.py` | Pool B 漏斗 + ISO 周缓存 + content_hash 失效 + 失败隔离 |
| `test_factors_analysis.py` | FactorAnalysisResult / compute_daily_ic / classify_regimes / half-life / analyze_factors / pick_top_factors |
| `test_factors_analysis_report.py` | HTML 渲染烟雾 + 空 regime |
| `test_cli_factors_analyze.py` | `factors analyze` 与 `factors pick-by-ic` CLI 烟雾 |
| `test_ml_dataset_labels.py` | forward_return / forward_return_panel 的 label_type 接口 |
| `test_ml_strategy_embargo.py` | walk-forward embargo:默认 auto=horizon,explicit 0 恢复旧行为,泄露 bar 被排除 |
| `test_sizing.py` | FixedLotSizer / VolTargetLotSizer 数学 + fallback + build_lot_sizer 工厂 |
| `test_portfolio_strategy.py` | `PrecomputedScoreStrategy` 语义 |
| `test_portfolio_scoring.py` | `precompute_scores_from_legacy`:happy / 失败隔离 / 全失败 / 缺 score_field / 不截断 history |
| `test_portfolio_engine.py` | PortfolioEngine:空面板 / 恒价等权不变 / 现金守恒 / T+1 fill / start_offset / 确定性 / rebalance diff / 末 bar 不执行 / 未知 code 过滤 / initial_cash |
| `test_portfolio_eligibility.py` | EligibilityFilter:min_history_bars / 流动性边界 / ST / date 截断 / 缺 volume / 阈值 0 跳过 |
| `test_portfolio_industry_cap.py` | `_select_top_k` 行业 cap 贪心 / cap=None / sector_map 空 / 全 unknown / 部分 unknown / engine 集成 |
| `test_cli_portfolio_backtest.py` | `portfolio-backtest`:smoke + `enabled=false` 退 2 + `--refresh-scores` + universe 自动扩 + `staggered_starts=3` ensemble |
| `test_portfolio_ensemble.py` | `StaggeredRunner`:N=1 等价 / rebalance bar disjoint / ensemble 均值数学 / envelope 列序 / aggregated_metrics / n<1 raises |
| `test_portfolio_report_ensemble.py` | `render_ensemble_report` HTML smoke + 空 ensemble 占位 |
| `test_portfolio_ab_config.py` | PortfolioABConfig:arms!=2 / extra=forbid / build_effective_cfg / 重算 content_hash |
| `test_portfolio_ab_runner.py` | run_portfolio_ab happy + per-arm failure isolation + empty-failed |
| `test_portfolio_ab_report.py` | render_portfolio_ab_report HTML smoke + 失败 arm 红 banner + 0 arm 占位 |
| `test_cli_portfolio_ab.py` | `portfolio-ab`:happy / unknown arm 退 2 / `--arm` 只 stdout / base `enabled=false` 退 2 |
| `test_panel_mask.py` | `_limit_threshold` 板块映射 + `_listing_mask` + `compute_tradability_mask` + `apply_mask` + IPO 日期路径 |
| `test_ops_mask_nan_safe.py` | `ts_mean/sum/std/product` min_periods + `decay_linear` NaN-safe 重归一化 |
| `test_ml_strategy_mask.py` | 各层 mask 参数语义 + sig 变化 + pooled/per_stock spy |
| `test_ml_preprocess.py` | 4 函数 + `apply_preprocess_pipeline`(mcap 残差化 + fundamental 跳过 + missing-panel warning + orthogonalize-last + size-guard)+ `_is_all_off` short-circuit + cs_zscore 含 ±inf 行按退化日归零 |
| `test_ml_preprocess_orthogonalize.py` | 对称(Löwdin)正交化 9 case:Gram 正交 / order-independent / 接近原因子 / 退化 passthrough / NaN 传播 / fundamental 跳过 / 单因子 / 近奇异 floor / 不 mutate |
| `test_factors_original_stats.py` | rolling 直接统计因子注册 + 数值 + look-ahead |
| `test_factors_ewma.py` | EWMA halflife 解析 + 公式对照 |
| `test_factors_vwap_deviation.py` | VWAP 偏离族注册 + 单调性 + 无 look-ahead |
| `test_factors_close_position.py` | 收盘位置 ∈ [0,1]、涨停封板 range=0 NaN 守护 |
| `test_factors_turnover_extra.py` | 短窗换手族、停牌日 volume=0 NaN 守护 |
| `test_factors_acceleration.py` | 二阶差分公式对照 + 无 look-ahead |
| `test_factors_single_stock_vol.py` | ATR/CCI/振幅/Parkinson 正性 + 无 look-ahead |
| `test_factors_composite.py` | 复合算子拼装注册 + 无 look-ahead |
| `test_factors_rank_correlation.py` | 秩相关 ∈ [-1,1] + 无 look-ahead |
| `test_factors_cross_sec_breadth.py` | 全市场标量广播 + 涨停股算入宽度分子 |
| `test_factors_fundamentals.py` | PIT:pubDate 之前 NaN、之后 ffill、亏损 PE NaN |
| `test_fundamentals_loader.py` | baostock mock + cache hit/stale/force_refresh/failure-fallback |
| `test_cli_refresh_fundamentals.py` | `--refresh-fundamentals` argparse wiring |
