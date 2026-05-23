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
python -m stockpool backtest --config config.yaml [--stocks 605589]

# 拉全市场 A 股缓存 (训练用,剔除 ST/科创/北交; mootdx 8 线程, 全量 ~1 分钟)
python -m stockpool fetch-universe [--workers 8] [--limit 100] [--refresh]

# 测试
python -m pytest tests/ -q

# 因子管理 (浏览 / 筛选 / 选择)
python -m stockpool factors list                          # 列全部 (~111 个)
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
```

## 模块地图

| 文件 | 职责 |
|---|---|
| `src/stockpool/cli.py` | 入口,定义 `run` 和 `backtest` 子命令 |
| `src/stockpool/config.py` | Pydantic schema + YAML 加载;**配置变更必须更新这里** |
| `src/stockpool/fetcher.py` | 公开 API + Parquet 缓存 + OHLCV 校验;按 `cfg.data.source` 派发后端;`fetch_universe`/`list_universe`/`load_universe_cache` 提供全市场批拉与读盘 |
| `src/stockpool/data_sources/mootdx_backend.py` | 通达信 TCP 后端(默认)。股票/指数/**行业板块(88xxxx)**;含当日盘中,TDX 占位 bar 会被丢弃 |
| `src/stockpool/data_sources/baostock_backend.py` | baostock 后端(无 token,收盘后约 18:00 更新);不支持板块,板块会自动走 mootdx |
| `src/stockpool/indicators.py` | MA / MACD / KDJ / RSI / BOLL / Volume / Breakout |
| `src/stockpool/signals.py` | `detect_signals` + `score_triggers` + `combine_daily_weekly` + `verdict_of` |
| `src/stockpool/factors/` | **连续因子库**(Factor ABC + 注册表 + 内置技术因子 + **WQ101**) |
| `src/stockpool/factors/ops.py` | **WQ 算子库** (ts_rank / rank / decay_linear / indneutralize / ...) |
| `src/stockpool/factors/wq101.py` | **WorldQuant 101 Formulaic Alphas** 全套实现 (Alpha001..Alpha101) |
| `src/stockpool/factors_picker.py` | **HTML 因子选择器** + `factors` CLI 子命令 |
| `src/stockpool/industry_map.py` | `code → 行业` 映射;多源(`auto` / `baostock` / `akshare`);缓存到 `data/stock_industry_map.parquet`,>30 天过期自动重拉。**mootdx 路径无效**:TDX 服务器对 `block_hy.dat` 返回 0 字节 |
| `src/stockpool/recommend_pool.py` | **Pool B**(全市场量化推荐池):漏斗 + 排序 + 周缓存;`compute_or_load_pool_b` 顶层 API |
| `src/stockpool/factors_analysis.py` | **因子分析**: 滚动 IC / IR / half-life / 相关性 / regime 切片;`analyze_factors` + `pick_top_factors` |
| `src/stockpool/factors_analysis_report.py` | pyecharts HTML 报告: 排名表 + IC 时序 + 相关性 heatmap + regime 拆分 |
| `src/stockpool/panel.py` | **Panel** 数据结构 (T×N 宽表 dict) + `build_panel_from_cache` |
| `src/stockpool/ml/` | **两步法 ML 组合**(dataset / Lasso 或 LightGBM selector / IC&IR&Equal&LightGBM weighter / TwoStepPipeline) |
| `src/stockpool/strategy_factory.py` | 按 `cfg.strategy.name` 工厂构造策略 + ML 通用 simulate;ml_factor 注入 `cache_dir` 以启用日报路径的月度训练缓存;`build_factor_panel` 顶层助手用于 CLI 预算 |
| `src/stockpool/report.py` | 日报 HTML(含市场背景、板块上下文);`_optimize_html` 做 echarts lib 去重 + `<details>` 默认折叠 + 图表懒加载,降低首屏开销 |
| `src/stockpool/backtest.py` | 单信号前瞻命中率 |
| `src/stockpool/backtesting/` | **回测框架**(策略 ABC + 引擎),见下 |
| `src/stockpool/backtest_composite.py` | 综合策略回测的旧 API 适配层,委托给框架 |
| `src/stockpool/backtest_report.py` | 回测 HTML 渲染 |

## 回测框架 (`stockpool.backtesting`)

**核心抽象**:

- `Strategy` ABC — 子类必须实现 `name` / `generate_signals` / `should_enter` / `should_exit`;
  `generate_signals` 输出至少含 `date / open / close / signal`(`open` 用于次日开盘成交;省略时引擎按 `close.shift(1)` 兜底);
  可选 `should_reset_timer`(返回 True 重置 N 天计时器)、`predict_latest(daily_df) -> dict`(给日报路径用,返回最后一根 bar 的 `{signal, final_score, ...}`;默认实现是跑全程 `generate_signals` 取末行,子类可重写做单点优化或加缓存)。
- `BacktestEngine` — 单仓位、long-only、T+1、单笔进出。
- `MultiLotBacktestEngine` — 每个 buy 开一个独立的 `position_size` 大小 lot,各自计 N、各自记账。
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

**CLI 默认引擎是 `multi_lot`**(见 `config.yaml:backtest.engine`),`position_size=0.1`。
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

## 配置 (`config.yaml`)

所有字段由 `config.py:AppConfig` 校验。结构概览:

- `stocks` — 股票池,每条含 `code` / `name` / `sector`
- `data` — `history_days` / `cache_dir` / `force_refresh` / **`source`** (`mootdx` 默认 / `baostock` / `akshare`)。注意:(1) 切换 source 时建议 `force_refresh: true`,不同后端 volume 计量单位不同(mootdx=手, baostock=股);(2) 行业板块仅在 `source=akshare` 时走东财,其他两种 source 都用 **mootdx 的通达信行业指数 (88xxxx)**;名字→代码映射见 `mootdx_backend._TDX_INDUSTRY_CODES`,也可在 `stocks[].sector` 直接填 6 位 TDX 代码
- `indicators` — MA 周期、MACD/KDJ/BOLL 参数
- `weights` — 各信号触发的得分
- `scoring` — `daily_weight` / `weekly_weight` / `resonance_bonus`(共振奖励)
- `verdicts` — `strong_buy` / `buy` / `sell` / `strong_sell` 分数阈值
- `backtest` — `forward_days` / `equity_curve_holding_days` / `risk_free_rate` / `costs` / **`engine`** / **`position_size`** / **`max_concurrent_lots`**
- `context` — `indices`(大盘指数列表,默认上证/深证成指)
- `report` — `output_dir` / `keep_history` / `klines_to_show`
- **`strategy`** — `name` (`composite_verdict` 默认 / `ml_factor`) + `ml_factor` 子配置(`factors` 或 **`factors_file`** / `horizon` / `train_window` / `refit_every` / `panel_mode` / **`training_universe`** / **`share_pool_fit`** / **`embargo_days`** / **`label_type`** / `selector.{lasso|lightgbm}` / `weighter` / `thresholds` / `*_verdicts`)。`factors_file` 指向 HTML picker 导出的 JSON,与 `factors` 列表二选一。**`training_universe`**: `pool`(默认,只用 cfg.stocks)/ `all`(全市场 cache,需先 `fetch-universe`;仅在 `panel_mode=pooled` 时生效)。**`share_pool_fit`**(默认 `true`,仅 `panel_mode=pooled` 生效):跨股共享 fit,缓存键 `(sig, year, month)`,同月内所有股、所有 refit_bar 复用同一 pipeline;训练集不再剔除 host,host 自己以 ~1/N 权重进入自己的训练。**`embargo_days`**(默认 `null` = auto = `horizon`,F2 PR-A 新增):walk-forward 训练集与测试集之间的额外间隔,消除 horizon 日前向收益的标签泄露;设 `0` 回到 pre-PR-A 行为。**`label_type`**(默认 `"return"`,F2 PR-A 接口位):训练标签变换 — `"return"` 已实装,`"vol_adjusted"` / `"cross_sec_rank"` 是占位 raise `NotImplementedError`,后续 PR 实装。**`selector.{lasso|lightgbm}`**(F2 PR-A 子段化 + PR-B1 加 LGB):`type` 默认 `"lightgbm"`,`lasso.{alpha,max_iter,tol}` 或 `lightgbm.{num_leaves,min_data_in_leaf,learning_rate,num_iterations,max_depth,random_state,top_k_factors,min_importance_ratio}` 子段二选一,顶层扁平字段被 Pydantic 拒绝。改 `selector` 任一字段后旧 ml_models pkl 自动失效。切到 `all` 或翻 `share_pool_fit`、改 `embargo_days` / `label_type` / `selector` 任一项后旧的 ml_models pkl 会因 sig 变化自动失效。**`weighter.{ic|ir|equal|lightgbm}`**(F2 PR-B2 子段化):`type` 默认 `"lightgbm"`,`ic.{use_rank,min_abs_ic}` / `ir.{n_chunks,use_rank,min_abs_ir}` / `equal` (无参) / `lightgbm.{num_leaves,min_data_in_leaf,learning_rate,num_iterations,max_depth,random_state}` 子段四选一,顶层扁平字段被 Pydantic 拒绝。
- **`recommend_pool`** — Pool B(全市场量化推荐池)。`enabled`(默认 `true`)/ `top_n`(30)/ `min_avg_amount_20d`(5e7 元;mootdx `vol*close*100`)/ `max_per_industry`(5;"未知" 桶在**所有股都未映射时**自动跳过 cap,否则正常计)/ `refresh`(`weekly`默认/`always`/`never`)/ `cache_dir`(`data/recommend_pool`)/ `industry_map_max_age_days`(30)/ **`industry_source`**(`auto` 默认 = baostock→akshare 链 / `baostock` / `akshare`)。**前置条件**:必须先跑 `python -m stockpool fetch-universe`;首次运行自动从所选 industry_source 拉映射(baostock ~5-10s,akshare ~1-2min)。**缓存键**含 `cfg.content_hash`,改 yaml 任一字段都失效

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

**Pool B(全市场量化推荐池)**在 `render_report` 之前由 `recommend_pool.compute_or_load_pool_b` 计算:
- 始终用 `load_universe_cache(cfg.data.cache_dir)` 作为**应用池**(独立于 strategy 训练池)
- 漏斗:**流动性**(近 20 日 `vol*close*100` ≥ `min_avg_amount_20d`) → **ST 二次防御**(name 含 ST) → 调当前 strategy 的 `predict_latest` 打分 → **行业上限贪心**(score 降序扫描,每行业 ≤ `max_per_industry`,收满 `top_n` 即停)
- 复用 `cli._prepare_ml_pool` 给 strategy 的 `pool_data` / `factor_panel`(ml_factor + training_universe=all 时跨 4000 票 cross-sec 真实横截面),不重复加载
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
- `recommend_pool/poolb_<content_hash>_<isoyear>w<NN>.parquet` — Pool B 本周排名缓存

报告:
- 日报:`reports/<YYYY-MM-DD>.html` + `reports/latest.html`
- 回测:`reports/backtest/<YYYY-MM-DD>.html` + `reports/backtest/latest.html`

## 测试

233 个,`pytest tests/ -q` 一次跑完。按域分布:

| 文件 | 覆盖 |
|---|---|
| `test_backtesting_framework.py` | 引擎契约、T+1、成本、扫 N、Strategy ABC |
| `test_multi_lot_engine.py` | 多仓位 lot 独立计时、现金约束、reset hook |
| `test_timer_reset.py` | strong_buy 刷新计时;reset 与 exit 同时为真时 reset 胜出 |
| `test_backtest_composite.py` | 适配层、综合策略 walk-forward 等价性 |
| `test_backtest.py` | 单信号命中率 |
| `test_cli_backtest.py` | CLI 烟雾测试 |
| `test_fetcher.py` | 缓存 + 增量更新 + `validate_ohlcv` |
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
| `test_config.py` | Pydantic 校验(含 `strategy` 段) |
| `test_report_smoke.py` | 全链路 `cmd_run` 烟雾 |
| `test_industry_map.py` | baostock + akshare 双源 mock,auto-fallback 链,parquet 缓存 / 过期 / failure-isolation |
| `test_recommend_pool.py` | Pool B 漏斗(流动性/ST/行业上限)+ ISO 周缓存 + content_hash 失效 + 失败隔离 |
| `test_factors_analysis.py` | FactorAnalysisResult / compute_daily_ic / classify_regimes / half-life / analyze_factors / pick_top_factors |
| `test_factors_analysis_report.py` | HTML 渲染烟雾 + 空 regime 处理 |
| `test_cli_factors_analyze.py` | `factors analyze` 与 `factors pick-by-ic` CLI 烟雾 |
| `test_ml_dataset_labels.py` | forward_return / forward_return_panel 的 label_type 接口(只 "return" 已实装) |
| `test_ml_strategy_embargo.py` | walk-forward embargo: 默认 auto=horizon,explicit 0 恢复旧行为,泄露 bar 被排除 |

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
           `time_series` / `cross_sectional` / `industry_neutral`

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
- 默认行为(尤其是 `engine`、`refresh_verdicts`、`position_size`、数据源行业板块路由)的切换
- 测试目录新增的"按域覆盖"文件 → CLAUDE.md 的测试表加一行
- 数据流 / 缓存路径 / 报告路径 / 数据源 行为变化
- 新增的因子来源/类型标签(影响 HTML picker 和 `factors list --source/--type` 用例)

### 怎么改

- CLAUDE.md:定位对应小节(模块地图 / 配置 / 测试 / 因子库 / 数据流 / 已知不支持),把改动点写进去
- README.md:更新 **快速开始** / **常用命令** 段;如果新增了用户级流程(如 `factors pick` HTML 选择器),加一段端到端示例
- 两边的命令示例要保持一致(同一份 `python -m stockpool ...` 命令不要在两份文档里写不同的参数)

**不需要每个 commit 都更新**。只在上述变化发生时跟一次,把对应小节改掉即可。
但**只更新 CLAUDE.md 不更新 README,或反过来,算未完成**——改动评审时按"两份都到位"判定。
