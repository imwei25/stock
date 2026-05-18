# stockpool — Claude Code 项目指南

> 本文为后续 Claude Code 会话的快速上手指南。**不需要通读整个项目**,按需查阅对应模块即可。
> 改动项目时请遵守文末的["更新本文件"](#改动后更新本文件)规则。

## 项目一句话

A 股技术信号分析工具:AKShare 拉数据 → 计算多指标 → 综合评级 → 输出 HTML 日报和走样式回测报告。

## 快速命令

```bash
# 日报 (拉数据 → 分析 → 渲染 HTML 到 reports/)
python -m stockpool run --config config.yaml

# 回测 (走样式综合策略 + 多 N 净值曲线)
python -m stockpool backtest --config config.yaml [--stocks 605589]

# 测试
python -m pytest tests/ -q
```

## 模块地图

| 文件 | 职责 |
|---|---|
| `src/stockpool/cli.py` | 入口,定义 `run` 和 `backtest` 子命令 |
| `src/stockpool/config.py` | Pydantic schema + YAML 加载;**配置变更必须更新这里** |
| `src/stockpool/fetcher.py` | AKShare 拉股票/指数/板块 + Parquet 缓存 + OHLCV 校验 |
| `src/stockpool/indicators.py` | MA / MACD / KDJ / RSI / BOLL / Volume / Breakout |
| `src/stockpool/signals.py` | `detect_signals` + `score_triggers` + `combine_daily_weekly` + `verdict_of` |
| `src/stockpool/factors/` | **连续因子库**(Factor ABC + 注册表 + 内置技术因子) |
| `src/stockpool/ml/` | **两步法 ML 组合**(dataset / Lasso selector / IC&IR weighter / TwoStepPipeline) |
| `src/stockpool/strategy_factory.py` | 按 `cfg.strategy.name` 工厂构造策略 + ML 通用 simulate |
| `src/stockpool/report.py` | 日报 HTML(含市场背景、板块上下文) |
| `src/stockpool/backtest.py` | 单信号前瞻命中率 |
| `src/stockpool/backtesting/` | **回测框架**(策略 ABC + 引擎),见下 |
| `src/stockpool/backtest_composite.py` | 综合策略回测的旧 API 适配层,委托给框架 |
| `src/stockpool/backtest_report.py` | 回测 HTML 渲染 |

## 回测框架 (`stockpool.backtesting`)

**核心抽象**:

- `Strategy` ABC — 子类必须实现 `name` / `generate_signals` / `should_enter` / `should_exit`;
  可选 `should_reset_timer`(返回 True 重置 N 天计时器)。
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

- **T+1**:决策只读 `signal[t-1]`,成交价 `close[t]`;末根 bar 信号无操作。
- **Look-ahead 安全契约**:`generate_signals` 第 `i` 行只能用 `daily_df.iloc[:i+1]`。
- **单仓位不加仓**;多仓位 lot 同信号入场。
- **`should_reset_timer` 胜出**:同时为真时优先于 `time_exit` 与 `should_exit`。
- **B&H 基准不扣手续费**(让策略对比更保守)。

## 配置 (`config.yaml`)

所有字段由 `config.py:AppConfig` 校验。结构概览:

- `stocks` — 股票池,每条含 `code` / `name` / `sector`
- `data` — `history_days` / `cache_dir` / `force_refresh`
- `indicators` — MA 周期、MACD/KDJ/BOLL 参数
- `weights` — 各信号触发的得分
- `scoring` — `daily_weight` / `weekly_weight` / `resonance_bonus`(共振奖励)
- `verdicts` — `strong_buy` / `buy` / `sell` / `strong_sell` 分数阈值
- `backtest` — `forward_days` / `equity_curve_holding_days` / `risk_free_rate` / `costs` / **`engine`** / **`position_size`** / **`max_concurrent_lots`**
- `context` — `indices`(大盘指数列表,默认上证/深证成指)
- `report` — `output_dir` / `keep_history` / `klines_to_show`
- **`strategy`** — `name` (`composite_verdict` 默认 / `ml_factor`) + `ml_factor` 子配置(`factors` / `horizon` / `train_window` / `refit_every` / `panel_mode` / `selector` / `weighter` / `thresholds` / `*_verdicts`)

## 数据流

```
AKShare → fetcher (cache parquet) → indicators (add_all)
       → signals (detect_signals → score → verdict_of)
       → report.render_report  /  backtest_composite.simulate_equity_curve
       → HTML
```

缓存(`data/`):
- `<code>_daily.parquet` — 股票
- `idx_<symbol>.parquet` — 指数(`stock_zh_index_daily` 全量替换)
- `sector_<name>.parquet` — 行业板块

报告:
- 日报:`reports/<YYYY-MM-DD>.html` + `reports/latest.html`
- 回测:`reports/backtest/<YYYY-MM-DD>.html` + `reports/backtest/latest.html`

## 测试

152 个,`pytest tests/ -q` 一次跑完。按域分布:

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
| `test_ml_strategy.py` | MLFactorStrategy walk-forward、per_stock/pooled、引擎集成 |
| `test_config.py` | Pydantic 校验(含 `strategy` 段) |
| `test_report_smoke.py` | 全链路 `cmd_run` 烟雾 |

写测试时:**用合成 OHLCV、`monkeypatch` 掉 AKShare 和 `_today`**(`test_cli_backtest.py` 是参考)。

## 已知不支持的能力

- 做空、多标的组合、盘中数据、部分成交、资金成本(融资融券)
- 仓位管理仅"满仓单笔"或"固定额度多笔"两种;无 Kelly / 比例追加

## 改动后更新本文件

**只要"后续 Claude 不读源码也能正确帮用户做事"这一原则受到威胁,就要同步本文件。**

具体触发场景:
- 新增 / 删除 / 重命名顶层模块或公开 API
- `Strategy` ABC、`BacktestEngine` / `MultiLotBacktestEngine` 的公开签名变化
- `config.py` 中 schema 字段新增 / 删除 / 默认值改变 / 语义改变
- CLI 子命令的增减、参数变化
- 默认行为(尤其是 `engine`、`refresh_verdicts`、`position_size`)的切换
- 测试目录新增的"按域覆盖"文件
- 数据流 / 缓存路径 / 报告路径变化

**不需要每个 commit 都更新**。只在上述变化发生时跟一次,把对应小节改掉即可。
