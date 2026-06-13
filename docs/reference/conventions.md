# 引擎约定 / 数据流 / 缓存 (重要)

## 回测框架抽象 (`stockpool.backtesting`)

完整 API 见 [../backtesting_framework.md](../backtesting_framework.md)。

- `Strategy` ABC — 子类实现 `name` / `generate_signals` / `should_enter` / `should_exit`;可选 `should_reset_timer`、`predict_latest(daily_df) -> dict`(日报路径,返回末根 bar `{signal, final_score, ...}`)。`generate_signals` 输出至少含 `date / open / close / signal`(`open` 用于次日开盘成交;省略时引擎按 `close.shift(1)` 兜底)。**同时输出 `score` 和 `final_score`**(数值相同),便于 portfolio `precompute_scores_from_legacy` 对接。
- `BacktestEngine` — 单仓位、long-only、T+1、单笔进出
- `MultiLotBacktestEngine` — 每个 buy 开独立 lot(仓位由 `LotSizer` 决定),各自计 N、各自记账
- `BacktestResult` — `signals` / `curve` / `trades` / `metrics`(全程)/ `metrics_active`(活跃段:从第一笔 trade 的 `entry_idx` 起切片重算,剔除 ml_factor 冷启动平头;无交易时 `None`)/ `max_holding_days` / `strategy_name`
- `TradeCosts(buy_cost, sell_cost)` — 比例(`0.001` = 0.1%)
- `Trade.ret` — 分母为**买入前**资金(单仓:`equity[t-1]`;多 lot:下单金额 `size`),双边成本都净掉
- `compute_metrics(eq, trades, rf, active_from_idx=None)` 纯函数 — total/ann return、max DD、Sharpe、win rate、avg trade ret;边界口径:有效天数 <60 → `annualized_return=None`,<20 → `sharpe=None`,无交易 → `win_rate`/`avg_trade_return_pct=None`(展示层显示 `—`)
- `buy_and_hold_baseline` — 不扣手续费的全仓基准;曲线与 `total_return` 都锚 `open[0]`(含 day-0 日内收益)

**内置策略**(`stockpool.backtesting.strategies`):`CompositeVerdictStrategy`(主策略,综合评级 + 日周共振)/ `MLFactorStrategy`(两步法,walk-forward 重训,支持 per_stock 和 pooled)/ `VerdictExecution` / `SMACrossStrategy`。

**verdict-based 策略默认参数**:

| 参数 | 默认 | 行为 |
|---|---|---|
| `buy_verdicts` | `("buy","strong_buy")` | 触发 `should_enter` |
| `sell_verdicts` | `("sell","strong_sell")` | 触发 `should_exit` |
| `refresh_verdicts` | `("strong_buy",)` | 持仓时触发 `should_reset_timer`(刷新 N 天计时);传 `()` 关闭 |

**CLI 默认引擎 `multi_lot`**,`sizing.type` 默认 `fixed`(2026-06-13 改进轮5 翻转:48 股生产口径 fixed 同时赢 return +5.4pp 与 sharpe +0.051;vol_target 对高波动股缩仓,恰好低配策略最赚钱的名字。求更低回撤可切回 `vol_target`)。切回单仓位:`engine: single`。

## T+1 撮合约定

- **T+1 + 次日开盘成交**:信号在第 `t-1` 根 bar 收盘后产生,**在第 `t` 根 bar 的 `open[t]` 成交**(A 股集合竞价价)。除非次日开盘直接涨停打不进单,否则视为可成交。
- **进场当日敞口** = `open[t] → close[t]`,扣 `buy_cost` 后乘 `close[t]/open[t]`;后续持仓日按 `close[t-1] → close[t]` 累计。
- **出场当日敞口** = `close[t-1] → open[t]`,然后扣 `sell_cost`,当日剩余时间空仓。
- **`Trade.entry_idx` / `exit_idx` 指向执行 bar `t`**(非决策 bar `t-1`);`entry_price` / `exit_price` 即 `open[t]`。
- **signal 帧必须带 `open` 列**;若缺,引擎回退到 `open[t]=close[t-1]`,新口径退化为旧 close-to-close 行为,旧测试算术保留。
- **Look-ahead 安全契约**:`generate_signals` 第 `i` 行只能用 `daily_df.iloc[:i+1]`。
- **单仓位不加仓**;多仓位 lot 同信号入场。
- **`should_reset_timer` 胜出**:同时为真时优先于 `time_exit` 与 `should_exit`。
- **B&H 基准不扣手续费**,锚定 `open[0]`:`equity[t] = close[t]/open[0]`。

## Sizing (F3 PR-C 起)

`MultiLotBacktestEngine` 不再硬编码 `position_size`,改由 `LotSizer` 注入:

- `FixedLotSizer(size)` — 老行为,每单恒定 `size` 比例
- `VolTargetLotSizer(baseline, ref_vol, window, min, max, fallback)` — 按个股近 `window` bars 滚动 std 反比调仓:`size = baseline × (ref_vol / recent_vol)`,clip 到 `[min, max]`
  - 冷启动(< window+1 bar)/ NaN / vol=0 → 走 `fallback`:`"fixed"` 退回 baseline,`"skip"` 返 0
  - 公式锚点 `baseline = cfg.backtest.sizing.fixed.size`:fixed 和 vol_target 切换时锚点不变,差异纯来自 vol-adjust
- 工厂 `build_lot_sizer(cfg.backtest.sizing)` 是顶层 wiring(cli / backtest_runner / backtest_composite / strategy_factory / ab/config 全走它)
- `Trade.lot_size` 记录每笔实际仓位,A/B 报告可用其归因

## 数据流

```
{mootdx | baostock | akshare} → fetcher (cache parquet) → indicators (add_all)
       → signals (detect_signals → score → verdict_of) + strategy.predict_latest()
       → report.render_report  /  backtest_composite.simulate_equity_curve
       → HTML
```

### 复权与缓存增量约定(2026-06 P0 修复)

- **全链路统一后复权 (hfq)**:akshare `adjust="hfq"`、baostock `adjustflag="1"`;
  mootdx 原始 bars 不复权,由 `mootdx_backend._apply_hfq` 叠加同源 xdxr 事件
  (TCP,`client.xdxr`)做**段内锚定 hfq**——返回段的段首因子=1,事件因子
  `P_prev/P_ex` 只依赖段内 prev_close;窗口外事件只贡献常数尺度,直接忽略。
  不用 mootdx 自带 `to_adjust`(走新浪 HTTP 且对部分窗口有 fillna(1.0) 边界 bug)。
  volume 各源均不复权。**为何 hfq 而非 qfq**:qfq 锚在最新价,每次除权全历史平移,
  与增量缓存根本不兼容;hfq 历史不变,增量追加自洽。
- **盘中保护**:`_drop_in_progress_bar` 在 15:05 前丢弃 `date==今天` 的行,
  半根盘中 bar 不会写入缓存(股票/指数/板块路径都生效)。
- **增量重叠拉取 + 接缝校验**:`fetch_daily` 增量从缓存最后一天**含**开始拉
  (非 last+1),`_reconcile_increment` 用重叠 bar 校验:close 偏差 >0.1%(复权
  基准漂移)或 volume 偏差 >1%(缓存末根是历史污染的半根 bar)→ 丢弃缓存全量
  重拉(自愈);mootdx 增量段用重叠 bar 锚定到缓存既有价格尺度。合并用
  `drop_duplicates(keep="last")`,新 bar 覆盖旧 bar。
- **绝对价格语义**:hfq 价格 ≠ 真实成交价(被复权因子缩放)。收益率/指标/回测
  PnL 正确;依赖绝对价位的逻辑(如 mcap = close×shares)是近似,见 review P2-22。

**日报路径 verdict 来自 `cfg.strategy.name`**(`cli._analyze_one`):
- 仍计算综合评级 triggers/scores/hit_rates 作展示补充
- 最终 `verdict` / `final_score` 由 `strategy.predict_latest(daily)` 给出
- `composite_verdict`:直接算最后一根 bar
- `ml_factor`:从 `<cache_dir>/ml_models/<sig>_<code>.pkl` 加载已训练 pipeline+quantiles,**每自然月最多重训一次**(同月内 predict-only,跨月自动重训覆写);`<sig>` = 8 位 MLFactorConfig 哈希
- `pooled` 模式下 `cmd_run` 预加载整池 `pool_data` 喂 `build_strategy`,保证 cross-sec 因子有真实横截面
- `cmd_run` 在 per-stock loop 前一次性预算 `sector_context_cache: {sector_name: ContextSignal}`,`_analyze_one` 复用,失败时存错误 str

**顶层各阶段计时**:`cmd_run` 按 `[TIME] setup+config / market_index_context / _prepare_ml_pool / sector_context prefetch / per_stock_loop / pool_b / render_report / TOTAL` 打印 stdout。

**Partial NaN 容忍**(`generate_signals` / `predict_latest`):predict 路径不要求 X 行所有因子非 NaN — 对 NaN 列 fill 0 impute(归一化后 0 是中性值),仅整行 NaN 时返 `signal=neutral, score=NaN`。原因:`selection.json` 含 alpha_037 这种 200 日 rolling correlation 因子,warmup 必然 36% NaN。

## Pool B(全市场量化推荐池)

在 `render_report` 前由 `recommend_pool.compute_or_load_pool_b` 计算:
- 始终用 `load_universe_cache(cfg.data.cache_dir)` 作**应用池**(独立于训练池)
- 漏斗:**流动性**(近 20 日 `vol*close*100` ≥ `min_avg_amount_20d`)→ **ST 二次防御**(name 含 ST)→ 调当前 strategy `predict_latest` 打分 → **行业上限贪心**(score 降序,每行业 ≤ `max_per_industry`,收满 `top_n` 即停)
- 复用 `cli._prepare_ml_pool` 给 strategy 的 `pool_data` / `factor_panel` / **`close_panel`**(必须透传,否则每股重跑 `build_close_panel(4000+ 股)`,每股 ~3s × 4007 → ~3 小时;透传后 build_avg ~3ms)
- 缓存键 `poolb_<content_hash>_<isoyear>w<isoweek>.parquet`;同周 + 同 yaml 读盘
- 失败隔离:per-stock predict 异常 log warning 跳过;Pool B 整体失败不影响 Pool A
- Pool B 内部按 200 股粒度打 `[TIME] Pool B i/total ... build_avg= predict_avg= ETA=` 进度

## 缓存 (`data/`)

| 文件 | 内容 |
|---|---|
| `<code>_daily.parquet` | 股票 K 线 |
| `idx_<symbol>.parquet` | 指数(`stock_zh_index_daily` 全量替换) |
| `sector_<name>.parquet` | 行业板块 |
| `ml_models/<sig>_<code>.pkl` | ml_factor 月度训练缓存(`share_pool_fit=false`) |
| `ml_models/<sig>_shared.pkl` | `share_pool_fit=true` 时所有股共享一份 |
| `universe.parquet` | `fetch-universe` 写入的全 A 股清单 (code/name/market) |
| `stock_industry_map.parquet` | Pool B 的 `code → 行业` 映射(akshare 东财,30 天有效) |
| `ipo_dates.parquet` | baostock `query_stock_basic` IPO 日期(5500+ 行,30 天有效);`mask.enabled=true` 时用 |
| `mcap_shares.parquet` | `scripts/pull_mcap_profit.py` 拉的全市场最新 `totalShare` 快照;`build_log_mcap_panel` 用其算 market_cap_neutralize 面板。无有效期/手动刷新(全量 ~50-75 min) |
| `fundamentals_{profit,growth,balance,cash_flow,dupont}.parquet` | baostock 季度财务长期缓存,30 天有效 |
| `recommend_pool/poolb_<hash>_<isoyear>w<NN>.parquet` | Pool B 本周排名缓存 |
| `factor_panels/<sig>/{manifest.json, close.parquet, <factor>.parquet × N}` | ml_factor pooled 因子面板 + close 宽表落盘缓存;sig hash 含 factors / sorted codes / last_date / preprocess。`--refresh-factor-panel` 旁路 |
| `.data_source` | 单行文本 `<source>:<adjust>`(如 `mootdx:hfq`),记录上次写入该 cache_dir 的数据源与复权模式;`fetch_*` 启动时比对,任一不匹配(含旧格式纯 source 的遗留 marker)触发 force_refresh 全量重拉 |

## 报告路径

- 日报:`reports/<YYYY-MM-DD>.html` + `reports/latest.html`
- 回测:`reports/backtest/<YYYY-MM-DD>.html` + `reports/backtest/latest.html`
- Portfolio:`reports/portfolio/<date>.html` + `latest.html`
- Portfolio AB:`reports/portfolio_ab/<date>.html` + `latest.html`
- A/B:`reports/ab/<date>.html` + `latest.html`
