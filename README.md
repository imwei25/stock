# stockpool — A 股技术信号 + 量化因子分析工具

每日扫描 A 股池,计算技术信号综合评分或两步法 ML 因子组合,结合大盘指数与行业板块环境,
产出交互式 HTML 日报、回测报告、A/B 对比和组合回测。

## 项目架构

```
数据源 (mootdx 默认 / baostock / akshare)  →  fetcher (parquet 缓存)
   ├─ 技术指标 (MA/MACD/KDJ/RSI/BOLL...)  →  信号综合评级
   └─ 连续因子库 (~165 个: WQ101 全集 + 内置技术 + 基本面)  →  两步法 ML (选因子 → 加权)
                                  │
                          策略 (综合评级 / ML 因子)
                                  │
   ┌──────────────┬──────────────┼───────────────┬────────────────┐
 日报 HTML      单股回测       组合回测          A/B 对比      全市场推荐池
 (Pool A+B)   (T+1 撮合)   (top-K 等权)    (两策略 PK)    (Pool B)
```

两条主路径:`run`(日报,每股最新评级)和 `backtest`/`portfolio-backtest`(历史回测)。

## 个股信号产生流程 · 各环节可配置方法

每只票的信号经过一条流水线产生,流水线的形态由 `strategy.name` 决定(综合评级 / ML 因子)。
下面按环节列出每一步**支持配置的方法**;字段全集见 [docs/reference/config.md](docs/reference/config.md)。

### 环节 0 — 数据获取与处理(两条路径共用)

| 环节 | 配置项 | 可选方法 / 取值 |
|---|---|---|
| 数据后端 | `data.source` | `mootdx`(默认,含盘中)/ `baostock`(盘后)/ `akshare`(东财兜底)。切换自动 force_refresh |
| 拉取窗口 | `data.history_days` / `data.force_refresh` | 历史天数 / 强制忽略缓存重拉 |
| 可交易性 mask | `strategy.ml_factor.mask.enabled` | `false`(默认)/ `true`。开启后涨停/停牌/新股头 N 天在训练标签层被剔除(不破坏因子输入)。阈值 `limit_up_threshold_{main,chinext,bse}` / `min_listing_days` 可调 |

### 路径 A — 综合评级策略(`strategy.name: composite_verdict`,默认)

```
指标计算 → 16 种信号触发 → 日/周线打分 → 加权合并(+共振)→ 五档判定
```

| 环节 | 配置段 | 可配置的方法 / 参数 |
|---|---|---|
| ① 指标计算 | `indicators` | `ma_periods`(均线周期)、`macd.{fast,slow,signal}`、`kdj.{n,m1,m2}`、`rsi_periods`、`boll.{n,k}`、`volume_ratio_window`、`breakout_window` |
| ② 信号触发打分 | `weights` | 16 种触发各自的分值:MA(`ma_cross_strong`/`ma_alignment`)、MACD(3 项)、KDJ(3 项)、RSI(2 项)、BOLL(2 项)、Volume(2 项)、Breakout(2 项)。单边分数 cap 到 [-10, +10] |
| ③ 日/周合并 | `scoring` | `daily_weight` / `weekly_weight`(默认 7:3)、`resonance_bonus`(共振奖励)、`resonance_daily_threshold` / `resonance_weekly_threshold`(触发共振的阈值) |
| ④ 五档判定 | `verdicts` | `strong_buy` / `buy` / `sell` / `strong_sell` 四个终分阈值 |

### 路径 B — ML 因子策略(`strategy.name: ml_factor`)

```
选因子 → 截面预处理流水线 → 选择器筛因子 → 加权器合成分 → 分位数映射五档
                     (walk-forward 周期重训)
```

| 环节 | 配置段 | 可配置的方法 / 取值 |
|---|---|---|
| ① 因子集 | `ml_factor.factors` 或 `factors_file` | 手列因子名,或引用 `factors pick` / `pick-by-ic` 导出的 selection.json(~165 个 base 因子可选) |
| ② 面板组织 | `ml_factor.panel_mode` / `training_universe` | `per_stock` / `pooled`(cross-sec 因子必须 pooled);训练池 `pool`(仅 cfg.stocks)/ `all`(全市场 ~4350 票) |
| ③ 截面预处理 | `ml_factor.preprocess` | `winsorize`(去极值,默认开)、`zscore`(标准化,默认开)、`market_cap_neutralize`(对 log 市值 OLS 残差化,默认开)、`industry_neutralize`(行业内 demean,默认关)、`symmetric_orthogonalize`(Löwdin 正交化去相关,默认关)、`min_pool_size`(小池子跳过保护)|
| ④ 选择器 (step-1) | `ml_factor.selector.type` | `lasso`(默认,`lasso.{alpha,max_iter,tol}`)/ `lightgbm`(`lightgbm.{num_leaves,min_data_in_leaf,learning_rate,num_iterations,max_depth,top_k_factors,min_importance_ratio}`) |
| ⑤ 加权器 (step-2) | `ml_factor.weighter.type` | `ic`(默认,`ic.{use_rank,min_abs_ic}`)/ `ir`(`ir.{n_chunks,use_rank,min_abs_ir}`)/ `equal`(等权,无参)/ `lightgbm`(`lightgbm.{...}`) |
| ⑥ 标签与重训 | `ml_factor.{horizon,train_window,refit_every,embargo_days,label_type}` | 前向收益天数 / 训练窗口 / 重训周期 / walk-forward embargo(默认 = horizon)/ 标签变换(目前仅 `return`) |
| ⑦ 分位数判定 | `ml_factor.thresholds` | `strong_buy` / `buy` / `sell` / `strong_sell` 四个分位点(0~1,训练集分位数映射) |

> **默认值已经过 A/B 定标**:`selector=lasso` + `weighter=ic`(LGB 在小训练集过拟合,已从默认回退);
> 预处理 winsorize+zscore+market_cap_neutralize 默认开。结论见 `docs/ab_validation_results.md`。

### 环节 9 — 进出场与仓位(回测路径)

| 环节 | 配置段 | 可配置的方法 / 取值 |
|---|---|---|
| 进出场判定 | `*_verdicts`(两策略各有) | `buy_verdicts` / `sell_verdicts` / `refresh_verdicts`(持仓刷新计时的判定集) |
| 撮合引擎 | `backtest.engine` | `multi_lot`(默认,每信号开独立 lot)/ `single`(单仓位)。统一 T+1 次日 `open[t]` 成交 |
| 仓位 sizing | `backtest.sizing.type` | `fixed`(默认,固定 `fixed.size`;2026-06-13 A/B 翻转)/ `vol_target`(按波动反比调仓,降回撤用,`reference_vol_annual`/`vol_window`/`min_size`/`max_size`/`fallback_to`)|
| 交易成本 | `backtest.costs` | `commission_rate` / `stamp_duty_rate` / `slippage_rate` |

> 日报路径(`run`)只取每股最后一根 bar 的判定(`predict_latest`),不涉及进出场/仓位 —— 那部分仅回测生效。

## 快速开始

```bash
# 1. 安装(需要 Python 3.10+)
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

# 2. 编辑股池(可选)
notepad config.yaml

# 3. 跑一次 + 看报告
.venv/Scripts/python -m stockpool run
start reports/latest.html
```

## 常用命令

```bash
# 日报
.venv/Scripts/python -m stockpool run                            # 默认全跑
.venv/Scripts/python -m stockpool run --refresh                  # 忽略缓存重拉
.venv/Scripts/python -m stockpool run --stocks 605589,603986     # 只跑两只

# 回测
.venv/Scripts/python -m stockpool backtest                       # 回测所有股票
.venv/Scripts/python -m stockpool backtest --refresh-factor-panel # 重算因子面板缓存

# 全市场缓存(ml_factor 训练池 + Pool B 必需,~4350 票,8 线程 ~1 分钟)
.venv/Scripts/python -m stockpool fetch-universe

# 因子库(~165 base / ~280-320 variants)
.venv/Scripts/python -m stockpool factors list [--source wq101] [--type fundamental]
.venv/Scripts/python -m stockpool factors show alpha_001
.venv/Scripts/python -m stockpool factors pick                   # HTML 选择器
.venv/Scripts/python -m stockpool factors analyze --universe all --output reports/factor_analysis
.venv/Scripts/python -m stockpool factors pick-by-ic --input reports/factor_analysis/<日期>.json --output reports/selection.json --top-n 20

# A/B 对比两策略 → reports/ab/latest.html
.venv/Scripts/python -m stockpool ab --config ab.yaml [--arm <name>]

# 组合回测(需 portfolio_backtest.enabled: true)→ reports/portfolio/latest.html
.venv/Scripts/python -m stockpool portfolio-backtest --config config.yaml
.venv/Scripts/python -m stockpool portfolio-ab --config portfolio_ab.yaml

# 测试
.venv/Scripts/python -m pytest
```

## 核心功能

### 信号评分(综合评级策略)

每个交易日对每只票运行技术信号检测(MA 金/死叉、MACD、KDJ、RSI、布林带、成交量、突破),
日线/周线各出一组,按 7:3 合并为终分,双线共振额外加分。终分映射五档判定:

| 终分 | 判定 | 终分 | 判定 |
|---|---|---|---|
| ≥ +6 | 🟢🟢 强烈买入 | ≤ -3 | 🔴 卖出观察 |
| ≥ +3 | 🟢 买入观察 | ≤ -6 | 🔴🔴 强烈卖出 |
| (-3, +3) | ⚪ 观望 | | |

报告顶部显示大盘指数 + 行业板块环境横栏,帮助判断个股信号是否与市场方向一致。

### ML 因子策略与因子库

`strategy.name: ml_factor` 切到两步法:Lasso 筛因子 → IC/IR/equal 加权 → 分位数映射判定,walk-forward 重训。

因子库 ~165 个 base 因子,双轴元数据组织:
- **来源**:`builtin`(技术因子 + 论文 B 家族 + EWMA)/ `wq101`(WorldQuant 101 Alphas 全套)/ `custom`(含基本面 PE/PB/ROE)
- **类型**:`momentum` / `reversal` / `trend` / `volatility` / `volume` / `time_series` / `cross_sectional` / `industry_neutral` / `fundamental`

用 `factors pick` 起本地 HTML 选择器勾选因子 → 写 `reports/selection.json` → `config.yaml` 引用:

```yaml
strategy:
  name: ml_factor
  ml_factor:
    factors_file: reports/selection.json
    panel_mode: pooled          # WQ101 cross-sec 必须 pooled
    training_universe: all      # 全市场训练(需先 fetch-universe);默认 pool 只用 cfg.stocks
    horizon: 5
```

> **训练池 vs 应用池**:`training_universe: all` 时训练在全市场 ~4350 票上跑(cross-sec 因子见到完整横截面),
> 日报/回测输出仍只针对 `cfg.stocks`。详见 [docs/reference/config.md](docs/reference/config.md)。

基本面因子首次需拉 baostock 财务数据(~30-60 分钟,30 天缓存),任意命令加 `--refresh-fundamentals` 触发。
按 **公告日(pubDate)** 对齐,确保无未来泄露。

### 历史回测

- **单信号命中率**:每种信号出现后 5/10/20 日平均涨幅与胜率
- **综合评级回测**:前向重建历史每日判定,按评级分桶统计收益
- **权益曲线**:模拟持有 N 日资金曲线,对比买入持有基准
- **T+1 撮合**:信号在 `t-1` 收盘后生成,用 `open[t]`(集合竞价价)作买卖成本;B&H 基准锚定 `open[0]`
- **仓位 sizing**:默认 `fixed`(固定每单仓位;2026-06-13 A/B:return/sharpe 双赢);`sizing.type: vol_target` 切按波动反比调仓(降回撤)

### Pool B — 全市场量化推荐池

日报底部多出一段"推荐池":对全市场 A 股调当前策略打分,经"流动性 + ST + 行业上限"漏斗取 top-30。
需先跑 `fetch-universe`。配合 `ml_factor` + `training_universe: all` 最有意义。

```yaml
recommend_pool:
  enabled: true
  top_n: 30
  min_avg_amount_20d: 50000000   # 最低成交额过滤,防次新/小盘刷分
  max_per_industry: 5
  refresh: weekly
```

### A/B 与组合回测

- **`ab`**:同一池/同段历史下并行跑两策略,出净值曲线 + 聚合差值表 + Sharpe 散点。arm 只能覆盖 `strategy:` / `backtest:`。
- **`portfolio-backtest`**:横截面 top-K 等权 + 周期 rebalance + 行业 cap + T+1。`staggered_starts > 1` 出 staggered ensemble + 包络图。
- **`portfolio-ab`**:组合级两策略 PK。

配置示例见 `ab.yaml.example` / `portfolio_ab.yaml.example`,字段说明见 [docs/reference/config.md](docs/reference/config.md)。

## 数据源 (`data.source`)

| 来源 | 特点 | 何时用 |
|---|---|---|
| `mootdx`(默认) | 通达信 TCP;**含当日盘中**(几分钟延迟);无 token | 日常、盘中 |
| `baostock` | 免费无 token;**收盘后约 18:00 更新** | 稳定历史回测、盘后跑批 |
| `akshare` | 东财 HTTP 爬虫;字段易变 | 想用东财行业板块时 |

**价格统一后复权 (hfq)**:三个源的除权除息跳空都已消除(mootdx 用同源 xdxr 事件复权);
盘中 15:05 前拉到的当日半根 K 线不会写入缓存;增量更新自带接缝校验,数据异常自动全量重拉;
volume 统一为"股"。**切换 source(或缓存口径升级)自动 force_refresh**(`data/.data_source` 标记比对)。
行业板块在 mootdx/baostock 下统一走 mootdx 的通达信行业指数(88xxxx)。

**回测执行真实性(2026-06)**:一字涨停拒买/跌停拒卖、multi_lot 默认仅信号边沿开仓
(`backtest.entry_mode: edge`)、组合差量调仓 + 换手率指标、退市持仓自动核销;
训练标签为 open-to-open(与 T+1 开盘成交对齐)。日报与回测的一致性有契约测试保障。

```yaml
data:
  history_days: 500
  cache_dir: "data"
  source: mootdx          # 或 baostock / akshare
  force_refresh: false
```

## 配置股池与指数

```yaml
stocks:
  - {code: "600519", name: "贵州茅台", sector: "白酒"}   # sector 填行业名或 6 位 TDX 代码,可选

context:
  indices:
    - {code: "sh000001", name: "上证指数"}
    - {code: "sz399001", name: "深证成指"}
```

完整配置字段见 **[docs/reference/config.md](docs/reference/config.md)**。

## 输出位置

- `reports/<日期>.html` + `reports/latest.html` — 日报(单股块默认折叠,首屏几乎瞬开)
- `reports/backtest/` `reports/portfolio/` `reports/ab/` `reports/portfolio_ab/` — 各类报告
- `data/*.parquet` — 行情/指数/板块/因子面板/财务等缓存(可删,下次自动重建)

## Windows 计划任务

复制 `scripts/stockpool_task.xml` 改项目路径,然后:

```cmd
schtasks /Create /XML scripts\stockpool_task.xml /TN "Stockpool Daily"
```

周一至周五 15:30 触发;脚本自查交易日历,节假日自动 exit 0。

## 给开发者 / AI 的文档

- **[CLAUDE.md](CLAUDE.md)** — 架构地图 + 关键设计决策(后续 Claude 会话入口)
- **[docs/reference/](docs/reference/)** — 模块 / 配置 / 引擎约定 / 因子库 / 测试 的明细
- `docs/ab_validation_results.md` — 所有 A/B 验证结论
- `docs/superpowers/specs/` — 各功能设计 spec

## ⚠️ 免责声明

本工具产出基于公开行情数据的技术指标计算,信号与打分仅供个人技术分析参考,**不构成任何投资建议**。
