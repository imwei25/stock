# stockpool — A 股养龙股池技术信号分析

每日扫描配置文件中的 A 股池,计算技术信号综合评分,结合大盘指数和行业板块环境,产出交互式 HTML 报告。

详细设计见 `docs/superpowers/specs/2026-05-17-a-share-signal-tool-design.md`。

## 快速开始

```bash
# 一键装好整套环境(Python venv + 项目依赖 + Rust 加速 + maturin)
# 跨平台:Windows (Git Bash) / macOS / Linux,idempotent
bash scripts/setup_env.sh

# 不想装 Rust?(因子 ops 走 pandas fallback,~5-10× 慢但可用)
bash scripts/setup_env.sh --skip-rust

# 编辑股池(可选)
notepad config.yaml          # macOS/Linux: $EDITOR config.yaml

# 跑一次(Windows 路径示例;macOS/Linux 用 .venv/bin/python)
.venv/Scripts/python -m stockpool run

# 看报告
start reports/latest.html    # macOS: open;Linux: xdg-open
```

`setup_env.sh` 干的事(手动安装时参考):

| 组件 | 必需 | 怎么装 |
|---|---|---|
| Python 3.10+ | 是 | <https://www.python.org/downloads/> |
| `.venv` + stockpool + dev deps | 是 | `python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"` |
| Rust stable + cargo | 推荐 | <https://rustup.rs/>(脚本走 winget on Windows / `sh.rustup.rs` on \*nix) |
| MSVC Build Tools (Windows) | Rust 才需要 | VS 2022 Community + "Desktop development with C++" workload |
| maturin | Rust 才需要 | `.venv/Scripts/pip install maturin` |
| `rust/stockpool_ops` crate | Rust 才需要 | `maturin develop --release --manifest-path rust/stockpool_ops/Cargo.toml`(PR-2 起存在) |

无 Rust 时所有 hot factor op 自动走 `_ops_py.py` 的 pandas oracle(`STOCKPOOL_USE_PYTHON_OPS=1` 环境变量可在 Rust 装好时强制走 pandas,用于 diff 比对)。

## 常用命令

```bash
.venv/Scripts/python -m stockpool run                            # 默认全跑
.venv/Scripts/python -m stockpool run --refresh                  # 忽略缓存重拉
.venv/Scripts/python -m stockpool run --stocks 605589,603986     # 只跑两只
.venv/Scripts/python -m stockpool run --skip-trading-day-check   # 周末调试
.venv/Scripts/python -m stockpool backtest                       # 回测所有股票
.venv/Scripts/python -m stockpool backtest --refresh-factor-panel # 重算 ml_factor 因子面板缓存 (data/factor_panels/)
.venv/Scripts/python -m stockpool fetch-universe                 # 拉全 A 股缓存 (训练用 + Pool B 必需)
.venv/Scripts/python -m pytest                                   # 全套单元测试

# 因子库 (~165 base names / ~280-320 variants:WQ101 全集 + 论文 B 9 家族精神复现 + EWMA + 基本面)
.venv/Scripts/python -m stockpool factors list                          # 列全部
.venv/Scripts/python -m stockpool factors list --source wq101           # 按来源筛
.venv/Scripts/python -m stockpool factors list --type cross_sectional   # 按类型筛
.venv/Scripts/python -m stockpool factors list --type fundamental       # 基本面因子(PE/PB/ROE 等)
.venv/Scripts/python -m stockpool factors show alpha_001                # 看元数据
.venv/Scripts/python -m stockpool factors pick                          # 打开 HTML 选择器
.venv/Scripts/python -m stockpool factors analyze --universe all --output reports/factor_analysis
.venv/Scripts/python -m stockpool factors pick-by-ic --input reports/factor_analysis/<日期>.json --output reports/selection.json --top-n 20
```

### 基本面因子(首次启用)

基本面因子族(PE/PB/ROE 等)需先从 baostock 拉财务数据(首次约 30-60 分钟,串行;30 天有效缓存):

```bash
# 任意命令加 --refresh-fundamentals 触发首次拉(或缓存过期重拉)
.venv/Scripts/python -m stockpool run --config config.yaml --refresh-fundamentals
.venv/Scripts/python -m stockpool backtest --config config.yaml --refresh-fundamentals
.venv/Scripts/python -m stockpool portfolio-backtest --config config.yaml --refresh-fundamentals

# 之后正常使用,基本面因子自动按 pubDate 前向填充
.venv/Scripts/python -m stockpool factors list --type fundamental
.venv/Scripts/python -m stockpool factors analyze --universe pool --output reports/factor_analysis
```

PIT 警告:基本面数据按 **公告日(pubDate)** 而非 **报告期末(statDate)** 对齐,确保无未来泄露。

```bash
# A/B 测试 (比较两个策略在同一池/同段历史下的优劣 → reports/ab/latest.html)
.venv/Scripts/python -m stockpool ab --config ab.yaml
.venv/Scripts/python -m stockpool ab --config ab.yaml --arm <name>          # 只跑单边, 打印指标到 stdout
.venv/Scripts/python -m stockpool ab --config ab.yaml --no-share-pool       # 强制每边独立 load 池

# Portfolio backtest — 横截面 top-K 等权 + 周期 rebalance + T+1 (PR-1/PR-2/PR-3)
# 需要在 config.yaml 设 portfolio_backtest.enabled: true
# PR-2 起 universe 自动从 data/universe.parquet 装 4000+ 票 (无该文件回退 cfg.stocks)
# PR-3 起 staggered_starts > 1 自动跑 N 个 offset 出 ensemble + 包络图
.venv/Scripts/python -m stockpool portfolio-backtest --config config.yaml
.venv/Scripts/python -m stockpool portfolio-backtest --refresh-scores       # 强制重算 score panel
.venv/Scripts/python -m stockpool portfolio-backtest --workers 8            # 并行打分 (默认 auto≤3; 每 worker ~6GB; 1=串行)

# Portfolio AB — 比较两套 portfolio 策略 (与 per-stock `ab` 平行)
.venv/Scripts/python -m stockpool portfolio-ab --config portfolio_ab.yaml
.venv/Scripts/python -m stockpool portfolio-ab --config portfolio_ab.yaml --arm <name>   # 调试单 arm, stdout 打印指标
.venv/Scripts/python -m stockpool portfolio-ab --config portfolio_ab.yaml --parallel-arms  # 两 arm 并行(subprocess; peak 内存 ~2×)
.venv/Scripts/python -m stockpool portfolio-ab --config portfolio_ab.yaml --workers 8       # 并行打分 (默认 auto≤3; 每 worker ~6GB; 1=串行)
```

`portfolio_ab.yaml` 最小示例(模板见 `portfolio_ab.yaml.example`):

```yaml
base_config: config.yaml
arms:
  arm_lasso_ic:
    strategy:
      name: ml_factor
      ml_factor: {selector: {type: lasso}, weighter: {type: ic}}
  arm_composite:
    strategy: {name: composite_verdict}
    portfolio_backtest:
      portfolio: {top_k: 15}        # 只改 top_k, 其他字段从 base 继承
```

每个 arm 只能覆盖 `strategy:`(整段替换)和 `portfolio_backtest:`(字段级合并),其他顶层字段(`stocks` / `data` / `indicators` / `weights` / `verdicts` 等)统统继承 base。两 arm 各算各的 score panel(per-arm content_hash 隔离)。输出 `reports/portfolio_ab/<日期>.html` + `latest.html`,含 banner + 聚合指标 + Δ + 双 arm 净值 overlay + per-stock 贡献分解 + 已交易 code 集合分析。

`portfolio_backtest` 段最小示例:

```yaml
portfolio_backtest:
  enabled: true                     # 默认 false (opt-in)
  portfolio:
    top_k: 20                       # 每次取分数前 K 等权
    rebalance_n_days: 5             # 每 N 个 bar 调仓一次
    max_per_industry: 5             # 同行业最多持仓数 (PR-2 起生效;null 关闭)
    initial_cash: 1.0
  eligibility:                      # 逐 bar 漏斗过滤 (PR-2 起生效)
    min_avg_amount_20d: 5e7         # 最近 20 bar 均成交额下限 (close * volume * 100)
    exclude_st: true                # 名称含 "ST" 排除
    min_history_bars: 60            # 历史不足这么多 bar 的剔除
  staggered_starts: 1               # >1 启用 staggered ensemble (PR-3): N 个 offset 串行各跑一份, 聚合成 envelope (min/p25/median/p75/max) + ensemble mean
  score_cache_dir: data/portfolio_scores
```

输出 `reports/portfolio/<日期>.html` + `latest.html`(净值曲线 + 等权 B&H 基准 + 持仓数量时间线 + 指标表)。Score panel 按 `cfg.content_hash` 缓存到 `data/portfolio_scores/<hash>.parquet`,改 yaml 任一字段都失效。PR-2 起 sector map 通过 `load_or_build_industry_map(source="auto")` 加载(首次运行 baostock 拉,后续走 30 天 parquet 缓存)。

## 数据源 (`data.source`)

支持三种后端,在 `config.yaml` 的 `data.source` 切换:

| 来源 | 特点 | 何时用 |
|---|---|---|
| `mootdx` (默认) | 通达信 TCP 直连;**含当日盘中数据**(几分钟延迟);无 token | 日常使用、盘中查看 |
| `baostock` | 完全免费,无 token;**收盘后约 18:00 才更新当日** | 稳定历史回测、盘后跑批 |
| `akshare` | 东方财富 HTTP 爬虫;上游字段易变,**仅作兜底** | 想用东财行业板块时显式选它 |

行业板块在 `source=mootdx` / `baostock` 下统一走 mootdx 的**通达信行业指数 (88xxxx)**,稳定性远高于东财爬虫。`config.yaml` 里 `stocks[].sector` 既可以填行业名(查内置映射表)也可以直接填 6 位 TDX 代码(如 `880305`)。内置映射见 `src/stockpool/data_sources/mootdx_backend.py::_TDX_INDUSTRY_CODES`,需要新行业时直接加一行即可。

**切换 source 会自动 force_refresh**:`data/.data_source` 记录上次拉数据的后端,`fetch_daily` / `fetch_index_daily` / `fetch_sector_daily` / `fetch_universe` 启动时比对,不一致就丢弃旧缓存重拉,无须手动 `force_refresh: true`。不同后端 volume 计量单位不同(mootdx = 手,baostock = 股),否则混源会让相对成交量指标失真。

`fetch-universe` 默认按 `cfg.data.source` 拉每只票 K 线;`--source` 仅用于临时覆盖(可选 `mootdx`/`baostock`/`akshare`)。**注**:全市场清单本身只有 mootdx 实现,所以"列清单"那一步永远走 mootdx。

```yaml
data:
  history_days: 500
  cache_dir: "data"
  source: mootdx      # 或 baostock / akshare
  force_refresh: false
```

## 加股票

打开 `config.yaml`,在 `stocks:` 列表里追加一行。`sector` 为可选字段,填写对应的东方财富行业板块名(不填则跳过板块信号):

```yaml
stocks:
  - {code: "600519", name: "贵州茅台", sector: "白酒"}
```

## 配置大盘指数

`context.indices` 指定每次运行时额外拉取并显示买卖信号的指数。默认上证 + 深证成指,可自行增减:

```yaml
context:
  indices:
    - {code: "sh000001", name: "上证指数"}
    - {code: "sz399001", name: "深证成指"}
    - {code: "sz399006", name: "创业板指"}
```

## 调整打分权重

`config.yaml` 的 `weights:` 段每个数字都可改。
信号定义见 `docs/superpowers/specs/2026-05-17-a-share-signal-tool-design.md` § 5。

## Windows 计划任务

复制 `scripts/stockpool_task.xml`,改里面的项目路径,然后:

```cmd
schtasks /Create /XML scripts\stockpool_task.xml /TN "Stockpool Daily"
```

任务设置为周一至周五 15:30 触发。脚本本身会查交易日历,节假日自动 exit 0。

## 输出位置

- `reports/YYYY-MM-DD/index.html` — 当日报告(单股块默认折叠,展开后再渲染图表;echarts 库全文只引一次,首屏几乎瞬开)
- `reports/YYYY-MM-DD/run.log` — 当日运行日志
- `reports/latest.html` — 永远是最新一份(任务栏快捷方式固定它)
- `reports/backtest/<date>.html` + `reports/backtest/latest.html` — 回测报告
- `reports/factors_picker.html` — HTML 因子选择器(`factors pick` 生成)
- `data/{code}_daily.parquet` — 个股行情缓存
- `data/idx_{symbol}.parquet` — 大盘指数缓存(如 `idx_sh000001.parquet`)
- `data/sector_{name}.parquet` — 行业板块缓存(如 `sector_化工.parquet`)

所有缓存可删除,下次运行自动重建。缓存超过 5 个自然日未更新时会自动触发增量拉取。

## 功能说明

### 信号评分

每个交易日对每只股票运行 16 种技术信号检测(MA 金/死叉、MACD、KDJ、RSI、布林带、成交量、突破),日线和周线各出一组信号分,按 7:3 权重合并为终分。双线共振时额外加分。

终分映射到五档判定:

| 终分 | 判定 |
|------|------|
| ≥ +6 | 🟢🟢 强烈买入 |
| ≥ +3 | 🟢 买入观察 |
| (-3, +3) | ⚪ 观望 |
| ≤ -3 | 🔴 卖出观察 |
| ≤ -6 | 🔴🔴 强烈卖出 |

### 大盘与板块环境

每次运行额外拉取配置中的大盘指数(默认上证 + 深证成指)和每只股票对应的行业板块,对它们运行相同的信号流水线。报告顶部显示市场环境横栏,每个股票详情页顶部显示大盘 + 板块的买卖判定,方便判断个股信号是否与市场方向一致。

### 历史回测

- **单信号命中率**:统计过去 500 日内每种信号出现后 5/10/20 日的平均涨幅与胜率
- **综合评级回测**:走前向回测重建历史每日判定,按评级分桶统计平均收益
- **权益曲线**:模拟持有 5/10/20 日的资金曲线,对比买入持有基准
- **撮合口径(T+1)**:信号在第 `t-1` 根 bar 收盘后生成,引擎用**第 `t` 根 bar 的 `open[t]`**(A 股集合竞价价)作为买入/卖出成本。除非次日开盘直接涨停打不进单,否则视为可成交。买入持有基准也锚定 `open[0]`

### ML 因子策略与因子库

除了综合评级,项目还内置一套**两步法 ML 因子组合策略**(`strategy.name: ml_factor`):
Lasso 在训练集筛因子 → IC/IR/equal 加权 → 训练集分位数映射到买卖判定,walk-forward 重训。

`stockpool run` 生成日报时也会读取 `strategy.name`,用对应策略给出每只票的最终评级。`ml_factor` 在日报路径下每个**自然月最多重训一次**(模型与分位数 pickle 到 `<cache_dir>/ml_models/<sig>_<code>.pkl`,`<sig>` 是 MLFactorConfig 的 8 位哈希,改 factors/horizon/selector 等任一项即自动失效),同月内只做单次预测,跨月自动重训。

**因子库**总共 165 个 base 因子(WQ101 全集 + 内置技术因子含论文 B 家族精神复现 + EWMA 变体 + 基本面 custom),通过双轴元数据组织,方便粗筛:

- **来源 (sources)**:`builtin`(51 个,含老技术因子 + 论文 B 9 家族精神复现 + EWMA 平滑变体)/ `wq101`(101 个,WorldQuant 101 Formulaic Alphas 全套)/ `custom`(13 个,含基本面 PE/PB/ROE/营收增速等)
- **类型 (types)**:`momentum` / `reversal` / `trend` / `volatility` / `volume` / `time_series` / `cross_sectional` / `industry_neutral` / `fundamental`(每个因子可同时打多个标签)

每个因子是 panel-in / panel-out 的纯函数(`T×N OHLCV 宽表 → T×N 因子值宽表`),
WQ101 里的 `rank` `decay_linear` `indneutralize` 等横截面算子全部支持。

**HTML 选择器** (`stockpool factors pick`):

```bash
.venv/Scripts/python -m stockpool factors pick                          # 起本地服务 + 自动开浏览器
.venv/Scripts/python -m stockpool factors pick --output my_sel.json     # 自定义 JSON 路径
.venv/Scripts/python -m stockpool factors pick --port 18765             # 固定端口
.venv/Scripts/python -m stockpool factors pick --static                 # 老的静态 HTML 文件模式
```

页面上勾选因子或整类后点 **"应用"** 按钮,后台直接写 `reports/selection.json`(无需手动下载/移动文件)。重开页面会自动从同一个文件载回之前的选择。

也提供 "下载 selection.json" 和 "复制 YAML" 作为离线兜底。

在 `config.yaml` 引用 selection.json:

```yaml
strategy:
  name: ml_factor
  ml_factor:
    factors_file: reports/selection.json   # 与 factors: [...] 二选一
    panel_mode: pooled                     # WQ101 cross-sec 必须 pooled
    training_universe: all                 # 训练用全市场 (默认 pool 只用 cfg.stocks)
    share_pool_fit: true                   # 跨股共享月度 fit(默认 true),省 N 倍 Lasso
    horizon: 5
    train_window: 250
```

> **`share_pool_fit`**(默认 `true`,仅 `panel_mode: pooled` 生效):同月内所有 cfg.stocks
> 共享一份 `(pipeline, quantiles)`,缓存键 `(sig, year, month)`。代价:训练池里不再
> 剔除 host 自己,host 以 ~1/N 权重(N≈pool 大小)进入自己的训练,IC 加权下偏差可忽略。
> 关掉的话每股、每个 refit_bar 都会重训。

#### 训练池 vs 应用池(`training_universe`)

`ml_factor` 默认 `training_universe: pool` —— 训练只用 `stocks` 里那几只票,样本太小,
cross-sec 因子(rank/scale/indneutralize 等)在 ~10 列的窄面板上几乎退化成常数。

切到 `training_universe: all`(仅在 `panel_mode: pooled` 下生效):

1. 先**一次性**拉全市场缓存(剔除 ST/科创/北交,~4350 只,8 线程 ~1 分钟):
   ```bash
   .venv/Scripts/python -m stockpool fetch-universe
   ```
   数据落到 `data/<code>_daily.parquet`,后续每天的 `stockpool run` 会自动从这里读全量,
   按 cfg.stocks 同样的逻辑做 incremental 更新(只补今天的几根 bar)。

2. 在 `config.yaml` 把 `training_universe` 改成 `all`。
3. 像往常一样 `stockpool run` / `backtest` —— **训练在全市场上跑(IC、Lasso 都见
   到完整的横截面),日报和回测的输出仍然只针对 `cfg.stocks`**。

模型缓存键 `<sig>` 会自动反映 `training_universe` 和 `share_pool_fit`,切换后旧缓存自动失效,首次跑会重训。`share_pool_fit=true` 时落盘文件名为 `<sig>_shared.pkl`(全部 cfg.stocks 共用),否则按老的 `<sig>_<code>.pkl` 每股一份。

行业中性化(`Alpha48` / `Alpha58` / `Alpha67` 等):自动用 `stocks[].sector` 当分组键;
`IndClass.subindustry` 退化到一级 `sector`(项目无子行业数据)。
`Alpha56` 需要总市值(`cap`),目前返回全 NaN —— 单独跳过即可。

### Pool B — 全市场量化推荐池

在保留 `cfg.stocks`(Pool A,手填 watchlist)的前提下,日报底部多出一段"推荐池":
对全市场 A 股调用当前 `strategy.predict_latest` 打分,经"流动性 + ST + 行业上限"
漏斗后取 top-30,两池**独立**、**允许重叠**(重叠股标 ⭐)。

```yaml
recommend_pool:
  enabled: true                  # 默认开启
  top_n: 30
  min_avg_amount_20d: 50000000   # 5000 万元;A 股次新/小盘容易刷分,设个最低成交额
  max_per_industry: 5            # 行业分散;"未知"也算一个桶
  refresh: weekly                # weekly(默认) | always | never
  cache_dir: "data/recommend_pool"
  industry_map_max_age_days: 30
```

**前置条件**:必须先跑过 `python -m stockpool fetch-universe`(否则全市场缓存
空,Pool B 退化为空表)。首次启用会拉一次行业映射,缓存到
`data/stock_industry_map.parquet`(30 天有效期)。数据源由
`industry_source` 控制,默认 `auto` = 先 **baostock**(~5-10 秒,无 token,
最稳)→ 失败再 akshare(~1-2 分钟,易受代理影响)。mootdx 路径不可用
(TDX 服务器对 `block_hy.dat` 返回 0 字节)。

> 行业映射全空(两个源都失败)时,Pool B 会跳过 `max_per_industry`
> 约束并打 warning,不至于把池砍到只剩 5 条。

**缓存策略**:`weekly` 模式下,每次启动算当前 ISO 周;同周直接读盘,跨周
重算。缓存键含 `cfg.content_hash`,任何 yaml 改动都自动失效 —— 换 strategy
后下次运行自动用新策略重排。

**Pool B 与 strategy 解耦**:不绑死 `ml_factor`;`composite_verdict` 也能给
`final_score`,只是后者不是为"全市场打分"设计的,结果偏离 ml_factor 的口径。
推荐配合 `ml_factor` + `training_universe: all` 使用 —— 训练池和应用池
天然对齐,cross-sec 因子在 ~4000 只票上拿到真实横截面,Pool B 排序最有意义。

**不做的事**:Pool B 不做回测(MVP);周频再平衡 30 股组合的曲线引擎留作
follow-up。日报上 Pool B 段失败不影响 Pool A 段。

### 数据质量检查

每次拉取后自动检测:
- 停牌K线(成交量为0)
- 单日涨跌幅 >20%(可能含数据异常)
- 日期间隔 >7 天(疑似长期停牌或数据缺失)

检测到问题时在报告中以橙色警告框显示。

### 关于 LightGBM selector / weighter(默认已回退到 Lasso+IC)

F2 PR-B1 引入了 LightGBM selector,PR-B2 引入了 LightGBM weighter,默认值曾切到 `"lightgbm"`。**2026-05-24 在 16 股 × 500 bar baseline 上做了 A/B 验证**(`docs/ab_validation_results.md`):

- **LGB+LGB vs Lasso+IC**:sharpe Δ=-0.20,total return Δ=-20%,7/16 股 B 胜 — **显著倒退**
- **拆解**:LGB selector 单独几乎平局(P1-1: Δ=-0.027),退化主要来自 LGB weighter 的过拟合(trade count 翻倍,churn 高 sharpe 低)
- **小训练集**:~250 bars × 16 股 ≈ 4k 行,LGB 的 `num_leaves=15`/`min_data_in_leaf=20` 默认在这规模上还是太宽

**结论**:已把 `selector.type` / `weighter.type` 默认值从 `"lightgbm"` 回退到 `"lasso"` / `"ic"`。LGB 仍可通过 YAML opt-in,但建议:
1. 先把 `num_leaves` 调到 7-10、`min_data_in_leaf` 调到 50+ 看是否能压住过拟合
2. 或先把 `training_universe` 切到 `"all"` 扩大训练集(需先 `python -m stockpool fetch-universe`)
3. 再用 `python -m stockpool ab --config <ab.yaml>` 对照 Lasso+IC baseline 看是否净改善

**结构性收益(仍保留)**:
- **PR-A embargo**(默认 `embargo_days: null` = auto = horizon):A/B 验证 tied,无副作用,继续保留
- **`pooled` panel_mode**(默认):A/B P3-1 验证 Δsharpe=+0.23,11/16 胜 — 真收益,继续默认

**关于 `weighter.contributions()`**:在 LGB weighter 下返回 SHAP 值,在 IC/IR/Equal 线性 weighter 下返回 `standardised(X) * weights`。两者形状一致(行 = 样本,列 = 因子),但 LGB 行和 ≈ `predict(X) - base_value`(SHAP convention)而非完全等于 `predict(X)`。

### AB 候选池(可选)

per-stock AB 默认在 `cfg.stocks`(几只)上对比,样本太小;全市场又太慢。
中间方案:构建一个 ~100 票的行业分层候选池,AB 对比时通过开关复用。

```bash
# 一次性构建(需先 fetch-universe)
.venv/Scripts/python -m stockpool ab-pool build
# 浏览器查看池子内容(支持行业/代码/名称筛选)
.venv/Scripts/python -m stockpool ab-pool show
```

在 `ab.yaml` 或 `portfolio_ab.yaml` 顶层加 `use_ab_pool: true` 即启用。
池子静态、手动重建(`--refresh`),保证历史 AB 结果可复现。

### 对比两个策略 — A/B testing

`stockpool ab` 在**同一份 stocks / data / indicators / context** 下并行跑两个策略,
产出 side-by-side 净值曲线 + 聚合差值表 + Sharpe 散点图 + 差值直方图,
方便回答"换了一个 selector / weighter / 因子集后到底有没有变好"。

新建一个 `ab.yaml`(完整可注释样例见 `ab.yaml.example`):

```yaml
base_config: config.yaml           # 共享的 stocks/data/indicators 来源

arms:                              # 恰好两个 arm,key 名自由
  composite:
    strategy: {name: composite_verdict}
    backtest: {equity_curve_holding_days: [10]}

  ml_lgbm:
    strategy:
      name: ml_factor
      ml_factor:
        factors_file: reports/selection.json
        selector: {type: lightgbm}
        weighter: {type: lightgbm}
    backtest: {equity_curve_holding_days: [10]}
```

跑:

```bash
.venv/Scripts/python -m stockpool ab --config ab.yaml
# 输出: reports/ab/<日期>.html  +  reports/ab/latest.html
```

调试单边(只跑 ml_lgbm,打印指标到 stdout、不出 HTML):

```bash
.venv/Scripts/python -m stockpool ab --config ab.yaml --arm ml_lgbm
```

**规则**:
- 每个 arm 只能覆盖 `strategy:` 和 `backtest:` 段;`indicators` / `weights` / `verdicts` / `scoring` 等顶层字段必须共享(强制 A/B 对比的"环境一致性")
- `equity_curve_holding_days` 强制单元素列表(A/B 在固定 N 下对比;扫 N 用普通 `backtest` 命令)
- 可选 `stocks_filter: [...]` 选股池子集(只能减,不能加;减完仍须在 base.stocks 内)
- ML 缓存通过 `effective_cfg.content_hash` 自动隔离两个 arm(`ml_models/<sig>_<code>.pkl`),改完任一边参数下次只重训那一边

**当前不支持**:portfolio-level 回测(每个 arm 仍是 per-stock 独立 + 跨股聚合)、统计显著性检验(样本太小)、覆盖 `weights`/`verdicts`/`scoring` 顶层字段(留作 follow-up,等 `composite_verdict` 参数下沉到 `strategy.composite_verdict.*` 子段后自动可用)。完整 schema + 设计权衡见 `docs/superpowers/specs/2026-05-24-ab-testing-design.md`。

### 仓位 sizing

(适用于所有回测模式 — `backtest` 命令和 A/B 测试均生效)

`backtest.sizing` 子段控制每笔买入的仓位大小:

- **`sizing.type: fixed`** — 每只票同样大小(`sizing.fixed.size`,默认 10%)
- **`sizing.type: vol_target`** (默认,F3 PR-C 起) — 按个股近期波动反比调仓,目标是让每只票贡献的风险大致相等。主效应是降低组合 max DD

老的 `backtest.position_size: 0.1` 仍能工作(自动迁移 + DeprecationWarning),但新代码请直接写 `sizing` 子段。

### Tradability mask (paper B mask-first, opt-in)

在 `config.yaml` 的 `strategy.ml_factor` 段开启:

```yaml
strategy:
  name: ml_factor
  ml_factor:
    # ...其他字段
    mask:
      enabled: true   # 默认 false。开启后训练标签对涨停/停牌/新股头 N 天做双向检查,
                      # 通过 stack_panel_to_xy dropna 自然剔除这些样本(不破坏因子输入值)
```

**关键**:`mask.enabled=true` 时 `MLFactorStrategy` 会自动调 `stockpool.ipo_dates.load_or_build_ipo_dates`
拉一次全 A 股 IPO 日期(baostock,缓存到 `data/ipo_dates.parquet`,30 天有效期)。
首次需要网络拉 ~3-5 秒;后续直接读盘。

启用后训练样本数下降 ~1-5%(取决于股池规模),大样本(训练池 = 全 A 股)上 Sharpe 预期提升 0.05-0.4(论文 B 在 4000+ 票 × 3 年 × 213 因子上报告 +0.44)。

A/B 验证(per-stock):
```bash
.venv/Scripts/python -m stockpool ab --config ab_mask.yaml   # 训练池 4358 + 应用池 16,实测 Δ Sharpe +0.07
```

Portfolio AB 验证(top-K 选股,更贴近 paper B 场景):
```bash
.venv/Scripts/python -m stockpool portfolio-ab --config portfolio_ab_mask.yaml
# 训练池 4358 + portfolio universe 16(top-8 选股),实测 Δ Sharpe +0.04
```

`portfolio_ab_mask.yaml` 的关键字段:
```yaml
portfolio_backtest:
  universe_codes:     # 解耦训练池与 portfolio universe
    - "605589"        # ml_factor training_universe=all 仍走 load_universe_cache (4358 票)
    - "603986"        # 但 precompute_scores + portfolio engine 仅在这个列表上跑
    # ...             # 避免在 4358 票全跑时 precompute_scores segfault
```

缓存:翻 `mask.enabled` 会改变 `cfg.content_hash` 和 ml_model sig hash → 自动重训 + 重算 score_panel。`factor_panel` 不变(mask 不影响因子值)。

### 因子预处理 (Phase 1 + Phase 2, opt-in)

Phase 1 因子预处理(winsorize / cs_zscore / industry_neutralize)+ Phase 2 `mcap_neutralize` 可在 `strategy.ml_factor.preprocess` 段开关,默认全关。
详见 `docs/superpowers/specs/2026-06-06-factor-preprocessing-phase1-design.md`。

```yaml
strategy:
  name: ml_factor
  ml_factor:
    # ...其他字段
    preprocess:
      winsorize: [0.01, 0.99]   # null = 关闭
      zscore: true              # 截面 z-score
      industry_neutralize: true # 跳过 factor types 含 "fundamental" 的因子
      mcap_neutralize: false    # Phase 2 (opt-in):对 log(market_cap) 做 per-day OLS 残差化,PE/PB 等含 mcap 因子自动跳过
```

**注意**:⚠️ Phase 1 AB 验证结果为 INDECISIVE(Δsharpe=+0.013,阈值 +0.05),默认保持全关。
主要问题:z-score 后 score 分布变平,`thresholds.strong_buy=0.9` 触发频率骤降,16 票中 5 票零交易。
Phase 1.5 将做阈值校准 + per-step ablation,见 `docs/ab_validation_results.md` P4-1。

## ⚠️ 免责声明

本工具产出基于公开行情数据的技术指标计算,信号与打分仅供个人技术分析参考,
**不构成任何投资建议**。
