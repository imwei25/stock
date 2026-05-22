# stockpool — A 股养龙股池技术信号分析

每日扫描配置文件中的 A 股池,计算技术信号综合评分,结合大盘指数和行业板块环境,产出交互式 HTML 报告。

详细设计见 `docs/superpowers/specs/2026-05-17-a-share-signal-tool-design.md`。

## 快速开始

```bash
# 1. 安装(需要 Python 3.10+)
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

# 2. 编辑股池(可选)
notepad config.yaml

# 3. 跑一次
.venv/Scripts/python -m stockpool run

# 4. 看报告
start reports/latest.html
```

## 常用命令

```bash
.venv/Scripts/python -m stockpool run                            # 默认全跑
.venv/Scripts/python -m stockpool run --refresh                  # 忽略缓存重拉
.venv/Scripts/python -m stockpool run --stocks 605589,603986     # 只跑两只
.venv/Scripts/python -m stockpool run --skip-trading-day-check   # 周末调试
.venv/Scripts/python -m stockpool backtest                       # 回测所有股票
.venv/Scripts/python -m stockpool fetch-universe                 # 拉全 A 股缓存 (训练用)
.venv/Scripts/python -m pytest                                   # 全套单元测试

# 因子库 (111 个:10 内置技术因子 + 101 WorldQuant Alpha)
.venv/Scripts/python -m stockpool factors list                          # 列全部
.venv/Scripts/python -m stockpool factors list --source wq101           # 按来源筛
.venv/Scripts/python -m stockpool factors list --type cross_sectional   # 按类型筛
.venv/Scripts/python -m stockpool factors show alpha_001                # 看元数据
.venv/Scripts/python -m stockpool factors pick                          # 打开 HTML 选择器
```

## 数据源 (`data.source`)

支持三种后端,在 `config.yaml` 的 `data.source` 切换:

| 来源 | 特点 | 何时用 |
|---|---|---|
| `mootdx` (默认) | 通达信 TCP 直连;**含当日盘中数据**(几分钟延迟);无 token | 日常使用、盘中查看 |
| `baostock` | 完全免费,无 token;**收盘后约 18:00 才更新当日** | 稳定历史回测、盘后跑批 |
| `akshare` | 东方财富 HTTP 爬虫;上游字段易变,**仅作兜底** | 想用东财行业板块时显式选它 |

行业板块在 `source=mootdx` / `baostock` 下统一走 mootdx 的**通达信行业指数 (88xxxx)**,稳定性远高于东财爬虫。`config.yaml` 里 `stocks[].sector` 既可以填行业名(查内置映射表)也可以直接填 6 位 TDX 代码(如 `880305`)。内置映射见 `src/stockpool/data_sources/mootdx_backend.py::_TDX_INDUSTRY_CODES`,需要新行业时直接加一行即可。

切换 source 时建议先设 `force_refresh: true` 跑一次:不同后端 volume 计量单位不同(mootdx = 手,baostock = 股),旧缓存若混入会让相对成交量指标失真。

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

- `reports/YYYY-MM-DD/index.html` — 当日报告
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

**因子库**总共 111 个,通过双轴元数据组织,方便粗筛:

- **来源 (sources)**:`builtin`(10 个移植自老信号的技术因子)/ `wq101`(WorldQuant 101 Formulaic Alphas 全套)
- **类型 (types)**:`momentum` / `reversal` / `trend` / `volatility` / `volume` / `time_series` / `cross_sectional` / `industry_neutral`(每个因子可同时打多个标签)

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

### 数据质量检查

每次拉取后自动检测:
- 停牌K线(成交量为0)
- 单日涨跌幅 >20%(可能含数据异常)
- 日期间隔 >7 天(疑似长期停牌或数据缺失)

检测到问题时在报告中以橙色警告框显示。

## ⚠️ 免责声明

本工具产出基于公开行情数据的技术指标计算,信号与打分仅供个人技术分析参考,
**不构成任何投资建议**。
