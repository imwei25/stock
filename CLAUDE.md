# stockpool — Claude Code 项目指南

> 后续 Claude 会话的快速上手地图。**不必通读** —— 本文给出架构全貌 + 关键约定,
> 细节按需查 `docs/reference/`。改动项目时遵守文末的[更新文档规则](#改动后更新文档)。

## 项目一句话

A 股技术信号分析工具:可插拔后端拉数据(mootdx/baostock/akshare)→ 计算多指标 / 连续因子
→ 综合评级 或 两步法 ML 因子组合 → 输出 HTML 日报、回测报告、A/B 对比、组合回测。

## 架构全貌

```
数据源 (mootdx 默认 / baostock / akshare)
  └─ fetcher.py        拉数 + parquet 缓存 + 源切换自动 force_refresh
       ├─ indicators.py        MA/MACD/KDJ/RSI/BOLL/Volume/Breakout
       │    └─ signals.py      detect → score → verdict (综合评级路径)
       └─ factors/             ~165 base 因子 (WQ101 全集 + 内置技术 + 基本面)
            └─ ml/             两步法:Lasso/LGB 选因子 → IC/IR/equal/LGB 加权
                               + preprocess.py 截面预处理流水线

策略层 (strategy_factory.py 按 cfg.strategy.name 工厂构造)
  ├─ CompositeVerdictStrategy  综合评级 + 日周共振 (默认)
  └─ MLFactorStrategy          两步法 ML,walk-forward 重训,per_stock/pooled

执行层
  ├─ backtesting/   单股回测引擎 (single / multi_lot + LotSizer sizing,T+1 次日开盘成交)
  ├─ portfolio/     组合回测 (top-K 等权 + 周期 rebalance + 行业 cap + staggered ensemble)
  ├─ ab/            per-stock A/B 对比两策略
  └─ portfolio_ab/  组合级 A/B

输出
  ├─ report.py           日报 HTML (Pool A watchlist + Pool B 全市场推荐池)
  ├─ backtest_report.py  回测 HTML
  └─ recommend_pool.py   Pool B:全市场打分 → 流动性/ST/行业漏斗 → top-N
```

**两条产出路径**:`stockpool run`(日报,每股 `predict_latest`)和 `stockpool backtest`/`portfolio-backtest`(回测,全程 `generate_signals`)。两者都经 `strategy_factory.build_strategy`,共享因子面板预算与缓存。

## 关键设计决策(读这几条避免踩坑)

- **训练池 vs 应用池分离**:`ml_factor` + `training_universe=all` 时,训练在全市场 ~4350 票上跑(cross-sec 因子/IC 看到完整横截面),日报/回测输出仍只针对 `cfg.stocks`。需先 `fetch-universe`,且 `panel_mode=pooled`。
- **因子面板落盘缓存**:`data/factor_panels/<sig>/`,sig = (factors + codes + last_date + preprocess) 哈希;任一变化自动重算。`--refresh-factor-panel` 旁路。
- **ml_factor 模型月度缓存**:`ml_models/<sig>_*.pkl`,日报路径每自然月最多重训一次,`<sig>` 改 factors/selector/weighter/... 即失效。
- **T+1 次日开盘成交**:信号在 `t-1` 收盘后产生,在 `open[t]` 成交。详见 [conventions.md](docs/reference/conventions.md)。
- **价格全链路后复权 (hfq)**:akshare/baostock 直接 hfq;mootdx 用同源 xdxr 做段内锚定 hfq(段首因子=1,增量段锚定缓存尺度)。盘中 15:05 前的当日半根 bar 不入缓存;增量拉取与缓存重叠一根 bar 做接缝校验,不一致自动全量重拉。`.data_source` marker 格式 `<source>:hfq`,旧缓存自动失效迁移。详见 [conventions.md](docs/reference/conventions.md)。
- **训练标签 open-to-open(2026-06 起默认)**:`label_basis=open` → 标签 = `open[t+1+h]/open[t+1]−1`,与 T+1 执行对齐(不含拿不到的隔夜段);embargo/截断数学自动 +1。`close` 可回退 legacy。
- **可交易性 mask 只在标签层**:涨停/停牌/新股的 mask 作用于训练标签(`forward_return_panel` 双向检查,open 基准查实际进出场 bar),**不破坏因子输入**;主板 ST 用 ±5% 阈值(`st_codes` 来自 `stock_basics.parquet` 干净名单)。
- **回测执行真实性(2026-06)**:三引擎一字涨停拒买/跌停拒卖(`backtesting/limits.py`,按代码前缀+ST 推断幅度);multi_lot 默认 `entry_mode=edge`(仅信号边沿开仓);组合 rebalance 为**差量调仓**(存活仓不动)+ turnover 指标;退市/长停持仓按 last_valid_close 计值、60 bar 无报价强制核销。
- **cross-sec 因子需注入 sector_map / mcap_panel**:build factor_panel 前 caller 必须经 `factors.context.set_sector_map` / `set_mcap_panel` 注入,否则 `IndustryRelativeStrengthFactor` 直接 raise(防 cache 中毒)。mcap 股本用 profit 表逐季 PIT(快照仅回退)。
- **选因子无 in-sample 偏差**:`factors analyze --end-date` 把 selection 窗口截止在回测起点之前;analyze 与生产共用同一预处理/mask/标签口径,ic_ir 带 Newey-West 修正。当前 `reports/selection.json` 选自 ≤2024-05-20 窗口,评估期 2024-05-21 起 500 根。
- **训练池保留 ST**:按当前名称整段剔除是前视;ST 仅在应用层(Pool B/推荐)当下剔除。退市股价格历史仍缺失(轻量 PIT 的已知残余偏差,文档明示)。
- **配置 fail loud**:全树 `extra=forbid`;verdicts 阈值排序、scoring 权重和、indicators 信号依赖周期都有校验;`content_hash` 是 resolved 配置哈希(含 selection.json 内容);因子覆盖率 <2% 直接 raise。
- **predict/backtest 一致性有契约测试**:`tests/test_consistency_contract.py`(composite + ml)+ 端到端黄金值锚点;ml 日报标注 `model_fit_date`。
- **预处理默认值经 A/B 定标**:winsorize+zscore 默认开,market_cap_neutralize 默认开,industry_neutralize / symmetric_orthogonalize 默认关。结论见 `docs/ab_validation_results.md`(2026-06 数据修复后重跑中,旧绝对数字不可采信)。
- **LGB selector/weighter 默认已回退到 Lasso/IC**:小训练集上 LGB 过拟合。仍可 opt-in。Lasso 对 y 标准化,选择不随波动状态漂移。

## 模块速查

| 域 | 关键文件 | 详细 |
|---|---|---|
| 入口 / 配置 | `cli.py` / `config.py` / `strategy_factory.py` | [modules.md](docs/reference/modules.md) |
| 数据获取 | `fetcher.py` / `data_sources/*` / `fundamentals_loader.py` / `industry_map.py` / `ipo_dates.py` | [modules.md](docs/reference/modules.md) |
| 信号 / 指标 | `indicators.py` / `signals.py` | [modules.md](docs/reference/modules.md) |
| 因子库 | `factors/`(ops / wq101 / 11 家族 / fundamentals)/ `factors_picker.py` / `factors_analysis.py` | [factors.md](docs/reference/factors.md) |
| Panel / ML | `panel.py` / `ml/`(selector / weighter / preprocess) | [modules.md](docs/reference/modules.md) |
| 回测框架 | `backtesting/`(engine + sizing)/ `backtest_runner.py` / `backtest_composite.py` | [backtesting_framework.md](docs/backtesting_framework.md) |
| 组合 | `portfolio/`(strategy / scoring / engine / eligibility / ensemble) | [modules.md](docs/reference/modules.md) |
| A/B | `ab/` / `portfolio_ab/` | [modules.md](docs/reference/modules.md) |
| 报告 / 推荐池 | `report.py` / `recommend_pool.py` | [conventions.md](docs/reference/conventions.md) |

完整模块表(含 API 契约与设计细节)见 **[docs/reference/modules.md](docs/reference/modules.md)**。

## 快速命令

```bash
# 日报 / 回测 / 全市场缓存
python -m stockpool run --config config.yaml
python -m stockpool backtest --config config.yaml [--stocks 605589] [--refresh-factor-panel]
python -m stockpool fetch-universe [--workers 8] [--limit 100] [--refresh] [--source baostock]

# 因子管理 / 分析 / 选因子
python -m stockpool factors list [--source wq101] [--type cross_sectional]
python -m stockpool factors show alpha_001
python -m stockpool factors pick                                    # HTML 选择器
python -m stockpool factors analyze --universe all --output reports/factor_analysis
python -m stockpool factors pick-by-ic --input <分析.json> --output reports/selection.json --top-n 20 --max-corr 0.6

# A/B / 组合回测 / 组合 A/B
python -m stockpool ab --config ab.yaml [--arm <name>] [--no-share-pool]
python -m stockpool portfolio-backtest --config config.yaml [--refresh-scores]   # 需 portfolio_backtest.enabled: true
python -m stockpool portfolio-ab --config portfolio_ab.yaml [--arm <name>]

# 测试
python -m pytest tests/ -q
```

## 参考文档

| 文档 | 内容 |
|---|---|
| [docs/reference/modules.md](docs/reference/modules.md) | 完整模块表 + API 契约 + 设计细节 |
| [docs/reference/config.md](docs/reference/config.md) | `config.yaml` 全字段 + ml_factor / preprocess / mask / portfolio / recommend_pool / A/B 配置 |
| [docs/reference/conventions.md](docs/reference/conventions.md) | 回测引擎约定(T+1/sizing)+ 数据流 + Pool B + 缓存/报告路径 |
| [docs/reference/factors.md](docs/reference/factors.md) | 因子库(Factor ABC / 双轴元数据 / WQ101 / HTML 选择器) |
| [docs/reference/testing.md](docs/reference/testing.md) | 615 测试的按域覆盖表 |
| [docs/backtesting_framework.md](docs/backtesting_framework.md) | 回测框架完整 API |
| `docs/ab_validation_results.md` | 所有 A/B 验证结论(P1..P4) |
| `docs/superpowers/specs/` | 各功能的设计 spec(按日期命名) |
| [docs/project_review_2026-06.md](docs/project_review_2026-06.md) | 全面审查报告:P0~P3 问题清单(复权/幸存者偏差/选因子泄漏等)+ 分阶段改进路线图 |

## 已知不支持的能力

- 做空、盘中数据、部分成交、资金成本(融资融券)
- 仓位管理仅"满仓单笔"或"固定/vol-target 多笔";无 Kelly / 比例追加(multi_lot 的 every_bar 模式可隐式金字塔,但默认 edge)
- 个股 → 板块的**自动**映射:`cfg.stocks[].sector` 仍需手填(Pool B 的 code→行业映射独立走 baostock/akshare;mootdx 路径不可用)
- **per-stock A/B 不能覆盖顶层 `indicators`/`weights`/`verdicts`/`scoring`**:这些 composite_verdict 参数尚未下沉到子段,arm 只允许覆盖 `strategy:` 和 `backtest:`
- **A/B 报告无统计显著性**:样本太小,只给均值/中位/差值/胜出计数(逐持有期 N 各一张表)
- **Portfolio**:staggered 串行(N=10 ≈ 10× 耗时);portfolio AB 两 arm 各算各的 score panel(无跨 arm 缓存共享);arm 仅允许覆盖 `strategy:` 和 `portfolio_backtest:`
- **盘中触板/部分成交未建模**:一字板开盘拒单已落地(`backtesting/limits.py`),但盘中触及涨跌停的部分成交概率(完整 mask_exec)仍未建模
- **退市股价格历史缺失**:PIT 名单(`stock_basics.parquet` 含 outDate)已落地,但 mootdx 拉不到退市股价格,训练样本仍缺退市前的大负收益段(幸存者偏差的残余,方向为高估)
- **mcap 用 hfq close**:绝对市值被各股复权因子缩放,截面 size 排序为近似(精确需 raw close 或复权因子列)

## 改动后更新文档

**新增/修改任何面向用户的功能或设计时,必须同时更新 `CLAUDE.md`、`README.md` 和对应的 `docs/reference/*.md`。**

- `CLAUDE.md` — 给 Claude 的架构地图(高层 + 关键决策 + 指针)。原则:**后续 Claude 不读源码也能正确帮用户做事**。明细放进 `docs/reference/`,别把 CLAUDE.md 撑回冗长。
- `README.md` — 给用户的入口(快速开始 + 常用命令 + 核心配置示例)。原则:**新用户从 README 能跑通核心场景**。
- `docs/reference/*.md` — 明细的归宿:模块 API、config 全字段、引擎约定、因子库、测试表。

**何时必须更新**(满足任一项,在同一次改动里改完):新增/删除/重命名顶层模块或公开 API;`Strategy` / 引擎公开签名变化;`config.py` schema 字段增删/默认值/语义变化;CLI 子命令或参数变化;默认行为切换(engine / sizing / 数据源行业路由 / preprocess 默认);测试新增按域覆盖文件;数据流/缓存/报告路径变化;新增因子来源/类型标签。

**怎么改**:定位对应小节,把改动点写进去;CLAUDE.md 只在影响架构全貌/关键决策时动,纯明细改 `docs/reference/`。两份用户级文档(CLAUDE.md + README.md)的命令示例保持一致。**不需要每个 commit 都更新**,只在上述变化发生时跟一次。
