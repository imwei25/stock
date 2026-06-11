# stockpool 项目全面审查报告与改进路线图(2026-06)

> 由 4 路并行代码审计产出(数据层 / 因子与 ML 层 / 回测与组合引擎 / 策略与工程质量),
> 所有发现均基于当前 `feat/composite-backtest` 工作区实际代码,多处用 `data/` 下真实
> parquet 数据做了实证。跨域重复发现已去重合并,行号对应审计时点。

---

## 修复状态(2026-06-12)

> **阶段一~四全部完成**(P0×6、P1×10、P2×30、P3 绝大部分),测试 857 全绿。
> 提交序列:`ccef8b4`(hfq/盘中bar/接缝)→ `fix(core)` 两批(volume/分页/基本面/
> baostock/标签 o2o/IC 统一/metrics)→ `feat(data)`(PIT listing/ST)→
> `fix(ml)`(analyze 对齐/互锁/覆盖率闸)→ `feat(engine)`(涨跌停拒单)→
> `feat(portfolio)`(差量调仓/turnover/退市估值)→ `feat(engine/ab)`(edge/
> 逐N报表/hit-rate)→ `fix(config)`(防呆/缓存键)→ `fix(report/signals)`
> (契约测试/黄金值/周线 repaint)→ `chore(d5/d6)`(粘性/卫生/lockfile)。
> 数据已按新口径全量重建(hfq+v2,4597/4598);selection.json 重选
> (窗口 ≤2024-05-20,无 in-sample 偏差);A/B 全量重跑进行中,结果将更新
> `docs/ab_validation_results.md`。未做项:仅「长期优化方向」与个别 P3
> 文案级(P3-19 注记、[TIME] print 双轨保留)。

## 执行摘要

**框架工程质量明显高于个人项目平均水平**:Strategy 抽象干净、T+1 open-fill 语义在三个引擎间一致、walk-forward embargo 数学正确、mask 只作用标签层、缓存签名体系完整、WQ101 公式翻译忠实(抽查 26 条仅约定级差异)、A/B 驱动默认值的方法论姿态正确、677 个测试覆盖面广。**"难做对的部分"基本做对了。**

**但地基有裂缝,当前所有回测绝对数字不可直接采信**,三个结构性根因:

1. **价格数据不可信(P0-1/2/3)**:默认源 mootdx 是不复权价,除权除息日的虚假跳空污染全部信号、训练标签与回测 PnL;备选源的前复权又与增量缓存机制根本不兼容;盘中运行还会把半根 K 线永久冻结进缓存。
2. **样本选择带未来信息(P0-4/6)**:训练池是"今天的存活名单"(幸存者偏差),因子清单是在含评估期的全样本上选出的(in-sample 选择偏差)。`docs/ab_validation_results.md` 的全部绝对数字都建立在这两层偏差上。
3. **组合 ml_factor 路径有接线 bug(P0-5)**:训练用预处理后的全市场因子面板,预测端却退化为未预处理的单股因子,`with_stock()` 在 src 下零调用——组合回测对项目主打的 cross-sec ML 策略基本无效。

修复顺序的总原则:**先修数据地基,再修方法论,然后才轮到执行真实性和调参**。在价格序列正确之前,其余一切优化都是在噪声上调参。

---

## P0 — 动摇回测可信度的地基问题

### P0-1 默认数据源 mootdx 返回不复权价,除权除息日产生虚假收益 ✅ 已修复 (2026-06-11)

> **修复**:mootdx 现用同源 xdxr 事件(TCP)做段内锚定 hfq(`mootdx_backend._apply_hfq`);
> akshare 切 `adjust="hfq"`、baostock 切 `adjustflag="1"`。marker 升级为 `<source>:hfq`,
> 旧的不复权缓存自动失效全量重拉。测试:`tests/test_fetcher_adjust.py`。
> **待办**:跑 `fetch-universe` 重建全市场缓存后,重跑全部 A/B 验证。

- **位置**:`src/stockpool/data_sources/mootdx_backend.py:85-97`(docstring 自认未做 xdxr 处理);`config.py:37` 与 `config.yaml` 默认 `source: mootdx`;`data/.data_source` 当前即 mootdx,4395 个 `*_daily.parquet` 全部是不复权价。
- **问题**:每个除权除息日是一个真实向下跳空。10 转 10 当日 "-50%",普通分红 1~5% 的虚假负收益每年系统性污染全市场几千个样本点。污染范围:技术指标(假死叉 / 假 breakout_new_low)、ML 标签(`forward_return_panel` 把除权日学成"暴跌样本")、回测 PnL、动量/反转类因子。与 `panel.py:96-107` 涨跌停 mask 还有隐蔽交互:除权幅度 >9.8% 被误判"跌停日"剔除,<9.8%(绝大多数分红)原样进入标签。`strategy_factory.py:200` 注释声称 "Cached close is 前复权" 与实情直接矛盾。
- **后果**:全市场 ML 训练、A/B 结论(P1..P4 verdict)、组合回测绝对数值都不可信;换数据源会换出一套不同的回测结论。
- **修复**:mootdx 路径接 `client.xdxr()` 做复权(推荐后复权,见 P0-2);或默认源切 baostock 并解决 P0-2;`validate_ohlcv` 把无复权标记的 >15% 缺口从 warning 升级为 error;回测报告头部标注数据源与复权口径。

### P0-2 前复权(qfq)+ 增量缓存追加 = 复权锚漂移,缓存永不因复权因子变化而失效 ✅ 已修复 (2026-06-11)

> **修复**:全链路统一后复权(历史不变,增量追加自洽);增量改为从缓存最后一天
> (含)重叠拉取,`_reconcile_increment` 接缝校验(close >0.1% 偏差 → 全量重拉)。

- **位置**:`fetcher.py:222-234`(增量 concat)、`fetcher.py:147`(akshare `adjust="qfq"`)、`baostock_backend.py:46`(`adjustflag="2"` 前复权);失效逻辑 `fetcher.py:62-64` 只看日期新鲜度。
- **问题**:前复权锚定最新收盘价,任何一次除权后**全部历史价格都会变**,但增量更新取 `start = last+1` 直接 concat——旧段锚在除权前、新段锚在除权后,拼接点出现等于分红/送转幅度的虚假跳变,之后整段历史不再自洽。源切换有 `.data_source` marker 防混用,但同源内复权基准漂移无任何检测。
- **后果**:换到 baostock/akshare 也无法逃脱 P0-1——只要持有缓存超过一个分红季,跨拼接点的收益率就是错的。
- **修复**:统一改后复权(hfq,锚在上市日,历史不变,增量追加天然自洽,回测收益率正确);或增量拉取时多拉最后 N 根与缓存重叠比对,不一致即全量刷新。

### P0-3 盘中运行把当日未走完的半根 K 线永久写入缓存,且永不修复 ✅ 已修复 (2026-06-11)

> **修复**:`_drop_in_progress_bar` 在 15:05 前丢弃当日 bar(股票/指数/板块路径);
> 合并改 `keep="last"`;重叠 bar volume 偏差 >1% 视为历史污染,触发全量重拉自愈。

- **位置**:`fetcher.py:62-64`(`_is_stale`)、`fetcher.py:226`(`start = last+1`)、`fetcher.py:229`(`drop_duplicates` 默认 `keep="first"`);`mootdx_backend.py:47-50` 只拦零成交占位行,拦不住有成交的盘中 bar。
- **问题**:10:30 跑 `stockpool run`,当日半成品 bar 写入 parquet;收盘后 `_is_stale` 判 False 不刷新;次日增量从 `last+1` 开始,半根 bar 永远不被完整版替换;即便日期重叠,`keep="first"` 仍保留旧值。**静默、不可逆的数据腐蚀**,每次盘中运行污染一根历史 bar。
- **修复**:盘中判定(交易日且当前 < 15:00)丢弃当日 bar;或增量从 `last` 当天开始并 `keep="last"` 覆盖。

### P0-4 幸存者偏差:训练 universe = 今天的存活名单,三层叠加

- **位置**:`mootdx_backend.py:162-182`(`list_a_shares` 拉当前在市名单)、`cli.py:710`(fetch-universe)、`fetcher.py:475-498`(`load_universe_cache` glob 全部 daily parquet)。
- **问题**:① 历史上已退市的票从不进入 universe;② ST 过滤按**当前**名称——2023 年健康、2026 年戴帽的票,其 2023 年截面样本也被整段剔除(用未来信息筛历史样本);③ `load_universe_cache` 不看名单文件,glob 目录全部 parquet——退市/戴帽票以冻结旧数据留在池里,与①②口径互相矛盾。
- **后果**:全市场 IC、ML 训练标签、组合回测系统性高估——踩雷退市/戴帽的负样本恰好都被剔掉;3 日 horizon 截面模型的分位数阈值(`strategies.py:712-717`)偏乐观。
- **修复**:baostock `query_stock_basic` 已含 `outDate`/`status`(`ipo_dates.py` 在用这张表却丢弃了这两个字段),用它构建含退市股的历史名单;ST 状态做成时变序列;短期至少在报告/文档明示该偏差。

### P0-5 组合回测 ml_factor 路径训练/预测特征不一致:`with_stock` 从未接线

- **位置**:`cli.py:463-469` 与 `portfolio_ab/runner.py:131-137`(`build_strategy` 未传 `current_stock_code`);`portfolio/scoring.py:41-43`(逐票直接 `generate_signals`,从不 `with_stock(code)`);`backtesting/strategies.py:536-546`(`current_stock_code is None` → 走单股退化路径)。
- **问题**:训练侧用预处理过的 T×N 因子面板(winsorize + cs_zscore + mcap 中性化);预测侧逐股重算**原始、单股**因子——cross-sec 因子(WQ101 rank 类、industry_relative)退化为常数/NaN,预处理全部缺失。模型在 z-score 分布上学的权重被拿去点乘原始量纲特征。`with_stock()` 在 src 下零调用(只有测试用)。
- **后果**:`portfolio-backtest` / `portfolio-ab` 在 ml_factor + cross-sec 因子配置下,score panel 的横截面排序基本失效;"训练池全市场跑 cross-sec 因子"的价值在组合层根本没兑现。
- **修复**:`precompute_scores_from_legacy` 中逐 code 调 `legacy.with_stock(code).generate_signals(daily)`;加回归测试断言预测用的 X 来自 `slice_stock_factor_matrix`。

### P0-6 因子清单在含评估期的全样本上选出 → in-sample 选择偏差

- **位置**:`factors_analysis.py:237-338`(全窗口逐日 IC)+ `pick_top_factors:341-386`;产物 `reports/selection.json` 被 `config.yaml` 引用。
- **问题**:`analyze_factors` 在全部 500 天历史上算 IC,`pick-by-ic` 据此从 ~165 个候选选 top-20,然后回测在**同一段 500 天**上评估。内层 walk-forward(Lasso/IC 权重)是诚实的,但外层因子清单用了评估期的未来信息——多重比较下纯噪声因子也会入选并在同段回测里"复现"。
- **后果**:`docs/ab_validation_results.md` 全部 A/B 结论的**绝对水平**不可信(相对比较受影响较小)。
- **修复**:selection 窗口截止在回测起点之前;或嵌套 walk-forward(定期重选因子);至少在文档把当前数字标注为 in-sample-selected。

---

## P1 — 显著扭曲收益 / 统计的问题

### P1-1 基本面因子 4/7 静默全 NaN:字段名对不上实际数据表(已实证)

- **位置**:`factors/fundamentals.py:103-110`(roa)、`:148-154`(revenue_yoy)、`:163-194`(pe)、`:203-223`(pb);静默逻辑 `:39-42`。
- **实证**(读 `data/fundamentals_*.parquet` schema):`roaAvg` 不在 profit 表;`YOYIncome` 不在 growth 表;`totalShare`/`totalShareholdersEquity` 不在 balance 表(`totalShare` 其实在 profit 表)。`_pit_align` 对缺失字段静默返回全 NaN panel,零日志。
- **连锁**:`stack_panel_to_xy(dropna=True)`(`ml/dataset.py:140`)丢任一因子为 NaN 的整行——把这三个因子任何一个加进 `ml_factor.factors`,**整个训练集变空**,策略静默输出全 neutral。当前 selection.json 恰好没选它们,问题潜伏。
- **修复**:pe/pb 的 `totalShare` 改从 profit 表取;roa 用 dupont 表 `dupontROA`;revenue_yoy 改 `YOYNI`/`YOYPNI` 或换源;`_pit_align` 对缺失字段 raise 或 `log.error`。

### P1-2 PE 的 TTM 算法错误:对"年初累计值"做 rolling(4).sum

- **位置**:`factors/fundamentals.py:180-188`。
- **实证**:baostock `netProfit` 是 YTD 累计(600519:2024Q1=249 亿 → FY=893 亿严格递增)。`rolling(4).sum()` 把 Q1+H1+9M+FY 加总 ≈ **2.5 倍年利润**,且倍数随披露节奏/盈利季节性逐季波动 → 截面排名失真(不只是绝对值)。另外按 `pubDate` 排序,数据里有 **6230 处同 code 同 pubDate 多 statDate**(年报+一季报同日披露是常态),窗口会重复计入同一季度。
- **修复**:TTM = 本期 YTD + 上年全年 − 上年同期 YTD(按 statDate 对齐去重);或直接用 profit 表现成的 `epsTTM × totalShare`。

### P1-3 涨跌停 / 停牌执行约束完全未建模(mask_exec 未落地)

- **位置**:`backtesting/framework.py:288-294, 317-319`(无条件按 `open[t]` 成交);`portfolio/engine.py:302-364` 同;`conventions.md:29` 的"除非涨停打不进"免责声明在代码里连这个"除非"也没实现。
- **偏差评估**:动量/趋势型信号与次日涨停高度正相关 → 买入端**正向选择偏差**;跌停日仍可卖出 → 止损端乐观、回撤被低估;组合 top-K 高分股恰是涨停概率最高的群体,top_k=20 + 5 日全换手下每期撞 1-2 只是常态。粗估动量类策略年化可虚高数个百分点。
- **修复**:执行层加拒单检查:`open[t] >= prev_close*(1+limit_pct)-ε` 时跳过买入,跌停同理延后卖出;limit_pct 按代码前缀推断(30/68 → 20%,ST → 5%)。这比完整 mask_exec 简单得多,能消掉最大头的偏差。

### P1-4 组合 rebalance"全清仓再买回":虚构双边换手 ~10%/年,且无换手指标

- **位置**:`portfolio/engine.py:311-321`(docstring 自认 fictitious churn 但把量级算成 "small ~0.1%")及 `:324-363`。
- **量级**:双边成本 ≈0.21%/轮 × 年约 50 次 rebalance(5 日周期)= 对**整个组合**(含本不该动的存活持仓)~10%/年的虚构成本拖累。同时 `PortfolioTrade` 全是 5 天碎片,`days_held`/胜率/单笔收益失真;**turnover 根本没有作为指标输出**;A/B 两 arm 参数不同时被污染方向不一致。
- **修复**:差量调仓(只卖 target 外、只买新进,存活仓做权重再平衡或不动);新增 turnover 指标。

### P1-5 退市 / 长期停牌持仓按入场价永久估值,尾部风险被抹掉

- **位置**:`portfolio/engine.py:127-131, 168-173, 324-329`。
- **问题**:close 缺失时兜底用的是**入场价**而非最后已知 close——涨 50% 后停牌一天的票当日市值被打回入场价,次日跳回,制造虚假波动污染 Sharpe/回撤;中途退市的票既卖不掉也不归零,期末强平 `ret ≈ -sell_cost`,现实中退市整理期 -90% 的尾部被完全抹掉。
- **修复**:维护 per-code `last_valid_close` 用于 mark;连续 N 天(如 60)无报价的持仓按最后价(或打折)强制核销,标 `exit_reason="delisted"`。

### P1-6 成交量单位跨源不一致:baostock 下流动性闸门失效 100 倍

- **位置**:`recommend_pool.py:173-174` 与 `portfolio/eligibility.py:66-69`(硬编码 `volume*close*100`,假定 mootdx 手);`baostock_backend.py:56-60` 的 volume 单位是**股**。
- **后果**:`source: baostock` 时 20 日均成交额高估 100 倍,`min_avg_amount_20d` 漏斗形同虚设,僵尸票涌入推荐池/组合 universe;反向切换则几乎全过滤。
- **修复**:在各 backend `_normalize` 统一 volume 为"股"(mootdx/akshare ×100),消费端去掉 ×100。

### P1-7 predict_latest(日报)与 generate_signals(回测)对同一天可能给不同信号

- **位置**:`backtesting/strategies.py:476-534`(日报:月度磁盘缓存,fit_date 同月即复用)vs `:585-591`(回测:`refit_every=20` bar 节奏)。
- **问题**:三个训练时点差异源(月初 vs 月中 fit、20-bar 节奏与月历错位、embargo 截断点随运行日漂移)→ pipeline 权重和 quantile 阈值不同 → 同一根 bar 日报 buy、回测 neutral 完全可能。这是实盘信号工具的命门,目前无任何测试或对账工具覆盖。
- **附带**:`cli.py:170-174` ml 策略失败时**静默回退 composite_verdict**,日报同表混排两种量纲的分数(ml ±0.0x vs composite ±10)。
- **修复**:加一致性契约测试(固定 fixture 下 `predict_latest == generate_signals 最后一行`);日报标注模型 fit_date;回退时在报告行明确标记策略名。

### P1-8 `detect_signals` 硬编码列名,配置改参数 → 信号无声消失

- **位置**:`signals.py:42`(ma5/ma20)、`:128`(rsi6)、`:154`(vol_ratio5);schema `config.py:57-64` 却允许任意 periods。
- **问题**:`volume_ratio_window: 10` → `curr.get("vol_ratio5", 1.0)` 静默回退 → 量价信号消失;`rsi_periods: [14]` → RSI 信号消失。无校验无告警。阈值(量比 1.5、J 值 20/80)与文案("20 日新高")也都不跟配置走。
- **修复**:`IndicatorsConfig` 加 model_validator 强制包含信号依赖的周期,或 `detect_signals` 从 cfg 动态取列名。

### P1-9 缓存键缺口两处:content_hash 不覆盖 factors_file;score panel 不含数据日期

- **位置**:`config.py:622-626`(hash yaml 原始字节)+ `config.yaml` 的 `factors_file`;消费方 `recommend_pool.py:79`、`cli.py:476`。`cli.py:474-490` 与 `portfolio_ab/runner.py:141-153` 的 score 缓存 key 仅 content_hash。
- **问题**:① 重跑 pick-by-ic 改写 selection.json 后 hash 不变 → Pool B 与 portfolio score panel 继续用旧因子组的缓存;反向地改个 yaml 注释就全量重算。② 数据更新后命中旧 score panel,新增日期 `predict_scores` 返回空 → 组合尾段**静默停止调仓**。附带:`config.py:484` 的 `factors_file` 相对 CWD 解析,换目录运行直接炸。
- **修复**:hash 用 resolved 后的 `cfg.model_dump()` 并纳入 factors_file 内容;score 缓存文件名加 last_date;factors_file 相对 config 文件目录解析。

### P1-10 mootdx 单次 800 根上限无分页:`history_days > 800` 静默截断 + 每次空拉

- **位置**:`mootdx_backend.py:19, 54-60`(`_offset_for_start(None)` 直接 800,分页循环未实现);`fetcher.py:215-219`。
- **后果**:回测窗口被静默缩短(实测缓存仅 813 行);`len(cached) < history_days` 永远为真 → 每次调用都触发无效网络拉取。`fetch_index`/`fetch_sector` 同病。
- **修复**:实现 offset 分页;或拿不满时 warning 并在缓存记录"已到头"。

---

## P2 — 统计口径 / 方法论 / 缓存 / 配置

### 统计与方法论

| # | 位置 | 问题 | 修复方向 |
|---|------|------|---------|
| P2-1 | `ml/weighters.py:155-178, 223-266` | ICWeighter 的"IC"是池化全样本 Spearman(时序+截面变异混杂),与选因子用的逐日截面 IC 不是同一统计量;IRWeighter 按行号分块,legacy 路径(stock-major 行序,`dataset.py:127-129`)下算的是"跨股稳定性"而非时间稳定性——同配置因缓存命中与否语义漂移 | 统一为"逐日截面 IC → 时间维聚合",按 date 分组,与行序解耦 |
| P2-2 | `ml/dataset.py:80`、`factors_analysis.py:282-298, 215-234` | horizon=3 + 逐日采样的重叠标签:ic_ir 的 std 低估 ~√3(`min_ir=0.05` 门槛形同虚设),half-life 被机械抬高,~25 万行训练集有效自由度只有 1/3 | IC 序列按 horizon 间隔抽样,或 ic_ir 做 lag-(h-1) Newey-West 修正 |
| P2-3 | `ml/dataset.py:80` vs T+1 open 执行 | 标签 close[t]→close[t+h] 含策略拿不到的隔夜跳空(A 股隔夜反转效应强),模型会把权重分给"只预测隔夜段"的因子,IC 好于实盘可实现部分 | 标签改 open[t+1]→open[t+1+h](加 build_open_panel),mask 逻辑不变 |
| P2-4 | `factors_analysis.py:272-273` vs `backtest_runner.py:94-98` | factors analyze 用 raw 因子 + 无 mask 标签选因子,生产训练用 preprocessed 因子 + mask 标签——mcap 中性化非秩不变,选出的排名与生产实际用的因子可能明显不同;无 mask 还偏向选"预测连板"的不可交易因子 | analyze 复用 `load_or_build_factor_panel(preprocess_cfg=...)` 并传 mask |
| P2-5 | `cross_sec_breadth.py:19-24` × `ml/preprocess.py:77-82` | 5 个 breadth 因子是全市场标量广播(每行常数),默认开启的 cs_zscore 对 σ≈0 的行整行置 0 → 加进 factors 列表后**恒为 0**,测的是空气,无告警 | 按因子 tag 跳过 zscore;退化行占比 >50% 时 warning |
| P2-6 | `strategy_factory.py:75` | `build_strategy` 内部回退路径建 factor_panel 时不传 `preprocess_cfg` → 拿到 raw 因子,且模型缓存 sig 含 preprocess 字段会跨路径串用(CLI 主路径都预建了所以暂未触发) | 补传 preprocess_cfg + maybe_inject_mcap_panel |
| P2-7 | `backtest_runner.py:59-68` 等 | 训练池资格判定用未来信息(与 P0-4 同源,ML 视角):回测期内退市/被 ST 的大负收益样本从训练集整段消失 | 同 P0-4 |

### 引擎与统计指标

| # | 位置 | 问题 | 修复方向 |
|---|------|------|---------|
| P2-8 | `framework.py:45, 293, 325, 543, 516`;`portfolio/engine.py:330-333` | `Trade.ret` 漏扣买入成本(buy_cost 出现在分母被约掉),与 docstring 和 `backtesting_framework.md:169` 矛盾;每笔 ret 虚高 0.08%,足以翻转贴近零的胜率统计(equity 曲线本身正确) | 修分母为买入前 equity,或改文档口径声明 |
| P2-9 | `portfolio/engine.py:145-149, 290-292` | top-K 选股窥视 `open[t+1]`(明日停牌信息),买不进时自动顺位补下一名——轻度 look-ahead + 乐观执行 | 用截至 t 的信息过滤;t+1 真停牌则该腿现金闲置 |
| P2-10 | `portfolio/engine.py:351-363` | `weight_at_entry` 分母用递减中的 cash:第 1 只记 1/n,最后一只恒记 ≈100%,归因分解失真(不影响 equity) | 循环前固定 total |
| P2-11 | `backtest.py:44-50` × `signals.py:79-87` | hit-rate 把双向复用的 signal_type 混进同一桶(direction 被最后一次触发覆盖)→ 日报"单信号命中率"在 6 类双向信号上是统计噪声;且 2 行窗口跑 detect 使 `macd_histogram_expand`(需 ≥4 行)永远不进命中率表但实时评级计分 | 桶 key 改 `(signal_type, direction)`;窗口对齐 |
| P2-12 | `strategies.py:548-610` × `metrics.py:46-48` | ml_factor 曲线含冷启动平头(min_train_samples 前全 neutral,可占 1/3-1/2),年化/Sharpe 被稀释;与 composite 曲线 A/B 时口径天然不齐 | metrics 增加 active-span 口径或并列两套数字 |
| P2-13 | `framework.py:539-551` × `strategies.py:932` | multi_lot 引擎 + ml_factor 连续 buy 信号 = 每根 bar 开新 lot 直到现金耗尽——隐式金字塔加仓,默认 multi_lot 下用户大概率没意识到,CLAUDE.md"无比例追加"与事实不符 | 加 `min_bars_between_entries` 或仅信号边沿开仓;报告标注开仓分布 |
| P2-14 | `ab/report.py:50-60` | A/B 汇总只取 `equity_curve_holding_days` 的第一个 N(配 [5,10,20] 时只反映 N=5,无提示);None→0 混入均值拉偏小样本统计 | 按 N 分组出表;None 跳过 |

### 数据与缓存

| # | 位置 | 问题 | 修复方向 |
|---|------|------|---------|
| P2-15 | `strategy_factory.py:290-326` | factor panel 缓存 sig 覆盖不全:`history_days` 改了 sig 不变(warmup 段错值)、industry_map 更新不失效(`custom.py:42-44` 自认)、`mcap_shares.parquet` 刷新不触发重建(manifest 只 glob `fundamentals_*`) | sig/manifest 纳入 first_date、industry_map mtime、mcap mtime |
| P2-16 | `fundamentals_loader.py:55-57` | 缓存键不含 codes 参数:先用小 codes 列表建缓存,30 天内全市场消费者命中子集,其余股票静默 NaN | 命中后校验 codes 覆盖率,不足则补拉 |
| P2-17 | `fundamentals_loader.py:124-125` | 仅拉"最近 16 季"且窗口随今天滚动:固定区间回测在不同月份运行时基本面覆盖不同,>4 年回测前段全 NaN 无告警 | 窗口与回测区间挂钩;覆盖不足告警 |
| P2-18 | `factors/fundamentals.py:53-56` | PIT 同日披露取舍方向错:`keep="last"` 在 `_recent_quarters` 降序下留下**较旧**报告期(年报+一季报同日披露留年报弃一季报,6230 处碰撞) | 排序键加 statDate,保留最新报告期 |
| P2-19 | `baostock_backend.py:12-23` × `fetcher.py:456` | baostock 全局单连接非线程安全,`fetch-universe --source baostock` 8 线程并发轻则报错、重则 A 股票拿到 B 的数据写进 A 的 parquet(mootdx 有 per-thread client,baostock 漏了) | 模块级锁或强制 max_workers=1 |
| P2-20 | mootdx(停牌无行)vs `baostock_backend.py:40-60`(volume=0 填充行) | 停牌日行语义跨源不一致,换源后时序指标输入形态不同;validate 只在 run 路径告警 | baostock 请求 tradestatus 并丢停牌行,统一"停牌无行" |
| P2-21 | `baostock_backend.py:54-55` × `fetcher.py:215-219` | 增量拉取无新数据时 raise:IPO 不足 history_days 的票每天触发空拉 → 重试 14 秒 → 抛异常 → 被剔出输出,样本无声缺失 | 增量场景空结果返回空 DataFrame |
| P2-22 | `strategy_factory.py:177-203`;`industry_map.py` | mcap = 前复权 close × **最新** totalShare 静态广播(轻度前视,且注释与 mootdx 不复权实情矛盾);行业映射是当前快照用于全历史 | profit 表已有逐季 PIT totalShare,用 `_pit_align` 即得;行业偏差至少文档明示 |
| P2-23 | `panel.py:96-107` | 训练标签涨跌停 mask 不区分 ST(±5%):手填池含 ST 时其涨跌停日低于 0.098 阈值不被 mask | mask 按 ST 状态分阈值 |

### 配置与防呆

| # | 位置 | 问题 | 修复方向 |
|---|------|------|---------|
| P2-24 | `config.py:87-99` | VerdictsConfig/ScoringConfig 零校验:`buy: 6, strong_buy: 3` 时 buy 区间被吞掉,静默错误;权重和、resonance_bonus 均无约束(对比 QuantileThresholds:337-345 有同类校验) | 补 model_validator |
| P2-25 | `config.py:597-614` | 顶层 7 个 Config 未设 `extra="forbid"`(ml_factor 子树全设了):yaml 写 `scorings:` 等 typo 静默走默认值 | 统一 `extra="forbid"` |
| P2-26 | `cli.py:1013-1047` | `--refresh-fundamentals` 是死参数:全仓库无消费点,`force_refresh` 形参无人传 True;对应测试只测 argparse 接受 flag | 穿透到 loader + 行为级测试 |
| P2-27 | `config.yaml` + `signals.py` | 综合评级权重/阈值无数据驱动来源(对比 ml 路径有完整 IC 工具链);`resample_to_weekly`(`fetcher.py:501-513`,W-FRI)含进行中不完整周 → 周线信号 repaint,而共振奖励 +2 占 buy 阈值 3 的 2/3,放大该噪声 | 周线只用已完成周;用现成 verdict_bucket_stats 做权重敏感性校准并记录 |
| P2-28 | `recommend_pool.py:246-282` | Pool B 跨周无 buffer/hysteresis,边界票每周进出 → 按池操作换手高;composite 路径 final_score 大量并列,并列序由 glob 文件序决定,top-30 边界实质随机 | 池粘性规则("已在池内降到 top-1.5N 保留");次级 tiebreak |
| P2-29 | `pyproject.toml:10-21` | 依赖全 `>=` 下界,无 lockfile;akshare 是爬虫库,小版本就可能改列名(LGB/Lasso 种子与确定性本身没问题) | 提交 lockfile;CI 固定 Python 小版本 |
| P2-30 | `backtesting/strategies.py:347-349` × `recommend_pool.py:205-210` | Pool B 打分循环每股深拷贝 close_panel(~17MB × 4000 股 ≈ 70GB 分配流量),纯浪费 | 共享只读引用;循环复用单 strategy + with_stock |

---

## P3 — 轻微 / 边界 / 工程卫生

| # | 位置 | 问题 |
|---|------|------|
| P3-1 | `fetcher.py:55-59` | `_last_business_day` 不识别中国节假日,长假期间每次运行全 universe 空拉(叠加 P2-21 放大为 14 秒/股) |
| P3-2 | `fetcher.py:33-48` | `_no_proxy` 全局 monkeypatch `requests.get`,并发嵌套 patch/restore 可能永久留下 patched 版本 |
| P3-3 | `fetcher.py:203` | 每次 fetch_daily 都写 `.data_source` marker,8 线程并发在 Windows 上可能 PermissionError |
| P3-4 | `mootdx_backend.py:176-178` | ST 过滤依赖双重编码乱码名称里的 ASCII "ST",可能误杀/漏杀 |
| P3-5 | `ops.py:35-46, 92-96` | `ts_sum`/`ts_product` 放宽 min_periods 到 60% 但不重标定:alpha_019/039/052 对历史短/停牌多的股票部分和系统性偏小,rank 后秩被结构性压低 |
| P3-6 | `ops.py:61-77` | `ts_argmax/argmin` 方向约定与主流复现相反(alpha_001/057/060/096/098/100 符号翻转;IC 加权自动纠符号,无实际损失,但与文献对比时会反) |
| P3-7 | `ops.py:174-181` | `indneutralize` 把无行业映射的股票置 0(假"完美中性")而 `custom.py:67-73` 同情形置 NaN,两套语义不一致 |
| P3-8 | `custom.py:100` | `limit_up_count` 统一 0.099 阈值,创业板 300/301(±20%)的普通大涨被计为涨停;`panel.py` 有正确分板逻辑未复用 |
| P3-9 | `strategies.py:527,599` | predict 时 `fillna(0.0)` 填原始空间的 0:关掉 zscore 后 rank 类因子 μ≈0.5,缺因子即注入方向偏置,应填 fit-time 均值 |
| P3-10 | `preprocess.py:172-174` | mcap 中性化对 mcap 缺失格保留原始值,与同行残差混截面,rank/Lasso 对无 mcap 股有结构偏置 |
| P3-11 | `strategies.py:815-818` vs `:884-888` | 非 sharing 模式下 fast/legacy 路径对 host 股的训练集处理不一致(剔除 vs 截断加回),同配置训练集随缓存可用性而不同 |
| P3-12 | `selectors.py:90-99` | Lasso `alpha=0.001` 是绝对量,与典型相关量级相同 → 选择数对市场波动状态敏感;选空时静默回退全因子(`pipeline.py:53-57`)无告警 |
| P3-13 | `fundamentals.py:61` | `_pit_align` 的 ffill 无时限,停止披露的公司财务值被无限沿用(应设 ~2 季上限) |
| P3-14 | `indicators.py:62-64` | RSI 在 avg_loss==0 时 fillna(50),应为 100 |
| P3-15 | `metrics.py:46-48, 74-81` | 回测期 <60 天时年化爆炸(10 天 +5% → +242%)无警示;无交易时 win_rate=0.0 而非 None |
| P3-16 | `framework.py:596-609, 474, 541` | B&H 曲线锚 open[0] 但 total_return 退化为 close→close,图表口径不一致;多 lot 仓位按**起始资金**固定比例不复利,长周期与单仓引擎不可比 |
| P3-17 | `sizing.py:21-30` | LotSizer 收到完整 opens/closes 数组,look-ahead 只靠 docstring 约束,自定义 sizer 易踩雷(应只传 `[:bar_idx]` 切片) |
| P3-18 | `ab/runner.py:130-137` | `--refresh` + 盘中 mootdx 时两 arm 可能拿到不同的最后一根 bar(应统一拉一次注入两 arm) |
| P3-19 | `backtest_composite.py:96-106` | verdict_bucket_stats 用信号 bar close→close 前瞻收益,与引擎次日 open 成交口径不一致,易被误读 |
| P3-20 | `report.py:279-283` | 报告公式硬编码 ×0.7/×0.3,不读 `cfg.scoring`,改配置后报告说谎 |
| P3-21 | `cli.py:52-68` vs `strategies.py:126-149` | `_compute_verdict` 与 CompositeVerdictStrategy.predict_latest 复制粘贴,改权重要改两处 |
| P3-22 | 仓库根 | `CLAUDE.md.bak`/`README.md.bak`/`debug.log`/12 个 ab*.yaml 实验残留;`cmd_run` 234 行单函数 + 大量 `print("[TIME]")` 与 logging 双轨;Pool B 单股失败只 log.debug(4000 股全挂也只见 ok=0) |
| P3-23 | `docs/reference/testing.md` | 写 615 个测试,实际 ~677,文档计数漂移;`framework.md` 的 Trade.ret 描述与实现不符(见 P2-8) |

---

## 测试盲区

现有 ~677 用例覆盖面整体不错(config / factors / 引擎 / portfolio / A/B 均有),但缺:

1. **端到端黄金值测试**:无"固定 OHLCV fixture → 完整回测 → 断言精确 equity/metrics"的回归锚点,引擎重构只能靠局部单测兜底。
2. **predict/backtest 一致性测试**(P1-7)——项目作为实盘信号工具的核心契约,零覆盖。
3. **非默认 indicators 配置测试**:P1-8 一个参数化测试就能抓住。
4. **数据层关键场景**:盘中半根 bar 污染(P0-3)、qfq 增量接缝(P0-2)、源切换×量纲组合(P1-6)。
5. **行为级 CLI 测试**:`test_cli_refresh_fundamentals.py` 只测 argparse 接受 flag,不测行为(P2-26 的死参数因此漏网)。
6. **因子覆盖率断言**:4 个基本面因子全 NaN 一年没被发现(P1-1),说明缺"因子有效值覆盖率 < x% 即 fail"的总闸测试。

---

## 改进路线图

### 阶段一:修数据地基(最高优先级,其余一切优化的前提)

| 任务 | 对应问题 | 预估 |
|---|---|---|
| ✅ 复权统一(已完成 2026-06-11):全链路 hfq,mootdx 段内锚定 xdxr 复权,增量重叠接缝校验 | P0-1, P0-2 | 2-3 天 |
| ✅ 盘中半根 bar(已完成 2026-06-11):15:05 前丢当日 bar + keep="last" + volume 接缝自愈 | P0-3 | 0.5 天 |
| volume 单位在 backend 层归一为"股",消费端去 ×100 | P1-6 | 0.5 天 |
| 基本面字段修复(pe/pb/roa/revenue_yoy 换正确表)+ TTM 公式重写 + `_pit_align` 缺字段 fail loud | P1-1, P1-2, P2-18 | 1 天 |
| mootdx 800 根分页 | P1-10 | 0.5 天 |
| **完成后:重跑全部 A/B 验证,在 ab_validation_results.md 标记 before/after** | — | 1 天 |

### 阶段二:方法论诚实化

| 任务 | 对应问题 | 预估 |
|---|---|---|
| selection 窗口前移到回测起点之前(改流程即可);中期做嵌套 walk-forward 定期重选因子 | P0-6 | 0.5 天起 |
| 退市/ST universe PIT 化:用 baostock `query_stock_basic` 的 outDate/status 建历史名单;短期先在报告明示偏差 | P0-4, P2-7 | 2-3 天 |
| 组合 ml_factor 接线 `with_stock` + 回归测试 | P0-5 | 1 天 |
| 标签改 open[t+1]→open[t+1+h](加 build_open_panel),与执行对齐 | P2-3 | 1 天 |
| IC 统计量统一:weighter 改逐日截面 IC,IR 按时间窗聚合,与行序解耦 | P2-1 | 1 天 |
| 重叠标签的 ic_ir Newey-West 修正;analyze 复用 preprocessed panel + mask | P2-2, P2-4 | 1 天 |

### 阶段三:执行真实性(组合回测拿到"绝对数字可讨论"资格)

| 任务 | 对应问题 | 预估 |
|---|---|---|
| 涨跌停拒单(一字板买不进/卖不出,按代码前缀推断 limit_pct) | P1-3 | 1-2 天 |
| rebalance 改差量调仓 + 输出 turnover 指标 | P1-4 | 1-2 天 |
| 退市/长停持仓:last_valid_close mark + N 天无报价强制核销 | P1-5 | 1 天 |
| Trade.ret 扣买入成本;top-K 去掉 opens_next 窥视 | P2-8, P2-9 | 0.5 天 |
| 成本模型补最低佣金 5 元 + 资金规模适用声明 | P2 | 0.5 天 |

### 阶段四:工程防呆与可信度护栏

| 任务 | 对应问题 | 预估 |
|---|---|---|
| predict/backtest 一致性契约测试 + 日报标注 fit_date + 回退显式标记 | P1-7 | 1 天 |
| 端到端黄金值回归测试(固定 fixture → 精确 metrics) | 测试盲区 1 | 1 天 |
| 配置收口:全树 `extra="forbid"`、Verdicts/Scoring 排序校验、indicators↔signals 列名校验 | P1-8, P2-24, P2-25 | 1 天 |
| content_hash 改 resolved-config(含 factors_file 内容);score 缓存加 last_date;factor panel sig 补 first_date/industry/mcap mtime | P1-9, P2-15 | 1 天 |
| 因子覆盖率总闸:compute_factor_panel 对有效值覆盖率 < x% 的因子 raise/warn | P1-1 类事故 | 0.5 天 |
| `--refresh-fundamentals` 接线 + 行为测试;清理 .bak/debug.log/根目录 ab 配置;print→logging | P2-26, P3-22 | 0.5 天 |
| 提交依赖 lockfile | P2-29 | 0.5 天 |

### 长期优化方向(地基修好之后)

1. **执行模型深化**:完整 mask_exec(盘中触及涨跌停的部分成交概率)、冲击成本模型、按资金规模的整手约束模拟。
2. **PIT 数据全面化**:行业分类历史版本、逐季 PIT 股本(profit 表已有数据,接上即可)、退市股价格序列回补。
3. **组合层进阶**:换手惩罚 / 进出池 buffer(P2-28)、行业/风格风险约束的组合优化(当前只有等权 + 行业 cap)、staggered ensemble 并行化(当前串行 10×耗时)。
4. **综合评级路径校准**:用现成的 verdict_bucket_stats 做权重网格敏感性分析,把"拍脑袋权重"升级为数据驱动,结论进 ab_validation_results;周线只用已完成周消除 repaint。
5. **监控与运维**:因子覆盖率 / IC 衰减 dashboard,日报与实盘信号对账命令(回放昨日报告逐票比对 generate_signals)。
6. **统计严谨性**:A/B 加 bootstrap 置信区间(样本小不做 t 检验是对的,但 bootstrap 可行);多 N 持有期分组报表(P2-14)。

### 总优先级一句话

**先让价格序列正确(阶段一),再让样本与标签诚实(阶段二)——这两步之前产出的任何回测数字都只宜用于策略间相对排序;阶段三之后组合回测的绝对数字才值得讨论;阶段四把"静默失败"类事故从根上消灭。**
