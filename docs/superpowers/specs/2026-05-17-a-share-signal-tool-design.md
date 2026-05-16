# A 股养龙股池技术信号分析工具 — 设计文档

- **状态**:Draft(等待用户最终确认后进入实现规划)
- **日期**:2026-05-17
- **作者**:Claude(与用户共同 brainstorm)
- **目标读者**:本工具的实现者(下一步由 writing-plans skill 转为实施计划)

---

## 1. 项目概述

为用户的"养龙股池"提供**每日技术信号分析**:从配置文件读取 A 股代码列表,自动拉取行情、计算技术指标、产出每只股票的综合打分(买入/卖出/观望),并生成单页交互式 HTML 报告。

**不在 v1 范围内**:基本面分析、消息面、资金流、形态识别、顶/底背离、组合回测、自动下单。

---

## 2. 锁定需求

| 维度 | 决策 |
|------|------|
| 股池来源 | `config.yaml`,代码+名称对,初始 8 只,上限 100 |
| 数据源 | AKShare(免费,免注册) |
| 指标集 | 全套:MA / MACD / KDJ / RSI / BOLL / 量能 / 价格突破 |
| 信号产出 | 综合打分 `[-10, +10]` + 触发明细 |
| K 线周期 | 日 K + 周 K(共振合成,周 K 权重 0.3) |
| 历史窗口 | 500 个交易日 |
| 输出形式 | 交互式 HTML(pyecharts),按日期归档 `reports/YYYY-MM-DD/index.html` |
| 运行模式 | 命令行手动 + Windows 计划任务(收盘后)定时,同一命令兼容两种触发 |
| 回测 | 信号命中率统计:对历史每次触发,统计 5/10/20 日后平均涨跌幅 + 胜率 |

**初始股池(写入默认 config):**

| 名称 | 代码 |
|------|------|
| 圣泉集团 | 605589 |
| 兆易创新 | 603986 |
| 柳工 | 000528 |
| 石大胜华 | 603026 |
| 大元泵业 | 603757 |
| 五洲新春 | 603667 |
| 金螳螂 | 002081 |
| 华电辽能 | 000922 |

---

## 3. 架构方案选型

考虑过三种:

1. **单文件脚本**:200 行写完,但难扩展、难测试。**淘汰**。
2. **模块化包 + 本地缓存(选定)**:数据/指标/信号/回测/报告分层,纯函数指标层易测,Parquet 缓存避免重复拉数据。
3. **套用 backtrader / qlib / vnpy 等框架**:杀鸡用牛刀,A 股复权/T+1 适配反而更慢。**淘汰**。

---

## 4. 数据流

```
config.yaml (股池+权重)
        │
        ▼
   fetcher.py        ── 检查 data/{code}_daily.parquet 最新日期
                    ── 过期则 AKShare 增量拉取
                    ── 周 K 由日 K resample('W'),不二次请求
        │
        ▼
   indicators.py     ── 纯函数:DataFrame → 附加列
                       (ma5/10/20/60, macd_*, kdj_*, rsi_*, boll_*, vol_ratio5)
        │
        ▼
   signals.py        ── 扫最新一根 K:输出 (signal_type, direction, weight)
                    ── 综合打分,日+周共振合成
        │
        ├──→ backtest.py   历史每次触发 → 5/10/20 日涨跌 + 胜率统计
        └──→ report.py     pyecharts 渲染单页 HTML
                                │
                                ▼
                        reports/YYYY-MM-DD/index.html
```

**关键约束:**
- `indicators.py` 是纯函数,不碰 IO
- `signals.py` 接收 DataFrame,不读文件
- 周 K 由日 K 重采样,避免被 AKShare 限频
- 缓存格式 Parquet(比 CSV 小 5-10 倍,保留 dtype)

---

## 5. 综合打分规则

### 5.1 单指标触发表

| 指标 | 强信号(±2) | 弱信号(±1) |
|------|-------------|-------------|
| MA | MA5 上/下穿 MA20 | 多头/空头排列(MA5>10>20>60 或反之) |
| MACD | 零轴上方金叉 / 零轴下方死叉 | 零轴下方金叉 / 零轴上方死叉;红/绿柱连续 3 日放大 |
| KDJ | J<20 时金叉 / J>80 时死叉 | 普通金叉/死叉 |
| RSI | RSI6<20 超卖 / RSI6>80 超买 | — |
| BOLL | 收盘上穿下轨 / 跌破上轨 | 突破/跌破中轨 |
| 量能 | 量比>1.5 且阳线 / 量比>1.5 且阴线 | — |
| 突破 | 收盘创 20 日新高 / 20 日新低 | — |

单根 K 线触发分截断到 `[-10, +10]`。

### 5.2 日 + 周共振合成

```
final_score = 0.7 × daily_score + 0.3 × weekly_score

共振加成:
  若 daily ≥ +3 且 weekly ≥ +1 → final += 2
  若 daily ≤ -3 且 weekly ≤ -1 → final -= 2

最终截断到 [-10, +10]
```

周 K 给小权重的设计意图:**作为方向过滤器**,避免日线一根反弹就喊买。

### 5.3 判定阈值

| final_score | 标签 | 颜色 |
|-------------|------|------|
| ≥ +6 | 🟢🟢 强烈买入 | 深绿 |
| +3 ~ +5 | 🟢 买入观察 | 浅绿 |
| -2 ~ +2 | ⚪ 观望 | 灰 |
| -5 ~ -3 | 🔴 卖出观察 | 浅红 |
| ≤ -6 | 🔴🔴 强烈卖出 | 深红 |

### 5.4 v1 主动不做

- 顶/底背离(MACD/RSI):写对很难、误触发多,留 v1.1
- 形态识别(头肩底、杯柄):工程量翻倍
- 板块/资金面:超出"纯技术"范围

---

## 6. HTML 报告结构

单个独立 HTML,**所有 pyecharts JS 内联,离线可看**。

### 6.1 页面布局

1. **页眉**:日期、扫描数、各档计数(🟢🟢/🟢/⚪/🔴/🔴🔴)、所用 config 文件路径 + 内容 hash(便于追溯报告是哪套权重产出)
2. **总览表**:按 `final_score` 降序,列 = 代码 / 名称 / 日分 / 周分 / 终分 / 主要触发摘要 / 标签彩条
3. **单股详情区**(每只一节,折叠 `<details>`,顶部目录锚点跳转):
   - 主图:日 K(默认 120 根)+ MA5/10/20/60 + BOLL 通道
   - 副图 1:成交量柱(红绿)+ 5 日均量线
   - 副图 2:MACD 柱状 + DIF/DEA
   - 副图 3:KDJ
   - 副图 4:RSI6/12
   - 4 副图与主图共享缩放(pyecharts `Grid` + `DataZoom`)
   - 触发信号明细列表(日 K + 周 K 分开列)+ 计算过程
   - 历史命中率小表(本股近 500 日内每类信号的 5/10/20 日表现)
4. **附录**:本次使用配置摘要、AKShare 接口/数据更新时间、免责声明

### 6.2 其他约定

- 报告生成无状态:同日重跑覆盖当日,带 `--keep-history` 则加时间戳后缀
- `reports/latest.html`(Win 下复制最新一份)永远是最新报告,任务栏快捷方式固定它
- 100 只股票仍在单页内,通过折叠 + 锚点导航不会卡

---

## 7. 配置文件 `config.yaml`

```yaml
stocks:
  - {code: "605589", name: "圣泉集团"}
  - {code: "603986", name: "兆易创新"}
  - {code: "000528", name: "柳工"}
  - {code: "603026", name: "石大胜华"}
  - {code: "603757", name: "大元泵业"}
  - {code: "603667", name: "五洲新春"}
  - {code: "002081", name: "金螳螂"}
  - {code: "000922", name: "华电辽能"}

data:
  history_days: 500
  cache_dir: "data"
  force_refresh: false

indicators:
  ma_periods: [5, 10, 20, 60]
  macd: {fast: 12, slow: 26, signal: 9}
  kdj: {n: 9, m1: 3, m2: 3}
  rsi_periods: [6, 12, 24]
  boll: {n: 20, k: 2}
  volume_ratio_window: 5
  breakout_window: 20

weights:
  ma_cross_strong: 2
  ma_alignment: 1
  macd_cross_above_zero: 2
  macd_cross_below_zero: 1
  macd_histogram_expand: 1
  kdj_oversold_cross: 2
  kdj_overbought_cross: 2
  kdj_normal_cross: 1
  rsi_oversold: 1
  rsi_overbought: 1
  boll_band_touch: 2
  boll_mid_cross: 1
  volume_surge_bullish: 1
  volume_surge_bearish: 1
  breakout_new_high: 2
  breakout_new_low: 2

scoring:
  daily_weight: 0.7
  weekly_weight: 0.3
  resonance_bonus: 2
  resonance_daily_threshold: 3
  resonance_weekly_threshold: 1

verdicts:
  strong_buy: 6
  buy: 3
  sell: -3
  strong_sell: -6

backtest:
  forward_days: [5, 10, 20]

report:
  output_dir: "reports"
  keep_history: true
  klines_to_show: 120
```

启动时做 schema 校验,任何字段缺失/类型错误立即报错并指出位置,不会跑到一半才崩。

---

## 8. 项目目录

```
stockpool/                  # 项目根(即 cwd)
├── config.yaml             # 用户唯一需编辑的文件
├── data/                   # Parquet 缓存(.gitignore)
├── reports/YYYY-MM-DD/     # 每日报告归档
├── reports/latest.html     # 始终是最新报告(脚本结束时复制)
├── src/
│   └── stockpool/
│       ├── __init__.py
│       ├── fetcher.py
│       ├── indicators.py
│       ├── signals.py
│       ├── backtest.py
│       ├── report.py
│       ├── config.py       # 加载 + 校验 yaml
│       └── cli.py          # 入口
├── tests/
│   ├── test_indicators.py
│   ├── test_signals.py
│   ├── test_backtest.py
│   ├── test_fetcher.py
│   ├── test_config.py
│   ├── test_report_smoke.py
│   └── fixtures/
├── pyproject.toml          # 依赖 + 入口脚本声明
├── README.md
└── docs/superpowers/specs/2026-05-17-a-share-signal-tool-design.md
```

**运行方式:**

```bash
python -m stockpool run              # 完整流程:拉数据→分析→生报告
python -m stockpool run --refresh    # 忽略缓存,全量重拉
python -m stockpool run --stocks 605589,603986   # 只跑指定股票
python -m stockpool backtest         # 只产出历史命中率表(不画 K 线图)
pytest tests/                        # 全套测试,目标 <5 秒
```

Windows 计划任务示例:周一至周五 15:30 触发 `python -m stockpool run`。脚本启动时通过 AKShare 的交易日历(`tool_trade_date_hist_sina`)判断当日是否为交易日,**非交易日直接 exit 0 不生成新报告**(避免节假日污染报告归档)。

---

## 9. 错误处理

| 故障 | 处理 | 用户感知 |
|------|------|---------|
| AKShare 限频/网络失败 | 指数退避重试 3 次(2s→4s→8s),失败则用缓存 | 该股标 ⚠️"数据截止 YYYY-MM-DD",其他股正常 |
| 股票代码不存在/已退市 | 跳过,记日志 | 报告底部"跳过股票"区块列出原因 |
| 新股历史 < 60 日 | 长周期指标(MA60)不参与打分 | 该股打分附 "*" + 注释"历史不足" |
| 配置文件缺字段/类型错 | 启动时 schema 校验失败立即退出 | 报错指出哪个字段哪个类型 |
| Parquet 缓存损坏 | 删除该文件 + 重新全量拉 | 仅该股慢一次 |
| pyecharts 渲染异常 | 单股渲染失败不影响其他股,退化为纯表格 | 该股显示"图表生成失败,仅展示信号文字" |
| 计划任务运行时无网 | 退出码 0,报告标题加"⚠️ 离线 fallback" | 第二天恢复 |

**原则:任何单股故障不影响其他股,日志写到 `reports/YYYY-MM-DD/run.log`。**

---

## 10. 测试策略

不追求覆盖率,追求"改了不出错":

- **`test_indicators.py`**:喂固定 20 根 K 的 DataFrame,断言 MA/MACD/KDJ 等列具体数值
- **`test_signals.py`**:构造已知指标值,断言触发哪些信号、打分多少
- **`test_backtest.py`**:合成 K 线(已知植入 N 次金叉),断言命中率统计正确
- **`test_fetcher.py`**:mock AKShare,断言"只拉缺失日期"的缓存逻辑
- **`test_config.py`**:缺字段/错类型都能被发现
- **`test_report_smoke.py`**:1 只股完整流程跑完,HTML 含关键字符串
- **`fixtures/`**:1 只票的真实历史快照 + 一组"已知答案"合成用例

目标:全套 `pytest` <5 秒,**CI 不依赖网络**。

---

## 11. 依赖清单

| 包 | 用途 |
|---|------|
| akshare | A 股数据 |
| pandas | DataFrame |
| numpy | 数值计算 |
| pyarrow | Parquet 读写 |
| pyecharts | 交互式图表 |
| pyyaml | 配置加载 |
| pydantic | 配置 schema 校验 |
| pytest | 测试 |

Python 版本:**3.10+**(为了较新类型注解和 `match` 等可选语法)。

---

## 12. 实施里程碑(粗粒度,详细计划由 writing-plans 拆)

1. 项目脚手架 + config 加载 + AKShare fetcher + 缓存
2. indicators 全套 + 单元测试
3. signals 触发 + 综合打分 + 单元测试
4. backtest 命中率统计
5. report HTML 生成(pyecharts)
6. CLI 串接 + 端到端 smoke 测试
7. README + 计划任务示例

---

## 13. 免责声明(写入报告页脚)

> 本工具产出基于公开行情数据的技术指标计算,信号与打分仅供个人技术分析参考,**不构成任何投资建议**。使用者应自行承担交易决策的全部责任。
