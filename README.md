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
.venv/Scripts/python -m pytest                                   # 全套单元测试
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

### 数据质量检查

每次拉取后自动检测:
- 停牌K线(成交量为0)
- 单日涨跌幅 >20%(可能含数据异常)
- 日期间隔 >7 天(疑似长期停牌或数据缺失)

检测到问题时在报告中以橙色警告框显示。

## ⚠️ 免责声明

本工具产出基于公开行情数据的技术指标计算,信号与打分仅供个人技术分析参考,
**不构成任何投资建议**。
