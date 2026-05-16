# stockpool — A 股养龙股池技术信号分析

每日扫描配置文件中的 A 股池,产出综合打分 + 交互式 HTML 报告。

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
.venv/Scripts/python -m pytest                                   # 全套单元测试
```

## 加股票

打开 `config.yaml`,在 `stocks:` 列表里追加一行即可:

```yaml
stocks:
  - {code: "600519", name: "贵州茅台"}
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
- `data/{code}_daily.parquet` — 行情缓存(可删除,下次自动重建)

## ⚠️ 免责声明

本工具产出基于公开行情数据的技术指标计算,信号与打分仅供个人技术分析参考,
**不构成任何投资建议**。
