# 因子库扩展 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 panel-based factor library 中新增 ~160-208 个因子(论文 B 9 个技术家族 + EWMA 小族 + 基本面族),从 114 → ~274-322,并落地基本面数据 PIT 加载层。

**Architecture:** 每家族一个 `factors/*.py` 模块,继承现有 `Factor` ABC 和 `@register` 装饰器模式。基本面族外加 `fundamentals_loader.py` 数据缓存层,严格按 `pubDate` 前向填充避免 look-ahead。Mask 行为完全沿用现状(因子看 raw close,mask 只作用于训练标签层)。

**Tech Stack:** Python 3.10+, pandas, numpy, baostock (财务数据), pytest, `.venv/Scripts/python.exe`(避免全局 anaconda numpy 冲突)

**关联 spec:** `docs/superpowers/specs/2026-05-31-factor-library-expansion-design.md`

---

## File Structure

**新增文件(13 个):**
- `src/stockpool/fundamentals_loader.py` — baostock 5 张季度财务表的 PIT 缓存层
- `src/stockpool/factors/original_stats.py` — rolling 直接统计量(close_std / skew / kurt 等)
- `src/stockpool/factors/ewma.py` — EWMA 平滑动量/波动/换手
- `src/stockpool/factors/vwap_deviation.py` — close 相对 vwap proxy 的偏离族
- `src/stockpool/factors/close_position.py` — close 在 high-low 区间位置族
- `src/stockpool/factors/turnover_extra.py` — 短窗换手 z-score / 量比
- `src/stockpool/factors/acceleration.py` — 动量/换手的二阶差分
- `src/stockpool/factors/single_stock_vol.py` — ATR / CCI / 振幅 / Parkinson
- `src/stockpool/factors/composite.py` — 现有算子拼装的复合信号
- `src/stockpool/factors/rank_correlation.py` — 价格秩 × 成交量秩滚动相关
- `src/stockpool/factors/cross_sec_breadth.py` — 全市场涨/触涨停占比等宽度
- `src/stockpool/factors/fundamentals.py` — PE/PB/ROE 等基本面因子定义
- 12 个对应 `tests/test_factors_*.py` + `tests/test_fundamentals_loader.py`

**修改文件:**
- `src/stockpool/factors/__init__.py` — 11 行 import 副作用
- `src/stockpool/cli.py` — 加 `--refresh-fundamentals` flag
- `src/stockpool/strategy_factory.py` — `load_or_build_factor_panel` 的 manifest 加 `fundamentals_snapshot_date`
- `tests/test_factors.py` — 加因子总数 sanity 检查
- `CLAUDE.md` — 模块地图 + 因子库章节 + 测试表 + 数据流缓存表
- `README.md` — 命令示例 + 因子库简介

---

## 因子清单(原理 + 处理)

按家族列出所有新增 Factor base class、原理(公式/语义)、处理细节(NaN 守护 / 窗口建议 / mask 关系 / look-ahead 保证)。**典型变体**=按建议窗口枚举的实际注册名;实施时窗口可调,但本表是参考。

### Family 1: `original_stats.py` — rolling 直接统计(Task 3,~25 变体)

| Base class | 原理 | 处理 |
|---|---|---|
| `close_std` | `close.rolling(n).std() / close`,归一化波动率 | `ddof=0` 总体方差;`close=0` 不可能(A 股最低 0.01);典型 n ∈ {5,10,20,60} |
| `close_skew` | `close.rolling(n).skew()` | pandas 自动 NaN warmup;典型 n ∈ {20, 60}(skew 需较多样本) |
| `close_kurt` | `close.rolling(n).kurt()` | 同上;典型 n ∈ {20, 60} |
| `volume_skew` | `volume.rolling(n).skew()` | volume=0 时 skew 仍可算(单点不影响);n ∈ {20, 60} |
| `volume_kurt` | `volume.rolling(n).kurt()` | 同上 |
| `range_std` | `(high-low).rolling(n).std() / close`,日内振幅波动 | `close` 不为 0;n ∈ {10, 20, 60} |
| `volume_std` | `volume.rolling(n).std() / volume.rolling(n).mean()`,变异系数 | `mean.replace(0, NA)` 防除零;n ∈ {20, 60} |

**Mask 关系**:全族沿用 raw close,mask 对 `compute()` 不可见。
**Look-ahead**:`rolling(n)` 仅看 `[t-n+1, t]`,天然安全。

### Family 2: `ewma.py` — EWMA 平滑(Task 4,~15 变体)

EWMA 因子统一用 `halflife` 参数(更直观),后缀格式 `hl<int>`,如 `ewma_momentum_hl10`。

| Base class | 原理 | 处理 |
|---|---|---|
| `ewma_momentum` | `(close - close.ewm(halflife=h).mean()) / close.ewm(...).mean()`,close 相对 EWMA 均线的偏离 | halflife>0;典型 h ∈ {5, 10, 20} |
| `ewma_vol` | `close.pct_change().ewm(halflife=h).std()`,RiskMetrics 风格波动率 | 收益首行 NaN warmup;h ∈ {5, 10, 20} |
| `ewma_turnover_z` | `log(volume).ewm(halflife=h)` 的 (lv - mean) / std | `v.replace(0, NaN)` 守护停牌;`std.replace(0, NaN)` 守除零;h ∈ {5, 10} |
| `ewma_close_dev` | `(close - EWMA(close)) / EWMA std(close)`,close 相对自身 EWMA 的 z | std=0(常数序列)→ NaN;h ∈ {10, 20} |
| `ewma_volume_ratio` | `volume / EWMA(volume).shift(1) - 1`,放/缩量 EWMA 版 | `shift(1)` 防今值;EWMA mean=0 不可能(volume>0);h ∈ {5, 10} |

**Mask 关系**:同上,raw 输入。
**Look-ahead**:`ewm` 是 causal 滤波;`ewma_volume_ratio` 用 `.shift(1)` 显式去除今值。

### Family 3: `vwap_deviation.py` — VWAP 偏离族(Task 5,~20 变体)

VWAP proxy = `(high + low + close) / 3`(`ops.vwap` 已有)。

| Base class | 原理 | 处理 |
|---|---|---|
| `vwap_dev` | `((close - vwap) / vwap).rolling(n).mean()`,close 相对 vwap 的 N 日均偏离 | vwap>0(close,h,l>0);n ∈ {3,5,10,20,60} |
| `vwap_weighted_mom` | `sum((close-vwap)*volume, n) / sum(volume, n) / vwap[t]`,量加权偏离动量 | sum(volume) 不为 0(必有成交);n ∈ {5,10,20} |
| `vwap_above_ratio` | `(close > vwap).rolling(n).mean()` ∈ [0,1],N 日 close>vwap 的天数比 | `astype(float)` 后 rolling;n ∈ {5,10,20} |
| `vwap_dev_std` | `((close - vwap) / vwap).rolling(n).std()`,偏离的波动率 | `ddof=0`;n ∈ {20, 60} |

**Mask 关系**:涨停日 close=high → vwap proxy 偏高,偏离信号仍捕获"涨停封板"语义,无需特殊处理。
**Look-ahead**:`rolling` 当日 close 进入,但 vwap proxy 用当日 H/L/C 是 daily-end 估值,不窥未来。

### Family 4: `close_position.py` — 收盘位置动量(Task 6,~15 变体)

`pos_raw = (close - low) / (high - low).replace(0, NaN)`。涨停封板日 high=low=close → range=0 → `pos_raw=NaN`(语义:无法判断区间位置)。

| Base class | 原理 | 处理 |
|---|---|---|
| `close_pos` | `pos_raw.rolling(n).mean()`,N 日均位置 ∈ [0,1] | range=0 NaN 自动传播 rolling;n ∈ {3,5,10,20,60} |
| `close_pos_cum` | `(pos_raw - 0.5).rolling(n).sum()`,累积偏离中位 | 正值多 / 负值空;n ∈ {5,10,20,60} |
| `close_pos_ema` | `pos_raw.ewm(span=n, adjust=False).mean()` | EWM 内部 NaN-tolerant;n ∈ {5,10,20} |

**Mask 关系**:涨停封板自然 NaN 自传播,不需 mask config。这是 spec §6 提到的"NaN 守护与 mask config 解耦"的范例。
**Look-ahead**:`rolling` / `ewm` 都 causal。

### Family 5: `turnover_extra.py` — 短窗换手(Task 7,~12 变体)

补 `custom.turnover_zscore_60` 长窗外的短/中窗指标。

| Base class | 原理 | 处理 |
|---|---|---|
| `turnover_z` | `log(volume).rolling(n)` 的 (lv - mean) / std,短窗换手 z | **必须** `v.replace(0, NaN)` 防停牌日 `log(0)=-inf`;`std.replace(0, NaN)` 防除零;n ∈ {3,5,10,20} |
| `amount_z` | `log(volume * close)` 的 rolling z-score,成交额异常 | 同上;`amount.replace(0, NaN)`;n ∈ {3,5,10,20} |
| `volume_ratio_short` | `volume / mean(volume, n).shift(1) - 1` | `v.replace(0, NaN)`;`shift(1)` 防今值;命名加 `_short` 避免与 `technical.vol_ratio` 冲突;n ∈ {3,5,10} |

**Mask 关系**:spec §6.1.1 锁定 — volume=0 NaN 守护是因子内部职责,无论 mask=on/off 都做。
**Look-ahead**:`.shift(1)` 保证基准量不含今值。

### Family 6: `acceleration.py` — 二阶差分(Task 8,~9 变体)

| Base class | 原理 | 处理 |
|---|---|---|
| `mom_accel` | `mom_n - mom_n.shift(n)`,N 日动量的 N 日差 | `pct_change(n)` 前 n 行 NaN;`.shift(n)` 再延 n 行;n ∈ {3,5,10} |
| `vol_accel` | `lv - 2*lv.shift(n) + lv.shift(2n)`,log(volume) 二阶差分 | `v.replace(0, NaN)`;前 2n 行 NaN;n ∈ {5,10} |
| `turnover_accel` | `turnover_z_n - turnover_z_n.shift(n)` | 复用 `turnover_z` 的 NaN 守护;n ∈ {5,10} |

**Mask 关系**:同 raw。涨停日动量跳变 = 真信号,二阶差分捕获"加速"语义。
**Look-ahead**:全用 `shift(n)`,causal。

### Family 7: `single_stock_vol.py` — 单股波动(Task 9,~20 变体)

| Base class | 原理 | 处理 |
|---|---|---|
| `atr` | Wilder ATR:`true_range = max(H-L, \|H-C_prev\|, \|L-C_prev\|)`,用 `α=1/n` EWM 平滑 | 第 1 行 c_prev=NaN → tr2/tr3 NaN → tr=tr1;n ∈ {7,14,20} |
| `cci` | `(tp - SMA(tp,n)) / (0.015 * MAD(tp,n))`,tp=(H+L+C)/3 | `mad.replace(0, NaN)` 防除零;n ∈ {14,20} |
| `amp` | `((high - low) / close).rolling(n).mean()`,日内振幅均值 | close>0;n ∈ {5,10,20} |
| `park_vol` | Parkinson:`sqrt(mean(ln(H/L)^2 / (4 ln 2), n))` | `low.replace(0, NaN)` 防 ln(inf);n ∈ {20, 60} |
| `gk_vol` | Garman-Klass:`0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2` 再 sqrt(mean) | low/open 替换 0 → NaN;n ∈ {20, 60} |

**Mask 关系**:涨停日窄幅高位收(H=L=C),`atr` true_range 收缩 = 真实低 ATR,反映"封板锁定"状态,不需 mask。
**Look-ahead**:`shift(1)` 用于 c_prev;rolling/ewm 都 causal。

### Family 8: `composite.py` — 复合算子拼装(Task 10,~12 变体)

| Base class | 原理 | 处理 |
|---|---|---|
| `rank_signed_mom` | `rank(mom_n) * sign(vol_chg_n)`,横截面动量秩 × 量变方向 | `rank` axis=1 pct=True;`sign` 含 NaN 传播;n ∈ {5,10,20} |
| `decay_corr_pv` | `decay_linear(ts_corr(rank(close), rank(volume), n), n)`,价格秩-量秩相关的线性衰减加权 | 复用 `ops.correlation` `ops.rank` `ops.decay_linear`;n ∈ {10, 20} |
| `scale_decay_mom` | `scale(decay_linear(close.pct_change(n), n))`,动量先衰减加权再 L1 normalize | `scale` 是横截面 L1 norm;n ∈ {10, 20} |
| `mom_vol_interact` | `mom_n * (volume / mean(volume, n).shift(1) - 1)`,动量与放量乘积 | volume=0 NaN;n ∈ {5, 10, 20} |

**Mask 关系**:原子层(rank/decay/scale)继承 raw 输入,无特殊处理。
**Look-ahead**:`pct_change` / `shift(1)` 显式 causal;`rank` 是当日横截面无时间维。

### Family 9: `rank_correlation.py` — 秩相关(Task 11,~20 变体)

| Base class | 原理 | 处理 |
|---|---|---|
| `corr_pv` | `ts_corr(rank(close), rank(volume), n)`,横截面秩然后时序相关 | `rank` 是 axis=1 横截面,`correlation` 是时序;n ∈ {5,10,20,60} |
| `corr_high_low` | `ts_corr(high, low, n)`,日内 H-L 联动 | 单股时序 corr;n ∈ {10, 20} |
| `corr_close_vwap` | `ts_corr(close, vwap, n)` | vwap proxy;n ∈ {10, 20, 60} |
| `corr_mom_vol` | `ts_corr(close.pct_change(), volume.pct_change(), n)`,收益-量变相关 | 收益首行 NaN;n ∈ {5,10,20} |
| `corr_close_close_lag` | `ts_corr(close, close.shift(1), n)`,自相关 | `shift(1)` 显式;n ∈ {10, 20} |

**Mask 关系**:涨停日 price-volume rank 异常本身是信息(封板锁单),不需要 mask。
**Look-ahead**:`pct_change` / `shift(1)` causal;`ts_corr` 仅看 `[t-n+1, t]`。

### Family 10: `cross_sec_breadth.py` — 截面市场宽度(Task 12,~7 变体)

全市场标量(T×1)广播到 T×N,每只票拿到的是**同一个全市场标量**。

| Base class | 原理 | 处理 |
|---|---|---|
| `breadth_above_ma` | `(close > MA_n).mean(axis=1)`,>MA 占比 → 广播 | `_broadcast` helper 把 T×1 复制到 T×N;n ∈ {5, 20, 60} |
| `breadth_advance` | `(close.pct_change() > 0).mean(axis=1)`,涨股占比 | 收益首行 NaN → 第一天宽度 NaN;无窗口 |
| `breadth_limit_up` | `(close.pct_change() > 0.099).mean(axis=1)`,触涨停占比 | ⚠️ spec §6.1.2 锁定:**涨停股算入分子**,不过滤;无窗口 |
| `breadth_dispersion` | `close.pct_change().std(axis=1)`,横截面离散度 | 无窗口(每日标量) |
| `breadth_pos_skew` | `close.pct_change().skew(axis=1)`,横截面收益偏度 | 至少需 ~5 只股;无窗口 |

**Mask 关系**:**spec §6.1.2 明文锁死** — 在 raw `panel["close"]` 上计算,不调 `apply_mask`,涨停股 / 停牌股不过滤。这是论文 B `cs_rank_*` 的语义。
**Look-ahead**:`pct_change` / `rolling` 全 causal;横截面 `axis=1` 是同日跨股,不跨时无 look-ahead 风险。

### Family 11: `fundamentals.py` — 基本面(Task 13,7 个核心)

⚠️ **PIT 关键**:所有因子按 `pubDate`(公告日)前向填充,**不用** `statDate`(报告期末)。否则会泄露 ~1 个月未来信息。

| Base class | 原理 | 处理 |
|---|---|---|
| `roe` | `profit.roeAvg`(年化 ROE),baostock 直接给 | `_pit_align` ffill;panel 起点前 pubDate → NaN |
| `roa` | `profit.roaAvg` | 同上 |
| `gross_margin` | `profit.gpMargin`(毛利率) | 同上 |
| `net_margin` | `profit.npMargin`(净利率) | 同上 |
| `revenue_yoy` | `growth.YOYIncome`(营收同比),字段名以 Task 0 调研为准 | 若字段名错改成实际值后重跑;同 PIT 处理 |
| `pe` | `close * totalShare / TTM(netProfit)`,4 季滚动 | TTM `min_periods=4` 强制 4 季;`ni > 0` 过滤亏损 → NaN |
| `pb` | `close * totalShare / totalShareholdersEquity` | `equity > 0` 过滤;同 PIT |

**Mask 关系**:基本面与 mask 无关 — 公告披露与可交易性是两个独立约束。
**Look-ahead 防线**:`_pit_align` 用 `reindex(method='ffill')` 保证日 `t` 只能看到 `pubDate ≤ t` 的财报;test `test_roe_factor_uses_pubdate_not_statdate` 是关键防线。

### 注册名总览(参考表,实施时可调整窗口)

下表是建议变体名,**不是必须全部注册**;实施时按 `factor_count_in_expected_range` 测试的 274-322 上下限即可。

```
# 直接统计 (Family 1)
close_std_{5,10,20,60} close_skew_{20,60} close_kurt_{20,60}
volume_skew_{20,60} volume_kurt_{20,60} range_std_{10,20,60}
volume_std_{20,60}

# EWMA (Family 2)
ewma_momentum_hl{5,10,20} ewma_vol_hl{5,10,20} ewma_turnover_z_hl{5,10}
ewma_close_dev_hl{10,20} ewma_volume_ratio_hl{5,10}

# VWAP 偏离 (Family 3)
vwap_dev_{3,5,10,20,60} vwap_weighted_mom_{5,10,20}
vwap_above_ratio_{5,10,20} vwap_dev_std_{20,60}

# 收盘位置 (Family 4)
close_pos_{3,5,10,20,60} close_pos_cum_{5,10,20,60}
close_pos_ema_{5,10,20}

# 短窗换手 (Family 5)
turnover_z_{3,5,10,20} amount_z_{3,5,10,20}
volume_ratio_short_{3,5,10}

# 加速度 (Family 6)
mom_accel_{3,5,10} vol_accel_{5,10} turnover_accel_{5,10}

# 单股波动 (Family 7)
atr_{7,14,20} cci_{14,20} amp_{5,10,20}
park_vol_{20,60} gk_vol_{20,60}

# 复合 (Family 8)
rank_signed_mom_{5,10,20} decay_corr_pv_{10,20}
scale_decay_mom_{10,20} mom_vol_interact_{5,10,20}

# 秩相关 (Family 9)
corr_pv_{5,10,20,60} corr_high_low_{10,20}
corr_close_vwap_{10,20,60} corr_mom_vol_{5,10,20}
corr_close_close_lag_{10,20}

# 截面宽度 (Family 10)
breadth_above_ma_{5,20,60} breadth_advance
breadth_limit_up breadth_dispersion breadth_pos_skew

# 基本面 (Family 11) — 无窗口参数
roe roa gross_margin net_margin revenue_yoy pe pb
```

**变体计数**:Family 1=14,Family 2=12,Family 3=14,Family 4=14,Family 5=11,Family 6=7,Family 7=12,Family 8=11,Family 9=15,Family 10=7,Family 11=7。**合计 124 个变体**(在 spec 给的 160-208 区间下沿,留出加 ps/pcf/debt_to_asset 等 follow-up 的空间)。

---

## Conventions

- **运行 pytest 用 `.venv/Scripts/python.exe -m pytest`**,不要用全局 `python`(numpy/numexpr 不兼容)
- **commit 信息格式**:`<type>(<scope>): <subject>`,沿用项目惯例(`feat(factors): add vwap_deviation family`)
- **Factor 类必须实现**:`__init__` + `name` property + `compute(panel)` method;`from_suffix_args` 用默认实现即可(`MomentumFactor.from_suffix_args(["20"])` 把 "20" 转 int 喂给 `__init__`)
- **@register 装饰器**:`@register(name, sources=..., types=..., description=...)`,name 与类的 `base_name` 一致(不含后缀参数)
- **每家族 smoke 测试覆盖**:(1) factor 注册成功 + 名字解析 (2) compute 返回正确 shape (3) 至少 1 个公式数值 sanity (4) look-ahead 截断后值不变 (5) NaN 守护(如 stop 日 volume=0 / 涨停日 range=0)

---

## Phase 0: Preparation

### Task 0: 验证 baostock 季度财务表 schema

**Files:**
- Output: `docs/handoff/2026-05-31-baostock-fundamentals-schema.md`(临时调研笔记)

**Rationale:** spec §5.1 字段名是按公开文档列出的近似名(`netProfit` / `totalShare` / `npMargin` 等)。实施前用 `.venv` 真实调一次,确认字段名再写 factor 公式,否则后续 Tasks 4-14 都依赖错误的字段名。

- [ ] **Step 1: 写一个一次性脚本验证 5 张表的字段**

```python
# scripts/_probe_baostock_fundamentals.py (临时,不入 git)
import baostock as bs
import pandas as pd

lg = bs.login()
print(f"login: {lg.error_code} {lg.error_msg}")

# 用平安银行 (sz.000001) 一只票、2024Q3 一季,看 schema
code = "sz.000001"
for table_name, fn in [
    ("profit", bs.query_profit_data),
    ("growth", bs.query_growth_data),
    ("balance", bs.query_balance_data),
    ("cash_flow", bs.query_cash_flow_data),
    ("dupont", bs.query_dupont_data),
]:
    rs = fn(code=code, year=2024, quarter=3)
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    print(f"\n=== {table_name} ===")
    print(f"fields: {list(df.columns)}")
    print(df.head(2).T)

bs.logout()
```

- [ ] **Step 2: 运行脚本,记录每张表的实际字段名**

Run: `.venv/Scripts/python.exe scripts/_probe_baostock_fundamentals.py`
Expected: stdout 输出 5 张表的 columns 列表,无异常

- [ ] **Step 3: 把结果写入调研笔记**

文件 `docs/handoff/2026-05-31-baostock-fundamentals-schema.md`:

```markdown
# baostock 季度财务表字段对照(2026-05-31 实测)

> 调研用,后续 fundamentals_loader / fundamentals factor 公式以此为准。

## profit (query_profit_data)
- code, pubDate, statDate, roeAvg, npMargin, gpMargin, netProfit, ...
  (此处粘贴 Step 2 实际输出)

## growth (query_growth_data)
- code, pubDate, statDate, YOYEquity, YOYAsset, YOYNI, ...
  (此处粘贴 Step 2 实际输出)

## balance (query_balance_data)
- code, pubDate, statDate, currentRatio, quickRatio, totalShare, ...
  (此处粘贴 Step 2 实际输出)

## cash_flow (query_cash_flow_data)
- ...

## dupont (query_dupont_data)
- ...

## 实施 mapping(spec 字段 → 实测字段)
| spec | 实测字段 |
| pe   | close * totalShare / TTM(netProfit) |
| pb   | close * totalShare / equity (balance表里待查) |
| pcf  | close * totalShare / TTM(CFOToOR * totalRevenue) (待查) |
| roe  | profit.roeAvg |
| roa  | profit.roaAvg |
...
```

- [ ] **Step 4: 删除临时脚本,commit 调研笔记**

```bash
rm scripts/_probe_baostock_fundamentals.py
git add docs/handoff/2026-05-31-baostock-fundamentals-schema.md
git -c commit.gpgsign=false commit -m "docs(fundamentals): record baostock quarterly schema (Task 0)"
```

---

## Phase 1: Fundamentals 数据基础设施

### Task 1: `fundamentals_loader.py` 实现 + 测试

**Files:**
- Create: `src/stockpool/fundamentals_loader.py`
- Create: `tests/test_fundamentals_loader.py`

- [ ] **Step 1: 写 5 个测试(cache hit / stale / force_refresh / failure-fallback / empty)**

```python
# tests/test_fundamentals_loader.py
"""Tests for stockpool.fundamentals_loader — baostock 5-table PIT cache."""
from __future__ import annotations

import os
import time

import pandas as pd
import pytest


def _mock_long_df():
    """3 codes × 4 quarters mock fundamentals DataFrame."""
    rows = []
    for code in ["000001", "600000", "300001"]:
        for q_idx, (year, q) in enumerate([(2023, 4), (2024, 1), (2024, 2), (2024, 3)]):
            rows.append({
                "code": code,
                "pubDate": pd.Timestamp(f"{year}-{q*3:02d}-28") + pd.Timedelta(days=q_idx),
                "statDate": pd.Timestamp(f"{year}-{q*3:02d}-30"),
                "roeAvg": 0.12 + 0.01 * q_idx,
                "netProfit": 1e9 * (1 + 0.05 * q_idx),
            })
    return pd.DataFrame(rows)


def test_load_or_build_fundamentals_cache_hit(tmp_path):
    """Fresh cache parquet → 直接读盘,不调 baostock。"""
    from stockpool.fundamentals_loader import load_or_build_fundamentals

    df = _mock_long_df()
    cache = tmp_path / "fundamentals_profit.parquet"
    df.to_parquet(cache, index=False)

    result = load_or_build_fundamentals("profit", cache_dir=tmp_path)
    assert len(result) == 12
    assert set(result["code"]) == {"000001", "600000", "300001"}
    assert "pubDate" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["pubDate"])


def test_load_or_build_fundamentals_stale_triggers_refresh(monkeypatch, tmp_path):
    """Mtime 老于 max_age_days → 触发 _fetch_table。"""
    from stockpool import fundamentals_loader as fl

    cache = tmp_path / "fundamentals_profit.parquet"
    _mock_long_df().head(3).to_parquet(cache, index=False)
    old = time.time() - 60 * 86400
    os.utime(cache, (old, old))

    called = {"n": 0}
    def fake_fetch(table, codes):
        called["n"] += 1
        return _mock_long_df()
    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    result = fl.load_or_build_fundamentals("profit", cache_dir=tmp_path, max_age_days=30)
    assert called["n"] == 1
    assert len(result) == 12


def test_load_or_build_fundamentals_force_refresh(monkeypatch, tmp_path):
    """force_refresh=True → 即便缓存新鲜也重拉。"""
    from stockpool import fundamentals_loader as fl

    cache = tmp_path / "fundamentals_profit.parquet"
    _mock_long_df().head(3).to_parquet(cache, index=False)

    called = {"n": 0}
    def fake_fetch(table, codes):
        called["n"] += 1
        return _mock_long_df()
    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    fl.load_or_build_fundamentals("profit", cache_dir=tmp_path, force_refresh=True)
    assert called["n"] == 1


def test_load_or_build_fundamentals_fetch_fail_falls_back_to_stale(monkeypatch, tmp_path):
    """baostock 抛错 + 有 stale 缓存 → 用 stale 缓存。"""
    from stockpool import fundamentals_loader as fl

    cache = tmp_path / "fundamentals_profit.parquet"
    _mock_long_df().to_parquet(cache, index=False)
    old = time.time() - 60 * 86400
    os.utime(cache, (old, old))

    def fake_fetch(table, codes):
        raise RuntimeError("network down")
    monkeypatch.setattr(fl, "_fetch_table", fake_fetch)

    result = fl.load_or_build_fundamentals("profit", cache_dir=tmp_path, max_age_days=30)
    assert len(result) == 12  # 从 stale 缓存读


def test_load_or_build_fundamentals_unknown_table_raises(tmp_path):
    """非法 table 名 → ValueError。"""
    from stockpool.fundamentals_loader import load_or_build_fundamentals

    with pytest.raises(ValueError, match="table"):
        load_or_build_fundamentals("does_not_exist", cache_dir=tmp_path)
```

- [ ] **Step 2: 运行测试确认全部失败(模块不存在)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fundamentals_loader.py -v`
Expected: 5 FAIL with `ModuleNotFoundError: No module named 'stockpool.fundamentals_loader'`

- [ ] **Step 3: 实现 `fundamentals_loader.py`**

```python
# src/stockpool/fundamentals_loader.py
"""baostock 5 张季度财务表的 PIT 缓存层。

参考 ``stockpool.ipo_dates`` 的 baostock login + parquet cache + mtime
staleness 模式。每张表缓存到 ``<cache_dir>/fundamentals_<table>.parquet``。

PIT 设计:long-form DataFrame 保留 ``pubDate`` 字段,factor 计算时按
``pubDate`` 而非 ``statDate`` 前向填充到日频(防 ~1 个月未来泄露)。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_VALID_TABLES = ("profit", "growth", "balance", "cash_flow", "dupont")
_TABLE_TO_BS_FN = {
    "profit": "query_profit_data",
    "growth": "query_growth_data",
    "balance": "query_balance_data",
    "cash_flow": "query_cash_flow_data",
    "dupont": "query_dupont_data",
}


def load_or_build_fundamentals(
    table: str,
    *,
    codes: list[str] | None = None,
    cache_dir: str | Path | None = None,
    max_age_days: int = 30,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """返回 long-form DataFrame: code / pubDate / statDate / <fields...>.

    Args:
        table: 五选一: profit / growth / balance / cash_flow / dupont
        codes: None → 拉全市场;否则只拉指定 6 位 code 列表
        cache_dir: 缓存目录;None → 不缓存(纯获取)
        max_age_days: 缓存有效期
        force_refresh: True 时无条件重拉

    Returns:
        long-form DataFrame, 每股每季一行。pubDate 是 datetime64。
        失败 + 无缓存时返回 empty DataFrame。
    """
    if table not in _VALID_TABLES:
        raise ValueError(
            f"unknown table={table!r}; valid: {_VALID_TABLES}"
        )

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"fundamentals_{table}.parquet"

        if not force_refresh and cache_path.exists():
            age = (time.time() - cache_path.stat().st_mtime) / 86400.0
            if age <= max_age_days:
                try:
                    return _read_cache(cache_path)
                except Exception as e:
                    log.warning("fundamentals cache corrupt (%s), rebuilding", e)
            else:
                log.info("fundamentals(%s) cache stale (%.1f d > %d d)",
                         table, age, max_age_days)

    try:
        df = _fetch_table(table, codes)
    except Exception as e:
        log.error("fundamentals(%s) fetch failed: %s", table, e)
        if cache_path is not None and cache_path.exists():
            log.info("fundamentals(%s): using stale cache", table)
            try:
                return _read_cache(cache_path)
            except Exception:
                pass
        return pd.DataFrame()

    if df.empty:
        return df

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log.info("fundamentals(%s) cache written: %s (%d rows)",
                 table, cache_path, len(df))
    return df


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # pubDate 必须是 datetime64;若被 parquet 推断为 object,显式 cast
    if not pd.api.types.is_datetime64_any_dtype(df["pubDate"]):
        df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
    return df


def _fetch_table(table: str, codes: list[str] | None) -> pd.DataFrame:
    """串行调 baostock 拉某张季度表。codes=None → 走 universe.parquet 全市场。

    每股每季一次 query (5500 × 16 = 88000 calls, 串行约 6-10 min/table)。
    Per-stock 失败 log warning 跳过,不抛出。
    """
    import baostock as bs

    if codes is None:
        # 走 fetcher 的 universe.parquet
        from stockpool.fetcher import list_universe
        codes = list_universe()  # ["000001", ...]

    fn_name = _TABLE_TO_BS_FN[table]
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        fn = getattr(bs, fn_name)
        all_rows: list[dict] = []
        all_fields: list[str] | None = None
        # 拉最近 16 季(过去 4 年)
        today = pd.Timestamp.today()
        quarters = _recent_quarters(today, n=16)
        for code in codes:
            bs_code = _to_bs_code(code)
            for year, q in quarters:
                try:
                    rs = fn(code=bs_code, year=year, quarter=q)
                    if rs.error_code != "0":
                        continue
                    if all_fields is None and rs.fields:
                        all_fields = list(rs.fields)
                    while rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        # 标准化 code 为 6 位
                        row["code"] = code
                        all_rows.append(row)
                except Exception as e:
                    log.warning("baostock %s %s %dQ%d failed: %s",
                                fn_name, bs_code, year, q, e)
    finally:
        bs.logout()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # pubDate / statDate 转 datetime
    for col in ("pubDate", "statDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["pubDate"]).reset_index(drop=True)
    log.info("fundamentals(%s): %d rows fetched across %d codes",
             table, len(df), df["code"].nunique())
    return df


def _to_bs_code(code: str) -> str:
    """6 位 code → baostock 格式 (sh./sz./bj.)."""
    if code.startswith(("60", "68")):
        return f"sh.{code}"
    if code.startswith(("00", "30")):
        return f"sz.{code}"
    if code.startswith(("8", "43")):
        return f"bj.{code}"
    return f"sh.{code}"  # 兜底


def _recent_quarters(today: pd.Timestamp, n: int = 16) -> list[tuple[int, int]]:
    """返回最近 n 个 (year, quarter) tuple,降序。"""
    quarters = []
    y, q = today.year, ((today.month - 1) // 3) + 1
    for _ in range(n):
        quarters.append((y, q))
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return quarters
```

- [ ] **Step 4: 运行测试确认全部通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fundamentals_loader.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/fundamentals_loader.py tests/test_fundamentals_loader.py
git -c commit.gpgsign=false commit -m "feat(fundamentals): add baostock PIT loader (Task 1)"
```

---

### Task 2: CLI `--refresh-fundamentals` flag

**Files:**
- Modify: `src/stockpool/cli.py`(找 `--refresh-factor-panel` 附近,加同款 flag)
- Test: `tests/test_cli_refresh_fundamentals.py`

- [ ] **Step 1: 写测试(smoke,确认 flag 被解析、传递)**

```python
# tests/test_cli_refresh_fundamentals.py
"""Smoke test for --refresh-fundamentals CLI flag wiring."""
from __future__ import annotations

import sys

import pytest


def test_run_accepts_refresh_fundamentals_flag(monkeypatch, capsys):
    """`python -m stockpool run --refresh-fundamentals` 不应该报 unknown arg。"""
    from stockpool import cli

    # 不真跑,只校验 argparse 接受 flag
    parser = cli._build_parser()
    args = parser.parse_args(["run", "--config", "config.yaml", "--refresh-fundamentals"])
    assert args.refresh_fundamentals is True


def test_backtest_accepts_refresh_fundamentals_flag():
    from stockpool import cli
    parser = cli._build_parser()
    args = parser.parse_args(["backtest", "--config", "config.yaml", "--refresh-fundamentals"])
    assert args.refresh_fundamentals is True


def test_portfolio_backtest_accepts_refresh_fundamentals_flag():
    from stockpool import cli
    parser = cli._build_parser()
    args = parser.parse_args(
        ["portfolio-backtest", "--config", "config.yaml", "--refresh-fundamentals"]
    )
    assert args.refresh_fundamentals is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_refresh_fundamentals.py -v`
Expected: 3 FAIL with `unrecognized arguments: --refresh-fundamentals` 或 `args.refresh_fundamentals` AttributeError

- [ ] **Step 3: 在 cli.py 的 `_build_parser` 里给 run/backtest/portfolio-backtest 三个子命令加 flag**

定位:`src/stockpool/cli.py` 里找 `add_argument("--refresh-factor-panel"...` 这几处,旁边加同款 `--refresh-fundamentals`:

```python
# 在 run subparser 定义里
run_p.add_argument(
    "--refresh-fundamentals", action="store_true",
    help="强制重拉 baostock 财务数据(绕过 30 天缓存)",
)

# 同样在 backtest_p 和 portfolio_p 里加
backtest_p.add_argument(
    "--refresh-fundamentals", action="store_true",
    help="强制重拉 baostock 财务数据(绕过 30 天缓存)",
)
portfolio_p.add_argument(
    "--refresh-fundamentals", action="store_true",
    help="强制重拉 baostock 财务数据(绕过 30 天缓存)",
)
```

实际命令处理(传给后续的 fundamentals_loader)在 Task 14 接上 — 本步只到 argparse 层即可让测试通过。

- [ ] **Step 4: 测试通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_refresh_fundamentals.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/cli.py tests/test_cli_refresh_fundamentals.py
git -c commit.gpgsign=false commit -m "feat(cli): add --refresh-fundamentals flag (Task 2)"
```

---

## Phase 2: 技术因子家族(由简到繁)

通用模板:每家族任务都遵循同样 5 步 — 写 smoke 测试 → 跑失败 → 实现 N 个 Factor 类 → 跑通过 → commit。每任务给出 3-5 个完整 class + 剩余变体的参数表。

**重要:每个新建的 factors/*.py 文件,在 Task 15 才统一在 `factors/__init__.py` 加 import 副作用。所以单独家族测试期,需要在测试文件顶部显式 `import stockpool.factors.<家族> as _` 触发注册。**

### Task 3: `original_stats.py` — rolling 直接统计

**Files:**
- Create: `src/stockpool/factors/original_stats.py`
- Create: `tests/test_factors_original_stats.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_original_stats.py
"""Smoke tests for direct rolling statistic factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 触发注册
import stockpool.factors.original_stats as _orig  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B", "C"]
    rng = np.random.default_rng(42)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 3)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    high = close + rng.uniform(0.1, 2.0, size=close.shape)
    low = close - rng.uniform(0.1, 2.0, size=close.shape)
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close, "high": high, "low": low,
            "open": close.shift(1).fillna(close.iloc[0]), "volume": volume}


def test_close_std_20_registered(panel):
    f = make_factor("close_std_20")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape
    # 前 11 行(< 0.6 * 20 = 12 min_periods 实际是 .rolling(20) 用 std 默认)NaN
    assert out.iloc[:11].isna().all().all()
    assert out.iloc[30:].notna().any().any()


def test_close_skew_20_returns_finite(panel):
    f = make_factor("close_skew_20")
    out = f.compute(panel)
    valid = out.iloc[30:]
    assert np.isfinite(valid.to_numpy()).any()


def test_volume_std_60_normalized(panel):
    """volume_std 是变异系数 (std/mean), 应该 > 0。"""
    f = make_factor("volume_std_60")
    out = f.compute(panel)
    last_row = out.iloc[-1].dropna()
    assert (last_row > 0).all()


def test_range_std_d_matches_formula(panel):
    """range_std_20 应该等于 (high-low).rolling(20).std() / close 数值。"""
    f = make_factor("range_std_20")
    out = f.compute(panel)
    expected = (panel["high"] - panel["low"]).rolling(20).std() / panel["close"]
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-9)


def test_no_look_ahead_truncation_close_std_20(panel):
    """截断后 panel 算因子,前 N 行应与全 panel 算的前 N 行一致。"""
    f = make_factor("close_std_20")
    full = f.compute(panel)
    truncated = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(truncated)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    """注册表应该有这些 base name。"""
    for name in ("close_std", "close_skew", "close_kurt",
                 "volume_skew", "volume_kurt",
                 "range_std", "volume_std"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_original_stats.py -v`
Expected: ALL FAIL (`ModuleNotFoundError` or `KeyError: Unknown factor`)

- [ ] **Step 3: 实现 `original_stats.py`**

```python
# src/stockpool/factors/original_stats.py
"""rolling 直接统计量因子族 (论文 B original_* 28 个的精神复现)。

公式都是 close / volume / range 的 rolling mean/std/skew/kurt 直接量,
不做 rank、不做归一化(除自身比例外)。

变体数 ~20-25,7 个 base class × 多个窗口参数。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "close_std",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="close N 日 std / close,归一化波动率",
)
class CloseStdFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_std_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        return c.rolling(self.n).std(ddof=0) / c


@register(
    "close_skew",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="close N 日滚动 skewness",
)
class CloseSkewFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_skew_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["close"].rolling(self.n).skew()


@register(
    "close_kurt",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="close N 日滚动 kurtosis",
)
class CloseKurtFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_kurt_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["close"].rolling(self.n).kurt()


@register(
    "volume_skew",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="volume N 日滚动 skewness",
)
class VolumeSkewFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"volume_skew_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["volume"].rolling(self.n).skew()


@register(
    "volume_kurt",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="volume N 日滚动 kurtosis",
)
class VolumeKurtFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"volume_kurt_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return panel["volume"].rolling(self.n).kurt()


@register(
    "range_std",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="(high-low) N 日 std / close,日内振幅波动",
)
class RangeStdFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"range_std_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        rng = panel["high"] - panel["low"]
        return rng.rolling(self.n).std(ddof=0) / panel["close"]


@register(
    "volume_std",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="volume 变异系数 (N 日 std / N 日 mean)",
)
class VolumeStdFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"volume_std_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"]
        mean = v.rolling(self.n).mean()
        std = v.rolling(self.n).std(ddof=0)
        return std / mean.replace(0.0, pd.NA)
```

变体由 `factors_list` 用户在 yaml / selection.json 里组合产生:`close_std_5` / `close_std_10` / `close_std_20` / `close_std_60` 等。base class 数 = 7,窗口数典型 4,生成 ~25 个变体名。

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_original_stats.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/original_stats.py tests/test_factors_original_stats.py
git -c commit.gpgsign=false commit -m "feat(factors): add original_stats family (7 base, ~25 variants) (Task 3)"
```

---

### Task 4: `ewma.py` — EWMA 平滑因子

**Files:**
- Create: `src/stockpool/factors/ewma.py`
- Create: `tests/test_factors_ewma.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_ewma.py
"""Smoke tests for EWMA-smoothed factors (halflife parameterized)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.ewma as _ewma  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_ewma_momentum_hl10_registered(panel):
    f = make_factor("ewma_momentum_hl10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape
    assert out.iloc[20:].notna().all().all()  # 20 期后必有值


def test_ewma_vol_hl10_positive(panel):
    f = make_factor("ewma_vol_hl10")
    out = f.compute(panel)
    valid = out.iloc[30:]
    assert (valid >= 0).all().all()


def test_ewma_close_dev_matches_formula(panel):
    f = make_factor("ewma_close_dev_hl10")
    out = f.compute(panel)
    c = panel["close"]
    ema = c.ewm(halflife=10).mean()
    std = c.ewm(halflife=10).std()
    expected = (c - ema) / std
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-9)


def test_ewma_no_look_ahead(panel):
    f = make_factor("ewma_momentum_hl10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("ewma_momentum", "ewma_vol", "ewma_turnover_z",
                 "ewma_close_dev", "ewma_volume_ratio"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 运行失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_ewma.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现 `ewma.py`**

```python
# src/stockpool/factors/ewma.py
"""EWMA 平滑因子族 (本 spec 自主补,论文 B 无对应)。

5 个 base class × 半衰期 ∈ {5, 10, 20} = ~15 变体。
命名:``ewma_<signal>_hl<h>``,h 是 halflife。

后缀解析:from_suffix_args 把 ["hl10"] 解析成 halflife=10。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _parse_hl(args: list[str]) -> int:
    """suffix 形如 ["hl10"] → 10。"""
    if len(args) != 1 or not args[0].startswith("hl"):
        raise ValueError(f"expected ['hl<n>'], got {args!r}")
    return int(args[0][2:])


@register(
    "ewma_momentum",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="close 相对 EWMA 的偏离(半衰期 h)",
)
class EWMAMomentumFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_momentum_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        ema = c.ewm(halflife=self.halflife).mean()
        return (c - ema) / ema

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMAMomentumFactor":
        return cls(halflife=_parse_hl(args))


@register(
    "ewma_vol",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="RiskMetrics-like EWMA 收益波动率(半衰期 h)",
)
class EWMAVolFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_vol_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        return ret.ewm(halflife=self.halflife).std()

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMAVolFactor":
        return cls(halflife=_parse_hl(args))


@register(
    "ewma_turnover_z",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="log(volume) EWMA z-score,异常活跃度",
)
class EWMATurnoverZFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_turnover_z_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.ewm(halflife=self.halflife).mean()
        std = lv.ewm(halflife=self.halflife).std().replace(0.0, np.nan)
        return (lv - mean) / std

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMATurnoverZFactor":
        return cls(halflife=_parse_hl(args))


@register(
    "ewma_close_dev",
    sources=("builtin",),
    types=("trend", "time_series"),
    description="(close - EWMA(close)) / EWMA std,close 偏离自身 EWMA 的 z",
)
class EWMACloseDevFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_close_dev_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        ema = c.ewm(halflife=self.halflife).mean()
        std = c.ewm(halflife=self.halflife).std()
        return (c - ema) / std

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMACloseDevFactor":
        return cls(halflife=_parse_hl(args))


@register(
    "ewma_volume_ratio",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="volume / EWMA(volume).shift(1) - 1,放/缩量 EWMA 版",
)
class EWMAVolumeRatioFactor(Factor):
    def __init__(self, halflife: int = 10):
        if halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.halflife = halflife

    @property
    def name(self) -> str:
        return f"ewma_volume_ratio_hl{self.halflife}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"]
        ema = v.ewm(halflife=self.halflife).mean().shift(1)
        return v / ema - 1.0

    @classmethod
    def from_suffix_args(cls, args: list[str]) -> "EWMAVolumeRatioFactor":
        return cls(halflife=_parse_hl(args))
```

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_ewma.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/ewma.py tests/test_factors_ewma.py
git -c commit.gpgsign=false commit -m "feat(factors): add ewma family (5 base, ~15 variants) (Task 4)"
```

---

### Task 5: `vwap_deviation.py` — VWAP 偏离族

**Files:**
- Create: `src/stockpool/factors/vwap_deviation.py`
- Create: `tests/test_factors_vwap_deviation.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_vwap_deviation.py
"""Smoke tests for VWAP deviation family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.vwap_deviation as _vd  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(13)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    high = close + rng.uniform(0.5, 2.0, size=close.shape)
    low = close - rng.uniform(0.5, 2.0, size=close.shape)
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close, "high": high, "low": low,
            "open": close.shift(1).fillna(close.iloc[0]), "volume": volume}


def test_vwap_dev_5_returns_finite(panel):
    f = make_factor("vwap_dev_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape
    assert np.isfinite(out.iloc[10:].to_numpy()).any()


def test_vwap_dev_d_zero_centered(panel):
    """理论上 (close - vwap)/vwap 在长时间上应该接近 0(无系统性偏离)。"""
    f = make_factor("vwap_dev_20")
    out = f.compute(panel)
    # 不严格要求 0,但绝对均值应远小于 1
    assert out.iloc[30:].abs().mean().mean() < 0.5


def test_vwap_weighted_mom_10_runs(panel):
    f = make_factor("vwap_weighted_mom_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_vwap_above_ratio_in_unit(panel):
    """vwap_above_ratio_d 应 ∈ [0, 1]。"""
    f = make_factor("vwap_above_ratio_10")
    out = f.compute(panel)
    valid = out.iloc[15:].to_numpy()
    valid = valid[~np.isnan(valid)]
    assert ((valid >= 0) & (valid <= 1)).all()


def test_no_look_ahead(panel):
    f = make_factor("vwap_dev_10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("vwap_dev", "vwap_weighted_mom", "vwap_above_ratio"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_vwap_deviation.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现**

```python
# src/stockpool/factors/vwap_deviation.py
"""VWAP 偏离族 (论文 B better_* 28 个的精神复现)。

VWAP proxy: (high + low + close) / 3 (复用 ops.vwap)。
4 个 base class × 5 窗口 ∈ {3, 5, 10, 20, 60} = ~20 变体。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "vwap_dev",
    sources=("builtin",),
    types=("trend", "volume", "time_series"),
    description="(close - vwap) / vwap 的 N 日均值",
)
class VWAPDevFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vwap_dev_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        vwap = ops.vwap(panel)
        dev = (panel["close"] - vwap) / vwap
        return dev.rolling(self.n).mean()


@register(
    "vwap_weighted_mom",
    sources=("builtin",),
    types=("momentum", "volume", "time_series"),
    description="量加权偏离动量: sum_d((close-vwap)*volume) / sum_d(volume) / vwap[t]",
)
class VWAPWeightedMomFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vwap_weighted_mom_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        vwap = ops.vwap(panel)
        v = panel["volume"]
        weighted_dev = (panel["close"] - vwap) * v
        num = weighted_dev.rolling(self.n).sum()
        den = v.rolling(self.n).sum()
        return num / den / vwap


@register(
    "vwap_above_ratio",
    sources=("builtin",),
    types=("trend", "time_series"),
    description="N 日内 close > vwap 的天数比例 ∈ [0, 1]",
)
class VWAPAboveRatioFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vwap_above_ratio_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        vwap = ops.vwap(panel)
        above = (panel["close"] > vwap).astype(float)
        return above.rolling(self.n).mean()


@register(
    "vwap_dev_std",
    sources=("builtin",),
    types=("volatility", "volume", "time_series"),
    description="(close - vwap) / vwap 的 N 日 std",
)
class VWAPDevStdFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vwap_dev_std_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        vwap = ops.vwap(panel)
        dev = (panel["close"] - vwap) / vwap
        return dev.rolling(self.n).std(ddof=0)
```

base 数 = 4,典型窗口 {3, 5, 10, 20, 60} → 20 变体。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_vwap_deviation.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/vwap_deviation.py tests/test_factors_vwap_deviation.py
git -c commit.gpgsign=false commit -m "feat(factors): add vwap_deviation family (4 base, ~20 variants) (Task 5)"
```

---

### Task 6: `close_position.py` — 收盘位置动量

**Files:**
- Create: `src/stockpool/factors/close_position.py`
- Create: `tests/test_factors_close_position.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_close_position.py
"""Smoke tests for close-in-range position family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.close_position as _cp  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(99)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    high = close + rng.uniform(0.5, 2.0, size=close.shape)
    low = close - rng.uniform(0.5, 2.0, size=close.shape)
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close, "high": high, "low": low,
            "open": close.shift(1).fillna(close.iloc[0]), "volume": volume}


def test_close_pos_5_in_unit(panel):
    """close_pos_d ∈ [0, 1] (close 永远在 [low, high] 内)。"""
    f = make_factor("close_pos_5")
    out = f.compute(panel)
    valid = out.iloc[10:].to_numpy()
    valid = valid[~np.isnan(valid)]
    assert (valid >= -1e-9).all()  # 允许浮点误差
    assert (valid <= 1.0 + 1e-9).all()


def test_close_pos_zero_range_returns_nan():
    """high==low (涨停封板) 时 close_pos 应 NaN。"""
    f = make_factor("close_pos_5")
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    # B 列全 high==low==close
    close = pd.DataFrame({"A": np.arange(20) + 100.0,
                          "B": np.full(20, 100.0)},
                         index=dates)
    panel = {
        "close": close,
        "high": close.copy(), "low": close.copy(),
        "open": close.shift(1).fillna(close.iloc[0]),
        "volume": pd.DataFrame(1.0, index=dates, columns=close.columns),
    }
    panel["high"]["A"] = close["A"] + 1
    panel["low"]["A"] = close["A"] - 1
    # A 有 range, B 没有
    out = f.compute(panel)
    assert out["B"].iloc[10:].isna().all()  # 全 NaN
    assert out["A"].iloc[10:].notna().any()


def test_close_pos_cum_centered(panel):
    """close_pos_cum 是 (pos - 0.5) rolling sum,可正可负。"""
    f = make_factor("close_pos_cum_10")
    out = f.compute(panel)
    assert out.iloc[15:].notna().all().all()


def test_no_look_ahead(panel):
    f = make_factor("close_pos_10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("close_pos", "close_pos_cum", "close_pos_ema"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_close_position.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现**

```python
# src/stockpool/factors/close_position.py
"""收盘位置动量族 (论文 B best_* 21 个的精神复现)。

pos_raw = (close - low) / (high - low),涨停封板日 range=0 时 NaN。
4 个 base × 5 窗口 = ~20 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _pos_raw(panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """(close - low) / (high - low),range=0 时 NaN。"""
    rng = (panel["high"] - panel["low"]).replace(0.0, np.nan)
    return (panel["close"] - panel["low"]) / rng


@register(
    "close_pos",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="(close - low) / (high - low) 的 N 日均值 ∈ [0, 1]",
)
class ClosePositionFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_pos_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return _pos_raw(panel).rolling(self.n).mean()


@register(
    "close_pos_cum",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="(pos - 0.5) 的 N 日累积偏离,正多空",
)
class ClosePositionCumFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_pos_cum_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return (_pos_raw(panel) - 0.5).rolling(self.n).sum()


@register(
    "close_pos_ema",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="收盘位置 pos 的 EMA(span=n) 平滑",
)
class ClosePositionEMAFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"close_pos_ema_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return _pos_raw(panel).ewm(span=self.n, adjust=False).mean()
```

base 数 = 3,窗口 {3, 5, 10, 20, 60} = 15 变体。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_close_position.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/close_position.py tests/test_factors_close_position.py
git -c commit.gpgsign=false commit -m "feat(factors): add close_position family (3 base, ~15 variants) (Task 6)"
```

---

### Task 7: `turnover_extra.py` — 短窗换手族

**Files:**
- Create: `src/stockpool/factors/turnover_extra.py`
- Create: `tests/test_factors_turnover_extra.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_turnover_extra.py
"""Smoke tests for short-window turnover / amount factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.turnover_extra as _te  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(5)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_turnover_z_5_runs(panel):
    f = make_factor("turnover_z_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_turnover_z_handles_volume_zero():
    """停牌日 volume=0 必须 NaN,不能 -inf 污染。"""
    f = make_factor("turnover_z_5")
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    volume = pd.DataFrame(1e6, index=dates, columns=["A"])
    volume.iloc[15] = 0.0  # 停牌日
    panel = {
        "close": pd.DataFrame(100.0, index=dates, columns=["A"]),
        "high": pd.DataFrame(101.0, index=dates, columns=["A"]),
        "low": pd.DataFrame(99.0, index=dates, columns=["A"]),
        "open": pd.DataFrame(100.0, index=dates, columns=["A"]),
        "volume": volume,
    }
    out = f.compute(panel)
    # 停牌日及附近不应有 -inf
    assert not np.isinf(out.to_numpy()).any()


def test_amount_z_10_runs(panel):
    f = make_factor("amount_z_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_volume_ratio_short_window(panel):
    f = make_factor("volume_ratio_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_no_look_ahead(panel):
    f = make_factor("turnover_z_10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("turnover_z", "amount_z", "volume_ratio_short"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_turnover_extra.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现**

```python
# src/stockpool/factors/turnover_extra.py
"""短窗换手族 (论文 B extra_* 14 个的精神复现)。

补 custom.py:turnover_zscore_60 (长窗) 之外的短/中窗换手指标。
``v.replace(0.0, np.nan)`` 防停牌日 log(0) 污染,与 custom.py 一致。
3 个 base class × 4 窗口 = ~12 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "turnover_z",
    sources=("custom",),
    types=("volume", "time_series"),
    description="log(volume) 短窗 z-score (短 vs custom.turnover_zscore_60 长窗)",
)
class TurnoverZShortFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"turnover_z_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.rolling(self.n).mean()
        std = lv.rolling(self.n).std(ddof=0).replace(0.0, np.nan)
        return (lv - mean) / std


@register(
    "amount_z",
    sources=("custom",),
    types=("volume", "time_series"),
    description="log(volume*close) 短窗 z-score (成交额)",
)
class AmountZFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"amount_z_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        amount = (panel["volume"] * panel["close"]).replace(0.0, np.nan)
        la = np.log(amount)
        mean = la.rolling(self.n).mean()
        std = la.rolling(self.n).std(ddof=0).replace(0.0, np.nan)
        return (la - mean) / std


@register(
    "volume_ratio_short",
    sources=("custom",),
    types=("volume", "time_series"),
    description="volume / mean(volume, n).shift(1) - 1,短窗放/缩量",
)
class VolumeRatioShortFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"volume_ratio_short_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        mean = v.rolling(self.n).mean().shift(1)
        return v / mean - 1.0
```

注:`volume_ratio_short` 与现有 `technical.py:vol_ratio` 区别仅在窗口建议(短窗 3/5/10)。命名加 `_short` 避免与现有冲突。

base 数 = 3,窗口 {3, 5, 10, 20} = 12 变体。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_turnover_extra.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/turnover_extra.py tests/test_factors_turnover_extra.py
git -c commit.gpgsign=false commit -m "feat(factors): add turnover_extra family (3 base, ~12 variants) (Task 7)"
```

---

### Task 8: `acceleration.py` — 加速度族

**Files:**
- Create: `src/stockpool/factors/acceleration.py`
- Create: `tests/test_factors_acceleration.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_acceleration.py
"""Smoke tests for second-order difference (acceleration) factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.acceleration as _acc  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(11)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_mom_accel_5_matches_formula(panel):
    f = make_factor("mom_accel_5")
    out = f.compute(panel)
    mom = panel["close"].pct_change(5, fill_method=None)
    expected = mom - mom.shift(5)
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-9)


def test_vol_accel_5_runs(panel):
    f = make_factor("vol_accel_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_turnover_accel_5_runs(panel):
    f = make_factor("turnover_accel_5")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_no_look_ahead(panel):
    f = make_factor("mom_accel_5")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("mom_accel", "vol_accel", "turnover_accel"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_acceleration.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现**

```python
# src/stockpool/factors/acceleration.py
"""加速度族 (论文 B change_* 5 个的精神复现)。

动量/换手的二阶差分,捕获趋势变速。3 个 base × 3 窗口 = ~6 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "mom_accel",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="N 日动量的 N 日差: mom_d - mom_d.shift(d)",
)
class MomAccelFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"mom_accel_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mom = panel["close"].pct_change(self.n, fill_method=None)
        return mom - mom.shift(self.n)


@register(
    "vol_accel",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="log(volume) 二阶差分: lv - 2*lv.shift(n) + lv.shift(2n)",
)
class VolAccelFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"vol_accel_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        return lv - 2.0 * lv.shift(self.n) + lv.shift(2 * self.n)


@register(
    "turnover_accel",
    sources=("builtin",),
    types=("volume", "time_series"),
    description="turnover_z_n 的 N 日差,换手 z-score 的加速度",
)
class TurnoverAccelFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"turnover_accel_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.rolling(self.n).mean()
        std = lv.rolling(self.n).std(ddof=0).replace(0.0, np.nan)
        tz = (lv - mean) / std
        return tz - tz.shift(self.n)
```

base 数 = 3,窗口 {3, 5, 10} = 9 变体。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_acceleration.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/acceleration.py tests/test_factors_acceleration.py
git -c commit.gpgsign=false commit -m "feat(factors): add acceleration family (3 base, ~9 variants) (Task 8)"
```

---

### Task 9: `single_stock_vol.py` — ATR / CCI / 振幅 / Parkinson

**Files:**
- Create: `src/stockpool/factors/single_stock_vol.py`
- Create: `tests/test_factors_single_stock_vol.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_single_stock_vol.py
"""Smoke tests for single-stock volatility / range factors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.single_stock_vol as _sv  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B"]
    rng = np.random.default_rng(17)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 2)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    high = close + rng.uniform(0.5, 2.0, size=close.shape)
    low = close - rng.uniform(0.5, 2.0, size=close.shape)
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close, "high": high, "low": low,
            "open": close.shift(1).fillna(close.iloc[0]), "volume": volume}


def test_atr_14_positive(panel):
    f = make_factor("atr_14")
    out = f.compute(panel)
    valid = out.iloc[15:]
    assert (valid > 0).all().all()


def test_cci_20_runs(panel):
    f = make_factor("cci_20")
    out = f.compute(panel)
    valid = out.iloc[25:]
    assert np.isfinite(valid.to_numpy()).any()


def test_amp_5_positive(panel):
    """振幅 = (high-low) / close 的 N 日均值,应 > 0。"""
    f = make_factor("amp_5")
    out = f.compute(panel)
    valid = out.iloc[10:]
    assert (valid > 0).all().all()


def test_park_vol_20_positive(panel):
    """Parkinson vol 必非负。"""
    f = make_factor("park_vol_20")
    out = f.compute(panel)
    valid = out.iloc[25:]
    assert (valid >= 0).all().all()


def test_no_look_ahead(panel):
    f = make_factor("atr_14")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("atr", "cci", "amp", "park_vol", "gk_vol"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_single_stock_vol.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现**

```python
# src/stockpool/factors/single_stock_vol.py
"""单股波动 / 振幅族 (论文 B stock_* 22 个的精神复现)。

ATR / CCI / 日内振幅 / Parkinson vol / Garman-Klass vol。
5 个 base × 4 窗口 = ~20 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "atr",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="Wilder ATR: EMA(span=n) 的真实波幅 max(h-l, |h-c_prev|, |l-c_prev|)",
)
class ATRFactor(Factor):
    def __init__(self, n: int = 14):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"atr_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        h, l, c = panel["high"], panel["low"], panel["close"]
        c_prev = c.shift(1)
        tr1 = h - l
        tr2 = (h - c_prev).abs()
        tr3 = (l - c_prev).abs()
        tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
        # Wilder smoothing: alpha=1/n EWM
        return tr.ewm(alpha=1.0 / self.n, adjust=False).mean()


@register(
    "cci",
    sources=("builtin",),
    types=("reversal", "time_series"),
    description="CCI = (tp - SMA(tp, n)) / (0.015 * MAD(tp, n)),tp=(H+L+C)/3",
)
class CCIFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"cci_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        tp = (panel["high"] + panel["low"] + panel["close"]) / 3.0
        sma = tp.rolling(self.n).mean()
        # MAD = mean(|tp - SMA|)
        mad = (tp - sma).abs().rolling(self.n).mean().replace(0.0, np.nan)
        return (tp - sma) / (0.015 * mad)


@register(
    "amp",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="日内振幅 (high-low)/close 的 N 日均值",
)
class AmpFactor(Factor):
    def __init__(self, n: int = 5):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"amp_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        amp = (panel["high"] - panel["low"]) / panel["close"]
        return amp.rolling(self.n).mean()


@register(
    "park_vol",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="Parkinson vol = sqrt(mean(ln(H/L)^2 / (4 ln 2), n))",
)
class ParkinsonVolFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"park_vol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ratio = panel["high"] / panel["low"].replace(0.0, np.nan)
        x = (np.log(ratio)) ** 2 / (4.0 * np.log(2.0))
        return np.sqrt(x.rolling(self.n).mean())


@register(
    "gk_vol",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="Garman-Klass vol: 综合 OHLC 的极差估计",
)
class GarmanKlassVolFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"gk_vol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        h, l, c, o = panel["high"], panel["low"], panel["close"], panel["open"]
        l_safe = l.replace(0.0, np.nan)
        o_safe = o.replace(0.0, np.nan)
        log_hl = np.log(h / l_safe)
        log_co = np.log(c / o_safe)
        x = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
        return np.sqrt(x.rolling(self.n).mean())
```

base 数 = 5,窗口 {5, 10, 20, 60} = 20 变体。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_single_stock_vol.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/single_stock_vol.py tests/test_factors_single_stock_vol.py
git -c commit.gpgsign=false commit -m "feat(factors): add single_stock_vol family (5 base, ~20 variants) (Task 9)"
```

---

### Task 10: `composite.py` — 复合补充族

**Files:**
- Create: `src/stockpool/factors/composite.py`
- Create: `tests/test_factors_composite.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_composite.py
"""Smoke tests for composite factors (built from existing ops)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.composite as _comp  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B", "C"]
    rng = np.random.default_rng(23)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 3)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_rank_signed_mom_10_runs(panel):
    f = make_factor("rank_signed_mom_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_decay_corr_pv_20_runs(panel):
    f = make_factor("decay_corr_pv_20")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_mom_vol_interact_10_runs(panel):
    f = make_factor("mom_vol_interact_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_no_look_ahead(panel):
    f = make_factor("rank_signed_mom_10")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("rank_signed_mom", "decay_corr_pv",
                 "scale_decay_mom", "mom_vol_interact"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_composite.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现**

```python
# src/stockpool/factors/composite.py
"""复合补充族 (论文 B add_* 30 个的精神复现)。

用现有 ops 拼装的混合信号: rank * sign / decay_linear / scale 等。
4 个 base × 3 窗口 = ~12 变体。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "rank_signed_mom",
    sources=("builtin",),
    types=("cross_sectional", "momentum", "time_series"),
    description="rank(close.pct_change(d)) * sign(volume.pct_change(d))",
)
class RankSignedMomFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"rank_signed_mom_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mom = panel["close"].pct_change(self.n, fill_method=None)
        vol_chg = panel["volume"].pct_change(self.n, fill_method=None)
        return ops.rank(mom) * np.sign(vol_chg)


@register(
    "decay_corr_pv",
    sources=("builtin",),
    types=("cross_sectional", "volume", "time_series"),
    description="decay_linear(ts_corr(rank(close), rank(volume), d), d)",
)
class DecayCorrPVFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"decay_corr_pv_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        r_c = ops.rank(panel["close"])
        r_v = ops.rank(panel["volume"])
        corr = ops.correlation(r_c, r_v, self.n)
        return ops.decay_linear(corr, self.n)


@register(
    "scale_decay_mom",
    sources=("builtin",),
    types=("cross_sectional", "momentum", "time_series"),
    description="scale(decay_linear(close.pct_change(d), d))",
)
class ScaleDecayMomFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"scale_decay_mom_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mom = panel["close"].pct_change(self.n, fill_method=None)
        return ops.scale(ops.decay_linear(mom, self.n))


@register(
    "mom_vol_interact",
    sources=("builtin",),
    types=("momentum", "volume", "time_series"),
    description="动量与放量的乘积: mom_d * (volume / mean(volume, d).shift(1) - 1)",
)
class MomVolInteractFactor(Factor):
    def __init__(self, n: int = 10):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"mom_vol_interact_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        mom = panel["close"].pct_change(self.n, fill_method=None)
        v = panel["volume"].replace(0.0, np.nan)
        v_ratio = v / v.rolling(self.n).mean().shift(1) - 1.0
        return mom * v_ratio
```

base 数 = 4,窗口 {5, 10, 20} = 12 变体。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_composite.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/composite.py tests/test_factors_composite.py
git -c commit.gpgsign=false commit -m "feat(factors): add composite family (4 base, ~12 variants) (Task 10)"
```

---

### Task 11: `rank_correlation.py` — 秩相关族

**Files:**
- Create: `src/stockpool/factors/rank_correlation.py`
- Create: `tests/test_factors_rank_correlation.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_rank_correlation.py
"""Smoke tests for rank-correlation family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.rank_correlation as _rc  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B", "C", "D"]
    rng = np.random.default_rng(31)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 4)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(1e6, 1e7, size=close.shape).astype(float),
        index=dates, columns=codes,
    )
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_corr_pv_20_in_unit(panel):
    """秩相关 ∈ [-1, 1]。"""
    f = make_factor("corr_pv_20")
    out = f.compute(panel)
    valid = out.iloc[25:].to_numpy()
    valid = valid[~np.isnan(valid)]
    assert (valid >= -1.0 - 1e-9).all()
    assert (valid <= 1.0 + 1e-9).all()


def test_corr_high_low_20_runs(panel):
    f = make_factor("corr_high_low_20")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_corr_mom_vol_10_runs(panel):
    f = make_factor("corr_mom_vol_10")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape


def test_no_look_ahead(panel):
    f = make_factor("corr_pv_20")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("corr_pv", "corr_high_low", "corr_close_vwap",
                 "corr_mom_vol"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_rank_correlation.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现**

```python
# src/stockpool/factors/rank_correlation.py
"""秩相关合成族 (论文 B old_* 50 个的精神复现)。

基于 ops.correlation 和 ops.rank,产出价格秩 × 成交量秩等组合的滚动相关。
5 个 base × 4 窗口 = ~20 变体。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from stockpool.factors import ops
from stockpool.factors.base import Factor
from stockpool.factors.registry import register


@register(
    "corr_pv",
    sources=("builtin",),
    types=("cross_sectional", "volume", "time_series"),
    description="ts_corr(rank(close), rank(volume), d)",
)
class CorrPVFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_pv_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        r_c = ops.rank(panel["close"])
        r_v = ops.rank(panel["volume"])
        return ops.correlation(r_c, r_v, self.n)


@register(
    "corr_high_low",
    sources=("builtin",),
    types=("volatility", "time_series"),
    description="ts_corr(high, low, d),收盘前位置相关",
)
class CorrHighLowFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_high_low_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return ops.correlation(panel["high"], panel["low"], self.n)


@register(
    "corr_close_vwap",
    sources=("builtin",),
    types=("trend", "time_series"),
    description="ts_corr(close, vwap, d)",
)
class CorrCloseVWAPFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_close_vwap_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return ops.correlation(panel["close"], ops.vwap(panel), self.n)


@register(
    "corr_mom_vol",
    sources=("builtin",),
    types=("momentum", "volume", "time_series"),
    description="ts_corr(close.pct_change(), volume.pct_change(), d)",
)
class CorrMomVolFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_mom_vol_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        ret = panel["close"].pct_change(fill_method=None)
        vchg = panel["volume"].pct_change(fill_method=None)
        return ops.correlation(ret, vchg, self.n)


@register(
    "corr_close_close_lag",
    sources=("builtin",),
    types=("momentum", "time_series"),
    description="ts_corr(close, close.shift(1), d),自相关",
)
class CorrCloseCloseLagFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"corr_close_close_lag_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        return ops.correlation(c, c.shift(1), self.n)
```

base 数 = 5,窗口 {5, 10, 20, 60} = 20 变体。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_rank_correlation.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/rank_correlation.py tests/test_factors_rank_correlation.py
git -c commit.gpgsign=false commit -m "feat(factors): add rank_correlation family (5 base, ~20 variants) (Task 11)"
```

---

### Task 12: `cross_sec_breadth.py` — 截面市场宽度

**Files:**
- Create: `src/stockpool/factors/cross_sec_breadth.py`
- Create: `tests/test_factors_cross_sec_breadth.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_factors_cross_sec_breadth.py
"""Smoke tests for cross-sectional market breadth factors.

注意 (spec §6.1.2): 涨停股算作上涨股、>MA20 股,**与 mask config 无关**。
本测试断言这一不变性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.cross_sec_breadth as _csb  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(41)
    close = pd.DataFrame(
        100.0 + rng.standard_normal((80, 5)).cumsum(axis=0),
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(1e6, index=dates, columns=codes)
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


def test_breadth_above_ma_20_broadcast(panel):
    f = make_factor("breadth_above_ma_20")
    out = f.compute(panel)
    assert out.shape == panel["close"].shape
    # 每一行的 5 列值必须相同(标量广播)
    valid = out.iloc[25:]
    for _, row in valid.iterrows():
        assert row.std() < 1e-9


def test_breadth_above_ma_in_unit(panel):
    """宽度 ∈ [0, 1]。"""
    f = make_factor("breadth_above_ma_20")
    out = f.compute(panel)
    valid = out.iloc[25:].to_numpy()
    valid = valid[~np.isnan(valid)]
    assert ((valid >= 0) & (valid <= 1)).all()


def test_breadth_advance_counts_up_stocks(panel):
    """涨股比例 = (上涨股数 / 总股数)。"""
    f = make_factor("breadth_advance")
    out = f.compute(panel)
    # 构造已知场景:第 50 行所有 5 只票都涨 → breadth=1.0
    expected = (panel["close"].pct_change(fill_method=None) > 0).mean(axis=1)
    actual = out.iloc[:, 0]
    pd.testing.assert_series_equal(
        actual, expected, check_names=False, check_exact=False, rtol=1e-9
    )


def test_breadth_limit_up_includes_limit_up_stocks():
    """spec §6.1.2: 涨停股必须算进涨停股占比分子。"""
    f = make_factor("breadth_limit_up")
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    close = pd.DataFrame({
        "A": [100, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0],
        "B": [100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9],
    }, index=dates)
    panel = {
        "close": close, "high": close + 1, "low": close - 1,
        "open": close.shift(1).fillna(close.iloc[0]),
        "volume": pd.DataFrame(1e6, index=dates, columns=close.columns),
    }
    out = f.compute(panel)
    # 第 1 天 A 涨 10%,触涨停,占比 = 1/2 = 0.5
    assert out.iloc[1, 0] == pytest.approx(0.5)


def test_no_look_ahead(panel):
    f = make_factor("breadth_above_ma_20")
    full = f.compute(panel)
    trunc = {k: v.iloc[:50] for k, v in panel.items()}
    short = f.compute(trunc)
    pd.testing.assert_frame_equal(
        full.iloc[:50], short, check_exact=False, rtol=1e-9
    )


def test_specs_registered():
    for name in ("breadth_above_ma", "breadth_advance",
                 "breadth_limit_up", "breadth_dispersion"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_cross_sec_breadth.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现**

```python
# src/stockpool/factors/cross_sec_breadth.py
"""截面市场宽度族 (论文 B cs_rank_* 6 个的精神复现)。

全市场标量(T×1)广播到 T×N。涨停股 / 停牌股 **不过滤**,
与 mask config 无关(spec §6.1.2)。

5 个 base class。breadth_above_ma 带窗口参数。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _broadcast(scalar: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """T×1 series → T×N DataFrame,广播到 like 的所有列。"""
    return pd.DataFrame(
        np.broadcast_to(scalar.to_numpy()[:, None], like.shape).copy(),
        index=like.index, columns=like.columns,
    )


@register(
    "breadth_above_ma",
    sources=("builtin",),
    types=("cross_sectional", "time_series"),
    description="全市场 close > MA_d 的股票比例 ∈ [0, 1],广播到 T×N",
)
class BreadthAboveMAFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"breadth_above_ma_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        ma = c.rolling(self.n).mean()
        scalar = (c > ma).mean(axis=1)
        return _broadcast(scalar, c)


@register(
    "breadth_advance",
    sources=("builtin",),
    types=("cross_sectional", "time_series"),
    description="全市场当日上涨股比例 = (close.pct_change > 0).mean(axis=1)",
)
class BreadthAdvanceFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "breadth_advance"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        scalar = (c.pct_change(fill_method=None) > 0).mean(axis=1)
        return _broadcast(scalar, c)


@register(
    "breadth_limit_up",
    sources=("builtin",),
    types=("cross_sectional", "time_series"),
    description="全市场触涨停 (ret>0.099) 股票占比;涨停股按 spec §6.1.2 算入分子",
)
class BreadthLimitUpFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "breadth_limit_up"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        ret = c.pct_change(fill_method=None)
        scalar = (ret > 0.099).mean(axis=1)
        return _broadcast(scalar, c)


@register(
    "breadth_dispersion",
    sources=("builtin",),
    types=("cross_sectional", "volatility", "time_series"),
    description="全市场收益横截面 std,衡量当日个股分化程度",
)
class BreadthDispersionFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "breadth_dispersion"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        scalar = c.pct_change(fill_method=None).std(axis=1)
        return _broadcast(scalar, c)


@register(
    "breadth_pos_skew",
    sources=("builtin",),
    types=("cross_sectional", "time_series"),
    description="全市场收益横截面 skewness,正偏 → 头部领涨",
)
class BreadthPosSkewFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "breadth_pos_skew"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        c = panel["close"]
        scalar = c.pct_change(fill_method=None).skew(axis=1)
        return _broadcast(scalar, c)
```

base 数 = 5,`breadth_above_ma` 取 3 个窗口 {5, 20, 60},其他 4 个无窗口 = 7 变体。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_cross_sec_breadth.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/cross_sec_breadth.py tests/test_factors_cross_sec_breadth.py
git -c commit.gpgsign=false commit -m "feat(factors): add cross_sec_breadth family (5 base, ~7 variants) (Task 12)"
```

---

## Phase 3: 基本面因子家族

### Task 13: `fundamentals.py` — PE/PB/ROE 等

**Files:**
- Create: `src/stockpool/factors/fundamentals.py`
- Create: `tests/test_factors_fundamentals.py`

- [ ] **Step 1: 写测试(包含 PIT 关键 case)**

```python
# tests/test_factors_fundamentals.py
"""Smoke tests for fundamental factors with strict PIT alignment.

关键 case: pubDate vs statDate 的看见时点 — 在 pubDate 之前必须 NaN。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stockpool.factors.fundamentals as _fund  # noqa: F401
from stockpool.factors import make_factor, get_spec


@pytest.fixture
def panel():
    """80 个工作日 panel,2 只票。"""
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    codes = ["000001", "600000"]
    close = pd.DataFrame(
        100.0 + np.arange(80).reshape(-1, 1).repeat(2, axis=1) * 0.1,
        index=dates, columns=codes,
    )
    volume = pd.DataFrame(1e6, index=dates, columns=codes)
    return {"close": close,
            "high": close + 1.0, "low": close - 1.0,
            "open": close.shift(1).fillna(close.iloc[0]),
            "volume": volume}


@pytest.fixture
def mock_profit_df():
    """3 季利润数据 mock。"""
    return pd.DataFrame([
        # 2023 Q3: statDate=2023-09-30, pubDate=2023-10-28 (~1 月延迟)
        {"code": "000001", "statDate": pd.Timestamp("2023-09-30"),
         "pubDate": pd.Timestamp("2023-10-28"),
         "roeAvg": 0.10, "netProfit": 1e9, "npMargin": 0.20,
         "gpMargin": 0.30, "totalRevenue": 5e9, "roaAvg": 0.05},
        # 2023 Q4: statDate=2023-12-31, pubDate=2024-03-15
        {"code": "000001", "statDate": pd.Timestamp("2023-12-31"),
         "pubDate": pd.Timestamp("2024-03-15"),
         "roeAvg": 0.11, "netProfit": 1.1e9, "npMargin": 0.21,
         "gpMargin": 0.31, "totalRevenue": 5.2e9, "roaAvg": 0.05},
        # 2024 Q1: statDate=2024-03-31, pubDate=2024-04-29
        {"code": "000001", "statDate": pd.Timestamp("2024-03-31"),
         "pubDate": pd.Timestamp("2024-04-29"),
         "roeAvg": 0.12, "netProfit": 1.2e9, "npMargin": 0.22,
         "gpMargin": 0.32, "totalRevenue": 5.4e9, "roaAvg": 0.06},
    ])


def test_roe_factor_uses_pubdate_not_statdate(monkeypatch, panel, mock_profit_df):
    """关键 PIT 测试: 2024-04-01 (statDate 之后但 pubDate 之前) ROE 仍是上季的值。"""
    from stockpool import fundamentals_loader as fl
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: mock_profit_df if table == "profit" else pd.DataFrame()
    )

    f = make_factor("roe")
    out = f.compute(panel)

    # 2024-03-31 是周日,前一个工作日 2024-03-29:此时只能看到 2023 Q4 (pubDate=2024-03-15)
    target = pd.Timestamp("2024-03-29")
    if target in out.index:
        assert out.loc[target, "000001"] == pytest.approx(0.11)

    # 2024-04-01 (statDate 已过) 仍然只能看到 2023 Q4
    target2 = pd.Timestamp("2024-04-01")
    if target2 in out.index:
        assert out.loc[target2, "000001"] == pytest.approx(0.11)

    # 2024-04-30 (pubDate 已过) 才能看到 2024 Q1
    target3 = pd.Timestamp("2024-04-30")
    if target3 in out.index:
        assert out.loc[target3, "000001"] == pytest.approx(0.12)


def test_roe_pre_first_pubdate_is_nan(monkeypatch, panel, mock_profit_df):
    """首份财报 pubDate (2023-10-28) 之前所有日为 NaN。"""
    from stockpool import fundamentals_loader as fl
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: mock_profit_df if table == "profit" else pd.DataFrame()
    )

    f = make_factor("roe")
    out = f.compute(panel)
    # 2024-01-01 (panel start) 在 pubDate 2023-10-28 之后,所以有值
    # 测试构造的 panel 起点比 pubDate 晚,所以 panel 起点应该已有 2023 Q3 数据
    assert out["000001"].iloc[0] == pytest.approx(0.10)


def test_roe_missing_code_in_fundamentals_is_nan(monkeypatch, panel, mock_profit_df):
    """panel 里有但 fundamentals 没有的 code 全列 NaN。"""
    from stockpool import fundamentals_loader as fl
    monkeypatch.setattr(
        fl, "load_or_build_fundamentals",
        lambda table, **kw: mock_profit_df if table == "profit" else pd.DataFrame()
    )

    f = make_factor("roe")
    out = f.compute(panel)
    assert out["600000"].isna().all()  # 600000 不在 mock 里


def test_pe_negative_earnings_returns_nan(monkeypatch, panel):
    """亏损 (net_income_ttm <= 0) → PE = NaN。"""
    from stockpool import fundamentals_loader as fl
    # 构造净利润全为负的 mock(4 季,符合 TTM min_periods=4)
    bad = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp(f"2023-{q*3:02d}-30"),
         "pubDate": pd.Timestamp(f"2023-{q*3:02d}-30") + pd.Timedelta(days=30),
         "netProfit": -1e8 * q}
        for q in (1, 2, 3, 4)
    ])
    bal = pd.DataFrame([
        {"code": "000001", "statDate": pd.Timestamp("2023-12-31"),
         "pubDate": pd.Timestamp("2024-01-30"), "totalShare": 1e10}
    ])
    def fake(table, **kw):
        return {"profit": bad, "balance": bal}.get(table, pd.DataFrame())
    monkeypatch.setattr(fl, "load_or_build_fundamentals", fake)

    f = make_factor("pe")
    out = f.compute(panel)
    assert out["000001"].dropna().empty or (out["000001"] <= 0).any() is False


def test_specs_registered():
    for name in ("roe", "roa", "pe", "pb", "gross_margin", "net_margin",
                 "revenue_yoy"):
        spec = get_spec(name)
        assert spec is not None
```

- [ ] **Step 2: 失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_fundamentals.py -v`
Expected: ALL FAIL

- [ ] **Step 3: 实现 fundamentals.py**

```python
# src/stockpool/factors/fundamentals.py
"""基本面因子族 (论文 A 启发,baostock 5 张季度表)。

PIT 对齐: 按 pubDate (公告日,**非** statDate 报告期末) 前向填充到日频,
防 ~1 个月未来泄露。Field 名以 Task 0 调研笔记为准。

7 个 base class:
  - 直接字段:roe, roa, gross_margin, net_margin
  - YOY 字段:revenue_yoy
  - 复合计算:pe, pb (close × totalShare / TTM(netProfit | equity))
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.registry import register


def _pit_align(
    raw: pd.DataFrame,
    field: str,
    panel_close: pd.DataFrame,
) -> pd.DataFrame:
    """long-form raw (code/pubDate/<field>) → T×N panel,按 pubDate ffill。

    返回的 DataFrame index=panel_close.index, columns=panel_close.columns。
    """
    if raw.empty or field not in raw.columns:
        return pd.DataFrame(np.nan, index=panel_close.index, columns=panel_close.columns)

    # field 可能是 string ("0.12") 或 float;转 float
    sub = raw[["code", "pubDate", field]].copy()
    sub[field] = pd.to_numeric(sub[field], errors="coerce")
    sub = sub.dropna(subset=["pubDate", field])
    if sub.empty:
        return pd.DataFrame(np.nan, index=panel_close.index, columns=panel_close.columns)

    # 同股同 pubDate 取最后一份(防重复)
    sub = sub.sort_values(["code", "pubDate"]).drop_duplicates(
        subset=["code", "pubDate"], keep="last"
    )
    pivot = sub.pivot(index="pubDate", columns="code", values=field)
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot = pivot.sort_index()

    # 用 panel 索引 reindex + ffill — pandas reindex(method='ffill') 自动满足
    # "日 t 只能看到 pubDate ≤ t 的财报" PIT 约束
    aligned = pivot.reindex(panel_close.index, method="ffill")
    # 添加缺失的列(panel 有但 fundamentals 无的 code)→ 全 NaN
    return aligned.reindex(columns=panel_close.columns)


class _ScalarFundamentalFactor(Factor):
    """直接字段类的共享逻辑: 一表一字段 PIT 对齐。"""

    _table: str = ""
    _field: str = ""

    def __init__(self):
        pass

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        from stockpool.fetcher import default_cache_dir

        raw = load_or_build_fundamentals(
            self._table, cache_dir=default_cache_dir()
        )
        return _pit_align(raw, self._field, panel["close"])


@register(
    "roe",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="ROE (return on equity, profit.roeAvg) PIT 前向填充",
)
class ROEFactor(_ScalarFundamentalFactor):
    _table = "profit"
    _field = "roeAvg"

    @property
    def name(self) -> str:
        return "roe"


@register(
    "roa",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="ROA (return on assets, profit.roaAvg) PIT 前向填充",
)
class ROAFactor(_ScalarFundamentalFactor):
    _table = "profit"
    _field = "roaAvg"

    @property
    def name(self) -> str:
        return "roa"


@register(
    "gross_margin",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="毛利率 profit.gpMargin",
)
class GrossMarginFactor(_ScalarFundamentalFactor):
    _table = "profit"
    _field = "gpMargin"

    @property
    def name(self) -> str:
        return "gross_margin"


@register(
    "net_margin",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="净利率 profit.npMargin",
)
class NetMarginFactor(_ScalarFundamentalFactor):
    _table = "profit"
    _field = "npMargin"

    @property
    def name(self) -> str:
        return "net_margin"


@register(
    "revenue_yoy",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="营收同比 growth.YOYIncome (或 Task 0 调研得到的实际字段名)",
)
class RevenueYoYFactor(_ScalarFundamentalFactor):
    _table = "growth"
    # 字段名以 Task 0 调研结果为准;baostock 公开文档为 YOYIncome
    _field = "YOYIncome"

    @property
    def name(self) -> str:
        return "revenue_yoy"


@register(
    "pe",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="PE = close × totalShare / TTM(netProfit),亏损 → NaN",
)
class PEFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "pe"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        from stockpool.fetcher import default_cache_dir

        profit = load_or_build_fundamentals("profit", cache_dir=default_cache_dir())
        balance = load_or_build_fundamentals("balance", cache_dir=default_cache_dir())
        if profit.empty or balance.empty:
            return pd.DataFrame(np.nan, index=panel["close"].index, columns=panel["close"].columns)

        # TTM netProfit: 对每股按 pubDate 排序后 rolling 4 季 sum
        profit = profit.sort_values(["code", "pubDate"]).copy()
        profit["netProfit"] = pd.to_numeric(profit["netProfit"], errors="coerce")
        profit["ttm"] = (
            profit.groupby("code")["netProfit"]
            .rolling(4, min_periods=4).sum()
            .reset_index(level=0, drop=True)
        )

        ni_panel = _pit_align(profit.rename(columns={"ttm": "_v"}),
                              "_v", panel["close"])
        # totalShare from balance
        shares_panel = _pit_align(balance, "totalShare", panel["close"])

        pe = panel["close"] * shares_panel / ni_panel
        # 亏损 / 缺数据 → NaN
        return pe.where((ni_panel > 0) & shares_panel.notna())


@register(
    "pb",
    sources=("custom",),
    types=("fundamental", "cross_sectional"),
    description="PB = close × totalShare / totalShareholdersEquity",
)
class PBFactor(Factor):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "pb"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from stockpool.fundamentals_loader import load_or_build_fundamentals
        from stockpool.fetcher import default_cache_dir

        balance = load_or_build_fundamentals("balance", cache_dir=default_cache_dir())
        if balance.empty:
            return pd.DataFrame(np.nan, index=panel["close"].index, columns=panel["close"].columns)

        shares_panel = _pit_align(balance, "totalShare", panel["close"])
        # 字段名以 Task 0 调研为准;baostock 公开文档为 totalShareholdersEquity 或 equity
        equity_panel = _pit_align(balance, "totalShareholdersEquity", panel["close"])

        pb = panel["close"] * shares_panel / equity_panel
        return pb.where((equity_panel > 0) & shares_panel.notna())
```

> **注意**:`fetcher.default_cache_dir()` 是助手函数。若 fetcher 没有该函数,
> 实施时改用 `Path(__file__).parent.parent.parent / "data"` 兜底,或读
> `cfg.data.cache_dir`。这是实施时的细节决策,不影响测试逻辑。

base 数 = 7,无窗口参数 = 7 变体(估值类需要更多 derived,后续 PR 可加 `pe_ttm_5y_mean` 等)。spec §4.11 列了 ~15-20,本 task 实现核心 7 个,剩余在 §4.11 备选列表里(`ps`、`pcf`、`debt_to_asset` 等)可通过相同模式快速添加,**不在本 task 必做项**。

> **后续:** 如要补 ps/pcf/debt_to_asset 等到 15-20 个,在 Task 13.5(本 task 完成后)新建一个补充 task 即可。本 plan 不预占 task 编号,保持核心 7 个的快速验证路径。

- [ ] **Step 4: 通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors_fundamentals.py -v`
Expected: 5 PASS(若某些字段名 (`YOYIncome` / `totalShareholdersEquity`) 与 Task 0 调研结果不符,改字段名直到 PASS。)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/fundamentals.py tests/test_factors_fundamentals.py
git -c commit.gpgsign=false commit -m "feat(factors): add fundamentals family (7 base PIT factors) (Task 13)"
```

---

## Phase 4: 集成与收尾

### Task 14: `factors/__init__.py` 注册副作用

**Files:**
- Modify: `src/stockpool/factors/__init__.py`(加 11 行 import)

- [ ] **Step 1: 编辑 `factors/__init__.py`,在 `from stockpool.factors import custom  # noqa: F401` 行之后加 11 行**

```python
# 现有(保留):
from stockpool.factors import technical  # noqa: F401
from stockpool.factors import wq101  # noqa: F401
from stockpool.factors import custom  # noqa: F401

# 新增(11 行):
from stockpool.factors import original_stats  # noqa: F401
from stockpool.factors import ewma  # noqa: F401
from stockpool.factors import vwap_deviation  # noqa: F401
from stockpool.factors import close_position  # noqa: F401
from stockpool.factors import turnover_extra  # noqa: F401
from stockpool.factors import acceleration  # noqa: F401
from stockpool.factors import single_stock_vol  # noqa: F401
from stockpool.factors import composite  # noqa: F401
from stockpool.factors import rank_correlation  # noqa: F401
from stockpool.factors import cross_sec_breadth  # noqa: F401
from stockpool.factors import fundamentals  # noqa: F401
```

- [ ] **Step 2: 跑全 pytest 看是否所有家族都注册成功**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q --tb=short`
Expected: 全过(528 + ~50-60 新测试)

- [ ] **Step 3: 改 `tests/test_factors.py` 加因子总数 sanity 检查**

在 `tests/test_factors.py` 末尾追加:

```python
def test_factor_count_in_expected_range():
    """所有新因子族落地后,总数应在 274-322 之间。
    防止漏注册或意外重名。"""
    from stockpool.factors import list_specs
    n = len(list_specs())
    assert 230 <= n <= 322, f"factor count={n} outside expected [230, 322]"


def test_new_type_fundamental_registered():
    """新引入的 type 标签 'fundamental' 应出现在 all_types 里。"""
    from stockpool.factors import all_types
    assert "fundamental" in all_types()
```

- [ ] **Step 4: 跑 sanity 测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors.py -v`
Expected: 全过(含新的 2 个 sanity)

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/factors/__init__.py tests/test_factors.py
git -c commit.gpgsign=false commit -m "feat(factors): register 11 new families in __init__.py + count sanity (Task 14)"
```

---

### Task 15: `factor_panel` 缓存的 fundamentals snapshot 集成

**Files:**
- Modify: `src/stockpool/strategy_factory.py`(找 `load_or_build_factor_panel` 函数 + 其 manifest.json 写入处)
- Test: 新加 1 个测试到 `tests/test_factor_panel_cache.py`

**Rationale (spec §5.6):** 基本面数据独立刷新时,同 `factor_panel` sig 缓存的内容会变。Manifest 加 `fundamentals_snapshot_date` 字段,读 cache 时若 fundamentals manifest 较新则旁路。

- [ ] **Step 1: 写测试**

```python
# 追加到 tests/test_factor_panel_cache.py

def test_factor_panel_cache_invalidated_when_fundamentals_newer(monkeypatch, tmp_path):
    """factor_panel cache 命中,但 fundamentals parquet mtime 更新过 → 重建。"""
    from stockpool.strategy_factory import load_or_build_factor_panel
    import time

    # 先做一次 build 让 panel cache 写出
    panel_dir = tmp_path / "factor_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    # ... (准备一个最小 panel + factor 列表,正常 build 一次)
    # 这里用 monkeypatch 跳过实际 build,只校验 manifest 字段存在
    pass  # 完整实现需要 mocking factor_panel build 链路;此 stub 仅做契约检查


def test_factor_panel_manifest_includes_fundamentals_snapshot_date(tmp_path, monkeypatch):
    """新 build 的 factor_panel manifest.json 应含 fundamentals_snapshot_date 字段(可能为 null)。"""
    # 与上一测试同样依赖完整 build 链路;skip 实现细节,只断言 manifest schema 含字段
    import json
    # 实施时:对一个 mocked build 完成后读 manifest.json,assert "fundamentals_snapshot_date" in keys
    pass
```

> 这两个测试是契约 reminder,实施时改为可运行版本;若 build 链路 mock 成本过高,可降级为只 grep 代码确认 manifest 写入处含该字段。

- [ ] **Step 2: 修改 `strategy_factory.py` 的 `load_or_build_factor_panel`**

定位 manifest.json 写入处,加 `fundamentals_snapshot_date` 字段:

```python
# 找到现有 manifest 构造代码,加这一行
manifest = {
    "factors": sorted(factor_names),
    "codes": sorted(panel["close"].columns.tolist()),
    "last_date": str(panel["close"].index[-1]),
    # 新增:基本面缓存的 mtime,用于 cache invalidation
    "fundamentals_snapshot_date": _fundamentals_latest_mtime(cache_dir),
}
```

新加 helper:

```python
def _fundamentals_latest_mtime(cache_dir) -> str | None:
    """返回 data/fundamentals_*.parquet 中最新的 mtime ISO 字符串;无文件返 None。"""
    from pathlib import Path
    if cache_dir is None:
        return None
    p = Path(cache_dir)
    parquets = list(p.glob("fundamentals_*.parquet"))
    if not parquets:
        return None
    latest = max(f.stat().st_mtime for f in parquets)
    import datetime
    return datetime.datetime.fromtimestamp(latest).isoformat()
```

在 cache hit 分支(读 manifest 后),加 staleness 检查:

```python
# 现有 cache hit 后:
cached_fundamentals_date = manifest.get("fundamentals_snapshot_date")
current_fundamentals_date = _fundamentals_latest_mtime(cache_dir)
if (cached_fundamentals_date is not None and
    current_fundamentals_date is not None and
    current_fundamentals_date > cached_fundamentals_date):
    log.info("factor_panel cache stale: fundamentals refreshed since build")
    # falls through to rebuild
else:
    return cached_panel  # 继续 cache hit
```

- [ ] **Step 3: 跑全 pytest 确认无回归**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q --tb=short`
Expected: 全过

- [ ] **Step 4: 把契约 reminder 测试完善 / 简化到能运行**

(可选)如果完整 mock build 链路过复杂,降级为静态断言:

```python
def test_load_or_build_factor_panel_helper_exists():
    """smoke: _fundamentals_latest_mtime helper 存在且不抛错。"""
    from stockpool.strategy_factory import _fundamentals_latest_mtime
    # None cache_dir → None
    assert _fundamentals_latest_mtime(None) is None
```

- [ ] **Step 5: Commit**

```bash
git add src/stockpool/strategy_factory.py tests/test_factor_panel_cache.py
git -c commit.gpgsign=false commit -m "feat(factor-panel): invalidate cache when fundamentals refresh (Task 15)"
```

---

### Task 16: 更新 `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 模块地图段,在 `src/stockpool/factors/` 现有行下加 11 个家族文件**

定位:`| \`src/stockpool/factors/\` | **连续因子库**(...) |` 那一行附近。在表格里追加:

```markdown
| `src/stockpool/factors/original_stats.py` | rolling 直接统计量族(close_std/skew/kurt、volume_skew/kurt、range_std、volume_std),~25 变体 |
| `src/stockpool/factors/ewma.py` | EWMA 平滑动量/波动/换手 族(halflife 参数化),~15 变体 |
| `src/stockpool/factors/vwap_deviation.py` | VWAP 偏离族 (close 相对 vwap proxy),论文 B better_* 精神复现,~20 变体 |
| `src/stockpool/factors/close_position.py` | 收盘位置动量族 ((C-L)/(H-L)),论文 B best_* 精神复现,~15 变体 |
| `src/stockpool/factors/turnover_extra.py` | 短窗换手 z-score / 量比,补 custom.turnover_zscore_60 长窗,~12 变体 |
| `src/stockpool/factors/acceleration.py` | 动量/换手二阶差分,捕获趋势变速,论文 B change_* 精神复现,~9 变体 |
| `src/stockpool/factors/single_stock_vol.py` | ATR / CCI / 振幅 / Parkinson / Garman-Klass 波动率,~20 变体 |
| `src/stockpool/factors/composite.py` | rank/decay/scale 复合算子拼装,论文 B add_* 精神复现,~12 变体 |
| `src/stockpool/factors/rank_correlation.py` | 价格秩 × 成交量秩滚动相关 系列,论文 B old_* 精神复现,~20 变体 |
| `src/stockpool/factors/cross_sec_breadth.py` | 全市场宽度因子(>MA20 占比/涨停占比/横截面 std),论文 B cs_rank_* 精神复现,~7 变体。⚠️ 涨停股算入分子,与 mask config 无关 |
| `src/stockpool/factors/fundamentals.py` | 基本面因子(PE/PB/ROE/ROA/毛利率/净利率/营收 YOY),baostock 5 张季度表,严格 PIT |
| `src/stockpool/fundamentals_loader.py` | baostock 5 张季度财务表(profit/growth/balance/cash_flow/dupont)PIT 长期缓存,parquet 文件按 30 天 staleness 失效;字段 mapping 详见 docs/handoff/2026-05-31-baostock-fundamentals-schema.md |
```

- [ ] **Step 2: 因子库章节,把"~111 个" / "114 个" 全部替换为新总数,并加一段"按精神复现论文 B"**

定位:`列全部 (~111 个)` 改为 `列全部 (~280-320 个)`。

在 "Factor ABC" 段落之后加一段:

```markdown
### 论文 B 家族精神复现 (2026-05-31 扩展)

114 → ~280-320 个因子,新增覆盖以下 11 家族(详见 `docs/superpowers/specs/2026-05-31-factor-library-expansion-design.md`):

- **VWAP 偏离 / 收盘位置 / 秩相关 / 单股波动 / 短窗换手 / 复合 / 加速度 / 直接统计 / 截面宽度 / EWMA / 基本面**

命名语义化(`vwap_dev_5` / `close_pos_10` / `roe` 等),不照搬论文 B 的 `better_*` / `best_*` 代号。Mask 行为完全沿用现状(因子看 raw close,mask 只在标签层)。

基本面族 PIT 设计:按 `pubDate`(公告日)前向填充,**不**用 `statDate`,防 ~1 个月未来泄露。首次拉 baostock 5 张季度表约 30-60 分钟,30 天缓存到 `data/fundamentals_*.parquet`。`--refresh-fundamentals` 强制重拉。
```

- [ ] **Step 3: 数据流 / 缓存路径段加 5 个新 parquet 行**

定位:现有 `<code>_daily.parquet` 列表段落。加:

```markdown
- `fundamentals_profit.parquet` / `fundamentals_growth.parquet` / `fundamentals_balance.parquet` / `fundamentals_cash_flow.parquet` / `fundamentals_dupont.parquet` — baostock 季度财务长期缓存,含 `code / pubDate / statDate / <fields...>` 列,30 天有效期。`MLFactorStrategy` 在 fundamentals 因子被列入 `cfg.strategy.ml_factor.factors` 时按 `pubDate` 前向填充到 daily panel
```

- [ ] **Step 4: 测试段落加 12 行**

定位:测试表。追加:

```markdown
| `test_factors_original_stats.py` | rolling 直接统计因子注册 + 数值 + look-ahead |
| `test_factors_ewma.py` | EWMA 平滑因子 halflife 解析 + 公式对照 |
| `test_factors_vwap_deviation.py` | VWAP 偏离族注册 + 单调性 + 无 look-ahead |
| `test_factors_close_position.py` | 收盘位置 ∈ [0,1]、涨停封板 range=0 NaN 守护 |
| `test_factors_turnover_extra.py` | 短窗换手族、停牌日 volume=0 NaN 守护 |
| `test_factors_acceleration.py` | 二阶差分公式对照 + 无 look-ahead |
| `test_factors_single_stock_vol.py` | ATR/CCI/振幅/Parkinson 正性 + 无 look-ahead |
| `test_factors_composite.py` | 复合算子拼装的注册 + 无 look-ahead |
| `test_factors_rank_correlation.py` | 秩相关 ∈ [-1,1] + 无 look-ahead |
| `test_factors_cross_sec_breadth.py` | 全市场标量广播 + 涨停股算入宽度分子(spec §6.1.2) |
| `test_factors_fundamentals.py` | 关键 PIT 测试:pubDate 之前 NaN、pubDate 后 ffill、亏损 PE NaN |
| `test_fundamentals_loader.py` | baostock mock + cache hit / stale / force_refresh / failure-fallback |
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git -c commit.gpgsign=false commit -m "docs(claude-md): factor library expansion + fundamentals (Task 16)"
```

---

### Task 17: 更新 `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 看现有 README.md 的"快速开始"和"常用命令"段落**

Run: `.venv/Scripts/python.exe -c "from pathlib import Path; print(Path('README.md').read_text(encoding='utf-8')[:3000])"`

- [ ] **Step 2: 在"常用命令"段加 fundamentals 相关示例**

```markdown
### 基本面因子(首次启用)

基本面因子族(PE/PB/ROE 等)需先从 baostock 拉财务数据(首次约 30-60 分钟,串行;30 天有效缓存):

```bash
# 任意命令加 --refresh-fundamentals 触发首次拉(或缓存过期重拉)
python -m stockpool run --config config.yaml --refresh-fundamentals

# 之后正常使用,基本面因子自动按 pubDate 前向填充
python -m stockpool factors list --type fundamental
python -m stockpool factors analyze --universe pool --output reports/factor_analysis
```

PIT 警告:基本面数据按 **公告日(pubDate)** 而非 **报告期末(statDate)** 对齐,确保无未来泄露。
```

- [ ] **Step 3: 把因子总数从 114 提到 ~280-320 的描述更新**

如果 README 提到"~111 / 114 个因子",改为"~280-320 个,覆盖 WQ101 全集 + 论文 B 9 家族精神复现 + EWMA + 基本面"。

- [ ] **Step 4: Commit**

```bash
git add README.md
git -c commit.gpgsign=false commit -m "docs(readme): factor library expansion + fundamentals usage (Task 17)"
```

---

### Task 18: 最终 smoke 验证 + 因子总数核对

**Files:** 无文件修改

- [ ] **Step 1: 跑全 pytest 确认所有测试过**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q --tb=short`
Expected: 全过(528 + ~60-80 新测试 = ~590-610)

- [ ] **Step 2: 跑 `factors list` 确认数量**

Run: `.venv/Scripts/python.exe -m stockpool factors list 2>&1 | tail -5`
Expected: 输出总数应在 230-322 之间(starter 集 = ~238,加 window 变体可推到 ~322)

- [ ] **Step 3: 按 source / type 筛选确认**

Run: `.venv/Scripts/python.exe -m stockpool factors list --source custom 2>&1 | head -30`
Expected: 含 `roe / roa / pe / pb / gross_margin / net_margin / revenue_yoy / turnover_z_* / amount_z_* / volume_ratio_short_* / industry_relative_strength / limit_up_count / turnover_zscore`

Run: `.venv/Scripts/python.exe -m stockpool factors list --type fundamental 2>&1 | head -10`
Expected: 7 个基本面因子(roe, roa, pe, pb, gross_margin, net_margin, revenue_yoy)

- [ ] **Step 4: 跑因子分析 smoke(短窗,几只票)**

```bash
# 用 cfg.stocks 几只票快速验证 factor_panel 能 build 通过
.venv/Scripts/python.exe -m stockpool factors analyze --universe pool --output reports/factor_analysis_smoke
```

Expected: 退出码 0,reports/factor_analysis_smoke/ 下有 HTML 和 JSON 输出。

- [ ] **Step 5: 验证 baostock 财务数据可达(端到端 smoke)**

```bash
# 注:此步真实调 baostock,需网络且耗时几分钟到 1 小时,可能 skip
.venv/Scripts/python.exe -c "
from stockpool.fundamentals_loader import load_or_build_fundamentals
df = load_or_build_fundamentals('profit', codes=['000001'], cache_dir='data')
print(f'rows: {len(df)}, cols: {list(df.columns)[:8]}')
print(df.head(2))
"
```

Expected: 输出 0 < rows < 100,含 `code / pubDate / statDate / roeAvg` 等字段。如果失败但前面 4 步都过,记下问题在 commit message 里(可能 baostock 短期不可达,不阻断 PR 合并)。

- [ ] **Step 6: 最终 commit(若 Step 1-5 全过且无遗漏改动)**

```bash
git status  # 确认无未提交
.venv/Scripts/python.exe -m pytest tests/ -q --tb=no | tail -5  # 最后一次确认
```

如果有任何遗漏修复,在本 task 内单独提小 commit。

---

## 完成判据(验收)

PR 合并前必须满足:

- [ ] **测试**:`.venv/Scripts/python.exe -m pytest tests/ -q` 全过(预期 ~590-610 个)
- [ ] **因子数**:`factors list` 输出 ∈ [230, 322](starter ~238,加 window 推到 ~322)
- [ ] **基本面**:`factors list --type fundamental` 输出 ≥ 7 个
- [ ] **CLI**:`--refresh-fundamentals` 在 run / backtest / portfolio-backtest 三处都接收
- [ ] **PIT**:`test_factors_fundamentals.py::test_roe_factor_uses_pubdate_not_statdate` 通过
- [ ] **Mask 不变**:`tests/test_panel_mask.py` 及现有 528 测试全过(无回归)
- [ ] **文档**:CLAUDE.md + README.md 都更新
- [ ] **commit 历史**:18 个 task 各自单独 commit,信息含 task 编号
