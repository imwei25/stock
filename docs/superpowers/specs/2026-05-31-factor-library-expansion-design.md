# 因子库扩展 — 设计文档

> 日期:2026-05-31
> 范围:落地论文 B 除 Alpha101 外的 9 个技术家族 + EWMA 小族 + 基本面族,从 114 → ~274-322 因子
> 关联调研:`docs/research/2026-05-31-a-share-quant-survey-comparison.md` §3.3、§3.4
> 关联前置:`docs/superpowers/specs/2026-05-31-tradability-mask-design.md`(本 spec 不改 mask 行为,仅声明新因子如何配合)

---

## 1. 动机

`docs/research/2026-05-31-a-share-quant-survey-comparison.md` §3.3 在对照论文 B
(arXiv 2507.07107)时识别出**因子库覆盖**是仅次于 mask 的次大缺口:

- 当前 114 个因子:WQ101 全集(101)+ 内置技术(10)+ A 股 custom(3)
- 论文 B 213 个因子(去 Alpha101 子集 9 后净增 ~204)分布在 10 个家族,
  论文表 6 给出**因子数→Sharpe 关系**:9→1.52、58→1.74、108→1.89、**213→2.05**
- 关键观察:**多样性 > 单家族质量** — 没有主导家族,9 个家族里任何
  单族贡献 < 0.1 Sharpe,组合贡献 +0.53 Sharpe

论文 A(arXiv 2506.06356)进一步引入**基本面因子**(估值/盈利/杠杆),作为
其 200+ 维特征工程的关键组成;论文 B 因为是合成数据+短窗,没做基本面。

目标:落地论文 B 的 9 个技术家族(除已覆盖的 Alpha101 子集,按精神复现)+
增加 EWMA 小族 + 引入基本面族,把项目因子数从 114 提到 ~274-322,
**期望与 §3.3 一致 +0.1~0.2 Sharpe(小样本上不如论文,量级一致)**。

### 1.1 为什么是"按精神复现"而非逐字对照

论文 B **未公开源码**,213 个因子里非 Alpha101 子集的 ~195 个无公开实现。
论文正文只给出家族名(`better_*` / `best_*` / `old_*` / ...)与一句话描述,
无完整公式。两个选择:

- **逐字复现**:不可能 — 缺源码
- **按精神复现**:每家族基于论文描述 + 常见配方设计 10-25 个变体,
  覆盖该家族的核心思想 + 多窗口参数扫描。论文 B 自己强调"多样性 > 单家族质量",
  这与按精神复现的目标一致

本 spec 采用按精神复现。新因子用**语义化命名**(见 §4),不与论文 B 的
`better_001` 类代号 1:1 映射。

---

## 2. 范围

### 2.1 In Scope

- **9 个技术家族(对应论文 B 除 Alpha101 外的 9 族)+ EWMA 小族**:共 ~145-188 个新技术因子,详见 §4
- **基本面族**:~15-20 个,基于 baostock 5 张季度财务表,**严格 point-in-time**
  (按 `pubDate` 而非 `statDate` 前向填充)
- **基础设施**:新增 `src/stockpool/fundamentals_loader.py` 数据缓存模块,
  pattern 同 `ipo_dates.py` / `industry_map.py`
- **文件组织**:每家族独立文件,共 11 个新 `factors/*.py` 模块
- **注册副作用**:`factors/__init__.py` import 新模块
- **测试**:每家族 smoke 测试 + 基本面 PIT 对齐测试 + 注册表 sanity
- **文档**:CLAUDE.md(模块地图 + 因子库章节 + 测试表)+ README.md
  (命令示例与 `factors list` 输出预期)

### 2.2 Out of Scope

- **mask 行为不动**:沿用 `compute_factor_panel` 不对 panel 应用 mask 的
  post-2026-05-31-refactor 约定。新因子也看原始 close,见 §6
- **mask-aware 因子族**:若日后需"涨停日 NaN-out 后再算因子"的家族,
  作为独立 PR
- **微观结构因子**(跳空/盘前量比/订单失衡):§3.3 已论证 ROI 低,本 PR 不做
- **AdjMSE 损失 / MVO+LW 组合优化 / GBM 数据增强**:§3 改进路线表里的
  其他项,独立 PR
- **预处理流水线**(winsorize / cs z-score / 默认行业中性):§3.4 项,
  独立 PR
- **因子分析重跑** & **selection.json 自动更新**:本 PR 落地后用户手动
  跑 `factors analyze` + `factors pick-by-ic` 即可,不在 spec 范围
- **真实 EPS / BookValue 单股自取**:复用 baostock `query_profit_data` 等
  接口,不引入 Tushare / Wind

---

## 3. 设计原则

### 3.1 Look-ahead 安全

所有新因子在 `compute(panel)` 第 `i` 行只能访问 `panel[fld].iloc[:i+1]`。
具体到实现:

- **时间序列算子**:用 `rolling(d, min_periods=...)` / `shift(d)` / `ewm(...)`,
  pandas 这些算子天然不窥未来
- **基本面前向填充**:严格按 `pubDate` 对齐,日 `t` 只能看到 `pubDate ≤ t` 的
  财报数据(§5.3 详述)
- **横截面**:用 `axis=1` 算子,跨股不跨时,无 look-ahead 风险

### 3.2 Panel-in / Panel-out

所有因子继承 `Factor` ABC,`compute(panel) -> pd.DataFrame`,返回 T×N 宽表对齐
`panel['close']`。**不得 mutate panel**。

### 3.3 NaN 行为

- **窗口期不足**:返回 NaN(由 `rolling.min_periods` 自然产生)
- **除零**:用 `.replace(0.0, np.nan)` 或 `safe_div` 显式 NaN-out
- **基本面 panel 起点前**:目标股 panel 开始时该股若无任何已公告的财报,
  PE/PB 等返回 NaN,前向填充触发后才有值

### 3.4 命名约定

语义化、人话、可识别家族:

| 家族 | 命名 prefix | 例子 |
|------|------------|------|
| VWAP 偏离 | `vwap_dev_*` / `vwap_weighted_mom_*` | `vwap_dev_5`、`vwap_weighted_mom_10` |
| 收盘位置 | `close_pos_*` / `close_pos_cum_*` | `close_pos_5`、`close_pos_cum_20` |
| 秩相关 | `corr_pv_*` / `corr_pp_*` | `corr_pv_20`、`corr_rank_close_vol_10` |
| 单股波动 | `atr_*` / `cci_*` / `amp_*` / `park_vol_*` | `atr_14`、`cci_20`、`amp_5` |
| 短窗换手 | `turnover_z_*` / `amount_z_*` / `volume_ratio_*` | `turnover_z_5`、`amount_z_10` |
| 复合 | `rank_signed_*` / `decay_*` / `scale_decay_*` | `rank_signed_mom_10`、`decay_corr_pv_20` |
| 加速度 | `mom_accel_*` / `vol_accel_*` / `turnover_accel_*` | `mom_accel_5`、`vol_accel_10` |
| 直接统计 | `close_std_*` / `close_skew_*` / `vol_kurt_*` | `close_std_20`、`vol_kurt_60` |
| 截面宽度 | `breadth_above_ma*` / `breadth_limit_up` / `breadth_advance` | `breadth_above_ma20`、`breadth_advance` |
| EWMA | `ewma_*_hl*` | `ewma_momentum_hl10`、`ewma_vol_hl20` |
| 基本面 | 字段名直接用 | `pe`、`pb`、`ps`、`pcf`、`roe`、`roa`、`gross_margin`、`net_margin`、`revenue_yoy`、`profit_yoy`、`debt_to_asset` 等 |

带数字后缀的因子继承 `Factor.from_suffix_args` 默认实现,可在
`config.yaml.strategy.ml_factor.factors` 里直接写 `vwap_dev_5` / `atr_14`
而无需修改 schema。

### 3.5 算子优先级:复用 > 新建

- WQ101 已有 `ops.rank` / `ts_rank` / `ts_corr` / `decay_linear` / `vwap` / `adv`,
  本 spec 不新增算子
- 若家族 X 需要 CCI(typical price 偏离的标准化),写在该家族文件内,
  不污染 `ops.py`(CCI 是 single-purpose,不像 `decay_linear` 可多用)

---

## 4. 家族详表

每家族给出:**描述、典型公式、窗口参数、变体数预估、放置文件**。

### 4.1 VWAP 偏离族 — `factors/vwap_deviation.py`

- **描述**:close 相对当日 VWAP proxy `(H+L+C)/3` 的相对位置 + 量加权动量。
  论文 B `better_*` 28 个的核心
- **公式**:
  - `vwap_dev_d = (close - vwap) / vwap` rolling mean over d
  - `vwap_weighted_mom_d = sum_d((close - vwap) * volume) / sum_d(volume)` ÷ vwap[t]
  - `vwap_above_ratio_d = sum_d(1 if close > vwap else 0) / d`
- **窗口**:`d ∈ {3, 5, 10, 20, 60}` × 4-5 个公式变体 = **~20-25**
- **放置**:`factors/vwap_deviation.py`
- **types**:`("trend", "volume", "time_series")`

### 4.2 收盘位置动量族 — `factors/close_position.py`

- **描述**:close 在当日 high-low 区间的位置 + N 日累积。论文 B `best_*` 21 个
- **NaN 守护**:定义 `pos_raw = (close - low) / (high - low).replace(0.0, np.nan)`。
  涨停封板日 high==low==close 时 range=0 → `pos_raw` 为 NaN(语义:无法判断
  区间位置),自然通过 rolling NaN-safe 传播
- **公式**:
  - `close_pos_d = pos_raw.rolling(d).mean()`(d 日均值)
  - `close_pos_cum_d = (pos_raw - 0.5).rolling(d).sum()`(累积偏离中位,正多空)
  - `close_pos_ema_d = pos_raw.ewm(span=d).mean()`
  - `close_pos_trend_d = (close_pos_5 - close_pos_d) / close_pos_d`(短期相对长期偏离)
- **窗口**:`d ∈ {3, 5, 10, 20, 60}` × 3-4 公式 = **~15-20**
- **放置**:`factors/close_position.py`
- **types**:`("momentum", "time_series")`

### 4.3 秩相关合成族 — `factors/rank_correlation.py`

- **描述**:价格秩与成交量秩的滚动相关,跨股价格秩动量。论文 B `old_*` 50 个
- **公式**:
  - `corr_pv_d = ts_corr(rank(close), rank(volume), d)`(横截面秩然后时序相关)
  - `corr_pp_d = ts_corr(rank(close), rank(close.shift(1)), d)`
  - `corr_rank_close_vol_d = ts_corr(close.rank(), volume.rank(), d)`(时序秩)
  - `corr_high_low_d = ts_corr(high, low, d)`
  - `corr_close_vwap_d = ts_corr(close, vwap, d)`
  - `corr_mom_vol_d = ts_corr(close.pct_change(), volume.pct_change(), d)`
- **窗口**:`d ∈ {5, 10, 20, 60}` × 6 变体 = **~25-30**
- **放置**:`factors/rank_correlation.py`
- **types**:`("cross_sectional", "volume", "time_series")`(每个再决定 industry_neutral 否)

### 4.4 单股波动族 — `factors/single_stock_vol.py`

- **描述**:ATR / CCI / 振幅 / Parkinson 波动率。论文 B `stock_*` 22 个
- **公式**:
  - `atr_d = ewm(true_range, span=d)`,其中 `tr = max(h-l, |h-c_prev|, |l-c_prev|)`
  - `cci_d = (tp - sma(tp, d)) / (0.015 * mad(tp, d))`,tp = (H+L+C)/3
  - `amp_d = mean((high - low) / close, d)`
  - `park_vol_d = sqrt(mean(ln(high/low)^2 / (4 ln 2), d))`(Parkinson 波动率)
  - `gk_vol_d`(Garman-Klass):用 OHLC 全部
- **窗口**:`d ∈ {5, 10, 20, 60}` × 4-5 公式 = **~15-20**
- **放置**:`factors/single_stock_vol.py`
- **types**:`("volatility", "time_series")`

### 4.5 短窗换手族 — `factors/turnover_extra.py`

- **描述**:补 `custom.py:turnover_zscore_60` 之外的短/中窗换手指标。论文 B `extra_*` 14 个
- **公式**:
  - `turnover_z_d = (log(volume) - mean(log(volume), d)) / std(log(volume), d)`,
    `d ∈ {3, 5, 10, 20}`
  - `amount_z_d = (log(volume*close) - mean(log(volume*close), d)) / std(...)`
  - `volume_ratio_d = volume / mean(volume, d).shift(1)`(放/缩量)
  - `volume_concentration_d = sum_d(top_q(volume, 0.2)) / sum_d(volume)`
    (近 d 日成交集中度)
- **窗口**:`d ∈ {3, 5, 10, 20}` × 3-4 公式 = **~10-15**
- **放置**:`factors/turnover_extra.py`
- **types**:`("volume", "time_series")`
- ⚠️ **NaN 处理**:`v.replace(0.0, np.nan)` 必须有,避免停牌日 `log(0)` 污染。
  与现有 `custom.py:TurnoverZScoreFactor` 一致

### 4.6 复合补充族 — `factors/composite.py`

- **描述**:用现有算子拼装的混合信号。论文 B `add_*` 30 个
- **公式**(每个都基于现有 ops):
  - `rank_signed_mom_d = rank(close.pct_change(d)) * sign(volume.pct_change(d))`
  - `decay_corr_pv_d = decay_linear(ts_corr(rank(close), rank(volume), d), d)`
  - `scale_decay_mom_d = scale(decay_linear(close.pct_change(d), d))`
  - `signed_rank_close_pos_d = sign(close_pos_d - 0.5) * rank(close_pos_d)`
  - `mom_vol_interact_d = close.pct_change(d) * volume_ratio_d`
- **窗口**:`d ∈ {5, 10, 20}` × 5-7 公式 = **~20-25**
- **放置**:`factors/composite.py`
- **types**:`("cross_sectional", "time_series")`(部分 industry_neutral)

### 4.7 加速度族 — `factors/acceleration.py`

- **描述**:动量/换手的二阶差分,捕获趋势变速。论文 B `change_*` 5 个
- **公式**:
  - `mom_accel_d = momentum_d - momentum_d.shift(d)`(动量本身的 d 日差)
  - `vol_accel_d = log(volume) - 2*log(volume).shift(d) + log(volume).shift(2d)`
    (二阶差分)
  - `turnover_accel_d = turnover_z_d - turnover_z_d.shift(d)`
  - `mom_jerk_d = (momentum_d - momentum_d.shift(d)) - (momentum_d.shift(d) - momentum_d.shift(2d))`
    (三阶变化)
- **窗口**:`d ∈ {3, 5, 10}` × 2-3 公式 = **~5-8**
- **放置**:`factors/acceleration.py`
- **types**:`("momentum", "time_series")`

### 4.8 直接统计族 — `factors/original_stats.py`

- **描述**:rolling 直接统计量,没有秩或归一化。论文 B `original_*` 28 个
- **公式**:
  - `close_std_d = close.rolling(d).std() / close`
  - `close_skew_d = close.rolling(d).skew()`
  - `close_kurt_d = close.rolling(d).kurt()`
  - `volume_skew_d / volume_kurt_d`
  - `range_std_d = (high-low).rolling(d).std() / close`
  - `volume_mean_d = volume.rolling(d).mean()`(原始量)
  - `volume_std_d = volume.rolling(d).std() / volume.rolling(d).mean()`(变异系数)
- **窗口**:`d ∈ {5, 10, 20, 60}` × 5-7 公式 = **~20-25**
- **放置**:`factors/original_stats.py`
- **types**:`("volatility" | "volume", "time_series")`(按公式分)

### 4.9 截面市场宽度族 — `factors/cross_sec_breadth.py`

- **描述**:全市场宽度指标,每只股拿到的是**同一个全市场标量**(T×1)广播到 N 列。
  论文 B `cs_rank_*` 6 个
- **公式**:
  - `breadth_above_ma_d = (panel_close > panel_close.rolling(d).mean()).mean(axis=1)`
    → broadcast 到 T×N。`d ∈ {5, 20, 60}`
  - `breadth_advance = (panel_close.pct_change() > 0).mean(axis=1)`(全市场涨股比例)
  - `breadth_limit_up = (panel_close.pct_change() > 0.099).mean(axis=1)`
    (涨停股占比)
  - `breadth_above_ema_d = (panel_close > panel_close.ewm(span=d).mean()).mean(axis=1)`
  - `breadth_pos_skew = panel_close.pct_change().skew(axis=1)`(全市场收益分布偏度)
  - `breadth_dispersion = panel_close.pct_change().std(axis=1)`(横截面离散度)
- **变体数**:**~5-8**
- **放置**:`factors/cross_sec_breadth.py`
- **types**:`("cross_sectional", "time_series")`
- ⚠️ **mask 关系**:这族的横截面统计是**全市场**(panel 内所有列)。涨停股算作
  上涨股、>MA20 股,这是论文 B 的口径,与 mask config 无关。Spec 明文记录,
  防止后续误改成"过滤涨停股再算宽度"

### 4.10 EWMA 小族 — `factors/ewma.py`

- **描述**:对动量/波动/换手做 EWMA 平滑,半衰期参数化。论文 B 无对应家族,
  本 spec 自主补
- **公式**:
  - `ewma_momentum_hl{h}`:`close.ewm(halflife=h).mean()` 相对 `close` 偏离
  - `ewma_vol_hl{h}`:`close.pct_change().ewm(halflife=h).std()`(RiskMetrics-like)
  - `ewma_turnover_z_hl{h}`:`log(volume).ewm(halflife=h).std()` 归一化的 z
  - `ewma_close_dev_hl{h}`:`(close - close.ewm(halflife=h).mean()) / close.ewm(halflife=h).std()`
  - `ewma_volume_ratio_hl{h}`:`volume / volume.ewm(halflife=h).mean().shift(1)`
- **半衰期**:`h ∈ {5, 10, 20}` × 5 公式 = **~10-12**
- **放置**:`factors/ewma.py`
- **types**:`("trend" | "volatility" | "volume" | "momentum", "time_series")`

### 4.11 基本面族 — `factors/fundamentals.py` + `fundamentals_loader.py`

- **描述**:估值 / 盈利 / 成长 / 杠杆等核心财务指标。论文 A 表 1 提到的
  4 大类基本面特征,日频前向填充
- **数据来源**:baostock 5 张季度财务表(详见 §5)
- **因子列表**:
  - **估值**:`pe`、`pb`、`ps`、`pcf`(price / cash flow per share)
  - **盈利能力**:`roe`、`roa`、`gross_margin`、`net_margin`
  - **成长**:`revenue_yoy`(营收同比)、`profit_yoy`(净利润同比)、`roe_yoy`
  - **杠杆与流动性**:`debt_to_asset`、`current_ratio`、`quick_ratio`
  - **运营**:`asset_turnover`、`inventory_turnover`
- **变体数**:**~15-20**
- **放置**:`factors/fundamentals.py`(因子定义)+ `src/stockpool/fundamentals_loader.py`
  (数据缓存层)
- **types**:`("fundamental", "cross_sectional")`(新增 "fundamental" type 标签)
- ⚠️ **PIT 对齐**:严格按 `pubDate`,见 §5.3

### 4.12 家族变体数汇总

| 家族 | 文件 | 变体数 |
|---|---|---|
| VWAP 偏离 | `vwap_deviation.py` | 20-25 |
| 收盘位置 | `close_position.py` | 15-20 |
| 秩相关 | `rank_correlation.py` | 25-30 |
| 单股波动 | `single_stock_vol.py` | 15-20 |
| 短窗换手 | `turnover_extra.py` | 10-15 |
| 复合补充 | `composite.py` | 20-25 |
| 加速度 | `acceleration.py` | 5-8 |
| 直接统计 | `original_stats.py` | 20-25 |
| 截面宽度 | `cross_sec_breadth.py` | 5-8 |
| EWMA | `ewma.py` | 10-12 |
| 基本面 | `fundamentals.py` | 15-20 |
| **合计新增** | | **~160-208** |
| **项目总数(含 114 现有)** | | **~274-322** |

---

## 5. 基本面数据基础设施

### 5.1 数据源:baostock 5 张季度财务表

baostock 提供以下接口(均为免费、公开):

| 接口 | 内容 | 关键字段 |
|------|------|---------|
| `query_profit_data(code, year, quarter)` | 盈利能力 | roe、roa、grossProfitMargin、npMargin、totalRevenue 等 |
| `query_growth_data(code, year, quarter)` | 成长性 | YOYEquity、YOYAsset、YOYNI、YOYEPS、YOYPNI 等 |
| `query_balance_data(code, year, quarter)` | 资产负债 | currentRatio、quickRatio、cashRatio、liabilityToAsset 等 |
| `query_cash_flow_data(code, year, quarter)` | 现金流 | CAToAsset、NCAToAsset、tangibleAssetToAsset、ebitToInterest 等 |
| `query_dupont_data(code, year, quarter)` | 杜邦分解 | dupontROE、dupontEquityMultiplier、dupontAssetTurn 等 |

每个接口返回 DataFrame(每股每季一行),含 `code` / `pubDate`(公告日)/
`statDate`(报告期末)/ 各字段。

> ⚠️ **字段名待实现阶段确认**:上表字段是按 baostock 公开文档列出的近似名,
> 实际可能有大小写或细微差异(例如 `npMargin` vs `netProfitMargin`)。
> 实施 plan 第一步是用 `.venv` 真实调一次接口、记录 schema、再写 factor 公式。
> 字段差异属实现细节,不影响本 spec 的架构决策。

估值类(PE/PB/PS/PCF)**不**从 baostock 直接拿(它们要求最新价 + 财报数据
拼接),而是用 `close * shares_outstanding / net_income_ttm` 等公式日频计算
(见 §5.4)。

### 5.2 `fundamentals_loader.py` 设计

复用 `ipo_dates.py` 的 baostock 鉴权 + parquet 缓存模式:

```python
# src/stockpool/fundamentals_loader.py 关键 API

def load_or_build_fundamentals(
    table: str,              # "profit" | "growth" | "balance" | "cash_flow" | "dupont"
    *,
    codes: list[str] | None = None,
    cache_dir: Path | None = None,
    max_age_days: int = 30,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """返回 long-form DataFrame: code / pubDate / statDate / <fields...>.

    缓存到 data/fundamentals_<table>.parquet,30 天有效期。codes=None 拉全市场。
    pubDate 是 datetime64,后续 build_panel 的 PIT 对齐键。
    """
```

- **首次拉数据**:5 张表 × 5500+ 股 × ~16 季(过去 4 年)≈ 440k+ 查询,
  baostock 串行预估 30-60 分钟。**通过 `--limit codes` 子集可加速开发**
- **缓存策略**:30 天有效期 + manifest.json 记录上次拉取日 + 字段 schema 版本
- **失败隔离**:per-stock query 失败 log warning 跳过,不阻断整体

### 5.3 PIT 对齐(关键)

`compute(panel) -> T×N` 实现时,把 long-form 财务 DataFrame **按 `pubDate`**
前向填充到日频:

```python
# 伪代码(factors/fundamentals.py 内)
def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    raw = load_or_build_fundamentals(self.table)  # long-form
    raw = raw.sort_values(["code", "pubDate"])
    pivot = raw.pivot(index="pubDate", columns="code", values=self.field)
    # ❗ 用 pubDate 作 index,然后 reindex 到 panel 的 trading dates 并 ffill
    pivot.index = pd.DatetimeIndex(pivot.index)
    result = pivot.reindex(panel["close"].index, method="ffill")
    return result.reindex(columns=panel["close"].columns)
```

**关键性质**:
- 日 `t` 只能看到 `pubDate ≤ t` 的财报 — 用 `reindex(method="ffill")` 自动满足
- 首份财报 `pubDate` 之前的所有日为 NaN — 符合"未公告则未知"
- ❌ **绝不能用 `statDate`** — `statDate = 2024-03-31` 实际公告可能在 `2024-04-29`,
  用 `statDate` 会泄露 ~1 个月未来信息

**估值类的特殊处理**(`pe` / `pb` / `ps` / `pcf`):
- `pe = close * total_shares / net_income_ttm`
- `net_income_ttm` = 最近 4 季度净利润之和(`profit_data` 表里 `netProfit`)
- `total_shares` 来自 `balance_data` 表里的 `totalShare`
- 注意 TTM 滚动 4 季度时也要按 `pubDate` 排序,以最近一季 `pubDate` 作可用日

### 5.4 估值因子 TTM 计算

```python
# 伪代码,以 PE 为例
profit = load_or_build_fundamentals("profit")  # 含 netProfit, pubDate
balance = load_or_build_fundamentals("balance")  # 含 totalShare, pubDate

# 计算 TTM 净利润:对每个 code,按 pubDate 排序,rolling 4 季和
ni_ttm = (
    profit.sort_values(["code", "pubDate"])
    .groupby("code")["netProfit"]
    .rolling(4, min_periods=4).sum()
    .reset_index(level=0)
)
ni_ttm["pubDate"] = profit.sort_values(["code", "pubDate"])["pubDate"].values

# Pivot + ffill 到日频(对齐 panel)
ni_panel = (
    ni_ttm.pivot(index="pubDate", columns="code", values="netProfit")
    .reindex(panel["close"].index, method="ffill")
)
shares_panel = (
    balance.pivot(index="pubDate", columns="code", values="totalShare")
    .reindex(panel["close"].index, method="ffill")
)

pe = panel["close"] * shares_panel / ni_panel
```

边界:
- 首份财报前 NaN
- TTM 不足 4 季的(新上市股)NaN(`min_periods=4` 强制)
- `ni_panel ≤ 0`(亏损)→ PE 无意义,用 `safe_div` NaN-out

### 5.5 缓存与刷新

- 默认 30 天 max_age,与现有 `ipo_dates.parquet` / `stock_industry_map.parquet` 一致
- CLI 加 `--refresh-fundamentals` flag(独立于 `--refresh-factor-panel`),
  挂在 `cli.py` 的 `run` / `backtest` / `portfolio-backtest` 下
- 本 spec 范围内基本面数据**只走 baostock**;若日后接入 akshare/Tushare 等
  备份源,属独立 PR 范围(列入 §15 follow-up)

### 5.6 `factor_panel` 缓存的 sig 重算

`load_or_build_factor_panel` 当前 sig 由 `(sorted factors + sorted codes + last_date)`
决定。新增基本面因子后,sig 内容**不变**(factors 列表已编码新名,如 `pe`、`roe`)。

但有个新情况:**基本面缓存独立刷新会让同一 sig 的 factor_panel 值变化**。

解决:`factor_panel` 的 manifest.json 加 `fundamentals_snapshot_date` 字段,
记录构建时所用基本面缓存的 manifest 写入日期。读取时若 fundamentals
manifest 较新,旁路 factor_panel 缓存重建。

---

## 6. Mask 关系声明(无新行为)

post-2026-05-31-refactor 的现状(`compute_factor_panel` docstring + CLAUDE.md
panel.py 段):

> mask **不** 应用到因子输入面板 — 时间序列因子需要看真实 close
> (涨停日 +9.9% 本身是有用信号)。Mask 只在 `forward_return_panel` 的双向
> 标签检查(`mask[t] ∧ mask[t+horizon]`)和训练样本 dropna 上生效。

**本 spec 沿用此约定。所有 ~160-208 个新因子都看原始价格,mask 对 `compute()`
完全不可见**。这等价于:
- 启用 mask 与否,新因子的 panel 值**完全相同**(同 sig 同 panel 缓存)
- mask 唯一影响的还是训练标签层 + 通过 `forward_return_panel` NaN 自然 dropna

### 6.1 两个家族的 mask 实现细节(spec 明文锁定)

**6.1.1 短窗换手 `extra_*`**

停牌日 `volume = 0` 会导致 `log(0) = -inf`,污染 rolling z-score。**必须**与
`custom.py:TurnoverZScoreFactor` 一致使用 `v.replace(0.0, np.nan)`:

```python
v = panel["volume"].replace(0.0, np.nan)
lv = np.log(v)
# ... rolling z-score on lv
```

这与 mask config 无关 — 是因子内部的 NaN 守护,无论 mask=on/off 都做。

**6.1.2 截面宽度 `cs_rank_*`**

横截面统计(如 `breadth_above_ma20`)在全市场 panel 上算时,**涨停股算上涨股、
>MA20 股**,这是论文 B 的口径。不过滤涨停股。Spec 明文锁定:

> `cross_sec_breadth.py` 内所有计算均在 raw `panel["close"]` 上进行,
> 不调用 `apply_mask`、不在 mask=on 时改变行为。涨停股算入宽度分子。

这避免后续维护者把 mask 与截面宽度搞混。

### 6.2 不引入 per-factor mask opt-in

最初考虑给少数因子加 `apply_mask=True` 选项,但会:
- 与"mask 在标签层、不动因子输入"的架构决策冲突
- 让同一组因子在 mask=on/off 下 panel 不一致(目前两 arm 共用同 sig panel,
  见 `mask-ab-investigation.md` §四)
- 让 "apply_mask 是工具不是默认" 的边界变模糊

放弃。日后若真需 mask-aware 因子族,作为独立 PR 决策。

---

## 7. 测试策略

每家族独立 smoke 测试文件,覆盖**注册、计算、look-ahead、NaN、单调性 sanity**。
不为每个变体写单测(160+ 个变体不现实),但每家族至少覆盖代表性 2-3 个变体。

### 7.1 测试文件清单

| 测试文件 | 覆盖 |
|----------|------|
| `tests/test_factors_vwap_deviation.py` | 注册名解析、3 个代表变体的数值 sanity、look-ahead 截断不变性 |
| `tests/test_factors_close_position.py` | 同上 |
| `tests/test_factors_rank_correlation.py` | 同上 + 跨股 corr 横截面对齐 |
| `tests/test_factors_single_stock_vol.py` | 同上 + ATR 与公式直算对照、CCI 边界 |
| `tests/test_factors_turnover_extra.py` | 同上 + `volume=0` 日 NaN 守护 |
| `tests/test_factors_composite.py` | 同上 + 复合算子结果与手工拼装对照 |
| `tests/test_factors_acceleration.py` | 同上 + 二阶差分 NaN warmup |
| `tests/test_factors_original_stats.py` | 同上 + skew/kurt 边界 |
| `tests/test_factors_cross_sec_breadth.py` | 全市场标量广播到 N 列正确性、涨停股被算入宽度 |
| `tests/test_factors_ewma.py` | EWMA 半衰期解析、与公式直算对照 |
| `tests/test_factors_fundamentals.py` | PIT 对齐(`pubDate` 而非 `statDate`)、ffill 行为、TTM 4 季 rolling、亏损 PE NaN |
| `tests/test_fundamentals_loader.py` | baostock mock + cache hit / stale / failure-fallback;pattern 同 `test_ipo_dates.py` |

### 7.2 PIT 对齐测试关键 case

在 `test_factors_fundamentals.py` 中显式构造一个 `pubDate < statDate + n_days`
的 mock,验证:
- 日 `t = statDate` 时 PE 返回 NaN(财报尚未公告)
- 日 `t = pubDate` 时 PE 才有值
- 公告日次日继续 ffill

这是本 PR 最关键的反 look-ahead 防线,必须覆盖。

### 7.3 注册表 sanity 测试

`tests/test_factors.py` 已有的注册表测试自动覆盖新因子。额外加一条:

```python
def test_factor_count_in_expected_range():
    """新因子族落地后总数应在 274-322 之间,作为 reminder 防止漏注册或重名。"""
    n = len(list_specs())
    assert 274 <= n <= 322, f"factor count={n} not in expected range"
```

### 7.4 总测试增量

预估约 **60-80 个新测试**,加上现有 528 个,落地后约 **590-610 测试**。
`.venv/Scripts/python.exe -m pytest tests/ -q` 应继续全过。

---

## 8. CLI / 用户接口变更

### 8.1 `factors list` 新输出

新因子自动出现在 `python -m stockpool factors list` 输出里。新增 source 标签 `"paper_b"`
不引入:**所有新因子都用 `("builtin",)` 或 `("custom",)` source 即可**。

具体:
- VWAP / 收盘位置 / 秩相关 / 单股波动 / 复合 / 加速度 / 直接统计 / EWMA / 截面宽度:`("builtin",)`
- 短窗换手 / 基本面:`("custom",)` (与现有 `turnover_zscore_60` / `industry_relative_strength_20` 风格一致)

`--source` 和 `--type` 筛选自动工作。`--type fundamental` 是新引入的 type 标签。

### 8.2 新 CLI 参数

```bash
# cli.py 新增 flag(可挂在 run / backtest / portfolio-backtest 下)
--refresh-fundamentals       # 强制重拉 baostock 财务数据(绕过 30 天缓存)
```

### 8.3 新数据缓存目录

- `data/fundamentals_profit.parquet`
- `data/fundamentals_growth.parquet`
- `data/fundamentals_balance.parquet`
- `data/fundamentals_cash_flow.parquet`
- `data/fundamentals_dupont.parquet`

每文件含 `code / pubDate / statDate / <fields...>` 列。

---

## 9. 文档变更

### 9.1 `CLAUDE.md`

- **模块地图**:`src/stockpool/factors/` 段加 11 个新家族文件描述
- **模块地图**:新增 `src/stockpool/fundamentals_loader.py` 一行
- **因子库章节**:新增"家族变体数表"(§4.12)、命名约定(§3.4)、基本面 PIT
  原则(§5.3)
- **数据流缓存表**:加 `fundamentals_<table>.parquet` 五行
- **测试表**:加 12 个新测试文件行
- **配置段**:`strategy.ml_factor.factors`/`factors_file` 可写入新因子名,
  无 schema 变化
- **已知不支持**:无新限制(但若 baostock 财务接口稳定性差,加一句"基本面
  数据有 30 天缓存,过期 + 网络不通时 PE/PB/ROE 等返回 NaN")

### 9.2 `README.md`

- **快速命令**:加 `--refresh-fundamentals` 例子
- **常用命令**:加一段"基本面因子首次启用"流程示例
- **因子库简介**:总数从 114 改为新数字 + 加一段"按精神复现论文 B 10 家族"

---

## 10. 配置兼容性

- **`MLFactorConfig.factors` 列表**:可直接写入新因子名,如 `[vwap_dev_5, atr_14, roe]`,
  无 schema 变化
- **`factors_file` (HTML picker selection.json)**:用户重跑 picker 即可看到
  新因子。**老 selection.json 仍可用**,但不包含新因子 — 用户自行选择是否
  重选
- **`factor_panels/<sig>/` 缓存**:不变。sig 由 factors 列表决定,加新名自动
  生成新 sig 目录,旧目录不受影响
- **`ml_models/<sig>_*.pkl`**:同上自动失效

完全向后兼容:用户不改 yaml,行为完全不变。

---

## 11. 性能与成本

### 11.1 因子计算

- 单因子 `compute(panel)` 在 4358 票 × 500 bar panel 上典型耗时:
  - 简单 rolling 类(VWAP / 直接统计 / EWMA):~0.1-0.3 秒
  - 秩相关 / argmin / argmax 类:~1-3 秒(`raw=False` 路径)
  - 截面宽度:~0.1 秒(全市场标量计算)
  - 基本面(已缓存):~0.05 秒(pivot + ffill)
- 全 ~160-208 新因子 + 现有 114(共 ~274-322):约 ~5-15 分钟 build 一次 panel
- 已通过 `factor_panels/<sig>/` 缓存,后续 refit 不重算

### 11.2 基本面首次拉取

- 5 张表 × 5500+ 股 × ~16 季 ≈ 440k+ baostock 查询
- baostock 串行约 30-60 分钟,后续 30 天命中缓存秒级
- **开发期 escape hatch**:`--limit <codes>` 只拉子集

### 11.3 存储

- `factor_panels/<sig>/`:每因子一个 parquet(T × N),205 个新因子约 200-500 MB
- `fundamentals_*.parquet`:5 张表合计 ~30-50 MB(long-form,有压缩)
- 总增量 < 1 GB,可接受

---

## 12. 风险与回滚

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| baostock 财务接口偶发返回空 | 中 | 部分股 PE/PB 缺失 | per-stock failure isolation,log warning,NaN 落地 |
| 某新因子全 NaN(类似 `industry_relative_strength` 历史 bug) | 低 | 拒绝该 bar 预测 | 测试覆盖 + `compute()` 内禁止 silent 全 NaN(必要时 raise) |
| 新因子高度共线导致 LightGBM 选择器困惑 | 低 | IC 略降 | 用户跑 `factors pick-by-ic --max-corr 0.6` 选 top-N |
| baostock 字段名升级 | 低 | loader crash | manifest.json 记录 schema 版本,不匹配时强制重拉 + log warning |
| PIT 对齐写错(用 statDate) | 高(实现陷阱) | look-ahead 泄露 | §7.2 专门 case 覆盖 |

### 回滚策略

每家族独立文件 + 注册副作用集中在 `factors/__init__.py`。要回滚某族,
注释掉 `factors/__init__.py` 的 import 即可,无需改其他代码。

完全回滚:revert 单一 PR commit。

---

## 13. 实施顺序(供 writing-plans 参考)

预估工作量在 1500-2500 行代码 + 测试 + 文档,单 PR。建议落地子顺序:

1. **基础设施**:`fundamentals_loader.py` + `tests/test_fundamentals_loader.py`
   (mock baostock)
2. **9 个技术家族**:依次实现 + 测试(由易到难)
   - 直接统计 `original_stats.py`(最简单,纯 rolling)
   - EWMA `ewma.py`
   - VWAP 偏离 `vwap_deviation.py`
   - 收盘位置 `close_position.py`
   - 短窗换手 `turnover_extra.py`(注意 volume=0 守护)
   - 加速度 `acceleration.py`
   - 单股波动 `single_stock_vol.py`(ATR / CCI / Parkinson)
   - 复合 `composite.py`
   - 秩相关 `rank_correlation.py`
   - 截面宽度 `cross_sec_breadth.py`(全市场标量广播)
3. **基本面**:`fundamentals.py` 因子定义 + `tests/test_factors_fundamentals.py`
   (PIT 测试是关键)
4. **注册副作用**:更新 `factors/__init__.py`,跑总 pytest
5. **CLI flag**:`--refresh-fundamentals` 挂到 `cli.py`
6. **文档**:CLAUDE.md + README.md 一并更新
7. **smoke 验证**:跑 `factors list` 确认数量、跑 `factors analyze --universe pool`
   小规模 sanity

---

## 14. 验收标准

PR 合并前必须达成:

- [ ] 528 + ~60-80 新测试全过(`.venv/Scripts/python.exe -m pytest tests/ -q`)
- [ ] `factors list` 输出总数 274-322(新增 ~160-208,基线 114)
- [ ] `factors list --source custom` 输出含基本面 + 短窗换手新条目
- [ ] `factors list --type fundamental` 输出 ~15-20 个基本面因子
- [ ] PIT 测试覆盖 `pubDate` 边界(`test_factors_fundamentals.py`)
- [ ] `factors analyze --universe pool --output reports/factor_analysis` 跑通,
  HTML 报告含新因子
- [ ] 老 yaml(`config.yaml`、`ab.yaml` 等)不改即可继续跑(向后兼容)
- [ ] CLAUDE.md 模块地图 + 因子库章节 + 测试表已更新
- [ ] README.md 命令示例 + 因子库简介已更新

---

## 15. Follow-up(spec 不做)

- §3.4 改进路线表中的预处理流水线(winsorize / cs z-score / 默认行业中性)
- AdjMSE 损失(§3.6)
- MVO + Ledoit-Wolf 加权(§3.8)
- GBM 数据增强(§3.7)
- 微观结构因子(若日后接入分钟数据)
- 基本面 akshare/Tushare 备份数据源
- 基本面 Bayesian-shrunk 估值因子(论文 A 提到)

这些是独立 PR,与本 spec 解耦。
