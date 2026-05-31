# baostock 季度财务表字段对照(2026-05-31 实测)

> Task 0 调研笔记。后续 `fundamentals_loader` / fundamentals factor 公式以此为准 — **不要**用 spec §5.1 的近似字段名。
>
> 实测样本:
> - `sz.000001` 平安银行(金融股,balance/cash_flow 多个比率为空 — 银行无流动比率等概念)
> - `sh.600519` 贵州茅台(非金融,ratios 都有值,做"满字段"基线)
> - 年份季度:2024Q3(pubDate 2024-10-19 / 2024-10-26;statDate 2024-09-30)
>
> baostock 版本:`.venv/Scripts/python.exe -m pip show baostock` 当前 install,login 返回 `error_code=0`,所有 5 张表 query 也 `error_code=0`。

---

## 1. profit (`query_profit_data`)

**fields (11):** `code, pubDate, statDate, roeAvg, npMargin, gpMargin, netProfit, epsTTM, MBRevenue, totalShare, liqaShare`

| 字段 | 茅台 2024Q3 | 平安银行 2024Q3 | 备注 |
|---|---|---|---|
| `roeAvg` | 0.268330 | 0.082528 | 平均 ROE(已年化的累计期间均值) |
| `npMargin` | 0.521887 | 0.356052 | 净利率 = netProfit / 营收 |
| `gpMargin` | 0.915314 | (空) | 毛利率;**银行无毛利概念,空值**|
| `netProfit` | 6.30e10 | 3.97e10 | 累计净利润(元) |
| `epsTTM` | 65.82 | 2.40 | 滚动 12 月 EPS |
| `MBRevenue` | (空) | (空) | 主营业务收入 — **两只都空**,baostock 该字段不可靠 |
| `totalShare` | 1,256,197,800 | 1.94e10 | 总股本(股) |
| `liqaShare` | 1,256,197,800 | 1.94e10 | 流通 A 股 |

⚠ `MBRevenue` 实测都空 — **不可作为营收来源**。推导营收用 `netProfit / npMargin`。

---

## 2. growth (`query_growth_data`)

**fields (8):** `code, pubDate, statDate, YOYEquity, YOYAsset, YOYNI, YOYEPSBasic, YOYPNI`

| 字段 | 茅台 | 平安银行 | 含义 |
|---|---|---|---|
| `YOYEquity` | 0.091368 | 0.052825 | 净资产同比 |
| `YOYAsset` | 0.094226 | 0.041621 | 总资产同比 |
| `YOYNI` | 0.149639 | 0.002372 | 净利润同比 |
| `YOYEPSBasic` | 0.150392 | 0.000000 | 基本每股收益同比 |
| `YOYPNI` | 0.150376 | 0.002372 | 归母净利润同比 |

⚠ **没有 `YOYIncome` / `YOYRevenue`** — spec 假设的"营收同比"字段不存在。
- 替代方案 A:用 `YOYPNI`(归母净利润同比)作 `revenue_yoy` 的代理 — 大多数情况下相关性高,但毛利率波动大的公司会失真
- 替代方案 B:从 profit 表的 `netProfit / npMargin` 推 `revenue_quarter`,然后自己算同比(需要两期数据)。fundamentals_loader 实现时优先选 B,A 作 fallback
- 替代方案 C:暂不提供 `revenue_yoy` 因子,改用 `YOYPNI`(`profit_yoy`)和 `YOYAsset`(`asset_yoy`)— **推荐**,简单可靠

---

## 3. balance (`query_balance_data`)

**fields (9):** `code, pubDate, statDate, currentRatio, quickRatio, cashRatio, YOYLiability, liabilityToAsset, assetToEquity`

| 字段 | 茅台 | 平安银行 | 含义 |
|---|---|---|---|
| `currentRatio` | 6.156912 | (空) | 流动比率 — 银行无此概念 |
| `quickRatio` | 4.911560 | (空) | 速动比率 — 同上 |
| `cashRatio` | 1.556805 | (空) | 现金比率 — 同上 |
| `YOYLiability` | 0.054643 | 0.040588 | 负债同比 |
| `liabilityToAsset` | 0.001363 | 0.009146 | 资产负债率(茅台值偏低 — 这是"有息负债/总资产",不是 1-equity/asset) |
| `assetToEquity` | 1.157789 | 11.715293 | 权益乘数 = 总资产 / 净资产 |

⚠ **没有 `totalShareholdersEquity`(净资产绝对额)、`totalAsset`(总资产)直接字段**。
- 推导净资产:`equity = netProfit / roeAvg`(精确,因为 `roeAvg = netProfit / 平均净资产`,**注意**茅台 roeAvg 是 9 个月累计的"半"年化值,不是终末点)
- 或:用 `dupontROE / dupontAssetStoEquity = NP/equity / asset/equity = NP/asset`,反推 `asset = NP / (dupontROE/dupontAssetStoEquity)`,再 `equity = asset / assetToEquity`
- **PB 因子推荐公式**:`PB = close * totalShare / equity`,`equity` 优选 `netProfit / roeAvg`(单查询可得)

⚠ `liabilityToAsset` 茅台 0.0014 反常低,可能是有息负债口径(非"总负债/总资产")。如果需要传统资产负债率,可用 `1 - 1/assetToEquity = 1 - 1/1.158 = 0.136` 约 13.6%(更合理)。**fundamentals factor 建议优先用 `assetToEquity` 派生,而非 `liabilityToAsset` 直接用**。

---

## 4. cash_flow (`query_cash_flow_data`)

**fields (10):** `code, pubDate, statDate, CAToAsset, NCAToAsset, tangibleAssetToAsset, ebitToInterest, CFOToOR, CFOToNP, CFOToGr`

| 字段 | 茅台 | 平安银行 | 含义 |
|---|---|---|---|
| `CAToAsset` | 0.831394 | (空) | 流动资产/总资产 — 银行空 |
| `NCAToAsset` | 0.168606 | (空) | 非流动资产/总资产 — 银行空 |
| `tangibleAssetToAsset` | 0.776988 | (空) | 有形资产/总资产 — 银行空 |
| `ebitToInterest` | (空) | (空) | 利息覆盖率 — **两只都空**,不可靠 |
| `CFOToOR` | 0.367799 | 1.229213 | 经营现金流/营收(OR=operating revenue) |
| `CFOToNP` | 0.704749 | 3.452340 | 经营现金流/净利润 |
| `CFOToGr` | 0.360790 | 1.229213 | 经营现金流/营业总收入(GR=gross revenue) |

⚠ **没有 CFO 绝对额字段**。推导:`CFO = CFOToNP * netProfit`(profit 表里有 netProfit)。

PCF 因子公式可行:
```
revenue ≈ netProfit / npMargin
CFO     ≈ CFOToOR * revenue = (CFOToOR / npMargin) * netProfit
PCF     = close * totalShare / CFO
```

---

## 5. dupont (`query_dupont_data`)

**fields (11):** `code, pubDate, statDate, dupontROE, dupontAssetStoEquity, dupontAssetTurn, dupontPnitoni, dupontNitogr, dupontTaxBurden, dupontIntburden, dupontEbittogr`

| 字段 | 茅台 | 平安银行 | 含义 |
|---|---|---|---|
| `dupontROE` | 0.268330 | 0.082528 | 杜邦 ROE(与 profit.roeAvg 一致) |
| `dupontAssetStoEquity` | 1.234004 | 11.771021 | 权益乘数(资产/净资产)— 与 balance.assetToEquity 略有差异(口径:期末 vs 平均) |
| `dupontAssetTurn` | 0.440140 | 0.019691 | 总资产周转率 |
| `dupontPnitoni` | 0.965035 | 1.000000 | 归母 NI / NI |
| `dupontNitogr` | 0.511941 | 0.356052 | NI / 营业总收入 = 净利率(与 npMargin 一致) |
| `dupontTaxBurden` | 0.749938 | 0.832126 | 税负 = NI / EBT |
| `dupontIntburden` | 1.014035 | (空) | 利息负担 = EBT / EBIT — 银行空 |
| `dupontEbittogr` | 0.673196 | (空) | EBIT / GR — 银行空 |

---

## 6. spec §5.1 字段名 → 实测字段 映射表

下面是 spec 里 fundamentals factor 的目标字段名,以及在实测 schema 下的可行推导。

| spec 目标因子 | spec 假设字段(可能错) | 实测可用字段 | 推导公式 |
|---|---|---|---|
| `pe` (TTM) | `netProfit` (累计→TTM) | `profit.netProfit`, `profit.totalShare`, `profit.epsTTM` | **PE_TTM = close / epsTTM**(最简,直接用 epsTTM) |
| `pb` | `totalShareholdersEquity` | 无直接字段 | **PB = close * totalShare / equity**, `equity = netProfit / roeAvg`(roeAvg 已年化均值口径) |
| `pcf` | `CFOToOR * totalRevenue` | `CFOToOR`, `npMargin`, `netProfit` | **PCF = close * totalShare / (CFOToOR * netProfit / npMargin)** |
| `ps` | `revenue` | `npMargin`, `netProfit` | **PS = close * totalShare / (netProfit / npMargin)** |
| `roe` | `roeAvg` | `profit.roeAvg` ✓ | 直接 |
| `roa` | `roaAvg` | **无!** | `roa = roeAvg / assetToEquity`(等价杜邦 NI/asset = NI/equity × equity/asset) |
| `gpm` (毛利率) | `gpMargin` | `profit.gpMargin` ✓ | 直接;**银行/保险股可能为空** |
| `npm` (净利率) | `npMargin` | `profit.npMargin` ✓ | 直接 |
| `revenue_yoy` | `YOYIncome` | **无!** | 推荐用 **`YOYPNI`**(归母净利同比)作代理,或用 `netProfit/npMargin` 自算同比(需 ≥2 期) |
| `profit_yoy` | `YOYNI` | `growth.YOYNI` ✓ 或 `YOYPNI` | 直接;建议用 `YOYPNI`(归母,更稳定) |
| `asset_yoy` | `YOYAsset` | `growth.YOYAsset` ✓ | 直接 |
| `equity_yoy` | `YOYEquity` | `growth.YOYEquity` ✓ | 直接 |
| `liability_yoy` | `YOYLiability` | `balance.YOYLiability` ✓ | 直接 |
| `eps_yoy` | `YOYEPSBasic` | `growth.YOYEPSBasic` ✓ | 直接 |
| `current_ratio` | `currentRatio` | `balance.currentRatio` ✓ | 直接;银行/保险空 |
| `quick_ratio` | `quickRatio` | `balance.quickRatio` ✓ | 直接;银行/保险空 |
| `cash_ratio` | `cashRatio` | `balance.cashRatio` ✓ | 直接;银行/保险空 |
| `debt_to_asset` | `liabilityToAsset` | `balance.liabilityToAsset` ⚠ 或 `1 - 1/assetToEquity` | **liabilityToAsset 是"有息负债/资产"口径,茅台 0.0014 反常低**;若需传统资产负债率用 `1 - 1/assetToEquity` |
| `equity_multiplier` | `assetToEquity` | `balance.assetToEquity` ✓ | 直接;杜邦表里 `dupontAssetStoEquity` 也有(口径略有差异) |
| `cfo_to_revenue` | `CFOToOR` | `cash_flow.CFOToOR` ✓ | 直接 |
| `cfo_to_netprofit` | `CFOToNP` | `cash_flow.CFOToNP` ✓ | 直接 |
| `asset_turnover` | `dupontAssetTurn` | `dupont.dupontAssetTurn` ✓ | 直接 |
| `tax_burden` | `dupontTaxBurden` | `dupont.dupontTaxBurden` ✓ | 直接 |
| `interest_burden` | `dupontIntburden` | `dupont.dupontIntburden` ⚠ 银行空 | 直接,需处理空值 |

---

## 7. 关键差异 / 后续 Task 注意事项

1. **没有 `roaAvg`,没有 `YOYIncome`,没有 `totalShareholdersEquity`,没有 `totalAsset`,没有 `revenue` 直接字段。** spec §5.1 至少 5 个假设字段不存在,fundamentals_loader 必须做派生计算。
2. **`MBRevenue` 字段存在但实测都空**,不要依赖。
3. **`ebitToInterest` 字段存在但实测都空**,不要依赖。
4. **金融股(银行/保险)在 balance / cash_flow 的多个比率字段实测都是空字符串 `""`**。fundamentals_loader 读取时:
   - 空字符串需当 NaN 处理(`pd.to_numeric(errors="coerce")`)
   - factor compute 应允许金融股某些因子为 NaN(已经天然按行业稀疏)
5. **`liabilityToAsset` 口径不是传统资产负债率**,茅台 0.14% 显然不对(实际约 13.6%)。优先用 `1 - 1/assetToEquity` 派生,或用 `assetToEquity` 直接作"权益乘数"因子。
6. **profit 表只有 `netProfit` 累计值,没有"当季 netProfit"**。TTM 计算需要拉 4 个连续季度数据,或者直接用 `epsTTM`(profit 表里就有)避开累计/TTM 转换。
7. **`pubDate` vs `statDate` 都要保留**。Point-in-time 安全:facto 在日期 D 用得了某季报,当且仅当 `pubDate ≤ D`(statDate 是会计期末,pubDate 是发布日;有 30-60 天延迟)。fundamentals_loader 必须按 `pubDate` 做时间对齐,**不能**按 `statDate`。
8. **登录调用**:`bs.login()` 返回 `error_code=0` 即成功,无需 token。生产代码可考虑用 context manager 包一层 `try / finally bs.logout()`。

---

## 8. Schema 一行速查

```
profit:    code,pubDate,statDate,roeAvg,npMargin,gpMargin,netProfit,epsTTM,MBRevenue,totalShare,liqaShare
growth:    code,pubDate,statDate,YOYEquity,YOYAsset,YOYNI,YOYEPSBasic,YOYPNI
balance:   code,pubDate,statDate,currentRatio,quickRatio,cashRatio,YOYLiability,liabilityToAsset,assetToEquity
cash_flow: code,pubDate,statDate,CAToAsset,NCAToAsset,tangibleAssetToAsset,ebitToInterest,CFOToOR,CFOToNP,CFOToGr
dupont:    code,pubDate,statDate,dupontROE,dupontAssetStoEquity,dupontAssetTurn,dupontPnitoni,dupontNitogr,dupontTaxBurden,dupontIntburden,dupontEbittogr
```
