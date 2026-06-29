# Design draft — Regime-conditional weighter (IC ↔ Sharpe)

> 状态:**研究方向草案(未实装)**。来源:改进循环 follow-up **F4**
> (`docs/improvement_loop/BACKLOG.md`)。这是一个**新研究方向**,不是把 default
> weighter 从 `ic` 换成 `sharpe` —— 后者已在 Layer B 15-yr × top1000 配对检验中
> **NOT CONFIRMED**(见 `docs/improvement_loop/WORKLOG.md` 的 "D3-Layer-B")。

## 1. 动机(为什么不是简单替换)

把 `weighter.type` 从 `ic` 切到 `sharpe` 的整体 ΔIC = −0.00057(15-yr,T=3533 日 IC
观测),95% block-bootstrap CI 含 0 —— **整体无普适增益**。但 regime 子段分析揭示
sharpe weighter 是 **regime-conditional alpha**:

| 区间 | n | ΔIC (sharpe−ic) | t | 方向 |
|---|---:|---:|---:|---|
| 2015 杠杆崩盘 + 熔断 (2015-06~2016-02) | 156 | −0.00326 | **−2.79** | IC 显著胜 |
| 平台监管 / 退市 (2022-01~2024-04) | 549 | +0.00162 | **+2.73** | **sharpe 显著胜** |

- **低波动 / 趋势期**(2022–2024 弱复苏):sharpe 显著占优(t = +2.73)。
- **极端波动期**(2015 去杠杆 + 熔断):sharpe 显著吃亏(t = −2.79)。
- 两者整体抵消 → 静态择一必然次优。

**假设**:若能**在线检测 regime** 并在该 regime 内选用更优的 weighter,可同时吃到两边的
条件 alpha,严格优于任一静态选择。

## 2. 设计要点

### 2.1 Regime detector(避免事件日期硬编码)

WORKLOG 的子段是用**已知历史事件日期**事后切的 —— 这是 look-ahead,不能直接进生产。
detector 必须**只用截至 t 的信息**,候选信号(都 look-ahead 安全):

- **市场波动率分位**:全市场日收益横截面中位数的 N 日(如 20/60)滚动 std,取其在
  扩张样本(expanding,只用过去)里的分位。高分位 = 高波动 regime。
- **横截面 dispersion 分位**:每日横截面收益 std(个股分化程度)的滚动均值分位。
- **趋势/反转强度**:大盘指数 N 日动量的符号 + 绝对值分位。

初版建议用**单一波动率分位**(最少自由度,最易证伪):
`regime_t = "high_vol" if vol_pct_t >= θ else "low_vol"`,θ 默认 0.7(过去样本分位)。
所有分位用 **expanding 窗口**(纯历史),绝不引入未来分位边界。

### 2.2 Weighter 选择

```
regime_t = detector(market_data[:t])           # look-ahead-safe
weighter = {"high_vol": IC, "low_vol": Sharpe}[regime_t]
```

映射方向由 §1 经验给出(高波动→IC,低波动→Sharpe),但**必须在 OOS 上验证**,不能直接
当成既定结论(子段是事后切的,符号可能不稳)。映射本身是 2 个超参(θ + 高/低 regime
各用哪个 weighter),需进 AB 网格。

### 2.3 落地形式(最小侵入)

- 不动现有 7 个 weighter 类。新增 `RegimeConditionalWeighter`(`ml/weighters.py`),
  内部持有两个子 weighter + 一个 detector callable,`fit/predict` 时按当前 regime 委派。
- detector 需要**市场级时间序列**(指数或全市场截面),当前 weighter 接口只拿到
  `(X, y)`。要么:(a) 在 `TwoStepPipeline` 装配时注入预算好的 `regime_series`
  (T→regime label),predict 时按 `y.index` 的 date 查表;(b) detector 作为独立 panel
  因子预算,和 factor_panel 同源缓存。倾向 (a),避免污染因子库。
- config:`weighter.type: "regime_conditional"` +
  `weighter.regime_conditional.{detector, vol_window, threshold, high_vol_weighter,
  low_vol_weighter}` 子段(沿用现有子段化 pattern)。

## 3. 评估计划

1. **Layer B 先行**(便宜):用 `docs/improvement_loop/analysis/layer_b_direct.py`
   扩一个 regime-conditional 分支,在 15-yr × top1000 上比 daily IC:
   `regime_conditional` vs `ic`(default)vs `sharpe`。要求点 ΔIC > 0 **且** CI 排除 0
   **且** 子段符号一致(硬化判定标准,见 `docs/improvement_loop/METHODOLOGY.md`)。
2. **Layer D**(组合 Sharpe):仅当 Layer B 通过再做 —— 需先修 **F1** 的 15-yr 死锁
   (`docs/handoff/2026-06-28-portfolio-ab-15yr-deadlock.md`),否则 portfolio Sharpe AB
   在 15-yr 上跑不完。
3. **过拟合护栏**:θ + 映射方向共 ≤3 个自由度;必须在**独立 regime 划分**(uniform
   5/8 buckets,非事件日期)上重测符号一致性,防止 detector 拟合到 2015/2022 两段。

## 4. 不做 / 风险

- **不**把它设为 default —— 它引入额外自由度,过拟合风险高于任一静态 weighter。
  只有在两层都 CONFIRMED 且子段稳健时才考虑 opt-in,且 default 仍是 `ic`。
- detector 的分位阈值是新的可调旋钮 → 容易事后调参拟合历史。坚持 expanding 分位 +
  独立子段验证。
- 若 Layer B 仍 NOT CONFIRMED(很可能),结论就是"regime 条件 alpha 太弱/太不稳,
  不值得 ensemble 复杂度",归档此草案即可,不进生产。
