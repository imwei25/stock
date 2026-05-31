# Tradability Mask for Factor Input — 设计文档

> 日期:2026-05-31
> 范围:A1 — 仅做 factor 输入侧 mask_price + 训练标签双向检查
> 关联文献:[arXiv 2507.07107 — Mask-First Multi-Factor Quant Trading](https://arxiv.org/html/2507.07107)
> 关联调研:`docs/research/2026-05-31-a-share-quant-survey-comparison.md` §3.2

---

## 1. 动机

A 股每日收盘价存在被 ±10% / ±20% 涨跌停规则截断的"censored price"问题。
当一只股票当日触及涨停 (close = +9.9%),这个 close 并不反映真实供需 —
真实清算价格可能想推到 +15%,但被规则封顶在 +9.9%。如果把这种截断价喂给
滚动因子(MA / ts_corr / cross-section rank),会产生两类后果:

1. **滚动统计被污染**:5 日均线在 t..t+4 这 5 天的窗口里平均了一个被低估
   的 close,后续 5 天的因子值都受影响,直到该窗口滚出。
2. **截面排名虚高**:涨停股当日 ret 排名第 1,但你**无法以这个价格交易**;
   open[t+1] 已经把信息通过 gap-up 提前定价,实际可捕获收益远小于 factor
   显示的动量。

论文 B 在真实 A 股(2022–2024)上消融实验:加 mask 后 Sharpe 从 1.61 →
**2.05** (+0.44)。这是论文最大单项贡献,超过模型架构(MLP→Transformer
+0.13)、损失函数(MSE→AdjMSE +0.27)等其他改动。

我们的目标:在本项目复现这个 mask-first 改造,**实测预期 +0.2 Sharpe
(小样本上不如论文,但量级一致)**。

---

## 2. 范围

### 2.1 In Scope (A1)

- **mask_price**:用于因子输入侧清洁,即"close[t] 作为 factor 输入是否可信"
  - 触发条件:涨/跌停日、停牌日、上市未满 1 年
  - 应用范围:`factors/` 下所有 panel-based 因子(WQ101 + technical + custom)
  - 应用方式:**预 mask panel**(panel 字段 `close/open/high/low/volume` 在
    mask=False 位置置 NaN),通过 NaN 自然传播到所有算子
- **标签双向检查**:`ml/dataset.py` 生成 `(X_t, y_{t+h})` 训练样本时,
  要求 `mask[t]=True ∧ mask[t+h]=True`(防 close[t]/close[t+h] 涨停导致
  label 虚高)
- **配置层**:`MLFactorConfig.mask` 子段,default `enabled=false`,完全
  向后兼容
- **缓存隔离**:mask 配置进入 `MLFactorConfig.content_hash`,翻 flag 自动
  让旧 `factor_panels/<sig>/` 失效

### 2.2 Out of Scope (留给后续 PR)

- **mask_exec**(open-side 执行可填性):用户的关切"开盘是否真的封板",
  对应 backtest engine 的 fill guard,单独 PR 处理。本 spec 不涉及
- **新因子家族**(VWAP 偏离 / 换手 z-score / 截面市场宽度):A2 / B 阶段
- **因子预处理**(winsorize / cs z-score):C 阶段
- **`composite_verdict` 策略路径**:其 `indicators.py` 走 per-stock 而非
  panel,污染问题轻微,不做改造
- **真实涨跌停价精确判定**:仍用启发式 `|ret| > threshold`,不引入新数据源

---

## 3. Mask 定义

### 3.1 阈值函数

```python
def _limit_threshold(code: str) -> float:
    """A 股涨跌停幅度按板块判定。

    返回值是"abs 收盘当日 ret 超过它即视为涨/跌停"的阈值。
    用 0.098 / 0.198 / 0.298 而非 0.10 / 0.20 / 0.30 是为了让真实涨停
    (实际是 ±9.99% 因 round-to-cent) 也能被捕获。
    """
    if code.startswith(("300", "301", "688")):
        return 0.198   # 创业板 + 科创板 ±20%
    if code.startswith(("83", "87", "43", "82")):
        return 0.298   # 北交所 ±30%(项目 universe 不会出现,留兜底)
    return 0.098       # 主板沪深 ±10%
```

ST ±5% 不处理:`universe.parquet` 和 `cfg.stocks` 都不含 ST。

### 3.2 Mask 三条件

T × N boolean DataFrame `mask`,以下三条**全部** True 才视为可用:

```python
ret = close / close.shift(1) - 1
thresholds = pd.Series(
    {code: _limit_threshold(code) for code in close.columns}
)

cond_not_limit = ret.abs().lt(thresholds, axis=1)   # 条件 1:不在涨跌停
cond_has_volume = volume > 0                         # 条件 2:有成交(非停牌)
cond_listed_long = _listing_mask(close, 252)         # 条件 3:上市 >1 年

mask = cond_not_limit & cond_has_volume & cond_listed_long
# 注:ret.shift(1) 在第 0 行 NaN → cond_not_limit 第 0 行 False(pandas NaN.lt = False)
```

`_listing_mask` 关键设计:**只对 panel 内新上市的股**应用 252 规则,
panel 开始就有 close 的股视为成熟股(因为它们的真实 IPO 早于 panel 起点,
项目无 IPO 元数据 → 保守假设已经满足上市 >1 年):

```python
def _listing_mask(close: pd.DataFrame, min_days: int = 252) -> pd.DataFrame:
    """Mask=False 仅适用于 panel 内"新上市后头 min_days 天"。"""
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    for code in close.columns:
        first_valid = close[code].first_valid_index()
        if first_valid is None:
            mask[code] = False
            continue
        first_pos = close.index.get_loc(first_valid)
        if first_pos == 0:
            continue  # Panel 起点就有数据 → 视为成熟股,全 True
        end_pos = min(first_pos + min_days, len(close))
        mask.iloc[first_pos:end_pos, mask.columns.get_loc(code)] = False
    return mask
```

对于 cfg.stocks(16 只全是老股)和 training_universe=all(4000+ 大部分老股),
此 mask 行为基本是"老股不动,新股头 252 天屏蔽"。**避免了把 500 天 panel
的前一半全部屏蔽的灾难性 bug**。

### 3.3 Apply Mask

```python
def apply_mask(panel: Mapping[str, pd.DataFrame],
               mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """返回 mask=False 位置置 NaN 的新 panel,原 panel 不动。"""
    return {field: df.where(mask) for field, df in panel.items()}
```

应用后 `panel["close"][涨停日, 该股] = NaN`,等等。所有下游算子通过 NaN
处理自然过滤。

---

## 4. 接入点

### 4.1 `strategy_factory.build_factor_panel`

```python
def build_factor_panel(
    panel: Mapping[str, pd.DataFrame],
    factors: Sequence[str],
    *,
    mask_config: MaskConfig | None = None,   # ← 新参数
) -> dict[str, pd.DataFrame]:
    if mask_config is not None and mask_config.enabled:
        from stockpool.panel import compute_tradability_mask, apply_mask
        mask = compute_tradability_mask(panel, mask_config)
        panel = apply_mask(panel, mask)
    # 后续因子计算逻辑不变,所有 ops.py 算子吃 NaN 传播
    ...
```

### 4.2 `cli._prepare_ml_pool`

预算 factor_panel 时传 `mask_config = cfg.strategy.ml_factor.mask`。

### 4.3 `strategy_factory.load_or_build_factor_panel`

缓存签名 `sig` 加入 `mask_config`(已通过 `content_hash` 间接生效)。
`manifest.json` 新增 `mask_enabled` / `mask_threshold_main` /
`mask_threshold_chinext` 字段供人肉调试。

### 4.4 `ml/dataset.py` 标签双向检查

`stack_panel_to_xy` (pooled) 和 `build_factor_matrix` (per-stock) 在生成
`(X_t, y_t)` 时:

```python
# 已有的 label 计算
y_t = close.shift(-horizon) / close - 1   # forward return

# 新增 mask 检查(只在 mask 模块开启时)
if mask is not None:
    label_valid = mask & mask.shift(-horizon)    # 双向 mask
    y_t = y_t.where(label_valid)
    # 后续 stack 时 NaN 行自动 dropna
```

**Mask 传递机制**:`stack_panel_to_xy` 和 `build_factor_matrix` 新增可选参
`mask: pd.DataFrame | None = None`(default None → 旧行为)。**不**塞进
panel_data 字典,理由:
- `panel_data` 当前签名是 OHLCV 字段,加 `"mask"` key 破坏类型语义
- 显式参数便于在 cli 层调用栈追踪 mask 是否被传下去
- default None 时所有现有调用站点零修改

### 4.5 `panel.py` 的 mask 是否成为 6th panel field?

**不要**。理由:
- `OHLCV_FIELDS` 是数据字段语义,mask 是元数据
- panel 字段统一参与 `apply_mask` 会导致 `mask.where(mask)` 自指环
- 改 `OHLCV_FIELDS` 会污染 `assert_panel_valid` / `panel_shape` 等多处
- 现有 374 个测试基于 5 字段 panel,扩到 6 字段是兼容性风险

mask 单独以 `pd.DataFrame` 形式从 `compute_tradability_mask` 返回,在
factor_panel 计算前一次性吸收,**不进 panel 字典**。

---

## 5. Ops.py 的 NaN 容忍度修补

经审计,以下算子在 NaN 输入下有问题,需要小修:

| 算子 | 当前行为 | 修补 |
|------|---------|------|
| `ts_mean` / `ts_std` | `min_periods=window` 太严,有 1 个 NaN 就整个窗口 NaN | 改 `min_periods=max(1, int(window * 0.6))` |
| `ts_corr` / `ts_covariance` | pandas rolling.corr 自带 NaN 跳过 | 无需改 |
| `ts_rank` | pandas rolling.rank NaN 行为已合理(NaN 跳过) | 无需改 |
| `rank`(cs,axis=1,pct=True) | NaN 自动得 NaN rank,排名分母用 valid count | 无需改 |
| `indneutralize` | 按 sector groupby + demean,NaN 自动剔除 | 无需改 |
| `decay_linear` | `np.dot(x, weights)` 遇 NaN 整列变 NaN | 改用加权 nansum + 重归一化:`(x * w).sum(skipna=True) / w[~x.isna()].sum()` |
| `ts_sum` / `ts_product` | `min_periods` 同 ts_mean | 同上放宽 |
| `delta` / `delay` | shift,NaN 自然 | 无需改 |
| `signedpower` | element-wise,NaN 自然 | 无需改 |
| `scale`(L1 norm) | 求和归一化,NaN 跳过即可 | 用 `np.nansum` 替代 `sum` |

**回归测试约束**:每个修补后的算子,**所有现有 ops 测试必须仍然通过**
(test_ops.py / test_factors.py)。因为旧测试没有大量 NaN 输入,放宽
`min_periods` 不改变其结果。

---

## 6. 配置 Schema

```python
# config.py
class MaskConfig(BaseModel):
    """Tradability mask for factor input quality (paper B mask-first)."""
    enabled: bool = False                          # 默认关闭,向后兼容
    limit_up_threshold_main: float = 0.098         # 主板沪深 ±10%
    limit_up_threshold_chinext: float = 0.198      # 创业板 + 科创板 ±20%
    limit_up_threshold_bse: float = 0.298          # 北交所 ±30%(留兜底)
    min_listing_days: int = 252                    # 上市 >1 年

    # 标签双向检查 — 跟随 enabled,无独立开关
    # (实测发现 enabled=true 但 label 不 mask 会减半效果,
    #  论文 B 也是 bundled)

    model_config = {"extra": "forbid"}

class MLFactorConfig(BaseModel):
    # ... 现有字段
    mask: MaskConfig = Field(default_factory=MaskConfig)
```

YAML 用法:

```yaml
strategy:
  name: ml_factor
  ml_factor:
    # 现有字段...
    mask:
      enabled: true              # 翻开关
      # 阈值留默认即可
```

---

## 7. 缓存失效

- `MLFactorConfig.content_hash` 已对 `mask` 段做哈希(Pydantic 子模型自动
  纳入),翻 `enabled=true` → 新 `sig` → 重算 factor_panel + 重训 ml_models
- `factor_panels/<sig>/manifest.json` 新增 `mask_enabled: bool` 等字段,
  供 cache 调试时人肉对比
- **不需要写迁移逻辑**:sig hash 变化自然让旧 cache 失效,只是占磁盘
  (用户可手动 `rm -rf data/factor_panels/`,或留着)

---

## 8. 测试

### 8.1 新建测试文件

**`tests/test_panel_mask.py`**(新):
- `test_compute_mask_limit_up_main`:主板涨停日 mask=False
- `test_compute_mask_limit_up_chinext`:创业板 +19.9% mask=False、
  +9.9% mask=True(对比主板会被误判)
- `test_compute_mask_suspended`:volume=0 mask=False
- `test_compute_mask_new_listing`:上市第 100 天 mask=False、第 252 天
  mask=True
- `test_apply_mask_correctness`:NaN 位置精确对应 mask=False
- `test_apply_mask_does_not_mutate`:原 panel 不变(深拷贝语义)
- `test_threshold_function`:`_limit_threshold` 对各代码前缀返回正确

**`tests/test_ops_mask_nan_safe.py`**(新):
- `test_ts_mean_relaxed_min_periods`:窗口内 1 个 NaN,结果 = 其余 N-1 个
  的均值(不是 NaN)
- `test_ts_corr_nan_pair_skip`:一段 NaN 后正常段,corr 用正常段计算
- `test_decay_linear_nan_renormalize`:权重和分母同步剔除 NaN
  位置后归一化
- `test_rank_pct_nan_excluded_from_denominator`:cross-section rank 分母
  = valid count
- `test_indneutralize_nan_skip`:NaN 股不参与 sector demean
- 对每个修补算子,**额外加一个"全 valid 输入下结果不变"测试**保证旧
  baseline 不破

**`tests/test_ml_strategy_mask.py`**(新):
- `test_mask_disabled_baseline_unchanged`:`enabled=false` 时 (X, y) 与
  当前实现 bitwise 等价
- `test_mask_enabled_drops_labels`:fixture 里植入 3 个涨停日,开启 mask
  后训练样本数减少 6 行(3 个 t 行 + 3 个 t-horizon 行,双向)
- `test_mask_content_hash_invalidates`:翻 `enabled` flag → sig 变化 →
  旧 pickle 失效(`_try_load_cached` 返回 None)

### 8.2 现有测试约束

`pytest tests/ -q` 全部 374 个测试必须仍然通过(default `enabled=false`
退化到旧路径)。如果有挂掉的,先调查是否真的破坏 baseline,**不允许**
为新测试通过而改旧测试预期值。

---

## 9. A/B 验证

落地后跑 `ab.yaml`:

```yaml
base_config: config.yaml
arms:
  baseline:
    strategy:
      ml_factor:
        mask:
          enabled: false
  with_mask:
    strategy:
      ml_factor:
        mask:
          enabled: true
```

预期产物 `reports/ab/<date>.html`:
- `with_mask` 整体 Sharpe ≥ baseline + 0.10(小样本下打折,论文是 0.44)
- IC 可能略降(apparent IC 下降是正常的 — 真实可交易 IC 才是上升的)
- 训练样本数下降 1-3%(典型市场年涨停率)

如果**没有**复现这个方向,有三种可能:
1. cfg.stocks 太小(16 只),mask 触发频次不够 → 扩到 training_universe=all
2. 阈值过严(罕见,但理论上)→ 检查 `compute_tradability_mask` 的统计
3. 论文不适用于本项目的样本/时段 → 不上线,留 spec 做警示文档

---

## 10. 文档更新

落地后,**同时**更新:
- `CLAUDE.md`:
  - "模块地图"`panel.py` 行添加 `compute_tradability_mask` /
    `apply_mask`
  - "配置 (`config.yaml`)" `strategy.ml_factor.mask.*` 子段
  - "测试" 表加 3 行
  - "已知不支持的能力" 添加注:"mask_exec(open-side fillability)单独 PR"
- `README.md`:常用命令段加 `mask.enabled` 配置示例(opt-in 用法)

---

## 11. 验收标准

- [ ] `pytest tests/ -q` 374 + 新增 ~21 = ~395 个测试全过(panel_mask 7 +
  ops_mask 11 + ml_strategy_mask 3)
- [ ] `python -m stockpool backtest --config config.yaml` 在 mask 开启时
  能跑通,产出 HTML 报告
- [ ] `python -m stockpool ab --config ab.yaml`(arms: baseline +
  with_mask)能产出 HTML 对比报告
- [ ] `with_mask` arm 在 16 股小样本上 Sharpe ≥ baseline + 0.05(松一点
  的 acceptance,严格 +0.2 留作 stretch goal)
- [ ] `enabled=false` 时 `cfg.content_hash` 与本 spec 落地**前**完全
  相同(回归检查 — 任何 hash 变化说明 default 路径被改了)
- [ ] CLAUDE.md + README.md 同步更新完毕

---

## 12. 已知风险与权衡

| 风险 | 评估 | 应对 |
|------|------|------|
| **小样本验证不出 Sharpe 提升** | 中:cfg.stocks 16 只,3 年涨停日总数 ~50-150 个,信噪比一般 | A/B 上 training_universe=all + portfolio-backtest 路径,扩到 4000 股 |
| **`ts_mean` 等放宽 `min_periods` 后旧测试值小数点变** | 低:旧测试不传 NaN,放宽 min_periods 不改变全 valid 输入结果 | CI 验证 |
| **`decay_linear` 重归一化方案下数值与论文 B 略偏** | 中:论文用 GPU op,我们用 pandas,边界数值差 1e-6 可接受 | test 用 `np.testing.assert_allclose(rtol=1e-4)` |
| **创业板 ±20% 也会有"非涨停大涨"(如 +15%) 是真信号被吃掉** | 低:阈值 0.198 已经留了 5bp 容差;真实涨停 ret ≈ 0.199–0.200,假阳性低 | 不处理 |
| **content_hash 变化导致用户已有的 ml_models pickle 全失效** | 低:本来就 opt-in,翻 flag 等价于重训请求 | 文档明示 |
| **panel `apply_mask` 拷贝整个 panel 内存翻倍** | 低:WQ101 已经在用更大的 factor_panel(213 columns),mask copy 是 O(T·N) 5 个字段,可接受 | 实测验证 |

---

## 13. 时间估算

- mask + apply_mask + 阈值函数 + 单元测试(panel 侧):**0.5 天**
- ops.py 修补 + 单元测试:**0.5 天**
- strategy_factory + cli + dataset 接入 + 集成测试:**0.5 天**
- ab.yaml 跑通 + 调试:**0.5 天**
- 文档更新:**0.25 天**

合计:**~2 天**(粗估,留 buffer)

---

## 14. 后续 PR 路线(本 spec 不做,仅记录)

1. **A2: mask_exec(open-side)** — backtest engine 加 fill guard,跳过
   open[t+1] 涨停的 entry 信号
2. **B: 新因子家族** — VWAP 偏离 / 换手 z-score / 截面市场宽度 ~15 个
3. **C: 因子预处理流水线** — winsorize + cs z-score + 默认行业中性
4. **D: AdjMSE 损失** — LightGBM custom objective(扩股池后才有意义)
5. **E: MVO + Ledoit-Wolf 加权** — PortfolioEngine 加 `weighting=mvo`
   选项

每个独立 spec + 独立 A/B 验证。
