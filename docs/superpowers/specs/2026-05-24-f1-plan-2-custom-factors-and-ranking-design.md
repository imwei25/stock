# F1 plan-2 — Custom factors + WQ101 ranking + A/B 验证

## 1. 背景与范围

属于 `docs/strategy_improvement_2026.md` §6 路线图的 **P2 项**。F1 plan-1(`factors_analysis` 模块 + `factors analyze` / `pick-by-ic` CLI + pyecharts 报告)已经完成,工具链就绪;F1 plan-2 是在那个工具链之上**真正跑一次全市场因子排名**,并决定是否更新 `config.yaml` 的默认 `factors:` 列表。

为了让 ranking 覆盖更全面、并补 WQ101 没覆盖的 A 股专属信号,本 PR 同时新增 3 个 custom 因子。其中 `industry_relative_strength` 需要 sector 映射,顺带把 WQ101 局部使用的 sector_map 提升为 `factors/` 子包共享的 first-class context。

### 1.1 解决的问题

1. **因子库利用率低**:`config.yaml` 默认 `factors:` 只用 8 个手挑技术因子;`wq101.py` 注册的 101 个 alpha 加 builtin technical 共 ~111 个因子,客观选 top-N 的工具链 (`factors analyze` + `pick-by-ic`) 已就绪但**没人真正跑过**。
2. **WQ101 缺 A 股特色信号**:`industry_relative_strength`(同业超额收益)、`limit_up_count`(短期涨停频次)、`turnover_zscore`(异常活跃度)是 A 股结构化的常用信号,WQ101 用 US-equity factor model 设计,不直接覆盖。
3. **sector_map 局部于 wq101.py**:目前 `_Wq101Context.sector_map` 是 wq101 模块的 ClassVar,通过 `factors.wq101.set_sector_map` 注入。新 custom 因子需要同一份 sector 信息,但从 wq101 子模块导入语义混乱。

### 1.2 非目标(明确避免 scope creep)

- 不改 `Factor` ABC(`compute(panel) -> DataFrame` 不变)
- 不改 `Panel` schema(仍是 `{open, high, low, close, volume} -> T×N`)
- 不引入新依赖(`industry_map.py` 已存在,baostock 是现成依赖)
- 不改 `MLFactorConfig` schema(`factors:` / `factors_file:` 二选一已是现有结构)
- 不实装 northbound_flow(需新数据源,留 followup;不在本 PR)
- 不调 IC ranking 算法本身(F1 plan-1 已实现并验证)
- 不动 selector / weighter / engine / sizing(F2/F3 范畴)

### 1.3 决策门(2026-05-24,roadmap §6 跨阶段约束)

> 任何改默认值的 PR 必须随附 A/B 报告,verdict 至少不是 ❌ regression。

本 PR 满足:Step 6 跑 `stockpool ab` 出 verdict,落地分支根据 verdict 决定(见 §5)。

## 2. 设计

### 2.1 共享 sector context(`src/stockpool/factors/context.py` 新文件)

把 `_Wq101Context` 中的 sector_map 提升到 `factors/` 子包共享:

```python
"""Shared factor context (sector_map et al.).

Lifted from ``factors/wq101.py`` to make sector-aware factors outside wq101
share the same injection point.
"""
from __future__ import annotations

from typing import ClassVar, Mapping

import pandas as pd

from stockpool.factors import ops


class _FactorContext:
    """Module-wide context for sector-aware factors.

    Set via ``set_sector_map`` at the strategy / analysis entry point;
    factors read via ``get_sector_map`` (returns a copy).
    """
    sector_map: ClassVar[dict[str, str]] = {}


def set_sector_map(mapping: Mapping[str, str]) -> None:
    """Inject ``{code: sector_name}`` for downstream factors."""
    _FactorContext.sector_map = dict(mapping)


def get_sector_map() -> dict[str, str]:
    """Return a snapshot of the current sector_map (empty if unset)."""
    return dict(_FactorContext.sector_map)


def indneutralize_with_context(x: pd.DataFrame) -> pd.DataFrame:
    """Industry-neutralise ``x`` (T×N) using current sector context.

    If sector_map is empty, falls back to horizontal demean
    (cross-sectional mean subtraction).
    """
    if _FactorContext.sector_map:
        return ops.indneutralize(x, _FactorContext.sector_map)
    return x.sub(x.mean(axis=1), axis=0)
```

**`factors/wq101.py` 兼容改造**:

```python
# wq101.py 顶部
from stockpool.factors.context import (
    _FactorContext,           # noqa: F401 — backward compat
    set_sector_map,           # noqa: F401 — re-export
    get_sector_map,           # noqa: F401 — re-export
    indneutralize_with_context as _indneutralise,
)

# 原 _Wq101Context / set_sector_map 局部定义全部删除
# 原 def _indneutralise(x): if _Wq101Context.sector_map: ... 改为直接 import
```

**外部调用方影响**:`from stockpool.factors.wq101 import set_sector_map` 仍可工作(re-export);新代码推荐 `from stockpool.factors.context import set_sector_map`。

### 2.2 新增 custom 因子(`src/stockpool/factors/custom.py` 新文件)

```python
"""A-share specific custom factors (panel-native).

补 WQ101 没覆盖的 A 股专属信号:同业超额收益、涨停频次、异常活跃度。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from stockpool.factors.base import Factor
from stockpool.factors.context import get_sector_map
from stockpool.factors.registry import register


@register(
    "industry_relative_strength",
    sources=("custom",),
    types=("momentum", "industry_neutral", "cross_sectional"),
    description="N 日动量减去同行业中位动量 (sector_map 通过 factors.context 注入)",
)
class IndustryRelativeStrengthFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"industry_relative_strength_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ret = close.pct_change(self.n, fill_method=None)   # T×N

        sector_map = get_sector_map()
        if not sector_map:
            return pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)

        # 给每只票打 sector 标签;不在 sector_map 里的票标 "__unknown__" 并最终 NaN
        sector_series = pd.Series(
            {code: sector_map.get(code, "__unknown__") for code in ret.columns},
            name="sector",
        )

        # groupby column -> sector, 对每日做 cross-sec median, broadcast 回去
        # transform(median) 在 sector 内只有 1 列时退化为该值本身;
        # 我们用 (count > 1) mask 把 singleton sector 置 NaN,避免自减自
        groups = ret.T.groupby(sector_series)
        sector_median = groups.transform("median").T              # T×N
        sector_count = groups.transform("count").T                # T×N
        sector_median = sector_median.where(sector_count >= 2, np.nan)

        result = ret - sector_median

        # __unknown__ 列整体置 NaN
        unknown_cols = [c for c in result.columns
                        if sector_map.get(c, "__unknown__") == "__unknown__"]
        if unknown_cols:
            result[unknown_cols] = np.nan
        return result


@register(
    "limit_up_count",
    sources=("custom",),
    types=("momentum", "time_series"),
    description="近 N 日触及涨停 (close > prev_close × 1.099) 的次数",
)
class LimitUpCountFactor(Factor):
    def __init__(self, n: int = 20):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"limit_up_count_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = panel["close"]
        ret = close.pct_change(fill_method=None)
        # 主板涨停 10%, 留 0.1% tolerance 免 round-off; ST/科创/北交已被 fetch-universe 过滤
        is_limit_up = (ret > 0.099).astype(float)
        # 第一行 ret 是 NaN -> astype(float) 是 0.0, 但我们希望 warmup 阶段是 NaN
        is_limit_up.iloc[0] = np.nan
        return is_limit_up.rolling(self.n, min_periods=self.n).sum()


@register(
    "turnover_zscore",
    sources=("custom",),
    types=("volume", "time_series"),
    description="log(volume) 的 N 日时间序列 z-score,反映异常活跃度",
)
class TurnoverZScoreFactor(Factor):
    def __init__(self, n: int = 60):
        if n <= 0:
            raise ValueError(f"window must be > 0, got {n}")
        self.n = n

    @property
    def name(self) -> str:
        return f"turnover_zscore_{self.n}"

    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        v = panel["volume"].replace(0.0, np.nan)
        lv = np.log(v)
        mean = lv.rolling(self.n, min_periods=self.n).mean()
        std = lv.rolling(self.n, min_periods=self.n).std(ddof=0)
        std = std.replace(0.0, np.nan)
        return (lv - mean) / std
```

### 2.3 sector_map 注入接线

5 个调用点要在分析 / 训练前调 `set_sector_map`:

1. **`cli.py:cmd_factors_analyze`** — `factors analyze` 子命令,build panel 前先 `load_or_build_industry_map(cfg.data.cache_dir, source="auto")` → `set_sector_map(...)`
2. **`cli.py:_prepare_ml_pool`** — ml_factor 训练 / predict 路径(both `training_universe=pool` and `=all`)
3. **`strategy_factory.py:build_factor_panel`** — 顶层 panel 预算 helper(若 panel 涉及任何 sector-aware 因子,通过检查 `factor_specs` 的 types 含 `industry_neutral` 决定;否则跳过避免不必要 IO)
4. **`recommend_pool.py:compute_or_load_pool_b`** — Pool B 路径已经有 `load_or_build_industry_map`(老用法),改成同样调 `set_sector_map` 后,共享一份缓存,避免重复加载
5. **`factors_analysis.py:analyze_factors`** — 不直接负责加载(由 CLI 注入),但要在 docstring 注明:"如果待分析因子列表里含 industry_neutral 类因子,调用方必须先 `set_sector_map`"

**注入幂等性**:`set_sector_map` 是 ClassVar 赋值,多次调用以最后一次为准,无累积副作用。

**性能**:`load_or_build_industry_map` 自带 mtime 缓存(30 日有效期),冷启动 5-10s(baostock)/ 1-2min(akshare),热路径 <100ms。

### 2.4 reports/selection.json 落盘格式

复用 F1 plan-1 已定义的格式,`factors pick-by-ic` 直接产出:

```json
{
  "factors": ["alpha_017", "industry_relative_strength_20", "macd_hist", ...],
  "metadata": {
    "generated_at": "2026-05-24T...",
    "source_analysis": "reports/factor_analysis/2026-05-24.json",
    "top_n": 20,
    "max_correlation": 0.6,
    "min_ir": 0.05,
    "universe": "all",
    "horizon": 3,
    "ic_window": 252
  }
}
```

`MLFactorConfig.factors_file` 现有逻辑只读 `factors` 字段,`metadata` 段是文档性,被 ignore。

## 3. A/B 执行计划

### 3.1 步骤

```
Step 0  前置检查:data/universe.parquet 存在?
        否 → 报错,要求先 `python -m stockpool fetch-universe`

Step 1  实现 3 个 custom 因子 + sector context 重构 + 测试通过
        pytest tests/test_factors_custom.py tests/test_factors_context.py
        pytest tests/  # 全套不退化

Step 2  全市场因子排名
        python -m stockpool factors analyze \
            --universe all --horizon 3 --ic-window 252 \
            --output reports/factor_analysis
        → reports/factor_analysis/<date>.json + .html
        → 覆盖 114 个因子 (10 builtin + 101 wq101 + 3 custom)

Step 3  Top-20 去相关
        python -m stockpool factors pick-by-ic \
            --input reports/factor_analysis/<date>.json \
            --output reports/selection.json \
            --top-n 20 --max-corr 0.6 --min-ir 0.05
        → reports/selection.json

Step 4  起 A/B yaml: docs/ab_runs/p5_1_old8_vs_top20.yaml
        base_config: config.yaml
        arms:
          baseline:
            strategy:
              name: ml_factor
              ml_factor:
                factors: [<当前 8 个>]
                # horizon/selector/weighter/... 继承 base
          candidate:
            strategy:
              name: ml_factor
              ml_factor:
                factors_file: reports/selection.json

Step 5  python -m stockpool ab --config docs/ab_runs/p5_1_old8_vs_top20.yaml
        → reports/ab/p5_1_old8_vs_top20_<date>.html

Step 6  读 verdict,按 §3.2 决策门落地
```

### 3.2 决策门

沿用 F1 plan-1 spec 的乘法门(与 `strategy_improvement_2026.md` §3.4 一致):

设 `r = sharpe_candidate / sharpe_baseline`(同时也对照年化收益的绝对差):

| 条件 | 判定 | 动作 |
|---|---|---|
| `r ≥ 1.10` **或** 年化收益绝对改善 ≥ +2pp | ✅ **success target** | `config.yaml` 默认改 `factors_file: reports/selection.json`;commit selection.json + 分析报告;F1 plan-2 在 roadmap §6 标 ✅ pass |
| `r ∈ [0.80, 1.10)` 且年化收益绝对差 ∈ (-2pp, +2pp) | ⚠️ **tied / no harm** | **保持** baseline 默认(沿用 P2-1 embargo 惯例);selection.json 仍 commit 作存档;roadmap §6 标 ⚠️ tied |
| `r < 0.80` **或** 年化收益绝对差 ≤ -2pp | ❌ **regression** | baseline 保留;**selection.json 不 commit**(避免被人误用为默认);只 commit 分析报告 + custom 因子实现;roadmap §6 标 🧊 暂搁 |

**边界**:若 `sharpe_baseline ≤ 0`(罕见,但要写清),乘法门失效,改用绝对差:`Δsharpe ≥ +0.10` 视为 pass,`Δsharpe ≤ -0.10` 视为 regression,中间视为 tied。

**当出现"sharpe 改善但年化收益退化"或反过来的冲突情况**:按 sharpe 主导(因为 sharpe 已经做了风险归一)。年化收益的 ±2pp 只作为"绝对收益显著时的额外 pass 通道"和"绝对收益显著退化时的额外 regression 触发"。

### 3.3 风险与边界

- **样本不匹配**:`factors analyze` 在 ~4350 票 IC ranking,但 A/B 回测在 16 只 cfg.stocks。已知"全市场选出的 top-20 可能在小池上不显著"——A/B 决定最终
- **industry_relative_strength 在小池子上的退化**:cfg.stocks 16 只票里 9 只半导体,industry_relative_strength_20 在"半导体内部减半导体中位"会主导信号,可能与简单 momentum_20 强相关。这是 candidate 在小池上的已知风险因素,A/B 揭示
- **fetch-universe 是否新鲜**:`load_universe_cache` 不检查 mtime,如果用户 universe 是几个月前的,会用老股票池跑 IC。Step 0 应该提醒用户检查 `data/universe.parquet` mtime,但不强制 fail-loud(用户可能有意如此)
- **可复现性**:F1 plan-1 spec 已声明"同 cache_dir 跑两次 hash 一致"。本 PR 的 custom 因子也满足 panel-wise 确定性

## 4. 测试覆盖

### 4.1 新增测试

**`tests/test_factors_context.py`**(~80 行)

| 测试 | 覆盖 |
|---|---|
| `test_set_get_sector_map_roundtrip` | `set_sector_map({...})` 后 `get_sector_map()` 返回相同 dict |
| `test_get_sector_map_returns_copy` | `get_sector_map()` 修改不影响内部状态(防外部污染) |
| `test_empty_sector_map_default` | 未调用 set 时,`get_sector_map()` 返回 `{}` |
| `test_indneutralize_with_context_empty_map` | 空 sector_map 下退化为整体 demean(=`x.sub(x.mean(axis=1), axis=0)`) |
| `test_indneutralize_with_context_nonempty` | 非空 sector_map 下走 `ops.indneutralize` |
| `test_wq101_set_sector_map_reexport` | `from stockpool.factors.wq101 import set_sector_map` 仍可工作 |
| `test_sector_map_isolation` | 测试间 teardown fixture 清理 ClassVar |

**`tests/test_factors_custom.py`**(~150 行)

| 测试 | 覆盖 |
|---|---|
| `test_industry_relative_strength_basic` | 合成 panel(2 sector × 3 票),验证因子 = 自身动量 - sector 中位 |
| `test_industry_relative_strength_no_sector_map` | sector_map 空时整列 NaN |
| `test_industry_relative_strength_singleton_sector` | sector 内只剩 1 只票 → NaN(避免自减自) |
| `test_industry_relative_strength_unmapped_stock` | 某只票不在 sector_map → 该列 NaN |
| `test_industry_relative_strength_look_ahead` | 截断 panel 末尾 5 行,前缀输出与原始一致 |
| `test_limit_up_count_basic` | 合成日涨幅序列,验证计数 |
| `test_limit_up_count_warmup_nan` | 前 n 行 NaN |
| `test_limit_up_count_look_ahead` | 截断不变性 |
| `test_turnover_zscore_basic` | 合成 volume 序列,验证 z-score 数值 |
| `test_turnover_zscore_zero_volume` | volume=0 行 → NaN(停牌日) |
| `test_turnover_zscore_warmup_nan` | rolling(60) warm-up |
| `test_turnover_zscore_look_ahead` | 截断不变性 |
| `test_custom_factors_registered` | `list_specs()` 含 3 个新名字 + 正确 sources=("custom",) + 正确 types |
| `test_custom_factors_factory` | `make_factor("industry_relative_strength_20")` 工作,后缀解析正确 |

### 4.2 回归保护

- **`test_wq101.py`** — 不改逻辑,只可能因 `set_sector_map` import 路径改 fail-import;通过 re-export 保证不破
- **`test_factors.py`** — 走注册表 + factory,自动覆盖新增 3 个因子(扫 `list_specs()`)
- **`test_factors_analysis.py`** / **`test_cli_factors_analyze.py`** — `analyze_factors` 入口签名不变;新增 `test_industry_neutral_factor_in_analyze` 用小型合成数据 + 显式 `set_sector_map` 验证含 industry_neutral 因子时 context 正确生效

### 4.3 不测的

- A/B 跑结果本身(执行步骤,非单元测试)
- `factors analyze --universe all` 端到端(~1 分钟,放 manual run)
- HTML 报告渲染(已由 `test_factors_analysis_report.py` 覆盖)

## 5. 落地分支(根据 A/B verdict)

### 5.1 ✅ Pass target(`sharpe_candidate / sharpe_baseline ≥ 1.10` 或 Δreturn ≥ +2pp)

**Commit 内容**:
- `src/stockpool/factors/context.py`(新)
- `src/stockpool/factors/custom.py`(新)
- `src/stockpool/factors/wq101.py`(改 import)
- `src/stockpool/factors_analysis.py` / `cli.py` / `strategy_factory.py` / `recommend_pool.py`(sector_map 注入)
- `tests/test_factors_context.py` / `tests/test_factors_custom.py`(新)
- `reports/factor_analysis/<date>.{json,html}`(分析存档)
- `reports/selection.json`(checked-in)
- `docs/ab_runs/p5_1_old8_vs_top20.yaml`(A/B 配置)
- `reports/ab/p5_1_old8_vs_top20_<date>.html`(A/B 报告)
- `config.yaml`:`strategy.ml_factor.factors:` 列表 → `factors_file: reports/selection.json`
- `CLAUDE.md`:模块地图加 `factors/custom.py`、`factors/context.py`;sweet spot 默认更新
- `README.md`:factors workflow 段补 selection.json checked-in 说明
- `docs/strategy_improvement_2026.md` §6:F1 plan-2 落到"已完成",标 ✅ pass
- `docs/ab_validation_results.md`:新增 P5-1 段

### 5.2 ⚠️ Tied(`sharpe_candidate / sharpe_baseline ∈ [0.80, 1.10)` 且 |Δreturn| < 2pp)

**Commit 内容**:同 §5.1,**但**:
- `config.yaml` 不改默认
- `CLAUDE.md` / `README.md` 描述 custom 因子和 selection.json **可用**但不是默认
- roadmap §6 标 ⚠️ tied / default unchanged

### 5.3 ❌ Regression(`sharpe_candidate / sharpe_baseline < 0.80` 或 Δreturn ≤ -2pp)

**Commit 内容**:
- `src/stockpool/factors/context.py`(新)— 重构本身有价值
- `src/stockpool/factors/custom.py`(新)— 因子实现保留供后续 P4 LGB 重启用
- `src/stockpool/factors/wq101.py` / 注入点(改)
- `tests/`(新)
- `reports/factor_analysis/<date>.{json,html}`(分析有价值)
- **不 commit `reports/selection.json`**(避免被误用)
- **不 commit `docs/ab_runs/p5_1_old8_vs_top20.yaml`**(或保留但加注释标失败)
- `docs/ab_validation_results.md`:记录"top-20 vs old 8 退化"
- roadmap §6:F1 plan-2 移到 🧊 暂搁,触发重启条件待 P4

## 6. Spec self-review 关注点

写完后自检:

1. **`set_sector_map` 重构不破 wq101 行为**:
   - grep 当前 `_Wq101Context.sector_map` 所有引用点
   - 验证 `factors/wq101.py` 重构后,`set_sector_map` import 路径变了但调用语义不变
   - `test_wq101.py` 应该零改动通过

2. **3 个因子无 look-ahead**:
   - `industry_relative_strength`:`pct_change(n)` 用 `close[t-n]` 和 `close[t]`,sector_map 是常量字典,无未来信息
   - `limit_up_count`:`pct_change()` + `rolling(n).sum()`,纯 backward-looking
   - `turnover_zscore`:`rolling(n).mean/std`,纯 backward-looking
   - 每个因子都有 `test_..._look_ahead` 测试

3. **决策门和现有 verdict 格式一致**:
   - 参考 `docs/ab_validation_results.md` 现有 P3-2 段,P5-1 写成同样的 7 字段指标表 + verdict 三态(✅ pass / ⚠️ tied / ❌ regression)

4. **`reports/selection.json` 的 fail-safe**:
   - 若 A/B 跑出来是 regression 但 selection.json 已经 commit 了,后续用户拉代码可能误以为"已经在用",所以决策门 §5.3 明确"不 commit selection.json"

## 7. 验证标准

走完 F1 plan-2 后,我们应该能回答:

1. **客观因子排名落地**:`reports/factor_analysis/<date>.{json,html}` 存在并 commit;包含 114 个因子的滚动 IC / IR / 半衰期 / 相关性数据
2. **A/B verdict 已出**:`reports/ab/p5_1_old8_vs_top20_<date>.html` 存在;7 指标对比表填好;`docs/ab_validation_results.md` 加 P5-1 段
3. **默认更新或保留有明确依据**:`config.yaml` 是否改默认由 §3.2 决策门决定,不由感觉决定
4. **可复现**:他人 clone repo + 跑 `factors analyze --universe all` 应该得到与 `reports/factor_analysis/<date>.json` 接近的排名(允许由于 universe.parquet 时点不同的小差异)
5. **零 wq101 回归**:`pytest tests/test_wq101.py` 通过(import 重构不破现有 alpha 行为)

## 8. 估算

| 子任务 | 估算 task 数 |
|---|---|
| sector context 重构(`factors/context.py` 新建 + wq101.py 改 import + 测试) | 1 |
| 3 个 custom 因子实现 + 测试 | 1-2 |
| 5 处 sector_map 注入接线 | 1 |
| 跑 Step 2-3(`factors analyze` + `pick-by-ic`)+ inspect 结果 | 0.5 |
| 起 ab.yaml + 跑 Step 5 A/B + 读 verdict | 0.5 |
| Step 6 落地(根据 verdict 走 §5.1/5.2/5.3 分支)+ 文档同步 | 1 |

**总计**:约 4-5 个 task,无新依赖,无 Pydantic schema 改动。
