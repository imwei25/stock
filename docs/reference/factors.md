# 因子库参考 (`stockpool.factors`)

约 165 个 base 因子(变体计含 ~280-320):WQ101 全集 + 内置技术因子(论文 B 9 家族精神复现 + EWMA)+ 基本面 custom。模块清单见 [modules.md](modules.md) 因子库段。

## Factor ABC — panel-in / panel-out

```python
class Factor(ABC):
    sources: tuple[str, ...] = ("builtin",)   # 来源标签
    types: tuple[str, ...] = ()               # 类型多标签
    description: str
    def compute(self, panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame: ...
```

Panel = `{"open"|"high"|"low"|"close"|"volume": T×N DataFrame}`,行 = date,列 = code。每个因子是纯函数(`T×N OHLCV 宽表 → T×N 因子值宽表`)。

## 双轴元数据

- **sources**:`builtin`(~51,老技术因子 + 论文 B 9 家族 + EWMA)/ `wq101`(101,WorldQuant 101 Formulaic Alphas)/ `custom`(~13,含基本面 PE/PB/ROE/营收增速)
- **types**:`momentum` / `reversal` / `trend` / `volatility` / `volume` / `time_series` / `cross_sectional` / `industry_neutral` / `fundamental`(每因子可多标签)

**注册表 API**:`list_specs()` / `filter_specs(sources=, types=, match='any')` / `all_sources()` / `all_types()` / `make_factor(name)`。

## 算子库 (`factors/ops.py`,WQ101 必需)

- **时间序列**:`ts_sum` `ts_mean` `ts_min/max` `ts_argmin/argmax` `ts_rank` `ts_std` `ts_product` `delta` `delay` `decay_linear` `correlation` `covariance`
  - `correlation` 出口数值卫生(2026-06):±inf → NaN、有限值 clip [-1, 1]。近常数窗口(平盘日 close 两天位级相等)下 pandas 矩量公式会产出 ±inf(数学真值 0/0 未定义)或 |ρ|>1;短窗口因子(如 alpha_045 的 `corr(close, volume, 2)`)平盘日约占截面 1.7%,inf 流出会绕过下游 isnan 防线毒化训练矩阵。修复后平盘日 → NaN → 训练整行剔除(边际样本损失 ~2.9%)、predict 均值填充(中性)。
- **横截面**:`rank`(axis=1, pct=True)`scale`(L1 norm)`signedpower`
- **行业中性**:`indneutralize(x, group_map)` — 按 group 分组 demean
- **工具**:`safe_div` `vwap`((H+L+C)/3 proxy)`adv(volume, d)`

## WQ101 (`factors/wq101.py`)

全 101 个 alpha,名字 `alpha_001` .. `alpha_101`。

- 注入 `set_sector_map({code: sector})` 后,所有 `IndNeutralize` 退化的 alpha 走分组 demean;未注入则退化为整体 demean
- `IndClass.subindustry` 一律 fall back 到 sector(项目无 subindustry 数据)
- `Alpha056` 需要 `cap`(总市值),目前返回全 NaN

**预算因子面板**:`strategy_factory.build_strategy` 在 `panel_mode=pooled` 且有 `pool_data` 时调 `build_factor_panel` 预算 `{factor_name: T×N}` 并注入 `MLFactorStrategy`。`generate_signals` 通过 `_build_x_full` 切出本股 X(`slice_stock_factor_matrix`),cross-sec 因子在 predict 阶段也用真实横截面。不注入时 fall back 到 `build_factor_matrix` 单股退化。CLI 在循环外预算 panel 一次。

**训练/应用池分离**:`training_universe=all` 时,`cli._prepare_ml_pool` 用 `load_universe_cache` 装全市场 ~4350 票作 pool_data(cfg.stocks 仍 merge 进去);predict 仍只对 cfg.stocks 跑。cross-sec 因子和 IC 加权拿到全市场横截面,日报/回测标的仍是 cfg.stocks。

**`IndustryRelativeStrengthFactor`** 在 `get_sector_map()` 为空时 **raise**(防 factor_panel cache 中毒)— sector_map 必须在 build factor_panel 前由 caller 经 `factors.context.set_sector_map(...)` 注入。

## 2026-05-31 因子扩展(11 家族 + 基本面)

114 → ~165 base。11 家族:VWAP 偏离 / 收盘位置 / 秩相关 / 单股波动 / 短窗换手 / 复合 / 加速度 / 直接统计 / 截面宽度 / EWMA / 基本面。命名语义化(`vwap_dev_5` / `close_pos_10` / `roe` 等)。

**基本面族 PIT 设计**:按 `pubDate`(公告日)前向填充,**不**用 `statDate`,防 ~1 个月未来泄露。首次拉 baostock 5 张季度表约 30-60 分钟,30 天缓存到 `data/fundamentals_*.parquet`,`--refresh-fundamentals` 强制重拉。`factor_panels/<sig>/manifest.json` 含 `fundamentals_snapshot_date`:fundamentals 刷新后 panel 缓存自动失效重建。

设计细节见 `docs/superpowers/specs/2026-05-31-factor-library-expansion-design.md`。

## HTML 选择器

```bash
python -m stockpool factors pick                          # 默认 server 模式
python -m stockpool factors pick --output my_sel.json     # 指定写入路径
python -m stockpool factors pick --port 18765             # 固定端口
python -m stockpool factors pick --static                 # 老的静态文件模式
```

**默认 server 模式**:起 `127.0.0.1` 本地 HTTP 服务(stdlib `http.server`),浏览器打开。顶栏 **"应用"** POST 到 `/save` 由服务端写 `reports/selection.json`(或 `--output`)。Ctrl-C 退出。页面打开时 GET `/selection.json` 把现有选择载回(服务端文件为权威源)。

**`--static` 模式**:生成静态 HTML(`file://`),"应用" 降级为"下载"。适合归档或防火墙挡 server。

左侧双轴筛选(来源 × 类型)+ 任一/全部模式;右侧卡片勾选。服务端路由(`factors_picker._make_handler`):`GET /` → 页面;`GET /selection.json` → 当前内容;`POST /save` → 写文件。

`config.yaml` 引用:

```yaml
strategy:
  ml_factor:
    factors_file: reports/selection.json   # 与 factors: [...] 二选一
```

## 因子分析 → 选因子

```bash
# 跑分析输出 HTML + JSON
python -m stockpool factors analyze --universe all --output reports/factor_analysis

# 从分析 JSON 自动选 top-N 去相关因子,写成 factors_file 兼容的 selection.json
python -m stockpool factors pick-by-ic \
  --input reports/factor_analysis/<日期>.json \
  --output reports/selection.json \
  --top-n 20 --max-corr 0.6 --min-ir 0.05
```

`factors_analysis.py` 算滚动 IC / IR / half-life / 相关性 / regime 切片(`analyze_factors` + `pick_top_factors`);报告由 `factors_analysis_report.py` 渲染。
