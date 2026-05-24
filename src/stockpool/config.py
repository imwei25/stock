"""Config schema + loader. Pydantic does the validation."""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Stock(BaseModel):
    code: str
    name: str
    sector: str | None = None


class IndexConfig(BaseModel):
    code: str  # e.g. "sh000001"
    name: str  # e.g. "上证指数"


class ContextConfig(BaseModel):
    indices: list[IndexConfig] = Field(default_factory=lambda: [
        IndexConfig(code="sh000001", name="上证指数"),
        IndexConfig(code="sz399001", name="深证成指"),
    ])


class DataConfig(BaseModel):
    history_days: int = Field(gt=0)
    cache_dir: str
    force_refresh: bool = False
    # 数据源后端: mootdx(通达信, 默认, 支持当日盘中) | baostock(无 token, 收盘后更新)
    # | akshare(东财爬虫, 兜底)。板块(行业)数据始终走 akshare,因为另两家不直接提供。
    source: Literal["mootdx", "baostock", "akshare"] = "mootdx"


class MACDConfig(BaseModel):
    fast: int
    slow: int
    signal: int


class KDJConfig(BaseModel):
    n: int
    m1: int
    m2: int


class BOLLConfig(BaseModel):
    n: int
    k: float


class IndicatorsConfig(BaseModel):
    ma_periods: list[int]
    macd: MACDConfig
    kdj: KDJConfig
    rsi_periods: list[int]
    boll: BOLLConfig
    volume_ratio_window: int
    breakout_window: int


class WeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ma_cross_strong: int
    ma_alignment: int
    macd_cross_above_zero: int
    macd_cross_below_zero: int
    macd_histogram_expand: int
    kdj_oversold_cross: int
    kdj_overbought_cross: int
    kdj_normal_cross: int
    rsi_oversold: int
    rsi_overbought: int
    boll_band_touch: int
    boll_mid_cross: int
    volume_surge_bullish: int
    volume_surge_bearish: int
    breakout_new_high: int
    breakout_new_low: int


class ScoringConfig(BaseModel):
    daily_weight: float
    weekly_weight: float
    resonance_bonus: int
    resonance_daily_threshold: int
    resonance_weekly_threshold: int


class VerdictsConfig(BaseModel):
    strong_buy: int
    buy: int
    sell: int
    strong_sell: int


class BacktestCostConfig(BaseModel):
    commission_rate: float = 0.0003   # 双边佣金 0.03%
    stamp_duty_rate: float = 0.0005   # 卖出印花税 0.05%（2023 年后减半）
    slippage_rate: float = 0.0005     # 单边冲击成本估算 0.05%

    @property
    def buy_cost(self) -> float:
        return self.commission_rate + self.slippage_rate

    @property
    def sell_cost(self) -> float:
        return self.commission_rate + self.stamp_duty_rate + self.slippage_rate


class FixedSizingConfig(BaseModel):
    """Constant lot size — every buy commits the same fraction of capital."""
    model_config = ConfigDict(extra="forbid")
    size: float = Field(default=0.1, gt=0.0, le=1.0)


class VolTargetSizingConfig(BaseModel):
    """Vol-target sizing — scale each lot inversely to recent stock vol.

    Formula (β, relative-to-baseline):
        size = fixed.size * (reference_vol_annual / recent_vol_annual)
        size = clip(size, min_size, max_size)

    ``fixed.size`` doubles as the baseline anchor: at recent_vol = reference_vol,
    the lot equals fixed.size. Vol estimator: simple rolling std over
    ``vol_window`` bars of daily simple returns, annualised with sqrt(252).
    """
    model_config = ConfigDict(extra="forbid")
    reference_vol_annual: float = Field(default=0.30, gt=0.0)
    vol_window: int = Field(default=20, gt=1)
    min_size: float = Field(default=0.03, gt=0.0, le=1.0)
    max_size: float = Field(default=0.20, gt=0.0, le=1.0)
    fallback_to: Literal["fixed", "skip"] = "fixed"

    @model_validator(mode="after")
    def _check_min_le_max(self) -> "VolTargetSizingConfig":
        if self.min_size > self.max_size:
            raise ValueError(
                f"min_size ({self.min_size}) must be <= max_size ({self.max_size})"
            )
        return self


class SizingConfig(BaseModel):
    """Per-lot sizing strategy. Default flipped to vol_target in F3 PR-C."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["fixed", "vol_target"] = "vol_target"
    fixed: FixedSizingConfig = Field(default_factory=FixedSizingConfig)
    vol_target: VolTargetSizingConfig = Field(default_factory=VolTargetSizingConfig)


class BacktestConfig(BaseModel):
    forward_days: list[int]
    equity_curve_holding_days: list[int] = Field(default_factory=lambda: [5, 10, 20])
    risk_free_rate: float = 0.02
    costs: BacktestCostConfig = Field(default_factory=BacktestCostConfig)
    engine: Literal["single", "multi_lot"] = "multi_lot"
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    # Deprecated alias for sizing.fixed.size. None = use sizing.
    # If set alongside a non-default sizing block, raises ValueError.
    # If set alone, auto-migrates to sizing.type=fixed + emits DeprecationWarning.
    position_size: float | None = Field(default=None, gt=0.0, le=1.0)
    max_concurrent_lots: int | None = Field(default=None, gt=0)

    @field_validator("equity_curve_holding_days")
    @classmethod
    def _validate_holding_days(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("equity_curve_holding_days must be a non-empty list")
        if any(n <= 0 for n in v):
            raise ValueError("equity_curve_holding_days entries must be positive integers")
        return v

    @model_validator(mode="after")
    def _migrate_position_size(self) -> "BacktestConfig":
        if self.position_size is None:
            return self
        # Heuristic: detect "user wrote sizing explicitly" by checking only
        # sizing.type and sizing.fixed.size. We deliberately do NOT inspect
        # vol_target.* sub-fields — combining legacy position_size with an
        # explicit vol_target block is incoherent (position_size is fixed-only
        # semantics), so silently dropping vol_target overrides on migration
        # is the documented tradeoff. Spec §2.1 calls this a tolerable false
        # negative. If sizing.type and sizing.fixed.size are both at defaults,
        # we cannot distinguish "user did not write sizing" from "user wrote
        # sizing with default values" — the latter collapses to migration.
        sizing_explicit = (
            self.sizing.type != "vol_target"
            or self.sizing.fixed.size != 0.1
        )
        if sizing_explicit:
            raise ValueError(
                "Cannot set both backtest.position_size (deprecated) and "
                "backtest.sizing. Migrate position_size into sizing.fixed.size."
            )
        warnings.warn(
            "backtest.position_size is deprecated; use "
            "backtest.sizing.fixed.size (with sizing.type=fixed) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.sizing = SizingConfig(
            type="fixed",
            fixed=FixedSizingConfig(size=self.position_size),
        )
        self.position_size = None
        return self


class ReportConfig(BaseModel):
    output_dir: str
    keep_history: bool
    klines_to_show: int


# === ML factor strategy ===

class LassoConfig(BaseModel):
    """Lasso-specific hyperparameters for ``selector.type == 'lasso'``.

    ``alpha`` is the L1 penalty on standardised features (typical range 1e-4 — 1e-1).
    """
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(default=0.001, ge=0.0)
    max_iter: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-6, gt=0.0)


class LightGBMSelectorConfig(BaseModel):
    """LightGBM-based selector hyperparameters.

    Defaults are conservative for walk-forward training (small per-refit
    training set; the embedded forest is intentionally shallow). Tighten
    ``num_leaves`` / increase ``min_data_in_leaf`` if observed IC is unstable
    across refits.
    """
    model_config = ConfigDict(extra="forbid")
    num_leaves: int = Field(default=15, gt=1)
    min_data_in_leaf: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_iterations: int = Field(default=200, gt=0)
    max_depth: int = Field(default=4, gt=0)
    random_state: int = Field(default=42, ge=0)
    top_k_factors: int = Field(default=20, gt=0)
    min_importance_ratio: float = Field(default=0.01, ge=0, le=1)
    verbose: int = Field(default=-1)


class SelectorConfig(BaseModel):
    """Step-1 (factor selection) settings.

    PR-A introduced ``selector.lasso.*`` subnesting. PR-B1 added
    ``selector.lightgbm.*`` as a parallel block; the default flipped to
    ``"lightgbm"`` initially but was rolled back to ``"lasso"`` on
    2026-05-24 after A/B validation
    (`docs/ab_validation_results.md`) showed LGB selector + LGB weighter
    regressed sharpe by 0.20 and total return by 20% on the 16-stock ×
    500-bar baseline. LGB remains opt-in via ``type: lightgbm``.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["lasso", "lightgbm"] = "lasso"
    lasso: LassoConfig = Field(default_factory=LassoConfig)
    lightgbm: LightGBMSelectorConfig = Field(default_factory=LightGBMSelectorConfig)


class ICWeighterConfig(BaseModel):
    """IC weighter hyperparameters (was flat fields on WeighterConfig pre-PR-B2)."""
    model_config = ConfigDict(extra="forbid")
    use_rank: bool = True
    min_abs_ic: float = Field(default=0.0, ge=0.0)


class IRWeighterConfig(BaseModel):
    """IR weighter hyperparameters.

    ``IRWeighter`` internally uses ``use_rank`` to choose Spearman vs Pearson
    when computing per-chunk IC; ``min_abs_ir`` filters factors by IR magnitude.
    """
    model_config = ConfigDict(extra="forbid")
    n_chunks: int = Field(default=6, gt=0)
    use_rank: bool = True
    min_abs_ir: float = Field(default=0.0, ge=0.0)


class EqualWeighterConfig(BaseModel):
    """Equal weighter has no hyperparameters; placeholder for uniform structure."""
    model_config = ConfigDict(extra="forbid")


class LightGBMWeighterConfig(BaseModel):
    """LightGBM weighter hyperparameters. Defaults match LightGBMSelectorConfig."""
    model_config = ConfigDict(extra="forbid")
    num_leaves: int = Field(default=15, gt=1)
    min_data_in_leaf: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_iterations: int = Field(default=200, gt=0)
    max_depth: int = Field(default=4, gt=0)
    random_state: int = Field(default=42, ge=0)
    verbose: int = Field(default=-1)


class WeighterConfig(BaseModel):
    """Step-2 (factor weighting) settings.

    PR-B2 refactors this from flat fields to subnested per-type blocks
    (ic / ir / equal / lightgbm), parallel to PR-A's SelectorConfig.
    Default ``type`` flipped to ``"lightgbm"`` initially but was rolled
    back to ``"ic"`` on 2026-05-24 after A/B validation
    (`docs/ab_validation_results.md`) showed LGB weighter contributed
    ~-12% return on the small training-set baseline. LGB remains opt-in
    via ``type: lightgbm``.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["ic", "ir", "equal", "lightgbm"] = "ic"
    ic: ICWeighterConfig = Field(default_factory=ICWeighterConfig)
    ir: IRWeighterConfig = Field(default_factory=IRWeighterConfig)
    equal: EqualWeighterConfig = Field(default_factory=EqualWeighterConfig)
    lightgbm: LightGBMWeighterConfig = Field(default_factory=LightGBMWeighterConfig)


class QuantileThresholds(BaseModel):
    """Map continuous predicted scores → discrete verdicts via training-set quantiles.

    Quantiles are computed once per refit on the training-window predictions.
    The order must be strong_sell < sell < buy < strong_buy ∈ (0, 1).
    """
    strong_buy: float = Field(default=0.90, gt=0.0, lt=1.0)
    buy: float = Field(default=0.70, gt=0.0, lt=1.0)
    sell: float = Field(default=0.30, gt=0.0, lt=1.0)
    strong_sell: float = Field(default=0.10, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_order(self) -> "QuantileThresholds":
        if not (
            self.strong_sell < self.sell < self.buy < self.strong_buy
        ):
            raise ValueError(
                "thresholds must satisfy strong_sell < sell < buy < strong_buy"
            )
        return self


class MLFactorConfig(BaseModel):
    """Settings for the two-step ML factor strategy.

    Decoupled from the legacy ``weights``/``scoring``/``verdicts`` block; both
    strategies can coexist in the same config (chosen via ``StrategyConfig``).
    """
    model_config = ConfigDict(extra="forbid")

    factors: list[str] = Field(default_factory=lambda: [
        "momentum_20", "macd_hist", "rsi_centered_14",
        "ma_distance_20", "vol_ratio_5", "boll_position_20",
    ])
    # 或者从 HTML 选择器导出的 JSON 文件加载因子列表(与 factors 二选一)。
    # JSON 格式: {"factors": ["alpha_001", "momentum_20", ...]}
    factors_file: str | None = None
    horizon: int = Field(default=5, gt=0)
    train_window: int = Field(
        default=250, gt=0,
        description=(
            "Per-stock recency window in bars. In per_stock mode this is also "
            "the total training-sample count. In pooled mode each stock "
            "contributes at most this many of its most-recent rows, so the "
            "total sample count ≈ train_window × (# active stocks)."
        ),
    )
    min_train_samples: int = Field(
        default=60, gt=0,
        description="Minimum non-NaN rows required before the first fit.",
    )
    refit_every: int = Field(default=20, gt=0)
    panel_mode: Literal["per_stock", "pooled"] = "per_stock"
    # 训练用股池:
    #   pool — 仅用 cfg.stocks(应用池,向后兼容,默认)
    #   all  — 用 data/ 缓存中已 fetch-universe 拉到的全市场 A 股(剔除 ST/科创/北交)
    # 仅在 panel_mode=pooled 时生效;per_stock 永远是单股训练,该选项被忽略。
    # 注意:all 需先跑 `python -m stockpool fetch-universe` 准备全市场缓存。
    training_universe: Literal["pool", "all"] = "pool"
    # pooled 模式下:训练池跨股票共享一次 fit/月,而非每股每 refit_bar 重训。
    # 启用后 `_build_truncated_pool` 不再剔除 host,训练集对所有 host 一致 →
    # 缓存键 (sig, year, month),同月内所有股、所有 refit_bar 复用同一 pipeline。
    # 代价:host 自身历史以 ~1/N 权重进入自己的训练(N=pool 大小),IC 加权下可忽略。
    share_pool_fit: bool = True
    # Walk-forward embargo: extra gap (in bars) between train window end and
    # the test bar, to prevent horizon-day forward returns from leaking into
    # training labels. ``None`` means "auto = horizon" (recommended default).
    # Set to ``0`` to opt out and reproduce pre-PR-A behavior.
    embargo_days: int | None = Field(default=None, ge=0)

    # Training-label transform. PR-A only implements "return" (the legacy
    # absolute forward return). "vol_adjusted" and "cross_sec_rank" are
    # interface placeholders — calls into the corresponding code path will
    # raise NotImplementedError until a later PR fills them in.
    label_type: Literal["return", "vol_adjusted", "cross_sec_rank"] = "return"

    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    weighter: WeighterConfig = Field(default_factory=WeighterConfig)
    thresholds: QuantileThresholds = Field(default_factory=QuantileThresholds)
    # Persistent enter/exit verdict sets (mirrors CompositeVerdictStrategy).
    buy_verdicts: list[str] = Field(default_factory=lambda: ["buy", "strong_buy"])
    sell_verdicts: list[str] = Field(default_factory=lambda: ["sell", "strong_sell"])
    refresh_verdicts: list[str] = Field(default_factory=lambda: ["strong_buy"])

    @model_validator(mode="after")
    def _load_factors_file(self) -> "MLFactorConfig":
        if self.factors_file:
            import json
            data = json.loads(Path(self.factors_file).read_text(encoding="utf-8"))
            if "factors" not in data or not isinstance(data["factors"], list):
                raise ValueError(
                    f"factors_file {self.factors_file!r} must contain "
                    f"a 'factors' list"
                )
            # 用文件覆盖默认 factors;若用户两边都给且不一致 → 报错。
            # 例外:如果 factors 已经等于文件内容(model_dump → model_validate
            # round-trip,例如经过 ab.build_effective_cfg),允许通过。
            file_factors = list(data["factors"])
            default_factors = [
                "momentum_20", "macd_hist", "rsi_centered_14",
                "ma_distance_20", "vol_ratio_5", "boll_position_20",
            ]
            if (
                self.factors
                and self.factors != default_factors
                and self.factors != file_factors
            ):
                raise ValueError(
                    "Cannot specify both 'factors' and 'factors_file'; "
                    "remove 'factors' to use the file"
                )
            object.__setattr__(self, "factors", file_factors)
        return self


class StrategyConfig(BaseModel):
    """Top-level strategy selector. ``name`` picks the strategy implementation."""
    model_config = ConfigDict(extra="forbid")
    name: Literal["composite_verdict", "ml_factor"] = "composite_verdict"
    ml_factor: MLFactorConfig = Field(default_factory=MLFactorConfig)


class RecommendPoolConfig(BaseModel):
    """Pool B — 全市场量化推荐池.

    在保留 ``cfg.stocks``(Pool A, 手填 watchlist)的前提下,对全市场 A 股
    调用 ``cfg.strategy`` 的 ``predict_latest`` 打分,经 "流动性 + ST 剔除 +
    行业上限" 漏斗后取 top-N,作为日报底部的推荐池。两池独立、允许重叠。

    周缓存:跨过 ISO 周边界才重算(``refresh="weekly"``,默认);缓存键含
    ``cfg.content_hash``,改任何 yaml 字段都自动失效。

    Pool B 不做回测(MVP);留作 follow-up。
    """
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    top_n: int = Field(default=30, gt=0)
    min_avg_amount_20d: float = Field(
        default=5e7, ge=0.0,
        description="最近 20 日均成交额下限 (元)。mootdx volume 单位是手, "
                    "amount = volume * close * 100。",
    )
    max_per_industry: int = Field(default=5, gt=0)
    refresh: Literal["weekly", "always", "never"] = "weekly"
    cache_dir: str = "data/recommend_pool"
    industry_map_max_age_days: int = Field(default=30, gt=0)
    # baostock = 一次性 5500+ 行,稳;akshare = 逐板块拉,慢且受代理影响;
    # auto = 先 baostock 后 akshare
    industry_source: Literal["auto", "baostock", "akshare"] = "auto"


class AppConfig(BaseModel):
    """Root config. `content_hash` is set post-load, not in YAML."""
    stocks: list[Stock]
    data: DataConfig
    indicators: IndicatorsConfig
    weights: WeightsConfig
    scoring: ScoringConfig
    verdicts: VerdictsConfig
    backtest: BacktestConfig
    report: ReportConfig
    context: ContextConfig = Field(default_factory=ContextConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    recommend_pool: RecommendPoolConfig = Field(default_factory=RecommendPoolConfig)

    content_hash: str = ""


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config and validate against schema.

    Raises pydantic.ValidationError on missing fields or wrong types.
    """
    raw_bytes = Path(path).read_bytes()
    parsed = yaml.safe_load(raw_bytes)
    cfg = AppConfig.model_validate(parsed)
    cfg.content_hash = hashlib.sha256(raw_bytes).hexdigest()[:8]
    return cfg
