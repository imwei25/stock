"""Config schema + loader. Pydantic does the validation."""
from __future__ import annotations

import hashlib
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


class BacktestConfig(BaseModel):
    forward_days: list[int]
    equity_curve_holding_days: list[int] = Field(default_factory=lambda: [5, 10, 20])
    risk_free_rate: float = 0.02
    costs: BacktestCostConfig = Field(default_factory=BacktestCostConfig)
    engine: Literal["single", "multi_lot"] = "multi_lot"
    position_size: float = Field(default=0.1, gt=0.0, le=1.0)
    max_concurrent_lots: int | None = Field(default=None, gt=0)

    @field_validator("equity_curve_holding_days")
    @classmethod
    def _validate_holding_days(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("equity_curve_holding_days must be a non-empty list")
        if any(n <= 0 for n in v):
            raise ValueError("equity_curve_holding_days entries must be positive integers")
        return v


class ReportConfig(BaseModel):
    output_dir: str
    keep_history: bool
    klines_to_show: int


# === ML factor strategy ===

class SelectorConfig(BaseModel):
    """Step-1 (factor selection) settings.

    Currently only ``type: lasso`` is supported; ``alpha`` is the L1 penalty
    strength on standardised features (typical range 1e-4 — 1e-1).
    """
    type: Literal["lasso"] = "lasso"
    alpha: float = Field(default=0.001, ge=0.0)
    max_iter: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-6, gt=0.0)


class WeighterConfig(BaseModel):
    """Step-2 (factor weighting) settings.

    * ``ic``: weight ∝ Spearman/Pearson IC against the target.
    * ``ir``: weight ∝ information ratio (mean(IC)/std(IC)) over sub-windows.
    * ``equal``: equal weight on every selected factor (baseline).
    """
    type: Literal["ic", "ir", "equal"] = "ic"
    use_rank: bool = True
    min_abs_ic: float = Field(default=0.0, ge=0.0)
    # IR-only
    n_chunks: int = Field(default=6, gt=0)
    min_abs_ir: float = Field(default=0.0, ge=0.0)


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
            # 用文件覆盖默认 factors;若用户两边都给且不一致 → 报错
            file_factors = list(data["factors"])
            default_factors = [
                "momentum_20", "macd_hist", "rsi_centered_14",
                "ma_distance_20", "vol_ratio_5", "boll_position_20",
            ]
            if self.factors and self.factors != default_factors:
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
