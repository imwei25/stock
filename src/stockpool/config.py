"""Config schema + loader. Pydantic does the validation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
