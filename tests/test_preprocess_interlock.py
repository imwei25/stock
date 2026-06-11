"""B5:breadth×zscore 互锁(P2-5)+ 因子覆盖率总闸 + build_strategy 回退口径(P2-6)。"""
import numpy as np
import pandas as pd
import pytest

from stockpool.config import PreprocessConfig
from stockpool.ml.preprocess import apply_preprocess_pipeline


def _wide(vals_by_code, n=20):
    idx = pd.bdate_range("2025-01-06", periods=n)
    return pd.DataFrame(vals_by_code, index=idx)


def test_broadcast_factor_skips_cs_zscore():
    """行常数的广播因子在 zscore 下会被整行置 0;broadcast 标签必须跳过。"""
    n = 20
    trend = np.linspace(0.1, 0.9, n)
    fp = {
        "breadth_x": _wide({"A": trend, "B": trend, "C": trend,
                            "D": trend, "E": trend}, n),
    }
    cfg = PreprocessConfig(zscore=True, market_cap_neutralize=False, min_pool_size=2)
    out = apply_preprocess_pipeline(
        fp, cfg, factor_types={"breadth_x": ("cross_sectional", "broadcast")},
        n_codes=5,
    )
    assert np.allclose(out["breadth_x"]["A"].to_numpy(), trend), (
        "broadcast 因子不应被截面变换抹掉"
    )


def test_non_broadcast_constant_rows_warns(caplog):
    n = 20
    trend = np.linspace(0.1, 0.9, n)
    fp = {"sneaky": _wide({c: trend for c in "ABCDE"}, n)}
    cfg = PreprocessConfig(zscore=True, market_cap_neutralize=False, min_pool_size=2)
    import logging
    with caplog.at_level(logging.WARNING):
        out = apply_preprocess_pipeline(fp, cfg, factor_types={}, n_codes=5)
    assert (out["sneaky"] == 0.0).all().all()
    assert any("broadcast" in r.message for r in caplog.records), (
        "未打标签的行常数因子被 zscore 抹零时必须告警"
    )


def test_breadth_factors_carry_broadcast_tag():
    from stockpool.factors.registry import make_factor
    for name in ("breadth_advance", "breadth_above_ma"):
        f = make_factor(name)
        assert "broadcast" in f.types, f"{name} 应带 broadcast 标签"


def test_dead_factor_coverage_raises():
    """全 NaN 因子必须 fail loud(P1-1 静默事故防线)。"""
    from stockpool.ml.dataset import compute_factor_panel
    from stockpool.factors.base import Factor
    from stockpool.factors import registry as reg

    class _DeadFactor(Factor):
        @property
        def name(self):
            return "dead_test_factor"
        types = ()
        sources = ()

        def compute(self, panel):
            close = panel["close"]
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # 不经注册表,直接验证总闸函数(面板规模需达到执法门槛 ≥10 列 ≥40 行)
    from stockpool.ml.dataset import _check_factor_coverage
    idx = pd.bdate_range("2025-01-06", periods=60)
    cols = [f"S{i:02d}" for i in range(12)]
    with pytest.raises(ValueError, match="覆盖率"):
        _check_factor_coverage("dead_test_factor",
                               pd.DataFrame(np.nan, index=idx, columns=cols))


def test_low_coverage_warns_not_raises(caplog):
    from stockpool.ml.dataset import _check_factor_coverage
    import logging
    idx = pd.bdate_range("2025-01-06", periods=100)
    cols = [f"S{i:02d}" for i in range(12)]
    wide = pd.DataFrame(np.nan, index=idx, columns=cols)
    wide.iloc[90:] = 1.0  # 10% 覆盖:介于 dead(2%) 与 warn(25%) 之间
    with caplog.at_level(logging.WARNING):
        _check_factor_coverage("warmup_factor", wide)
    assert any("覆盖率" in r.message for r in caplog.records)


def test_small_panel_exempt_from_coverage_gate():
    from stockpool.ml.dataset import _check_factor_coverage
    idx = pd.bdate_range("2025-01-06", periods=80)
    # 2 列玩具面板:全 NaN 也不应 raise(单测夹具场景)
    _check_factor_coverage("toy", pd.DataFrame(np.nan, index=idx, columns=["A", "B"]))


def test_build_strategy_fallback_applies_preprocess(monkeypatch):
    """P2-6:build_strategy 内部回退建面板时必须带 preprocess_cfg。"""
    import stockpool.strategy_factory as sf
    from stockpool.config import load_config
    captured = {}

    def fake_build_factor_panel(names, pool_data, preprocess_cfg=None):
        captured["preprocess_cfg"] = preprocess_cfg
        return {}

    monkeypatch.setattr(sf, "build_factor_panel", fake_build_factor_panel)
    monkeypatch.setattr(sf, "maybe_inject_mcap_panel", lambda *a, **k: None)

    import textwrap, tempfile, os
    yaml_text = textwrap.dedent("""
    stocks: [{code: "000001", name: "x"}]
    data: {history_days: 100, cache_dir: "data"}
    indicators:
      ma_periods: [5, 20, 60]
      macd: {fast: 12, slow: 26, signal: 9}
      kdj: {n: 9, m1: 3, m2: 3}
      rsi_periods: [6, 12, 24]
      boll: {n: 20, k: 2.0}
      volume_ratio_window: 5
      breakout_window: 20
    weights:
      ma_cross_strong: 2
      ma_alignment: 1
      macd_cross_above_zero: 2
      macd_cross_below_zero: 2
      macd_histogram_expand: 1
      kdj_oversold_cross: 2
      kdj_overbought_cross: 2
      kdj_normal_cross: 1
      rsi_oversold: 1
      rsi_overbought: 1
      boll_band_touch: 1
      boll_mid_cross: 1
      volume_surge_bullish: 1
      volume_surge_bearish: 1
      breakout_new_high: 2
      breakout_new_low: 2
    scoring:
      daily_weight: 0.7
      weekly_weight: 0.3
      resonance_bonus: 2
      resonance_daily_threshold: 3
      resonance_weekly_threshold: 1
    verdicts: {strong_buy: 6, buy: 3, sell: -3, strong_sell: -6}
    backtest: {forward_days: [5]}
    report: {output_dir: "reports", keep_history: true, klines_to_show: 60}
    strategy:
      name: ml_factor
      ml_factor: {panel_mode: pooled}
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(yaml_text)
        path = fh.name
    try:
        cfg = load_config(path)
        daily = pd.DataFrame({
            "date": pd.bdate_range("2025-01-06", periods=30),
            "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
            "volume": 1e6,
        })
        sf.build_strategy(cfg, pool_data={"000001": daily})
    finally:
        os.unlink(path)

    assert captured.get("preprocess_cfg") is not None, (
        "回退路径必须透传 preprocess_cfg"
    )
