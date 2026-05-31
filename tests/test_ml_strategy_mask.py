"""Tests for tradability mask integration in ml/dataset pipeline.

设计哲学(2026-05-31 重构后):
  - mask **不** 应用到时间序列因子输入面板(`compute_factor_panel` /
    `build_factor_panel` / `build_factor_matrix` 都不接受 mask)。
    时间序列因子需要看真实 close(包括涨停日 +9.9% 这种有用信号)。
  - mask **只** 应用到:
      1. forward_return_panel 双向标签检查
      2. 训练样本筛选(通过 label NaN 自然 dropna)
  - factor_panel 缓存 sig 是 mask-agnostic 的(同因子列表 + 同股池 +
    同 last_date → 同 sig,与 mask 设定无关)。

详见 docs/handoff/2026-05-31-mask-ab-investigation.md 的演变。
"""
import numpy as np
import pandas as pd
import pytest


def _make_panel(close_dict):
    codes = list(close_dict.keys())
    n = len(next(iter(close_dict.values())))
    idx = pd.date_range("2024-01-01", periods=n)
    close = pd.DataFrame(close_dict, index=idx)
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": pd.DataFrame({c: [1000.0] * n for c in codes}, index=idx),
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_factor_panel — 不再接 mask 参数
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_factor_panel_no_mask_param():
    """compute_factor_panel 不接受 mask kwarg(2026-05-31 重构移除)。"""
    from stockpool.ml.dataset import compute_factor_panel
    panel = _make_panel({"600000": list(np.linspace(10, 11, 30))})
    # 没有 mask kwarg
    out = compute_factor_panel(panel, ["momentum_5"])
    assert "momentum_5" in out
    # 试图传 mask=... 会 TypeError
    with pytest.raises(TypeError, match="mask"):
        compute_factor_panel(panel, ["momentum_5"], mask=None)


def test_compute_factor_panel_sees_real_close():
    """涨停日的 +9.9% close 也会进入因子计算(不被 NaN 掉)。"""
    from stockpool.ml.dataset import compute_factor_panel
    closes = list(np.linspace(10, 11, 30))
    closes[10] = closes[9] * 1.099  # 第 10 天涨停
    panel = _make_panel({"600000": closes})
    out = compute_factor_panel(panel, ["momentum_5"])
    # 第 10 天 momentum_5(close[t]/close[t-5]-1)应该是较大正值
    # 注:具体值随 momentum 定义而异,只要不是 NaN(因为我们不 mask)
    assert pd.notna(out["momentum_5"].iloc[10, 0])


# ─────────────────────────────────────────────────────────────────────────────
# forward_return_panel — 仍带 mask(标签层是 mask 唯一应用点)
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_return_panel_no_mask_unchanged():
    from stockpool.ml.dataset import forward_return_panel
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]})
    y_a = forward_return_panel(close, horizon=2)
    y_b = forward_return_panel(close, horizon=2, mask=None)
    pd.testing.assert_frame_equal(y_a, y_b)


def test_forward_return_panel_bidirectional_mask():
    """mask[t]=False 或 mask[t+h]=False 时 y[t] = NaN(标签可执行性双向检查)。"""
    from stockpool.ml.dataset import forward_return_panel
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]})
    mask = pd.DataFrame({"A": [True, True, False, True, True]})
    y = forward_return_panel(close, horizon=2, mask=mask)
    # t=0: mask[0]=T ∧ mask[2]=F → NaN
    # t=1: mask[1]=T ∧ mask[3]=T → (13-11)/11
    # t=2: mask[2]=F → NaN
    assert np.isnan(y["A"].iloc[0])
    assert y["A"].iloc[1] == pytest.approx(2.0 / 11.0)
    assert np.isnan(y["A"].iloc[2])


# ─────────────────────────────────────────────────────────────────────────────
# build_factor_panel — 不再接 mask_config
# ─────────────────────────────────────────────────────────────────────────────

def test_build_factor_panel_no_mask_param():
    """build_factor_panel 不接受 mask_config kwarg(2026-05-31 重构移除)。"""
    from stockpool.strategy_factory import build_factor_panel
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "open": np.linspace(10, 11, 30),
        "high": np.linspace(10.1, 11.1, 30),
        "low": np.linspace(9.9, 10.9, 30),
        "close": np.linspace(10, 11, 30),
        "volume": [1000.0] * 30,
    })
    pool_data = {"600000": df}
    out = build_factor_panel(["momentum_5"], pool_data)
    assert "momentum_5" in out
    # mask_config kwarg 已移除
    with pytest.raises(TypeError, match="mask_config"):
        build_factor_panel(["momentum_5"], pool_data, mask_config=None)


def test_build_factor_panel_does_not_nan_limit_up_days():
    """限定到第 10 天 +9.9%(主板涨停);build_factor_panel 不再 NaN 掉这天的因子。"""
    from stockpool.strategy_factory import build_factor_panel
    n = 30
    closes = np.linspace(10, 11, n).copy()
    closes[10] = closes[9] * 1.099
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": [1000.0] * n,
    })
    pool_data = {"600000": df}
    out = build_factor_panel(["momentum_5"], pool_data)
    # 第 10 天因子值仍有效(因 momentum_5 需要至少 5 bar 前历史)
    assert pd.notna(out["momentum_5"].iloc[10, 0])


# ─────────────────────────────────────────────────────────────────────────────
# build_panel(pooled top-level)— 仍接 mask_config,只为 forward_return_panel
# ─────────────────────────────────────────────────────────────────────────────

def test_build_panel_no_mask_unchanged():
    from stockpool.ml.dataset import build_panel
    n = 30
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10, 11, n),
        "volume": [1000.0] * n,
    })
    stocks_data = {"600000": df}
    X_a, y_a = build_panel(stocks_data, ["momentum_5"], horizon=2)
    X_b, y_b = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=None)
    pd.testing.assert_frame_equal(X_a, X_b)
    pd.testing.assert_series_equal(y_a, y_b)


def test_build_panel_mask_drops_samples_via_labels():
    """启用 mask 时,通过 forward_return 双向检查在涨停日产生 NaN label,
    被 stack_panel_to_xy 的 dropna 剔除。**因子值本身不变**,只是带 NaN
    label 的行被砍掉。"""
    from stockpool.ml.dataset import build_panel
    from stockpool.config import MaskConfig
    n = 30
    closes = np.linspace(10, 11, n).copy()
    closes[15] = closes[14] * 1.099
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": [1000.0] * n,
    })
    stocks_data = {"600000": df}
    cfg_no = MaskConfig(enabled=False)
    cfg_yes = MaskConfig(enabled=True, min_listing_days=0)
    _, y_no = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=cfg_no)
    _, y_yes = build_panel(stocks_data, ["momentum_5"], horizon=2, mask_config=cfg_yes)
    # 启用 mask 后样本数减少(label NaN 行被丢)
    assert len(y_yes) < len(y_no)


# ─────────────────────────────────────────────────────────────────────────────
# build_factor_matrix(per-stock)— 不再接 mask_config
# ─────────────────────────────────────────────────────────────────────────────

def test_build_factor_matrix_no_mask_param():
    """build_factor_matrix 不接受 mask_config kwarg(2026-05-31 重构移除)。"""
    from stockpool.ml.dataset import build_factor_matrix
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20),
        "open": np.linspace(10, 11, 20),
        "high": np.linspace(10.1, 11.1, 20),
        "low": np.linspace(9.9, 10.9, 20),
        "close": np.linspace(10, 11, 20),
        "volume": [1000.0] * 20,
    })
    out = build_factor_matrix(df, ["momentum_5"])
    assert "momentum_5" in out.columns
    with pytest.raises(TypeError, match="mask_config"):
        build_factor_matrix(df, ["momentum_5"], mask_config=None)


# ─────────────────────────────────────────────────────────────────────────────
# MLFactorStrategy — 仍把 cfg.mask 透传给 build_panel(pooled),
# 但不再向 build_factor_matrix 传 mask_config(per_stock 路径)
# ─────────────────────────────────────────────────────────────────────────────

def test_ml_factor_strategy_mask_changes_sig():
    """翻 cfg.mask.enabled → _strategy_signature 变化 → 旧 ml_models pkl 失效。
    (mask 仍在 cfg 里,因为 build_panel 仍读它做标签双向检查。)"""
    from stockpool.config import MLFactorConfig
    from stockpool.backtesting.strategies import MLFactorStrategy
    cfg_no = MLFactorConfig.model_validate({
        "factors": ["momentum_5"],
        "mask": {"enabled": False},
    })
    cfg_yes = MLFactorConfig.model_validate({
        "factors": ["momentum_5"],
        "mask": {"enabled": True},
    })
    s_no = MLFactorStrategy(cfg_no)
    s_yes = MLFactorStrategy(cfg_yes)
    assert s_no._strategy_signature() != s_yes._strategy_signature()


def test_ml_factor_strategy_pooled_path_uses_mask(monkeypatch):
    """pooled `_try_fit` 内 build_panel 调用带上 mask_config=self.cfg.mask
    (build_panel 用 mask 来做 forward_return 标签的双向检查,不影响因子值)。"""
    from stockpool.config import MLFactorConfig
    from stockpool.backtesting.strategies import MLFactorStrategy
    import stockpool.ml.dataset as ds

    captured = {}
    orig = ds.build_panel
    def spy_build_panel(stocks_data, factor_names, horizon, *, mask_config=None):
        captured["mask_config"] = mask_config
        return orig(stocks_data, factor_names, horizon, mask_config=mask_config)
    monkeypatch.setattr(
        "stockpool.backtesting.strategies.build_panel", spy_build_panel
    )

    cfg = MLFactorConfig.model_validate({
        "factors": ["momentum_5"],
        "panel_mode": "pooled",
        "horizon": 2,
        "train_window": 20,
        "min_train_samples": 5,
        "refit_every": 5,
        "share_pool_fit": False,
        "mask": {"enabled": True, "min_listing_days": 0},
    })
    n = 50
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10, 11, n),
        "volume": [1000.0] * n,
    })
    pool_data = {"600000": df, "600001": df.copy()}
    strat = MLFactorStrategy(cfg, pool_data=pool_data, current_stock_code="600000")
    _ = strat.generate_signals(df)
    assert captured.get("mask_config") is not None
    assert captured["mask_config"].enabled is True


def test_ml_factor_strategy_per_stock_path_does_not_pass_mask(monkeypatch):
    """per_stock 路径 build_factor_matrix 调用 **不再** 带 mask_config —
    2026-05-31 重构后 build_factor_matrix 也不接受这个参数(per-stock 因子
    退化语义下也无需 mask 化输入)。"""
    from stockpool.config import MLFactorConfig
    from stockpool.backtesting.strategies import MLFactorStrategy
    import stockpool.ml.dataset as ds

    captured = {"call_kwargs": None}
    orig = ds.build_factor_matrix
    def spy_build_fm(df, factor_names, **kwargs):
        captured["call_kwargs"] = kwargs
        return orig(df, factor_names, **kwargs)
    monkeypatch.setattr(
        "stockpool.backtesting.strategies.build_factor_matrix", spy_build_fm
    )

    cfg = MLFactorConfig.model_validate({
        "factors": ["momentum_5"],
        "panel_mode": "per_stock",
        "horizon": 2,
        "train_window": 20,
        "min_train_samples": 5,
        "refit_every": 5,
        "mask": {"enabled": True, "min_listing_days": 0},
    })
    n = 50
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low": np.linspace(9.9, 10.9, n),
        "close": np.linspace(10, 11, n),
        "volume": [1000.0] * n,
    })
    strat = MLFactorStrategy(cfg)
    _ = strat.generate_signals(df)
    # 调用确实发生了
    assert captured["call_kwargs"] is not None
    # 但 mask_config 不在 kwargs 里
    assert "mask_config" not in captured["call_kwargs"]
