"""Tests for stockpool.panel mask functions (tradability mask for factor input)."""
import numpy as np
import pandas as pd
import pytest


def test_limit_threshold_main_board():
    from stockpool.panel import _limit_threshold
    assert _limit_threshold("600000") == 0.098
    assert _limit_threshold("601398") == 0.098
    assert _limit_threshold("603986") == 0.098
    assert _limit_threshold("605589") == 0.098
    assert _limit_threshold("000001") == 0.098
    assert _limit_threshold("002001") == 0.098
    assert _limit_threshold("003001") == 0.098


def test_limit_threshold_chinext_star():
    from stockpool.panel import _limit_threshold
    assert _limit_threshold("300001") == 0.198
    assert _limit_threshold("301001") == 0.198
    assert _limit_threshold("688001") == 0.198


def test_limit_threshold_bse():
    from stockpool.panel import _limit_threshold
    assert _limit_threshold("830001") == 0.298
    assert _limit_threshold("870001") == 0.298
    assert _limit_threshold("820001") == 0.298
    assert _limit_threshold("430001") == 0.298


def test_listing_mask_mature_stock_all_true():
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=300)
    close = pd.DataFrame({"600000": np.arange(300, dtype=float)}, index=idx)
    mask = _listing_mask(close, min_days=252)
    assert mask["600000"].all()


def test_listing_mask_new_listing_blocks_first_n_days():
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=400)
    close = pd.DataFrame({
        "300001": [np.nan] * 50 + list(range(350)),
    }, index=idx)
    mask = _listing_mask(close, min_days=252)
    assert not mask["300001"].iloc[50:50+252].any()
    assert mask["300001"].iloc[50+252:].all()


def test_listing_mask_all_nan_stock_all_false():
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=100)
    close = pd.DataFrame({"600000": [np.nan] * 100}, index=idx)
    mask = _listing_mask(close, min_days=252)
    assert not mask["600000"].any()


# ────────────────────────────────────────────────────────────────────────────
# _listing_mask with explicit ipo_dates (preferred path; fixes the
# first_valid_index heuristic bug where mature stocks with shorter
# cache windows got mis-flagged as newly listed).
# ────────────────────────────────────────────────────────────────────────────


def test_listing_mask_ipo_dates_mature_stock_short_cache():
    """成熟股(IPO 远早于 panel 起点)即使前几行 NaN(panel union 早于
    该股缓存起点),有 ipo_dates 时仍全部 True。这正是修复 first_valid_index
    启发式 bug 的关键场景。"""
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-05-08", periods=500)
    close = pd.DataFrame({
        # 模拟"成熟股,缓存历史不齐":前 14 行 NaN
        "600584": [np.nan] * 14 + list(np.linspace(10, 12, 486)),
    }, index=idx)
    ipo_dates = {"600584": pd.Timestamp("2003-08-26")}  # 长电科技实际 IPO
    mask = _listing_mask(close, min_days=252, ipo_dates=ipo_dates)
    assert mask["600584"].all()


def test_listing_mask_ipo_dates_recent_ipo_within_panel():
    """IPO 落在 panel 范围内 → 屏蔽 IPO 后约 366 自然日(≈252 交易日)以内。"""
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=500)
    close = pd.DataFrame({"301308": list(np.linspace(10, 12, 500))}, index=idx)
    ipo_dates = {"301308": pd.Timestamp("2024-03-15")}
    mask = _listing_mask(close, min_days=252, ipo_dates=ipo_dates)
    cutoff = pd.Timestamp("2024-03-15") + pd.Timedelta(days=int(252 * 1.45))
    # cutoff 之前应该 mask=False
    assert not mask["301308"].loc[idx < cutoff].any()
    # cutoff 及之后 mask=True
    assert mask["301308"].loc[idx >= cutoff].all()


def test_listing_mask_ipo_dates_missing_code_assumes_mature():
    """ipo_dates 里没有的 code → 不 mask(保守假设成熟股)。"""
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-01-01", periods=100)
    close = pd.DataFrame({
        "600000": list(np.linspace(10, 12, 100)),
        "999999": list(np.linspace(10, 12, 100)),
    }, index=idx)
    ipo_dates = {"600000": pd.Timestamp("2000-01-01")}
    mask = _listing_mask(close, min_days=252, ipo_dates=ipo_dates)
    assert mask["600000"].all()
    assert mask["999999"].all()


def test_listing_mask_ipo_dates_overrides_first_valid_index():
    """直接对比启发式 vs ipo_dates:前者错误 mask 14+252 行,后者正确返回全 True。"""
    from stockpool.panel import _listing_mask
    idx = pd.date_range("2024-05-08", periods=500)
    close = pd.DataFrame({
        "600584": [np.nan] * 14 + list(np.linspace(10, 12, 486)),
    }, index=idx)
    # 启发式(无 ipo_dates):mask=False 应该有约 266 行
    heuristic_mask = _listing_mask(close, min_days=252, ipo_dates=None)
    n_false_heuristic = (~heuristic_mask["600584"]).sum()
    assert n_false_heuristic > 200  # 大量被错误 mask

    # 用真实 IPO 日期:0 个 mask=False
    real_mask = _listing_mask(
        close, min_days=252,
        ipo_dates={"600584": pd.Timestamp("2003-08-26")},
    )
    assert (~real_mask["600584"]).sum() == 0


def _make_panel(close_dict, volume_dict=None):
    codes = list(close_dict.keys())
    idx = pd.date_range("2024-01-01", periods=len(next(iter(close_dict.values()))))
    close = pd.DataFrame(close_dict, index=idx)
    if volume_dict is None:
        volume = pd.DataFrame({c: [1000.0] * len(idx) for c in codes}, index=idx)
    else:
        volume = pd.DataFrame(volume_dict, index=idx)
    return {
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close,
        "volume": volume,
    }


def test_compute_mask_main_board_limit_up():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {
        "600000": [10.0, 10.99, 11.0, 11.01],
        "300001": [10.0, 10.99, 11.0, 11.01],
    }
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.loc[panel["close"].index[1], "600000"] == False
    assert mask.loc[panel["close"].index[1], "300001"] == True


def test_compute_mask_suspension_volume_zero():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {"600000": [10.0, 10.05, 10.1, 10.15]}
    volume_dict = {"600000": [1000.0, 0.0, 1000.0, 1000.0]}
    panel = _make_panel(close_dict, volume_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.loc[panel["close"].index[1], "600000"] == False


def test_compute_mask_three_conditions_intersect():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {"600000": [10.0, 10.05, 10.10, 10.15]}
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.iloc[0, 0] == False
    assert mask.iloc[1:, 0].all()


def test_compute_mask_shape_matches_close():
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    close_dict = {f"600{i:03d}": [10.0 + i * 0.01] * 50 for i in range(5)}
    panel = _make_panel(close_dict)
    cfg = MaskConfig(enabled=True, min_listing_days=0)
    mask = compute_tradability_mask(panel, cfg)
    assert mask.shape == panel["close"].shape
    assert mask.index.equals(panel["close"].index)
    assert mask.columns.equals(panel["close"].columns)


def test_compute_tradability_mask_accepts_ipo_dates_kwarg():
    """compute_tradability_mask 接 ipo_dates → 透传给 _listing_mask。
    对成熟股,带 ipo_dates 修复"缓存短"的误判。"""
    from stockpool.panel import compute_tradability_mask
    from stockpool.config import MaskConfig
    idx = pd.date_range("2024-05-08", periods=500)
    close_values = [np.nan] * 14 + list(np.linspace(10, 12, 486))
    close_df = pd.DataFrame({"600584": close_values}, index=idx)
    panel = {
        "open": close_df.copy(),
        "high": close_df.copy(),
        "low": close_df.copy(),
        "close": close_df,
        "volume": pd.DataFrame({"600584": [1000.0] * 500}, index=idx),
    }
    cfg = MaskConfig(enabled=True, min_listing_days=252)

    # No ipo_dates: heuristic mis-flags ~266 rows
    mask_heuristic = compute_tradability_mask(panel, cfg)
    n_false_heuristic = (~mask_heuristic["600584"]).sum()
    assert n_false_heuristic > 200

    # With real ipo_dates: only NaN-induced mask=False (rows 0-13 from
    # NaN close + row 14 from NaN ret 第一日)
    mask_real = compute_tradability_mask(
        panel, cfg,
        ipo_dates={"600584": pd.Timestamp("2003-08-26")},
    )
    n_false_real = (~mask_real["600584"]).sum()
    assert n_false_real < 20  # 大幅减少


def test_apply_mask_nulls_correct_positions():
    from stockpool.panel import apply_mask
    idx = pd.date_range("2024-01-01", periods=4)
    panel = {
        "close": pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=idx),
        "open": pd.DataFrame({"A": [10.1, 11.1, 12.1, 13.1]}, index=idx),
        "high": pd.DataFrame({"A": [10.5, 11.5, 12.5, 13.5]}, index=idx),
        "low": pd.DataFrame({"A": [9.5, 10.5, 11.5, 12.5]}, index=idx),
        "volume": pd.DataFrame({"A": [100.0, 200.0, 300.0, 400.0]}, index=idx),
    }
    mask = pd.DataFrame({"A": [True, False, True, False]}, index=idx)
    out = apply_mask(panel, mask)
    for field in ("open", "high", "low", "close", "volume"):
        assert np.isnan(out[field].iloc[1, 0])
        assert np.isnan(out[field].iloc[3, 0])
        assert out[field].iloc[0, 0] == panel[field].iloc[0, 0]
        assert out[field].iloc[2, 0] == panel[field].iloc[2, 0]


def test_apply_mask_does_not_mutate_input():
    from stockpool.panel import apply_mask
    idx = pd.date_range("2024-01-01", periods=3)
    panel = {
        "close": pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=idx),
        "volume": pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=idx),
    }
    mask = pd.DataFrame({"A": [True, False, True]}, index=idx)
    _ = apply_mask(panel, mask)
    assert panel["close"].iloc[1, 0] == 11.0
    assert panel["volume"].iloc[1, 0] == 200.0
