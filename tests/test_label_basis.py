"""P2-2/审查 P2-3:训练标签从 close→close 改为 open[t+1]→open[t+1+h]。

与 T+1 次日开盘成交的执行口径对齐:决策 bar t 的标签 = 实际可实现的
开盘到开盘收益,不再包含拿不到的 close[t]→open[t+1] 隔夜段。
"""
import numpy as np
import pandas as pd
import pytest

from stockpool.ml.dataset import forward_return, forward_return_panel


def _wide(values_by_code: dict, start="2026-01-05") -> pd.DataFrame:
    n = len(next(iter(values_by_code.values())))
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame(values_by_code, index=idx)


# ---------------------------------------------------------------------------
# forward_return_panel: open 基准
# ---------------------------------------------------------------------------

def test_panel_open_basis_math():
    """h=2:y[t] = open[t+3] / open[t+1] − 1(在 open[t+1] 买、open[t+1+h] 卖)。"""
    open_ = _wide({"A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]})
    close = open_ + 0.5
    y = forward_return_panel(close, horizon=2, open_=open_)
    # t=0: open[3]/open[1] - 1 = 13/11 - 1
    assert y["A"].iloc[0] == pytest.approx(13.0 / 11.0 - 1.0)
    # t=1: open[4]/open[2] - 1 = 14/12 - 1
    assert y["A"].iloc[1] == pytest.approx(14.0 / 12.0 - 1.0)
    # 末尾 h+1 行无未来 open → NaN
    assert y["A"].iloc[-3:].isna().all()


def test_panel_open_basis_excludes_overnight_gap():
    """close 基准会把 close[t]→open[t+1] 隔夜跳空计入标签;open 基准不含。
    构造:close 恒定 10,open 恒定 20(极端隔夜跳空)→ open 基准收益恒 0。"""
    close = _wide({"A": [10.0] * 6})
    open_ = _wide({"A": [20.0] * 6})
    y_open = forward_return_panel(close, horizon=2, open_=open_)
    y_close = forward_return_panel(close, horizon=2)
    assert y_open["A"].iloc[0] == pytest.approx(0.0)
    assert y_close["A"].iloc[0] == pytest.approx(0.0)  # close 恒定也为 0
    # open 基准的值由 open 决定,与 close 无关
    open2 = _wide({"A": [20.0, 20.0, 22.0, 20.0, 20.0, 20.0]})
    y2 = forward_return_panel(close, horizon=2, open_=open2)
    assert y2["A"].iloc[0] == pytest.approx(20.0 / 20.0 - 1.0)  # open[3]/open[1]
    assert y2["A"].iloc[1] == pytest.approx(20.0 / 22.0 - 1.0)  # open[4]/open[2]


def test_panel_open_basis_mask_checks_entry_and_exit_bars():
    """mask 双向检查应作用于实际进出场 bar(t+1 与 t+1+h),而非 t 与 t+h。"""
    open_ = _wide({"A": [10.0] * 8})
    close = open_.copy()
    mask = pd.DataFrame(True, index=open_.index, columns=open_.columns)

    # h=2,t=0 的进场 bar 是 t+1=1,出场 bar 是 t+3=3
    mask.iloc[1, 0] = False  # 进场 bar 不可交易
    y = forward_return_panel(close, horizon=2, mask=mask, open_=open_)
    assert np.isnan(y["A"].iloc[0]), "进场 bar(t+1)不可交易时标签应为 NaN"
    # t=1 的进出场 bar 是 2 和 4,均可交易 → 有值
    assert not np.isnan(y["A"].iloc[1])

    mask2 = pd.DataFrame(True, index=open_.index, columns=open_.columns)
    mask2.iloc[3, 0] = False  # t=0 的出场 bar
    y2 = forward_return_panel(close, horizon=2, mask=mask2, open_=open_)
    assert np.isnan(y2["A"].iloc[0]), "出场 bar(t+1+h)不可交易时标签应为 NaN"


def test_panel_close_basis_unchanged():
    """不传 open_ 时保持旧语义(close[t+h]/close[t]−1,mask 查 t 与 t+h)。"""
    close = _wide({"A": [10.0, 11.0, 12.0, 13.0, 14.0]})
    y = forward_return_panel(close, horizon=2)
    assert y["A"].iloc[0] == pytest.approx(12.0 / 10.0 - 1.0)


# ---------------------------------------------------------------------------
# forward_return(单股): basis 参数
# ---------------------------------------------------------------------------

def _daily(opens, closes, start="2026-01-05") -> pd.DataFrame:
    n = len(opens)
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": opens, "high": [x + 0.2 for x in closes],
        "low": [x - 0.2 for x in closes], "close": closes,
        "volume": [1000.0] * n,
    })


def test_forward_return_open_basis():
    df = _daily([10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [10.5, 11.5, 12.5, 13.5, 14.5, 15.5])
    y = forward_return(df, horizon=2, basis="open")
    assert y.iloc[0] == pytest.approx(13.0 / 11.0 - 1.0)
    assert y.iloc[-3:].isna().all()


def test_forward_return_close_basis_default():
    df = _daily([10.0, 11.0, 12.0, 13.0], [10.0, 11.0, 12.0, 13.0])
    y = forward_return(df, horizon=2)
    assert y.iloc[0] == pytest.approx(12.0 / 10.0 - 1.0)


# ---------------------------------------------------------------------------
# 配置与策略接线
# ---------------------------------------------------------------------------

def test_mlfactor_config_default_label_basis_is_open():
    from stockpool.config import MLFactorConfig
    cfg = MLFactorConfig()
    assert cfg.label_basis == "open"


def test_embargo_accounts_for_extra_open_lag():
    """open 基准标签多看 1 根 bar(open[t+1+h]),embargo 数学必须同步 +1。"""
    from stockpool.backtesting.strategies import MLFactorStrategy
    from stockpool.config import MLFactorConfig

    cfg_open = MLFactorConfig(horizon=5, embargo_days=0, label_basis="open")
    cfg_close = MLFactorConfig(horizon=5, embargo_days=0, label_basis="close")
    s_open = MLFactorStrategy(cfg_open)
    s_close = MLFactorStrategy(cfg_close)
    assert s_close._embargoed_label_end(100) == 100 - 5
    assert s_open._embargoed_label_end(100) == 100 - 5 - 1, (
        "open 基准下标签终点应再往回退 1 根 bar"
    )


def test_build_panel_open_basis_labels():
    from stockpool.ml.dataset import build_panel
    n = 16  # 因子 warmup(5)+ 标签前瞻(h+1=3)都要留足
    opens = [10.0 + i for i in range(n)]
    closes = [x + 0.3 for x in opens]
    pool = {
        "AAA": _daily(opens, closes),
        "BBB": _daily([x * 2 for x in opens], [x * 2 for x in closes]),
    }
    X, y = build_panel(pool, ["momentum_5"], horizon=2, label_basis="open")
    # 取 AAA 第一个有标签的样本核对 open-to-open 数学
    sub = y.xs("AAA", level="stock")
    valid = sub.dropna()
    assert len(valid) > 0
    t = valid.index[0]
    i = list(pd.bdate_range("2026-01-05", periods=n)).index(t)
    expected = opens[i + 3] / opens[i + 1] - 1.0
    assert valid.iloc[0] == pytest.approx(expected)
