"""B4:factors analyze 与生产口径对齐(P2-4)+ selection 窗口(P0-6)+ NW 修正(P2-2)。"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import stockpool.factors_analysis as fa


def _panel(n=60, codes=("000001", "000002", "000003", "000004", "000005",
                        "000006", "000007", "000008")):
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2025-01-06", periods=n)
    out = {}
    base = {c: 10 + np.cumsum(rng.normal(0, 0.1, n)) for c in codes}
    for field in ("open", "high", "low", "close"):
        out[field] = pd.DataFrame(
            {c: base[c] + (0.05 if field == "high" else 0) for c in codes},
            index=dates)
    out["volume"] = pd.DataFrame(1e6, index=dates, columns=list(codes))
    return out


def test_end_date_cuts_analysis_window():
    """P0-6:selection 必须能截止在回测起点之前。"""
    panel = _panel(60)
    cutoff = panel["close"].index[39]
    res = fa.analyze_factors(panel, ["momentum_5"], horizon=2, end_date=cutoff)
    assert res.end_date <= cutoff
    for s in res.daily_ic.values():
        assert s.index.max() <= cutoff


def test_labels_use_open_basis_by_default():
    """口径一致:analyze 的标签默认 open-to-open(与训练/执行一致)。"""
    panel = _panel(40)
    captured = {}
    real = fa.forward_return_panel

    def spy(close, horizon, *args, **kwargs):
        captured.update(kwargs)
        return real(close, horizon, *args, **kwargs)

    with patch.object(fa, "forward_return_panel", side_effect=spy):
        fa.analyze_factors(panel, ["momentum_5"], horizon=2)
    assert captured.get("open_") is not None, "默认应使用 open 基准标签"


def test_mask_passed_to_labels():
    panel = _panel(40)
    mask = pd.DataFrame(True, index=panel["close"].index,
                        columns=panel["close"].columns)
    captured = {}
    real = fa.forward_return_panel

    def spy(close, horizon, *args, **kwargs):
        captured.update(kwargs)
        return real(close, horizon, *args, **kwargs)

    with patch.object(fa, "forward_return_panel", side_effect=spy):
        fa.analyze_factors(panel, ["momentum_5"], horizon=2, mask=mask)
    assert captured.get("mask") is not None


def test_preprocess_cfg_applied():
    """P2-4:传 preprocess_cfg 时,IC 在预处理后的因子上算(与生产一致)。"""
    from stockpool.config import PreprocessConfig
    panel = _panel(40)
    called = {}

    def fake_preproc(raw, factor_names, preprocess_cfg, n_codes):
        called["yes"] = True
        return raw

    with patch("stockpool.strategy_factory.apply_production_preprocess",
               side_effect=fake_preproc):
        fa.analyze_factors(
            panel, ["momentum_5"], horizon=2,
            preprocess_cfg=PreprocessConfig(),
        )
    assert called.get("yes")


def test_newey_west_std_inflates_for_overlapping_ic():
    """P2-2:正自相关(重叠标签)序列的 NW σ 应大于朴素 σ;白噪声两者接近。"""
    rng = np.random.default_rng(11)
    # AR(1) 正自相关
    n = 500
    e = rng.normal(0, 1, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    s_ar = pd.Series(ar)
    s_wn = pd.Series(rng.normal(0, 1, n))

    nw_ar = fa._newey_west_std(s_ar, lag=4)
    naive_ar = float(s_ar.std(ddof=0))
    assert nw_ar > naive_ar * 1.3

    nw_wn = fa._newey_west_std(s_wn, lag=4)
    naive_wn = float(s_wn.std(ddof=0))
    assert nw_wn == pytest.approx(naive_wn, rel=0.25)


def test_ic_ir_uses_nw_std():
    """ic_ir 的分母应做 lag=(label_lag−1) 的 NW 修正(重叠日 IC 独立性虚高)。"""
    panel = _panel(60)
    res = fa.analyze_factors(panel, ["momentum_5"], horizon=5)
    name = res.factor_names[0]
    raw_ir = res.mean_ic[name] / res.daily_ic[name].std(ddof=0)
    if np.isfinite(raw_ir) and abs(raw_ir) > 1e-9:
        assert abs(res.ic_ir[name]) <= abs(raw_ir) + 1e-12, (
            "NW 修正后的 |ic_ir| 不应大于朴素值(正自相关情形)"
        )
