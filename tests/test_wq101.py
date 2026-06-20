"""WQ101 因子库基础校验。

不验证每个 alpha 的数值正确性(那需要论文级回归),只验证:
  * 全部 101 注册成功 (alpha_001 .. alpha_101)
  * 元数据合规 (source=wq101, 至少 1 个 type)
  * 在合成 panel 上能 compute,且形状与 panel['close'] 一致
  * look-ahead 安全: 截断 panel 算因子,值与全长版本前缀一致
  * cap 缺失的 alpha 是全 NaN(预期行为)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.factors import make_factor, list_specs


def _panel(T: int = 300, N: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=T, freq="B")
    codes = [f"c{i}" for i in range(N)]
    close = pd.DataFrame(
        100 + np.cumsum(rng.standard_normal((T, N)) * 0.5, axis=0),
        index=dates, columns=codes,
    )
    open_ = close.shift(1).fillna(close)
    high = close * (1 + np.abs(rng.standard_normal((T, N))) * 0.01)
    low = close * (1 - np.abs(rng.standard_normal((T, N))) * 0.01)
    vol = pd.DataFrame(rng.uniform(1e5, 1e6, (T, N)), index=dates, columns=codes)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


def _wq_specs():
    return [s for s in list_specs() if "wq101" in s.sources]


# ─────────────────────────────────────────────────────────────────────────────
# 注册 & 元数据
# ─────────────────────────────────────────────────────────────────────────────

def test_all_101_alphas_registered():
    names = sorted(s.default_name for s in _wq_specs())
    expected = [f"alpha_{i:03d}" for i in range(1, 102)]
    assert names == expected


def test_metadata_well_formed():
    for spec in _wq_specs():
        assert "wq101" in spec.sources
        assert len(spec.types) > 0, f"{spec.default_name} has no types"
        assert spec.description, f"{spec.default_name} has empty description"


# ─────────────────────────────────────────────────────────────────────────────
# Compute smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_all_alphas_compute_without_errors():
    panel = _panel(300, 6)
    errs: list[str] = []
    for spec in _wq_specs():
        f = make_factor(spec.default_name)
        try:
            out = f.compute(panel)
            assert out.shape == panel["close"].shape, f"{f.name}: shape {out.shape}"
        except Exception as e:
            errs.append(f"{f.name}: {type(e).__name__}: {e}")
    assert not errs, "errors:\n" + "\n".join(errs)


def test_alpha_056_is_all_nan_due_to_missing_cap():
    panel = _panel(200, 4)
    f = make_factor("alpha_056")
    out = f.compute(panel)
    assert out.isna().all().all()


# ─────────────────────────────────────────────────────────────────────────────
# Look-ahead safety: 截断面板算的因子值,与全长前缀严格相等
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha_name", [
    "alpha_001", "alpha_003", "alpha_006", "alpha_012",
    "alpha_041", "alpha_054", "alpha_101",
])
def test_no_lookahead(alpha_name):
    panel = _panel(200, 5, seed=42)
    cut = 100
    full = make_factor(alpha_name).compute(panel)
    trunc_panel = {k: v.iloc[:cut] for k, v in panel.items()}
    trunc = make_factor(alpha_name).compute(trunc_panel)
    # 同位置应当数值相等(NaN 也匹配 NaN)
    full_head = full.iloc[:cut]
    pd.testing.assert_frame_equal(full_head, trunc, check_exact=False, rtol=1e-9)
