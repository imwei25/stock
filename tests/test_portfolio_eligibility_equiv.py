"""Lock the precomputed EligibilityFilter to the original per-bar reference logic.

The optimization replaced an O(rebalances x N x T) per-bar re-parse/re-aggregate
with a one-shot per-panel precompute + searchsorted. This test asserts the new
implementation returns *identical* eligible sets to a verbatim copy of the old
logic across many dates and several eligibility configs (incl. the liquidity
boundary, ST exclusion, missing-volume, and date-truncation edge cases).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.config import PortfolioEligibilityConfig
from stockpool.portfolio.eligibility import EligibilityFilter, _is_st


def _reference_eligible(cfg, name_map, date_t, panel_data):
    """Verbatim pre-optimization eligible() logic."""
    date_t = pd.Timestamp(date_t)
    out = set()
    for code, daily in panel_data.items():
        if cfg.exclude_st and _is_st(name_map.get(code, "")):
            continue
        if "date" not in daily.columns or "close" not in daily.columns:
            continue
        df = daily[pd.to_datetime(daily["date"]) <= date_t]
        if len(df) < cfg.min_history_bars:
            continue
        if cfg.min_avg_amount_20d > 0:
            if "volume" not in df.columns:
                continue
            recent = df.tail(20)
            if len(recent) == 0:
                continue
            avg_amount = float(
                (recent["close"].astype(float) * recent["volume"].astype(float) * 100.0).mean()
            )
            if pd.isna(avg_amount) or avg_amount < cfg.min_avg_amount_20d:
                continue
        out.add(code)
    return out


def _mk_panel(seed: int, n_codes: int = 25, n_bars: int = 400):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_bars, freq="B")
    panel = {}
    name_map = {}
    for c in range(n_codes):
        code = f"{c:06d}"
        # Varied histories: some short, some with volume regime shifts near the
        # liquidity threshold to exercise boundary flips.
        start = rng.integers(0, 80)
        d = dates[start:]
        close = 10 + np.abs(rng.standard_normal(len(d)).cumsum() * 0.1)
        vol = rng.uniform(40_000, 60_000, len(d))  # straddles 5e7 amount
        panel[code] = pd.DataFrame({"date": d, "close": close, "volume": vol})
        name_map[code] = "*ST x" if c % 7 == 0 else "正常"
    return panel, name_map


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("cfg", [
    PortfolioEligibilityConfig(min_avg_amount_20d=5e7, exclude_st=True, min_history_bars=60),
    PortfolioEligibilityConfig(min_avg_amount_20d=0, exclude_st=True, min_history_bars=60),
    PortfolioEligibilityConfig(min_avg_amount_20d=5e7, exclude_st=False, min_history_bars=20),
])
def test_fast_eligibility_matches_reference(seed, cfg):
    panel, name_map = _mk_panel(seed)
    all_dates = sorted(set(pd.concat([d["date"] for d in panel.values()])))
    test_dates = all_dates[40::11]
    f = EligibilityFilter(cfg, name_map=name_map)
    for dt in test_dates:
        ref = _reference_eligible(cfg, name_map, dt, panel)
        got = f.eligible(dt, panel)
        assert got == ref, f"mismatch at {dt}: only_ref={ref - got} only_new={got - ref}"


def test_missing_volume_and_close_columns():
    cfg = PortfolioEligibilityConfig(min_avg_amount_20d=5e7, exclude_st=False, min_history_bars=1)
    panel = {
        "novol": pd.DataFrame({"date": pd.date_range("2024-01-02", periods=30, freq="B"),
                               "close": [10.0] * 30}),
        "noclose": pd.DataFrame({"date": pd.date_range("2024-01-02", periods=30, freq="B"),
                                 "volume": [100_000.0] * 30}),
        "ok": pd.DataFrame({"date": pd.date_range("2024-01-02", periods=30, freq="B"),
                            "close": [10.0] * 30, "volume": [60_000.0] * 30}),
    }
    f = EligibilityFilter(cfg)
    got = f.eligible(pd.Timestamp("2024-12-31"), panel)
    ref = _reference_eligible(cfg, {}, pd.Timestamp("2024-12-31"), panel)
    assert got == ref == {"ok"}
