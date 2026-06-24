"""Tests for ab.score_ic — cross-sectional rank-IC of strategy score in A/B.

These tests exercise the close-basis path (label_basis defaults to "open" but
falls back to close-to-close when no ``open_`` panel is supplied). open-basis
math is covered in test_label_basis.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpool.ab.score_ic import (
    arm_score_ic,
    cross_sectional_score_ic,
    panels_from_per_stock,
)
from stockpool.ml.dataset import forward_return_panel


@dataclass
class _FakeResult:
    score_frame: pd.DataFrame | None


def _panel(values, dates, codes):
    return pd.DataFrame(values, index=pd.to_datetime(dates), columns=codes)


def test_perfect_score_gives_high_positive_ic():
    """Score equal to the realized forward return → cross-sectional IC ≈ +1."""
    dates = pd.bdate_range("2024-01-01", periods=60)
    codes = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(0)
    close = _panel(rng.uniform(10, 20, (len(dates), len(codes))), dates, codes)
    h = 3
    score = forward_return_panel(close, h, "return")  # score == realized fwd ret
    res = cross_sectional_score_ic(score, close, h)
    assert res["mean_ic"] is not None and res["mean_ic"] > 0.99
    assert res["n_days"] > 0 and res["n_stocks"] == 5


def test_negated_score_gives_negative_ic():
    dates = pd.bdate_range("2024-01-01", periods=60)
    codes = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(1)
    close = _panel(rng.uniform(10, 20, (len(dates), len(codes))), dates, codes)
    h = 3
    score = -forward_return_panel(close, h, "return")
    res = cross_sectional_score_ic(score, close, h)
    assert res["mean_ic"] < -0.99


def test_random_score_gives_near_zero_ic():
    dates = pd.bdate_range("2024-01-01", periods=150)
    codes = [f"S{i}" for i in range(8)]
    rng = np.random.default_rng(2)
    close = _panel(rng.uniform(10, 20, (len(dates), len(codes))), dates, codes)
    score = _panel(rng.normal(size=(len(dates), len(codes))), dates, codes)
    res = cross_sectional_score_ic(score, close, 3)
    assert abs(res["mean_ic"]) < 0.2


def test_empty_score_returns_none():
    res = cross_sectional_score_ic(pd.DataFrame(), pd.DataFrame(), 3)
    assert res["mean_ic"] is None and res["n_days"] == 0


def _score_frame(dates, scores, closes):
    return pd.DataFrame({
        "date": dates, "open": closes, "close": closes, "final_score": scores,
    })


def test_panels_and_arm_score_ic_end_to_end():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(3)
    per_stock = []
    for code in ["A", "B", "C", "D"]:
        c = rng.uniform(10, 20, len(dates))
        per_stock.append((code, code, _FakeResult(_score_frame(dates, rng.normal(size=len(dates)), c))))
    score, open_, close = panels_from_per_stock(per_stock)
    assert score.shape == (40, 4) and close.shape == (40, 4)
    assert open_.shape == (40, 4)
    out = arm_score_ic(per_stock, [3, 5])
    assert set(out) == {3, 5}
    assert out[3]["n_stocks"] == 4


def test_arm_score_ic_empty_when_no_score_frames():
    per_stock = [("A", "A", _FakeResult(None))]
    out = arm_score_ic(per_stock, [3])
    assert out[3]["mean_ic"] is None
