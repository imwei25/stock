"""Tests for portfolio.scoring.precompute_scores_from_legacy."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from stockpool.portfolio.scoring import precompute_scores_from_legacy


class _StubLegacy:
    """Returns a signal frame with date + final_score columns per code."""
    def __init__(self, score_fn):
        self._score_fn = score_fn
        self.calls: list[int] = []   # daily_df row counts seen

    def generate_signals(self, daily):
        self.calls.append(len(daily))
        scores = [self._score_fn(d, daily) for d in daily["date"].tolist()]
        return pd.DataFrame({
            "date": daily["date"].tolist(),
            "close": daily["close"].tolist(),
            "final_score": scores,
            "signal": ["neutral"] * len(daily),
        })


class _ExplodingLegacy:
    def generate_signals(self, daily):
        raise RuntimeError("boom")


class _NoScoreLegacy:
    def generate_signals(self, daily):
        return pd.DataFrame({
            "date": daily["date"], "close": daily["close"],
            "signal": ["neutral"] * len(daily),
        })


def _mk_daily(dates, base=10.0):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": [base + i for i in range(len(dates))]})


def test_happy_path_builds_panel():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    panel_data = {
        "A": _mk_daily(dates),
        "B": _mk_daily(dates),
    }
    legacy = _StubLegacy(score_fn=lambda d, df: float(d.day))
    panel = precompute_scores_from_legacy(legacy, panel_data)
    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == 3
    assert panel.loc[pd.Timestamp("2024-01-03"), "A"] == 3.0


def test_failure_isolated_per_stock(caplog):
    dates = ["2024-01-02", "2024-01-03"]
    legacy_ok = _StubLegacy(score_fn=lambda d, df: 0.5)
    panel_data = {"A": _mk_daily(dates), "B": _mk_daily(dates)}

    class _Mixed:
        def generate_signals(self, daily):
            if daily["close"].iloc[0] == 99.0:
                raise RuntimeError("boom")
            return legacy_ok.generate_signals(daily)

    # B will explode, A succeeds
    panel_data = {"A": _mk_daily(dates), "B": _mk_daily(dates, base=99.0)}
    with caplog.at_level(logging.WARNING, logger="stockpool"):
        panel = precompute_scores_from_legacy(_Mixed(), panel_data)
    assert list(panel.columns) == ["A"]
    assert any("generate_signals failed" in r.message for r in caplog.records)


def test_all_fail_returns_empty():
    panel_data = {"A": _mk_daily(["2024-01-02"]), "B": _mk_daily(["2024-01-02"])}
    panel = precompute_scores_from_legacy(_ExplodingLegacy(), panel_data)
    assert panel.empty


def test_missing_score_field_skipped(caplog):
    panel_data = {"A": _mk_daily(["2024-01-02"])}
    with caplog.at_level(logging.WARNING, logger="stockpool"):
        panel = precompute_scores_from_legacy(_NoScoreLegacy(), panel_data)
    assert panel.empty
    assert any("missing 'final_score'" in r.message for r in caplog.records)


def test_passes_full_daily_history_to_legacy():
    """Helper itself doesn't truncate; look-ahead lives in legacy.generate_signals."""
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    legacy = _StubLegacy(score_fn=lambda d, df: 1.0)
    panel_data = {"A": _mk_daily(dates)}
    precompute_scores_from_legacy(legacy, panel_data)
    # Helper called legacy once with the whole daily frame (legacy is
    # responsible for walk-forward).
    assert legacy.calls == [4]
