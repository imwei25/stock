"""Snapshot equivalence test.

PR-1: re-runs every registered factor on the OHLCV fixture and diffs
the output against the snapshot fixture. Because PR-1 has no Rust path,
this is pandas-vs-pandas self-validation — its purpose is to prove the
fixture round-trips and to be the test that breaks (in a controlled way)
when PR-2+ swaps in Rust ops that diverge.

Tolerance: atol=1e-9, rtol=1e-7 (per design spec
docs/superpowers/specs/2026-06-20-rust-ops-acceleration-design.md).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpool.factors import list_factors, make_factor
from stockpool.factors.context import set_sector_map
from stockpool.industry_map import load_or_build_industry_map

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PANEL_PATH = FIXTURES / "ops_panel.parquet"
SNAPSHOT_PATH = FIXTURES / "ops_snapshot.parquet"


def _load_panel() -> dict[str, pd.DataFrame]:
    """Reconstruct a panel dict from the long-form fixture parquet."""
    long = pd.read_parquet(PANEL_PATH)
    long["date"] = pd.to_datetime(long["date"])
    panel: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        wide = long.pivot(index="date", columns="code", values=field).sort_index()
        wide.index.name = "date"
        wide.columns.name = "code"
        panel[field] = wide
    return panel


@pytest.fixture(scope="module")
def panel() -> Iterator[dict[str, pd.DataFrame]]:
    if not PANEL_PATH.exists():
        pytest.skip(f"{PANEL_PATH} missing — run scripts/gen_ops_snapshot.py")
    # indneutralize-using factors need sector_map; load it once.
    sector_map = load_or_build_industry_map(ROOT / "data", source="auto")
    set_sector_map(sector_map or {})
    yield _load_panel()
    # Reset global sector_map so this test doesn't pollute later tests.
    set_sector_map({})


@pytest.fixture(scope="module")
def expected() -> dict[str, pd.Series]:
    """{factor_name: Series indexed by (date, code) → value}"""
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"{SNAPSHOT_PATH} missing — run scripts/gen_ops_snapshot.py")
    long = pd.read_parquet(SNAPSHOT_PATH)
    long["date"] = pd.to_datetime(long["date"])
    long = long.set_index(["factor", "date", "code"])["value"]
    return {
        name: long.loc[name]
        for name in long.index.get_level_values(0).unique()
    }


@pytest.mark.parametrize("factor_name", list_factors())
def test_factor_matches_snapshot(
    factor_name: str,
    panel: dict[str, pd.DataFrame],
    expected: dict[str, pd.Series],
) -> None:
    if factor_name not in expected:
        pytest.skip(f"factor {factor_name} not in snapshot — regenerate fixture")
    f = make_factor(factor_name)
    wide = f.compute(panel)
    actual_long = wide.stack(dropna=False)
    exp = expected[factor_name]
    # Align actual to the snapshot's (date, code) index ordering
    actual = actual_long.reorder_levels(["date", "code"]).reindex(exp.index)
    np.testing.assert_allclose(
        actual.values, exp.values,
        atol=1e-9, rtol=1e-7, equal_nan=True,
        err_msg=f"factor {factor_name} diverged from snapshot",
    )
