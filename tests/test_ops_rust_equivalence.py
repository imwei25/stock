"""Per-op equivalence tests for the Rust acceleration.

Each test computes the op two ways — once via the Rust path
(stockpool_ops_rs through the ops.py dispatcher) and once via the
pandas oracle (_ops_py) — and asserts allclose(atol=1e-9, rtol=1e-7).
Skipped when the Rust module isn't installed.

Tolerance + contract details live in
docs/superpowers/specs/2026-06-20-rust-ops-acceleration-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpool.factors import _ops_py, ops

_RUST_AVAILABLE = getattr(ops, "_USE_RUST", False)
pytestmark = pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="stockpool_ops_rs not importable (build it with `maturin develop --release`)",
)


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.RangeIndex(values.shape[0], name="row"),
        columns=[f"c{i:02d}" for i in range(values.shape[1])],
    )


def _assert_equiv(rust, py) -> None:
    np.testing.assert_allclose(
        rust.values, py.values, atol=1e-9, rtol=1e-7, equal_nan=True,
    )
    # Wrapper must preserve index / column identity (no copies)
    assert rust.index.equals(py.index)
    assert list(rust.columns) == list(py.columns)


# ─────────────────────────────────────────────────────────────────────────────
# rank
# ─────────────────────────────────────────────────────────────────────────────

class TestRank:
    def test_random_no_nan(self):
        rng = np.random.default_rng(42)
        df = _frame(rng.standard_normal((30, 20)))
        _assert_equiv(ops.rank(df), _ops_py.rank(df))

    def test_scattered_nan_5pct(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((30, 20))
        mask = rng.random(x.shape) < 0.05
        x[mask] = np.nan
        df = _frame(x)
        _assert_equiv(ops.rank(df), _ops_py.rank(df))

    def test_full_nan_column(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((20, 10))
        x[:, 3] = np.nan
        df = _frame(x)
        _assert_equiv(ops.rank(df), _ops_py.rank(df))

    def test_full_nan_row(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((20, 10))
        x[7, :] = np.nan
        df = _frame(x)
        _assert_equiv(ops.rank(df), _ops_py.rank(df))

    def test_ties(self):
        # Hand-traceable: ties in row 0 must get average rank.
        x = np.array(
            [
                [1.0, 2.0, 2.0, 3.0, 3.0],  # ranks 1, 2.5, 2.5, 4.5, 4.5 → pcts 0.2, 0.5, 0.5, 0.9, 0.9
                [5.0, 5.0, 5.0, 5.0, 5.0],  # all tied → avg rank 3 → pct 0.6 each
            ]
        )
        df = _frame(x)
        rust = ops.rank(df)
        py = _ops_py.rank(df)
        _assert_equiv(rust, py)
        # Spot-check exact values
        np.testing.assert_allclose(
            rust.iloc[0].values, [0.2, 0.5, 0.5, 0.9, 0.9], atol=1e-12,
        )
        np.testing.assert_allclose(
            rust.iloc[1].values, [0.6, 0.6, 0.6, 0.6, 0.6], atol=1e-12,
        )

    @pytest.mark.parametrize("ncols", [1, 2])
    def test_narrow_frames(self, ncols):
        rng = np.random.default_rng(4)
        df = _frame(rng.standard_normal((10, ncols)))
        _assert_equiv(ops.rank(df), _ops_py.rank(df))
