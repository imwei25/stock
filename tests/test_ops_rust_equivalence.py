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


# ─────────────────────────────────────────────────────────────────────────────
# ts_std
# ─────────────────────────────────────────────────────────────────────────────

class TestTsStd:
    @pytest.mark.parametrize("d", [2, 5, 20, 60])
    def test_random_no_nan(self, d):
        rng = np.random.default_rng(100 + d)
        df = _frame(rng.standard_normal((100, 8)))
        _assert_equiv(ops.ts_std(df, d), _ops_py.ts_std(df, d))

    @pytest.mark.parametrize("d", [5, 20])
    def test_scattered_nan(self, d):
        rng = np.random.default_rng(200 + d)
        x = rng.standard_normal((80, 8))
        mask = rng.random(x.shape) < 0.05
        x[mask] = np.nan
        df = _frame(x)
        _assert_equiv(ops.ts_std(df, d), _ops_py.ts_std(df, d))

    def test_full_nan_column(self):
        rng = np.random.default_rng(300)
        x = rng.standard_normal((60, 5))
        x[:, 2] = np.nan
        df = _frame(x)
        _assert_equiv(ops.ts_std(df, 10), _ops_py.ts_std(df, 10))

    def test_nan_burst(self):
        rng = np.random.default_rng(400)
        x = rng.standard_normal((60, 5))
        x[10:18, 1] = np.nan
        df = _frame(x)
        _assert_equiv(ops.ts_std(df, 10), _ops_py.ts_std(df, 10))


# ─────────────────────────────────────────────────────────────────────────────
# ts_argmax / ts_argmin
# ─────────────────────────────────────────────────────────────────────────────

class TestTsArgmaxArgmin:
    @pytest.mark.parametrize("d", [3, 5, 10, 20])
    def test_random_no_nan(self, d):
        rng = np.random.default_rng(500 + d)
        df = _frame(rng.standard_normal((60, 8)))
        _assert_equiv(ops.ts_argmax(df, d), _ops_py.ts_argmax(df, d))
        _assert_equiv(ops.ts_argmin(df, d), _ops_py.ts_argmin(df, d))

    def test_any_nan_in_window_yields_nan(self):
        rng = np.random.default_rng(600)
        x = rng.standard_normal((40, 4))
        x[10, 0] = np.nan
        df = _frame(x)
        rs = ops.ts_argmax(df, 5)
        py = _ops_py.ts_argmax(df, 5)
        _assert_equiv(rs, py)
        # Spot check: positions t=10, 11, 12, 13, 14 should ALL be NaN for col 0
        # (since their windows include the NaN at t=10)
        for t in range(10, 15):
            assert np.isnan(rs.iloc[t, 0])

    def test_ties_first_occurrence(self):
        # All-equal window: numpy argmax returns 0 (first); our op returns d-1.
        x = np.array([[5.0, 5.0, 5.0, 5.0]]).T  # shape (4, 1)
        df = _frame(x)
        out = ops.ts_argmax(df, 3)
        # Window [t-2, t-1, t] all 5.0. argmax = 0 (first), pos = d-1-0 = 2.
        assert out.iloc[2, 0] == 2.0
        assert out.iloc[3, 0] == 2.0
        _assert_equiv(out, _ops_py.ts_argmax(df, 3))


# ─────────────────────────────────────────────────────────────────────────────
# correlation — NOT yet wired through Rust dispatcher (see execution log
# 2026-06-20-rust-ops-execution-log.md "PR-3 correlation deferral"). The
# Rust impl exists in rust/stockpool_ops/src/rolling.rs (Welford form) but
# the ops.py wrapper still goes straight to _ops_py.correlation. Reason:
# pandas Rolling.corr is FP-unstable at near-zero-variance windows, which
# cascades through downstream rank/ts_sum into O(1) factor-output
# divergence for ~20 WQ alphas. Achieving bit-near equivalence requires
# replicating pandas' exact Welford / NaN-propagation FP path, which is
# scoped for a follow-up. Until then, correlation stays on pandas in the
# dispatcher, so no Layer A test belongs here.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# decay_linear — Rust impl in rust/stockpool_ops/src/decay.rs.
# ops.py currently re-exports _ops_py.decay_linear (not wired through
# dispatcher yet), so we test stockpool_ops_rs.decay_linear directly
# against the pandas oracle _ops_py.decay_linear.
# ─────────────────────────────────────────────────────────────────────────────

_rust = ops._rust  # stockpool_ops_rs module (or None when Rust unavailable)


@pytest.mark.parametrize("T,N,d", [
    (50, 20, 5),
    (30, 50, 10),
    (10, 5, 3),
])
def test_decay_linear_happy(T, N, d):
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((T, N))
    df = _frame(arr)
    rust_out = pd.DataFrame(
        _rust.decay_linear(np.ascontiguousarray(arr), d),
        index=df.index, columns=df.columns,
    )
    py_out = _ops_py.decay_linear(df, d)
    np.testing.assert_allclose(
        rust_out.values, py_out.values,
        rtol=1e-7, atol=1e-9, equal_nan=True,
    )


def test_decay_linear_all_nan_window():
    arr = np.full((10, 3), np.nan)
    arr[5:, :] = 1.0  # latter half valid
    rust_out = _rust.decay_linear(arr, 5)
    py_out = _ops_py.decay_linear(_frame(arr), 5).values
    np.testing.assert_allclose(rust_out, py_out, rtol=1e-7, atol=1e-9, equal_nan=True)


def test_decay_linear_partial_window_mid():
    """d=20, single column with 12 valid data points (mid-partial window) — weight tail alignment boundary."""
    arr = np.arange(1, 13, dtype=float).reshape(12, 1)
    rust_out = _rust.decay_linear(np.ascontiguousarray(arr), 20)
    py_out = _ops_py.decay_linear(_frame(arr), 20).values
    np.testing.assert_allclose(rust_out, py_out, rtol=1e-7, atol=1e-9, equal_nan=True)


def test_decay_linear_constant_input():
    arr = np.full((10, 3), 5.0)
    rust_out = _rust.decay_linear(arr, 3)
    py_out = _ops_py.decay_linear(_frame(arr), 3).values
    np.testing.assert_allclose(rust_out, py_out, rtol=1e-7, atol=1e-9, equal_nan=True)


def test_decay_linear_scattered_nan():
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((30, 10))
    # 10% scattered NaN
    nan_mask = rng.random((30, 10)) < 0.1
    arr[nan_mask] = np.nan
    rust_out = _rust.decay_linear(np.ascontiguousarray(arr), 5)
    py_out = _ops_py.decay_linear(_frame(arr), 5).values
    np.testing.assert_allclose(rust_out, py_out, rtol=1e-7, atol=1e-9, equal_nan=True)


# ─────────────────────────────────────────────────────────────────────────────
# indneutralize — Rust impl in rust/stockpool_ops/src/lib.rs.
# ops._rust.indneutralize(arr, sector_ids) takes an int32 sector_id array
# where -1 means "solo group" (code not in sector_map).
# Tested directly against _ops_py.indneutralize (pandas oracle).
# ─────────────────────────────────────────────────────────────────────────────


def _encode_sectors(codes, sector_map):
    """{code -> str sector} -> int32 array; codes missing from map get -1."""
    label_to_id: dict[str, int] = {}
    sector_ids = np.empty(len(codes), dtype=np.int32)
    for i, c in enumerate(codes):
        s = sector_map.get(c)
        if s is None:
            sector_ids[i] = -1
            continue
        if s not in label_to_id:
            label_to_id[s] = len(label_to_id)
        sector_ids[i] = label_to_id[s]
    return sector_ids


@pytest.mark.parametrize("T,N,n_sec", [
    (20, 30, 3),
    (50, 100, 8),
    (5, 12, 2),
])
def test_indneutralize_happy(T, N, n_sec):
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((T, N))
    codes = [f"S{i:04d}" for i in range(N)]
    sector_map = {c: f"ind_{i % n_sec}" for i, c in enumerate(codes)}
    sector_ids = _encode_sectors(codes, sector_map)
    rust_out = _rust.indneutralize(np.ascontiguousarray(arr), sector_ids)
    py_out = _ops_py.indneutralize(pd.DataFrame(arr, columns=codes), sector_map).values
    np.testing.assert_allclose(rust_out, py_out, rtol=1e-7, atol=1e-9, equal_nan=True)


def test_indneutralize_with_nan():
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((10, 8))
    arr[0:3, 0] = np.nan
    arr[5, :] = np.nan
    codes = [f"S{i}" for i in range(8)]
    sector_map = {c: ("A" if i < 4 else "B") for i, c in enumerate(codes)}
    sector_ids = _encode_sectors(codes, sector_map)
    rust_out = _rust.indneutralize(np.ascontiguousarray(arr), sector_ids)
    py_out = _ops_py.indneutralize(pd.DataFrame(arr, columns=codes), sector_map).values
    np.testing.assert_allclose(rust_out, py_out, rtol=1e-7, atol=1e-9, equal_nan=True)


def test_indneutralize_missing_code_solo_group():
    """Codes not in sector_map -> Rust sector_id=-1 -> output 0 (self-demean)."""
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((5, 4))
    codes = ["A", "B", "C", "D"]
    sector_map = {"A": "x", "B": "x"}  # C, D not in map
    sector_ids = _encode_sectors(codes, sector_map)
    rust_out = _rust.indneutralize(np.ascontiguousarray(arr), sector_ids)
    py_out = _ops_py.indneutralize(pd.DataFrame(arr, columns=codes), sector_map).values
    np.testing.assert_allclose(rust_out, py_out, rtol=1e-7, atol=1e-9, equal_nan=True)


def test_indneutralize_single_member_group():
    """Single-member industry -> mean = self -> demeaned = 0."""
    arr = np.array([[1.0, 2.0, 3.0]])
    codes = ["A", "B", "C"]
    sector_map = {"A": "x", "B": "y", "C": "z"}  # each code is its own group
    sector_ids = _encode_sectors(codes, sector_map)
    rust_out = _rust.indneutralize(np.ascontiguousarray(arr), sector_ids)
    py_out = _ops_py.indneutralize(pd.DataFrame(arr, columns=codes), sector_map).values
    np.testing.assert_allclose(rust_out, py_out, rtol=1e-7, atol=1e-9, equal_nan=True)
