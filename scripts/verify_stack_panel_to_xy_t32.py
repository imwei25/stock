"""Verify Rust stack_factors_long vs numpy column_stack are bit-exact.

T3.2 candidate verification script.
"""
import os
import sys
import importlib

import numpy as np
import pandas as pd

# Ensure stockpool is importable from worktree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _build_panel(seed=42):
    rng = np.random.default_rng(seed)
    T, N, F = 200, 100, 25
    dates = pd.date_range("2024-01-02", periods=T, freq="B")
    codes = [f"S{i:03d}" for i in range(N)]
    factor_panel = {}
    for f in range(F):
        name = f"factor_{f:02d}"
        arr = rng.standard_normal((T, N))
        # ~5% NaN to test dropna
        nan_mask = rng.random((T, N)) < 0.05
        arr[nan_mask] = np.nan
        factor_panel[name] = pd.DataFrame(arr, index=dates, columns=codes)
    fwd_ret = pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=codes)
    return factor_panel, fwd_ret


def main():
    factor_panel, fwd_ret = _build_panel()

    # -- Python path --
    os.environ["STOCKPOOL_USE_PYTHON_OPS"] = "1"
    from stockpool.ml import dataset as _ds
    importlib.reload(_ds)
    X_py, y_py = _ds.stack_panel_to_xy(factor_panel, fwd_ret, dropna=True)

    # -- Rust path --
    os.environ.pop("STOCKPOOL_USE_PYTHON_OPS", None)
    importlib.reload(_ds)
    X_rust, y_rust = _ds.stack_panel_to_xy(factor_panel, fwd_ret, dropna=True)

    # Verify bit-exact equality
    np.testing.assert_array_equal(
        X_py.values, X_rust.values,
        err_msg="X values differ between numpy and Rust paths"
    )
    pd.testing.assert_index_equal(X_py.index, X_rust.index)
    pd.testing.assert_index_equal(X_py.columns, X_rust.columns)
    np.testing.assert_array_equal(
        y_py.values, y_rust.values,
        err_msg="y values differ between numpy and Rust paths"
    )
    pd.testing.assert_index_equal(y_py.index, y_rust.index)

    print(f"OK: stack_panel_to_xy Rust == numpy (bit-exact); X.shape={X_py.shape}")


if __name__ == "__main__":
    main()
