"""Minimal example: industry_neutralize_panel(log_mcap=None) old vs new
must be bit-exact.

Why bit-exact: the Rust ops.indneutralize is Kahan-compensated (verified
in commit 576f8cf via snapshot 167/167 alphas). The pandas
groupby().transform("mean") produces identical output to Kahan-summed
group mean. So Python's old impl and the Rust-dispatched new impl are
mathematically identical implementations of the same operation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _old_legacy_impl(df, sector_map):
    """Snapshot of the pre-T2.2 implementation."""
    industries = pd.Series(
        {c: sector_map.get(c, "_unknown_") for c in df.columns},
        name="industry",
    )
    transposed = df.T.copy()
    transposed["__industry__"] = industries
    date_cols = [c for c in transposed.columns if c != "__industry__"]
    demeaned = transposed.groupby("__industry__")[date_cols].transform(
        lambda s: s - s.mean()
    )
    return demeaned.T


def main():
    from stockpool.ml.preprocess import industry_neutralize_panel

    rng = np.random.default_rng(42)
    T, N = 50, 200
    n_industries = 7
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    codes = [f"S{i:04d}" for i in range(N)]

    df = pd.DataFrame(
        rng.standard_normal((T, N)),
        index=dates, columns=codes,
    )
    # Scattered NaN
    df.iloc[3:7, :30] = np.nan
    df.iloc[:, ::17] = np.nan
    # Sector map with some codes intentionally unmapped -> land in "_unknown_"
    sector_map = {
        c: f"ind_{i % n_industries}"
        for i, c in enumerate(codes)
        if i % 23 != 0  # skip ~4% of codes -> unmapped
    }

    out_old = _old_legacy_impl(df, sector_map)
    out_new = industry_neutralize_panel(df, sector_map, log_mcap=None)

    np.testing.assert_allclose(
        out_new.values, out_old.values,
        rtol=0, atol=0, equal_nan=True,
    )
    print(
        f"OK: industry_neutralize_panel(log_mcap=None) Rust dispatch "
        f"== pandas groupby (bit-exact); shape={out_new.shape}"
    )


if __name__ == "__main__":
    main()
