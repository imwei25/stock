//! Shared utilities for time-series rolling ops.
//!
//! `rolling_apply_col` and `rolling_apply_col_pair` parallelize over the
//! N (column / stock) axis via rayon. For each column we materialize a
//! single contiguous Vec<f64> of length T once, then slide a window over
//! it (zero-copy slices into the Vec) so the closure `f` sees `&[f64]`.

use ndarray::{Array2, ArrayView2, Axis, Zip};

/// Mirrors `_ops_py._min_periods`: relax to 60% of the window
/// (`max(1, int(d * 0.6))`).
#[inline]
pub fn min_periods(d: usize) -> usize {
    let v = ((d as f64) * 0.6) as usize;
    v.max(1)
}

/// Apply `f` to every trailing-d window in every column. Output[t, j]
/// is set to `f(window)` only when `window.len() == d` AND
/// `n_non_nan(window) >= min_periods`; otherwise NaN (matching pandas
/// rolling.apply with min_periods).
pub fn rolling_apply_col<F>(
    x: ArrayView2<f64>,
    d: usize,
    mp: usize,
    f: F,
) -> Array2<f64>
where
    F: Fn(&[f64]) -> f64 + Sync + Send,
{
    let (nrows, ncols) = x.dim();
    let mut out = Array2::from_elem((nrows, ncols), f64::NAN);
    Zip::from(out.axis_iter_mut(Axis(1)))
        .and(x.axis_iter(Axis(1)))
        .par_for_each(|mut out_col, in_col| {
            // Make the column contiguous so windows are zero-copy slices.
            let col: Vec<f64> = in_col.iter().copied().collect();
            for t in 0..nrows {
                if t + 1 < d {
                    continue;
                }
                let start = t + 1 - d;
                let window = &col[start..=t];
                let n_valid = window.iter().filter(|v| !v.is_nan()).count();
                if n_valid >= mp {
                    out_col[t] = f(window);
                }
            }
        });
    out
}

/// Two-array variant for `correlation` / `covariance`. Paired windows;
/// NaN in EITHER side (at any position) counts as missing for that t.
/// `min_periods` must be satisfied by the paired-valid count.
pub fn rolling_apply_col_pair<F>(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    d: usize,
    mp: usize,
    f: F,
) -> Array2<f64>
where
    F: Fn(&[f64], &[f64]) -> f64 + Sync + Send,
{
    let (nrows, ncols) = x.dim();
    assert_eq!(x.dim(), y.dim(), "x and y must have identical shape");
    let mut out = Array2::from_elem((nrows, ncols), f64::NAN);
    Zip::from(out.axis_iter_mut(Axis(1)))
        .and(x.axis_iter(Axis(1)))
        .and(y.axis_iter(Axis(1)))
        .par_for_each(|mut out_col, in_x, in_y| {
            let xc: Vec<f64> = in_x.iter().copied().collect();
            let yc: Vec<f64> = in_y.iter().copied().collect();
            for t in 0..nrows {
                if t + 1 < d {
                    continue;
                }
                let start = t + 1 - d;
                let wx = &xc[start..=t];
                let wy = &yc[start..=t];
                // Pandas Rolling.corr treats NaN strictly: any NaN at any
                // window position (in x OR y) zeroes the pairwise-valid
                // count locally. Caller's f must handle NaN itself; we
                // only enforce min_periods on positions where BOTH are
                // non-NaN, matching pandas.
                let n_valid = wx
                    .iter()
                    .zip(wy.iter())
                    .filter(|(a, b)| !a.is_nan() && !b.is_nan())
                    .count();
                if n_valid >= mp {
                    out_col[t] = f(wx, wy);
                }
            }
        });
    out
}
