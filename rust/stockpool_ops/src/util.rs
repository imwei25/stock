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

/// Apply `f` to every trailing-d window in every column.
///
/// For each row `t`:
///   - If `strict_d` is true: only process when `t + 1 >= d` (full window required),
///     matching `min_periods=d` pandas semantics (used by ts_argmax/ts_argmin/correlation).
///   - If `strict_d` is false: process from t=0 with window = col[0..=t] when t+1 < d,
///     or col[t+1-d..=t] when t+1 >= d. This matches pandas rolling with `min_periods < d`.
///
/// In both cases, `n_non_nan(window) >= mp` must hold; otherwise output is NaN.
pub fn rolling_apply_col<F>(
    x: ArrayView2<f64>,
    d: usize,
    mp: usize,
    strict_d: bool,
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
                let start = if t + 1 < d {
                    if strict_d {
                        // Not enough rows yet for a full window.
                        continue;
                    }
                    0
                } else {
                    t + 1 - d
                };
                let window = &col[start..=t];
                // Count finite values (pandas treats inf as NaN in rolling stats).
                let n_valid = window.iter().filter(|v| v.is_finite()).count();
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
                // Pandas Rolling.corr treats NaN (and inf) strictly: any
                // non-finite value at any position (in x OR y) counts as
                // missing. Caller's f must handle non-finite itself; we
                // only enforce min_periods on positions where BOTH are
                // finite, matching pandas.
                let n_valid = wx
                    .iter()
                    .zip(wy.iter())
                    .filter(|(a, b)| a.is_finite() && b.is_finite())
                    .count();
                if n_valid >= mp {
                    out_col[t] = f(wx, wy);
                }
            }
        });
    out
}
