//! Stack a dict of T×N wide DataFrames into long-form (T·N × F) ndarray.
//!
//! The Python side handles index construction (MultiIndex of stock × date)
//! and the dropna mask. Rust here does the transpose + contiguous copy
//! efficiently, dispatching large cases to rayon.
//!
//! Layout contract (matches Python's F-order ravel per factor):
//!   panels: (F, T, N) contiguous C-order input.
//!   output: (T*N, F) row-major where
//!     output[stock_idx * T + date_idx, f] = panels[f, date_idx, stock_idx]
//!
//! This equals: panels.transpose(2,1,0).reshape(N*T, F)   (numpy equivalent).

use ndarray::{Array2, ArrayView3};
use rayon::prelude::*;

/// Stack a (F, T, N) factor panel array into a (T*N, F) long-form array.
///
/// Equivalent to the Python:
///   col_arrays = [panels[f].ravel(order="F") for f in range(F)]
///   X_arr = np.column_stack(col_arrays)
///
/// Which equals: panels.transpose(2,1,0).reshape(N*T, F)
pub fn stack_factors_long(panels: ArrayView3<'_, f64>) -> Array2<f64> {
    let (f_count, t, n) = panels.dim();
    let rows = t * n; // = N*T

    // Allocate output flat buffer.
    // Output layout (row-major): element [r, f] = flat[r * f_count + f]
    // We fill it by iterating over (stock, date, factor) with stock-outer, date-inner.
    // For a given (stock s, date d): row r = s * t + d
    //   output[r, f] = panels[f, d, s]

    // Use rayon to parallelize over stocks (N dimension).
    // Each stock owns rows [s*t .. (s+1)*t) in the output — disjoint slices.
    // We pre-allocate and chunk the output by stock.

    let mut flat = vec![0.0f64; rows * f_count];

    // Split flat into N chunks of (t * f_count) elements each — one per stock.
    // rayon::par_chunks_mut is ideal here.
    flat.par_chunks_mut(t * f_count)
        .enumerate()
        .for_each(|(stock_idx, chunk)| {
            // chunk has length t * f_count
            // chunk[date_idx * f_count + f] = output[stock_idx*t + date_idx, f]
            //                               = panels[f, date_idx, stock_idx]
            for date_idx in 0..t {
                for f in 0..f_count {
                    // SAFETY: date_idx < t, f < f_count, stock_idx < n — all in bounds.
                    let val = unsafe { *panels.uget([f, date_idx, stock_idx]) };
                    chunk[date_idx * f_count + f] = val;
                }
            }
        });

    Array2::from_shape_vec((rows, f_count), flat)
        .expect("stack_factors_long: shape mismatch (should never happen)")
}
