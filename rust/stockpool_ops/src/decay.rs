//! Linearly-weighted moving average (NaN-safe, partial-window aware).
//!
//! Mirrors `_ops_py.decay_linear`:
//!   * weights = 1..=d (oldest -> newest)
//!   * NaN positions excluded from numerator AND denominator,
//!     remaining weights renormalize
//!   * all-NaN window -> NaN
//!   * min_periods = max(1, int(d * 0.6))
//!   * partial windows (t+1 < d): weights[-len(window):] tail-aligned

use ndarray::{Array2, ArrayView2};

use crate::util::{min_periods, rolling_apply_col};

pub fn decay_linear(x: ArrayView2<f64>, d: usize) -> Array2<f64> {
    let mp = min_periods(d);
    // Pre-compute weights[1..=d] once; per-window we slice the tail.
    let full_weights: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    rolling_apply_col(x, d, mp, false, move |window: &[f64]| {
        let w_slice = &full_weights[full_weights.len() - window.len()..];
        let mut num = 0.0;
        let mut den = 0.0;
        let mut any_valid = false;
        for (v, &w) in window.iter().zip(w_slice.iter()) {
            if v.is_finite() {
                num += v * w;
                den += w;
                any_valid = true;
            }
        }
        if !any_valid {
            f64::NAN
        } else {
            num / den
        }
    })
}
