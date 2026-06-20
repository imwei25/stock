//! Rolling time-series ops (single-array).

use ndarray::{Array2, ArrayView2, Axis, Zip};

use crate::util::{min_periods, rolling_apply_col, rolling_apply_col_pair};

/// Rolling population stddev (ddof=0), pandas-equivalent
/// `x.rolling(d, min_periods=_min_periods(d)).std(ddof=0)`. NaN/inf-skip.
///
/// Uses the same online (running-sum) algorithm as pandas rolling to ensure
/// bit-identical results: var = (n*sum_x2 - sum_x^2) / n^2. Pandas treats
/// inf as NaN in rolling statistics, so we skip both.
pub fn ts_std(x: ArrayView2<f64>, d: usize) -> Array2<f64> {
    let mp = min_periods(d);
    let (nrows, ncols) = x.dim();
    let mut out = Array2::from_elem((nrows, ncols), f64::NAN);

    Zip::from(out.axis_iter_mut(Axis(1)))
        .and(x.axis_iter(Axis(1)))
        .par_for_each(|mut out_col, in_col| {
            let col: Vec<f64> = in_col.iter().copied().collect();
            // Online running sums, mirroring pandas rolling cython internals.
            // sum_x and sum_x2 accumulate over the entire column; we add the
            // new value and subtract the old one that leaves the window.
            // This produces the same floating-point path as pandas and ensures
            // boundary-condition parity with the pandas oracle.
            let mut sum_x = 0.0_f64;
            let mut sum_x2 = 0.0_f64;
            let mut cnt = 0usize;

            for t in 0..nrows {
                let new_v = col[t];
                // Add incoming value (skip NaN/inf like pandas).
                if new_v.is_finite() {
                    sum_x += new_v;
                    sum_x2 += new_v * new_v;
                    cnt += 1;
                }
                // Drop outgoing value when window exceeds d.
                if t >= d {
                    let old_v = col[t - d];
                    if old_v.is_finite() {
                        sum_x -= old_v;
                        sum_x2 -= old_v * old_v;
                        cnt -= 1;
                    }
                }
                if cnt >= mp {
                    // Population variance: (sum_x2 - sum_x^2/n) / n, clamped >= 0.
                    let n = cnt as f64;
                    let var = (sum_x2 - sum_x * sum_x / n) / n;
                    out_col[t] = var.max(0.0_f64).sqrt();
                }
                // else: leave as NaN (pre-filled)
            }
        });
    out
}

/// Position of the max within the trailing-d window:
/// 0 = today, d-1 = oldest. ANY NaN or inf in window -> NaN.
/// Tie-break: first occurrence (oldest) -- numpy argmax semantics.
///
/// Note: pandas rolling.apply internally converts inf to NaN, so windows
/// containing inf also produce NaN — we match that behaviour here.
pub fn ts_argmax(x: ArrayView2<f64>, d: usize) -> Array2<f64> {
    // Strict min_periods=d: only full-d windows; any NaN/inf in window -> NaN.
    rolling_apply_col(x, d, d, true, |w| {
        // any NaN or inf -> NaN (pandas treats inf as NaN in rolling.apply)
        if w.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        // numpy argmax: FIRST occurrence wins for ties -- update only on
        // strictly greater. Rust's iter().max_by returns LAST, so do it by hand.
        let mut max_i = 0usize;
        let mut max_v = w[0];
        for (i, &v) in w.iter().enumerate().skip(1) {
            if v > max_v {
                max_v = v;
                max_i = i;
            }
        }
        (w.len() - 1 - max_i) as f64
    })
}

/// Position of the min within the trailing-d window: 0 = today, d-1 = oldest.
/// ANY NaN or inf in window -> NaN (pandas rolling.apply treats inf as NaN).
pub fn ts_argmin(x: ArrayView2<f64>, d: usize) -> Array2<f64> {
    rolling_apply_col(x, d, d, true, |w| {
        if w.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        let mut min_i = 0usize;
        let mut min_v = w[0];
        for (i, &v) in w.iter().enumerate().skip(1) {
            if v < min_v {
                min_v = v;
                min_i = i;
            }
        }
        (w.len() - 1 - min_i) as f64
    })
}

/// Rolling Pearson correlation between paired columns of x and y.
/// Pandas `x.rolling(d, min_periods=d).corr(y)` equivalent:
///   * strict min_periods=d
///   * ANY NaN/inf at any position in EITHER window -> NaN
///   * constant series (variance effectively 0 by std<1e-7) -> NaN
///   * |corr| > 1 (FP garbage) -> NaN
///
/// Uses Welford's online algorithm to match pandas exactly: for
/// bit-identical constant windows pandas returns std=0 and our
/// Welford accumulator also stays at 0 (deltas are 0 each step).
/// A naive two-pass `sum((x - mean)^2)` would leak FP noise from the
/// sum-then-mean path and produce std~1e-17 on constants -- enough
/// to skip the guard and emit garbage downstream.
pub fn correlation(x: ArrayView2<f64>, y: ArrayView2<f64>, d: usize) -> Array2<f64> {
    rolling_apply_col_pair(x, y, d, d, |wx, wy| {
        if wx.iter().any(|v| !v.is_finite()) || wy.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        // Welford-style online mean + Co-moment accumulators.
        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;
        let mut m2x = 0.0_f64;
        let mut m2y = 0.0_f64;
        let mut cxy = 0.0_f64;
        let mut n = 0.0_f64;
        for (&xv, &yv) in wx.iter().zip(wy.iter()) {
            n += 1.0;
            let dx = xv - mean_x;
            mean_x += dx / n;
            let dy_raw = yv - mean_y;
            mean_y += dy_raw / n;
            m2x += dx * (xv - mean_x);
            m2y += dy_raw * (yv - mean_y);
            // Co-moment Welford: Cxy += dx * (yv - mean_y_new)
            cxy += dx * (yv - mean_y);
        }
        let std_x = (m2x / n).sqrt();
        let std_y = (m2y / n).sqrt();
        if std_x < 1e-7 || std_y < 1e-7 {
            return f64::NAN;
        }
        let denom = (m2x * m2y).sqrt();
        let result = cxy / denom;
        if !result.is_finite() || result.abs() > 1.0 {
            return f64::NAN;
        }
        result
    })
}
