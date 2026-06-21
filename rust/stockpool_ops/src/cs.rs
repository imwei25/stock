//! Cross-sectional ops (operate along axis=1 / per-row).

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Zip};

/// Cross-sectional pct-rank per row.
///
/// Mirrors pandas `df.rank(axis=1, pct=True, method="average")`:
///   * NaN cells stay NaN, excluded from the ranking.
///   * Ties get the mean of their (1-based) ranks.
///   * Result = avg_rank / n_valid, ∈ (0, 1] for non-NaN cells.
///
/// Empty (all-NaN) rows produce all-NaN output (no division by zero).
pub fn rank(x: ArrayView2<f64>) -> Array2<f64> {
    let (nrows, ncols) = x.dim();
    let mut out = Array2::from_elem((nrows, ncols), f64::NAN);
    Zip::from(out.axis_iter_mut(Axis(0)))
        .and(x.axis_iter(Axis(0)))
        .par_for_each(|out_row, in_row| rank_row_into(in_row, out_row));
    out
}

fn rank_row_into(x: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
    let mut valid: Vec<(usize, f64)> = x
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.is_nan() { None } else { Some((i, v)) })
        .collect();
    if valid.is_empty() {
        // out is already pre-filled with NaN; nothing to do.
        return;
    }
    let n_valid = valid.len() as f64;
    // Sort by value (NaN already filtered; partial_cmp is total here).
    valid.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("NaN already filtered"));
    // Average-rank tie handling. 1-based ranks.
    // Exact equality (`==` on f64) matches pandas method="average" tie
    // semantics — pandas also requires bit-identical values for ties.
    // Epsilon-tolerant tie detection would DIVERGE from the oracle.
    let mut i = 0;
    while i < valid.len() {
        let mut j = i + 1;
        while j < valid.len() && valid[j].1 == valid[i].1 {
            j += 1;
        }
        // Group [i..j) is tied: positions i+1, i+2, ..., j (1-based).
        // Average = (i+1 + j) / 2 = (i + j + 1) / 2.
        let avg_rank = (i as f64 + j as f64 + 1.0) / 2.0;
        let pct = avg_rank / n_valid;
        for k in i..j {
            out[valid[k].0] = pct;
        }
        i = j;
    }
}
