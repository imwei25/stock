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

/// Group demean per row.
///
/// `sector_ids[c]` is the integer sector id for column c. Codes with id < 0
/// are treated as solo groups (output = 0 for finite cells — matches the
/// pandas oracle's `__solo__<code>` semantics: `self - self == 0`).
///
/// NaN cells are excluded from group means and remain NaN in the output.
pub fn indneutralize(
    x: ArrayView2<f64>,
    sector_ids: ArrayView1<i32>,
) -> Array2<f64> {
    let (nrows, ncols) = x.dim();
    let mut out = Array2::from_elem((nrows, ncols), f64::NAN);
    // Compute max sector id once (skip negatives — solo bucket).
    let max_id: i32 = sector_ids
        .iter()
        .copied()
        .filter(|&v| v >= 0)
        .max()
        .unwrap_or(-1);
    if max_id < 0 {
        // All solo -> output is 0 for finite cells, NaN otherwise.
        Zip::from(out.axis_iter_mut(Axis(0)))
            .and(x.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, in_row| {
                for (o, v) in out_row.iter_mut().zip(in_row.iter()) {
                    if v.is_finite() {
                        *o = 0.0;
                    }
                }
            });
        return out;
    }
    let n_buckets = (max_id + 1) as usize;
    Zip::from(out.axis_iter_mut(Axis(0)))
        .and(x.axis_iter(Axis(0)))
        .par_for_each(|mut out_row, in_row| {
            // Kahan compensated summation per sector to match pandas' groupby
            // numerically-stable reduce path (avoids ULP divergence that would
            // cascade through downstream rank() calls).
            let mut sum_by_sector = vec![0.0f64; n_buckets];
            let mut comp_by_sector = vec![0.0f64; n_buckets];  // Kahan compensator
            let mut cnt_by_sector = vec![0usize; n_buckets];
            for (c, v) in in_row.iter().enumerate() {
                let sid = sector_ids[c];
                if sid < 0 || !v.is_finite() {
                    continue;
                }
                let s = sid as usize;
                // Kahan step
                let y = *v - comp_by_sector[s];
                let t = sum_by_sector[s] + y;
                comp_by_sector[s] = (t - sum_by_sector[s]) - y;
                sum_by_sector[s] = t;
                cnt_by_sector[s] += 1;
            }
            for (c, v) in in_row.iter().enumerate() {
                if !v.is_finite() {
                    continue;
                }
                let sid = sector_ids[c];
                if sid < 0 {
                    out_row[c] = 0.0;  // solo: self - self
                    continue;
                }
                let n = cnt_by_sector[sid as usize];
                if n == 0 {
                    continue;  // shouldn't happen since v is finite
                }
                let mean = sum_by_sector[sid as usize] / (n as f64);
                out_row[c] = *v - mean;
            }
        });
    out
}
