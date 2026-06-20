//! PyO3 acceleration for stockpool factor hot ops.
//!
//! See docs/superpowers/specs/2026-06-20-rust-ops-acceleration-design.md
//! for the contract (pandas oracle in src/stockpool/factors/_ops_py.py).
//!
//! Each #[pyfunction] takes numpy arrays (zero-copy via numpy::PyReadonlyArray2)
//! and returns numpy via to_pyarray. The GIL is released around the actual
//! computation via py.allow_threads, so rayon's per-row parallelism actually
//! runs in parallel.

use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

mod cs;

/// Cross-sectional pct-rank per row.
///
/// Equivalent to ``df.rank(axis=1, pct=True, method="average")`` element-wise
/// within atol=1e-9, rtol=1e-7.
#[pyfunction]
fn rank<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| cs::rank(view));
    out.to_pyarray_bound(py)
}

#[pymodule]
fn stockpool_ops_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank, m)?)?;
    Ok(())
}
