//! PyO3 acceleration for stockpool factor hot ops.
//!
//! See docs/superpowers/specs/2026-06-20-rust-ops-acceleration-design.md
//! for the contract (pandas oracle in src/stockpool/factors/_ops_py.py).

use pyo3::prelude::*;

#[pymodule]
fn stockpool_ops_rs(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
