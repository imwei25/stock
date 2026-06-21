//! PyO3 acceleration for stockpool factor hot ops.
//!
//! See docs/superpowers/specs/2026-06-20-rust-ops-acceleration-design.md
//! for the contract (pandas oracle in src/stockpool/factors/_ops_py.py).
//!
//! Each #[pyfunction] takes numpy arrays (zero-copy via numpy::PyReadonlyArray2)
//! and returns numpy via to_pyarray. The GIL is released around the actual
//! computation via py.allow_threads, so rayon's per-row parallelism actually
//! runs in parallel.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;

mod cs;
mod decay;
mod rolling;
mod stack;
mod util;

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

/// Rolling population stddev (ddof=0). NaN-skip; min_periods = max(1, int(d*0.6)).
#[pyfunction]
#[pyo3(name = "ts_std")]
fn ts_std_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, d: usize) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| rolling::ts_std(view, d));
    out.to_pyarray_bound(py)
}

/// Position of max in trailing-d window (0=today, d-1=oldest). Any NaN -> NaN.
#[pyfunction]
#[pyo3(name = "ts_argmax")]
fn ts_argmax_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, d: usize) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| rolling::ts_argmax(view, d));
    out.to_pyarray_bound(py)
}

/// Position of min in trailing-d window (0=today, d-1=oldest). Any NaN -> NaN.
#[pyfunction]
#[pyo3(name = "ts_argmin")]
fn ts_argmin_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, d: usize) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| rolling::ts_argmin(view, d));
    out.to_pyarray_bound(py)
}

/// Time-series quantile rank within trailing-d window. Strict; any NaN -> NaN.
#[pyfunction]
#[pyo3(name = "ts_rank")]
fn ts_rank_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, d: usize) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| rolling::ts_rank(view, d));
    out.to_pyarray_bound(py)
}

/// Rolling Pearson correlation. Strict min_periods=d; any NaN/inf -> NaN; constant -> NaN.
#[pyfunction]
#[pyo3(name = "correlation")]
fn correlation_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    d: usize,
) -> Bound<'py, PyArray2<f64>> {
    let x_view = x.as_array();
    let y_view = y.as_array();
    let out = py.allow_threads(|| rolling::correlation(x_view, y_view, d));
    out.to_pyarray_bound(py)
}

/// Rolling sum. NaN/inf-skip; min_periods = max(1, int(d*0.6)).
#[pyfunction]
#[pyo3(name = "ts_sum")]
fn ts_sum_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, d: usize) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| rolling::ts_sum(view, d));
    out.to_pyarray_bound(py)
}

/// Rolling mean. NaN/inf-skip; min_periods = max(1, int(d*0.6)).
#[pyfunction]
#[pyo3(name = "ts_mean")]
fn ts_mean_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, d: usize) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| rolling::ts_mean(view, d));
    out.to_pyarray_bound(py)
}

/// Rolling min. Strict min_periods = d; any NaN/inf in window → NaN.
#[pyfunction]
#[pyo3(name = "ts_min")]
fn ts_min_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, d: usize) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| rolling::ts_min(view, d));
    out.to_pyarray_bound(py)
}

/// Rolling max. Strict min_periods = d; any NaN/inf in window → NaN.
#[pyfunction]
#[pyo3(name = "ts_max")]
fn ts_max_py<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, d: usize) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| rolling::ts_max(view, d));
    out.to_pyarray_bound(py)
}

/// Linearly-weighted moving average. Weights 1..=d (oldest -> newest);
/// NaN positions drop from numerator + denominator; min_periods = max(1, int(d*0.6)).
#[pyfunction]
#[pyo3(name = "decay_linear")]
fn decay_linear_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    d: usize,
) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| decay::decay_linear(view, d));
    out.to_pyarray_bound(py)
}

/// Group demean per row. sector_ids[c] = integer sector id for column c
/// (negative = solo group / not in map).
#[pyfunction]
#[pyo3(name = "indneutralize")]
fn indneutralize_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    sector_ids: PyReadonlyArray1<'py, i32>,
) -> Bound<'py, PyArray2<f64>> {
    let x_view = x.as_array();
    let s_view = sector_ids.as_array();
    let out = py.allow_threads(|| cs::indneutralize(x_view, s_view));
    out.to_pyarray_bound(py)
}

/// Stack a (F, T, N) factor panel into long-form (T*N, F).
///
/// Equivalent to per-factor `panel.ravel(order='F')` column-stacked.
/// `output[stock * T + date, f] = panels[f, date, stock]`
///
/// panels must be a contiguous C-order float64 array of shape (F, T, N).
#[pyfunction]
#[pyo3(name = "stack_factors_long")]
fn stack_factors_long_py<'py>(
    py: Python<'py>,
    panels: PyReadonlyArray3<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let view = panels.as_array();
    let out = py.allow_threads(|| stack::stack_factors_long(view));
    out.to_pyarray_bound(py)
}

#[pymodule]
fn stockpool_ops_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank, m)?)?;
    m.add_function(wrap_pyfunction!(ts_std_py, m)?)?;
    m.add_function(wrap_pyfunction!(ts_argmax_py, m)?)?;
    m.add_function(wrap_pyfunction!(ts_argmin_py, m)?)?;
    m.add_function(wrap_pyfunction!(ts_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(ts_sum_py, m)?)?;
    m.add_function(wrap_pyfunction!(ts_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(ts_min_py, m)?)?;
    m.add_function(wrap_pyfunction!(ts_max_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_py, m)?)?;
    m.add_function(wrap_pyfunction!(decay_linear_py, m)?)?;
    m.add_function(wrap_pyfunction!(indneutralize_py, m)?)?;
    m.add_function(wrap_pyfunction!(stack_factors_long_py, m)?)?;
    Ok(())
}
