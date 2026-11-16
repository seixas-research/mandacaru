//! Bindings of the unit Slater fits (`sto_fit`).
use crate::arrays::error;
use pyo3::prelude::*;

/// `(exponents, coefficients, overlap^2)` of the `n_gaussians`-term fit of the unit `(n, l)`
/// Slater orbital (`sto_fit::fit_sto_reference`).
#[pyfunction]
pub fn fit_sto_reference(
    py: Python<'_>,
    n: usize,
    l: usize,
    n_gaussians: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
    py.detach(|| mandacaru_core::sto_fit::fit_sto_reference(n, l, n_gaussians))
        .map_err(error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_sto_reference, m)?)?;
    Ok(())
}
