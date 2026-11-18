//! Binding of the all-electron logarithmic derivative check used by dataset diagnostics.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::{models::relativity::Treatment, paw::checks};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature=(context, r, v, l, energies, r_cut, z_eff, treatment="none", kappa=None))]
fn paw_log_derivatives_ae<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    l: usize,
    energies: Vec<f64>,
    r_cut: f64,
    z_eff: f64,
    treatment: &str,
    kappa: Option<i32>,
) -> PyResult<Array<'py, f64>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (v, _b) = copy(&context.inner, &v)?;
    let treatment = Treatment::parse(treatment).map_err(error)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| {
            checks::log_derivatives_ae(ctx, &r, &v, l, &energies, r_cut, z_eff, treatment, kappa)
        })
        .map_err(error)?;
    array(py, &[out.len()], out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_log_derivatives_ae, m)?)?;
    Ok(())
}
