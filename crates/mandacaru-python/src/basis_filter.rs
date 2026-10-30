//! Bindings of the Fourier filter of a radial basis function. Arrays are copied in
//! and returned owned; the numerics are in `mandacaru_core::basis::filter`.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::basis::filter::{self, Method, Options};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};

/// `(filtered, report)`: the table with the content above `k_c` removed and
/// `(residual_before, residual_after, support_radius, mask_radius, masked,
/// norm_kept, max_change)`. A `None` option takes the engine's default.
#[pyfunction]
#[pyo3(signature = (context, r, values, l, k_c, method=None, rolloff=None, oversample=None, renormalize=true, switch_margin=None))]
#[allow(clippy::type_complexity)]
fn radial_filter<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    l: usize,
    k_c: f64,
    method: Option<&str>,
    rolloff: Option<f64>,
    oversample: Option<usize>,
    renormalize: bool,
    switch_margin: Option<f64>,
) -> PyResult<(Array<'py, f64>, (f64, f64, f64, f64, bool, f64, f64))> {
    let ctx = context.inner.clone();
    let defaults = Options::default();
    let options = Options {
        method: match method {
            Some(name) => Method::parse(name).map_err(error)?,
            None => defaults.method,
        },
        rolloff: rolloff.unwrap_or(defaults.rolloff),
        oversample: oversample.unwrap_or(defaults.oversample),
        renormalize,
        switch_margin: switch_margin.unwrap_or(defaults.switch_margin),
    };
    let (r, _a) = copy(&ctx, &r)?;
    let (values, _b) = copy(&ctx, &values)?;
    let (filtered, report) = py
        .detach(|| filter::filter(&ctx, &r, &values, l, k_c, &options))
        .map_err(error)?;
    let report = (
        report.residual_before,
        report.residual_after,
        report.support_radius,
        report.mask_radius,
        report.masked,
        report.norm_kept,
        report.max_change,
    );
    Ok((array(py, &[filtered.len()], filtered)?, report))
}

/// Fraction of `int R^2 r^2 dr` carried by wave-vectors below `k_c`.
#[pyfunction]
#[pyo3(signature = (context, r, values, l, k_c, oversample=None))]
fn radial_band_weight(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    r: PyReadonlyArray1<'_, f64>,
    values: PyReadonlyArray1<'_, f64>,
    l: usize,
    k_c: f64,
    oversample: Option<usize>,
) -> PyResult<f64> {
    let ctx = context.inner.clone();
    let oversample = oversample.unwrap_or(Options::default().oversample);
    if oversample == 0 {
        return Err(PyValueError::new_err(
            "the transform oversampling must be at least 1",
        ));
    }
    let (r, _a) = copy(&ctx, &r)?;
    let (values, _b) = copy(&ctx, &values)?;
    py.detach(|| filter::band_weight(&ctx, &r, &values, l, k_c, oversample))
        .map_err(error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The filter methods, the default first: one definition, in Rust.
    m.add("RADIAL_FILTER_METHODS", Method::NAMES.to_vec())?;
    m.add_function(wrap_pyfunction!(radial_filter, m)?)?;
    m.add_function(wrap_pyfunction!(radial_band_weight, m)?)?;
    Ok(())
}
