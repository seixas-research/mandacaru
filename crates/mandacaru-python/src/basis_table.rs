//! Bindings of the tabulated radial functions of the PAW-LCAO basis: spline
//! evaluation with the continuation below the first node, and the piecewise
//! cubic coefficients. Arrays are copied in and returned owned; the numerics are
//! in `mandacaru_core::basis::table`.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::basis::{
    sampling::{self, Orbital, Sampled},
    table,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// `R(r)` (or `dR/dr`) of the table `(r, values)` for angular momentum `l` at `nodes`.
#[pyfunction]
fn radial_table_evaluate<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    l: usize,
    nodes: PyReadonlyArray1<'py, f64>,
    slope: bool,
) -> PyResult<Array<'py, f64>> {
    let ctx = context.inner.clone();
    let (r, _a) = copy(&ctx, &r)?;
    let (values, _b) = copy(&ctx, &values)?;
    let (nodes, _c) = copy(&ctx, &nodes)?;
    let out = py
        .detach(|| table::evaluate(&ctx, &r, &values, l, &nodes, slope))
        .map_err(error)?;
    array(py, &[out.len()], out)
}

/// `(4, n - 1)` piecewise-cubic coefficients of the table, highest power first.
#[pyfunction]
fn radial_table_coefficients<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let ctx = context.inner.clone();
    let (r, _a) = copy(&ctx, &r)?;
    let (values, _b) = copy(&ctx, &values)?;
    let out = py
        .detach(|| table::coefficients(&ctx, &r, &values))
        .map_err(error)?;
    array(py, &[4, r.len() - 1], out)
}

/// `R(r) Y_lm` (or, with `gradient`, its `(3, n)` gradient) of the table `(r, values)` centered at
/// `center`, at the points `x, y, z`: `float64` for real harmonics, `complex128` otherwise.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn radial_orbital_sample<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    l: usize,
    m: i32,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    center: [f64; 3],
    real: bool,
    gradient: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let ctx = context.inner.clone();
    let (r, _a) = copy(&ctx, &r)?;
    let (values, _b) = copy(&ctx, &values)?;
    let (x, _c) = copy(&ctx, &x)?;
    let (y, _d) = copy(&ctx, &y)?;
    let (z, _e) = copy(&ctx, &z)?;
    let n = x.len();
    let sampled = py
        .detach(|| {
            let orbital = Orbital { r: &r, values: &values, l, m, center, real };
            if gradient {
                sampling::gradient(&ctx, &orbital, &x, &y, &z)
            } else {
                sampling::values(&ctx, &orbital, &x, &y, &z)
            }
        })
        .map_err(error)?;
    let shape: &[usize] = if gradient { &[3, n] } else { &[n] };
    Ok(match sampled {
        Sampled::Real(out) => array(py, shape, out)?.into_any(),
        Sampled::Complex(out) => array(py, shape, out)?.into_any(),
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(radial_orbital_sample, m)?)?;
    m.add_function(wrap_pyfunction!(radial_table_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(radial_table_coefficients, m)?)?;
    Ok(())
}
