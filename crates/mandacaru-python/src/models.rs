//! Typed boundaries for scientific models shared by multiple solver families.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::models::spherical;
use numpy::PyReadonlyArray1;
use pyo3::{prelude::*, types::PyDict};

pub fn radial_hartree<'py>(
    context: &PyContext,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    density: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let (r, _r) = copy(&context.inner, &r)?;
    let (density, _d) = copy(&context.inner, &density)?;
    let out = py
        .detach(|| spherical::hartree(&context.inner, &r, &density))
        .map_err(error)?;
    array(py, &[r.len()], out.as_slice().to_vec())
}
pub fn radial_xc<'py>(
    context: &PyContext,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    density: PyReadonlyArray1<'py, f64>,
    functional: &str,
    relativistic: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (r, _r) = copy(&context.inner, &r)?;
    let (density, _d) = copy(&context.inner, &density)?;
    let out = py
        .detach(|| {
            spherical::exchange_correlation(&context.inner, &r, &density, functional, relativistic)
        })
        .map_err(error)?;
    let result = PyDict::new(py);
    result.set_item("energy", out.energy)?;
    result.set_item(
        "per_particle",
        array(py, &[r.len()], out.per_particle.as_slice().to_vec())?,
    )?;
    result.set_item(
        "potential",
        array(py, &[r.len()], out.potential.as_slice().to_vec())?,
    )?;
    Ok(result)
}
