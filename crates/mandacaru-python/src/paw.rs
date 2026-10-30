//! Typed PAW boundaries; all radial mathematics executes in the core crate.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::paw::{channel, radial};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub fn local<'py>(
    context: &PyContext,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    radius: f64,
    shift: f64,
    fit_window: Option<f64>,
) -> PyResult<Array<'py, f64>> {
    let (r, _r) = copy(&context.inner, &r)?;
    let (potential, _v) = copy(&context.inner, &potential)?;
    let out = py
        .detach(|| {
            radial::local_potential(&context.inner, &r, &potential, radius, shift, fit_window)
        })
        .map_err(error)?;
    array(py, &[r.len()], out.as_slice().to_vec())
}
pub fn channel<'py>(
    context: &PyContext,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    ae: PyReadonlyArray2<'py, f64>,
    potential: PyReadonlyArray1<'py, f64>,
    local: PyReadonlyArray1<'py, f64>,
    l: usize,
    radius: f64,
    local_radius: f64,
    energies: Vec<f64>,
    wavevectors: Vec<f64>,
    coefficients: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if ae.shape() != [energies.len(), r.shape()[0]]
        || coefficients.shape() != [energies.len(), wavevectors.len()]
    {
        return Err(PyValueError::new_err(
            "PAW waves must be (references,radii) and coefficients (references,wavevectors)",
        ));
    }
    let (r, _r) = copy(&context.inner, &r)?;
    let (ae, _a) = copy(&context.inner, &ae)?;
    let (potential, _v) = copy(&context.inner, &potential)?;
    let (local, _l) = copy(&context.inner, &local)?;
    let (coefficients, _c) = copy(&context.inner, &coefficients)?;
    let count = energies.len();
    let out = py
        .detach(|| {
            channel::construct(
                &context.inner,
                &channel::ChannelInput {
                    r: &r,
                    ae: &ae,
                    vae: &potential,
                    local: &local,
                    l,
                    radius,
                    local_radius,
                    energies: &energies,
                    wavevectors: &wavevectors,
                    coefficients: &coefficients,
                },
            )
        })
        .map_err(error)?;
    let result = PyDict::new(py);
    result.set_item(
        "pseudo_waves",
        array(py, &[count, r.len()], out.pseudo.as_slice().to_vec())?,
    )?;
    result.set_item(
        "projectors",
        array(py, &[count, r.len()], out.projectors.as_slice().to_vec())?,
    )?;
    for (name, matrix) in [
        ("overlap", out.overlap),
        ("screened", out.screened),
        ("kinetic", out.kinetic),
        ("potential", out.potential),
    ] {
        let values = (0..count)
            .flat_map(|i| {
                let matrix = &matrix;
                (0..count).map(move |j| matrix[(i, j)])
            })
            .collect();
        result.set_item(name, array(py, &[count, count], values)?)?;
    }
    result.set_item("duality_error", out.duality_error)?;
    result.set_item("asymmetry", out.asymmetry)?;
    Ok(result)
}
