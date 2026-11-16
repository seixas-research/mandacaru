//! Bindings of the spherical harmonics (`math::harmonics`): samples, Cartesian gradients and the
//! real-from-complex transform. Arrays are copied in and returned owned.
use crate::arrays::{array, error, layout, Array};
use mandacaru_core::derived;
use mandacaru_core::math::harmonics::{self, GradientSamples, Samples};
use numpy::{Complex64, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
pub fn real_harmonic_transform<'py>(py: Python<'py>, l: usize) -> PyResult<Array<'py, Complex64>> {
    let dim = 2 * l + 1;
    array(py, &[dim, dim], harmonics::real_transform(l))
}

#[pyfunction]
pub fn spherical_harmonic_evaluate<'py>(
    py: Python<'py>,
    l: usize,
    m: i32,
    theta: PyReadonlyArray1<'py, f64>,
    phi: PyReadonlyArray1<'py, f64>,
    real: bool,
) -> PyResult<Bound<'py, PyAny>> {
    layout(&theta)?;
    layout(&phi)?;
    let n = theta.as_slice()?.len();
    Ok(
        match harmonics::harmonic_samples(l, m, theta.as_slice()?, phi.as_slice()?, real)
            .map_err(error)?
        {
            Samples::Real(values) => array(py, &[n], values)?.into_any(),
            Samples::Complex(values) => array(py, &[n], values)?.into_any(),
        },
    )
}

type Gradient<'py> = (
    Bound<'py, PyAny>,
    (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>),
);

#[pyfunction]
pub fn spherical_harmonic_gradient_evaluate<'py>(
    py: Python<'py>,
    l: usize,
    m: i32,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    real: bool,
) -> PyResult<Gradient<'py>> {
    layout(&x)?;
    layout(&y)?;
    layout(&z)?;
    let n = x.as_slice()?.len();
    let samples = harmonics::harmonic_gradient_samples(
        l,
        m,
        x.as_slice()?,
        y.as_slice()?,
        z.as_slice()?,
        real,
    )
    .map_err(error)?;
    Ok(match samples {
        GradientSamples::Real(values, [gx, gy, gz]) => (
            array(py, &[n], values)?.into_any(),
            (
                array(py, &[n], gx)?.into_any(),
                array(py, &[n], gy)?.into_any(),
                array(py, &[n], gz)?.into_any(),
            ),
        ),
        GradientSamples::Complex(values, [gx, gy, gz]) => (
            array(py, &[n], values)?.into_any(),
            (
                array(py, &[n], gx)?.into_any(),
                array(py, &[n], gy)?.into_any(),
                array(py, &[n], gz)?.into_any(),
            ),
        ),
    })
}

/// The magnetic quantum numbers `-l, ..., l` (`derived::magnetic_numbers`).
#[pyfunction]
fn magnetic_numbers(l: usize) -> Vec<i32> {
    derived::magnetic_numbers(l)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(magnetic_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(real_harmonic_transform, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_gradient_evaluate, m)?)?;
    Ok(())
}
