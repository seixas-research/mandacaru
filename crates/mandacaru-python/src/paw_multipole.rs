//! Bindings of the PAW compensation multipoles and the nonlinear core
//! correction. Arrays are copied in and returned owned; the work is in
//! `mandacaru_core::paw`.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::paw::{
    core_correction,
    multipole::{self, gaunt, gaunt_grid},
    multipole_pairs::{self as pairs, ChannelWaves, Pair},
};
use numpy::{Complex64, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyfunction]
fn paw_shape_function<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    radius: PyReadonlyArray1<'py, f64>,
    r_g: f64,
    l: usize,
) -> PyResult<Array<'py, f64>> {
    let (radius, _guard) = copy(&context.inner, &radius)?;
    let out = py
        .detach(|| multipole::shape_function(&radius, r_g, l))
        .map_err(error)?;
    array(py, &[out.len()], out)
}

#[pyfunction]
fn paw_shape_potential<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    radius: PyReadonlyArray1<'py, f64>,
    r_g: f64,
    l: usize,
) -> PyResult<Array<'py, f64>> {
    let (radius, _guard) = copy(&context.inner, &radius)?;
    let out = py
        .detach(|| multipole::shape_potential(&radius, r_g, l))
        .map_err(error)?;
    array(py, &[out.len()], out)
}

#[pyfunction]
fn paw_multipole_range(l1: usize, l2: usize) -> Vec<usize> {
    multipole::multipole_range(l1, l2)
}

/// `keys` is `(n, 6)`: `L, M, l1, m1, l2, m2`.
#[pyfunction]
fn paw_gaunt<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    keys: PyReadonlyArray2<'py, i64>,
    real: bool,
) -> PyResult<Array<'py, Complex64>> {
    let (keys, _guard) = copy(&context.inner, &keys)?;
    if keys.len() % 6 != 0 || keys.iter().any(|k| k.unsigned_abs() > 64) {
        return Err(PyValueError::new_err(
            "Gaunt keys are rows of L, M, l1, m1, l2, m2",
        ));
    }
    let n = keys.len() / 6;
    let out = py
        .detach(|| {
            let grid = gaunt_grid()?;
            Ok::<_, mandacaru_core::Error>(
                keys.chunks(6)
                    .map(|k| {
                        let (l, a, b) = (
                            k[0].max(0) as usize,
                            k[2].max(0) as usize,
                            k[4].max(0) as usize,
                        );
                        if k[0] < 0 || k[2] < 0 || k[4] < 0 {
                            return Complex64::new(0., 0.);
                        }
                        let v = gaunt(
                            [l, a, b],
                            [k[1] as i32, k[3] as i32, k[5] as i32],
                            real,
                            &grid,
                        );
                        Complex64::new(v.re, v.im)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .map_err(error)?;
    array(py, &[n], out)
}

#[pyfunction]
fn paw_angular_grid<'py>(
    py: Python<'py>,
    polar: usize,
    azimuthal: usize,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>, Array<'py, f64>)> {
    let (theta, phi, weight) = multipole::angular_grid(polar, azimuthal).map_err(error)?;
    let n = theta.len();
    Ok((
        array(py, &[n], theta)?,
        array(py, &[n], phi)?,
        array(py, &[n], weight)?,
    ))
}

#[pyfunction]
fn paw_smooth_waves<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    l: usize,
    r: PyReadonlyArray1<'py, f64>,
    r_cut: f64,
    ae: PyReadonlyArray2<'py, f64>,
    coefficients: PyReadonlyArray2<'py, f64>,
    wavevectors: PyReadonlyArray2<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let waves = ae.shape()[0];
    let (r, _a) = copy(&context.inner, &r)?;
    let (ae, _b) = copy(&context.inner, &ae)?;
    let (coefficients, _c) = copy(&context.inner, &coefficients)?;
    let (wavevectors, _d) = copy(&context.inner, &wavevectors)?;
    let out = py
        .detach(|| {
            pairs::smooth_waves(
                &r,
                &ChannelWaves {
                    l,
                    r_cut,
                    ae: &ae,
                    coefficients: &coefficients,
                    wavevectors: &wavevectors,
                    waves,
                },
            )
        })
        .map_err(error)?;
    array(py, &[waves, r.len()], out)
}

#[pyfunction]
fn paw_radial_moments<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    l: usize,
    r: PyReadonlyArray1<'py, f64>,
    r_cut: f64,
    ae1: PyReadonlyArray2<'py, f64>,
    smooth1: PyReadonlyArray2<'py, f64>,
    ae2: PyReadonlyArray2<'py, f64>,
    smooth2: PyReadonlyArray2<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let waves = [ae1.shape()[0], ae2.shape()[0]];
    let (r, _a) = copy(&context.inner, &r)?;
    let (ae1, _b) = copy(&context.inner, &ae1)?;
    let (s1, _c) = copy(&context.inner, &smooth1)?;
    let (ae2, _d) = copy(&context.inner, &ae2)?;
    let (s2, _e) = copy(&context.inner, &smooth2)?;
    let out = py
        .detach(|| {
            pairs::radial_moments(
                &Pair {
                    r: &r,
                    r_cut,
                    ae: [&ae1, &ae2],
                    smooth: [&s1, &s2],
                    waves,
                },
                l,
            )
        })
        .map_err(error)?;
    array(py, &[waves[0], waves[1]], out)
}

#[pyfunction]
fn paw_radial_remainder<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    l: usize,
    r: PyReadonlyArray1<'py, f64>,
    r_cut: f64,
    ae1: PyReadonlyArray2<'py, f64>,
    smooth1: PyReadonlyArray2<'py, f64>,
    ae2: PyReadonlyArray2<'py, f64>,
    smooth2: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    delta: PyReadonlyArray2<'py, f64>,
    shape: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let waves = [ae1.shape()[0], ae2.shape()[0]];
    let (r, _a) = copy(&context.inner, &r)?;
    let (ae1, _b) = copy(&context.inner, &ae1)?;
    let (s1, _c) = copy(&context.inner, &smooth1)?;
    let (ae2, _d) = copy(&context.inner, &ae2)?;
    let (s2, _e) = copy(&context.inner, &smooth2)?;
    let (q, _f) = copy(&context.inner, &q)?;
    let (delta, _g) = copy(&context.inner, &delta)?;
    let (shape, _h) = copy(&context.inner, &shape)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| {
            pairs::radial_remainder(
                ctx,
                &Pair {
                    r: &r,
                    r_cut,
                    ae: [&ae1, &ae2],
                    smooth: [&s1, &s2],
                    waves,
                },
                l,
                &q,
                &delta,
                &shape,
            )
        })
        .map_err(error)?;
    array(py, &[q.len(), waves[0], waves[1]], out)
}

#[pyfunction]
fn paw_crossover_radius(
    context: PyRef<'_, PyContext>,
    r: PyReadonlyArray1<'_, f64>,
    core: PyReadonlyArray1<'_, f64>,
    valence: PyReadonlyArray1<'_, f64>,
    ratio: f64,
) -> PyResult<Option<f64>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (core, _b) = copy(&context.inner, &core)?;
    let (valence, _c) = copy(&context.inner, &valence)?;
    core_correction::crossover_radius(&r, &core, &valence, ratio).map_err(error)
}

#[pyfunction]
fn paw_smooth_core_density<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    core: PyReadonlyArray1<'py, f64>,
    r_nlcc: f64,
) -> PyResult<(Array<'py, f64>, f64, f64)> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (core, _b) = copy(&context.inner, &core)?;
    let (smooth, a, b) = py
        .detach(|| core_correction::smooth_core_density(&r, &core, r_nlcc))
        .map_err(error)?;
    Ok((array(py, &[smooth.len()], smooth)?, a, b))
}

#[pyfunction]
#[pyo3(signature=(context, r, core, valence, r_nlcc=None, ratio=1.0))]
fn paw_partial_core_density<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    core: PyReadonlyArray1<'py, f64>,
    valence: PyReadonlyArray1<'py, f64>,
    r_nlcc: Option<f64>,
    ratio: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (core, _b) = copy(&context.inner, &core)?;
    let (valence, _c) = copy(&context.inner, &valence)?;
    let out = py
        .detach(|| core_correction::partial_core_density(&r, &core, &valence, r_nlcc, ratio))
        .map_err(error)?;
    let result = PyDict::new(py);
    result.set_item("density", array(py, &[out.density.len()], out.density)?)?;
    result.set_item("r_nlcc", out.r_nlcc)?;
    result.set_item("amplitude", out.amplitude)?;
    result.set_item("wavevector", out.wavevector)?;
    result.set_item("core_electrons", out.core_electrons)?;
    result.set_item("partial_core_electrons", out.partial_core_electrons)?;
    Ok(result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_shape_function, m)?)?;
    m.add_function(wrap_pyfunction!(paw_shape_potential, m)?)?;
    m.add_function(wrap_pyfunction!(paw_multipole_range, m)?)?;
    m.add_function(wrap_pyfunction!(paw_gaunt, m)?)?;
    m.add_function(wrap_pyfunction!(paw_angular_grid, m)?)?;
    m.add_function(wrap_pyfunction!(paw_smooth_waves, m)?)?;
    m.add_function(wrap_pyfunction!(paw_radial_moments, m)?)?;
    m.add_function(wrap_pyfunction!(paw_radial_remainder, m)?)?;
    m.add_function(wrap_pyfunction!(paw_crossover_radius, m)?)?;
    m.add_function(wrap_pyfunction!(paw_smooth_core_density, m)?)?;
    m.add_function(wrap_pyfunction!(paw_partial_core_density, m)?)?;
    Ok(())
}
