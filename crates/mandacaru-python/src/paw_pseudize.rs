//! Bindings of the PAW smooth-wave construction and its checks. Arrays are
//! copied in and returned owned; the numerics are in `mandacaru_core::paw`.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::{
    models::{numerov, relativity::Treatment},
    paw::{
        checks,
        pseudize::{self, Equation, NormCondition, PseudoInput},
        pseudize_tables as tables, smooth,
    },
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

fn equation(treatment: &str, kappa: Option<i32>, z: f64) -> PyResult<Equation> {
    Ok(Equation {
        treatment: Treatment::parse(treatment).map_err(error)?,
        kappa,
        z,
    })
}

#[pyfunction]
fn paw_bessel_table<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    l: usize,
    q: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let (q, _a) = copy(&context.inner, &q)?;
    let (r, _b) = copy(&context.inner, &r)?;
    let out = py.detach(|| tables::bessel_table(l, &q, &r));
    array(py, &[q.len(), r.len()], out)
}

#[pyfunction]
fn paw_bessel_derivatives<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    l: usize,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let (x, _a) = copy(&context.inner, &x)?;
    let n = x.len();
    let rows = x.iter().map(|&x| smooth::bessel(l, x)).collect::<Vec<_>>();
    array(
        py,
        &[4, n],
        (0..4)
            .flat_map(|k| rows.iter().map(move |row| row[k]))
            .collect(),
    )
}

#[pyfunction]
fn paw_bessel_wavevectors(l: usize, r_cut: f64, count: usize) -> PyResult<Vec<f64>> {
    smooth::wavevectors(l, r_cut, count).map_err(error)
}

#[pyfunction]
fn paw_resample<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    r_new: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (values, _b) = copy(&context.inner, &values)?;
    let (r_new, _c) = copy(&context.inner, &r_new)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| tables::resample(ctx, &r, &values, &r_new))
        .map_err(error)?;
    array(py, &[out.len()], out)
}

#[pyfunction]
fn paw_inner_grid<'py>(py: Python<'py>, r_cut: f64) -> PyResult<Array<'py, f64>> {
    let grid = tables::inner_grid(r_cut);
    array(py, &[grid.len()], grid)
}

#[pyfunction]
fn paw_snap(r: PyReadonlyArray1<'_, f64>, radius: f64) -> PyResult<f64> {
    Ok(pseudize::snap(r.as_slice()?, radius))
}

#[pyfunction]
#[pyo3(signature=(context, r, u, v, l, energy, r_cut, treatment="none", kappa=None, z_eff=0.0))]
fn paw_matching_targets<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    l: usize,
    energy: f64,
    r_cut: f64,
    treatment: &str,
    kappa: Option<i32>,
    z_eff: f64,
) -> PyResult<Array<'py, f64>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (u, _b) = copy(&context.inner, &u)?;
    let (v, _c) = copy(&context.inner, &v)?;
    let eq = equation(treatment, kappa, z_eff)?;
    let out = py
        .detach(|| pseudize::matching_targets(&r, &u, &v, l, energy, r_cut, &eq))
        .map_err(error)?;
    array(py, &[4], out.to_vec())
}

fn matrix<'py>(py: Python<'py>, m: &nalgebra::DMatrix<f64>) -> PyResult<Array<'py, f64>> {
    array(
        py,
        &[m.nrows(), m.ncols()],
        (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
            .collect(),
    )
}

#[pyfunction]
#[pyo3(signature=(context, r, waves, energies, r_cut, v, treatment="none", kappa=None, z_eff=0.0))]
fn paw_norm_targets<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    waves: PyReadonlyArray2<'py, f64>,
    energies: Vec<f64>,
    r_cut: f64,
    v: PyReadonlyArray1<'py, f64>,
    treatment: &str,
    kappa: Option<i32>,
    z_eff: f64,
) -> PyResult<Array<'py, f64>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (waves, _b) = copy(&context.inner, &waves)?;
    let (v, _c) = copy(&context.inner, &v)?;
    let eq = equation(treatment, kappa, z_eff)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| pseudize::norm_targets(ctx, &r, &waves, &energies, r_cut, &v, &eq))
        .map_err(error)?;
    matrix(py, &out)
}

#[pyfunction]
fn paw_inner_overlaps<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    waves: PyReadonlyArray2<'py, f64>,
    r_cut: f64,
) -> PyResult<Array<'py, f64>> {
    let count = waves.shape()[0];
    let (r, _a) = copy(&context.inner, &r)?;
    let (waves, _b) = copy(&context.inner, &waves)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| pseudize::inner_overlaps(ctx, &r, &waves, count, r_cut))
        .map_err(error)?;
    matrix(py, &out)
}

/// `norm_factor=None` is the free (unconstrained) minimization.
#[pyfunction]
#[pyo3(signature=(context, r, v_ae, l, waves, energies, r_cut, q_cut, n_bessel, norm_factor=Some(1.0), treatment="none", kappa=None, z_eff=0.0))]
fn paw_optimize_pseudo_waves<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    v_ae: PyReadonlyArray1<'py, f64>,
    l: usize,
    waves: PyReadonlyArray2<'py, f64>,
    energies: Vec<f64>,
    r_cut: f64,
    q_cut: f64,
    n_bessel: usize,
    norm_factor: Option<f64>,
    treatment: &str,
    kappa: Option<i32>,
    z_eff: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if waves.shape()[0] != energies.len() {
        return Err(PyRuntimeError::new_err(
            "one energy per reference wave is required",
        ));
    }
    let (r, _a) = copy(&context.inner, &r)?;
    let (v_ae, _b) = copy(&context.inner, &v_ae)?;
    let (waves, _c) = copy(&context.inner, &waves)?;
    let equation = equation(treatment, kappa, z_eff)?;
    let norm = norm_factor.map_or(NormCondition::Free, NormCondition::Factor);
    let ctx = &context.inner;
    let out = py
        .detach(|| {
            pseudize::optimize(
                ctx,
                &PseudoInput {
                    r: &r,
                    v_ae: &v_ae,
                    l,
                    waves: &waves,
                    energies: &energies,
                    r_cut,
                    q_cut,
                    n_bessel,
                    norm,
                    equation,
                },
            )
        })
        .map_err(error)?;
    let result = PyDict::new(py);
    result.set_item("r_cut", out.r_cut)?;
    result.set_item(
        "wavevectors",
        array(py, &[out.wavevectors.len()], out.wavevectors.clone())?,
    )?;
    result.set_item("coefficients", matrix(py, &out.coefficients)?)?;
    result.set_item("residual_kinetic", out.residuals)?;
    result.set_item("norms", matrix(py, &out.norms)?)?;
    result.set_item("achieved", matrix(py, &out.achieved)?)?;
    result.set_item("targets", matrix(py, &out.targets)?)?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature=(context, r, v, l, energy, z_eff, treatment="none", kappa=None))]
fn paw_scattering_wave<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    l: usize,
    energy: f64,
    z_eff: f64,
    treatment: &str,
    kappa: Option<i32>,
) -> PyResult<Array<'py, f64>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (v, _b) = copy(&context.inner, &v)?;
    let treatment = Treatment::parse(treatment).map_err(error)?;
    let ctx = &context.inner;
    let out = py
        .detach(|| numerov::scattering_wave(ctx, &r, &v, l, energy, z_eff, treatment, kappa))
        .map_err(error)?;
    array(py, &[out.len()], out)
}

/// The Numerov bound state `(u, energy)` (`radial=True`: `R = u / r`), or
/// `RuntimeError` when no bracket exists.
#[pyfunction]
#[pyo3(signature=(context, r, v, l, energy_guess, z_eff, window=5e-3, treatment="none", kappa=None, radial=false))]
fn paw_bound_state<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    l: usize,
    energy_guess: f64,
    z_eff: f64,
    window: f64,
    treatment: &str,
    kappa: Option<i32>,
    radial: bool,
) -> PyResult<(Array<'py, f64>, f64)> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (v, _b) = copy(&context.inner, &v)?;
    let eq = equation(treatment, kappa, z_eff)?;
    let ctx = &context.inner;
    let (wave, energy) = py
        .detach(|| {
            if radial {
                pseudize::bound_reference(ctx, &r, &v, l, energy_guess, &eq, Some(window))
            } else {
                numerov::bound_state(
                    ctx,
                    &r,
                    &v,
                    l,
                    energy_guess,
                    z_eff,
                    window,
                    eq.treatment,
                    kappa,
                )?
                .ok_or_else(|| {
                    mandacaru_core::Error::Invalid(format!(
                        "no Numerov bound state near {energy_guess:.6} Ha for l={l}"
                    ))
                })
            }
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((array(py, &[wave.len()], wave)?, energy))
}

/// The scattering wave at `energy`, normalized to unit weight inside `radius`
/// (`int u^2 dr = 1` there), as the radial function `R = u / r`.
#[pyfunction]
#[pyo3(signature=(context, r, v, l, energy, z_eff, radius, treatment="none", kappa=None))]
fn paw_reference_waves<'py>(
    context: PyRef<'_, PyContext>,
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    l: usize,
    energy: f64,
    z_eff: f64,
    radius: f64,
    treatment: &str,
    kappa: Option<i32>,
) -> PyResult<Array<'py, f64>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (v, _b) = copy(&context.inner, &v)?;
    let eq = equation(treatment, kappa, z_eff)?;
    let ctx = &context.inner;
    let wave = py
        .detach(|| pseudize::scattering_reference(ctx, &r, &v, l, energy, &eq, radius))
        .map_err(error)?;
    array(py, &[wave.len()], wave)
}

#[pyfunction]
fn paw_constrained_minimum<'py>(
    py: Python<'py>,
    k_matrix: PyReadonlyArray2<'py, f64>,
    k_vector: PyReadonlyArray1<'py, f64>,
    constraints: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray1<'py, f64>,
    metric: PyReadonlyArray2<'py, f64>,
    norm: f64,
) -> PyResult<Array<'py, f64>> {
    let dense = |a: &PyReadonlyArray2<'py, f64>| -> PyResult<nalgebra::DMatrix<f64>> {
        Ok(nalgebra::DMatrix::from_row_slice(
            a.shape()[0],
            a.shape()[1],
            a.as_slice()?,
        ))
    };
    let vector = |a: &PyReadonlyArray1<'py, f64>| -> PyResult<nalgebra::DVector<f64>> {
        Ok(nalgebra::DVector::from_column_slice(a.as_slice()?))
    };
    let (k, g, a) = (dense(&k_matrix)?, dense(&metric)?, dense(&constraints)?);
    let (linear, b) = (vector(&k_vector)?, vector(&targets)?);
    let out = py
        .detach(|| mandacaru_core::math::constrained::minimum(&k, &linear, &a, &b, &g, norm))
        .map_err(error)?;
    array(py, &[out.len()], out.iter().copied().collect())
}

#[pyfunction]
fn paw_log_derivative_of_u(
    r: PyReadonlyArray1<'_, f64>,
    u: PyReadonlyArray1<'_, f64>,
    r_cut: f64,
) -> PyResult<f64> {
    checks::log_derivative_of_u(r.as_slice()?, u.as_slice()?, r_cut).map_err(error)
}

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

#[pyfunction]
fn paw_node_count(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    r: PyReadonlyArray1<'_, f64>,
    v: PyReadonlyArray1<'_, f64>,
    l: usize,
    energy: f64,
    z_eff: f64,
    treatment: &str,
) -> PyResult<(usize, f64)> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (v, _b) = copy(&context.inner, &v)?;
    let treatment = Treatment::parse(treatment).map_err(error)?;
    let ctx = &context.inner;
    py.detach(|| checks::node_count(ctx, &r, &v, l, energy, z_eff, treatment))
        .map_err(error)
}

/// Levels in `(lower, upper]`; `lower=None` is the potential minimum less one Hartree.
#[pyfunction]
#[pyo3(signature=(context, r, v, l, lower=None, upper=0.0))]
fn paw_bound_levels(
    context: PyRef<'_, PyContext>,
    py: Python<'_>,
    r: PyReadonlyArray1<'_, f64>,
    v: PyReadonlyArray1<'_, f64>,
    l: usize,
    lower: Option<f64>,
    upper: f64,
) -> PyResult<Vec<f64>> {
    let (r, _a) = copy(&context.inner, &r)?;
    let (v, _b) = copy(&context.inner, &v)?;
    py.detach(|| checks::bound_levels(&r, &v, l, lower, upper))
        .map_err(error)
}

#[pyfunction]
fn paw_peak_radius(
    r: PyReadonlyArray1<'_, f64>,
    radial: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    checks::peak_radius(r.as_slice()?, radial.as_slice()?).map_err(error)
}

/// `f = 2[V + l(l+1)/2r^2 - E]` on the grid with the origin prepended.
#[pyfunction]
fn paw_radial_f<'py>(
    py: Python<'py>,
    r0: PyReadonlyArray1<'py, f64>,
    v0: PyReadonlyArray1<'py, f64>,
    l: usize,
    energy: f64,
) -> PyResult<Array<'py, f64>> {
    let f = numerov::radial_f(r0.as_slice()?, v0.as_slice()?, l, energy);
    array(py, &[f.len()], f)
}

#[pyfunction]
fn paw_window_offsets<'py>(py: Python<'py>, width: f64, step: f64) -> PyResult<Array<'py, f64>> {
    let out = checks::window_offsets(width, step);
    array(py, &[out.len()], out)
}

#[pyfunction]
fn paw_phase_summary(
    pseudo: PyReadonlyArray1<'_, f64>,
    all_electron: PyReadonlyArray1<'_, f64>,
    offsets: PyReadonlyArray1<'_, f64>,
    window: f64,
) -> PyResult<(f64, f64, Option<usize>)> {
    checks::phase_summary(
        pseudo.as_slice()?,
        all_electron.as_slice()?,
        offsets.as_slice()?,
        window,
    )
    .map_err(error)
}

#[pyfunction]
fn paw_interpolate_linear<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    nodes: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    let out = checks::interpolate_linear(x.as_slice()?, nodes.as_slice()?, values.as_slice()?)
        .map_err(error)?;
    array(py, &[out.len()], out)
}

#[pyfunction]
fn paw_spurious_nodes(
    l: usize,
    r_cut: f64,
    r: PyReadonlyArray1<'_, f64>,
    ae: PyReadonlyArray2<'_, f64>,
    wavevectors: PyReadonlyArray2<'_, f64>,
    coefficients: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<usize>> {
    checks::spurious_nodes(
        l,
        r_cut,
        r.as_slice()?,
        ae.as_slice()?,
        wavevectors.as_slice()?,
        coefficients.as_slice()?,
        ae.shape()[0],
    )
    .map_err(error)
}

#[pyfunction]
fn paw_spectrum_extent(
    r: PyReadonlyArray1<'_, f64>,
    wave: PyReadonlyArray1<'_, f64>,
    floor: f64,
    lower: f64,
    upper: f64,
) -> PyResult<f64> {
    checks::spectrum_extent(r.as_slice()?, wave.as_slice()?, floor, lower, upper).map_err(error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paw_bessel_table, m)?)?;
    m.add_function(wrap_pyfunction!(paw_bessel_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(paw_bessel_wavevectors, m)?)?;
    m.add_function(wrap_pyfunction!(paw_resample, m)?)?;
    m.add_function(wrap_pyfunction!(paw_inner_grid, m)?)?;
    m.add_function(wrap_pyfunction!(paw_snap, m)?)?;
    m.add_function(wrap_pyfunction!(paw_matching_targets, m)?)?;
    m.add_function(wrap_pyfunction!(paw_norm_targets, m)?)?;
    m.add_function(wrap_pyfunction!(paw_inner_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(paw_optimize_pseudo_waves, m)?)?;
    m.add_function(wrap_pyfunction!(paw_constrained_minimum, m)?)?;
    m.add_function(wrap_pyfunction!(paw_reference_waves, m)?)?;
    m.add_function(wrap_pyfunction!(paw_scattering_wave, m)?)?;
    m.add_function(wrap_pyfunction!(paw_bound_state, m)?)?;
    m.add_function(wrap_pyfunction!(paw_log_derivative_of_u, m)?)?;
    m.add_function(wrap_pyfunction!(paw_log_derivatives_ae, m)?)?;
    m.add_function(wrap_pyfunction!(paw_node_count, m)?)?;
    m.add_function(wrap_pyfunction!(paw_bound_levels, m)?)?;
    m.add_function(wrap_pyfunction!(paw_peak_radius, m)?)?;
    m.add_function(wrap_pyfunction!(paw_radial_f, m)?)?;
    m.add_function(wrap_pyfunction!(paw_window_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(paw_phase_summary, m)?)?;
    m.add_function(wrap_pyfunction!(paw_interpolate_linear, m)?)?;
    m.add_function(wrap_pyfunction!(paw_spurious_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(paw_spectrum_extent, m)?)?;
    Ok(())
}
