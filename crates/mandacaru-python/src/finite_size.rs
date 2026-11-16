//! Python bindings for finite-size corrections (KZK, MPC, CCMH).
use crate::arrays::{array, error, layout, Array};
use crate::context::PyContext;
use mandacaru_core::math::wigner_seitz::{
    minimum_image_kernel, voxel_self_potential as core_voxel_self_potential,
};
use mandacaru_core::math::SpatialGrid;
use mandacaru_core::periodic::finite_size::{
    ccmh_extrapolate as core_ccmh_extrapolate, kzk as core_kzk, kzk_flexible, CcmhShell,
};
use nalgebra::Matrix3;
use numpy::{
    Complex64, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Helper functions for scalar/array operations
// ---------------------------------------------------------------------------

fn apply_scalar_or_array<'py, F>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    f: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Fn(f64) -> f64,
{
    if let Ok(v) = input.extract::<f64>() {
        return Ok(f(v).into_pyobject(py)?.into_any());
    }
    let arr = input.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    layout(&arr)?;
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<f64> = arr.as_slice()?.iter().copied().map(f).collect();
    let out = array(py, &shape, data)?;
    Ok(out.into_any())
}

fn apply_scalar_or_array_pair<'py, F>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    f: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: Fn(f64) -> (f64, f64),
{
    if let Ok(v) = input.extract::<f64>() {
        let (a, b) = f(v);
        return Ok((a, b).into_pyobject(py)?.into_any());
    }
    let arr = input.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    layout(&arr)?;
    let shape: Vec<usize> = arr.shape().to_vec();
    let slice = arr.as_slice()?;
    let mut data1 = Vec::with_capacity(slice.len());
    let mut data2 = Vec::with_capacity(slice.len());
    for &x in slice {
        let (a, b) = f(x);
        data1.push(a);
        data2.push(b);
    }
    let out1 = array(py, &shape, data1)?;
    let out2 = array(py, &shape, data2)?;
    Ok((out1, out2).into_pyobject(py)?.into_any())
}

// ---------------------------------------------------------------------------
// KZK functions
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn kzk_a0() -> f64 {
    core_kzk::a0()
}

#[pyfunction]
pub fn kzk_effective_length(volume: f64) -> f64 {
    core_kzk::effective_length(volume)
}

#[pyfunction]
pub fn kzk_rs_at(electrons: f64, length: f64) -> f64 {
    core_kzk::rs_at(electrons, length)
}

#[pyfunction]
pub fn kzk_pz_correlation_rydberg<'py>(
    py: Python<'py>,
    rs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    apply_scalar_or_array_pair(py, rs, core_kzk::pz_correlation_rydberg)
}

#[pyfunction]
pub fn kzk_g<'py>(py: Python<'py>, rs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    apply_scalar_or_array(py, rs, core_kzk::g_function)
}

#[pyfunction]
pub fn kzk_dg<'py>(py: Python<'py>, rs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    apply_scalar_or_array(py, rs, core_kzk::g_derivative)
}

#[pyfunction]
pub fn kzk_fs_exchange<'py>(
    py: Python<'py>,
    rs: &Bound<'py, PyAny>,
    length: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let fit = core_kzk::Fit::new(length).map_err(error)?;
    apply_scalar_or_array(py, rs, move |r| fit.exchange(r))
}

#[pyfunction]
pub fn kzk_fs_correlation<'py>(
    py: Python<'py>,
    rs: &Bound<'py, PyAny>,
    length: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let fit = core_kzk::Fit::new(length).map_err(error)?;
    apply_scalar_or_array(py, rs, move |r| fit.correlation(r))
}

#[pyfunction]
pub fn kzk_fs_exchange_correlation<'py>(
    py: Python<'py>,
    rs: &Bound<'py, PyAny>,
    length: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let fit = core_kzk::Fit::new(length).map_err(error)?;
    apply_scalar_or_array(py, rs, move |r| fit.exchange_correlation(r))
}

#[pyfunction]
pub fn kzk_infinite_exchange_correlation<'py>(
    py: Python<'py>,
    rs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    apply_scalar_or_array(py, rs, core_kzk::infinite_exchange_correlation)
}

#[pyfunction]
pub fn kzk_intermediate_cubic(length: f64) -> PyResult<(f64, f64, f64, f64)> {
    let fit = core_kzk::Fit::new(length).map_err(error)?;
    Ok((fit.alpha, fit.beta, fit.gamma_h, fit.gamma_l))
}

#[pyfunction]
pub fn kzk_leading_term_cancellation<'py>(
    rs: PyReadonlyArrayDyn<'py, f64>,
    length: f64,
) -> PyResult<f64> {
    layout(&rs)?;
    core_kzk::leading_term_cancellation(rs.as_slice()?, length).map_err(error)
}

#[pyfunction]
#[pyo3(signature = (density, dv, cell, n_cells=1, floor=1e-12, max_vacuum_fraction=None))]
pub fn kzk_correction<'py>(
    py: Python<'py>,
    density: PyReadonlyArrayDyn<'py, f64>,
    dv: f64,
    cell: PyReadonlyArray2<'py, f64>,
    n_cells: usize,
    floor: f64,
    max_vacuum_fraction: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    layout(&density)?;
    layout(&cell)?;

    let cell_shape = cell.shape();
    if cell_shape != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cell must have shape (3, 3), got {:?}",
            cell_shape
        )));
    }

    let cell_slice = cell.as_slice()?;
    let cell_matrix = [
        [cell_slice[0], cell_slice[1], cell_slice[2]],
        [cell_slice[3], cell_slice[4], cell_slice[5]],
        [cell_slice[6], cell_slice[7], cell_slice[8]],
    ];

    let density_slice = density.as_slice()?;
    let result = kzk_flexible(
        density_slice,
        dv,
        &cell_matrix,
        n_cells,
        floor,
        false,
        max_vacuum_fraction,
    )
    .map_err(error)?;

    let dict = PyDict::new(py);
    dict.set_item("correction", result.correction)?;
    dict.set_item("correction_per_cell", result.correction_per_cell())?;
    dict.set_item("xc_infinite", result.xc_infinite)?;
    dict.set_item("xc_finite", result.xc_finite)?;
    dict.set_item("length", result.length)?;
    dict.set_item("volume", result.scope.volume)?;
    dict.set_item("n_cells", result.scope.n_cells)?;
    dict.set_item("n_electrons", result.n_electrons)?;
    dict.set_item("electrons_below_floor", result.electrons_below_floor)?;
    dict.set_item("electrons_beyond_gamma_l", result.electrons_beyond_gamma_l)?;
    dict.set_item("cubic_deviation", result.cubic_deviation)?;
    dict.set_item("rs_range", (result.rs_range.0, result.rs_range.1))?;

    Ok(dict)
}

// ---------------------------------------------------------------------------
// CCMH functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (k_shells, s_shells, volume, n_cells=1, n_electrons=None))]
pub fn ccmh_extrapolate<'py>(
    py: Python<'py>,
    k_shells: PyReadonlyArray1<'py, f64>,
    s_shells: PyReadonlyArray1<'py, f64>,
    volume: f64,
    n_cells: usize,
    n_electrons: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    layout(&k_shells)?;
    layout(&s_shells)?;

    let k_slice = k_shells.as_slice()?;
    let s_slice = s_shells.as_slice()?;
    if k_slice.len() != s_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "k_shells length ({}) must match s_shells length ({})",
            k_slice.len(),
            s_slice.len()
        )));
    }

    let ne = n_electrons.unwrap_or(1.0);
    let shells: Vec<CcmhShell> = k_slice
        .iter()
        .copied()
        .zip(s_slice.iter().copied())
        .map(|(k, s)| CcmhShell { k, s, count: 1 })
        .collect();

    let result = core_ccmh_extrapolate(&shells, volume, n_cells, ne).map_err(error)?;

    let dict = PyDict::new(py);
    dict.set_item("correction", result.correction)?;
    dict.set_item("correction_per_electron", result.correction_per_electron)?;
    dict.set_item("correction_per_cell", result.correction_per_cell())?;
    dict.set_item("limit", result.limit)?;
    dict.set_item("quadratic_spread", result.quadratic_spread)?;
    dict.set_item("reliable", result.reliable)?;
    dict.set_item("rpa_limit", result.rpa_limit)?;
    dict.set_item("rpa_correction", result.rpa_correction)?;
    dict.set_item("plasma_frequency", result.plasma_frequency)?;
    dict.set_item("density", result.density)?;
    dict.set_item("n_electrons", result.n_electrons)?;
    dict.set_item("n_cells", result.scope.n_cells)?;
    dict.set_item("volume", result.scope.volume)?;

    Ok(dict)
}

#[pyfunction]
pub fn ccmh_shells<'py>(
    py: Python<'py>,
    magnitudes: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    count: usize,
) -> PyResult<(Array<'py, f64>, Array<'py, f64>)> {
    layout(&magnitudes)?;
    layout(&values)?;
    let (k, v) = mandacaru_core::periodic::finite_size::ccmh::shell_means(
        magnitudes.as_slice()?,
        values.as_slice()?,
        count,
    );
    let n = k.len();
    Ok((array(py, &[n], k)?, array(py, &[n], v)?))
}

#[pyfunction]
pub fn ccmh_reciprocal_vectors<'py>(
    py: Python<'py>,
    cell: PyReadonlyArray2<'py, f64>,
    limit: i64,
) -> PyResult<(Array<'py, i64>, Array<'py, f64>)> {
    layout(&cell)?;
    if cell.shape() != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err("cell must be (3, 3)"));
    }
    let c: [f64; 9] = cell.as_slice()?.try_into().expect("a (3, 3) cell");
    let (triples, kvectors) =
        mandacaru_core::periodic::finite_size::ccmh::reciprocal_triples(&c, limit).map_err(error)?;
    let n = triples.len() / 3;
    Ok((array(py, &[n, 3], triples)?, array(py, &[n, 3], kvectors)?))
}

#[pyfunction]
pub fn compute_structure_factor<'py>(
    context: PyRef<'_, super::PyContext>,
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
    shape: (usize, usize, usize),
    dv: f64,
    cell: PyReadonlyArray2<'py, f64>,
    limit: i64,
    a_matrix: PyReadonlyArray2<'py, Complex64>,
    d_matrix: PyReadonlyArray2<'py, f64>,
    gamma_matrix: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    layout(&psi)?;
    layout(&cell)?;
    layout(&a_matrix)?;
    layout(&d_matrix)?;
    layout(&gamma_matrix)?;

    let n_ao = psi.shape()[0];
    let n_grid = psi.shape()[1];
    let (nx, ny, nz) = shape;
    if n_grid != nx * ny * nz {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "psi ngrid {} != shape product {}",
            n_grid,
            nx * ny * nz
        )));
    }

    let a_rows = a_matrix.shape()[0];
    let a_cols = a_matrix.shape()[1];
    if a_rows != n_ao {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "a_matrix rows {} != n_ao {}",
            a_rows, n_ao
        )));
    }
    let d_slice = d_matrix.as_slice()?;
    if d_slice.len() != a_cols * a_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "d_matrix length {} != a_cols^2 {}",
            d_slice.len(),
            a_cols * a_cols
        )));
    }
    let gamma_slice = gamma_matrix.as_slice()?;
    if gamma_slice.len() != a_cols * a_cols * a_cols * a_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gamma_matrix length {} != a_cols^4 {}",
            gamma_slice.len(),
            a_cols.pow(4)
        )));
    }

    let (psi, cell, a) = (psi.as_slice()?.to_vec(), cell.as_slice()?.to_vec(), a_matrix.as_slice()?.to_vec());
    let (d, gamma) = (d_slice.to_vec(), gamma_slice.to_vec());
    let ctx = &context.inner;
    let out = py
        .detach(|| mandacaru_core::periodic::finite_size::structure_factor(
            ctx, &psi, [nx, ny, nz], dv, &cell, limit, &a, a_cols, &d, &gamma))
        .map_err(error)?;
    let n_triples = out.values.len();
    let dict = PyDict::new(py);
    dict.set_item("kvectors", array(py, &[n_triples, 3], out.kvectors)?)?;
    dict.set_item("values", array(py, &[n_triples], out.values)?)?;
    dict.set_item("magnitudes", array(py, &[n_triples], out.magnitudes)?)?;
    dict.set_item("n_electrons", out.n_electrons)?;
    dict.set_item("overlap_error", out.overlap_error)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// MPC functions
// ---------------------------------------------------------------------------


#[pyfunction]
pub fn voxel_self_potential(step: PyReadonlyArray2<f64>) -> PyResult<f64> {
    layout(&step)?;
    let s = step.shape();
    if s != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "step must be a 3x3 matrix, got {:?}",
            s
        )));
    }
    let slice = step.as_slice()?;
    let step_matrix = Matrix3::new(
        slice[0], slice[1], slice[2],
        slice[3], slice[4], slice[5],
        slice[6], slice[7], slice[8],
    );
    core_voxel_self_potential(&step_matrix).map_err(error)
}

#[pyfunction]
#[pyo3(signature = (shape, step, shells=2, _context=None))]
pub fn wigner_seitz_kernel<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize),
    step: PyReadonlyArray2<'py, f64>,
    shells: usize,
    _context: Option<&Bound<'py, PyContext>>,
) -> PyResult<Array<'py, f64>> {
    layout(&step)?;
    let s = step.shape();
    if s != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "step must be a 3x3 matrix, got {:?}",
            s
        )));
    }
    let slice = step.as_slice()?;
    let step_array = [
        slice[0], slice[1], slice[2],
        slice[3], slice[4], slice[5],
        slice[6], slice[7], slice[8],
    ];

    let ctx = match _context {
        Some(c) => c.borrow().inner.clone(),
        None => mandacaru_core::Context::new(None, 64 << 20).map_err(error)?,
    };

    let grid = SpatialGrid::new(
        [shape.0, shape.1, shape.2],
        [0.0; 3],
        step_array,
        true,
    )
    .map_err(error)?;

    let kernel_buf = minimum_image_kernel(&ctx, &grid, shells).map_err(error)?;
    let data = kernel_buf.as_slice().to_vec();

    array(py, &[shape.0, shape.1, shape.2], data)
}

#[pyfunction]
pub fn periodic_poisson_kernel<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize),
    step: PyReadonlyArray2<'py, f64>,
    dv: f64,
) -> PyResult<Array<'py, f64>> {
    layout(&step)?;
    if step.shape() != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err("step must be 3x3"));
    }
    let s: [f64; 9] = step.as_slice()?.try_into().expect("a (3, 3) step");
    let shape = [shape.0, shape.1, shape.2];
    let k = mandacaru_core::periodic::finite_size::mpc::periodic_poisson_kernel(shape, &s, dv)
        .map_err(error)?;
    array(py, &shape, k)
}

#[pyfunction]
pub fn truncated_coulomb_kernel<'py>(
    py: Python<'py>,
    real_kernel: PyReadonlyArrayDyn<'py, f64>,
    shape: (usize, usize, usize),
) -> PyResult<Array<'py, f64>> {
    layout(&real_kernel)?;
    let shape = [shape.0, shape.1, shape.2];
    let ctx = mandacaru_core::Context::new(None, 64 << 20).map_err(error)?;
    let out = mandacaru_core::periodic::finite_size::mpc::truncated_coulomb_kernel(
        &ctx,
        real_kernel.as_slice()?,
        shape,
    )
    .map_err(error)?;
    array(py, &shape, out)
}

#[pyfunction]
pub fn poisson_solve_stack<'py>(
    py: Python<'py>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    shape: (usize, usize, usize),
    rho_stack: PyReadonlyArray2<'py, f64>,
    dv: f64,
) -> PyResult<Array<'py, f64>> {
    layout(&kernel)?;
    layout(&rho_stack)?;
    let shape = [shape.0, shape.1, shape.2];
    let n_grid = shape.iter().product::<usize>();
    if kernel.as_slice()?.len() != n_grid {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "kernel size does not match shape",
        ));
    }
    if rho_stack.shape()[1] != n_grid {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rho_stack columns do not match shape product",
        ));
    }
    let p_count = rho_stack.shape()[0];
    let ctx = mandacaru_core::Context::new(None, 64 << 20).map_err(error)?;
    let out = mandacaru_core::periodic::finite_size::mpc::poisson_solve_stack(
        &ctx,
        kernel.as_slice()?,
        shape,
        rho_stack.as_slice()?,
        dv,
    )
    .map_err(error)?;
    array(py, &[p_count, n_grid], out)
}

#[pyfunction]
pub fn wigner_seitz_metrics<'py>(
    kernel: PyReadonlyArrayDyn<'py, f64>,
    dv: f64,
) -> PyResult<(f64, f64)> {
    layout(&kernel)?;
    Ok(mandacaru_core::periodic::finite_size::mpc::wigner_seitz_metrics(kernel.as_slice()?, dv))
}

#[pyfunction]
pub fn mpc_hole_energies<'py>(
    eri_periodic: PyReadonlyArrayDyn<'py, f64>,
    eri_truncated: PyReadonlyArrayDyn<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    gamma: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(f64, f64, f64)> {
    layout(&eri_periodic)?;
    layout(&eri_truncated)?;
    layout(&d)?;
    layout(&gamma)?;

    let p_shape = eri_periodic.shape();
    let t_shape = eri_truncated.shape();
    let g_shape = gamma.shape();
    let d_shape = d.shape();

    if p_shape.len() != 4 || t_shape.len() != 4 || g_shape.len() != 4 || d_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid dimensions for mpc_hole_energies",
        ));
    }
    if p_shape != t_shape || p_shape != g_shape {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tensor shapes must match",
        ));
    }
    let m = p_shape[0];
    if p_shape != [m, m, m, m] || d_shape != [m, m] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tensors must be (M, M, M, M) and D must be (M, M)",
        ));
    }

    mandacaru_core::periodic::finite_size::mpc::hole_energies(
        eri_periodic.as_slice()?,
        eri_truncated.as_slice()?,
        d.as_slice()?,
        gamma.as_slice()?,
        m,
    )
    .map_err(error)
}

#[pyfunction]
pub fn spatial_rdms<'py>(
    py: Python<'py>,
    gamma: PyReadonlyArray2<'py, f64>,
    gamma2: Option<PyReadonlyArrayDyn<'py, f64>>,
    n_orbitals: usize,
) -> PyResult<(Array<'py, f64>, Option<Array<'py, f64>>)> {
    layout(&gamma)?;
    let m = n_orbitals;
    let g_shape = gamma.shape();
    if g_shape != [2 * m, 2 * m] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected a {}x{} one-body RDM, got {:?}",
            2 * m,
            2 * m,
            g_shape
        )));
    }
    if let Some(g2) = &gamma2 {
        layout(g2)?;
        if g2.shape() != [2 * m, 2 * m, 2 * m, 2 * m] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected a (2M, 2M, 2M, 2M) two-body RDM, got {:?}",
                g2.shape()
            )));
        }
    }
    let g2 = gamma2.as_ref().map(|g| g.as_slice()).transpose()?;
    let (d, g) = mandacaru_core::periodic::finite_size::two_body::spatial_rdms(
        gamma.as_slice()?,
        g2,
        m,
    )
    .map_err(error)?;
    let g = g.map(|g| array(py, &[m, m, m, m], g)).transpose()?;
    Ok((array(py, &[m, m], d)?, g))
}

#[pyfunction]
pub fn compute_pair_densities<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&psi)?;
    let (m, points) = (psi.shape()[0], psi.shape()[1]);
    let out =
        mandacaru_core::periodic::finite_size::two_body::pair_densities(psi.as_slice()?, m, points);
    array(py, &[m * m, points], out)
}

fn optional_real(a: Option<PyReadonlyArray2<'_, f64>>) -> PyResult<Option<Vec<f64>>> {
    a.map(|a| { layout(&a)?; Ok(a.as_slice()?.to_vec()) }).transpose()
}

fn optional_complex(a: Option<PyReadonlyArray2<'_, Complex64>>) -> PyResult<Option<Vec<Complex64>>> {
    a.map(|a| { layout(&a)?; Ok(a.as_slice()?.to_vec()) }).transpose()
}

/// `A = X V` (`periodic::finite_size::two_body::orbital_rotation`).
#[pyfunction]
#[pyo3(signature = (m, x, v))]
pub fn orbital_rotation<'py>(
    py: Python<'py>,
    m: usize,
    x: Option<PyReadonlyArray2<'py, f64>>,
    v: Option<PyReadonlyArray2<'py, Complex64>>,
) -> PyResult<Array<'py, Complex64>> {
    let (x, v) = (optional_real(x)?, optional_complex(v)?);
    let a = mandacaru_core::periodic::finite_size::two_body::orbital_rotation(m, x.as_deref(), v.as_deref())
        .map_err(super::error)?;
    array(py, &[m, m], a)
}

/// The density of `d` in the orbitals `A = X V` of `psi` (`two_body::rotated_density`).
#[pyfunction]
#[pyo3(signature = (context, psi, x, v, d))]
pub fn rotated_density<'py>(
    context: PyRef<'_, super::PyContext>,
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    v: Option<PyReadonlyArray2<'py, Complex64>>,
    d: PyReadonlyArray2<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&psi)?;
    layout(&d)?;
    let (m, points) = (psi.shape()[0], psi.shape()[1]);
    let (x, v) = (optional_real(x)?, optional_complex(v)?);
    let (psi, d) = (psi.as_slice()?.to_vec(), d.as_slice()?.to_vec());
    let ctx = &context.inner;
    let out = py.detach(|| {
        use mandacaru_core::periodic::finite_size::two_body::{orbital_rotation, rotated_density};
        let a = orbital_rotation(m, x.as_deref(), v.as_deref())?;
        rotated_density(ctx, &psi, points, &a, &d, m)
    }).map_err(super::error)?;
    array(py, &[points], out)
}

#[pyfunction]
#[pyo3(signature = (context, rho, phi, dv, x, v, m))]
pub fn pair_two_body<'py>(
    context: PyRef<'_, super::PyContext>,
    py: Python<'py>,
    rho: PyReadonlyArray2<'py, f64>,
    phi: PyReadonlyArray2<'py, f64>,
    dv: f64,
    x: Option<PyReadonlyArray2<'py, f64>>,
    v: Option<PyReadonlyArray2<'py, f64>>,
    m: usize,
) -> PyResult<Array<'py, f64>> {
    for array in [&rho, &phi] {
        layout(array)?;
    }
    if rho.shape() != phi.shape() || rho.shape()[0] != m * m {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the pair densities and potentials must both be (M^2, points)",
        ));
    }
    let points = rho.shape()[1];
    let (rho, phi) = (rho.as_slice()?.to_vec(), phi.as_slice()?.to_vec());
    let x = x.map(|a| { layout(&a)?; Ok::<_, PyErr>(a.as_slice()?.to_vec()) }).transpose()?;
    let v = v.map(|a| { layout(&a)?; Ok::<_, PyErr>(a.as_slice()?.to_vec()) }).transpose()?;
    let ctx = &context.inner;
    let out = py
        .detach(|| mandacaru_core::periodic::finite_size::two_body::pair_two_body(
            ctx, &rho, &phi, points, dv, x.as_deref(), v.as_deref(), m))
        .map_err(super::error)?;
    array(py, &[m, m, m, m], out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kzk_a0, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_effective_length, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_rs_at, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_pz_correlation_rydberg, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_g, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_dg, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_fs_exchange, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_fs_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_fs_exchange_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_infinite_exchange_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_intermediate_cubic, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_leading_term_cancellation, m)?)?;
    m.add_function(wrap_pyfunction!(kzk_correction, m)?)?;

    m.add_function(wrap_pyfunction!(ccmh_extrapolate, m)?)?;
    m.add_function(wrap_pyfunction!(ccmh_shells, m)?)?;
    m.add_function(wrap_pyfunction!(ccmh_reciprocal_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(compute_structure_factor, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_self_potential, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_seitz_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_poisson_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_coulomb_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_solve_stack, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_seitz_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(mpc_hole_energies, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_rdms, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pair_densities, m)?)?;
    m.add_function(wrap_pyfunction!(pair_two_body, m)?)?;
    m.add_function(wrap_pyfunction!(orbital_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(rotated_density, m)?)?;
    Ok(())
}
