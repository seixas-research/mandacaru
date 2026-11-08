//! Python bindings for finite-size corrections (KZK, MPC, CCMH).
use crate::arrays::{array, error, layout, Array};
use crate::context::PyContext;
use mandacaru_core::math::fft::Fft3;
use mandacaru_core::math::wigner_seitz::{
    minimum_image_kernel, voxel_self_potential as core_voxel_self_potential,
};
use mandacaru_core::math::SpatialGrid;
use mandacaru_core::periodic::finite_size::{
    ccmh_extrapolate as core_ccmh_extrapolate, kzk as core_kzk, kzk_flexible, CcmhShell,
};
use nalgebra::{Matrix3, Vector3};
type C64 = nalgebra::Complex<f64>;
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
    let k_slice = magnitudes.as_slice()?;
    let s_slice = values.as_slice()?;
    if k_slice.is_empty() || count == 0 {
        return Ok((array(py, &[0], vec![])?, array(py, &[0], vec![])?));
    }
    let mut pairs: Vec<(f64, f64)> = k_slice
        .iter()
        .copied()
        .zip(s_slice.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let max_k = pairs.last().map(|p| p.0).unwrap_or(1.0);
    let tol = 1e-8 * max_k.max(1.0);

    let mut k_means = Vec::new();
    let mut s_means = Vec::new();

    let mut current_k_sum = pairs[0].0;
    let mut current_s_sum = pairs[0].1;
    let mut current_count = 1usize;
    let mut prev_k = pairs[0].0;

    for &(k, s) in &pairs[1..] {
        if (k - prev_k) > tol {
            k_means.push(current_k_sum / current_count as f64);
            s_means.push(current_s_sum / current_count as f64);
            if k_means.len() >= count {
                break;
            }
            current_k_sum = 0.0;
            current_s_sum = 0.0;
            current_count = 0;
        }
        current_k_sum += k;
        current_s_sum += s;
        current_count += 1;
        prev_k = k;
    }
    if current_count > 0 && k_means.len() < count {
        k_means.push(current_k_sum / current_count as f64);
        s_means.push(current_s_sum / current_count as f64);
    }

    let n = k_means.len();
    Ok((array(py, &[n], k_means)?, array(py, &[n], s_means)?))
}

#[pyfunction]
pub fn ccmh_reciprocal_vectors<'py>(
    py: Python<'py>,
    cell: PyReadonlyArray2<'py, f64>,
    limit: i64,
) -> PyResult<(Array<'py, i64>, Array<'py, f64>)> {
    layout(&cell)?;
    if cell.shape() != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cell must be (3, 3)",
        ));
    }
    let c = cell.as_slice()?;
    let mat = Matrix3::new(
        c[0], c[1], c[2],
        c[3], c[4], c[5],
        c[6], c[7], c[8],
    );
    let inv_t = mat
        .try_inverse()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("singular cell matrix"))?
        .transpose();
    let b = inv_t * (2.0 * std::f64::consts::PI);

    let mut triples = Vec::new();
    let mut kvectors = Vec::new();

    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if i == 0 && j == 0 && k == 0 {
                    continue;
                }
                triples.push(i);
                triples.push(j);
                triples.push(k);
                let m = Vector3::new(i as f64, j as f64, k as f64);
                let kx = m[0] * b[(0, 0)] + m[1] * b[(1, 0)] + m[2] * b[(2, 0)];
                let ky = m[0] * b[(0, 1)] + m[1] * b[(1, 1)] + m[2] * b[(2, 1)];
                let kz = m[0] * b[(0, 2)] + m[1] * b[(1, 2)] + m[2] * b[(2, 2)];
                kvectors.push(kx);
                kvectors.push(ky);
                kvectors.push(kz);
            }
        }
    }

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
#[pyo3(signature = (hole_model, hole_periodic, v_madelung, n_electrons, n_cells=1, volume=None, cell=None))]
pub fn mpc_correction<'py>(
    py: Python<'py>,
    hole_model: f64,
    hole_periodic: f64,
    v_madelung: f64,
    n_electrons: f64,
    n_cells: usize,
    volume: Option<f64>,
    cell: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let vol = if let Some(v) = volume {
        v
    } else if let Some(ref c) = cell {
        layout(c)?;
        let s = c.as_slice()?;
        let m = Matrix3::new(
            s[0], s[1], s[2],
            s[3], s[4], s[5],
            s[6], s[7], s[8],
        );
        m.determinant().abs()
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "either volume or cell must be provided to mpc_correction",
        ));
    };

    let result = mandacaru_core::periodic::finite_size::mpc_from_hole_energies(
        hole_model,
        hole_periodic,
        v_madelung,
        n_electrons,
        n_cells,
        vol,
    )
    .map_err(error)?;

    let dict = PyDict::new(py);
    dict.set_item("correction", result.correction)?;
    dict.set_item("correction_per_cell", result.correction_per_cell())?;
    dict.set_item("hole_model", result.hole_model)?;
    dict.set_item("hole_periodic", result.hole_periodic)?;
    dict.set_item("madelung_shift", result.madelung_shift)?;
    dict.set_item("v_madelung", result.v_madelung)?;
    dict.set_item("n_electrons", result.n_electrons)?;
    dict.set_item("n_cells", result.scope.n_cells)?;
    dict.set_item("volume", result.scope.volume)?;
    dict.set_item("raw_correction", result.raw_correction)?;

    Ok(dict)
}

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
    let s = step.as_slice()?;
    let step_mat = Matrix3::new(
        s[0], s[1], s[2],
        s[3], s[4], s[5],
        s[6], s[7], s[8],
    );
    let diag = Matrix3::new(
        shape.0 as f64, 0.0, 0.0,
        0.0, shape.1 as f64, 0.0,
        0.0, 0.0, shape.2 as f64,
    );
    let cell = step_mat * diag;
    let inv_t = cell
        .try_inverse()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("singular cell"))?
        .transpose();
    let reciprocal = inv_t * (2.0 * std::f64::consts::PI);

    let (nx, ny, nz) = shape;
    let mut k = Vec::with_capacity(nx * ny * nz);

    for i in 0..nx {
        let mx = if i < (nx + 1) / 2 {
            i as f64
        } else {
            i as f64 - nx as f64
        };
        for j in 0..ny {
            let my = if j < (ny + 1) / 2 {
                j as f64
            } else {
                j as f64 - ny as f64
            };
            for k_idx in 0..nz {
                let mz = if k_idx < (nz + 1) / 2 {
                    k_idx as f64
                } else {
                    k_idx as f64 - nz as f64
                };
                let m = Vector3::new(mx, my, mz);
                let kvec = reciprocal * m;
                let g2 = kvec.dot(&kvec);
                if g2 > 0.0 {
                    k.push((4.0 * std::f64::consts::PI / g2) / dv);
                } else {
                    k.push(0.0);
                }
            }
        }
    }
    array(py, &[nx, ny, nz], k)
}

#[pyfunction]
pub fn truncated_coulomb_kernel<'py>(
    py: Python<'py>,
    real_kernel: PyReadonlyArrayDyn<'py, f64>,
    shape: (usize, usize, usize),
) -> PyResult<Array<'py, f64>> {
    layout(&real_kernel)?;
    let (nx, ny, nz) = shape;
    let slice = real_kernel.as_slice()?;
    if slice.len() != nx * ny * nz {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "real_kernel size does not match shape",
        ));
    }

    let ctx = mandacaru_core::Context::new(None, 64 << 20).map_err(error)?;
    let mut fft = Fft3::new(&ctx, [nx, ny, nz]).map_err(error)?;

    let mut buf: Vec<C64> = slice.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft.transform(&mut buf, false).map_err(error)?;

    let real_data: Vec<f64> = buf.iter().map(|c| c.re).collect();
    array(py, &[nx, ny, nz], real_data)
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

    let (nx, ny, nz) = shape;
    let n_grid = nx * ny * nz;
    let k_slice = kernel.as_slice()?;
    if k_slice.len() != n_grid {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "kernel size does not match shape",
        ));
    }
    let p_count = rho_stack.shape()[0];
    let rho_cols = rho_stack.shape()[1];
    if rho_cols != n_grid {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rho_stack columns do not match shape product",
        ));
    }

    let ctx = mandacaru_core::Context::new(None, 64 << 20).map_err(error)?;
    let mut fft = Fft3::new(&ctx, [nx, ny, nz]).map_err(error)?;

    let rho_slice = rho_stack.as_slice()?;
    let mut out = Vec::with_capacity(p_count * n_grid);
    let mut buf = vec![C64::new(0.0, 0.0); n_grid];

    for p in 0..p_count {
        let rho_p = &rho_slice[p * n_grid..(p + 1) * n_grid];
        for i in 0..n_grid {
            buf[i] = C64::new(rho_p[i], 0.0);
        }
        fft.transform(&mut buf, false).map_err(error)?;
        for i in 0..n_grid {
            buf[i] = buf[i] * k_slice[i];
        }
        fft.transform(&mut buf, true).map_err(error)?;
        for i in 0..n_grid {
            out.push(buf[i].re * dv);
        }
    }

    array(py, &[p_count, n_grid], out)
}

#[pyfunction]
pub fn wigner_seitz_metrics<'py>(
    kernel: PyReadonlyArrayDyn<'py, f64>,
    dv: f64,
) -> PyResult<(f64, f64)> {
    layout(&kernel)?;
    let slice = kernel.as_slice()?;
    let mut count = 0usize;
    let mut max_inv = 0.0_f64;
    for &v in slice {
        if v != 0.0 {
            count += 1;
        }
        if v > 0.0 {
            let inv = 1.0 / v;
            if inv > max_inv {
                max_inv = inv;
            }
        }
    }
    Ok((count as f64 * dv, max_inv))
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

    let p_slice = eri_periodic.as_slice()?;
    let t_slice = eri_truncated.as_slice()?;
    let g_slice = gamma.as_slice()?;
    let d_slice = d.as_slice()?;

    let mut periodic = 0.0_f64;
    let mut truncated = 0.0_f64;

    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                let d_pr = d_slice[p * m + r];
                for s in 0..m {
                    let idx = ((p * m + q) * m + r) * m + s;
                    let hole = g_slice[idx] - d_pr * d_slice[q * m + s];
                    periodic += p_slice[idx] * hole;
                    truncated += t_slice[idx] * hole;
                }
            }
        }
    }

    let mut n_electrons = 0.0_f64;
    for i in 0..m {
        n_electrons += d_slice[i * m + i];
    }

    Ok((0.5 * periodic, 0.5 * truncated, n_electrons))
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
    let g_slice = gamma.as_slice()?;
    let mut d = vec![0.0_f64; m * m];
    for p in 0..m {
        for q in 0..m {
            d[p * m + q] = g_slice[p * (2 * m) + q] + g_slice[(p + m) * (2 * m) + (q + m)];
        }
    }
    let d_arr = array(py, &[m, m], d)?;

    let g_arr = if let Some(g2) = gamma2 {
        layout(&g2)?;
        let g2_shape = g2.shape();
        if g2_shape != [2 * m, 2 * m, 2 * m, 2 * m] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected a (2M, 2M, 2M, 2M) two-body RDM, got {:?}",
                g2_shape
            )));
        }
        let g2_slice = g2.as_slice()?;
        let dim = 2 * m;
        let mut g = vec![0.0_f64; m * m * m * m];
        for p in 0..m {
            for q in 0..m {
                for r in 0..m {
                    for s in 0..m {
                        let idx_m = ((p * m + q) * m + r) * m + s;
                        let val_aaaa =
                            g2_slice[((p * dim + q) * dim + r) * dim + s];
                        let val_abab =
                            g2_slice[((p * dim + (q + m)) * dim + r) * dim + (s + m)];
                        let val_baba =
                            g2_slice[(((p + m) * dim + q) * dim + (r + m)) * dim + s];
                        let val_bbbb =
                            g2_slice[(((p + m) * dim + (q + m)) * dim + (r + m)) * dim + (s + m)];
                        g[idx_m] = val_aaaa + val_abab + val_baba + val_bbbb;
                    }
                }
            }
        }
        Some(array(py, &[m, m, m, m], g)?)
    } else {
        None
    };

    Ok((d_arr, g_arr))
}

#[pyfunction]
pub fn compute_pair_densities<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray2<'py, f64>,
) -> PyResult<Array<'py, f64>> {
    layout(&psi)?;
    let m = psi.shape()[0];
    let ngrid = psi.shape()[1];
    let slice = psi.as_slice()?;
    let mut rho_stack = vec![0.0_f64; m * m * ngrid];
    for b in 0..m {
        let psi_b = &slice[b * ngrid..(b + 1) * ngrid];
        for d in 0..m {
            let psi_d = &slice[d * ngrid..(d + 1) * ngrid];
            let row = &mut rho_stack[(b * m + d) * ngrid..(b * m + d + 1) * ngrid];
            for g in 0..ngrid {
                row[g] = psi_b[g] * psi_d[g];
            }
        }
    }
    array(py, &[m * m, ngrid], rho_stack)
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

    m.add_function(wrap_pyfunction!(mpc_correction, m)?)?;
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
