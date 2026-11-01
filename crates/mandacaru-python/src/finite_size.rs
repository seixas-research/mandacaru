//! Python bindings for finite-size corrections (KZK, MPC, CCMH).
use crate::arrays::{array, error, layout, Array};
use crate::context::PyContext;
use mandacaru_core::math::wigner_seitz::{minimum_image_kernel, voxel_self_potential as core_voxel_self_potential};
use mandacaru_core::math::SpatialGrid;
use mandacaru_core::periodic::finite_size::{
    ccmh_extrapolate as core_ccmh_extrapolate, kzk as core_kzk,
    mpc_from_hole_energies as core_mpc_from_hole_energies, CcmhShell, Vacuum,
};
use nalgebra::Matrix3;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (density, dv, cell, n_cells=1, floor=1e-12, max_vacuum_fraction=0.05))]
pub fn kzk_correction<'py>(
    py: Python<'py>,
    density: PyReadonlyArray1<'py, f64>,
    dv: f64,
    cell: PyReadonlyArray2<'py, f64>,
    n_cells: usize,
    floor: f64,
    max_vacuum_fraction: f64,
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
    let vacuum = Vacuum {
        floor,
        max_fraction: max_vacuum_fraction,
    };

    let result = core_kzk(density_slice, dv, &cell_matrix, n_cells, vacuum).map_err(error)?;

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
        let m = Matrix3::new(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]);
        m.determinant().abs()
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "either volume or cell must be provided to mpc_correction",
        ));
    };

    let result = core_mpc_from_hole_energies(
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kzk_correction, m)?)?;
    m.add_function(wrap_pyfunction!(mpc_correction, m)?)?;
    m.add_function(wrap_pyfunction!(ccmh_extrapolate, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_self_potential, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_seitz_kernel, m)?)?;
    Ok(())
}
