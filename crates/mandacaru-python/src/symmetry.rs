//! Python bindings for crystal symmetry detection and space group identification.
use crate::arrays::{array, error, layout};
use crate::context::PyContext;
use mandacaru_core::periodic::space_group::{find, find_with_context, identify, Cell};
use nalgebra::Matrix3;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (cell, positions, numbers, symprec=1e-5, _context=None))]
pub fn crystal_symmetry<'py>(
    py: Python<'py>,
    cell: PyReadonlyArray2<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
    numbers: PyReadonlyArray1<'py, i64>,
    symprec: f64,
    _context: Option<&Bound<'py, PyContext>>,
) -> PyResult<Bound<'py, PyDict>> {
    layout(&cell)?;
    layout(&positions)?;
    layout(&numbers)?;

    let cell_shape = cell.shape();
    if cell_shape != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cell must have shape (3, 3), got {:?}",
            cell_shape
        )));
    }

    let pos_shape = positions.shape();
    if pos_shape.len() != 2 || pos_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "positions must have shape (N, 3), got {:?}",
            pos_shape
        )));
    }

    let n_atoms = pos_shape[0];
    if numbers.shape() != [n_atoms] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "numbers length ({}) does not match positions count ({})",
            numbers.shape()[0],
            n_atoms
        )));
    }

    let cell_slice = cell.as_slice()?;
    let lat_matrix = Matrix3::new(
        cell_slice[0], cell_slice[1], cell_slice[2],
        cell_slice[3], cell_slice[4], cell_slice[5],
        cell_slice[6], cell_slice[7], cell_slice[8],
    );

    let pos_slice = positions.as_slice()?;
    let mut frac_positions = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        frac_positions.push([
            pos_slice[3 * i],
            pos_slice[3 * i + 1],
            pos_slice[3 * i + 2],
        ]);
    }

    let num_slice = numbers.as_slice()?;
    let labels: Vec<u32> = num_slice.iter().map(|&z| z as u32).collect();

    let ctx_opt = _context.map(|c| c.borrow().inner.clone());
    let (ops, id) = py
        .detach(|| {
            let crystal_cell = Cell {
                lattice: lat_matrix,
                fractional: &frac_positions,
                labels: &labels,
            };
            let ops = match &ctx_opt {
                Some(ctx) => find_with_context(ctx, &crystal_cell, symprec)?,
                None => find(&crystal_cell, symprec)?,
            };
            let id = identify(&ops, &crystal_cell)?;
            Ok::<_, mandacaru_core::Error>((ops, id))
        })
        .map_err(error)?;

    let n_ops = ops.rotations.len();
    let mut rot_flat = Vec::with_capacity(n_ops * 9);
    for r in &ops.rotations {
        rot_flat.push(r[(0, 0)]);
        rot_flat.push(r[(0, 1)]);
        rot_flat.push(r[(0, 2)]);
        rot_flat.push(r[(1, 0)]);
        rot_flat.push(r[(1, 1)]);
        rot_flat.push(r[(1, 2)]);
        rot_flat.push(r[(2, 0)]);
        rot_flat.push(r[(2, 1)]);
        rot_flat.push(r[(2, 2)]);
    }

    let mut trans_flat = Vec::with_capacity(n_ops * 3);
    for t in &ops.translations {
        trans_flat.extend_from_slice(t);
    }

    let py_rotations = array(py, &[n_ops, 3, 3], rot_flat)?;
    let py_translations = array(py, &[n_ops, 3], trans_flat)?;

    let pg_symbol = mandacaru_core::periodic::space_group::catalog::lookup(id.number)
        .map(|g| g.point_group)
        .unwrap_or(ops.point_group.symbol);
    let dict = PyDict::new(py);
    dict.set_item("number", id.number)?;
    dict.set_item("international", id.symbol)?;
    dict.set_item("point_group", pg_symbol)?;
    dict.set_item("rotations", py_rotations)?;
    dict.set_item("translations", py_translations)?;
    dict.set_item("n_operations", n_ops)?;
    dict.set_item("n_pure_translations", ops.n_pure_translations)?;
    dict.set_item("symprec", symprec)?;
    dict.set_item("symprec_used", ops.tolerance_used)?;

    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crystal_symmetry, m)?)?;
    Ok(())
}
