//! Bindings of the band-path and irreducible-zone views: validated inputs are copied in, owned
//! arrays come back; the numerics are in `mandacaru_core::periodic::kpath`.
use crate::arrays::{array, copy, error};
use crate::context::PyContext;
use mandacaru_core::periodic::kpath::{
    irreducible_zone as reduce, select, BandPath, Metric, PathPoints,
};
use mandacaru_core::periodic::spectrum;
use nalgebra::Matrix3;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The cell vectors as rows.
pub(crate) fn lattice(cell: &PyReadonlyArray2<'_, f64>) -> PyResult<Matrix3<f64>> {
    if cell.shape() != [3, 3] {
        return Err(PyValueError::new_err("cell must have shape (3, 3)"));
    }
    let values = cell.as_array();
    Ok(Matrix3::from_fn(|i, j| values[[i, j]]))
}

/// The band path of `path` (named points, `None` for the lattice's default path) or of the
/// explicit `points`, and the reciprocal metric of the periodic axes.
pub(crate) fn band_path_of(
    cell: &Matrix3<f64>,
    pbc: [bool; 3],
    path: Option<&str>,
    points: Option<Vec<(String, [f64; 3])>>,
) -> PyResult<(BandPath, Metric)> {
    let metric = Metric::new(cell, pbc).map_err(error)?;
    let band = match points {
        Some(points) => {
            if path.is_some() {
                return Err(PyValueError::new_err(
                    "give the band path as text or as explicit points, not both",
                ));
            }
            BandPath::explicit(&points)
        }
        None => BandPath::named(cell, pbc, path),
    }
    .map_err(error)?;
    Ok((band, metric))
}

pub(crate) fn path_dict<'py>(
    py: Python<'py>,
    selected: &PathPoints,
) -> PyResult<Bound<'py, PyDict>> {
    let n = selected.kpoints.len();
    let out = PyDict::new(py);
    out.set_item("convention", &selected.convention)?;
    out.set_item("kpoints", array(py, &[n, 3], selected.kpoints.concat())?)?;
    out.set_item("mesh_index", selected.mesh_index.clone())?;
    out.set_item("distances", array(py, &[n], selected.distances.clone())?)?;
    out.set_item("labels", selected.labels.clone())?;
    out.set_item(
        "label_distances",
        array(
            py,
            &[selected.label_distances.len()],
            selected.label_distances.clone(),
        )?,
    )?;
    out.set_item("missing", selected.missing.clone())?;
    out.set_item("complete", selected.complete())?;
    Ok(out)
}

/// The mesh points on a band path of the cell (`cell` rows, `pbc`), for the `size` mesh.
#[pyfunction]
#[pyo3(signature=(context, cell, pbc, size, *, gamma=false, path=None, points=None))]
#[allow(clippy::too_many_arguments)]
pub fn band_path<'py>(
    py: Python<'py>,
    context: &PyContext,
    cell: PyReadonlyArray2<'py, f64>,
    pbc: [bool; 3],
    size: [usize; 3],
    gamma: bool,
    path: Option<&str>,
    points: Option<Vec<(String, [f64; 3])>>,
) -> PyResult<Bound<'py, PyDict>> {
    let lattice = lattice(&cell)?;
    let (band, metric) = band_path_of(&lattice, pbc, path, points)?;
    let selected = py
        .detach(|| {
            let mesh = spectrum::mesh(&context.inner, size, gamma, &[], false)?;
            select(&band, &metric, &mesh.full)
        })
        .map_err(error)?;
    path_dict(py, &selected)
}

/// The space group of the cell and the irreducible zone of the `size` mesh under it.
#[pyfunction]
#[pyo3(signature=(context, cell, positions, numbers, size, *, gamma=false, symprec=1e-5))]
#[allow(clippy::too_many_arguments)]
pub fn irreducible_zone<'py>(
    py: Python<'py>,
    context: &PyContext,
    cell: PyReadonlyArray2<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
    numbers: PyReadonlyArray1<'py, i64>,
    size: [usize; 3],
    gamma: bool,
    symprec: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let lattice = lattice(&cell)?;
    let ctx = &context.inner;
    if positions.ndim() != 2
        || positions.shape()[1] != 3
        || numbers.shape() != [positions.shape()[0]]
    {
        return Err(PyValueError::new_err(
            "positions must be (n, 3) and numbers (n,)",
        ));
    }
    let (flat, _positions) = copy(ctx, &positions)?;
    let fractional: Vec<[f64; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let (numbers, _numbers) = copy(ctx, &numbers)?;
    let labels = numbers
        .iter()
        .map(|&z| {
            u32::try_from(z).map_err(|_| PyValueError::new_err("atomic numbers must be positive"))
        })
        .collect::<PyResult<Vec<u32>>>()?;
    let zone = py
        .detach(|| reduce(ctx, &lattice, &fractional, &labels, size, gamma, symprec))
        .map_err(error)?;
    let out = PyDict::new(py);
    let mesh = &zone.mesh;
    out.set_item(
        "full",
        array(py, &[mesh.full.len(), 3], mesh.full.concat())?,
    )?;
    out.set_item(
        "points",
        array(py, &[mesh.points.len(), 3], mesh.points.concat())?,
    )?;
    out.set_item("weights", mesh.weights.clone())?;
    out.set_item("mapping", mesh.mapping.clone())?;
    out.set_item(
        "rotations",
        array(py, &[zone.rotations.len(), 3, 3], zone.rotations.concat())?,
    )?;
    out.set_item("space_group_operations", zone.space_group_operations)?;
    out.set_item("distinct_rotations", zone.distinct_rotations)?;
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(band_path, m)?)?;
    m.add_function(wrap_pyfunction!(irreducible_zone, m)?)?;
    Ok(())
}
