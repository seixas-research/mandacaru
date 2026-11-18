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
        cell_slice[0],
        cell_slice[1],
        cell_slice[2],
        cell_slice[3],
        cell_slice[4],
        cell_slice[5],
        cell_slice[6],
        cell_slice[7],
        cell_slice[8],
    );

    let pos_slice = positions.as_slice()?;
    let mut frac_positions = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        frac_positions.push([pos_slice[3 * i], pos_slice[3 * i + 1], pos_slice[3 * i + 2]]);
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

fn length_scale(units: &str) -> PyResult<f64> {
    if units.eq_ignore_ascii_case("angstrom") || units.eq_ignore_ascii_case("ang") {
        Ok(mandacaru_core::units::ANGSTROM_TO_BOHR)
    } else if units.eq_ignore_ascii_case("bohr") {
        Ok(1.0)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported length unit in symmetry report: {units}"
        )))
    }
}

fn parse_moments<'py>(
    moments: Option<&Bound<'py, PyAny>>,
    vectors: Option<&Bound<'py, PyAny>>,
    n_atoms: usize,
) -> PyResult<(Option<Vec<[f64; 3]>>, bool)> {
    if let Some(obj) = vectors {
        if !obj.is_none() {
            let arr2 = obj.extract::<PyReadonlyArray2<'py, f64>>()?;
            layout(&arr2)?;
            if arr2.shape() != [n_atoms, 3] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "moment_vectors array must have shape (N, 3)",
                ));
            }
            let s = arr2.as_slice()?;
            let vecs: Vec<[f64; 3]> = (0..n_atoms)
                .map(|i| [s[3 * i], s[3 * i + 1], s[3 * i + 2]])
                .collect();
            let nonzero = vecs.iter().flatten().any(|v| v.abs() > 1e-4);
            return Ok((Some(vecs), nonzero));
        }
    }
    let Some(obj) = moments else {
        return Ok((None, false));
    };
    if obj.is_none() {
        return Ok((None, false));
    }
    if let Ok(arr2) = obj.extract::<PyReadonlyArray2<'py, f64>>() {
        layout(&arr2)?;
        if arr2.shape() != [n_atoms, 3] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "moments array must have shape (N, 3)",
            ));
        }
        let s = arr2.as_slice()?;
        let vecs: Vec<[f64; 3]> = (0..n_atoms)
            .map(|i| [s[3 * i], s[3 * i + 1], s[3 * i + 2]])
            .collect();
        let nonzero = vecs.iter().flatten().any(|v| v.abs() > 1e-4);
        return Ok((Some(vecs), nonzero));
    }
    if let Ok(arr1) = obj.extract::<PyReadonlyArray1<'py, f64>>() {
        layout(&arr1)?;
        if arr1.shape() != [n_atoms] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "collinear moments array must have shape (N,)",
            ));
        }
        let s = arr1.as_slice()?;
        let vecs: Vec<[f64; 3]> = s.iter().map(|&mz| [0.0, 0.0, mz]).collect();
        let nonzero = vecs.iter().flatten().any(|v| v.abs() > 1e-4);
        return Ok((Some(vecs), nonzero));
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "moments must be a 1D (N,) or 2D (N, 3) float64 array",
    ))
}

#[pyfunction]
#[pyo3(signature = (
    positions,
    numbers,
    *,
    units = "Angstrom",
    cell = None,
    pbc = None,
    moments = None,
    moment_vectors = None,
    spins = 1,
    num_particles = None,
    unrestricted = false,
    magnetization = None,
    method = None,
    magnetic = false,
    spin_orbit = false,
    time_reversal = true,
    kmesh_reduced = false,
    kept_operations = None,
    context = None,
))]
pub fn symmetry_log_fields<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    numbers: PyReadonlyArray1<'py, i64>,
    units: &str,
    cell: Option<PyReadonlyArray2<'py, f64>>,
    pbc: Option<[bool; 3]>,
    moments: Option<Bound<'py, PyAny>>,
    moment_vectors: Option<Bound<'py, PyAny>>,
    spins: usize,
    num_particles: Option<[usize; 2]>,
    unrestricted: bool,
    magnetization: Option<f64>,
    method: Option<&str>,
    magnetic: bool,
    spin_orbit: bool,
    time_reversal: bool,
    kmesh_reduced: bool,
    kept_operations: Option<usize>,
    context: Option<&Bound<'py, PyContext>>,
) -> PyResult<Bound<'py, PyDict>> {
    layout(&positions)?;
    layout(&numbers)?;
    let pos_shape = positions.shape();
    if pos_shape.len() != 2 || pos_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions must have shape (N, 3)",
        ));
    }
    let n_atoms = pos_shape[0];
    if numbers.shape() != [n_atoms] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "numbers length must match positions count",
        ));
    }
    let scale = length_scale(units)?;
    let pos_slice = positions.as_slice()?;
    let positions_bohr: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| {
            [
                pos_slice[3 * i] * scale,
                pos_slice[3 * i + 1] * scale,
                pos_slice[3 * i + 2] * scale,
            ]
        })
        .collect();
    let num_slice = numbers.as_slice()?;
    let numbers_u8: Vec<u8> = num_slice
        .iter()
        .map(|&z| {
            u8::try_from(z)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("atomic number out of range"))
        })
        .collect::<PyResult<_>>()?;
    let labels: Vec<u32> = numbers_u8.iter().map(|&z| u32::from(z)).collect();
    let has_noncollinear = moment_vectors.as_ref().is_some_and(|v| !v.is_none());
    let (moment_vecs, moments_nonzero) =
        parse_moments(moments.as_ref(), moment_vectors.as_ref(), n_atoms)?;
    let is_ghf = method.is_some_and(|m| m.eq_ignore_ascii_case("ghf"));
    let spin_orbit = spin_orbit || is_ghf || has_noncollinear;
    let magnetic = magnetic
        || unrestricted
        || spins > 1
        || num_particles.is_some_and(|[a, b]| a != b)
        || magnetization.is_some_and(|m| m.abs() > 1e-4)
        || moments_nonzero;
    let time_reversal = time_reversal && !(magnetic && spin_orbit);
    let is_periodic = pbc.unwrap_or([false; 3]).iter().any(|&flag| flag);
    let lattice_rows: Option<[[f64; 3]; 3]> = match cell {
        Some(c) => {
            layout(&c)?;
            if c.shape() != [3, 3] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cell must have shape (3, 3)",
                ));
            }
            let s = c.as_slice()?;
            Some([
                [s[0] * scale, s[1] * scale, s[2] * scale],
                [s[3] * scale, s[4] * scale, s[5] * scale],
                [s[6] * scale, s[7] * scale, s[8] * scale],
            ])
        }
        None => None,
    };
    let ctx = match context {
        Some(c) => c.borrow().inner.clone(),
        None => mandacaru_core::Context::new(Some(1), usize::MAX).map_err(error)?,
    };

    let pairs = py
        .detach(|| {
            if is_periodic {
                if let Some(rows) = lattice_rows {
                    return mandacaru_core::periodic::space_group::report::crystal_symmetry_fields(
                        &ctx,
                        rows,
                        &positions_bohr,
                        &labels,
                        moment_vecs.as_deref(),
                        magnetic,
                        spin_orbit,
                        time_reversal,
                        kmesh_reduced,
                        kept_operations,
                    );
                }
            }
            mandacaru_core::molecular::symmetry::molecular_symmetry_fields(
                &positions_bohr,
                &numbers_u8,
                moment_vecs.as_deref(),
                magnetic,
                spin_orbit,
            )
        })
        .map_err(error)?;

    let dict = PyDict::new(py);
    for (k, v) in pairs {
        dict.set_item(k, v)?;
    }
    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crystal_symmetry, m)?)?;
    m.add_function(wrap_pyfunction!(symmetry_log_fields, m)?)?;
    Ok(())
}
