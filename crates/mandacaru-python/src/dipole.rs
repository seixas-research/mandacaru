//! Bindings of the length-gauge dipole coupling (`mandacaru_core::quantum::fermion_terms`).
use crate::arrays::error;
use mandacaru_core::quantum::{fermion_terms, C64};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// `strength * direction` for a unit Cartesian `direction` (`fermion_terms::field_vector`).
#[pyfunction]
pub fn electric_field_vector(direction: [f64; 3], strength: f64) -> PyResult<[f64; 3]> {
    fermion_terms::field_vector(direction, strength).map_err(error)
}

/// `(h, constant)` of `V = -E . mu`: `h = E . <p|r|q>` (in alpha and beta blocks with
/// `spatial_orbitals`) and `-E . mu_nuc` (`fermion_terms::dipole_one_body`).
#[pyfunction]
#[pyo3(signature=(position_integrals, field, nuclear_dipole, spatial_orbitals=true))]
pub fn dipole_one_body<'py>(
    py: Python<'py>,
    position_integrals: PyReadonlyArray3<'py, C64>,
    field: [f64; 3],
    nuclear_dipole: [f64; 3],
    spatial_orbitals: bool,
) -> PyResult<(Bound<'py, PyArray2<C64>>, f64)> {
    let shape = position_integrals.shape();
    if shape[0] != 3 || shape[1] == 0 || shape[1] != shape[2] {
        return Err(error(mandacaru_core::Error::Invalid(
            "position_integrals must have shape (3, m, m), m > 0".into(),
        )));
    }
    let m = shape[1];
    let positions = position_integrals.as_slice()?.to_vec();
    let (h, modes, constant) = py
        .detach(|| fermion_terms::dipole_one_body(&positions, m, field, nuclear_dipole, spatial_orbitals))
        .map_err(error)?;
    let matrix = h.into_pyarray(py).reshape([modes, modes])?;
    Ok((matrix, constant))
}
