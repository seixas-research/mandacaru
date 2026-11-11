//! Bindings of the ordered term-list algebra (`mandacaru_core::quantum::term_lists`) behind the
//! Python `PauliSum` and `Fermion` builders. Terms cross as `(key, re, im)`.
use crate::arrays::error;
use mandacaru_core::quantum::{term_lists, C64};
use pyo3::prelude::*;

type Pauli = (String, f64, f64);
type FermionTerm = (Vec<(usize, bool)>, f64, f64);

fn inward<K>(terms: Vec<(K, f64, f64)>) -> Vec<(K, C64)> {
    terms.into_iter().map(|(k, re, im)| (k, C64::new(re, im))).collect()
}
fn outward<K>(terms: Vec<(K, C64)>) -> Vec<(K, f64, f64)> {
    terms.into_iter().map(|(k, v)| (k, v.re, v.im)).collect()
}

/// Equal labels added where they first occur (`term_lists::sum`).
#[pyfunction]
pub fn pauli_term_sum(terms: Vec<Pauli>) -> PyResult<Vec<Pauli>> {
    term_lists::sum(inward(terms)).map(outward).map_err(error)
}
/// Every coefficient times `real + i imaginary` (`term_lists::scale`).
#[pyfunction]
pub fn pauli_term_scale(terms: Vec<Pauli>, real: f64, imaginary: f64) -> PyResult<Vec<Pauli>> {
    term_lists::scale(&inward(terms), C64::new(real, imaginary)).map(outward).map_err(error)
}
/// The terms above `atol` in magnitude (`term_lists::drop_small`).
#[pyfunction]
pub fn pauli_term_drop(terms: Vec<Pauli>, atol: f64) -> PyResult<Vec<Pauli>> {
    term_lists::drop_small(&inward(terms), atol).map(outward).map_err(error)
}
/// Every coefficient above `drop` has an imaginary part below `atol` (`term_lists::hermitian`).
#[pyfunction]
#[pyo3(signature=(terms, atol=1e-9, drop=1e-12))]
pub fn pauli_term_hermitian(terms: Vec<Pauli>, atol: f64, drop: f64) -> bool {
    term_lists::hermitian(&inward(terms), atol, drop)
}
/// Equal ladder products added where they first occur (`term_lists::sum`).
#[pyfunction]
pub fn fermion_term_sum(terms: Vec<FermionTerm>) -> PyResult<Vec<FermionTerm>> {
    term_lists::sum(inward(terms)).map(outward).map_err(error)
}
/// Every coefficient times `real + i imaginary` (`term_lists::scale`).
#[pyfunction]
pub fn fermion_term_scale(terms: Vec<FermionTerm>, real: f64, imaginary: f64) -> PyResult<Vec<FermionTerm>> {
    term_lists::scale(&inward(terms), C64::new(real, imaginary)).map(outward).map_err(error)
}
/// The product of two fermion operators (`term_lists::fermion_product`).
#[pyfunction]
pub fn fermion_term_product(left: Vec<FermionTerm>, right: Vec<FermionTerm>) -> PyResult<Vec<FermionTerm>> {
    term_lists::fermion_product(&inward(left), &inward(right)).map(outward).map_err(error)
}
/// The Hermitian conjugate of a fermion operator (`term_lists::fermion_dagger`).
#[pyfunction]
pub fn fermion_term_dagger(terms: Vec<FermionTerm>) -> PyResult<Vec<FermionTerm>> {
    term_lists::fermion_dagger(&inward(terms)).map(outward).map_err(error)
}
/// One more than the largest mode of any term (`term_lists::fermion_modes`).
#[pyfunction]
pub fn fermion_term_modes(terms: Vec<FermionTerm>) -> usize {
    term_lists::fermion_modes(&inward(terms))
}
