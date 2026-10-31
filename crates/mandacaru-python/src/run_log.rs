//! Bindings of the small facts a run log states about native objects: the term
//! count of a Hamiltonian, how the two-body integrals are stored, what a
//! checkpoint was written by and what a pool generates. Each reads a field of
//! an object that already exists; nothing is computed here and no array is
//! copied.
use super::error;
use super::molecular::PyIntegrals;
use super::persistence::PySnapshot;
use super::quantum::PyProblem;
use mandacaru_core::quantum::pools;
use pyo3::{prelude::*, types::PyDict};

/// Number of Pauli terms of the problem's qubit Hamiltonian.
#[pyfunction]
fn problem_term_count(problem: PyRef<'_, PyProblem>) -> usize {
    problem.inner.hamiltonian.terms.len()
}

/// How the integrals are stored. `representation`, `augmentation_rank`,
/// `cholesky_rank` and `cholesky_residual` are `None` while the two-body
/// integrals are not built (this call never builds them), and the rank and
/// residual are `None` for a representation without them. `atom_of_orbital`
/// maps each basis function to its atom.
#[pyfunction]
fn integrals_report<'py>(
    py: Python<'py>,
    integrals: PyRef<'_, PyIntegrals>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let report = if integrals.inner.has_eri() {
        Some(integrals.inner.eri().map_err(error)?.report())
    } else {
        None
    };
    out.set_item("representation", report.as_ref().map(|r| r.representation))?;
    out.set_item(
        "augmentation_rank",
        report.as_ref().and_then(|r| r.augmentation_rank),
    )?;
    let cholesky = report.as_ref().and_then(|r| r.cholesky);
    out.set_item("cholesky_rank", cholesky.map(|(rank, _)| rank))?;
    out.set_item("cholesky_residual", cholesky.and_then(|(_, r)| r))?;
    out.set_item("n_orbitals", integrals.inner.m())?;
    out.set_item(
        "atom_of_orbital",
        integrals
            .inner
            .basis
            .orbitals
            .iter()
            .map(|o| o.atom)
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

/// What a checkpoint was written by: `method`, `energy_ha`, `num_operators`,
/// `num_parameters` and `status` (JSON text). Raises for a Hamiltonian snapshot.
#[pyfunction]
fn snapshot_lineage<'py>(
    py: Python<'py>,
    snapshot: PyRef<'_, PySnapshot>,
) -> PyResult<Bound<'py, PyDict>> {
    let lineage = snapshot.inner.lineage().map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("method", lineage.method)?;
    out.set_item("energy_ha", lineage.energy)?;
    out.set_item("num_operators", lineage.operators)?;
    out.set_item("num_parameters", lineage.parameters)?;
    out.set_item("status", lineage.status)?;
    Ok(out)
}

/// What the ADAPT pool `name` generates. Raises for an unknown pool.
#[pyfunction]
fn pool_kind(name: &str) -> PyResult<&'static str> {
    pools::kind(name).map_err(error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(problem_term_count, m)?)?;
    m.add_function(wrap_pyfunction!(integrals_report, m)?)?;
    m.add_function(wrap_pyfunction!(snapshot_lineage, m)?)?;
    m.add_function(wrap_pyfunction!(pool_kind, m)?)?;
    Ok(())
}
