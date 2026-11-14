//! Bindings of the provider-independent measurement arithmetic (`mandacaru_core::execution`):
//! the shot-noise bound, twirled-instance shot sharing, the run-budget check, the Estimator
//! precision and the matrix-product-state job memory. Python formats what these return.
use crate::arrays::error;
use mandacaru_core::execution;
use pyo3::prelude::*;

/// `sum |c_j| / sqrt(shots)` over the non-identity strings above `atol` (`execution::shot_noise`).
#[pyfunction]
#[pyo3(signature=(terms, shots, atol=1e-12))]
pub fn shot_noise(terms: Vec<(String, f64, f64)>, shots: u64, atol: f64) -> PyResult<f64> {
    execution::shot_noise(&terms, shots, atol).map_err(error)
}

/// `(randomizations, shots per randomization)` of twirled instances sharing `shots`; `None`
/// is the estimator's automatic choice (`execution::twirled_instances`).
#[pyfunction]
#[pyo3(signature=(shots, randomizations=None, shots_per_randomization=None))]
pub fn twirled_instances(
    shots: u64,
    randomizations: Option<u64>,
    shots_per_randomization: Option<u64>,
) -> PyResult<(u64, u64)> {
    execution::twirled_instances(shots, randomizations, shots_per_randomization).map_err(error)
}

/// `((jobs, total_shots) after, [(name, total, limit), ...] exceeded)` of `times` submissions
/// of `per = (jobs, shots)` after `used` (`execution::budget_check`). Nothing is recorded.
#[pyfunction]
#[pyo3(signature=(used, per, times=1, job_limit=None, shot_limit=None))]
#[allow(clippy::type_complexity)]
pub fn budget_check(
    used: (u64, u64),
    per: (u64, u64),
    times: u64,
    job_limit: Option<u64>,
    shot_limit: Option<u64>,
) -> PyResult<((u64, u64), Vec<(String, u64, u64)>)> {
    let (after, exceeded) =
        execution::budget_check(used, per, times, (job_limit, shot_limit)).map_err(error)?;
    Ok((after, exceeded.into_iter().map(|(name, total, limit)| (name.to_owned(), total, limit)).collect()))
}

/// `1 / sqrt(shots)`, zero without shots (`execution::estimator_precision`).
#[pyfunction]
pub fn estimator_precision(shots: u64) -> f64 {
    execution::estimator_precision(shots)
}

/// GB a matrix-product-state simulator holds for one job of `bases` bases
/// (`execution::mps_job_gigabytes`).
#[pyfunction]
pub fn mps_job_gigabytes(bases: u64) -> f64 {
    execution::mps_job_gigabytes(bases)
}

/// `(rdm_circuits, instances, total_shots)` of a device force (`execution::device_force_cost`).
#[pyfunction]
fn device_force_cost(rotations: u64, circuits: u64, bases: u64, shots: u64) -> PyResult<(u64, u64, u64)> {
    execution::device_force_cost(rotations, circuits, bases, shots).map_err(error)
}

/// The jobs of a submission, `(request indices, bases)` each (`execution::pack_jobs`).
#[pyfunction]
#[pyo3(signature = (loads, limit = None))]
fn pack_jobs(loads: Vec<u64>, limit: Option<u64>) -> Vec<(Vec<usize>, u64)> {
    execution::pack_jobs(&loads, limit)
}

/// `(entangling, single, total)` gate counts of `(name, qubits, count)` gates
/// (`execution::gate_tally`).
#[pyfunction]
#[pyo3(signature = (gates, skipped, entanglers = None))]
fn gate_tally(
    gates: Vec<(String, usize, u64)>,
    skipped: Vec<String>,
    entanglers: Option<Vec<String>>,
) -> (u64, u64, u64) {
    execution::gate_tally(&gates, &skipped, entanglers.as_deref())
}

/// The wire of qubit `k` in an SDK that numbers its wires from the other end
/// (`execution::reversed_wire`).
#[pyfunction]
fn reversed_wire(n_qubits: usize, k: usize) -> PyResult<usize> {
    execution::reversed_wire(n_qubits, k).map_err(error)
}

/// The `[start, end)` bounds of `parts` equal consecutive chunks of `total` items
/// (`execution::equal_chunks`).
#[pyfunction]
fn equal_chunks(total: usize, parts: usize) -> Vec<(usize, usize)> {
    execution::equal_chunks(total, parts)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reversed_wire, m)?)?;
    m.add_function(wrap_pyfunction!(equal_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(device_force_cost, m)?)?;
    m.add_function(wrap_pyfunction!(pack_jobs, m)?)?;
    m.add_function(wrap_pyfunction!(gate_tally, m)?)?;
    Ok(())
}
