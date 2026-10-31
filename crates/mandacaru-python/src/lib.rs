//! NumPy/PyO3 bindings. Scientific algorithms live in mandacaru-core.
// Scientific controls are exposed as named keyword arguments.
#![allow(clippy::too_many_arguments)]
use pyo3::prelude::*;
mod advanced;
mod arrays;
mod atom;
mod basis_confinement;
mod basis_filter;
mod basis_table;
mod context;
mod geometry;
mod hubbard;
mod models;
mod molecular;
mod parquet;
mod paw;
mod paw_multipole;
mod paw_onecenter;
mod paw_pseudize;
mod periodic;
mod persistence;
mod quantum;
mod run_log;
mod wannier;
use arrays::{array, complex, complex_array, copy, error, layout, Array};
use context::PyContext;
use geometry::{PyGeometry, PyGrid, PySpatialGrid, Workspace};
// Importing this module into a free-threaded interpreter re-enables the GIL.
// NumPy input snapshots are taken while attached; only Rust-owned data is used
// in detached work. Free-threaded support needs a separate NumPy alias audit.
#[pymodule(gil_used = true)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BOHR_TO_ANGSTROM", mandacaru_core::units::BOHR_TO_ANGSTROM)?;
    m.add("HARTREE_TO_EV", mandacaru_core::units::HARTREE_TO_EV)?;
    m.add(
        "FUNCTIONALS",
        mandacaru_core::models::FUNCTIONALS
            .iter()
            .map(|f| f.name)
            .collect::<Vec<_>>(),
    )?;
    m.add("ABI_VERSION", 1)?;
    m.add(
        "OPTIMIZERS",
        mandacaru_core::quantum::optimizer::Method::NAMES.to_vec(),
    )?;
    m.add("THREAD_POLICY_VERSION", 1)?;
    // The smallest settings a complex64 state resolves: one definition, in Rust.
    m.add(
        "FLOAT32_ENERGY_FLOOR_HA",
        mandacaru_core::quantum::amplitude::FLOAT32_FLOORS.energy,
    )?;
    m.add(
        "FLOAT32_GRADIENT_FLOOR",
        mandacaru_core::quantum::amplitude::FLOAT32_FLOORS.gradient,
    )?;
    m.add_function(wrap_pyfunction!(quantum::optimizer_evaluation_bound, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::state_norm_squared, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::state_inner, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::require_normalized_state, m)?)?;
    paw_multipole::register(m)?;
    paw_onecenter::register(m)?;
    paw_pseudize::register(m)?;
    parquet::register(m)?;
    basis_confinement::register(m)?;
    basis_filter::register(m)?;
    basis_table::register(m)?;
    run_log::register(m)?;
    hubbard::register(m)?;
    m.add_class::<PyContext>()?;
    m.add_class::<persistence::PySnapshot>()?;
    m.add_class::<persistence::PyLedger>()?;
    m.add_class::<PyGrid>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<Workspace>()?;
    m.add_class::<quantum::PyOperator>()?;
    m.add_class::<advanced::PyPropagator>()?;
    m.add_class::<quantum::PyProblem>()?;
    m.add_class::<PySpatialGrid>()?;
    m.add_class::<molecular::PyIntegrals>()?;
    m.add_class::<molecular::PyMeanField>()?;
    m.add_class::<molecular::PyHandover>()?;
    m.add_class::<molecular::PyModel>()?;
    m.add_class::<periodic::PyCrystal>()?;
    m.add_class::<periodic::PyPeriodicResult>()?;
    m.add_class::<wannier::PyWannierSetup>()?;
    m.add_class::<wannier::PyWannierResult>()?;
    Ok(())
}
