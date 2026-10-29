//! NumPy/PyO3 bindings. Scientific algorithms live in mandacaru-core.
// Scientific controls are exposed as named keyword arguments.
#![allow(clippy::too_many_arguments)]
use pyo3::prelude::*;
mod advanced;
mod arrays;
mod context;
mod geometry;
mod models;
mod molecular;
mod paw;
mod periodic;
mod persistence;
mod quantum;
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
    m.add_function(wrap_pyfunction!(quantum::optimizer_evaluation_bound, m)?)?;
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
