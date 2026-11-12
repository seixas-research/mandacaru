//! NumPy/PyO3 bindings. Scientific algorithms live in mandacaru-core.
// Scientific controls are exposed as named keyword arguments.
#![allow(clippy::too_many_arguments)]
use pyo3::prelude::*;
mod advanced;
mod angular;
mod arrays;
mod atom;
mod basis_confinement;
mod basis_filter;
mod basis_table;
mod basis_zetas;
mod berry;
mod bloch;
mod context;
mod derived;
mod dipole;
mod factorization;
mod finite_size;
mod geometry;
mod hubbard;
mod kpath;
mod measurement;
mod term_lists;
mod models;
mod molecular;
mod orbitals;
mod optimizers;
mod parquet;
mod paw_basis;
mod paw_diagnose;
mod paw_generate;
mod paw_multipole;
mod paw_onecenter;
mod paw_pseudize;
mod paw_record;
mod periodic;
mod persistence;
mod quantum;
mod run_log;
mod symmetry;
mod sparse;
mod viewer;
mod workflows;
mod downfold;
mod excitations;
mod wannier;
mod xc_grid;
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
    m.add("ANGSTROM_TO_BOHR", mandacaru_core::units::ANGSTROM_TO_BOHR)?;
    m.add("EV_TO_HARTREE", mandacaru_core::units::EV_TO_HARTREE)?;
    // The defaults of the basis options, one definition, in Rust.
    m.add(
        "DEFAULT_ENERGY_SHIFT",
        mandacaru_core::paw::basis_recipe::DEFAULT_ENERGY_SHIFT,
    )?;
    m.add(
        "MIN_ENERGY_SHIFT",
        mandacaru_core::paw::basis_recipe::MIN_ENERGY_SHIFT,
    )?;
    m.add(
        "CONFINEMENT_AMPLITUDE",
        mandacaru_core::basis::confinement::AMPLITUDE,
    )?;
    m.add(
        "CONFINEMENT_INNER_FRACTION",
        mandacaru_core::basis::confinement::INNER_FRACTION,
    )?;
    m.add("FILTER_NYQUIST_FRACTION", mandacaru_core::basis::filter::NYQUIST_FRACTION)?;
    m.add("DEFAULT_RMS_RATIO", mandacaru_core::basis::diffuse::DEFAULT_RMS_RATIO)?;
    m.add("DEFAULT_SHELL_RATIO", mandacaru_core::basis::diffuse::default_shell_ratio())?;
    m.add("DEFAULT_CUTOFF_NORM", mandacaru_core::basis::diffuse::DEFAULT_CUTOFF_NORM)?;
    m.add("MAX_CUTOFF_NORM", mandacaru_core::basis::diffuse::MAX_CUTOFF_NORM)?;
    m.add(
        "LINEAR_DEPENDENCE_TOL",
        mandacaru_core::paw::basis_recipe::DIFFUSE_LINEAR_DEPENDENCE_TOL,
    )?;
    m.add(
        "FUNCTIONALS",
        mandacaru_core::models::FUNCTIONALS
            .iter()
            .map(|f| f.name)
            .collect::<Vec<_>>(),
    )?;
    m.add(
        "ACTIVE_SPACE_METHODS",
        mandacaru_core::molecular::active::METHODS.to_vec(),
    )?;
    m.add(
        "OPTIMIZERS",
        mandacaru_core::quantum::optimizer::Method::NAMES.to_vec(),
    )?;
    // The smallest settings a complex64 state resolves: one definition, in Rust.
    m.add(
        "FLOAT32_ENERGY_FLOOR_HA",
        mandacaru_core::quantum::amplitude::FLOAT32_FLOORS.energy,
    )?;
    m.add(
        "FLOAT32_GRADIENT_FLOOR",
        mandacaru_core::quantum::amplitude::FLOAT32_FLOORS.gradient,
    )?;
    m.add_function(wrap_pyfunction!(context::num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(molecular::two_body_representation, m)?)?;
    m.add_function(wrap_pyfunction!(excitations::molecular_excitations, m)?)?;
    m.add_function(wrap_pyfunction!(excitations::line_spectrum, m)?)?;
    m.add_class::<excitations::PyExcitations>()?;
    m.add_function(wrap_pyfunction!(context::grid_blas_threads, m)?)?;
    m.add_function(wrap_pyfunction!(context::worker_threads, m)?)?;
    m.add_function(wrap_pyfunction!(derived::quotients, m)?)?;
    m.add_function(wrap_pyfunction!(derived::rounded_total, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::optimizer_evaluation_bound, m)?)?;
    m.add_function(wrap_pyfunction!(measurement::shot_noise, m)?)?;
    m.add_function(wrap_pyfunction!(derived::to_user_units, m)?)?;
    m.add_function(wrap_pyfunction!(derived::free_wavevector, m)?)?;
    m.add_function(wrap_pyfunction!(derived::free_energy_ev, m)?)?;
    m.add_function(wrap_pyfunction!(derived::energy_shift_radius, m)?)?;
    m.add_function(wrap_pyfunction!(derived::nyquist_wavevector, m)?)?;
    m.add_function(wrap_pyfunction!(derived::force_summary, m)?)?;
    m.add_function(wrap_pyfunction!(derived::iteration_values, m)?)?;
    m.add_function(wrap_pyfunction!(derived::cell_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(derived::relative_to_first, m)?)?;
    m.add_function(wrap_pyfunction!(derived::difference_to_user, m)?)?;
    m.add_function(wrap_pyfunction!(derived::electron_counts, m)?)?;
    m.add_function(wrap_pyfunction!(derived::energy_from_hartree, m)?)?;
    m.add_function(wrap_pyfunction!(derived::energy_to_hartree, m)?)?;
    m.add_function(wrap_pyfunction!(derived::convert_energy, m)?)?;
    m.add_function(wrap_pyfunction!(derived::length_to_bohr, m)?)?;
    m.add_function(wrap_pyfunction!(derived::energy_unit_factor, m)?)?;
    m.add_function(wrap_pyfunction!(derived::total, m)?)?;
    m.add_function(wrap_pyfunction!(derived::mesh_fractions, m)?)?;
    m.add_function(wrap_pyfunction!(derived::vector_norm, m)?)?;
    m.add_function(wrap_pyfunction!(derived::distance, m)?)?;
    m.add_function(wrap_pyfunction!(derived::gaps, m)?)?;
    m.add_function(wrap_pyfunction!(derived::uniform_weights, m)?)?;
    m.add_function(wrap_pyfunction!(derived::highest_occupied, m)?)?;
    m.add_function(wrap_pyfunction!(derived::energy_axis, m)?)?;
    m.add_function(wrap_pyfunction!(derived::column_groups, m)?)?;
    m.add_function(wrap_pyfunction!(derived::displaced, m)?)?;
    m.add_function(wrap_pyfunction!(derived::ratio, m)?)?;
    m.add_function(wrap_pyfunction!(derived::grid_cell, m)?)?;
    m.add_function(wrap_pyfunction!(derived::grid_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(derived::decimal_digits, m)?)?;
    m.add_function(wrap_pyfunction!(derived::rounded, m)?)?;
    m.add_function(wrap_pyfunction!(derived::descending_weights, m)?)?;
    m.add_function(wrap_pyfunction!(derived::max_abs, m)?)?;
    m.add_function(wrap_pyfunction!(derived::array_minimum, m)?)?;
    m.add_function(wrap_pyfunction!(derived::array_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(derived::sample_times, m)?)?;
    m.add_function(wrap_pyfunction!(derived::fourier_bin_spacing, m)?)?;
    m.add_function(wrap_pyfunction!(derived::nyquist_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(derived::squared, m)?)?;
    m.add_function(wrap_pyfunction!(derived::modulus_squared, m)?)?;
    m.add_function(wrap_pyfunction!(derived::pulse_bound, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::pauli_term_sum, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::pauli_term_scale, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::pauli_term_drop, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::pauli_term_hermitian, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::fermion_term_sum, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::fermion_term_scale, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::fermion_term_product, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::fermion_term_dagger, m)?)?;
    m.add_function(wrap_pyfunction!(term_lists::fermion_term_modes, m)?)?;
    m.add_function(wrap_pyfunction!(dipole::electric_field_vector, m)?)?;
    m.add_function(wrap_pyfunction!(dipole::dipole_one_body, m)?)?;
    m.add_function(wrap_pyfunction!(measurement::twirled_instances, m)?)?;
    m.add_function(wrap_pyfunction!(measurement::budget_check, m)?)?;
    m.add_function(wrap_pyfunction!(measurement::estimator_precision, m)?)?;
    m.add_function(wrap_pyfunction!(measurement::mps_job_gigabytes, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::state_norm_squared, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::state_norm, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::state_inner, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::require_normalized_state, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::periodic_quantum_forces_fd, m)?)?;
    m.add_function(wrap_pyfunction!(bloch::supercell_integrals, m)?)?;
    m.add_function(wrap_pyfunction!(bloch::bloch_supercell, m)?)?;
    m.add_function(wrap_pyfunction!(bloch::bloch_bands, m)?)?;
    m.add_function(wrap_pyfunction!(bloch::bloch_spectral_function, m)?)?;
    m.add_function(wrap_pyfunction!(bloch::bloch_path_spectral_function, m)?)?;
    m.add_function(wrap_pyfunction!(bloch::tetrahedron_dos, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::periodic_quantum_stress_fd, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::periodic_quantum_forces_analytic, m)?)?;
    m.add_function(wrap_pyfunction!(quantum::periodic_quantum_stress_analytic, m)?)?;
    kpath::register(m)?;
    paw_basis::register(m)?;
    paw_diagnose::register(m)?;
    paw_generate::register(m)?;
    paw_multipole::register(m)?;
    paw_onecenter::register(m)?;
    paw_pseudize::register(m)?;
    paw_record::register(m)?;
    parquet::register(m)?;
    basis_confinement::register(m)?;
    basis_filter::register(m)?;
    basis_table::register(m)?;
    basis_zetas::register(m)?;
    run_log::register(m)?;
    hubbard::register(m)?;
    symmetry::register(m)?;
    finite_size::register(m)?;
    viewer::register(m)?;
    workflows::register(m)?;
    angular::register(m)?;
    models::register(m)?;
    geometry::register(m)?;
    orbitals::register(m)?;
    optimizers::register(m)?;
    sparse::register(m)?;
    xc_grid::register(m)?;
    m.add_class::<PyContext>()?;
    m.add_class::<persistence::PySnapshot>()?;
    m.add_class::<persistence::PyLedger>()?;
    m.add_class::<PyGrid>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<Workspace>()?;
    m.add_class::<quantum::PyOperator>()?;
    m.add_class::<advanced::PyPropagator>()?;
    m.add_function(wrap_pyfunction!(advanced::fourier_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(advanced::qpe_memory, m)?)?;
    m.add_function(wrap_pyfunction!(advanced::givens_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(factorization::double_factorization, m)?)?;
    m.add_function(wrap_pyfunction!(factorization::spatial_double_factorization, m)?)?;
    m.add_function(wrap_pyfunction!(factorization::pair_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(factorization::factorized_energy_rdm, m)?)?;
    m.add_class::<quantum::PyProblem>()?;
    m.add_class::<quantum::PyRdmMeasurement>()?;
    m.add_class::<PySpatialGrid>()?;
    m.add_class::<molecular::PyIntegrals>()?;
    m.add_class::<molecular::PyMeanField>()?;
    m.add_class::<molecular::PyHandover>()?;
    m.add_class::<molecular::PyModel>()?;
    m.add_class::<periodic::PyCrystal>()?;
    m.add_class::<periodic::PyPeriodicSymmetry>()?;
    m.add_class::<periodic::PyPeriodicResult>()?;
    m.add_class::<wannier::PyWannierSetup>()?;
    m.add_class::<wannier::PyWannierResult>()?;
    m.add_class::<downfold::PyDownfolded>()?;
    Ok(())
}
