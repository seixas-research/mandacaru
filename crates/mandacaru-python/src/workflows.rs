//! Bindings of the values the workflow result views derive (`mandacaru_core::derived`,
//! `quantum::dynamics`, `quantum::term_lists` and the NumPy rounding rules of `math::numpy_rules`):
//! the magnitudes of a spectrum, its peaks, the readout of a QPE register, the normalization of a
//! state and the real terms of a propagator. Validated copies in, owned arrays out.
use crate::arrays::{array, error, layout};
use mandacaru_core::{
    derived,
    math::numpy_rules,
    quantum::{dynamics, term_lists, C64},
};
use numpy::{Complex32, IntoPyArray, PyReadonlyArray1};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyDict};

fn floats(input: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    layout(input)?;
    Ok(input.as_slice()?.to_vec())
}

/// `np.abs` of a complex128 vector (`derived::magnitudes`).
#[pyfunction]
pub fn spectrum_magnitudes<'py>(
    py: Python<'py>,
    amplitudes: PyReadonlyArray1<'py, numpy::Complex64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    layout(&amplitudes)?;
    let values: Vec<C64> = amplitudes.as_slice()?.iter().map(|v| C64::new(v.re, v.im)).collect();
    Ok(derived::magnitudes(&values).into_pyarray(py))
}

/// The positive-energy peak indices of an echo spectrum (`dynamics::echo_peaks`).
#[pyfunction]
pub fn echo_peaks<'py>(
    py: Python<'py>,
    energies: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    relative_threshold: f64,
) -> PyResult<crate::arrays::Array<'py, i64>> {
    let (energies, intensities) = (floats(&energies)?, floats(&intensities)?);
    let peaks = dynamics::echo_peaks(&energies, &intensities, relative_threshold).map_err(error)?;
    let count = peaks.len();
    array(py, &[count], peaks.into_iter().map(|i| i as i64).collect())
}

/// `{"phases", "energies", "peaks", "resolution"}` of a QPE register (`dynamics::qpe_readout`).
#[pyfunction]
pub fn qpe_readout<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArray1<'py, f64>,
    evaluation: usize,
    lo: f64,
    hi: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let probabilities = floats(&probabilities)?;
    let readout = dynamics::qpe_readout(&probabilities, evaluation, [lo, hi]).map_err(error)?;
    let out = PyDict::new(py);
    out.set_item("phases", readout.phases.into_pyarray(py))?;
    out.set_item("energies", readout.energies.into_pyarray(py))?;
    out.set_item("peaks", readout.peaks)?;
    out.set_item("resolution", readout.resolution)?;
    Ok(out)
}

/// The probability in the ground eigenspace (`dynamics::qpe_ground_overlap`).
#[pyfunction]
pub fn qpe_ground_overlap(
    exact: PyReadonlyArray1<'_, f64>,
    overlaps: PyReadonlyArray1<'_, f64>,
    hartree: bool,
) -> PyResult<f64> {
    dynamics::qpe_ground_overlap(&floats(&exact)?, &floats(&overlaps)?, hartree).map_err(error)
}

/// `state / divisor` of a complex64 or complex128 state by a Python float, by NumPy's complex
/// quotient (`numpy_rules::divide_by_real`), in the state's own dtype.
#[pyfunction]
pub fn divide_state<'py>(py: Python<'py>, state: &Bound<'py, PyAny>, divisor: f64) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(values) = state.extract::<PyReadonlyArray1<'py, numpy::Complex64>>() {
        layout(&values)?;
        let out: Vec<numpy::Complex64> = values
            .as_slice()?
            .iter()
            .map(|v| numpy_rules::divide_by_real(C64::new(v.re, v.im), divisor))
            .map(|z| numpy::Complex64::new(z.re, z.im))
            .collect();
        Ok(out.into_pyarray(py).into_any())
    } else if let Ok(values) = state.extract::<PyReadonlyArray1<'py, Complex32>>() {
        layout(&values)?;
        let out: Vec<Complex32> = values
            .as_slice()?
            .iter()
            .map(|&v| numpy_rules::divide_by_real_single(v, divisor))
            .collect();
        Ok(out.into_pyarray(py).into_any())
    } else {
        Err(PyTypeError::new_err(
            "a state must be a one-dimensional complex64 or complex128 array",
        ))
    }
}

/// `(terms, norm_bound)`: the real terms of a Hermitian Pauli sum and the sum of their magnitudes
/// (`term_lists::real_pauli_terms`). Terms cross as `(label, re, im)`.
#[pyfunction]
pub fn real_pauli_terms(n_qubits: usize, terms: Vec<(String, f64, f64)>) -> PyResult<(Vec<(String, f64)>, f64)> {
    let terms: Vec<(String, C64)> = terms.into_iter().map(|(k, re, im)| (k, C64::new(re, im))).collect();
    term_lists::real_pauli_terms(n_qubits, &terms).map_err(error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spectrum_magnitudes, m)?)?;
    m.add_function(wrap_pyfunction!(echo_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(qpe_readout, m)?)?;
    m.add_function(wrap_pyfunction!(qpe_ground_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(divide_state, m)?)?;
    m.add_function(wrap_pyfunction!(real_pauli_terms, m)?)?;
    Ok(())
}
