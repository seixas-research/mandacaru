//! Thin boundaries for native dynamics and correlation.
use super::quantum::{input_state, parse_precision, state_array, typed};
use super::{array, copy, error};
use mandacaru_core::{
    math::storage::Precision,
    quantum::{dynamics, C64},
    Context,
};
use numpy::{PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};

#[pyclass(name = "Propagator", module = "mandacaru._native", frozen)]
pub struct PyPropagator {
    context: Context,
    inner: dynamics::Propagator,
    /// Storage precision of the states this propagator takes and returns.
    precision: Precision,
}
impl PyPropagator {
    pub fn new(
        context: &Context,
        n: usize,
        terms: Vec<(String, f64, f64)>,
        order: usize,
        precision: &str,
    ) -> PyResult<Self> {
        let precision = parse_precision(precision)?;
        let terms: Vec<_> = terms
            .into_iter()
            .map(|(l, r, i)| (l, C64::new(r, i)))
            .collect();
        Ok(Self {
            context: context.clone(),
            inner: dynamics::Propagator::new(context, n, &terms, order).map_err(error)?,
            precision,
        })
    }
}
#[pymethods]
impl PyPropagator {
    fn schedule(&self) -> (f64, Vec<(String, f64)>) {
        self.inner.schedule()
    }

    #[getter]
    fn precision(&self) -> &'static str {
        self.precision.name()
    }
    #[pyo3(signature=(state,time,*,steps=1,inverse=false,normalized=true))]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
        time: f64,
        steps: usize,
        inverse: bool,
        normalized: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = input_state(&self.context, self.precision, state)?;
        let length = input.len();
        let out = py
            .detach(|| {
                typed!(input, values => {
                    self.inner
                        .evolve(&self.context, values, time, steps, inverse, normalized)
                        .map(mandacaru_core::quantum::Amplitude::wrap)
                })
            })
            .map_err(error)?;
        state_array(py, &[length], out)
    }
    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = input_state(&self.context, self.precision, state)?;
        let length = input.len();
        let out = py
            .detach(|| {
                typed!(input, values => {
                    self.inner
                        .apply(&self.context, values)
                        .map(mandacaru_core::quantum::Amplitude::wrap)
                })
            })
            .map_err(error)?;
        state_array(py, &[length], out)
    }
}
pub fn qpe<'py>(
    py: Python<'py>,
    operator: &super::quantum::PyOperator,
    state: &Bound<'py, PyAny>,
    evaluation_qubits: usize,
    window: [f64; 2],
    keep: bool,
    precision: Precision,
) -> PyResult<Bound<'py, PyDict>> {
    let input = input_state(&operator.context, precision, state)?;
    let length = input.len();
    let out = py
        .detach(|| {
            typed!(input, values => dynamics::qpe(
                &operator.context,
                &operator.inner,
                values,
                evaluation_qubits,
                window,
                keep,
            ))
        })
        .map_err(error)?;
    let result = PyDict::new(py);
    let bins = out.probabilities.as_slice().len();
    result.set_item(
        "probabilities",
        array(py, &[bins], out.probabilities.as_slice().to_vec())?,
    )?;
    if let Some(amplitudes) = out.amplitudes {
        result.set_item("amplitudes", state_array(py, &[bins, length], amplitudes)?)?;
    } else {
        result.set_item("amplitudes", py.None())?;
    }
    result.set_item("exact_energies", out.exact_energies)?;
    result.set_item("overlaps", out.overlaps)?;
    Ok(result)
}
pub fn mp2_result<'py>(
    py: Python<'py>,
    out: mandacaru_core::molecular::correlation::Mp2Result,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let o = out.occupied_occupations.as_slice().len();
    let v = out.virtual_occupations.as_slice().len();
    result.set_item("correlation_energy", out.correlation_energy)?;
    result.set_item(
        "occupied_occupations",
        array(py, &[o], out.occupied_occupations.as_slice().to_vec())?,
    )?;
    result.set_item(
        "virtual_occupations",
        array(py, &[v], out.virtual_occupations.as_slice().to_vec())?,
    )?;
    result.set_item(
        "rotation",
        array(py, &[o + v, o + v], out.rotation.as_slice().to_vec())?,
    )?;
    result.set_item(
        "occupied_density",
        array(py, &[o, o], out.occupied_density.as_slice().to_vec())?,
    )?;
    Ok(result)
}
pub fn mp2<'py>(
    py: Python<'py>,
    ctx: &Context,
    one: PyReadonlyArray2<'py, C64>,
    two: PyReadonlyArrayDyn<'py, C64>,
    occupied: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let m = one.shape()[0];
    if one.shape() != [m, m] || two.shape() != [m, m, m, m] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "MP2 requires square one-body and matching four-index integrals",
        ));
    }
    let (one, _one) = copy(ctx, &one)?;
    let (two, _two) = copy(ctx, &two)?;
    let out = py
        .detach(|| {
            mandacaru_core::molecular::correlation::natural_orbitals(ctx, m, &one, &two, occupied)
        })
        .map_err(error)?;
    mp2_result(py, out)
}
