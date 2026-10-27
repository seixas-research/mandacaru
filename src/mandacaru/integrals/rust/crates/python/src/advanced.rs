//! Thin boundaries for native dynamics and correlation.
use super::{array, copy, error, Array};
use mandacaru_core::{
    quantum::{dynamics, C64},
    Context,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};

#[pyclass(name = "Propagator", module = "mandacaru._native", frozen)]
pub struct PyPropagator {
    context: Context,
    inner: dynamics::Propagator,
}
impl PyPropagator {
    pub fn new(
        context: &Context,
        n: usize,
        terms: Vec<(String, f64, f64)>,
        order: usize,
    ) -> PyResult<Self> {
        let terms: Vec<_> = terms
            .into_iter()
            .map(|(l, r, i)| (l, C64::new(r, i)))
            .collect();
        Ok(Self {
            context: context.clone(),
            inner: dynamics::Propagator::new(context, n, &terms, order).map_err(error)?,
        })
    }
}
#[pymethods]
impl PyPropagator {
    #[pyo3(signature=(state,time,*,steps=1,inverse=false,normalized=true))]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, C64>,
        time: f64,
        steps: usize,
        inverse: bool,
        normalized: bool,
    ) -> PyResult<Array<'py, C64>> {
        let (state, _memory) = copy(&self.context, &state)?;
        let out = py
            .detach(|| {
                self.inner
                    .evolve(&self.context, &state, time, steps, inverse, normalized)
            })
            .map_err(error)?;
        array(py, &[state.len()], out.as_slice().to_vec())
    }
    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, C64>,
    ) -> PyResult<Array<'py, C64>> {
        let (state, _memory) = copy(&self.context, &state)?;
        let out = py
            .detach(|| self.inner.apply(&self.context, &state))
            .map_err(error)?;
        array(py, &[state.len()], out.as_slice().to_vec())
    }
}
pub fn qpe<'py>(
    py: Python<'py>,
    operator: &super::quantum::PyOperator,
    state: PyReadonlyArray1<'py, C64>,
    evaluation_qubits: usize,
    window: [f64; 2],
    keep: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (state, _memory) = copy(&operator.context, &state)?;
    let out = py
        .detach(|| {
            dynamics::qpe(
                &operator.context,
                &operator.inner,
                &state,
                evaluation_qubits,
                window,
                keep,
            )
        })
        .map_err(error)?;
    let result = PyDict::new(py);
    let bins = out.probabilities.as_slice().len();
    result.set_item(
        "probabilities",
        array(py, &[bins], out.probabilities.as_slice().to_vec())?,
    )?;
    if let Some(amplitudes) = out.amplitudes {
        result.set_item(
            "amplitudes",
            array(py, &[bins, state.len()], amplitudes.as_slice().to_vec())?,
        )?;
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
