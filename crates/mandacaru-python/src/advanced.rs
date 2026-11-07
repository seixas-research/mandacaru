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

#[pyfunction]
#[pyo3(signature = (x, height=None))]
pub fn find_peaks_1d<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray1<'py, f64>,
    height: Option<f64>,
) -> PyResult<crate::arrays::Array<'py, i64>> {
    let s = x.as_slice()?;
    let min_h = height.unwrap_or(f64::NEG_INFINITY);
    let mut peaks = Vec::new();
    let n = s.len();
    if n >= 3 {
        let mut i = 1;
        while i < n - 1 {
            if s[i] > s[i - 1] {
                let mut j = i;
                while j < n - 1 && s[j + 1] == s[i] {
                    j += 1;
                }
                if j < n - 1 && s[j + 1] < s[i] && s[i] >= min_h {
                    peaks.push(i as i64);
                }
                i = j + 1;
            } else {
                i += 1;
            }
        }
    }
    let len = peaks.len();
    array(py, &[len], peaks)
}

#[pyfunction]
#[pyo3(signature = (unitary, tol=1e-12))]
pub fn givens_decomposition<'py>(
    py: Python<'py>,
    unitary: numpy::PyReadonlyArray2<'py, f64>,
    tol: f64,
) -> PyResult<(Vec<(usize, f64)>, crate::arrays::Array<'py, f64>)> {
    let shape = unitary.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected a square matrix, got {:?}",
            shape
        )));
    }
    let m = shape[0];
    let arr = unitary.as_array();
    let mut u = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..m {
            u[i * m + j] = arr[[i, j]];
        }
    }

    for i in 0..m {
        for j in 0..m {
            let mut dot = 0.0;
            for k in 0..m {
                dot += u[i * m + k] * u[j * m + k];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > 1e-8 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the Givens decomposition needs a real orthogonal matrix; this one is not",
                ));
            }
        }
    }

    let mut rotations = Vec::new();
    for column in 0..(m - 1) {
        for row in (column + 1..m).rev() {
            let below = u[row * m + column];
            let above = u[(row - 1) * m + column];
            if below.abs() <= tol {
                continue;
            }
            let theta = below.atan2(above);
            let cos = theta.cos();
            let sin = theta.sin();
            for col in 0..m {
                let r1 = u[(row - 1) * m + col];
                let r2 = u[row * m + col];
                u[(row - 1) * m + col] = cos * r1 + sin * r2;
                u[row * m + col] = -sin * r1 + cos * r2;
            }
            rotations.push((row - 1, theta));
        }
    }

    let signs: Vec<f64> = (0..m).map(|i| u[i * m + i]).collect();
    let py_signs = crate::arrays::array(py, &[m], signs)?;
    Ok((rotations, py_signs))
}
