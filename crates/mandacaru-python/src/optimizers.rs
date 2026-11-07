//! Variational parameter optimization bindings.
use crate::arrays::{array, error};
use crate::context::PyContext;
use mandacaru_core::quantum::optimizer::{self, Method, Options};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (
    context,
    method,
    x0,
    cost,
    gradient = None,
    max_iterations = 1000,
    max_evaluations = 10000,
    tolerance = None,
    seed = 0,
    track_best = true,
    patience = 5,
    eps = 1.4901161193847656e-8
))]
pub fn native_minimize<'py>(
    py: Python<'py>,
    context: &PyContext,
    method: &str,
    x0: PyReadonlyArray1<'py, f64>,
    cost: Bound<'py, PyAny>,
    gradient: Option<Bound<'py, PyAny>>,
    max_iterations: usize,
    max_evaluations: usize,
    tolerance: Option<f64>,
    seed: u64,
    track_best: bool,
    patience: usize,
    eps: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let initial = x0.as_slice()?.to_vec();
    let n = initial.len();
    let parsed_method = Method::parse(method).map_err(error)?;
    let options = Options {
        method: parsed_method,
        max_iterations,
        max_evaluations,
        tolerance,
        seed,
        track_best,
        patience,
    };

    let is_derivative_free = matches!(
        parsed_method,
        Method::Cobyla | Method::NelderMead | Method::Spsa
    );

    let mut python_err: Option<PyErr> = None;
    let mut objective = |x: &[f64]| -> mandacaru_core::Result<(f64, Vec<f64>)> {
        if python_err.is_some() {
            return Err(mandacaru_core::Error::Kernel("python callback error".into()));
        }
        Python::attach(|py| {
            if let Some(grad_fn) = &gradient {
                let py_x = array(py, &[x.len()], x.to_vec()).map_err(|e| {
                    python_err = Some(e);
                    mandacaru_core::Error::Kernel("array allocation failed".into())
                })?;
                match grad_fn.call1((py_x,)) {
                    Ok(res) => {
                        if let Ok((val, grad_obj)) = res.extract::<(f64, Bound<'_, PyAny>)>() {
                            if let Ok(grad_arr) = grad_obj.extract::<PyReadonlyArray1<'_, f64>>() {
                                if let Ok(grad_slice) = grad_arr.as_slice() {
                                    return Ok((val, grad_slice.to_vec()));
                                }
                            }
                        }
                        python_err = Some(PyValueError::new_err(
                            "gradient callable must return (float, 1D array)",
                        ));
                        Err(mandacaru_core::Error::Kernel("invalid gradient return".into()))
                    }
                    Err(e) => {
                        python_err = Some(e);
                        Err(mandacaru_core::Error::Kernel("python callback error".into()))
                    }
                }
            } else if is_derivative_free {
                let py_x = array(py, &[x.len()], x.to_vec()).map_err(|e| {
                    python_err = Some(e);
                    mandacaru_core::Error::Kernel("array allocation failed".into())
                })?;
                match cost.call1((py_x,)) {
                    Ok(res) => match res.extract::<f64>() {
                        Ok(val) => Ok((val, vec![0.0; x.len()])),
                        Err(e) => {
                            python_err = Some(e);
                            Err(mandacaru_core::Error::Kernel("cost must return float".into()))
                        }
                    },
                    Err(e) => {
                        python_err = Some(e);
                        Err(mandacaru_core::Error::Kernel("python callback error".into()))
                    }
                }
            } else {
                let py_x = array(py, &[x.len()], x.to_vec()).map_err(|e| {
                    python_err = Some(e);
                    mandacaru_core::Error::Kernel("array allocation failed".into())
                })?;
                let f0: f64 = match cost.call1((py_x,)) {
                    Ok(res) => match res.extract::<f64>() {
                        Ok(v) => v,
                        Err(e) => {
                            python_err = Some(e);
                            return Err(mandacaru_core::Error::Kernel("cost must return float".into()));
                        }
                    },
                    Err(e) => {
                        python_err = Some(e);
                        return Err(mandacaru_core::Error::Kernel("python callback error".into()));
                    }
                };

                let mut g = Vec::with_capacity(n);
                let mut x_pert = x.to_vec();
                for i in 0..n {
                    let orig = x_pert[i];
                    let h = eps * orig.abs().max(1.0);
                    x_pert[i] = orig + h;
                    let py_x_pert = array(py, &[x.len()], x_pert.clone()).map_err(|e| {
                        python_err = Some(e);
                        mandacaru_core::Error::Kernel("array allocation failed".into())
                    })?;
                    let fi: f64 = match cost.call1((py_x_pert,)) {
                        Ok(res) => match res.extract::<f64>() {
                            Ok(v) => v,
                            Err(e) => {
                                python_err = Some(e);
                                return Err(mandacaru_core::Error::Kernel("cost must return float".into()));
                            }
                        },
                        Err(e) => {
                            python_err = Some(e);
                            return Err(mandacaru_core::Error::Kernel("python callback error".into()));
                        }
                    };
                    x_pert[i] = orig;
                    g.push((fi - f0) / h);
                }
                Ok((f0, g))
            }
        })
    };

    let outcome = optimizer::minimize(&context.inner, &initial, &options, &mut objective);
    if let Some(err) = python_err {
        return Err(err);
    }
    let outcome = outcome.map_err(error)?;

    let dict = PyDict::new(py);
    let n_res = outcome.parameters.len();
    dict.set_item("x", array(py, &[n_res], outcome.parameters)?)?;
    dict.set_item("fun", outcome.energy)?;
    dict.set_item("steps", outcome.steps)?;
    dict.set_item("success", outcome.success)?;
    dict.set_item("message", outcome.message)?;
    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_minimize, m)?)?;
    Ok(())
}
