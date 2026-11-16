//! The compressed-row view of an operator's matrix (`PauliOperator.sparse_matrix`): shape, the
//! stored arrays, the conjugate and transpose, the dense form and `@` with a dense operand. The
//! numerics are `math::csr`; this copies arrays in and out.
use crate::arrays::array;
use mandacaru_core::math::csr::ComplexCsr;
use numpy::{Complex64, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

#[derive(Clone)]
#[pyclass(name = "CSRMatrix")]
pub struct PyCSRMatrix {
    pub inner: ComplexCsr,
}

#[pymethods]
impl PyCSRMatrix {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.rows, self.inner.cols)
    }
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }
    #[getter]
    fn format(&self) -> &'static str {
        "csr"
    }
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(array(py, &[self.inner.nnz()], self.inner.values.clone())?.into_any())
    }
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let indices: Vec<i64> = self.inner.columns.iter().map(|&c| c as i64).collect();
        Ok(array(py, &[indices.len()], indices)?.into_any())
    }
    #[getter]
    fn indptr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let offsets: Vec<i64> = self.inner.offsets.iter().map(|&o| o as i64).collect();
        Ok(array(py, &[offsets.len()], offsets)?.into_any())
    }
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("numpy")?.getattr("complex128")
    }
    #[getter]
    #[allow(non_snake_case)]
    fn T(&self) -> Self {
        self.transpose()
    }
    fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }
    fn conj(&self) -> Self {
        Self {
            inner: self.inner.conj(),
        }
    }
    fn conjugate(&self) -> Self {
        self.conj()
    }
    fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(array(py, &[self.inner.rows, self.inner.cols], self.inner.dense())?.into_any())
    }
    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rows = self.inner.rows;
        if let Ok(v) = other.extract::<PyReadonlyArray1<'_, Complex64>>() {
            let v = v.as_array().to_vec();
            return Ok(array(py, &[rows], self.inner.apply(&v, 1, |a, x| a * x))?.into_any());
        }
        if let Ok(v) = other.extract::<PyReadonlyArray1<'_, f64>>() {
            let v = v.as_array().to_vec();
            return Ok(array(py, &[rows], self.inner.apply(&v, 1, |a, x| a * x))?.into_any());
        }
        if let Ok(m) = other.extract::<PyReadonlyArray2<'_, Complex64>>() {
            let width = m.as_array().shape()[1];
            let m: Vec<Complex64> = m.as_array().iter().copied().collect();
            return Ok(array(
                py,
                &[rows, width],
                self.inner.apply(&m, width, |a, x| a * x),
            )?
            .into_any());
        }
        if let Ok(m) = other.extract::<PyReadonlyArray2<'_, f64>>() {
            let width = m.as_array().shape()[1];
            let m: Vec<f64> = m.as_array().iter().copied().collect();
            return Ok(array(
                py,
                &[rows, width],
                self.inner.apply(&m, width, |a, x| a * x),
            )?
            .into_any());
        }
        Err(PyValueError::new_err(
            "CSRMatrix @ takes a one- or two-dimensional float or complex array",
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCSRMatrix>()?;
    Ok(())
}
