use super::{array, error, Array};
use mandacaru_core::{persistence::Document, Context};
use pyo3::{prelude::*, types::PyDict};
use std::{path::Path, sync::Mutex};

#[pyclass(name = "Snapshot", module = "mandacaru._native", frozen)]
pub struct PySnapshot {
    pub context: Context,
    pub inner: Document,
}
#[pymethods]
impl PySnapshot {
    #[getter]
    fn json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.text()).map_err(error)
    }
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.inner.save(Path::new(path)))
            .map_err(error)
    }
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, mandacaru_core::quantum::C64>> {
        let state = py
            .detach(|| self.inner.state(&self.context))
            .map_err(error)?;
        array(py, &[state.as_slice().len()], state.as_slice().to_vec())
    }
    fn expectation(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.expectation(&self.context))
            .map_err(error)
    }
    #[pyo3(signature=(*,sector=true,dense=false,include_singles=true,sparse=false,pool="fermionic"))]
    fn problem(
        &self,
        py: Python<'_>,
        sector: bool,
        dense: bool,
        include_singles: bool,
        sparse: bool,
        pool: &str,
    ) -> PyResult<super::quantum::PyProblem> {
        let mut inner = py
            .detach(|| {
                self.inner
                    .problem(&self.context, sector, dense, include_singles)
            })
            .map_err(error)?;
        if pool != "fermionic" {
            py.detach(|| inner.use_pool(pool, include_singles))
                .map_err(error)?;
        }
        if sparse {
            inner.sparse = Some(
                py.detach(|| {
                    mandacaru_core::quantum::sparse::SparseMatrix::new(
                        &self.context,
                        &inner.basis,
                        &inner.hamiltonian,
                    )
                })
                .map_err(error)?,
            );
        }
        Ok(super::quantum::PyProblem { inner })
    }
}
#[pyclass(name = "RunLedger", module = "mandacaru._native", frozen)]
pub struct PyLedger {
    pub inner: Mutex<mandacaru_core::execution::RunLedger>,
}
#[pymethods]
impl PyLedger {
    fn reserve(&self, py: Python<'_>, jobs: u64, shots: u64) -> PyResult<Vec<u64>> {
        py.detach(|| self.inner.lock().unwrap().reserve(jobs, shots))
            .map_err(error)
    }
    fn finish(&self, py: Python<'_>, ticket: u64, success: bool) -> PyResult<()> {
        py.detach(|| self.inner.lock().unwrap().finish(ticket, success))
            .map_err(error)
    }
    #[getter]
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let usage = self.inner.lock().unwrap().usage.clone();
        let out = PyDict::new(py);
        out.set_item("attempted_jobs", usage.attempted_jobs)?;
        out.set_item("completed_jobs", usage.completed_jobs)?;
        out.set_item("failed_jobs", usage.failed_jobs)?;
        out.set_item("total_shots", usage.total_shots)?;
        Ok(out)
    }
}
