//! Owned Wannier setup/result bindings. Numerical work detaches from Python.
use super::{array, copy, error, periodic::input_arrays, Array};
use mandacaru_core::{
    periodic::{
        input::kpoint_arrays,
        wannier::{
            setup::{Setup, Wannier},
            Options,
        },
    },
    product,
    quantum::C64,
};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};
use std::sync::Arc;

#[pyclass(name = "WannierSetup", module = "mandacaru._native")]
pub struct PyWannierSetup {
    pub inner: Option<Setup>,
}
#[pymethods]
impl PyWannierSetup {
    fn add_point(
        &mut self,
        py: Python<'_>,
        metadata_json: &str,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let setup = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Wannier setup has already been consumed")
        })?;
        let c = setup.crystal.clone();
        let _metadata = c
            .context
            .reserve(product(&[metadata_json.len(), 32]).map_err(error)?)
            .map_err(error)?;
        let metadata: serde_json::Value = serde_json::from_str(metadata_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let raw = metadata
            .as_array()
            .filter(|v| v.len() == 1)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "transfer one full-mesh Wannier k-point at a time",
                )
            })?;
        let mut arrays = input_arrays(&c.context, values, metadata_json, Some(c.precision))?;
        let _work = c
            .context
            .reserve(
                product(&[
                    c.bands(),
                    c.bands()
                        .checked_add(c.p)
                        .ok_or_else(|| error(mandacaru_core::Error::Overflow))?,
                    16,
                    16,
                ])
                .map_err(error)?,
            )
            .map_err(error)?;
        py.detach(|| {
            let point = kpoint_arrays(
                &c.context,
                &raw[0],
                c.m,
                c.p,
                c.grid.index.points(),
                &mut arrays,
                c.precision,
            )?;
            if !arrays.0.is_empty() {
                return Err(mandacaru_core::Error::Invalid(
                    "unreferenced Wannier k-point arrays".into(),
                ));
            }
            setup.add_point(point)
        })
        .map_err(error)
    }
    #[pyo3(signature=(*,max_iterations=2000,tolerance=1e-10,step=0.5))]
    fn localize(
        &mut self,
        py: Python<'_>,
        max_iterations: usize,
        tolerance: f64,
        step: f64,
    ) -> PyResult<PyWannierResult> {
        let setup = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Wannier setup has already been consumed")
        })?;
        let inner = py
            .detach(|| {
                setup.finish(&Options {
                    max_iterations,
                    tolerance,
                    step,
                })
            })
            .map_err(error)?;
        Ok(PyWannierResult {
            inner: Arc::new(inner),
        })
    }
}
#[pyclass(name = "WannierResult", module = "mandacaru._native", frozen)]
pub struct PyWannierResult {
    pub inner: Arc<Wannier>,
}
#[pymethods]
impl PyWannierResult {
    #[getter]
    fn n_spins(&self) -> usize {
        self.inner.channels.len()
    }
    #[pyo3(signature=(*,spin=0))]
    fn snapshot<'py>(&self, py: Python<'py>, spin: usize) -> PyResult<Bound<'py, PyDict>> {
        let owner = &self.inner;
        let channel = owner.channels.get(spin).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid Wannier spin channel")
        })?;
        let localized = &channel.localized;
        let n = owner.trials.len();
        let out = PyDict::new(py);
        out.set_item(
            "centers",
            array(
                py,
                &[n, 3],
                localized
                    .spread
                    .centers
                    .as_slice()
                    .iter()
                    .flatten()
                    .copied()
                    .collect(),
            )?,
        )?;
        out.set_item(
            "spreads",
            array(
                py,
                &[n],
                localized
                    .spread
                    .second_moments
                    .as_slice()
                    .iter()
                    .zip(localized.spread.centers.as_slice())
                    .map(|(second, r)| second - r.iter().map(|v| v * v).sum::<f64>())
                    .collect(),
            )?,
        )?;
        out.set_item("omega_invariant", localized.spread.invariant)?;
        out.set_item("omega_offdiagonal", localized.spread.off_diagonal)?;
        out.set_item("omega_diagonal", localized.spread.diagonal)?;
        out.set_item("converged", localized.converged)?;
        out.set_item(
            "disentanglement_converged",
            channel.disentanglement_converged,
        )?;
        out.set_item(
            "history",
            array(
                py,
                &[localized.history.len(), 3],
                localized.history.iter().flatten().copied().collect(),
            )?,
        )?;
        out.set_item("disentanglement", channel.disentanglement.clone())?;
        out.set_item(
            "unitary",
            array(
                py,
                &[localized.gauge.len(), owner.bands.len(), n],
                localized
                    .gauge
                    .iter()
                    .flat_map(|m| {
                        (0..m.nrows()).flat_map(move |i| (0..m.ncols()).map(move |j| m[(i, j)]))
                    })
                    .collect::<Vec<C64>>(),
            )?,
        )?;
        out.set_item(
            "kpoints",
            array(
                py,
                &[channel.frames.len(), 3],
                channel.frames.iter().flat_map(|f| f.fractional).collect(),
            )?,
        )?;
        out.set_item(
            "lattice_vectors",
            array(
                py,
                &[channel.hoppings.vectors.as_slice().len(), 3],
                channel
                    .hoppings
                    .vectors
                    .as_slice()
                    .iter()
                    .flatten()
                    .copied()
                    .collect(),
            )?,
        )?;
        out.set_item(
            "hamiltonian_R",
            array(
                py,
                &[channel.hoppings.vectors.as_slice().len(), n, n],
                channel.hoppings.values.as_slice().to_vec(),
            )?,
        )?;
        out.set_item(
            "labels",
            owner
                .trials
                .iter()
                .map(|t| t.label.clone())
                .collect::<Vec<_>>(),
        )?;
        out.set_item("bands", owner.bands.clone())?;
        out.set_item("size", owner.size)?;
        Ok(out)
    }
    #[pyo3(signature=(fractional_kpoints,*,spin=0))]
    fn interpolate<'py>(
        &self,
        py: Python<'py>,
        fractional_kpoints: PyReadonlyArray2<'py, f64>,
        spin: usize,
    ) -> PyResult<Array<'py, f64>> {
        if fractional_kpoints.shape()[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fractional k-points need shape (n,3)",
            ));
        }
        let channel = self.inner.channels.get(spin).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("invalid Wannier spin channel")
        })?;
        let ctx = &self.inner.crystal.context;
        let (values, _memory) = copy(ctx, &fractional_kpoints)?;
        let points: Vec<[f64; 3]> = values.chunks_exact(3).map(|r| [r[0], r[1], r[2]]).collect();
        let out = py
            .detach(|| channel.hoppings.interpolate(ctx, &points))
            .map_err(error)?;
        array(
            py,
            &[points.len(), channel.hoppings.functions],
            out.as_slice().to_vec(),
        )
    }
}
