//! Owned periodic inputs and results. Numerical loops run detached in Rust.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::{
    math::storage::MatrixSource,
    periodic::{
        spectrum::{self, Smearing},
        Crystal, KPoint, Options, PeriodicResult,
    },
    product,
    quantum::C64,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};
use std::sync::Arc;
pub(super) fn input_arrays(
    ctx: &mandacaru_core::Context,
    values: &Bound<'_, PyDict>,
    metadata: &str,
    precision: Option<mandacaru_core::math::storage::Precision>,
) -> PyResult<mandacaru_core::periodic::input::InputArrays> {
    use mandacaru_core::math::storage::{ComplexSamples, Precision};
    use mandacaru_core::periodic::input::{InputArray, InputArrays};
    let _metadata = ctx
        .reserve(product(&[metadata.len(), 32]).map_err(error)?)
        .map_err(error)?;
    let metadata: serde_json::Value = serde_json::from_str(metadata)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let precision = match precision {
        Some(p) => p,
        None => {
            Precision::parse(metadata["precision"].as_str().unwrap_or("double")).map_err(error)?
        }
    };
    let raw = metadata
        .as_array()
        .or_else(|| metadata["kpoints"].as_array());
    let sample_keys: std::collections::BTreeSet<_> = raw
        .into_iter()
        .flatten()
        .filter_map(|row| row["psi"].as_str())
        .collect();
    let mut out = InputArrays::default();
    let mut sample_bytes = 0usize;
    for key in &sample_keys {
        if let Some(value) = values.get_item(*key)? {
            let length = if let Ok(data) = value.extract::<PyReadonlyArrayDyn<'_, C64>>() {
                data.len()
            } else {
                value
                    .extract::<PyReadonlyArrayDyn<'_, numpy::Complex32>>()?
                    .len()
            };
            let bytes = product(&[
                length,
                if precision == Precision::Single {
                    8
                } else {
                    16
                },
            ])
            .map_err(error)?;
            sample_bytes = sample_bytes
                .checked_add(bytes)
                .ok_or_else(|| error(mandacaru_core::Error::Overflow))?;
        }
    }
    let archive = sample_bytes > ctx.memory_limit() / 8;
    for (key, value) in values.iter() {
        let key: String = key.extract()?;
        let owned = if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            super::layout(&array)?;
            InputArray::Real(ctx.copy_slice(array.as_slice()?).map_err(error)?)
        } else if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, C64>>() {
            super::layout(&array)?;
            if sample_keys.contains(key.as_str()) || metadata["hats"].as_str() == Some(key.as_str())
            {
                let is_hat = metadata["hats"].as_str() == Some(key.as_str());
                let precision = if is_hat { Precision::Double } else { precision };
                let archive = if is_hat {
                    product(&[array.len(), 16]).map_err(error)? > ctx.memory_limit() / 8
                } else {
                    archive
                };
                InputArray::Samples(
                    if archive {
                        ComplexSamples::archived_copy(ctx, array.as_slice()?, precision)
                    } else {
                        ComplexSamples::copy(ctx, array.as_slice()?, precision)
                    }
                    .map_err(error)?,
                )
            } else {
                InputArray::Complex(ctx.copy_slice(array.as_slice()?).map_err(error)?)
            }
        } else if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, numpy::Complex32>>() {
            super::layout(&array)?;
            if !sample_keys.contains(key.as_str()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "complex64 inputs are reserved for Bloch samples",
                ));
            }
            InputArray::Samples(
                ComplexSamples::copy_single(ctx, array.as_slice()?, precision, archive)
                    .map_err(error)?,
            )
        } else if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, u64>>() {
            super::layout(&array)?;
            InputArray::Indices(ctx.copy_slice(array.as_slice()?).map_err(error)?)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array {key} must have float64, complex128, complex64 Bloch samples or uint64 dtype"
            )));
        };
        out.0.insert(key, owned);
    }
    Ok(out)
}
fn matrix_values(a: &nalgebra::DMatrix<C64>) -> Vec<C64> {
    (0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |j| a[(i, j)]))
        .collect()
}
#[pyclass(name = "PeriodicProblem", module = "mandacaru._native", frozen)]
pub struct PyCrystal {
    pub inner: Arc<Crystal>,
}
#[pymethods]
impl PyCrystal {
    #[getter]
    fn context(&self) -> PyContext {
        PyContext {
            inner: self.inner.context.clone(),
        }
    }
    #[getter]
    fn functional(&self) -> &str {
        &self.inner.functional
    }
    #[getter]
    fn precision(&self) -> &'static str {
        self.inner.precision.name()
    }
    #[getter]
    fn spins(&self) -> usize {
        self.inner.spins
    }
    #[getter]
    fn spinor(&self) -> bool {
        self.inner.spinor
    }
    #[getter]
    fn time_reversal(&self) -> bool {
        self.inner.time_reversal
    }
    #[getter]
    fn n_orbitals(&self) -> usize {
        self.inner.m
    }
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.inner.save(path)).map_err(error)
    }
    #[pyo3(signature=(*,max_iterations=300,tolerance=1e-7,density_tolerance=1e-5,smearing="fermi-dirac",width_ha=0.003674932217565499,mixing=0.25,history=7))]
    fn solve(
        &self,
        py: Python<'_>,
        max_iterations: usize,
        tolerance: f64,
        density_tolerance: f64,
        smearing: &str,
        width_ha: f64,
        mixing: f64,
        history: usize,
    ) -> PyResult<PyPeriodicResult> {
        let options = Options {
            max_iterations,
            tolerance,
            density_tolerance,
            smearing: Smearing::parse(smearing).map_err(error)?,
            width: width_ha,
            mixing,
            history,
        };
        let inner = py.detach(|| self.inner.solve(&options)).map_err(error)?;
        Ok(PyPeriodicResult {
            inner: Arc::new(inner),
            crystal: self.inner.clone(),
        })
    }
}
#[pyclass(name = "PeriodicResult", module = "mandacaru._native", frozen)]
pub struct PyPeriodicResult {
    pub inner: Arc<PeriodicResult>,
    pub crystal: Arc<Crystal>,
}
#[pymethods]
impl PyPeriodicResult {
    fn wannier_setup(
        &self,
        py: Python<'_>,
        metadata_json: &str,
    ) -> PyResult<super::wannier::PyWannierSetup> {
        let inner = py
            .detach(|| {
                mandacaru_core::periodic::wannier::setup::Setup::from_json(
                    self.crystal.clone(),
                    self.inner.clone(),
                    metadata_json,
                )
            })
            .map_err(error)?;
        Ok(super::wannier::PyWannierSetup { inner: Some(inner) })
    }
    #[getter]
    fn spin_orbit_energy(&self) -> Option<f64> {
        self.inner.spin_orbit_energy
    }
    #[getter]
    fn n_iterations(&self) -> usize {
        self.inner.history.len()
    }
    fn projectability<'py>(
        &self,
        py: Python<'py>,
        trials: PyReadonlyArray2<'py, C64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let targets = trials.shape()[1];
        if trials.shape()[0] != self.crystal.bands() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "trials need shape (n_basis,n_targets)",
            ));
        }
        let (values, _input) = copy(&self.crystal.context, &trials)?;
        let out = py
            .detach(|| {
                mandacaru_core::periodic::analysis::projectability(
                    &self.crystal,
                    &self.inner,
                    &values,
                    targets,
                )
            })
            .map_err(error)?;
        let c = &self.crystal;
        let result = PyDict::new(py);
        result.set_item(
            "weights",
            array(
                py,
                &[c.spins, c.points.len(), c.bands()],
                out.weights.as_slice().to_vec(),
            )?,
        )?;
        result.set_item(
            "hamiltonians",
            array(
                py,
                &[c.spins, c.points.len(), targets, targets],
                out.hamiltonians.as_slice().to_vec(),
            )?,
        )?;
        Ok(result)
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[pyo3(signature=(*,include_matrices=true))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        include_matrices: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.crystal;
        let r = &self.inner;
        let nk = c.points.len();
        let m = c.bands();
        let s = c.spins;
        let n = c.grid.index.points();
        let out = PyDict::new(py);
        for (name, value) in [
            ("energy", r.energy),
            ("free_energy", r.free_energy),
            ("extrapolated_energy", r.extrapolated_energy),
            ("fermi_level", r.fermi),
            ("magnetic_moment", r.magnetization),
        ] {
            out.set_item(name, value)?;
        }
        out.set_item("functional", &c.functional)?;
        out.set_item("n_spins", s)?;
        out.set_item("spinor", c.spinor)?;
        out.set_item("time_reversal", c.time_reversal)?;
        out.set_item("spin_orbit_energy", r.spin_orbit_energy)?;
        out.set_item("converged", r.converged)?;
        out.set_item("n_iterations", r.history.len())?;
        out.set_item("precision", c.precision.name())?;
        if let Some(tau) = &r.kinetic_density {
            out.set_item(
                "kinetic_density",
                array(py, &[s, n], tau.as_slice().to_vec())?,
            )?;
            out.set_item(
                "kinetic_residuals",
                r.history
                    .iter()
                    .map(|it| it.kinetic_residual)
                    .collect::<Vec<_>>(),
            )?;
        }
        out.set_item("smearing", (r.options.smearing.name(), r.options.width))?;
        out.set_item(
            "eigenvalues",
            array(
                py,
                &[s, nk, m],
                r.eigenvalues.iter().flatten().copied().collect(),
            )?,
        )?;
        out.set_item(
            "occupations",
            array(
                py,
                &[s, nk, m],
                r.occupations.iter().flatten().copied().collect(),
            )?,
        )?;
        if include_matrices {
            let coefficients = r
                .coefficients
                .iter()
                .map(|v| v.matrix(&c.context).map(|v| matrix_values(&v)))
                .collect::<mandacaru_core::Result<Vec<_>>>()
                .map_err(error)?;
            out.set_item(
                "coefficients",
                array(
                    py,
                    &[s, nk, m, m],
                    coefficients.into_iter().flatten().collect(),
                )?,
            )?;
            let matrices = r
                .matrices
                .iter()
                .map(|v| v.matrix(&c.context).map(|v| matrix_values(&v)))
                .collect::<mandacaru_core::Result<Vec<_>>>()
                .map_err(error)?;
            out.set_item(
                "density_matrices",
                array(py, &[s, nk, m, m], matrices.into_iter().flatten().collect())?,
            )?;
        }
        out.set_item(
            "density",
            array(py, &[c.field_channels(), n], r.density.as_slice().to_vec())?,
        )?;
        out.set_item(
            "moments",
            array(py, &[c.channels], r.moments.as_slice().to_vec())?,
        )?;
        out.set_item(
            "kpoints",
            array(py, &[nk, 3], c.points.iter().flat_map(|p| p.k).collect())?,
        )?;
        out.set_item(
            "fractional_kpoints",
            array(
                py,
                &[nk, 3],
                c.points.iter().flat_map(|p| p.fractional).collect(),
            )?,
        )?;
        out.set_item(
            "weights",
            c.points.iter().map(|p| p.weight).collect::<Vec<_>>(),
        )?;
        let terms = PyDict::new(py);
        for (name, value) in ["band_fixed", "electrostatic", "xc", "constant", "entropy"]
            .iter()
            .zip(r.terms)
        {
            terms.set_item(*name, value)?;
        }
        out.set_item("terms", terms)?;
        out.set_item(
            "history",
            r.history
                .iter()
                .map(|i| {
                    (
                        i.iteration,
                        i.free_energy,
                        i.delta,
                        i.residual,
                        i.moment_residual,
                        i.magnetization,
                    )
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }
    #[pyo3(signature=(energies_ha,*,width_ha,projected=false))]
    fn dos<'py>(
        &self,
        py: Python<'py>,
        energies_ha: PyReadonlyArray1<'py, f64>,
        width_ha: f64,
        projected: bool,
    ) -> PyResult<Array<'py, f64>> {
        let c = &self.crystal;
        let (energies, _memory) = copy(&c.context, &energies_ha)?;
        let out = py
            .detach(|| {
                let weights = (0..c.spins)
                    .flat_map(|_| c.points.iter().map(|p| p.weight))
                    .collect::<Vec<_>>();
                if projected {
                    return mandacaru_core::periodic::analysis::projected_dos(
                        c,
                        &self.inner,
                        &energies,
                        width_ha,
                    );
                }
                spectrum::broaden(
                    &c.context,
                    &energies,
                    &self.inner.eigenvalues,
                    &weights,
                    width_ha,
                    c.degeneracy(),
                    None,
                )
            })
            .map_err(error)?;
        array(
            py,
            &[energies.len(), if projected { c.bands() } else { 1 }],
            out.as_slice().to_vec(),
        )
    }
    /// Extra density-independent k-point matrices supplied by the setup adapter.
    fn bands<'py>(
        &self,
        py: Python<'py>,
        data_json: &str,
        arrays: &Bound<'_, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.crystal;
        let mut arrays = input_arrays(&c.context, arrays, data_json, Some(c.precision))?;
        let _parse = c
            .context
            .reserve(product(&[data_json.len(), 32]).map_err(error)?)
            .map_err(error)?;
        let owned = data_json.to_owned();
        let (eps, projections, _memory) = py
            .detach(|| {
                let v: serde_json::Value = serde_json::from_str(&owned)
                    .map_err(|e| mandacaru_core::Error::Invalid(e.to_string()))?;
                let raw = v.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
                    mandacaru_core::Error::Invalid("bands need a nonempty k-point list".into())
                })?;
                let memory = c.context.reserve(product(&[
                    raw.len(),
                    c.spins,
                    c.bands(),
                    c.bands()
                        .checked_add(c.p)
                        .ok_or(mandacaru_core::Error::Overflow)?,
                    16,
                    16,
                ])?)?;
                let points = raw
                    .iter()
                    .map(|v| {
                        mandacaru_core::periodic::input::kpoint_arrays(
                            &c.context,
                            v,
                            c.m,
                            c.p,
                            c.grid.index.points(),
                            &mut arrays,
                            c.precision,
                        )
                    })
                    .collect::<mandacaru_core::Result<Vec<KPoint>>>()?;
                if !arrays.0.is_empty() {
                    return Err(mandacaru_core::Error::Invalid(
                        "unreferenced band input arrays".into(),
                    ));
                }
                let mut eps = vec![];
                let mut projections = vec![];
                let n = c.grid.index.points();
                for s in 0..c.spins {
                    for p in &points {
                        let (e, _, y) = mandacaru_core::periodic::solver::diagonalize(
                            c,
                            p,
                            if c.spinor {
                                self.inner.potential.as_slice()
                            } else {
                                &self.inner.potential.as_slice()[s * n..(s + 1) * n]
                            },
                            self.inner.moment_potential.as_slice(),
                            self.inner
                                .kinetic_potential
                                .as_ref()
                                .map(|v| &v.as_slice()[s * n..(s + 1) * n]),
                        )?;
                        eps.extend(e.iter().map(|e| e + c.reference));
                        for a in 0..c.bands() {
                            for b in 0..c.bands() {
                                projections.push(y[(a, b)].norm_sqr());
                            }
                        }
                    }
                }
                Ok::<_, mandacaru_core::Error>((eps, projections, memory))
            })
            .map_err(error)?;
        let nk = eps.len() / (c.spins * c.bands());
        let out = PyDict::new(py);
        out.set_item("eigenvalues", array(py, &[c.spins, nk, c.bands()], eps)?)?;
        out.set_item(
            "projections",
            array(py, &[c.spins, nk, c.bands(), c.bands()], projections)?,
        )?;
        Ok(out)
    }
}
