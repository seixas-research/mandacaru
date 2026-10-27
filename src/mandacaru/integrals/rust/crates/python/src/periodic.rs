//! Owned periodic inputs and results. Numerical loops run detached in Rust.
use super::{array, copy, error, Array, PyContext};
use mandacaru_core::{
    periodic::{
        spectrum::{self, Smearing},
        Crystal, KPoint, Options, PeriodicResult,
    },
    product,
    quantum::C64,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};
use std::sync::Arc;
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
    fn spins(&self) -> usize {
        self.inner.spins
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
    fn projectability<'py>(
        &self,
        py: Python<'py>,
        trials: PyReadonlyArray2<'py, C64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let targets = trials.shape()[1];
        if trials.shape()[0] != self.crystal.m {
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
                &[c.spins, c.points.len(), c.m],
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
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.crystal;
        let r = &self.inner;
        let nk = c.points.len();
        let m = c.m;
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
        out.set_item("converged", r.converged)?;
        out.set_item("n_iterations", r.history.len())?;
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
        out.set_item(
            "coefficients",
            array(
                py,
                &[s, nk, m, m],
                r.coefficients.iter().flat_map(matrix_values).collect(),
            )?,
        )?;
        out.set_item(
            "density_matrices",
            array(
                py,
                &[s, nk, m, m],
                r.matrices.iter().flat_map(matrix_values).collect(),
            )?,
        )?;
        out.set_item(
            "density",
            array(py, &[s, n], r.density.as_slice().to_vec())?,
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
                spectrum::broaden(
                    &c.context,
                    &energies,
                    &self.inner.eigenvalues,
                    &weights,
                    width_ha,
                    if c.spins == 1 { 2. } else { 1. },
                    if projected {
                        Some((self.inner.projections.as_slice(), c.m))
                    } else {
                        None
                    },
                )
            })
            .map_err(error)?;
        array(
            py,
            &[energies.len(), if projected { c.m } else { 1 }],
            out.as_slice().to_vec(),
        )
    }
    /// Extra density-independent k-point matrices supplied by the setup adapter.
    fn bands<'py>(&self, py: Python<'py>, data_json: &str) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.crystal;
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
                    c.m,
                    c.m.checked_add(c.p)
                        .ok_or(mandacaru_core::Error::Overflow)?,
                    16,
                    16,
                ])?)?;
                let points = raw
                    .iter()
                    .map(|v| {
                        mandacaru_core::periodic::input::kpoint(
                            &c.context,
                            v,
                            c.m,
                            c.p,
                            c.grid.index.points(),
                        )
                    })
                    .collect::<mandacaru_core::Result<Vec<KPoint>>>()?;
                let mut eps = vec![];
                let mut projections = vec![];
                let n = c.grid.index.points();
                for s in 0..c.spins {
                    for p in &points {
                        let (e, _, y) = mandacaru_core::periodic::solver::diagonalize(
                            c,
                            p,
                            &self.inner.potential.as_slice()[s * n..(s + 1) * n],
                            self.inner.moment_potential.as_slice(),
                        )?;
                        eps.extend(e.iter().map(|e| e + c.reference));
                        for a in 0..c.m {
                            for b in 0..c.m {
                                projections.push(y[(a, b)].norm_sqr());
                            }
                        }
                    }
                }
                Ok::<_, mandacaru_core::Error>((eps, projections, memory))
            })
            .map_err(error)?;
        let nk = eps.len() / (c.spins * c.m);
        let out = PyDict::new(py);
        out.set_item("eigenvalues", array(py, &[c.spins, nk, c.m], eps)?)?;
        out.set_item(
            "projections",
            array(py, &[c.spins, nk, c.m, c.m], projections)?,
        )?;
        Ok(out)
    }
}
