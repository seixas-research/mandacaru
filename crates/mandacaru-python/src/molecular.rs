//! Python/ASE boundary for the Python-independent molecular engine.
use super::quantum::{PyOperator, PyProblem};
use super::{array, copy, error, Array, PySpatialGrid};
use mandacaru_core::{
    molecular::{transform::Model, MeanField, MolecularIntegrals, ScfOptions},
    quantum::{Basis, Encoding, C64},
    Context,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use std::sync::Arc;

fn matrix<'py>(py: Python<'py>, value: &nalgebra::DMatrix<C64>) -> PyResult<Array<'py, C64>> {
    array(
        py,
        &[value.nrows(), value.ncols()],
        (0..value.nrows())
            .flat_map(|i| (0..value.ncols()).map(move |j| value[(i, j)]))
            .collect(),
    )
}
#[pyclass(name = "MolecularIntegrals", module = "mandacaru._native", frozen)]
pub struct PyIntegrals {
    pub inner: Arc<MolecularIntegrals>,
}
#[pymethods]
impl PyIntegrals {
    #[pyo3(signature=(previous,*,max_iterations=300,tolerance=1e-9,diis=true))]
    fn continue_mean_field(
        &self,
        py: Python<'_>,
        previous: &PyMeanField,
        max_iterations: usize,
        tolerance: f64,
        diis: bool,
    ) -> PyResult<PyMeanField> {
        let options = ScfOptions {
            max_iterations,
            tolerance,
            diis,
            guesses: 1,
        };
        let inner = py
            .detach(|| self.inner.continue_mean_field(&previous.inner, &options))
            .map_err(error)?;
        Ok(PyMeanField {
            inner: Arc::new(inner),
            integrals: self.inner.clone(),
        })
    }
    fn at_geometry(&self, py: Python<'_>, geometry: &super::PyGeometry) -> PyResult<Self> {
        py.detach(|| self.inner.at_geometry(geometry.inner.clone()))
            .map(|inner| Self {
                inner: Arc::new(inner),
            })
            .map_err(error)
    }
    #[getter]
    fn has_eri(&self) -> bool {
        self.inner.has_eri()
    }

    #[getter]
    fn resolution_ratios(&self) -> Vec<f64> {
        self.inner.resolution_ratios.clone()
    }
    #[pyo3(signature=(*,charge=0,magnetic_moment=0.0,n_electrons=None))]
    fn particles(
        &self,
        charge: i32,
        magnetic_moment: f64,
        n_electrons: Option<usize>,
    ) -> PyResult<[usize; 2]> {
        self.inner
            .particles(charge, magnetic_moment, n_electrons)
            .map_err(error)
    }
    #[getter]
    fn n_orbitals(&self) -> usize {
        self.inner.m()
    }
    #[getter]
    fn grid(&self) -> PySpatialGrid {
        PySpatialGrid {
            inner: self.inner.grid.clone(),
        }
    }
    #[pyo3(signature=(*,include_samples=false,include_hamiltonian=false))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        include_samples: bool,
        include_hamiltonian: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let m = self.inner.m();
        let out = PyDict::new(py);
        out.set_item("precision", self.inner.samples.precision().name())?;
        for (name, buffer) in [
            ("overlap", &self.inner.overlap),
            ("kinetic", &self.inner.kinetic),
            ("potential", &self.inner.potential),
            ("one_body", &self.inner.one),
        ] {
            out.set_item(name, array(py, &[m, m], buffer.as_slice().to_vec())?)?;
        }
        let lowdin: Vec<_> = (0..m)
            .flat_map(|i| (0..m).map(move |j| self.inner.lowdin[(i, j)]))
            .collect();
        out.set_item("lowdin", array(py, &[m, m], lowdin)?)?;
        if include_hamiltonian {
            let eri = py
                .detach(|| self.inner.eri()?.dense(&self.inner.context))
                .map_err(error)?;
            out.set_item(
                "two_body",
                array(py, &[m, m, m, m], eri.as_slice().to_vec())?,
            )?;
        }
        out.set_item("nuclear_repulsion", self.inner.nuclear)?;
        out.set_item("overlap_min", self.inner.overlap_min)?;
        out.set_item(
            "atom_of_orbital",
            self.inner
                .basis
                .orbitals
                .iter()
                .map(|o| o.atom)
                .collect::<Vec<_>>(),
        )?;
        if include_samples {
            use mandacaru_core::math::storage::RealSamples;
            for (name, samples) in [
                ("samples", &self.inner.samples),
                ("orthonormal_samples", &self.inner.orthonormal_samples),
            ] {
                match samples {
                    RealSamples::Double(values) => out.set_item(
                        name,
                        array(
                            py,
                            &[m, self.inner.grid.index.points()],
                            values.as_slice().to_vec(),
                        )?,
                    )?,
                    RealSamples::Single(values) => out.set_item(
                        name,
                        array(
                            py,
                            &[m, self.inner.grid.index.points()],
                            values.as_slice().to_vec(),
                        )?,
                    )?,
                }
            }
        }
        out.set_item("has_eri", self.inner.has_eri())?;
        if self.inner.has_eri() {
            match self.inner.eri().map_err(error)?.as_ref() {
                mandacaru_core::molecular::integrals::Eri::Augmented {
                    base, correction, ..
                } => {
                    out.set_item("representation", "cholesky+augmentation")?;
                    out.set_item("augmentation_rank", correction.factors.len())?;
                    if let mandacaru_core::molecular::integrals::Eri::Cholesky {
                        rank,
                        residual,
                        ..
                    } = base.as_ref()
                    {
                        out.set_item("cholesky_rank", rank)?;
                        out.set_item("cholesky_residual", residual)?;
                    }
                }
                mandacaru_core::molecular::integrals::Eri::Packed { .. } => {
                    out.set_item("representation", "packed")?
                }
                mandacaru_core::molecular::integrals::Eri::Cholesky { rank, residual, .. } => {
                    out.set_item("representation", "cholesky")?;
                    out.set_item("cholesky_rank", rank)?;
                    out.set_item("cholesky_residual", residual)?;
                }
            }
        }
        Ok(out)
    }
    #[pyo3(signature=(method,num_particles,*,max_iterations=300,tolerance=1e-9,diis=true,guesses=3))]
    #[allow(clippy::too_many_arguments)]
    fn mean_field(
        &self,
        py: Python<'_>,
        method: &str,
        num_particles: [usize; 2],
        max_iterations: usize,
        tolerance: f64,
        diis: bool,
        guesses: usize,
    ) -> PyResult<PyMeanField> {
        let options = ScfOptions {
            max_iterations,
            tolerance,
            diis,
            guesses,
        };
        let inner = py
            .detach(|| self.inner.mean_field(method, num_particles, &options))
            .map_err(error)?;
        Ok(PyMeanField {
            inner: Arc::new(inner),
            integrals: self.inner.clone(),
        })
    }
    #[pyo3(signature = (functional, num_particles, *, unrestricted=false, max_iterations=300, tolerance=1e-9, diis=true, guesses=3))]
    fn kohn_sham(
        &self,
        py: Python<'_>,
        functional: &str,
        num_particles: [usize; 2],
        unrestricted: bool,
        max_iterations: usize,
        tolerance: f64,
        diis: bool,
        guesses: usize,
    ) -> PyResult<PyMeanField> {
        let options = ScfOptions {
            max_iterations,
            tolerance,
            diis,
            guesses,
        };
        let inner = py
            .detach(|| {
                self.inner
                    .kohn_sham(functional, num_particles, unrestricted, &options)
            })
            .map_err(error)?;
        Ok(PyMeanField {
            inner: Arc::new(inner),
            integrals: self.inner.clone(),
        })
    }
}
#[pyclass(name = "MolecularHandover", module = "mandacaru._native", frozen)]
pub struct PyHandover {
    inner: mandacaru_core::quantum::transfer::Handover,
}

#[pyclass(name = "MeanField", module = "mandacaru._native", frozen)]
pub struct PyMeanField {
    inner: Arc<MeanField>,
    integrals: Arc<MolecularIntegrals>,
}
#[pymethods]
impl PyMeanField {
    fn capture_transfer(
        &self,
        py: Python<'_>,
        model: &PyModel,
        problem: &PyProblem,
        indices: Vec<usize>,
        parameters: Vec<f64>,
    ) -> PyResult<PyHandover> {
        use mandacaru_core::quantum::transfer::{self, Frame};
        let inner = py
            .detach(|| {
                transfer::capture(
                    Frame {
                        integrals: &self.integrals,
                        reference: &self.inner,
                        model: model.inner(),
                        problem: &problem.inner,
                    },
                    &indices,
                    &parameters,
                )
            })
            .map_err(error)?;
        Ok(PyHandover { inner })
    }
    #[pyo3(signature=(previous,model,problem,*,threshold=0.9))]
    fn transfer_from<'py>(
        &self,
        py: Python<'py>,
        previous: &PyHandover,
        model: &PyModel,
        problem: &PyProblem,
        threshold: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        use mandacaru_core::quantum::transfer::{self, Frame};
        let seed = py
            .detach(|| {
                transfer::carry(
                    &previous.inner,
                    Frame {
                        integrals: &self.integrals,
                        reference: &self.inner,
                        model: model.inner(),
                        problem: &problem.inner,
                    },
                    threshold,
                )
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("indices", seed.indices)?;
        out.set_item("parameters", seed.parameters)?;
        out.set_item("accepted", seed.accepted)?;
        out.set_item("confidence", seed.confidence)?;
        out.set_item("reason", seed.reason)?;
        Ok(out)
    }
    #[pyo3(signature=(problem,model,state,*,active_spec=None,step=1e-3,internal=false,extrapolate=true))]
    #[allow(clippy::too_many_arguments)]
    fn quantum_gradient<'py>(
        &self,
        py: Python<'py>,
        problem: &PyProblem,
        model: &PyModel,
        state: PyReadonlyArray1<'py, C64>,
        active_spec: Option<&str>,
        step: f64,
        internal: bool,
        extrapolate: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (values, _memory) = copy(&self.integrals.context, &state)?;
        let response = py
            .detach(|| {
                mandacaru_core::molecular::response::gradient(
                    &self.integrals,
                    &self.inner,
                    model.inner(),
                    &problem.inner,
                    &values,
                    &mandacaru_core::molecular::response::Options {
                        active_spec,
                        step,
                        internal,
                        extrapolate,
                    },
                )
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item(
            "gradient",
            array(
                py,
                &[self.integrals.geometry.numbers().len(), 3],
                response.gradient.as_slice().to_vec(),
            )?,
        )?;
        out.set_item("energy_ha", response.energy)?;
        out.set_item("directions", response.directions)?;
        out.set_item("step_bohr", response.step)?;
        out.set_item("scheme", response.scheme)?;
        out.set_item("step_error_estimate", response.estimate)?;
        out.set_item("orbital_response", "displaced-SCF-and-selector")?;
        Ok(out)
    }
    #[pyo3(signature=(*,gamma=None,active=None,frozen=Vec::new(),model=None,grid=None,quantity=None,index=0,component="auto"))]
    fn observables<'py>(
        &self,
        py: Python<'py>,
        gamma: Option<PyReadonlyArray2<'py, C64>>,
        active: Option<Vec<usize>>,
        frozen: Vec<usize>,
        model: Option<&PyModel>,
        grid: Option<&PySpatialGrid>,
        quantity: Option<&str>,
        index: usize,
        component: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let m = self.integrals.m();
        let (active, frozen, rotation) = if let Some(model) = model {
            if active.is_some() || !frozen.is_empty() {
                return Err(PyValueError::new_err(
                    "choose a native orbital model or explicit active/frozen indices",
                ));
            }
            (
                model.inner().orbital_indices.clone(),
                model.inner().frozen_indices.clone(),
                model.inner().orbital_rotation.clone(),
            )
        } else {
            (
                active.unwrap_or_else(|| (0..m).filter(|p| !frozen.contains(p)).collect()),
                frozen,
                None,
            )
        };
        let gamma = if let Some(gamma) = gamma {
            let n = active
                .len()
                .checked_mul(2)
                .ok_or_else(|| error(mandacaru_core::Error::Overflow))?;
            if gamma.shape() != [n, n] {
                return Err(PyValueError::new_err(
                    "one-particle RDM shape must match the active spin orbitals",
                ));
            }
            Some(copy(&self.integrals.context, &gamma)?)
        } else {
            None
        };
        let grid = grid.map_or(&self.integrals.grid, |grid| &grid.inner);
        let (analysis, field) = py
            .detach(|| {
                use mandacaru_core::molecular::observables::{self, ActiveDensity};
                let input = gamma.as_ref().map(|(values, _)| ActiveDensity {
                    gamma: values,
                    active: &active,
                    frozen: &frozen,
                    rotation: rotation.as_ref().map(|r| &r.matrix),
                });
                let analysis = observables::analyze(&self.integrals, &self.inner, input)?;
                let field = quantity
                    .map(|quantity| {
                        observables::sample(
                            &self.integrals,
                            &analysis,
                            grid,
                            quantity,
                            index,
                            component,
                        )
                    })
                    .transpose()?;
                Ok::<_, mandacaru_core::Error>((analysis, field))
            })
            .map_err(error)?;
        let out = PyDict::new(py);
        out.set_item("occupations", array(py, &[m], analysis.occupations)?)?;
        out.set_item("coefficients", matrix(py, &analysis.coefficients)?)?;
        out.set_item(
            "mo_coefficients",
            analysis
                .mo_coefficients
                .as_ref()
                .map(|c| matrix(py, c))
                .transpose()?,
        )?;
        out.set_item("n_electrons", analysis.electrons)?;
        if let Some(field) = field {
            out.set_item(
                "data",
                array(py, &grid.index.shape(), field.values.as_slice().to_vec())?,
            )?;
            out.set_item("integral", field.integral)?;
            out.set_item("norm", field.norm)?;
            out.set_item("component", field.component)?;
            out.set_item("augmentation_charge", field.augmentation_charge)?;
            out.set_item("n_electrons", field.electrons)?;
            out.set_item(
                "charges",
                self.integrals.corrections.as_ref().map_or_else(
                    || {
                        self.integrals
                            .geometry
                            .numbers()
                            .iter()
                            .map(|&z| z as f64)
                            .collect::<Vec<_>>()
                    },
                    |c| c.charges.clone(),
                ),
            )?;
            out.set_item(
                "grid",
                PySpatialGrid {
                    inner: grid.clone(),
                },
            )?;
            out.set_item(
                "numbers",
                self.integrals
                    .geometry
                    .numbers()
                    .iter()
                    .map(|&z| usize::from(z))
                    .collect::<Vec<_>>(),
            )?;
            out.set_item("positions", self.integrals.geometry.positions().to_vec())?;
        }
        Ok(out)
    }
    #[getter]
    fn n_iterations(&self) -> usize {
        self.inner.iterations.len()
    }
    #[getter]
    fn num_particles(&self) -> [usize; 2] {
        self.inner.particles
    }
    #[getter]
    fn n_occupied(&self) -> PyResult<usize> {
        if self.inner.method == "ghf" {
            return Ok(self.inner.particles.iter().sum());
        }
        if self.inner.particles[0] != self.inner.particles[1] {
            return Err(PyValueError::new_err(
                "open-shell orbitals have separate spin occupations",
            ));
        }
        Ok(self.inner.particles[0])
    }
    #[getter]
    fn mo_energies<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        let energies = &self.inner.energies;
        if energies.len() == 1 {
            array(py, &[energies[0].len()], energies[0].clone())
        } else {
            array(
                py,
                &[energies.len(), energies[0].len()],
                energies.iter().flatten().copied().collect(),
            )
        }
    }
    #[getter]
    fn has_model(&self) -> bool {
        self.inner.has_model()
    }

    fn gradient<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let gradient = py
            .detach(|| self.integrals.gradient(&self.inner))
            .map_err(error)?;
        let shape = [self.integrals.geometry.numbers().len(), 3];
        let out = PyDict::new(py);
        out.set_item(
            "gradient",
            array(py, &shape, gradient.total.as_slice().to_vec())?,
        )?;
        for (key, part) in [
            ("hellmann_feynman", &gradient.hellmann_feynman),
            ("pulay", &gradient.pulay),
        ] {
            if let Some(part) = part {
                out.set_item(key, array(py, &shape, part.as_slice().to_vec())?)?;
            }
        }
        out.set_item("scheme", gradient.scheme)?;
        if let Some(step) = gradient.step {
            out.set_item("step_bohr", step)?;
        }
        if let Some(estimate) = gradient.estimate {
            out.set_item("step_error_estimate", estimate)?;
        }
        Ok(out)
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[pyo3(signature=(mapping="jordan_wigner"))]
    fn state<'py>(&self, py: Python<'py>, mapping: &str) -> PyResult<Array<'py, C64>> {
        let out = py
            .detach(|| {
                let m = self.integrals.m();
                let particles = self.inner.particles;
                let encoding = Encoding::new(mapping, 2 * m, particles)?;
                let mut state = self.integrals.context.allocate::<C64>(
                    1usize
                        .checked_shl(encoding.n_qubits() as u32)
                        .ok_or(mandacaru_core::Error::Overflow)?,
                )?;
                if self.inner.method != "uhf" {
                    state.as_mut_slice()[encoding.reference() as usize] = C64::new(1., 0.);
                    return Ok(state);
                }
                let occupation_encoding = Encoding::new("jordan_wigner", 2 * m, particles)?;
                let basis = Basis::new(&self.integrals.context, &occupation_encoding, true)?;
                let rotations: Vec<_> = self
                    .inner
                    .coefficients
                    .iter()
                    .map(|c| self.inner.model_orbitals.adjoint() * c)
                    .collect();
                for &occupation in basis.indices.as_slice() {
                    let mut amplitude = C64::new(1., 0.);
                    for spin in 0..2 {
                        let occupied: Vec<_> = (0..m)
                            .filter(|&i| occupation & (1u64 << (2 * m - 1 - spin * m - i)) != 0)
                            .collect();
                        let determinant =
                            nalgebra::DMatrix::from_fn(particles[spin], particles[spin], |i, j| {
                                rotations[spin][(occupied[i], j)]
                            })
                            .determinant();
                        amplitude *= determinant;
                    }
                    state.as_mut_slice()[encoding.encode(occupation) as usize] = amplitude;
                }
                Ok::<_, mandacaru_core::Error>(state)
            })
            .map_err(error)?;
        array(py, &[out.as_slice().len()], out.as_slice().to_vec())
    }
    fn rdm<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, C64>> {
        if self.inner.method == "ghf" {
            return Err(PyValueError::new_err(
                "GHF spinor population densities are not yet wired",
            ));
        }
        let m = self.integrals.m();
        let out = py
            .detach(|| {
                let _work = self
                    .integrals
                    .context
                    .reserve(mandacaru_core::product(&[m, m, 16, 8])?)?;
                let mut out = self
                    .integrals
                    .context
                    .allocate::<C64>(mandacaru_core::product(&[2 * m, 2 * m])?)?;
                for spin in 0..2 {
                    let density = &self.inner.densities[if self.inner.densities.len() == 1 {
                        0
                    } else {
                        spin
                    }];
                    let transformed =
                        self.inner.model_orbitals.adjoint() * density * &self.inner.model_orbitals;
                    for p in 0..m {
                        for q in 0..m {
                            out.as_mut_slice()[(spin * m + p) * 2 * m + spin * m + q] =
                                transformed[(p, q)];
                        }
                    }
                }
                Ok::<_, mandacaru_core::Error>(out)
            })
            .map_err(error)?;
        array(py, &[2 * m, 2 * m], out.as_slice().to_vec())
    }
    fn active_model(&self, py: Python<'_>, spec: &str) -> PyResult<PyModel> {
        let inner = py
            .detach(|| self.inner.active_model(&self.integrals, spec))
            .map_err(error)?;
        Ok(PyModel {
            context: self.integrals.context.clone(),
            source: Arc::new(inner),
        })
    }
    #[getter]
    fn model(&self, py: Python<'_>) -> PyResult<PyModel> {
        let model = py
            .detach(|| self.inner.model(&self.integrals))
            .map_err(error)?;
        Ok(PyModel {
            context: self.integrals.context.clone(),
            source: model,
        })
    }
    #[pyo3(signature=(*,include_hamiltonian=false))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        include_hamiltonian: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let value = &self.inner;
        let n = value.model_orbitals.ncols();
        let out = PyDict::new(py);
        out.set_item("method", &value.method)?;
        out.set_item("functional", &value.functional)?;
        out.set_item("hartree_energy", value.hartree_energy)?;
        out.set_item("xc_energy", value.xc_energy)?;
        out.set_item("electronic_energy", value.electronic_energy)?;
        out.set_item("reference_energy", value.reference_energy)?;
        out.set_item("nuclear_repulsion", self.integrals.nuclear)?;
        out.set_item("converged", value.converged)?;
        out.set_item("n_iterations", value.iterations.len())?;
        out.set_item("energies", value.energies.clone())?;
        let c: PyResult<Vec<_>> = value.coefficients.iter().map(|c| matrix(py, c)).collect();
        out.set_item("coefficients", c?)?;
        let d: PyResult<Vec<_>> = value.densities.iter().map(|d| matrix(py, d)).collect();
        out.set_item("densities", d?)?;
        out.set_item("model_orbitals", matrix(py, &value.model_orbitals)?)?;
        out.set_item("natural_occupations", value.natural_occupations.clone())?;
        if include_hamiltonian {
            let model = py.detach(|| value.model(&self.integrals)).map_err(error)?;
            out.set_item("h_mo", array(py, &[n, n], model.one.as_slice().to_vec())?)?;
            out.set_item(
                "eri_mo",
                array(py, &[n, n, n, n], model.two.as_slice().to_vec())?,
            )?;
        }
        out.set_item("has_model", value.has_model())?;
        out.set_item("has_eri", self.integrals.has_eri())?;
        out.set_item("num_particles", value.particles)?;
        out.set_item("kramers_pairing", value.kramers_pairing)?;
        out.set_item(
            "history",
            value
                .iterations
                .iter()
                .map(|it| (it.iteration, it.energy, it.delta, it.residual))
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }
}
#[pyclass(name = "MolecularModel", module = "mandacaru._native", frozen)]
pub struct PyModel {
    context: Context,
    source: Arc<Model>,
}
impl PyModel {
    fn inner(&self) -> &Model {
        &self.source
    }
}
#[pymethods]
impl PyModel {
    #[getter]
    fn orbital_symmetry_sets(&self) -> Vec<Vec<usize>> {
        self.inner().orbital_symmetry_sets.clone()
    }
    #[getter]
    fn one_body<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, C64>> {
        let n = if self.inner().spinor {
            2 * self.inner().m
        } else {
            self.inner().m
        };
        array(py, &[n, n], self.inner().one.as_slice().to_vec())
    }
    #[getter]
    fn two_body<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, C64>> {
        let n = if self.inner().spinor {
            2 * self.inner().m
        } else {
            self.inner().m
        };
        array(py, &[n, n, n, n], self.inner().two.as_slice().to_vec())
    }
    #[getter]
    fn orbital_rotation<'py>(&self, py: Python<'py>) -> PyResult<Option<Array<'py, C64>>> {
        self.inner()
            .orbital_rotation
            .as_ref()
            .map(|r| matrix(py, &r.matrix))
            .transpose()
    }
    #[getter]
    fn selection_occupations<'py>(&self, py: Python<'py>) -> PyResult<Option<Array<'py, f64>>> {
        self.inner()
            .orbital_rotation
            .as_ref()
            .map(|r| array(py, &[r.occupations.len()], r.occupations.clone()))
            .transpose()
    }
    #[getter]
    fn mp2_correlation_energy(&self) -> Option<f64> {
        self.inner()
            .orbital_rotation
            .as_ref()
            .map(|r| r.correlation_energy)
    }
    fn mp2<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let model = self.inner();
        if model.spinor || model.particles[0] != model.particles[1] {
            return Err(PyValueError::new_err(
                "native MP2 requires a closed-shell spatial model",
            ));
        }
        let out = py
            .detach(|| {
                mandacaru_core::molecular::correlation::natural_orbitals(
                    &self.context,
                    model.m,
                    model.one.as_slice(),
                    model.two.as_slice(),
                    model.particles[0],
                )
            })
            .map_err(error)?;
        super::advanced::mp2_result(py, out)
    }
    #[getter]
    fn orbital_indices(&self) -> Vec<usize> {
        self.inner().orbital_indices.clone()
    }
    #[getter]
    fn frozen_indices(&self) -> Vec<usize> {
        self.inner().frozen_indices.clone()
    }
    #[getter]
    fn context(&self) -> super::PyContext {
        super::PyContext {
            inner: self.context.clone(),
        }
    }
    #[pyo3(signature=(path,*,mapping="jordan_wigner"))]
    fn save(&self, py: Python<'_>, path: String, mapping: &str) -> PyResult<()> {
        py.detach(|| {
            self.inner()
                .save(&self.context, std::path::Path::new(&path), mapping)
        })
        .map_err(error)
    }
    #[getter]
    fn n_spatial_orbitals(&self) -> usize {
        self.inner().m
    }
    #[getter]
    fn num_particles(&self) -> [usize; 2] {
        self.inner().particles
    }
    #[getter]
    fn constant_energy(&self) -> f64 {
        self.inner().constant
    }
    #[getter]
    fn spinor(&self) -> bool {
        self.inner().spinor
    }
    fn reduce(&self, py: Python<'_>, frozen: Vec<usize>, active: Vec<usize>) -> PyResult<Self> {
        let model = py
            .detach(|| self.inner().reduce(&self.context, &frozen, &active))
            .map_err(error)?;
        Ok(Self {
            context: self.context.clone(),
            source: Arc::new(model),
        })
    }
    fn fermion_terms(&self, py: Python<'_>) -> PyResult<super::quantum::FermionTerms> {
        let n = 2 * self.inner().m;
        let _terms = self
            .context
            .reserve(mandacaru_core::product(&[n, n, n, n, 128]).map_err(error)?)
            .map_err(error)?;
        Ok(py
            .detach(|| self.inner().fermion_terms())
            .into_iter()
            .map(|(l, c)| (l, c.re, c.im))
            .collect())
    }
    #[pyo3(signature=(mapping="jordan_wigner"))]
    fn operator(&self, py: Python<'_>, mapping: &str) -> PyResult<PyOperator> {
        let inner = py
            .detach(|| self.inner().operator(&self.context, mapping))
            .map_err(error)?;
        Ok(PyOperator {
            context: self.context.clone(),
            inner,
        })
    }
    #[pyo3(signature=(*,mapping="jordan_wigner",sector=true,dense=false,include_singles=true,sparse=false,pool="fermionic"))]
    fn problem(
        &self,
        py: Python<'_>,
        mapping: &str,
        sector: bool,
        dense: bool,
        include_singles: bool,
        sparse: bool,
        pool: &str,
    ) -> PyResult<PyProblem> {
        if sparse && dense {
            return Err(PyValueError::new_err(
                "choose either dense or sparse storage",
            ));
        }
        let mut inner = py
            .detach(|| {
                self.inner()
                    .problem(&self.context, mapping, sector, dense, include_singles)
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
        Ok(PyProblem { inner })
    }
    #[pyo3(signature=(mapping="jordan_wigner"))]
    fn reference_state<'py>(&self, py: Python<'py>, mapping: &str) -> PyResult<Array<'py, C64>> {
        let out = py
            .detach(|| {
                let encoding = Encoding::new(mapping, 2 * self.inner().m, self.inner().particles)?;
                let basis = Basis::register(&self.context, encoding.n_qubits())?;
                let mut state = self.context.allocate::<C64>(basis.len())?;
                state.as_mut_slice()[encoding.reference() as usize] = C64::new(1., 0.);
                Ok::<_, mandacaru_core::Error>(state)
            })
            .map_err(error)?;
        array(py, &[out.as_slice().len()], out.as_slice().to_vec())
    }
}

pub fn poisson<'py>(
    py: Python<'py>,
    ctx: &Context,
    grid: &PySpatialGrid,
    density: PyReadonlyArray1<'py, C64>,
    periodic: bool,
    slab_axis: Option<usize>,
) -> PyResult<Array<'py, C64>> {
    let (rho, _memory) = copy(ctx, &density)?;
    let out = py
        .detach(|| {
            let mut solver = if let Some(axis) = slab_axis {
                mandacaru_core::math::fft::Poisson::slab(ctx, &grid.inner, axis)?
            } else if periodic {
                mandacaru_core::math::fft::Poisson::periodic(ctx, &grid.inner)?
            } else {
                mandacaru_core::math::fft::Poisson::isolated(ctx, &grid.inner)?
            };
            let mut out = ctx.allocate(rho.len())?;
            solver.solve_into(&rho, out.as_mut_slice())?;
            Ok::<_, mandacaru_core::Error>(out)
        })
        .map_err(error)?;
    array(py, &[out.as_slice().len()], out.as_slice().to_vec())
}
