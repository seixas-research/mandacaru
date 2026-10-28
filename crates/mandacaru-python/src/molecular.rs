//! Python/ASE boundary for the Python-independent molecular engine.
use super::quantum::{PyOperator, PyProblem};
use super::{array, copy, error, Array, PySpatialGrid};
use mandacaru_core::{
    molecular::{transform::Model, MeanField, MolecularIntegrals, ScfOptions},
    quantum::{Basis, Encoding, C64},
    Context,
};
use numpy::PyReadonlyArray1;
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
            out.set_item(
                "samples",
                array(
                    py,
                    &[m, self.inner.grid.index.points()],
                    self.inner.samples.as_slice().to_vec(),
                )?,
            )?;
            out.set_item(
                "orthonormal_samples",
                array(
                    py,
                    &[m, self.inner.grid.index.points()],
                    self.inner.orthonormal_samples.as_slice().to_vec(),
                )?,
            )?;
        }
        out.set_item("has_eri", self.inner.has_eri())?;
        if self.inner.has_eri() {
            match self.inner.eri().map_err(error)?.as_ref() {
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
#[pyclass(name = "MeanField", module = "mandacaru._native", frozen)]
pub struct PyMeanField {
    inner: Arc<MeanField>,
    integrals: Arc<MolecularIntegrals>,
}
#[pymethods]
impl PyMeanField {
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
        out.set_item(
            "hellmann_feynman",
            array(py, &shape, gradient.hellmann_feynman.as_slice().to_vec())?,
        )?;
        out.set_item(
            "pulay",
            array(py, &shape, gradient.pulay.as_slice().to_vec())?,
        )?;
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
            .detach(|| {
                self.inner.model(&self.integrals)?.select(
                    &self.integrals.context,
                    self.integrals.geometry.numbers(),
                    spec,
                )
            })
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
    #[pyo3(signature=(*,mapping="jordan_wigner",sector=true,dense=false,include_singles=true,sparse=false))]
    fn problem(
        &self,
        py: Python<'_>,
        mapping: &str,
        sector: bool,
        dense: bool,
        include_singles: bool,
        sparse: bool,
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
