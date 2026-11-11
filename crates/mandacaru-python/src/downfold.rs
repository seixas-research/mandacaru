//! The downfolded many-body problem of a Wannier fragment as owned arrays. No numerics here:
//! `mandacaru_core::periodic::wannier::downfold` builds the problem, this only hands it over.
use super::{array, error, Array};
use mandacaru_core::{
    periodic::wannier::downfold::{problem::Matrix, Downfolded},
    quantum::C64,
};
use pyo3::prelude::*;

#[pyclass(name = "Downfolded", module = "mandacaru._native", frozen)]
pub struct PyDownfolded {
    pub inner: Downfolded,
    pub screening: Option<String>,
    pub coulomb: String,
}

fn stacked(matrices: &[Matrix]) -> Vec<C64> {
    matrices
        .iter()
        .flat_map(|m| (0..m.nrows()).flat_map(move |i| (0..m.ncols()).map(move |j| m[(i, j)])))
        .collect()
}

#[pymethods]
impl PyDownfolded {
    /// The fragment's modes `(function, (R1, R2, R3))`.
    #[getter]
    fn functions(&self) -> Vec<(usize, [i64; 3])> {
        self.inner.problem.modes.clone()
    }
    /// One channel for a restricted crystal, two (up, down) for a spin-polarized one.
    #[getter]
    fn spins(&self) -> usize {
        self.inner.problem.spins
    }
    #[getter]
    fn n_spatial_orbitals(&self) -> usize {
        self.inner.problem.n
    }
    #[getter]
    fn num_particles(&self) -> (usize, usize) {
        let [up, down] = self.inner.problem.num_particles;
        (up, down)
    }
    #[getter]
    fn dielectric_head(&self) -> Option<f64> {
        self.inner.problem.dielectric_head
    }
    #[getter]
    fn screening(&self) -> Option<String> {
        self.screening.clone()
    }
    #[getter]
    fn coulomb(&self) -> String {
        self.coulomb.clone()
    }
    /// Truncation radius of the interaction (Bohr).
    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius
    }
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }
    #[getter]
    fn kohn_sham<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, C64>> {
        let p = &self.inner.problem;
        array(py, &[p.spins, p.n, p.n], stacked(&p.kohn_sham))
    }
    #[getter]
    fn density_matrix<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, C64>> {
        let p = &self.inner.problem;
        array(py, &[p.spins, p.n, p.n], stacked(&p.density_matrix))
    }
    #[getter]
    fn double_counting<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, C64>> {
        let p = &self.inner.problem;
        array(py, &[p.spins, p.n, p.n], stacked(&p.double_counting))
    }
    /// `h - V^dc` in the Wannier functions (`basis="wannier"`) or the natural orbitals.
    fn one_body<'py>(&self, py: Python<'py>, basis: &str) -> PyResult<Array<'py, C64>> {
        let p = &self.inner.problem;
        array(
            py,
            &[p.spins, p.n, p.n],
            stacked(&p.basis(basis).map_err(error)?.one_body),
        )
    }
    /// `<pq|rs>` of spin pair `(s, t)` in the Wannier functions or the natural orbitals.
    fn two_body<'py>(&self, py: Python<'py>, basis: &str) -> PyResult<Array<'py, C64>> {
        let p = &self.inner.problem;
        let data = p
            .basis(basis)
            .map_err(error)?
            .two_body
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        array(py, &[p.spins, p.spins, p.n, p.n, p.n, p.n], data)
    }
    /// Occupations of the natural orbitals, most occupied first, per spin channel.
    #[getter]
    fn natural_occupations<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        let p = &self.inner.problem;
        array(
            py,
            &[p.spins, p.n],
            p.occupations.iter().flatten().copied().collect(),
        )
    }
    /// The natural orbitals as columns in the fragment's functions, per spin channel.
    #[getter]
    fn natural_rotation<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, C64>> {
        let p = &self.inner.problem;
        array(py, &[p.spins, p.n, p.n], stacked(&p.rotation))
    }
}
