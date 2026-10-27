//! Python boundary: inputs are copied into Rust before detaching the interpreter;
//! returned NumPy arrays own independent storage. No borrowed NumPy memory is
//! retained by a calculation or accessed from a detached worker.

use mandacaru_core::{
    kernels, product, Buffer, Context, Error, Geometry, Grid, LengthUnit, Reservation,
};
use numpy::{
    ndarray::{ArrayD, Dimension, IxDyn},
    Complex64, IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyMemoryError, PyOverflowError, PyRuntimeError, PyValueError},
    prelude::*,
};

type Array<'py, T> = Bound<'py, PyArrayDyn<T>>;
type NativeComplex = mandacaru_core::Complex64;
mod molecular;
mod quantum;

fn error(error: Error) -> PyErr {
    match error {
        Error::Invalid(_) => PyValueError::new_err(error.to_string()),
        Error::Overflow => PyOverflowError::new_err(error.to_string()),
        Error::Budget { .. } | Error::Allocation => PyMemoryError::new_err(error.to_string()),
        Error::Kernel(_) => PyRuntimeError::new_err(error.to_string()),
    }
}

fn array<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    data: Vec<T>,
) -> PyResult<Array<'py, T>> {
    Ok(ArrayD::from_shape_vec(IxDyn(shape), data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .into_pyarray(py))
}

fn layout<T: numpy::Element, D: Dimension>(input: &PyReadonlyArray<'_, T, D>) -> PyResult<()> {
    if !input.is_c_contiguous() || (input.data() as usize) % std::mem::align_of::<T>() != 0 {
        return Err(PyValueError::new_err(
            "array must be aligned and C-contiguous",
        ));
    }
    Ok(())
}

fn copy<T: numpy::Element + Copy, D: Dimension>(
    ctx: &Context,
    input: &PyReadonlyArray<'_, T, D>,
) -> PyResult<(Vec<T>, Reservation)> {
    layout(input)?;
    let reservation = ctx
        .reserve(product(&[input.len(), std::mem::size_of::<T>()]).map_err(error)?)
        .map_err(error)?;
    Ok((input.as_slice()?.to_vec(), reservation))
}

fn complex<D: Dimension>(
    ctx: &Context,
    input: &PyReadonlyArray<'_, Complex64, D>,
) -> PyResult<(Vec<NativeComplex>, Reservation)> {
    layout(input)?;
    let reservation = ctx
        .reserve(product(&[input.len(), 16]).map_err(error)?)
        .map_err(error)?;
    Ok((
        input
            .as_slice()?
            .iter()
            .map(|v| NativeComplex::new(v.re, v.im))
            .collect(),
        reservation,
    ))
}

fn complex_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    data: &Buffer<NativeComplex>,
) -> PyResult<Array<'py, Complex64>> {
    array(
        py,
        shape,
        data.as_slice()
            .iter()
            .map(|v| Complex64::new(v.re, v.im))
            .collect(),
    )
}

#[pyclass(name = "Geometry", module = "mandacaru._native", frozen)]
struct PyGeometry {
    inner: Geometry,
}
#[pymethods]
impl PyGeometry {
    #[new]
    #[pyo3(signature = (numbers, positions, cell, pbc=[false;3], units="angstrom"))]
    fn new(
        numbers: Vec<u8>,
        positions: PyReadonlyArray2<'_, f64>,
        cell: PyReadonlyArray2<'_, f64>,
        pbc: [bool; 3],
        units: &str,
    ) -> PyResult<Self> {
        layout(&positions)?;
        layout(&cell)?;
        if positions.shape() != [numbers.len(), 3] || cell.shape() != [3, 3] {
            return Err(PyValueError::new_err(
                "positions must be (N,3) and cell must be (3,3)",
            ));
        }
        let positions = positions
            .as_slice()?
            .chunks_exact(3)
            .map(|r| [r[0], r[1], r[2]])
            .collect();
        let c = cell.as_slice()?;
        Ok(Self {
            inner: Geometry::new(
                numbers,
                positions,
                [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]],
                pbc,
                LengthUnit::parse(units).map_err(error)?,
            )
            .map_err(error)?,
        })
    }
    #[getter]
    fn numbers(&self) -> Vec<u16> {
        self.inner.numbers().iter().map(|&z| u16::from(z)).collect()
    }
    #[getter]
    fn pbc(&self) -> [bool; 3] {
        self.inner.pbc()
    }
    #[getter]
    fn positions_bohr<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        array(
            py,
            &[self.inner.numbers().len(), 3],
            self.inner.positions().iter().flatten().copied().collect(),
        )
    }
    #[getter]
    fn cell_bohr<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        array(
            py,
            &[3, 3],
            self.inner.cell().iter().flatten().copied().collect(),
        )
    }
}

#[pyclass(name = "Grid", module = "mandacaru._native", frozen)]
struct PyGrid {
    inner: Grid,
}
#[pymethods]
impl PyGrid {
    #[new]
    fn new(
        shape: [usize; 3],
        inverse_metric: PyReadonlyArray2<'_, f64>,
        dv_bohr3: f64,
    ) -> PyResult<Self> {
        layout(&inverse_metric)?;
        if inverse_metric.shape() != [3, 3] {
            return Err(PyValueError::new_err("inverse metric must be (3,3)"));
        }
        let mut metric = [0.0; 9];
        metric.copy_from_slice(inverse_metric.as_slice()?);
        Ok(Self {
            inner: Grid::new(shape, metric, dv_bohr3).map_err(error)?,
        })
    }
    #[staticmethod]
    fn orthogonal(shape: [usize; 3], spacing_bohr: [f64; 3]) -> PyResult<Self> {
        Ok(Self {
            inner: Grid::orthogonal(shape, spacing_bohr).map_err(error)?,
        })
    }
    #[getter]
    fn shape(&self) -> [usize; 3] {
        self.inner.shape()
    }
    #[getter]
    fn points(&self) -> usize {
        self.inner.points()
    }
    #[getter]
    fn dv_bohr3(&self) -> f64 {
        self.inner.dv()
    }
    #[getter]
    fn inverse_metric<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        array(py, &[3, 3], self.inner.metric().to_vec())
    }
}

#[pyclass(name = "Workspace", module = "mandacaru._native")]
struct Workspace {
    buffer: Buffer<f64>,
}
#[pymethods]
impl Workspace {
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        array(
            py,
            &[self.buffer.as_slice().len()],
            self.buffer.as_slice().to_vec(),
        )
    }
    fn fill(&mut self, value: f64) -> PyResult<()> {
        if !value.is_finite() {
            return Err(PyValueError::new_err("workspace value must be finite"));
        }
        self.buffer.as_mut_slice().fill(value);
        Ok(())
    }
}

#[pyclass(name = "Context", module = "mandacaru._native", frozen)]
struct PyContext {
    inner: Context,
}
#[pymethods]
impl PyContext {
    #[pyo3(signature=(geometry,coordinates,*,basis_spec="{\"name\":\"HAO\"}"))]
    fn sample_basis<'py>(
        &self,
        py: Python<'py>,
        geometry: &PyGeometry,
        coordinates: PyReadonlyArray2<'py, f64>,
        basis_spec: &str,
    ) -> PyResult<Array<'py, f64>> {
        if coordinates.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "coordinates must have shape (N,3) in Bohr",
            ));
        }
        let (points, _memory) = copy(&self.inner, &coordinates)?;
        let (m, out) = py
            .detach(|| {
                mandacaru_core::finite(&points, "basis coordinates")?;
                let basis = mandacaru_core::molecular::basis::Basis::build(
                    &self.inner,
                    &geometry.inner,
                    basis_spec,
                )?;
                let m = basis.orbitals.len();
                let n = points.len() / 3;
                let mut out = self.inner.allocate(mandacaru_core::product(&[m, n])?)?;
                for (a, orbital) in basis.orbitals.iter().enumerate() {
                    for k in 0..n {
                        out.as_mut_slice()[a * n + k] =
                            orbital.evaluate(points[k * 3..k * 3 + 3].try_into().unwrap());
                    }
                }
                Ok::<_, Error>((m, out))
            })
            .map_err(error)?;
        array(py, &[m, points.len() / 3], out.as_slice().to_vec())
    }
    #[pyo3(signature=(geometry,*,basis_spec="{\"name\":\"HAO\"}",spacing_bohr=0.37794522492515404,grid=None,kinetic="spectral",cholesky_tolerance=None))]
    #[allow(clippy::too_many_arguments)]
    fn molecular_integrals(
        &self,
        py: Python<'_>,
        geometry: &PyGeometry,
        basis_spec: &str,
        spacing_bohr: f64,
        grid: Option<&molecular::PySpatialGrid>,
        kinetic: &str,
        cholesky_tolerance: Option<f64>,
    ) -> PyResult<molecular::PyIntegrals> {
        let geometry = geometry.inner.clone();
        let grid = match grid {
            Some(grid) => grid.inner.clone(),
            None => mandacaru_core::molecular::SpatialGrid::from_geometry(
                &geometry,
                [spacing_bohr; 3],
                false,
                false,
            )
            .map_err(error)?,
        };
        let inner = py
            .detach(|| {
                mandacaru_core::molecular::MolecularIntegrals::new(
                    &self.inner,
                    geometry,
                    grid,
                    basis_spec,
                    kinetic,
                    cholesky_tolerance,
                )
            })
            .map_err(error)?;
        Ok(molecular::PyIntegrals {
            inner: std::sync::Arc::new(inner),
        })
    }
    #[pyo3(signature=(grid,density,*,periodic=false,slab_axis=None))]
    fn poisson<'py>(
        &self,
        py: Python<'py>,
        grid: &molecular::PySpatialGrid,
        density: PyReadonlyArray1<'py, Complex64>,
        periodic: bool,
        slab_axis: Option<usize>,
    ) -> PyResult<Array<'py, Complex64>> {
        molecular::poisson(py, &self.inner, grid, density, periodic, slab_axis)
    }
    fn pauli_operator(
        &self,
        py: Python<'_>,
        num_qubits: usize,
        terms: Vec<(String, f64, f64)>,
    ) -> PyResult<quantum::PyOperator> {
        py.detach(|| quantum::operator(&self.inner, num_qubits, terms))
    }
    #[pyo3(signature=(n_modes,num_particles,terms,*,mapping="jordan_wigner"))]
    fn fermion_operator(
        &self,
        py: Python<'_>,
        n_modes: usize,
        num_particles: [usize; 2],
        terms: quantum::FermionTerms,
        mapping: &str,
    ) -> PyResult<quantum::PyOperator> {
        py.detach(|| quantum::fermion(&self.inner, n_modes, num_particles, mapping, terms))
    }
    #[pyo3(signature=(path,*,sector=true,dense=false,include_singles=true,sparse=false))]
    fn load_hamiltonian(
        &self,
        py: Python<'_>,
        path: String,
        sector: bool,
        dense: bool,
        include_singles: bool,
        sparse: bool,
    ) -> PyResult<quantum::PyProblem> {
        if sparse && dense {
            return Err(PyValueError::new_err(
                "choose either dense or sparse storage",
            ));
        }
        let mut inner = py
            .detach(|| {
                mandacaru_core::quantum::CachedProblem::load(
                    &self.inner,
                    std::path::Path::new(&path),
                    sector,
                    dense,
                    include_singles,
                )
            })
            .map_err(error)?;
        if sparse {
            inner.sparse = Some(
                py.detach(|| {
                    mandacaru_core::quantum::sparse::SparseMatrix::new(
                        &self.inner,
                        &inner.basis,
                        &inner.hamiltonian,
                    )
                })
                .map_err(error)?,
            );
        }
        Ok(quantum::PyProblem { inner })
    }
    #[new]
    #[pyo3(signature = (threads=None, memory_limit_bytes=536870912))]
    fn new(threads: Option<usize>, memory_limit_bytes: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Context::new(threads, memory_limit_bytes).map_err(error)?,
        })
    }
    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }
    #[getter]
    fn memory_limit_bytes(&self) -> usize {
        self.inner.memory_limit()
    }
    #[getter]
    fn memory_used_bytes(&self) -> usize {
        self.inner.memory_used()
    }
    #[getter]
    fn memory_peak_bytes(&self) -> usize {
        self.inner.memory_peak()
    }
    #[getter]
    fn worker_peak(&self) -> usize {
        self.inner.workers_peak()
    }
    #[getter]
    fn workers_launched(&self) -> usize {
        self.inner.workers_total()
    }

    fn workspace(&self, elements: usize) -> PyResult<Workspace> {
        Ok(Workspace {
            buffer: self.inner.allocate(elements).map_err(error)?,
        })
    }

    fn laplacian<'py>(
        &self,
        py: Python<'py>,
        grid: &PyGrid,
        fields: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let shape = fields.shape().to_vec();
        if shape.len() != 4 || shape[1..] != grid.inner.shape() {
            return Err(PyValueError::new_err("fields must have shape (N,nx,ny,nz)"));
        }
        let (input, _input) = copy(&self.inner, &fields)?;
        let result = py
            .detach(|| kernels::laplacian(&self.inner, &grid.inner, shape[0], &input))
            .map_err(error)?;
        array(py, &shape, result.as_slice().to_vec())
    }
    fn laplacian_f32<'py>(
        &self,
        py: Python<'py>,
        grid: &PyGrid,
        fields: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Array<'py, f32>> {
        let shape = fields.shape().to_vec();
        if shape.len() != 4 || shape[1..] != grid.inner.shape() {
            return Err(PyValueError::new_err("fields must have shape (N,nx,ny,nz)"));
        }
        let (input, _input) = copy(&self.inner, &fields)?;
        let result = py
            .detach(|| kernels::laplacian_f32(&self.inner, &grid.inner, shape[0], &input))
            .map_err(error)?;
        array(py, &shape, result.as_slice().to_vec())
    }
    fn one_body<'py>(
        &self,
        py: Python<'py>,
        grid: &PyGrid,
        orbitals: PyReadonlyArray2<'py, Complex64>,
        potential: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Array<'py, Complex64>, Array<'py, Complex64>)> {
        let m = orbitals.shape()[0];
        let (psi, _psi) = complex(&self.inner, &orbitals)?;
        let (potential, _potential) = copy(&self.inner, &potential)?;
        let (kinetic, local) = py
            .detach(|| kernels::one_body(&self.inner, &grid.inner, m, &psi, &potential))
            .map_err(error)?;
        Ok((
            complex_array(py, &[m, m], &kinetic)?,
            complex_array(py, &[m, m], &local)?,
        ))
    }
    fn project<'py>(
        &self,
        py: Python<'py>,
        dv_bohr3: f64,
        orbitals: PyReadonlyArray2<'py, Complex64>,
        projectors: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Array<'py, Complex64>> {
        let (m, g, p) = (
            orbitals.shape()[0],
            orbitals.shape()[1],
            projectors.shape()[0],
        );
        if projectors.shape()[1] != g {
            return Err(PyValueError::new_err(
                "projector and orbital grid lengths differ",
            ));
        }
        let (psi, _psi) = complex(&self.inner, &orbitals)?;
        let (chi, _chi) = complex(&self.inner, &projectors)?;
        let result = py
            .detach(|| kernels::project(&self.inner, g, m, p, dv_bohr3, &psi, &chi))
            .map_err(error)?;
        complex_array(py, &[m, p], &result)
    }
    fn bessel<'py>(
        &self,
        py: Python<'py>,
        l: i32,
        radii: PyReadonlyArray1<'py, f64>,
        wave_numbers: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let (r, _r) = copy(&self.inner, &radii)?;
        let (k, _k) = copy(&self.inner, &wave_numbers)?;
        let result = py
            .detach(|| kernels::bessel(&self.inner, l, &r, &k))
            .map_err(error)?;
        array(py, &[r.len(), k.len()], result.as_slice().to_vec())
    }
    fn fourier_bessel<'py>(
        &self,
        py: Python<'py>,
        l: i32,
        radii: PyReadonlyArray1<'py, f64>,
        weighted: PyReadonlyArray1<'py, f64>,
        wave_numbers: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Array<'py, f64>> {
        let (r, _r) = copy(&self.inner, &radii)?;
        let (w, _w) = copy(&self.inner, &weighted)?;
        let (k, _k) = copy(&self.inner, &wave_numbers)?;
        let result = py
            .detach(|| kernels::fourier_bessel(&self.inner, l, &r, &w, &k))
            .map_err(error)?;
        array(py, &[k.len()], result.as_slice().to_vec())
    }
    fn eigenpair<'py>(
        &self,
        py: Python<'py>,
        diagonal: PyReadonlyArray1<'py, f64>,
        off_diagonal: PyReadonlyArray1<'py, f64>,
        index: usize,
    ) -> PyResult<(f64, Array<'py, f64>)> {
        let (d, _d) = copy(&self.inner, &diagonal)?;
        let (e, _e) = copy(&self.inner, &off_diagonal)?;
        let (value, vector) = py
            .detach(|| kernels::eigenpair(&self.inner, &d, &e, index))
            .map_err(error)?;
        Ok((value, array(py, &[d.len()], vector.as_slice().to_vec())?))
    }
    #[pyo3(signature=(coefficient, source, initial, step_squared, boundary, inward=false))]
    fn numerov<'py>(
        &self,
        py: Python<'py>,
        coefficient: PyReadonlyArray1<'py, f64>,
        source: PyReadonlyArray1<'py, f64>,
        initial: PyReadonlyArray1<'py, f64>,
        step_squared: f64,
        boundary: usize,
        inward: bool,
    ) -> PyResult<Array<'py, f64>> {
        let (f, _f) = copy(&self.inner, &coefficient)?;
        let (s, _s) = copy(&self.inner, &source)?;
        let (u, _u) = copy(&self.inner, &initial)?;
        let result = py
            .detach(|| kernels::numerov(&self.inner, &f, &s, &u, step_squared, boundary, inward))
            .map_err(error)?;
        array(py, &[f.len()], result.as_slice().to_vec())
    }
    #[pyo3(signature=(coordinates, center, support, knots, coefficients, cutoff, l, ms, derivative=-1))]
    fn real_shell<'py>(
        &self,
        py: Python<'py>,
        coordinates: PyReadonlyArray2<'py, f64>,
        center: [f64; 3],
        support: f64,
        knots: PyReadonlyArray1<'py, f64>,
        coefficients: PyReadonlyArray1<'py, f64>,
        cutoff: f64,
        l: i32,
        ms: PyReadonlyArray1<'py, i32>,
        derivative: i32,
    ) -> PyResult<Array<'py, f64>> {
        if coordinates.shape()[0] != 3 {
            return Err(PyValueError::new_err("coordinates must have shape (3,G)"));
        }
        let points = coordinates.shape()[1];
        let (xyz, _xyz) = copy(&self.inner, &coordinates)?;
        let (b, _b) = copy(&self.inner, &knots)?;
        let (c, _c) = copy(&self.inner, &coefficients)?;
        let (ms, _ms) = copy(&self.inner, &ms)?;
        let spline = kernels::Spline::new(&b, &c, cutoff, l).map_err(error)?;
        let result = py
            .detach(|| {
                kernels::real_shell(
                    &self.inner,
                    [&xyz[..points], &xyz[points..2 * points], &xyz[2 * points..]],
                    center,
                    support,
                    &spline,
                    &ms,
                    derivative,
                )
            })
            .map_err(error)?;
        array(py, &[ms.len(), points], result.as_slice().to_vec())
    }
}

// Importing this module into a free-threaded interpreter re-enables the GIL.
// NumPy input snapshots are taken while attached; only Rust-owned data is used
// in detached work. Free-threaded support needs a separate NumPy alias audit.
#[pymodule(gil_used = true)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI_VERSION", 1)?;
    m.add("THREAD_POLICY_VERSION", 1)?;
    m.add_class::<PyContext>()?;
    m.add_class::<PyGrid>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<Workspace>()?;
    m.add_class::<quantum::PyOperator>()?;
    m.add_class::<quantum::PyProblem>()?;
    m.add_class::<molecular::PySpatialGrid>()?;
    m.add_class::<molecular::PyIntegrals>()?;
    m.add_class::<molecular::PyMeanField>()?;
    m.add_class::<molecular::PyModel>()?;
    Ok(())
}
