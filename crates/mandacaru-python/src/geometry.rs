//! Geometry, grid-index and workspace values at the language boundary.
use super::{array, error, layout, Array};
use mandacaru_core::{math::SpatialGrid, Buffer, Geometry, Grid, LengthUnit};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
#[pyclass(name = "Geometry", module = "mandacaru._native", frozen)]
pub(crate) struct PyGeometry {
    pub(crate) inner: Geometry,
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
pub(crate) struct PyGrid {
    pub(crate) inner: Grid,
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
pub(crate) struct Workspace {
    pub(crate) buffer: Buffer<f64>,
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

#[pyclass(name = "SpatialGrid", module = "mandacaru._native", frozen)]
pub(crate) struct PySpatialGrid {
    pub(crate) inner: SpatialGrid,
}
#[pymethods]
impl PySpatialGrid {
    #[new]
    #[pyo3(signature=(shape,origin_bohr,step_bohr,*,periodic=false))]
    fn new(
        shape: [usize; 3],
        origin_bohr: [f64; 3],
        step_bohr: PyReadonlyArray2<'_, f64>,
        periodic: bool,
    ) -> PyResult<Self> {
        layout(&step_bohr)?;
        if step_bohr.shape() != [3, 3] {
            return Err(PyValueError::new_err("step must have shape (3,3)"));
        }
        Ok(Self {
            inner: SpatialGrid::new(
                shape,
                origin_bohr,
                step_bohr.as_slice()?.try_into().unwrap(),
                periodic,
            )
            .map_err(error)?,
        })
    }
    #[staticmethod]
    #[pyo3(signature=(geometry,spacing_bohr,*,periodic=false,skew=false))]
    fn from_geometry(
        geometry: &PyGeometry,
        spacing_bohr: [f64; 3],
        periodic: bool,
        skew: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SpatialGrid::from_geometry(&geometry.inner, spacing_bohr, periodic, skew)
                .map_err(error)?,
        })
    }
    #[getter]
    fn shape(&self) -> [usize; 3] {
        self.inner.index.shape()
    }
    #[getter]
    fn origin_bohr(&self) -> [f64; 3] {
        self.inner.origin
    }
    #[getter]
    fn step_bohr<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        array(py, &[3, 3], self.inner.step_values().to_vec())
    }
    #[getter]
    fn dv_bohr3(&self) -> f64 {
        self.inner.index.dv()
    }
    #[getter]
    fn points(&self) -> usize {
        self.inner.index.points()
    }
    #[getter]
    fn periodic(&self) -> bool {
        self.inner.periodic
    }
}
