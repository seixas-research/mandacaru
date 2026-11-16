//! Geometry, grid-index and workspace values at the language boundary.
use super::{array, error, layout, Array};
use mandacaru_core::{
    math::{
        grid::{self, GridBox},
        SpatialGrid,
    },
    Buffer, Geometry, Grid, LengthUnit,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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
    #[getter]
    fn spacings(&self) -> [f64; 3] {
        self.inner.spacings()
    }
    fn is_orthogonal(&self) -> bool {
        let s = self.inner.step;
        s[(0, 1)].abs() < 1e-12
            && s[(0, 2)].abs() < 1e-12
            && s[(1, 0)].abs() < 1e-12
            && s[(1, 2)].abs() < 1e-12
            && s[(2, 0)].abs() < 1e-12
            && s[(2, 1)].abs() < 1e-12
    }
    fn is_orthorhombic(&self) -> bool {
        self.is_orthogonal()
    }
    fn is_cubic(&self) -> bool {
        let [nx, ny, nz] = self.inner.index.shape();
        if nx != ny || ny != nz {
            return false;
        }
        if !self.is_orthorhombic() {
            return false;
        }
        let [dx, dy, dz] = self.inner.spacings();
        (dx - dy).abs() <= 1e-12 * dx.abs().max(1.0)
            && (dy - dz).abs() <= 1e-12 * dy.abs().max(1.0)
    }
    fn metric_inverse<'py>(&self, py: Python<'py>) -> PyResult<Array<'py, f64>> {
        array(py, &[3, 3], self.inner.index.metric().to_vec())
    }
    fn coordinates<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>, Array<'py, f64>)> {
        let [nx, ny, nz] = self.inner.index.shape();
        let n = nx * ny * nz;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for idx in 0..n {
            let pt = self.inner.point(idx);
            x.push(pt[0]);
            y.push(pt[1]);
            z.push(pt[2]);
        }
        Ok((
            array(py, &[nx, ny, nz], x)?,
            array(py, &[nx, ny, nz], y)?,
            array(py, &[nx, ny, nz], z)?,
        ))
    }
    fn flat_coordinates<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Array<'py, f64>, Array<'py, f64>, Array<'py, f64>)> {
        let n = self.inner.index.points();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for idx in 0..n {
            let pt = self.inner.point(idx);
            x.push(pt[0]);
            y.push(pt[1]);
            z.push(pt[2]);
        }
        Ok((
            array(py, &[n], x)?,
            array(py, &[n], y)?,
            array(py, &[n], z)?,
        ))
    }
    #[staticmethod]
    #[pyo3(signature = (
        center,
        spacing,
        box_size=None,
        cell=None,
        units="angstrom",
        periodic=false,
        skew=false,
        commensurate=None
    ))]
    fn from_specs(
        center: &Bound<'_, PyAny>,
        spacing: &Bound<'_, PyAny>,
        box_size: Option<&Bound<'_, PyAny>>,
        cell: Option<&Bound<'_, PyAny>>,
        units: &str,
        periodic: bool,
        skew: bool,
        commensurate: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let unit = match units.to_ascii_lowercase().as_str() {
            "angstrom" | "a" => LengthUnit::Angstrom,
            "bohr" | "au" | "a0" => LengthUnit::Bohr,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown length unit {:?}; use 'angstrom' or 'bohr'",
                    units
                )))
            }
        };

        // `center=None` with a `cell` is the cell's center, half the sum of its vectors, summed in
        // `units` before the conversion (the center of a periodic grid).
        let center_vec: Vec<f64> = if center.is_none() {
            let arr: PyReadonlyArray2<'_, f64> = match cell {
                Some(c) if !c.is_none() => c
                    .extract()
                    .map_err(|_| PyValueError::new_err("cell must be a (3, 3) tensor"))?,
                _ => {
                    return Err(PyValueError::new_err(
                        "a grid without a center needs a cell: its center is the middle of the cell",
                    ))
                }
            };
            layout(&arr)?;
            if arr.shape() != [3, 3] {
                return Err(PyValueError::new_err(format!(
                    "cell must be a (3, 3) tensor, got {:?}",
                    arr.shape()
                )));
            }
            let sl: [f64; 9] = arr.as_slice()?.try_into().expect("a (3, 3) tensor");
            grid::cell_center(&sl).to_vec()
        } else if let Ok(v) = center.extract::<Vec<f64>>() {
            v
        } else if let Ok(arr) = center.extract::<PyReadonlyArray1<'_, f64>>() {
            layout(&arr)?;
            arr.as_slice()?.to_vec()
        } else {
            return Err(PyValueError::new_err(
                "center must be a length-3 coordinate sequence",
            ));
        };
        if center_vec.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "center must have 3 coordinates, got {}",
                center_vec.len()
            )));
        }
        let center_bohr = [
            unit.to_bohr(center_vec[0]),
            unit.to_bohr(center_vec[1]),
            unit.to_bohr(center_vec[2]),
        ];

        let (h_axes, h_scalar) = if let Ok(s) = spacing.extract::<f64>() {
            let hb = unit.to_bohr(s).abs();
            ([hb, hb, hb], true)
        } else if let Ok(arr) = spacing.extract::<PyReadonlyArray1<'_, f64>>() {
            layout(&arr)?;
            let sl = arr.as_slice()?;
            if sl.len() == 1 {
                let hb = unit.to_bohr(sl[0]).abs();
                ([hb, hb, hb], true)
            } else if sl.len() == 3 {
                (
                    [
                        unit.to_bohr(sl[0]).abs(),
                        unit.to_bohr(sl[1]).abs(),
                        unit.to_bohr(sl[2]).abs(),
                    ],
                    false,
                )
            } else {
                return Err(PyValueError::new_err(format!(
                    "grid_spacing must be a scalar or length-3 array, got length {}",
                    sl.len()
                )));
            }
        } else if let Ok(vec) = spacing.extract::<Vec<f64>>() {
            if vec.len() == 1 {
                let hb = unit.to_bohr(vec[0]).abs();
                ([hb, hb, hb], true)
            } else if vec.len() == 3 {
                (
                    [
                        unit.to_bohr(vec[0]).abs(),
                        unit.to_bohr(vec[1]).abs(),
                        unit.to_bohr(vec[2]).abs(),
                    ],
                    false,
                )
            } else {
                return Err(PyValueError::new_err(format!(
                    "grid_spacing must be a scalar or length-3 sequence, got length {}",
                    vec.len()
                )));
            }
        } else {
            return Err(PyValueError::new_err(
                "grid_spacing must be a float or length-3 sequence",
            ));
        };

        let cell_matrix = if let Some(c) = cell {
            if c.is_none() {
                None
            } else {
                let arr: PyReadonlyArray2<'_, f64> = c
                    .extract()
                    .map_err(|_| PyValueError::new_err("cell must be a (3, 3) tensor"))?;
                layout(&arr)?;
                let s = arr.shape();
                if s != [3, 3] {
                    return Err(PyValueError::new_err(format!(
                        "cell must be a (3, 3) tensor, got {:?}",
                        s
                    )));
                }
                let sl = arr.as_slice()?;
                let m = [
                    [
                        unit.to_bohr(sl[0]),
                        unit.to_bohr(sl[1]),
                        unit.to_bohr(sl[2]),
                    ],
                    [
                        unit.to_bohr(sl[3]),
                        unit.to_bohr(sl[4]),
                        unit.to_bohr(sl[5]),
                    ],
                    [
                        unit.to_bohr(sl[6]),
                        unit.to_bohr(sl[7]),
                        unit.to_bohr(sl[8]),
                    ],
                ];
                Some(m)
            }
        } else {
            None
        };

        let box_spec = if cell_matrix.is_some() {
            GridBox::None
        } else if let Some(b) = box_size {
            if b.is_none() {
                GridBox::None
            } else if let Ok(s) = b.extract::<f64>() {
                GridBox::Scalar(unit.to_bohr(s).abs())
            } else if let Ok(arr) = b.extract::<PyReadonlyArray1<'_, f64>>() {
                layout(&arr)?;
                let sl = arr.as_slice()?;
                if sl.len() == 1 {
                    GridBox::Scalar(unit.to_bohr(sl[0]).abs())
                } else if sl.len() == 3 {
                    GridBox::Axes([
                        unit.to_bohr(sl[0]).abs(),
                        unit.to_bohr(sl[1]).abs(),
                        unit.to_bohr(sl[2]).abs(),
                    ])
                } else {
                    return Err(PyValueError::new_err(format!(
                        "box_size must be a scalar or a length-3 array of per-axis half-extents, got shape ({},)",
                        sl.len()
                    )));
                }
            } else if let Ok(vec) = b.extract::<Vec<f64>>() {
                if vec.len() == 1 {
                    GridBox::Scalar(unit.to_bohr(vec[0]).abs())
                } else if vec.len() == 3 {
                    GridBox::Axes([
                        unit.to_bohr(vec[0]).abs(),
                        unit.to_bohr(vec[1]).abs(),
                        unit.to_bohr(vec[2]).abs(),
                    ])
                } else {
                    return Err(PyValueError::new_err(format!(
                        "box_size must be a scalar or a length-3 array of per-axis half-extents, got shape ({},)",
                        vec.len()
                    )));
                }
            } else {
                return Err(PyValueError::new_err(
                    "box_size must be a scalar or a length-3 array of per-axis half-extents",
                ));
            }
        } else {
            GridBox::None
        };

        let divisors = if let Some(cm) = commensurate {
            if cm.is_none() {
                None
            } else if let Ok(v) = cm.extract::<Vec<usize>>() {
                if v.len() != 3 {
                    return Err(PyValueError::new_err(format!(
                        "commensurate must be three positive integers, one per lattice vector; got {:?}",
                        v
                    )));
                }
                Some([v[0], v[1], v[2]])
            } else {
                return Err(PyValueError::new_err(
                    "commensurate must be three positive integers, one per lattice vector",
                ));
            }
        } else {
            None
        };

        Ok(Self {
            inner: grid::grid_from_spec(
                center_bohr,
                h_axes,
                h_scalar,
                box_spec,
                cell_matrix,
                periodic,
                skew,
                divisors,
            )
            .map_err(error)?,
        })
    }
}

#[pyfunction]
pub fn grid_step_max_norm(step: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    layout(&step)?;
    if step.shape() != [3, 3] {
        return Err(PyValueError::new_err("step must have shape (3,3)"));
    }
    let s: [f64; 9] = step.as_slice()?.try_into().expect("a (3, 3) tensor");
    Ok(grid::step_max_norm(&s))
}

fn rows(cell: &PyReadonlyArray2<'_, f64>) -> PyResult<[[f64; 3]; 3]> {
    layout(cell)?;
    if cell.shape() != [3, 3] {
        return Err(PyValueError::new_err("cell must have shape (3, 3)"));
    }
    let c = cell.as_slice()?;
    Ok([[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]])
}

fn points(positions: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    layout(positions)?;
    if positions.ndim() != 2 || positions.shape()[1] != 3 {
        return Err(PyValueError::new_err("positions must have shape (n, 3)"));
    }
    Ok(positions.as_slice()?.chunks_exact(3).map(|r| [r[0], r[1], r[2]]).collect())
}

/// The fractional coordinates of `positions` in the cell (rows) in the same length unit; the
/// periodic ones are wrapped into `[0, 1)` unless `wrap` is false (`geometry::fractional_coordinates`).
#[pyfunction]
#[pyo3(signature = (cell, positions, pbc, wrap = true))]
fn fractional_coordinates<'py>(
    py: Python<'py>,
    cell: PyReadonlyArray2<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
    pbc: [bool; 3],
    wrap: bool,
) -> PyResult<Array<'py, f64>> {
    let (cell, positions) = (rows(&cell)?, points(&positions)?);
    let out = py
        .detach(|| mandacaru_core::geometry::fractional_coordinates(&cell, &positions, pbc, wrap))
        .map_err(error)?;
    array(py, &[out.len(), 3], out.concat())
}

/// `(numbers, positions, cell, moments)` of the supercell of `repeats` cells along the cell
/// vectors; `moments` is one value per atom, repeated with them (`geometry::repeat_cell`).
#[pyfunction]
fn repeat_cell<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: PyReadonlyArray2<'py, f64>,
    cell: PyReadonlyArray2<'py, f64>,
    moments: Vec<f64>,
    repeats: [usize; 3],
) -> PyResult<(Vec<u8>, Array<'py, f64>, Array<'py, f64>, Vec<f64>)> {
    let (cell, positions) = (rows(&cell)?, points(&positions)?);
    if moments.len() != numbers.len() {
        return Err(PyValueError::new_err("one moment per atom"));
    }
    let out = py
        .detach(|| mandacaru_core::geometry::repeat_cell(&cell, &positions, &numbers, repeats))
        .map_err(error)?;
    let moments = out.source.iter().map(|&i| moments[i]).collect();
    Ok((
        out.numbers,
        array(py, &[out.positions.len(), 3], out.positions.concat())?,
        array(py, &[3, 3], out.cell.concat())?,
        moments,
    ))
}

/// For each atom, the vectors to its neighbors closer than `cutoff`, periodic images included
/// (`geometry::neighbor_vectors`), as an `(m, 3)` array per atom.
#[pyfunction]
fn neighbor_vectors<'py>(
    py: Python<'py>,
    cell: PyReadonlyArray2<'py, f64>,
    pbc: [bool; 3],
    positions: PyReadonlyArray2<'py, f64>,
    cutoff: f64,
) -> PyResult<Vec<Array<'py, f64>>> {
    let (cell, positions) = (rows(&cell)?, points(&positions)?);
    let found = py
        .detach(|| mandacaru_core::geometry::neighbor_vectors(&cell, pbc, &positions, cutoff))
        .map_err(error)?;
    found.into_iter().map(|v| array(py, &[v.len(), 3], v.concat())).collect()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fractional_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(repeat_cell, m)?)?;
    m.add_function(wrap_pyfunction!(neighbor_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(grid_step_max_norm, m)?)?;
    Ok(())
}

