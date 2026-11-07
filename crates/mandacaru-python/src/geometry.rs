//! Geometry, grid-index and workspace values at the language boundary.
use super::{array, error, layout, Array};
use mandacaru_core::{math::SpatialGrid, Buffer, Geometry, Grid, LengthUnit};
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
    fn max_step_norm(&self) -> f64 {
        let s = self.inner.step_values();
        let col0 = (s[0] * s[0] + s[3] * s[3] + s[6] * s[6]).sqrt();
        let col1 = (s[1] * s[1] + s[4] * s[4] + s[7] * s[7]).sqrt();
        let col2 = (s[2] * s[2] + s[5] * s[5] + s[8] * s[8]).sqrt();
        col0.max(col1).max(col2)
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

        let center_vec: Vec<f64> = if let Ok(v) = center.extract::<Vec<f64>>() {
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

        enum BoxSpec {
            None,
            Scalar(f64),
            Axes([f64; 3]),
        }
        let box_spec = if cell_matrix.is_some() {
            BoxSpec::None
        } else if let Some(b) = box_size {
            if b.is_none() {
                BoxSpec::None
            } else if let Ok(s) = b.extract::<f64>() {
                BoxSpec::Scalar(unit.to_bohr(s).abs())
            } else if let Ok(arr) = b.extract::<PyReadonlyArray1<'_, f64>>() {
                layout(&arr)?;
                let sl = arr.as_slice()?;
                if sl.len() == 1 {
                    BoxSpec::Scalar(unit.to_bohr(sl[0]).abs())
                } else if sl.len() == 3 {
                    BoxSpec::Axes([
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
                    BoxSpec::Scalar(unit.to_bohr(vec[0]).abs())
                } else if vec.len() == 3 {
                    BoxSpec::Axes([
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
            BoxSpec::None
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

        let is_cubic_path = cell_matrix.is_none()
            && matches!(box_spec, BoxSpec::Scalar(_))
            && h_scalar
            && !periodic;

        if is_cubic_path {
            let box_bohr = match box_spec {
                BoxSpec::Scalar(val) => val,
                _ => unreachable!(),
            };
            let points = ((2.0 * box_bohr / h_axes[0]).round() as isize + 1).max(2) as usize;
            let dx = 2.0 * box_bohr / (points - 1) as f64;
            let shape = [points, points, points];
            let step = [
                dx, 0.0, 0.0,
                0.0, dx, 0.0,
                0.0, 0.0, dx,
            ];
            let origin = [
                center_bohr[0] - box_bohr,
                center_bohr[1] - box_bohr,
                center_bohr[2] - box_bohr,
            ];
            return Ok(Self {
                inner: SpatialGrid::new(shape, origin, step, false).map_err(error)?,
            });
        }

        if periodic {
            let cell_m = cell_matrix.ok_or_else(|| {
                PyValueError::new_err("a periodic grid needs the cell it is periodic in")
            })?;
            let mut lengths = [0.0; 3];
            for m in 0..3 {
                lengths[m] =
                    (cell_m[m][0].powi(2) + cell_m[m][1].powi(2) + cell_m[m][2].powi(2)).sqrt();
            }
            let mut counts = [
                ((lengths[0] / h_axes[0]).round() as isize).max(1) as usize,
                ((lengths[1] / h_axes[1]).round() as isize).max(1) as usize,
                ((lengths[2] / h_axes[2]).round() as isize).max(1) as usize,
            ];
            if let Some(divs) = divisors {
                for m in 0..3 {
                    let d = divs[m].max(1);
                    let n = counts[m];
                    let rem = (-(n as isize)).rem_euclid(d as isize) as usize;
                    counts[m] = n + rem;
                }
            }
            let shape = counts;
            let mut step = [0.0; 9];
            for m in 0..3 {
                for row in 0..3 {
                    step[row * 3 + m] = cell_m[m][row] / counts[m] as f64;
                }
            }
            let middle = [
                0.5 * (cell_m[0][0] + cell_m[1][0] + cell_m[2][0]),
                0.5 * (cell_m[0][1] + cell_m[1][1] + cell_m[2][1]),
                0.5 * (cell_m[0][2] + cell_m[1][2] + cell_m[2][2]),
            ];
            let origin = [
                center_bohr[0] - middle[0],
                center_bohr[1] - middle[1],
                center_bohr[2] - middle[2],
            ];
            return Ok(Self {
                inner: SpatialGrid::new(shape, origin, step, true).map_err(error)?,
            });
        }

        if let Some(cell_m) = cell_matrix {
            if skew {
                let mut lengths = [0.0; 3];
                for m in 0..3 {
                    lengths[m] = (cell_m[m][0].powi(2)
                        + cell_m[m][1].powi(2)
                        + cell_m[m][2].powi(2))
                    .sqrt();
                }
                let counts = [
                    ((lengths[0] / h_axes[0]).round() as isize + 1).max(2) as usize,
                    ((lengths[1] / h_axes[1]).round() as isize + 1).max(2) as usize,
                    ((lengths[2] / h_axes[2]).round() as isize + 1).max(2) as usize,
                ];
                let shape = counts;
                let mut step = [0.0; 9];
                for m in 0..3 {
                    for row in 0..3 {
                        step[row * 3 + m] = cell_m[m][row] / (counts[m] - 1) as f64;
                    }
                }
                let r_center = [
                    0.5 * (cell_m[0][0] + cell_m[1][0] + cell_m[2][0]),
                    0.5 * (cell_m[0][1] + cell_m[1][1] + cell_m[2][1]),
                    0.5 * (cell_m[0][2] + cell_m[1][2] + cell_m[2][2]),
                ];
                let origin = [
                    center_bohr[0] - r_center[0],
                    center_bohr[1] - r_center[1],
                    center_bohr[2] - r_center[2],
                ];
                return Ok(Self {
                    inner: SpatialGrid::new(shape, origin, step, false).map_err(error)?,
                });
            }
        }

        let half = if let Some(cell_m) = cell_matrix {
            let mut extent = [0.0; 3];
            for axis in 0..3 {
                extent[axis] =
                    cell_m[0][axis].abs() + cell_m[1][axis].abs() + cell_m[2][axis].abs();
            }
            [0.5 * extent[0], 0.5 * extent[1], 0.5 * extent[2]]
        } else {
            match box_spec {
                BoxSpec::Scalar(val) => [val, val, val],
                BoxSpec::Axes(axes) => axes,
                BoxSpec::None => {
                    return Err(PyValueError::new_err(
                        "either cell or box_size must be provided",
                    ))
                }
            }
        };
        let counts = [
            ((2.0 * half[0] / h_axes[0]).round() as isize + 1).max(2) as usize,
            ((2.0 * half[1] / h_axes[1]).round() as isize + 1).max(2) as usize,
            ((2.0 * half[2] / h_axes[2]).round() as isize + 1).max(2) as usize,
        ];
        let shape = counts;
        let step = [
            h_axes[0], 0.0, 0.0,
            0.0, h_axes[1], 0.0,
            0.0, 0.0, h_axes[2],
        ];
        let origin = [
            center_bohr[0] - 0.5 * h_axes[0] * (counts[0] - 1) as f64,
            center_bohr[1] - 0.5 * h_axes[1] * (counts[1] - 1) as f64,
            center_bohr[2] - 0.5 * h_axes[2] * (counts[2] - 1) as f64,
        ];
        Ok(Self {
            inner: SpatialGrid::new(shape, origin, step, false).map_err(error)?,
        })
    }
}

#[pyfunction]
pub fn positions_center(positions: PyReadonlyArray2<'_, f64>) -> PyResult<[f64; 3]> {
    layout(&positions)?;
    let shape = positions.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err("positions must have shape (N, 3)"));
    }
    let n = shape[0];
    if n == 0 {
        return Ok([0.0, 0.0, 0.0]);
    }
    let slice = positions.as_slice()?;
    let mut sum = [0.0; 3];
    for row in slice.chunks_exact(3) {
        sum[0] += row[0];
        sum[1] += row[1];
        sum[2] += row[2];
    }
    let inv_n = 1.0 / (n as f64);
    Ok([sum[0] * inv_n, sum[1] * inv_n, sum[2] * inv_n])
}

#[pyfunction]
pub fn grid_step_max_norm(step: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    layout(&step)?;
    if step.shape() != [3, 3] {
        return Err(PyValueError::new_err("step must have shape (3,3)"));
    }
    let s = step.as_slice()?;
    let col0 = (s[0] * s[0] + s[3] * s[3] + s[6] * s[6]).sqrt();
    let col1 = (s[1] * s[1] + s[4] * s[4] + s[7] * s[7]).sqrt();
    let col2 = (s[2] * s[2] + s[5] * s[5] + s[8] * s[8]).sqrt();
    Ok(col0.max(col1).max(col2))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(positions_center, m)?)?;
    m.add_function(wrap_pyfunction!(grid_step_max_norm, m)?)?;
    Ok(())
}

