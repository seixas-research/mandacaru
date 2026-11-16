use super::invalid;
use crate::{finite, Geometry, Grid, Result};
use nalgebra::{Matrix3, Vector3};

/// An index grid in Bohr. Step vectors are columns; cell vectors are rows.
#[derive(Clone, Debug)]
pub struct SpatialGrid {
    pub index: Grid,
    pub origin: [f64; 3],
    pub step: Matrix3<f64>,
    pub periodic: bool,
}
impl SpatialGrid {
    pub fn new(
        shape: [usize; 3],
        origin: [f64; 3],
        step: [f64; 9],
        periodic: bool,
    ) -> Result<Self> {
        finite(&origin, "grid origin")?;
        finite(&step, "grid step")?;
        let step = Matrix3::from_row_slice(&step);
        let inverse = step
            .try_inverse()
            .ok_or_else(|| invalid("grid step must be nonsingular"))?;
        let metric = inverse * inverse.transpose();
        let mut values = [0.; 9];
        for i in 0..3 {
            for j in 0..3 {
                values[3 * i + j] = metric[(i, j)];
            }
        }
        let index = Grid::new(shape, values, step.determinant().abs())?;
        Ok(Self {
            index,
            origin,
            step,
            periodic,
        })
    }
    pub fn from_geometry(
        geometry: &Geometry,
        spacing: [f64; 3],
        periodic: bool,
        skew: bool,
    ) -> Result<Self> {
        if spacing.iter().any(|h| !h.is_finite() || *h <= 0.) {
            return Err(invalid("grid spacing must be positive and finite"));
        }
        let cell = geometry.cell();
        let mut shape = [0; 3];
        let mut step = [0.; 9];
        for axis in 0..3 {
            let length = if periodic || skew {
                cell[axis].iter().map(|v| v * v).sum::<f64>().sqrt()
            } else {
                cell.iter().map(|row| row[axis].abs()).sum()
            };
            if length <= 0. || !length.is_finite() {
                return Err(invalid(
                    "molecular integration requires a nondegenerate cell or an explicit grid",
                ));
            }
            let count = (length / spacing[axis]).round_ties_even();
            if count > i32::MAX as f64 - 1. {
                return Err(crate::Error::Overflow);
            }
            shape[axis] = if periodic {
                count.max(1.) as usize
            } else {
                (count as usize + 1).max(2)
            };
            for row in 0..3 {
                step[row * 3 + axis] = if periodic || skew {
                    cell[axis][row] / (shape[axis] - usize::from(!periodic)) as f64
                } else if row == axis {
                    spacing[axis]
                } else {
                    0.
                };
            }
        }
        let center: Vector3<f64> = if periodic {
            (Vector3::from(cell[0]) + Vector3::from(cell[1]) + Vector3::from(cell[2])) * 0.5
        } else {
            geometry
                .positions()
                .iter()
                .map(|r| Vector3::from(*r))
                .sum::<Vector3<f64>>()
                / geometry.positions().len() as f64
        };
        let matrix = Matrix3::from_row_slice(&step);
        let middle = Vector3::from_fn(|i, _| (shape[i] - usize::from(!periodic)) as f64 / 2.);
        let origin = center - matrix * middle;
        Self::new(shape, origin.into(), step, periodic)
    }
    pub fn point(&self, index: usize) -> [f64; 3] {
        let [_, ny, nz] = self.index.shape();
        let coordinate = Vector3::new(
            (index / (ny * nz)) as f64,
            ((index / nz) % ny) as f64,
            (index % nz) as f64,
        );
        (Vector3::from(self.origin) + self.step * coordinate).into()
    }
    pub fn spacings(&self) -> [f64; 3] {
        std::array::from_fn(|j| self.step.column(j).norm())
    }
    pub fn step_values(&self) -> [f64; 9] {
        std::array::from_fn(|k| self.step[(k / 3, k % 3)])
    }
}

/// The box of a grid without a cell: half-extents in Bohr, one for every axis or one per axis.
#[derive(Clone, Copy, Debug)]
pub enum GridBox {
    None,
    Scalar(f64),
    Axes([f64; 3]),
}

/// The grid of a calculation from its specification, every length in Bohr: a cubic box around
/// `center_bohr` for a scalar spacing and box, a periodic grid of `round(|a_m| / h_m)` nodes per
/// lattice vector (raised to a multiple of `divisors[m]`, centered on `center_bohr`), a skew
/// grid spanning a nonperiodic cell, or an orthogonal grid over the cell's extent or the box.
#[allow(clippy::too_many_arguments)]
pub fn grid_from_spec(
    center_bohr: [f64; 3],
    h_axes: [f64; 3],
    h_scalar: bool,
    box_spec: GridBox,
    cell_matrix: Option<[[f64; 3]; 3]>,
    periodic: bool,
    skew: bool,
    divisors: Option<[usize; 3]>,
) -> Result<SpatialGrid> {
    let is_cubic_path =
        cell_matrix.is_none() && matches!(box_spec, GridBox::Scalar(_)) && h_scalar && !periodic;

    if is_cubic_path {
        let box_bohr = match box_spec {
            GridBox::Scalar(val) => val,
            _ => unreachable!(),
        };
        let points = ((2.0 * box_bohr / h_axes[0]).round() as isize + 1).max(2) as usize;
        let dx = 2.0 * box_bohr / (points - 1) as f64;
        let shape = [points, points, points];
        let step = [dx, 0.0, 0.0, 0.0, dx, 0.0, 0.0, 0.0, dx];
        let origin = [
            center_bohr[0] - box_bohr,
            center_bohr[1] - box_bohr,
            center_bohr[2] - box_bohr,
        ];
        return SpatialGrid::new(shape, origin, step, false);
    }

    if periodic {
        let cell_m = cell_matrix
            .ok_or_else(|| invalid("a periodic grid needs the cell it is periodic in"))?;
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
        return SpatialGrid::new(shape, origin, step, true);
    }

    if let Some(cell_m) = cell_matrix {
        if skew {
            let mut lengths = [0.0; 3];
            for m in 0..3 {
                lengths[m] =
                    (cell_m[m][0].powi(2) + cell_m[m][1].powi(2) + cell_m[m][2].powi(2)).sqrt();
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
            return SpatialGrid::new(shape, origin, step, false);
        }
    }

    let half = if let Some(cell_m) = cell_matrix {
        let mut extent = [0.0; 3];
        for axis in 0..3 {
            extent[axis] = cell_m[0][axis].abs() + cell_m[1][axis].abs() + cell_m[2][axis].abs();
        }
        [0.5 * extent[0], 0.5 * extent[1], 0.5 * extent[2]]
    } else {
        match box_spec {
            GridBox::Scalar(val) => [val, val, val],
            GridBox::Axes(axes) => axes,
            GridBox::None => return Err(invalid("either cell or box_size must be provided")),
        }
    };
    let counts = [
        ((2.0 * half[0] / h_axes[0]).round() as isize + 1).max(2) as usize,
        ((2.0 * half[1] / h_axes[1]).round() as isize + 1).max(2) as usize,
        ((2.0 * half[2] / h_axes[2]).round() as isize + 1).max(2) as usize,
    ];
    let shape = counts;
    let step = [
        h_axes[0], 0.0, 0.0, 0.0, h_axes[1], 0.0, 0.0, 0.0, h_axes[2],
    ];
    let origin = [
        center_bohr[0] - 0.5 * h_axes[0] * (counts[0] - 1) as f64,
        center_bohr[1] - 0.5 * h_axes[1] * (counts[1] - 1) as f64,
        center_bohr[2] - 0.5 * h_axes[2] * (counts[2] - 1) as f64,
    ];
    SpatialGrid::new(shape, origin, step, false)
}

/// The largest column norm of a `(3, 3)` row-major step matrix: the coarsest grid step.
pub fn step_max_norm(s: &[f64; 9]) -> f64 {
    let col0 = (s[0] * s[0] + s[3] * s[3] + s[6] * s[6]).sqrt();
    let col1 = (s[1] * s[1] + s[4] * s[4] + s[7] * s[7]).sqrt();
    let col2 = (s[2] * s[2] + s[5] * s[5] + s[8] * s[8]).sqrt();
    col0.max(col1).max(col2)
}

/// Half the sum of the cell vectors (rows of `cell`, row-major): the middle of the cell.
pub fn cell_center(cell: &[f64; 9]) -> [f64; 3] {
    std::array::from_fn(|i| 0.5 * (cell[i] + cell[3 + i] + cell[6 + i]))
}

/// `2 pi (A^-1)^T` of the row-major matrix `a`, row-major: the reciprocal vectors of a lattice
/// whose vectors are the columns of `a` come out as the columns.
pub fn reciprocal_matrix(a: &[f64; 9]) -> Result<[f64; 9]> {
    let a = Matrix3::from_row_slice(a);
    let inv = a.try_inverse().ok_or_else(|| invalid("singular lattice matrix"))?;
    let b = inv.transpose() * (2.0 * std::f64::consts::PI);
    Ok(std::array::from_fn(|k| b[(k / 3, k % 3)]))
}
