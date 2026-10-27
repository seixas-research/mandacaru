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
