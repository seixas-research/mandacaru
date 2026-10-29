//! Bounded tiled projection of a local field into a real orbital basis.
use crate::{finite, product, Buffer, Context, Error, Reservation, Result};
use nalgebra::{DMatrixView, DMatrixViewMut};

pub struct RealProjector<'a> {
    samples: &'a [f64],
    orbitals: usize,
    points: usize,
    tile: usize,
    left: Buffer<f64>,
    right: Buffer<f64>,
    matrix: Buffer<f64>,
    _packing: Reservation,
}
impl<'a> RealProjector<'a> {
    pub fn new(ctx: &Context, samples: &'a [f64], orbitals: usize, points: usize) -> Result<Self> {
        if orbitals == 0 || points == 0 || samples.len() != product(&[orbitals, points])? {
            return Err(Error::Invalid(
                "orbital samples have inconsistent projection dimensions".into(),
            ));
        }
        finite(samples, "orbital samples")?;
        let tile = points.min(2048);
        Ok(Self {
            samples,
            orbitals,
            points,
            tile,
            left: ctx.allocate(product(&[orbitals, tile])?)?,
            right: ctx.allocate(product(&[orbitals, tile])?)?,
            matrix: ctx.allocate(product(&[orbitals, orbitals])?)?,
            _packing: ctx.reserve(product(&[orbitals, tile, 16])?)?,
        })
    }
    /// Column-major matrix `sum_k phi[a,k] field[k] phi[b,k]`.
    pub fn project(&mut self, field: &[f64]) -> Result<&[f64]> {
        if field.len() != self.points {
            return Err(Error::Invalid(
                "local field does not match the orbital grid".into(),
            ));
        }
        finite(field, "local field")?;
        let m = self.orbitals;
        self.matrix.as_mut_slice().fill(0.);
        for start in (0..self.points).step_by(self.tile) {
            let count = self.tile.min(self.points - start);
            for a in 0..m {
                for k in 0..count {
                    let value = self.samples[a * self.points + start + k];
                    self.left.as_mut_slice()[k * m + a] = value;
                    self.right.as_mut_slice()[a * self.tile + k] = value * field[start + k];
                }
            }
            let left = DMatrixView::from_slice(self.left.as_slice(), m, self.tile);
            let right = DMatrixView::from_slice(self.right.as_slice(), self.tile, m);
            let mut matrix = DMatrixViewMut::from_slice(self.matrix.as_mut_slice(), m, m);
            matrix.gemm(1., &left.columns(0, count), &right.rows(0, count), 1.);
        }
        Ok(self.matrix.as_slice())
    }
}
