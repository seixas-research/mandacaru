//! The lattice images of an atom-centered function that reach a region, and the Bloch phases
//! of those images.
//!
//! A Bloch sum `sum_R e^{i k.R} phi(r - c - R)` over a ball of points needs only the images whose
//! support touches the ball: `|c + R - center| <= support + radius`. The same test decides which
//! functions reach a quadrature sphere at all ([`reaching`]). Atomic units; lattice vectors are the
//! columns of the matrix, and the images come out `n1` slowest like the grid.
use super::shapes::visit_lattice;
use crate::{Buffer, Context, Reservation, Result};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
use std::collections::HashMap;

/// A ball of points: the home cell's bounding ball, or a quadrature sphere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub center: [f64; 3],
    pub radius: f64,
}

/// The lattice vectors `R` for which a function centered at `center` with the given support
/// reaches the region. The list is charged to the `Context` that built it.
#[derive(Debug)]
pub struct Images {
    vectors: Vec<[f64; 3]>,
    _held: Reservation,
}

impl Images {
    /// Images of the function at `center` (support radius `support`) that reach `region`.
    pub fn around(
        ctx: &Context,
        lattice: &Matrix3<f64>,
        region: &Region,
        center: [f64; 3],
        support: f64,
    ) -> Result<Self> {
        let reach = support + region.radius;
        let offset = (Vector3::from(center) - Vector3::from(region.center)).norm();
        let mut vectors = Vec::new();
        visit_lattice(lattice, reach + offset, false, 4096, &mut |block| {
            for r in block {
                let d = Vector3::from(center) + Vector3::from(*r) - Vector3::from(region.center);
                if d.norm() <= reach {
                    vectors.push(*r);
                }
            }
            Ok(())
        })?;
        let held = ctx.reserve(vectors.len().saturating_mul(24))?;
        Ok(Self {
            vectors,
            _held: held,
        })
    }
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    pub fn vectors(&self) -> &[[f64; 3]] {
        &self.vectors
    }
    /// `exp(i sign k.R)` for every k-point (row) and image (column), row-major. A Bloch sum of
    /// values takes `sign = +1`; the projection of a Bloch sum, which conjugates the function,
    /// takes `sign = -1`.
    pub fn phases(
        &self,
        ctx: &Context,
        kpoints: &[[f64; 3]],
        sign: f64,
    ) -> Result<Buffer<Complex64>> {
        let mut out = ctx.allocate::<Complex64>(kpoints.len().saturating_mul(self.len()))?;
        for (row, k) in out
            .as_mut_slice()
            .chunks_mut(self.len().max(1))
            .zip(kpoints)
        {
            for (phase, r) in row.iter_mut().zip(&self.vectors) {
                let angle = sign * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2]);
                *phase = Complex64::new(angle.cos(), angle.sin());
            }
        }
        Ok(out)
    }
}

/// Whether each body `(center, support)` has an image reaching the region. Bodies that share a
/// center and a support share one verdict.
pub fn reaching(
    ctx: &Context,
    lattice: &Matrix3<f64>,
    region: &Region,
    bodies: &[([f64; 3], f64)],
) -> Result<Vec<bool>> {
    let mut verdicts: HashMap<[i64; 4], bool> = HashMap::new();
    let key = |x: f64| (x * 1e12).round() as i64;
    let mut out = Vec::with_capacity(bodies.len());
    for (center, support) in bodies {
        let id = [
            key(center[0]),
            key(center[1]),
            key(center[2]),
            key(*support),
        ];
        let verdict = match verdicts.get(&id) {
            Some(v) => *v,
            None => {
                let v = !Images::around(ctx, lattice, region, *center, *support)?.is_empty();
                verdicts.insert(id, v);
                v
            }
        };
        out.push(verdict);
    }
    Ok(out)
}
