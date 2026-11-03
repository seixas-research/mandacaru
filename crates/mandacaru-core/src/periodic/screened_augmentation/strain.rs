//! The strain derivative of the pair shapes of the exact exchange (H-3 of the stress plan).
//!
//! `T_a(G) = hat_a(Q) e^{+iG.r0} / dv` with `hat_a(Q) = 4 pi (-i)^L Y_LM(Q^) F_L(|Q|) e^{-iQ.R}`
//! and `Q` the box vector of the grid index shifted by `k_diff` and wrapped into the FFT box (one
//! plain `Q`, no alias average: [`Augmentation::pair_shapes`]). Under the strain
//! `x -> (1 + eps) x`, at fixed fractional `k_diff` and fixed integer indices:
//!
//! ```text
//! Q -> (1 - eps^T) Q,   Q.R and G.r0 are invariant,   dv -> dv (1 + tr eps)
//! dT_a / d eps_ab = - delta_ab T_a - Q_a (d_b f_a)(Q) e^{-iQ.R} e^{iG.r0} / dv
//! ```
//!
//! where `f_a = 4 pi (-i)^L Y_LM F_L` and `d_b` is the Cartesian gradient in `Q` of
//! [`Shapes::transform_gradients`]: the angular gradient of the harmonic and the SLOPE of the
//! piecewise-linear shape table (the interpolant the energy reads, not the transform it
//! tabulates). The first term is the `1 / dv` of the compensation charge and is
//! [`ShapeStrain`]'s caller's (it needs only `T_a`); this module returns the second. At `Q = 0`
//! the factor `Q_a` removes it.
use super::{invalid, Augmentation, Crystal};
use crate::{
    math::{fft::frequency, reciprocal},
    periodic::setup::shapes::Transient,
    quantum::C64,
    Buffer, Context, Result,
};
use nalgebra::Vector3;

/// The gradients of the pair shapes in `Q` and the wrapped `Q` they are taken at, for one
/// `k_diff = q - k`.
pub struct ShapeStrain {
    rows: usize,
    n: usize,
    /// `e^{iG.r0}/dv * e^{-iQ.R} d_b f_a(Q)` at `[(row * 3 + b) * n + v]`.
    gradients: Buffer<C64>,
    /// The wrapped `Q` at `[axis * n + v]`.
    wrapped: Buffer<f64>,
}

impl ShapeStrain {
    /// The number of compensation channels.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The wrapped `Q_axis` of index `v`.
    pub fn wave(&self, axis: usize, v: usize) -> f64 {
        self.wrapped.as_slice()[axis * self.n + v]
    }

    /// `-Q_a (d_b f_row)(Q) e^{-iQ.R} e^{iG.r0} / dv` for every channel and index
    /// (`rows * n`, row-major): the form-factor part of `dT/d eps_ab`.
    pub fn form(&self, ctx: &Context, a: usize, b: usize) -> Result<Buffer<C64>> {
        if a > 2 || b > 2 {
            return Err(invalid("a strain component is a pair of axes 0, 1 or 2"));
        }
        let n = self.n;
        let mut out = ctx.allocate::<C64>(self.rows * n)?;
        let q = &self.wrapped.as_slice()[a * n..(a + 1) * n];
        for row in 0..self.rows {
            let slope = &self.gradients.as_slice()[(row * 3 + b) * n..(row * 3 + b + 1) * n];
            for (v, to) in out.as_mut_slice()[row * n..(row + 1) * n]
                .iter_mut()
                .enumerate()
            {
                *to = slope[v] * -q[v];
            }
        }
        Ok(out)
    }
}

impl Augmentation<'_> {
    /// The pieces of the strain derivative of [`Augmentation::pair_shapes`] for the pairs of two
    /// points whose wave vectors differ by `k_diff = q - k`: the wrapped `Q` of
    /// [`Augmentation::pair_shapes`] and the gradients of its shapes (see the module).
    pub fn pair_shape_strain(&self, c: &Crystal, k_diff: [f64; 3]) -> Result<ShapeStrain> {
        let rows = self.sites.channels.len();
        let n = c.grid.index.points();
        let shape = c.grid.index.shape();
        let dv = c.grid.index.dv();
        let mut gradients = c.context.allocate::<C64>(rows * 3 * n)?;
        let mut wrapped = c.context.allocate::<f64>(3 * n)?;
        if rows == 0 {
            return Ok(ShapeStrain {
                rows,
                n,
                gradients,
                wrapped,
            });
        }
        let b = reciprocal::reciprocal_cell(&c.grid)?;
        let kappa = b
            .try_inverse()
            .ok_or_else(|| invalid("singular reciprocal cell"))?
            * Vector3::from(k_diff);
        let block = (Transient::of(&c.context).dense_bytes / (rows * 3 * 16 + 64 + 56)).max(1);
        let _scratch = c
            .context
            .reserve(crate::product(&[block.min(n), rows * 3 * 16 + 64 + 56])?)?;
        let mut vectors: Vec<[f64; 3]> = Vec::with_capacity(block.min(n));
        let mut phases: Vec<C64> = Vec::with_capacity(block.min(n));
        for start in (0..n).step_by(block) {
            vectors.clear();
            phases.clear();
            for at in start..(start + block).min(n) {
                let index = [
                    at / (shape[1] * shape[2]),
                    at / shape[2] % shape[1],
                    at % shape[2],
                ];
                // The wrapping of `shape_rows`, line for line: one plain Q per index.
                let box_vector = Vector3::from_fn(|a, _| {
                    let size = shape[a] as f64;
                    let x = frequency(index[a], shape[a]) + kappa[a];
                    x - size * ((x + 0.5 * size + 1e-9) / size).floor()
                });
                let q = b * box_vector;
                let g = q - Vector3::from(k_diff);
                let angle =
                    g[0] * c.grid.origin[0] + g[1] * c.grid.origin[1] + g[2] * c.grid.origin[2];
                for axis in 0..3 {
                    wrapped.as_mut_slice()[axis * n + at] = q[axis];
                }
                vectors.push(q.into());
                phases.push(C64::from_polar(1. / dv, angle));
            }
            self.shapes.transform_gradients(
                &self.sites,
                &vectors,
                &mut gradients.as_mut_slice()[start..],
                n,
            )?;
            for row in 0..rows {
                for axis in 0..3 {
                    for (v, phase) in phases.iter().enumerate() {
                        gradients.as_mut_slice()[(row * 3 + axis) * n + start + v] *= phase;
                    }
                }
            }
        }
        Ok(ShapeStrain {
            rows,
            n,
            gradients,
            wrapped,
        })
    }
}
