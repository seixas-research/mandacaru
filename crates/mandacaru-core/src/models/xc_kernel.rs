//! The adiabatic exchange-correlation kernel of LDA, PBE and the semilocal part of HSE06 on a
//! molecular grid.
//!
//! The kernel is the second derivative of the discretized energy
//! `E = dv sum_k f(n_up, n_dn, s_uu, s_ud, s_dd)` that
//! [`evaluate_fields_relativistic`](super::xc::evaluate_fields_relativistic) differentiates once:
//! the same point function ([`point_hessian`](super::xc::point_hessian) runs it forward over
//! forward), the same grid stencil for the density gradient (`s_st = grad n_s . grad n_t`) and
//! the same skew-adjoint divergence of the flux. So [`Kernel::apply`] is exactly the
//! derivative of that potential along a density perturbation: `dv_s = d(v_s) / d(n) . dn`.
//! The perturbation `dn` is the valence response; the smooth core in `densities` is frozen.
//!
//! Atomic units: densities in Bohr^-3, potentials in Hartree.
use super::{invalid, xc};
use crate::{math::SpatialGrid, product, Buffer, Context, Result};

/// The packed upper triangle of a symmetric 5 x 5 matrix.
const PACKED: usize = 15;
fn packed(i: usize, j: usize) -> usize {
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    a * 5 - a * (a + 1) / 2 + b
}

/// The point kernels of one ground-state density.
pub struct Kernel {
    gradient: bool,
    points: usize,
    /// `[point][15]`: the second partials in `(n_up, n_dn, s_uu, s_ud, s_dd)`.
    second: Buffer<f64>,
    /// `[point][3]`: `df/ds_uu`, `df/ds_ud`, `df/ds_dd` (gradient functionals only).
    first: Buffer<f64>,
    /// `[spin][axis][point]`: the ground-state density gradients (gradient functionals only).
    slopes: Buffer<f64>,
}

/// The work arrays of one [`Kernel::apply`]; one per lane.
pub struct Scratch {
    slopes: Buffer<f64>,
    flux: Buffer<f64>,
    work: Buffer<f64>,
}

impl Kernel {
    /// The kernel of the semilocal functional `name` (`"lda"`, `"pbe"` or `"hse06"`, the
    /// semilocal part of the screened hybrid) at the two spin
    /// densities `densities` (`[spin][point]`, the smooth core included exactly as the
    /// potential sees it). `relativistic` is the relativistic exchange factor of the potential.
    pub fn new(
        ctx: &Context,
        grid: &SpatialGrid,
        densities: &[f64],
        name: &str,
        relativistic: bool,
    ) -> Result<Self> {
        let name = match name {
            "lda" => "lda",
            "pbe" => "pbe",
            "hse06" => "hse06",
            _ => return Err(invalid("the XC kernel is defined for lda, pbe and hse06")),
        };
        let n = grid.index.points();
        if densities.len() != product(&[2, n])? || densities.iter().any(|v| !v.is_finite()) {
            return Err(invalid("the XC kernel needs two finite grid densities"));
        }
        let gradient = name != "lda";
        let mut second = ctx.allocate::<f64>(product(&[n, PACKED])?)?;
        let mut first = ctx.allocate::<f64>(if gradient { product(&[n, 3])? } else { 0 })?;
        let mut slopes = ctx.allocate::<f64>(if gradient { product(&[6, n])? } else { 0 })?;
        if gradient {
            for s in 0..2 {
                for a in 0..3 {
                    xc::derivative(
                        grid,
                        &densities[s * n..(s + 1) * n],
                        a,
                        &mut slopes.as_mut_slice()[(s * 3 + a) * n..(s * 3 + a + 1) * n],
                    )?;
                }
            }
        }
        for k in 0..n {
            // The potential gives a point below the weight threshold no gradient
            // contraction, so its kernel has no gradient part there either.
            let weighted = densities[k] + densities[n + k] > 1e-10;
            let mut sigma = [0.; 3];
            if gradient && weighted {
                let g = slopes.as_slice();
                for a in 0..3 {
                    let u = g[a * n + k];
                    let d = g[(3 + a) * n + k];
                    sigma[0] += u * u;
                    sigma[1] += u * d;
                    sigma[2] += d * d;
                }
            }
            let (_, partials, hessian) = xc::point_hessian(
                name,
                [densities[k], densities[n + k], sigma[0], sigma[1], sigma[2]],
                relativistic,
            )?;
            let row = &mut second.as_mut_slice()[k * PACKED..(k + 1) * PACKED];
            for i in 0..5 {
                for j in i..5 {
                    let keep = (i < 2 && j < 2) || (gradient && weighted);
                    row[packed(i, j)] = if keep { hessian[i][j] } else { 0. };
                }
            }
            if gradient && weighted {
                first.as_mut_slice()[k * 3..k * 3 + 3].copy_from_slice(&partials[2..5]);
            }
        }
        Ok(Self {
            gradient,
            points: n,
            second,
            first,
            slopes,
        })
    }
    pub fn points(&self) -> usize {
        self.points
    }
    /// Work arrays for [`Self::apply`].
    pub fn scratch(&self, ctx: &Context) -> Result<Scratch> {
        let n = self.points;
        let size = if self.gradient { product(&[6, n])? } else { 0 };
        Ok(Scratch {
            slopes: ctx.allocate(size)?,
            flux: ctx.allocate(size)?,
            work: ctx.allocate(if self.gradient { n } else { 0 })?,
        })
    }
    /// `out[spin][point] = sum_t f_st dn_t` with the gradient terms: the first-order change of
    /// the potential of each spin under the density change `delta` (`[spin][point]`).
    pub fn apply(
        &self,
        grid: &SpatialGrid,
        delta: &[f64],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let n = self.points;
        if grid.index.points() != n || delta.len() != 2 * n || out.len() != 2 * n {
            return Err(invalid(
                "the XC kernel and its density change disagree in size",
            ));
        }
        if self.gradient {
            for s in 0..2 {
                for a in 0..3 {
                    xc::derivative(
                        grid,
                        &delta[s * n..(s + 1) * n],
                        a,
                        &mut scratch.slopes.as_mut_slice()[(s * 3 + a) * n..(s * 3 + a + 1) * n],
                    )?;
                }
            }
        }
        let g = self.slopes.as_slice();
        let dg = scratch.slopes.as_slice();
        let flux = scratch.flux.as_mut_slice();
        for k in 0..n {
            let h = &self.second.as_slice()[k * PACKED..(k + 1) * PACKED];
            let mut dx = [delta[k], delta[n + k], 0., 0., 0.];
            if self.gradient {
                for a in 0..3 {
                    let (u, d) = (g[a * n + k], g[(3 + a) * n + k]);
                    let (du, dd) = (dg[a * n + k], dg[(3 + a) * n + k]);
                    dx[2] += 2. * u * du;
                    dx[3] += u * dd + d * du;
                    dx[4] += 2. * d * dd;
                }
            }
            let response: [f64; 5] =
                std::array::from_fn(|i| (0..5).map(|j| h[packed(i, j)] * dx[j]).sum());
            out[k] = response[0];
            out[n + k] = response[1];
            if self.gradient {
                let f = &self.first.as_slice()[k * 3..k * 3 + 3];
                for a in 0..3 {
                    let (u, d) = (g[a * n + k], g[(3 + a) * n + k]);
                    let (du, dd) = (dg[a * n + k], dg[(3 + a) * n + k]);
                    flux[a * n + k] =
                        2. * response[2] * u + 2. * f[0] * du + response[3] * d + f[1] * dd;
                    flux[(3 + a) * n + k] =
                        2. * response[4] * d + 2. * f[2] * dd + response[3] * u + f[1] * du;
                }
            }
        }
        if self.gradient {
            let work = scratch.work.as_mut_slice();
            for s in 0..2 {
                for a in 0..3 {
                    xc::derivative(grid, &flux[(s * 3 + a) * n..(s * 3 + a + 1) * n], a, work)?;
                    for k in 0..n {
                        out[s * n + k] -= work[k];
                    }
                }
            }
        }
        Ok(())
    }
}
