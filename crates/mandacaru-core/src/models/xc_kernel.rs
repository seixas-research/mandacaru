//! The adiabatic exchange-correlation kernel of LDA, PBE and the semilocal part of HSE06 on a
//! molecular grid, and r2SCAN's point kernel ([`MetaKernel`]).
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

/// Half the second change of the LDA, PBE or HSE06 semilocal potential of each spin along a density change:
/// `out[spin][point]` is the `t^2` coefficient of `v_s(n + t delta)`, the third derivative of
/// the energy contracted twice with `delta` over two (`d^2 v / dt^2 = 2 out`). It is the
/// derivative of [`Kernel::apply`] along `delta` (`densities` as in [`Kernel::new`], the smooth
/// core included), with the same stencil, threshold and divergence: the third-order term of the
/// excited-state gradient (`dev_plan/TDDFT_PLAN.md`, T7).
pub fn second_change(
    ctx: &Context,
    grid: &SpatialGrid,
    densities: &[f64],
    name: &str,
    relativistic: bool,
    delta: &[f64],
) -> Result<Buffer<f64>> {
    use xc::Taylor2;
    let gradient = match name {
        "lda" => false,
        "pbe" | "hse06" => true,
        _ => {
            return Err(invalid(
                "the third-order XC kernel is defined for lda, pbe and hse06",
            ))
        }
    };
    let n = grid.index.points();
    if densities.len() != product(&[2, n])?
        || delta.len() != 2 * n
        || densities.iter().chain(delta).any(|v| !v.is_finite())
    {
        return Err(invalid(
            "the third-order XC kernel needs two finite densities and changes",
        ));
    }
    let size = if gradient { product(&[6, n])? } else { 0 };
    let (mut g, mut dg, mut flux) = (
        ctx.allocate::<f64>(size)?,
        ctx.allocate::<f64>(size)?,
        ctx.allocate::<f64>(size)?,
    );
    if gradient {
        for s in 0..2 {
            for a in 0..3 {
                let at = (s * 3 + a) * n..(s * 3 + a + 1) * n;
                xc::derivative(
                    grid,
                    &densities[s * n..(s + 1) * n],
                    a,
                    &mut g.as_mut_slice()[at.clone()],
                )?;
                xc::derivative(
                    grid,
                    &delta[s * n..(s + 1) * n],
                    a,
                    &mut dg.as_mut_slice()[at],
                )?;
            }
        }
    }
    let mut out = ctx.allocate::<f64>(2 * n)?;
    let (g, dg) = (g.as_slice(), dg.as_slice());
    for k in 0..n {
        let weighted = densities[k] + densities[n + k] > 1e-10;
        let mut sigma = [Taylor2::new([0.; 3]); 3];
        if gradient && weighted {
            for a in 0..3 {
                let u = Taylor2::new([g[a * n + k], dg[a * n + k], 0.]);
                let d = Taylor2::new([g[(3 + a) * n + k], dg[(3 + a) * n + k], 0.]);
                sigma[0] = sigma[0] + u * u;
                sigma[1] = sigma[1] + u * d;
                sigma[2] = sigma[2] + d * d;
            }
        }
        let p = xc::curve_partials(
            name,
            [
                Taylor2::new([densities[k], delta[k], 0.]),
                Taylor2::new([densities[n + k], delta[n + k], 0.]),
                sigma[0],
                sigma[1],
                sigma[2],
            ],
            relativistic,
        )?;
        out.as_mut_slice()[k] = p[0].c[2];
        out.as_mut_slice()[n + k] = p[1].c[2];
        if gradient && weighted {
            // flux_up = 2 f_uu grad n_up + f_ud grad n_dn (and its mirror), each factor along the
            // curve; the t^2 coefficient of the product.
            let two = |f: Taylor2, x: f64, dx: f64| f.c[2] * x + f.c[1] * dx;
            let flux = flux.as_mut_slice();
            for a in 0..3 {
                let (u, d) = (g[a * n + k], g[(3 + a) * n + k]);
                let (du, dd) = (dg[a * n + k], dg[(3 + a) * n + k]);
                flux[a * n + k] = 2. * two(p[2], u, du) + two(p[3], d, dd);
                flux[(3 + a) * n + k] = 2. * two(p[4], d, dd) + two(p[3], u, du);
            }
        }
    }
    if gradient {
        let mut work = ctx.allocate::<f64>(n)?;
        for s in 0..2 {
            for a in 0..3 {
                xc::derivative(
                    grid,
                    &flux.as_slice()[(s * 3 + a) * n..(s * 3 + a + 1) * n],
                    a,
                    work.as_mut_slice(),
                )?;
                for k in 0..n {
                    out.as_mut_slice()[s * n + k] -= work.as_slice()[k];
                }
            }
        }
    }
    Ok(out)
}

/// The packed upper triangle of a symmetric 7 x 7 matrix.
const PACKED_META: usize = 28;
fn packed_meta(i: usize, j: usize) -> usize {
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    a * 7 - a * (a + 1) / 2 + b
}

/// The point kernels of r2SCAN at one ground state, in the form the molecular meta-GGA potential
/// takes (`evaluate_fields_relativistic` with the orbitals' own density gradient): the density
/// gradient and the kinetic densities are inputs built from the orbitals, and the potential of the
/// gradient is the flux `df / d(grad n_s)`, contracted by the caller with the orbital products.
/// [`MetaKernel::apply`] is the exact derivative of `(dE/dn, dE/dtau, flux)` along a change of the
/// three inputs.
pub struct MetaKernel {
    points: usize,
    /// `[point][28]`: second partials in `(n_up, n_dn, s_uu, s_ud, s_dd, tau_up, tau_dn)`.
    second: Buffer<f64>,
    /// `[point][3]`: `df/ds_uu`, `df/ds_ud`, `df/ds_dd`.
    first: Buffer<f64>,
    /// `[spin][axis][point]`: the ground-state density gradients the potential saw.
    gradients: Buffer<f64>,
}

/// What [`MetaKernel::apply`] returns per spin: the changes of `dE/dn` (`local`), of `dE/dtau`
/// (`tau`) and of the flux (`flux`, `[spin][axis][point]`).
pub struct MetaChange<'a> {
    pub local: &'a mut [f64],
    pub tau: &'a mut [f64],
    pub flux: &'a mut [f64],
}

impl MetaKernel {
    /// The r2SCAN kernel at the spin densities `densities` (`[spin][point]`), kinetic densities
    /// `kinetic` (`[spin][point]`) and density gradients `gradients` (`[spin][axis][point]`), each
    /// exactly as the potential sees them (the smooth core included).
    pub fn new(
        ctx: &Context,
        densities: &[f64],
        kinetic: &[f64],
        gradients: &[f64],
    ) -> Result<Self> {
        let n = densities.len() / 2;
        if densities.len() != 2 * n
            || kinetic.len() != 2 * n
            || gradients.len() != product(&[6, n])?
            || densities
                .iter()
                .chain(kinetic)
                .chain(gradients)
                .any(|v| !v.is_finite())
        {
            return Err(invalid(
                "the meta-GGA kernel needs finite densities, kinetic densities and gradients",
            ));
        }
        let mut second = ctx.allocate::<f64>(product(&[n, PACKED_META])?)?;
        let mut first = ctx.allocate::<f64>(product(&[n, 3])?)?;
        for k in 0..n {
            // The potential gives a point below the weight threshold nothing at all.
            if densities[k] + densities[n + k] <= 1e-10 {
                continue;
            }
            let mut sigma = [0.; 3];
            for a in 0..3 {
                let u = gradients[a * n + k];
                let d = gradients[(3 + a) * n + k];
                sigma[0] += u * u;
                sigma[1] += u * d;
                sigma[2] += d * d;
            }
            let (_, partials, hessian) = xc::point_hessian_meta([
                densities[k],
                densities[n + k],
                sigma[0],
                sigma[1],
                sigma[2],
                kinetic[k],
                kinetic[n + k],
            ])?;
            let row = &mut second.as_mut_slice()[k * PACKED_META..(k + 1) * PACKED_META];
            for i in 0..7 {
                for j in i..7 {
                    row[packed_meta(i, j)] = hessian[i][j];
                }
            }
            first.as_mut_slice()[k * 3..k * 3 + 3].copy_from_slice(&partials[2..5]);
        }
        Ok(Self {
            points: n,
            second,
            first,
            gradients: ctx.copy_slice(gradients)?,
        })
    }
    pub fn points(&self) -> usize {
        self.points
    }
    /// The first-order changes of the potential's three parts along the changes `density`
    /// (`[spin][point]`), `gradient` (`[spin][axis][point]`) and `tau` (`[spin][point]`).
    pub fn apply(
        &self,
        density: &[f64],
        gradient: &[f64],
        tau: &[f64],
        out: MetaChange<'_>,
    ) -> Result<()> {
        let n = self.points;
        if density.len() != 2 * n
            || tau.len() != 2 * n
            || gradient.len() != 6 * n
            || out.local.len() != 2 * n
            || out.tau.len() != 2 * n
            || out.flux.len() != 6 * n
        {
            return Err(invalid(
                "the meta-GGA kernel and its changes disagree in size",
            ));
        }
        let g = self.gradients.as_slice();
        for k in 0..n {
            let h = &self.second.as_slice()[k * PACKED_META..(k + 1) * PACKED_META];
            let mut dx = [density[k], density[n + k], 0., 0., 0., tau[k], tau[n + k]];
            for a in 0..3 {
                let (u, d) = (g[a * n + k], g[(3 + a) * n + k]);
                let (du, dd) = (gradient[a * n + k], gradient[(3 + a) * n + k]);
                dx[2] += 2. * u * du;
                dx[3] += u * dd + d * du;
                dx[4] += 2. * d * dd;
            }
            let response: [f64; 7] =
                std::array::from_fn(|i| (0..7).map(|j| h[packed_meta(i, j)] * dx[j]).sum());
            out.local[k] = response[0];
            out.local[n + k] = response[1];
            out.tau[k] = response[5];
            out.tau[n + k] = response[6];
            let f = &self.first.as_slice()[k * 3..k * 3 + 3];
            for a in 0..3 {
                let (u, d) = (g[a * n + k], g[(3 + a) * n + k]);
                let (du, dd) = (gradient[a * n + k], gradient[(3 + a) * n + k]);
                out.flux[a * n + k] =
                    2. * response[2] * u + 2. * f[0] * du + response[3] * d + f[1] * dd;
                out.flux[(3 + a) * n + k] =
                    2. * response[4] * d + 2. * f[2] * dd + response[3] * u + f[1] * du;
            }
        }
        Ok(())
    }
}

/// Half the second change of r2SCAN's three potential parts along a curve of its inputs: the
/// `t^2` coefficients of `dE/dn`, `dE/dtau` and the flux (`[local 2n | tau 2n | flux 6n]`,
/// spin-major, the flux `[spin][axis][point]`) at the densities `n + t dn`, the density gradients
/// `g + t dg` and the kinetic densities `tau + t dtau + t^2 tau2` (`tau2` the curvature of a
/// gauge-invariant kinetic density along an imaginary density change, or zero). The ground fields
/// are those of [`MetaKernel::new`]; the curve's `t` coefficients reproduce [`MetaKernel::apply`],
/// so this is the third-order term of the excited-state gradient (`dev_plan/TDDFT_PLAN.md`, T7).
#[allow(clippy::too_many_arguments)]
pub fn second_change_meta(
    ctx: &Context,
    densities: &[f64],
    kinetic: &[f64],
    gradients: &[f64],
    density: &[f64],
    gradient: &[f64],
    tau: &[f64],
    tau2: &[f64],
) -> Result<Buffer<f64>> {
    use xc::Taylor2;
    let n = densities.len() / 2;
    if densities.len() != 2 * n
        || [kinetic, density, tau, tau2]
            .iter()
            .any(|v| v.len() != 2 * n)
        || [gradients, gradient].iter().any(|v| v.len() != 6 * n)
        || [densities, kinetic, gradients, density, gradient, tau, tau2]
            .iter()
            .any(|v| v.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid(
            "the third-order meta-GGA kernel needs finite fields and changes of one grid",
        ));
    }
    let mut buffer = ctx.allocate::<f64>(product(&[10, n])?)?;
    let (local, rest) = buffer.as_mut_slice().split_at_mut(2 * n);
    let (kinetic_out, flux) = rest.split_at_mut(2 * n);
    for k in 0..n {
        // The potential gives a point below the weight threshold nothing at all.
        if densities[k] + densities[n + k] <= 1e-10 {
            continue;
        }
        let mut sigma = [Taylor2::new([0.; 3]); 3];
        for a in 0..3 {
            let u = Taylor2::new([gradients[a * n + k], gradient[a * n + k], 0.]);
            let d = Taylor2::new([gradients[(3 + a) * n + k], gradient[(3 + a) * n + k], 0.]);
            sigma[0] = sigma[0] + u * u;
            sigma[1] = sigma[1] + u * d;
            sigma[2] = sigma[2] + d * d;
        }
        let line = |base: &[f64], change: &[f64], s: usize| {
            Taylor2::new([base[s * n + k], change[s * n + k], 0.])
        };
        let p = xc::curve_partials_meta([
            line(densities, density, 0),
            line(densities, density, 1),
            sigma[0],
            sigma[1],
            sigma[2],
            Taylor2::new([kinetic[k], tau[k], tau2[k]]),
            Taylor2::new([kinetic[n + k], tau[n + k], tau2[n + k]]),
        ])?;
        local[k] = p[0].c[2];
        local[n + k] = p[1].c[2];
        kinetic_out[k] = p[5].c[2];
        kinetic_out[n + k] = p[6].c[2];
        // flux_up = 2 f_uu g_up + f_ud g_dn (and its mirror), each factor along the curve; the
        // t^2 coefficient of the product.
        let two = |f: Taylor2, x: f64, dx: f64| f.c[2] * x + f.c[1] * dx;
        for a in 0..3 {
            let (u, d) = (gradients[a * n + k], gradients[(3 + a) * n + k]);
            let (du, dd) = (gradient[a * n + k], gradient[(3 + a) * n + k]);
            flux[a * n + k] = 2. * two(p[2], u, du) + two(p[3], d, dd);
            flux[(3 + a) * n + k] = 2. * two(p[4], d, dd) + two(p[3], u, du);
        }
    }
    Ok(buffer)
}

/// The largest `|m|`, relative to the largest charge, of a reference the noncollinear kernel
/// takes as unpolarized: a closed-shell spinor SCF leaves a magnetization of roundoff size.
pub const UNPOLARIZED: f64 = 1e-10;

/// The adiabatic noncollinear kernel of LDA, PBE or HSE06's semilocal part on a molecular grid:
/// the derivative of the Pauli-channel potentials `(v_0, v_x, v_y, v_z)` of a spinor state
/// (`models::noncollinear`: `E = E_coll[(n +- |m|) / 2]`, `v_a = (v_up - v_dn) m_a / 2|m|`)
/// along a change `(dn, dm)`. The longitudinal change `m_hat . dm` goes through the collinear
/// [`Kernel`] of the frame densities; the transverse change turns `m_hat` and takes the weight
/// `(v_up - v_dn) / 2|m|`. Where every `|m|` is at or below [`UNPOLARIZED`] times the largest
/// charge (a Kramers-unpolarized reference, whose magnetization is roundoff) the
/// energy depends on `|dm|`, which is not differentiable for a gradient functional; the
/// kernel is then the collinear spin kernel along each axis, the derivative along every change
/// whose magnetization keeps one direction and sign (`dev_plan/DECISIONS.md`, T8).
pub struct NoncollinearKernel {
    collinear: Kernel,
    points: usize,
    /// The frame densities `(n +- |m|) / 2` the collinear kernel sees, the smooth core included.
    frame: Buffer<f64>,
    name: String,
    relativistic: bool,
    /// `[axis][point]`: `m / |m|`, zero at or below the floor.
    axes: Buffer<f64>,
    /// `(v_up - v_dn) / 2|m|`, zero at or below the floor.
    transverse: Buffer<f64>,
    /// `|m|`, zero at or below the floor.
    norms: Buffer<f64>,
    isotropic: bool,
}

impl NoncollinearKernel {
    /// The kernel of the semilocal functional `name` at the Pauli-channel densities `fields`
    /// (`[n | m_x | m_y | m_z]`, the smooth core in the charge as the potential sees it).
    pub fn new(
        ctx: &Context,
        grid: &SpatialGrid,
        fields: &[f64],
        name: &str,
        relativistic: bool,
    ) -> Result<Self> {
        use super::noncollinear::{rotate, NORM_FLOOR};
        let n = grid.index.points();
        if fields.len() != product(&[4, n])? {
            return Err(invalid(
                "the noncollinear kernel needs four Pauli-channel densities",
            ));
        }
        let largest = |values: &[f64]| values.iter().fold(0f64, |a, v| a.max(v.abs()));
        let norm = (0..n)
            .map(|k| {
                (1..4)
                    .map(|a| fields[a * n + k].powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0f64, f64::max);
        let isotropic = norm <= UNPOLARIZED * largest(&fields[..n]);
        // An unpolarized reference is its charge alone: the roundoff magnetization has no axis.
        let mut fields = fields.to_vec();
        if isotropic {
            fields[n..].fill(0.);
        }
        let fields = fields.as_slice();
        let rotated = rotate(ctx, n, fields, None)?;
        // The frame densities as the spinor potential sees them: nonnegative up to roundoff.
        let mut frame = ctx.copy_slice(rotated.densities.as_slice())?;
        for value in frame.as_mut_slice() {
            *value = value.max(0.);
        }
        let collinear = Kernel::new(ctx, grid, frame.as_slice(), name, relativistic)?;
        let terms = xc::evaluate_fields_relativistic(
            ctx,
            grid,
            frame.as_slice(),
            None,
            None,
            name,
            relativistic,
        )?;
        let v = terms.potentials.as_slice();
        let norms = rotated.norms.as_slice();
        let mut axes = ctx.allocate::<f64>(product(&[3, n])?)?;
        let mut transverse = ctx.allocate::<f64>(n)?;
        let mut kept = ctx.allocate::<f64>(n)?;
        for k in 0..n {
            if norms[k] > NORM_FLOOR {
                kept.as_mut_slice()[k] = norms[k];
                transverse.as_mut_slice()[k] = 0.5 * (v[k] - v[n + k]) / norms[k];
                for a in 0..3 {
                    axes.as_mut_slice()[a * n + k] = fields[(a + 1) * n + k] / norms[k];
                }
            }
        }
        Ok(Self {
            collinear,
            points: n,
            frame,
            name: name.to_string(),
            relativistic,
            axes,
            transverse,
            norms: kept,
            isotropic,
        })
    }
    pub fn points(&self) -> usize {
        self.points
    }
    /// Whether the reference is Kramers-unpolarized (the isotropic kernel).
    pub fn unpolarized(&self) -> bool {
        self.isotropic
    }
    /// Half the second change of the Pauli-channel potentials along `delta` (`4 x points`): the
    /// `t^2` coefficient of `(v_0, v_x, v_y, v_z)` along the change, the third-order term of the
    /// excited-state gradient (`d^2 v / dt^2 = 2 out`). It is the derivative of [`Self::apply`]
    /// along the charge of the reference. At a Kramers-unpolarized reference each axis is the
    /// collinear [`second_change`] of the frame change `(dn +- dm_a) / 2`: its spin part is
    /// `v_a`'s, and the charge part, counted once per axis, loses the twice-extra pure charge
    /// term. Odd orders in the magnetization vanish there, so the axes do not mix.
    ///
    /// At a magnetized reference the frame densities `(n + t dn +- |m + t dm|) / 2` move by
    /// `t r1 + t^2 r2`, with `r1 = (dn +- s) / 2` (`s = m_hat . dm`) and `r2 = +-|dm_perp|^2 / 4|m|`,
    /// and the axis turns by `t u1 + t^2 u2`, `u1 = dm_perp / |m|` and
    /// `u2 = -(s dm_perp + m_hat |dm_perp|^2 / 2) / |m|^2`. With the collinear potentials
    /// `V = V0 + t K[r1] + t^2 (K[r2] + S[r1])` (`S` the collinear second change) and
    /// `w = (V_up - V_dn) / 2`, the `t^2` coefficients are `v_0: (V_up + V_dn) / 2` and
    /// `v_a: w2 m_hat_a + w1 u1_a + w0 u2_a`. Points at or below the norm floor keep no
    /// magnetization response, as in [`Self::apply`].
    pub fn second_change(
        &self,
        ctx: &Context,
        grid: &SpatialGrid,
        delta: &[f64],
    ) -> Result<Buffer<f64>> {
        let n = self.points;
        if delta.len() != 4 * n {
            return Err(invalid(
                "the noncollinear second change needs four Pauli-channel changes",
            ));
        }
        let mut spins = ctx.allocate::<f64>(2 * n)?;
        if !self.isotropic {
            let (axes, norms) = (self.axes.as_slice(), self.norms.as_slice());
            let along = |k: usize| {
                (0..3)
                    .map(|a| axes[a * n + k] * delta[(a + 1) * n + k])
                    .sum::<f64>()
            };
            let perp = |k: usize, a: usize| delta[(a + 1) * n + k] - axes[a * n + k] * along(k);
            let squared = |k: usize| (0..3).map(|a| perp(k, a).powi(2)).sum::<f64>();
            let mut curve = ctx.allocate::<f64>(2 * n)?;
            for k in 0..n {
                let s = along(k);
                spins.as_mut_slice()[k] = 0.5 * (delta[k] + s);
                spins.as_mut_slice()[n + k] = 0.5 * (delta[k] - s);
                let bend = if norms[k] > 0. {
                    0.25 * squared(k) / norms[k]
                } else {
                    0.
                };
                curve.as_mut_slice()[k] = bend;
                curve.as_mut_slice()[n + k] = -bend;
            }
            let mut v2 = second_change(
                ctx,
                grid,
                self.frame.as_slice(),
                &self.name,
                self.relativistic,
                spins.as_slice(),
            )?;
            let mut scratch = self.collinear.scratch(ctx)?;
            let mut response = ctx.allocate::<f64>(2 * n)?;
            self.collinear.apply(
                grid,
                curve.as_slice(),
                response.as_mut_slice(),
                &mut scratch,
            )?;
            for (a, b) in v2.as_mut_slice().iter_mut().zip(response.as_slice()) {
                *a += b;
            }
            self.collinear.apply(
                grid,
                spins.as_slice(),
                response.as_mut_slice(),
                &mut scratch,
            )?;
            let (v2, v1) = (v2.as_slice(), response.as_slice());
            let transverse = self.transverse.as_slice();
            let mut out = ctx.allocate::<f64>(4 * n)?;
            let o = out.as_mut_slice();
            for k in 0..n {
                o[k] = 0.5 * (v2[k] + v2[n + k]);
                if norms[k] > 0. {
                    let (s, p2, norm) = (along(k), squared(k), norms[k]);
                    let (w2, w1, w0) = (
                        0.5 * (v2[k] - v2[n + k]),
                        0.5 * (v1[k] - v1[n + k]),
                        transverse[k] * norm,
                    );
                    for a in 0..3 {
                        let (u, d) = (axes[a * n + k], perp(k, a));
                        o[(a + 1) * n + k] =
                            w2 * u + w1 * d / norm - w0 * (s * d + 0.5 * u * p2) / (norm * norm);
                    }
                }
            }
            return Ok(out);
        }
        let mut collinear = |spin: Option<usize>| -> Result<Buffer<f64>> {
            for k in 0..n {
                let s = spin.map_or(0., |a| delta[a * n + k]);
                spins.as_mut_slice()[k] = 0.5 * (delta[k] + s);
                spins.as_mut_slice()[n + k] = 0.5 * (delta[k] - s);
            }
            second_change(
                ctx,
                grid,
                self.frame.as_slice(),
                &self.name,
                self.relativistic,
                spins.as_slice(),
            )
        };
        let mut out = ctx.allocate::<f64>(4 * n)?;
        let charge = collinear(None)?;
        for k in 0..n {
            let c = charge.as_slice();
            out.as_mut_slice()[k] = -(c[k] + c[n + k]);
        }
        for a in 1..4 {
            let r = collinear(Some(a))?;
            let r = r.as_slice();
            for k in 0..n {
                out.as_mut_slice()[k] += 0.5 * (r[k] + r[n + k]);
                out.as_mut_slice()[a * n + k] = 0.5 * (r[k] - r[n + k]);
            }
        }
        Ok(out)
    }
    /// Work arrays for [`Self::apply`].
    pub fn scratch(&self, ctx: &Context) -> Result<Scratch> {
        self.collinear.scratch(ctx)
    }
    /// `out = d(v_0, v_x, v_y, v_z)` (`4 x points`) along the change `delta` of the four
    /// Pauli-channel densities.
    pub fn apply(
        &self,
        ctx: &Context,
        grid: &SpatialGrid,
        delta: &[f64],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let n = self.points;
        if delta.len() != 4 * n || out.len() != 4 * n {
            return Err(invalid(
                "the noncollinear kernel and its change disagree in size",
            ));
        }
        let mut spins = ctx.allocate::<f64>(2 * n)?;
        let mut response = ctx.allocate::<f64>(2 * n)?;
        // The collinear kernel of the frame change `(dn +- ds) / 2`: `(dv_up + dv_dn) / 2` and
        // `(dv_up - dv_dn) / 2`.
        let mut collinear = |charge: &dyn Fn(usize) -> f64,
                             spin: &dyn Fn(usize) -> f64,
                             out: &mut dyn FnMut(usize, f64, f64)|
         -> Result<()> {
            for k in 0..n {
                spins.as_mut_slice()[k] = 0.5 * (charge(k) + spin(k));
                spins.as_mut_slice()[n + k] = 0.5 * (charge(k) - spin(k));
            }
            self.collinear
                .apply(grid, spins.as_slice(), response.as_mut_slice(), scratch)?;
            let r = response.as_slice();
            for k in 0..n {
                out(k, 0.5 * (r[k] + r[n + k]), 0.5 * (r[k] - r[n + k]));
            }
            Ok(())
        };
        out.fill(0.);
        if self.isotropic {
            // Unpolarized: charge and spin decouple, each axis takes the collinear spin kernel.
            for a in 1..4 {
                let charge = |k: usize| if a == 1 { delta[k] } else { 0. };
                let spin = |k: usize| delta[a * n + k];
                collinear(&charge, &spin, &mut |k, average, half| {
                    out[k] += average;
                    out[a * n + k] = half;
                })?;
            }
            return Ok(());
        }
        let (axes, transverse) = (self.axes.as_slice(), self.transverse.as_slice());
        let along = |k: usize| {
            (0..3)
                .map(|a| axes[a * n + k] * delta[(a + 1) * n + k])
                .sum::<f64>()
        };
        collinear(&|k| delta[k], &along, &mut |k, average, half| {
            out[k] = average;
            let s = along(k);
            for a in 0..3 {
                let u = axes[a * n + k];
                out[(a + 1) * n + k] = u * half + transverse[k] * (delta[(a + 1) * n + k] - u * s);
            }
        })
    }
}
