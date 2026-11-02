//! The compensation charges of Bloch pair densities among themselves: the dense reciprocal sum
//! `(1/V) sum_G v(|G + p|) g_a^*(G + p) g_b(G + p)` at a Bloch shift `p` under an arbitrary
//! kernel `v` (a screened hybrid's exchange), with the kernel-weighted on-site tail beyond the
//! cutoff, and the same sum weighted by `i (G + p)` whose product with `delta_aA - delta_bA` is
//! its derivative when atom `A` moves.
//!
//! [`super::electrostatics::couplings`] is the special case `p = 0`, bare Coulomb, `G = 0`
//! dropped.
use super::shapes::{visit_lattice, Shapes, Sites, Transient};
use crate::{math::radial::integrate, periodic::invalid, product, Context, Result};
use num_complex::Complex64;
use std::f64::consts::PI;

/// `(rows x rows)` row-major matrices of the dense sum, and with it the three `i (G + p)`
/// weighted ones when `gradient` is set.
fn sum(
    ctx: &Context,
    shapes: &Shapes,
    sites: &Sites,
    cutoff: f64,
    shift: [f64; 3],
    kernel: &dyn Fn(f64) -> f64,
    gradient: bool,
) -> Result<Vec<Vec<Complex64>>> {
    sites.check()?;
    if !(cutoff.is_finite() && cutoff > 0.) || shift.iter().any(|x| !x.is_finite()) {
        return Err(invalid(
            "a dense sum needs a positive cutoff and a finite shift",
        ));
    }
    let rows = sites.channels.len();
    let volume = sites.volume();
    let count = if gradient { 3 } else { 1 };
    let mut out = vec![vec![Complex64::default(); rows * rows]; count];
    let block = (Transient::of(ctx).dense_bytes / ((rows + 2) * 16 + 64)).max(1);
    let _scratch = ctx.reserve(product(&[block, (rows + 2) * 16 + 64])?)?;
    let reach = cutoff + shift.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut f = vec![Complex64::default(); rows * block];
    visit_lattice(&sites.reciprocal()?, reach, false, block, &mut |vectors| {
        let shifted: Vec<[f64; 3]> = vectors
            .iter()
            .map(|g| [g[0] + shift[0], g[1] + shift[1], g[2] + shift[2]])
            .filter(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2] <= cutoff * cutoff)
            .collect();
        let m = shifted.len();
        if m == 0 {
            return Ok(());
        }
        shapes.transforms(sites, &shifted, &mut f[..rows * m], m)?;
        let v: Vec<f64> = shifted
            .iter()
            .map(|g| kernel(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]))
            .collect();
        for slot in 0..count {
            // Axis `slot - 1` of `i (G + p)` for the gradient sums, one for the plain sum.
            let weight: Vec<Complex64> = (0..m)
                .map(|g| {
                    if gradient {
                        Complex64::new(0., v[g] * shifted[g][slot])
                    } else {
                        Complex64::new(v[g], 0.)
                    }
                })
                .collect();
            for a in 0..rows {
                let left: Vec<Complex64> =
                    (0..m).map(|g| f[a * m + g].conj() * weight[g]).collect();
                for b in 0..rows {
                    out[slot][a * rows + b] +=
                        (0..m).map(|g| left[g] * f[b * m + g]).sum::<Complex64>();
                }
            }
        }
        Ok(())
    })?;
    for matrix in &mut out {
        matrix.iter_mut().for_each(|x| *x /= volume);
    }
    Ok(out)
}

/// `8 int_cutoff^inf F_L(q)^2 [v(q^2) q^2 / 4 pi] dq` on the shape table, trapezoid: the on-site
/// self-energy of a unit `g_L Y_LM` beyond the cutoff under the kernel `v` (the bare Coulomb
/// kernel makes the bracket one).
fn tail(
    shapes: &Shapes,
    radius: f64,
    l: usize,
    cutoff: f64,
    kernel: &dyn Fn(f64) -> f64,
) -> Result<f64> {
    let table = shapes.table(radius, l)?;
    let from = table.q.partition_point(|v| *v < cutoff);
    let weighted: Vec<f64> = table.q[from..]
        .iter()
        .zip(&table.f[from..])
        .map(|(q, f)| f * f * kernel(q * q) * q * q / (4. * PI))
        .collect();
    Ok(8. * integrate(&table.q[from..], &weighted)?)
}

/// `U[a, b] = (1/V) sum_{|G + p| <= cutoff} v(|G + p|^2) g_a^*(G + p) g_b(G + p)` plus the
/// kernel-weighted on-site tail on the diagonal; `kernel` maps `|G + p|^2` to `v` and must be
/// finite at zero. Row-major `channels x channels`.
pub fn dense_compensation(
    ctx: &Context,
    shapes: &Shapes,
    sites: &Sites,
    cutoff: f64,
    shift: [f64; 3],
    kernel: &dyn Fn(f64) -> f64,
) -> Result<Vec<Complex64>> {
    let mut u = sum(ctx, shapes, sites, cutoff, shift, kernel, false)?.remove(0);
    let rows = sites.channels.len();
    for (a, &(atom, l, _)) in sites.channels.iter().enumerate() {
        let radius = sites.atoms[atom].compensation_radius;
        u[a * rows + a] += tail(shapes, radius, l, cutoff, kernel)?;
    }
    Ok(u)
}

/// The sum of [`dense_compensation`] with every term weighted by `i (G + p)_axis`, no tail: for
/// each axis, `d U[a, b] / d R_A = out[axis][a, b] * (delta_aA - delta_bA)`.
pub fn dense_compensation_gradient(
    ctx: &Context,
    shapes: &Shapes,
    sites: &Sites,
    cutoff: f64,
    shift: [f64; 3],
    kernel: &dyn Fn(f64) -> f64,
) -> Result<[Vec<Complex64>; 3]> {
    let mut parts = sum(ctx, shapes, sites, cutoff, shift, kernel, true)?;
    let z = parts.pop().unwrap();
    let y = parts.pop().unwrap();
    let x = parts.pop().unwrap();
    Ok([x, y, z])
}
