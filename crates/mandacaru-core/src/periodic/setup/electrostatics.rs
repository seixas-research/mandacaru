//! Electrostatics of the compact charges: the Coulomb kernel, the Gaussian ions
//! and the compensation shapes on the grid's reciprocal set (each frequency the
//! mean over its lattice aliases), and the dense lattice sums `U`, `U_ion` and
//! `E_ion` of the shapes and ions among themselves.
//!
//! Every dense sum runs in column blocks whose bytes are reserved from the
//! context; the block size follows `Transient::dense_bytes`, never the grid.
use super::shapes::{visit_lattice, Shapes, Sites, Transient};
use crate::{
    math::{reciprocal, SpatialGrid},
    Buffer, Context, Result,
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// `4 pi / G^2`, with `G = 0` dropped.
pub fn coulomb(g: [f64; 3]) -> f64 {
    let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    if g2 > 0. {
        4. * PI / g2
    } else {
        0.
    }
}

/// The Coulomb kernel on the grid's reciprocal set (C order), alias-averaged.
pub fn kernel(ctx: &Context, grid: &SpatialGrid) -> Result<Buffer<f64>> {
    reciprocal::alias_mean(ctx, grid, coulomb)
}

/// The Gaussian ions at `g` (electron sign): `sum_A -Z_A exp(-sigma^2 G^2 / 2) exp(-i G.R_A)`.
pub fn ion_transform(sites: &Sites, g: [f64; 3]) -> Complex64 {
    let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    let envelope = (-0.5 * sites.sigma * sites.sigma * g2).exp();
    let mut total = Complex64::default();
    for atom in sites.atoms {
        let angle = -(g[0] * atom.center[0] + g[1] * atom.center[1] + g[2] * atom.center[2]);
        total += Complex64::new(angle.cos(), angle.sin()) * (-atom.charge * envelope);
    }
    total
}

/// The Gaussian ions on the grid's reciprocal set, alias-averaged.
pub fn ion_charge(ctx: &Context, grid: &SpatialGrid, sites: &Sites) -> Result<Buffer<Complex64>> {
    sites.check()?;
    reciprocal::alias_mean(ctx, grid, |g| ion_transform(sites, g))
}

/// Largest wave number `hats` and `ion_charge` evaluate on this grid, aliases
/// included: the `q_max` its `Shapes` must cover.
pub fn max_wavenumber(ctx: &Context, grid: &SpatialGrid) -> Result<f64> {
    let b = reciprocal::reciprocal_cell(grid)?;
    let shape = grid.index.shape();
    let norm = |g: [f64; 3]| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let plain = (0..grid.index.points()).map(|at| norm(reciprocal::box_vector(&b, shape, at)));
    let table = reciprocal::alias_table(ctx, grid)?;
    Ok(plain
        .chain(table.vectors.iter().map(|v| norm(*v)))
        .fold(0., f64::max))
}

fn columns(ctx: &Context, rows: usize) -> usize {
    (Transient::of(ctx).dense_bytes / (rows * 16 + 64)).max(1)
}

/// `(channels, n)` row-major: every channel's compensation shape on the grid's
/// reciprocal set, alias-averaged. Grid points are processed in blocks of
/// `Transient::dense_bytes`.
pub fn hats(
    ctx: &Context,
    grid: &SpatialGrid,
    shapes: &Shapes,
    sites: &Sites,
) -> Result<Buffer<Complex64>> {
    sites.check()?;
    let rows = sites.channels.len();
    let shape = grid.index.shape();
    let n = grid.index.points();
    let mut out = ctx.allocate::<Complex64>(rows * n)?;
    if rows == 0 {
        return Ok(out);
    }
    let block = columns(ctx, rows);
    let _scratch = ctx.reserve(crate::product(&[block, rows * 16 + 64])?)?;
    let b = reciprocal::reciprocal_cell(grid)?;
    for start in (0..n).step_by(block) {
        let vectors: Vec<[f64; 3]> = (start..(start + block).min(n))
            .map(|at| reciprocal::box_vector(&b, shape, at))
            .collect();
        shapes.transforms(sites, &vectors, &mut out.as_mut_slice()[start..], n)?;
    }
    let table = reciprocal::alias_table(ctx, grid)?;
    let mut alias = vec![Complex64::default(); block.min(table.len()).max(1) * rows];
    for range in table.blocks(block) {
        let count = range.len();
        if alias.len() < count * rows {
            alias.resize(count * rows, Complex64::default());
        }
        shapes.transforms(sites, &table.vectors[range.clone()], &mut alias, count)?;
        for row in 0..rows {
            table.combine(
                &mut out.as_mut_slice()[row * n..(row + 1) * n],
                range.clone(),
                &alias[row * count..(row + 1) * count],
            );
        }
    }
    Ok(out)
}

/// `U[a, b]` (row-major, `channels x channels`), `U_ion[a]` and `E_ion`.
pub struct Couplings {
    pub u: Buffer<Complex64>,
    pub u_ion: Buffer<Complex64>,
    pub e_ion: f64,
}

/// `U[a,b] = (1/Omega) sum_{G != 0, |G| <= cutoff} g_a^*(G) (4 pi / G^2) g_b(G)`
/// plus the on-site tail beyond the cutoff on the diagonal, `U_ion[a]` the same
/// sum against the Gaussian ions and `E_ion = (1/2 Omega) sum 4 pi/G^2 |ion|^2`.
pub fn couplings(ctx: &Context, shapes: &Shapes, sites: &Sites, cutoff: f64) -> Result<Couplings> {
    sites.check()?;
    let rows = sites.channels.len();
    let volume = sites.volume();
    let mut u = ctx.allocate::<Complex64>(rows * rows)?;
    let mut u_ion = ctx.allocate::<Complex64>(rows)?;
    let mut e_ion = 0.;
    let block = columns(ctx, rows.max(1));
    let _scratch = ctx.reserve(crate::product(&[block, rows * 16 + 64])?)?;
    let mut f = vec![Complex64::default(); rows * block];
    let mut weighted = vec![Complex64::default(); block];
    let (mut k, mut ion) = (vec![0.; block], vec![Complex64::default(); block]);
    visit_lattice(&sites.reciprocal()?, cutoff, true, block, &mut |vectors| {
        let m = vectors.len();
        shapes.transforms(sites, vectors, &mut f[..rows * m], m)?;
        for (g, v) in vectors.iter().enumerate() {
            k[g] = coulomb(*v);
            ion[g] = ion_transform(sites, *v);
            e_ion += 0.5 * k[g] * ion[g].norm_sqr();
        }
        for a in 0..rows {
            for g in 0..m {
                weighted[g] = f[a * m + g].conj() * k[g];
            }
            for b in a..rows {
                let mut sum = Complex64::default();
                for g in 0..m {
                    sum += weighted[g] * f[b * m + g];
                }
                u.as_mut_slice()[a * rows + b] += sum;
            }
            let mut sum = Complex64::default();
            for g in 0..m {
                sum += weighted[g] * ion[g];
            }
            u_ion.as_mut_slice()[a] += sum;
        }
        Ok(())
    })?;
    for a in 0..rows {
        for b in a..rows {
            let value = u.as_slice()[a * rows + b] / volume;
            u.as_mut_slice()[a * rows + b] = value;
            u.as_mut_slice()[b * rows + a] = value.conj();
        }
        u_ion.as_mut_slice()[a] /= volume;
        let (atom, l, _) = sites.channels[a];
        u.as_mut_slice()[a * rows + a] +=
            shapes.tail(sites.atoms[atom].compensation_radius, l, cutoff)?;
    }
    Ok(Couplings {
        u,
        u_ion,
        e_ion: e_ion / volume,
    })
}
