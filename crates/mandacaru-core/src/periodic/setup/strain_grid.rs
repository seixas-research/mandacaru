//! Strain derivatives of the grid arrays of the periodic setup: the Coulomb kernel, the Gaussian
//! ions, the compensation shapes of the channel moments, all on the grid's reciprocal set with each
//! frequency the mean over its lattice aliases, and the smooth core on the grid nodes.
//!
//! Convention (the stress plan's; see `periodic::stress`): the strain `eps` carries lattice
//! vectors, atoms, the grid origin and the grid nodes by `x -> (1 + eps) x`, so a wave vector moves
//! as `G -> (1 - eps^T) G`, `G.R` and `G.r0` are invariant and `d f(G) / d eps_ab = -G_a d_b f`.
//! What is pinned stays at its reference value: the node counts, the lattice operations of the
//! alias average, the Gaussian width, the shape tables and the integer triples of every cutoff
//! list. A derivative of an alias mean is the alias mean of the derivative at the SAME alias
//! vectors. The kernel and the ions are symmetric in `ab` and return the six Voigt rows (xx, yy,
//! zz, yz, xz, xy); the shapes are not, and return the nine rows `a * 3 + b`. Every result is
//! the array `d f / d eps_ab`; the contractions with the density, which carry the Jacobians
//! of `dv` and `Omega`, are the stress terms', not these.
use super::{
    electrostatics::{coulomb, ion_transform},
    shapes::{Shapes, Sites, Transient},
};
use crate::{
    math::{reciprocal, spline::Cubic, SpatialGrid},
    paw::fields::place_by,
    periodic::invalid,
    Buffer, Context, Result,
};
use num_complex::Complex64;
use std::ops::{Add, Mul, Sub};

/// The index pairs of the six Voigt rows.
pub const VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

/// `rows` functions of the wave vector on the grid's reciprocal set (`out[row * n + at]`), each
/// frequency the alias mean `f(box) + sum w (f(alias) - f(box))` of what `evaluate` returns.
/// `evaluate(vectors, out)` fills `out[row * vectors.len() + v]`; it holds at most
/// `per_vector` bytes per vector, which sizes the blocks the vectors come in.
fn mean_rows<T>(
    ctx: &Context,
    grid: &SpatialGrid,
    rows: usize,
    per_vector: usize,
    evaluate: &mut dyn FnMut(&[[f64; 3]], &mut [T]) -> Result<()>,
) -> Result<Buffer<T>>
where
    T: Default + Copy + Add<Output = T> + Sub<Output = T> + Mul<f64, Output = T>,
{
    let n = grid.index.points();
    let shape = grid.index.shape();
    let b = reciprocal::reciprocal_cell(grid)?;
    let mut out = ctx.allocate::<T>(crate::product(&[rows, n])?)?;
    let bytes = rows * size_of::<T>() + per_vector + 24 + 64;
    let block = (Transient::of(ctx).dense_bytes / bytes).max(1);
    let _scratch = ctx.reserve(crate::product(&[block.min(n), bytes])?)?;
    let mut values = vec![T::default(); rows * block.min(n)];
    for start in (0..n).step_by(block) {
        let vectors: Vec<[f64; 3]> = (start..(start + block).min(n))
            .map(|at| reciprocal::box_vector(&b, shape, at))
            .collect();
        let m = vectors.len();
        evaluate(&vectors, &mut values[..rows * m])?;
        for row in 0..rows {
            out.as_mut_slice()[row * n + start..row * n + start + m]
                .copy_from_slice(&values[row * m..(row + 1) * m]);
        }
    }
    let table = reciprocal::alias_table(ctx, grid)?;
    for range in table.blocks(block) {
        let m = range.len();
        if values.len() < rows * m {
            values.resize(rows * m, T::default());
        }
        evaluate(&table.vectors[range.clone()], &mut values[..rows * m])?;
        for row in 0..rows {
            table.combine(
                &mut out.as_mut_slice()[row * n..(row + 1) * n],
                range.clone(),
                &values[row * m..(row + 1) * m],
            );
        }
    }
    Ok(out)
}

/// `d K / d eps_ab` of [`super::electrostatics::kernel`], `(6, n)`: per alias
/// `2 K G_a G_b / G^2` (from `K = 4 pi / G^2`, `d G^2 = -2 G_a G_b`), zero at `G = 0` where the
/// kernel is dropped.
pub fn kernel_strain(ctx: &Context, grid: &SpatialGrid) -> Result<Buffer<f64>> {
    mean_rows(
        ctx,
        grid,
        6,
        0,
        &mut |vectors: &[[f64; 3]], out: &mut [f64]| {
            let m = vectors.len();
            for (v, g) in vectors.iter().enumerate() {
                let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                for (row, &(a, b)) in VOIGT.iter().enumerate() {
                    out[row * m + v] = if g2 > 0. {
                        2. * coulomb(*g) * g[a] * g[b] / g2
                    } else {
                        0.
                    };
                }
            }
            Ok(())
        },
    )
}

/// `d I / d eps_ab` of [`super::electrostatics::ion_charge`], `(6, n)`: the Gaussian envelope
/// `exp(-sigma^2 G^2 / 2)` at the pinned width gives `sigma^2 G_a G_b I(G)` and the atom phases
/// do not move.
pub fn ions_strain(ctx: &Context, grid: &SpatialGrid, sites: &Sites) -> Result<Buffer<Complex64>> {
    sites.check()?;
    let sigma2 = sites.sigma * sites.sigma;
    mean_rows(
        ctx,
        grid,
        6,
        0,
        &mut |vectors: &[[f64; 3]], out: &mut [Complex64]| {
            let m = vectors.len();
            for (v, g) in vectors.iter().enumerate() {
                let ion = ion_transform(sites, *g);
                for (row, &(a, b)) in VOIGT.iter().enumerate() {
                    out[row * m + v] = ion * (sigma2 * g[a] * g[b]);
                }
            }
            Ok(())
        },
    )
}

/// `d h / d eps_ab` of the compensation charge `h(G) = sum_c moments[c] hat_c(G)` of
/// [`super::electrostatics::hats`], `(9, n)` with row `a * 3 + b`: `-G_a sum_c q_c d_b hat_c`
/// per alias, the derivative `d_b` taken by [`Shapes::transform_gradients`] (the slope of the
/// shape table's interpolant, the harmonics' angular gradient, the atom phases fixed). The
/// shapes are not symmetric in `ab`, nor is this.
pub fn hats_strain(
    ctx: &Context,
    grid: &SpatialGrid,
    shapes: &Shapes,
    sites: &Sites,
    moments: &[Complex64],
) -> Result<Buffer<Complex64>> {
    sites.check()?;
    let rows = sites.channels.len();
    if moments.len() != rows {
        return Err(invalid("one moment per compensation channel"));
    }
    let mut gradients = Vec::new();
    mean_rows(
        ctx,
        grid,
        9,
        rows * 3 * 16,
        &mut |vectors: &[[f64; 3]], out: &mut [Complex64]| {
            let m = vectors.len();
            gradients.resize(rows * 3 * m, Complex64::default());
            shapes.transform_gradients(sites, vectors, &mut gradients, m)?;
            for (v, g) in vectors.iter().enumerate() {
                for b in 0..3 {
                    let slope: Complex64 = (0..rows)
                        .map(|c| moments[c] * gradients[(c * 3 + b) * m + v])
                        .sum();
                    for a in 0..3 {
                        out[(a * 3 + b) * m + v] = slope * -g[a];
                    }
                }
            }
            Ok(())
        },
    )
}

/// `d rho_core / d eps_ab` on the grid nodes for the smooth core `core(radius)` of atoms at
/// `centers` (all their images): `rho'(|d|) d_a d_b / |d|` with `d = r - R` the node-to-image
/// offset, from the analytic slope of the SAME spline the placement samples (zero at the
/// nucleus). Nodes and centers move together, so only the distance changes; symmetric in `ab`.
pub fn core_strain(
    ctx: &Context,
    grid: &SpatialGrid,
    radius: &[f64],
    core: &[f64],
    centers: &[[f64; 3]],
    axes: (usize, usize),
) -> Result<Buffer<f64>> {
    if axes.0 > 2 || axes.1 > 2 {
        return Err(invalid("a strain component is a pair of axes 0, 1 or 2"));
    }
    let profile = Cubic::new(ctx, ctx.copy_slice(radius)?, ctx.copy_slice(core)?)?;
    place_by(ctx, grid, &profile, centers, &|distance, offset| {
        if distance > 0. {
            profile.value_slope(distance).1 * offset[axes.0] * offset[axes.1] / distance
        } else {
            0.
        }
    })
}
