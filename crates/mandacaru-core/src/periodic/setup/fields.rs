//! The density-independent fields of a crystal on its grid: the superposed
//! smooth valence density and its magnetization, the smooth cores, the
//! reference monopoles, and the mixing and phase arrays of the reciprocal set.
use super::shapes::Sites;
use crate::{
    math::{reciprocal, spline::Cubic, SpatialGrid},
    paw::fields::{place, sample_many, RadialSource},
    periodic::invalid,
    Buffer, Context, Result,
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// The radial tables of one species the fields read.
pub struct Tables<'a> {
    pub radius: &'a [f64],
    pub valence_density: &'a [f64],
    /// The smooth core the dataset was unscreened with, if any.
    pub core: Option<&'a [f64]>,
}
/// One atom: its species (an index into the tables) and its center (Bohr).
#[derive(Clone, Copy, Debug)]
pub struct Placed {
    pub species: usize,
    pub center: [f64; 3],
}

/// The core table a dataset's record selects: its smooth core when the
/// unscreening applied it, none when it did not or the table is all zero; any
/// other source (a core re-pseudized at an unstored radius) raises.
pub fn core_table<'a>(applied: bool, source: &str, smooth: &'a [f64]) -> Result<Option<&'a [f64]>> {
    if !applied {
        return Ok(None);
    }
    if source != "smooth_core" {
        return Err(crate::Error::Unsupported(format!(
            "the dataset was unscreened with a core re-pseudized ({source:?}), which it does not store"
        )));
    }
    Ok(smooth.iter().any(|v| *v != 0.).then_some(smooth))
}

/// `|G|^2 / (|G|^2 + 1)` on the FFT box (`G = 0` is one): the Kerker weights.
pub fn kerker(ctx: &Context, grid: &SpatialGrid) -> Result<Buffer<f64>> {
    let b = reciprocal::reciprocal_cell(grid)?;
    let shape = grid.index.shape();
    let mut out = ctx.allocate::<f64>(grid.index.points())?;
    for (at, value) in out.as_mut_slice().iter_mut().enumerate() {
        let g = reciprocal::box_vector(&b, shape, at);
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        *value = g2 / (g2 + 1.);
    }
    out.as_mut_slice()[0] = 1.;
    Ok(out)
}

/// `exp(-i G.r0)` on the FFT box, `r0` the grid's first node.
pub fn phase(ctx: &Context, grid: &SpatialGrid) -> Result<Buffer<Complex64>> {
    let b = reciprocal::reciprocal_cell(grid)?;
    let shape = grid.index.shape();
    let mut out = ctx.allocate::<Complex64>(grid.index.points())?;
    for (at, value) in out.as_mut_slice().iter_mut().enumerate() {
        let g = reciprocal::box_vector(&b, shape, at);
        let angle = -(g[0] * grid.origin[0] + g[1] * grid.origin[1] + g[2] * grid.origin[2]);
        *value = Complex64::new(angle.cos(), angle.sin());
    }
    Ok(out)
}

/// `2 pi sigma^2 sum Z / Omega`: the potential shift of the Gaussian ions.
pub fn reference(sites: &Sites) -> f64 {
    2. * PI * sites.sigma * sites.sigma * sites.atoms.iter().map(|a| a.charge).sum::<f64>()
        / sites.volume()
}

/// A valence density counts out to the last radius where it exceeds this
/// fraction of its peak. The oracle's threshold (1e-12), kept so that the
/// superposed density reproduces its number; the shared placement alone would
/// reach to 1e-14 of the peak.
pub const SUPPORT_TOLERANCE: f64 = 1e-12;

/// The profile of a valence density: a not-a-knot spline of the table cut after
/// its last significant radius, so that the placement's own support is that radius.
fn profile(ctx: &Context, radius: &[f64], values: &[f64]) -> Result<Cubic> {
    if radius.len() != values.len() {
        return Err(invalid("a radial table needs one value per radius"));
    }
    let peak = values.iter().fold(0f64, |m, v| m.max(v.abs()));
    let last = values
        .iter()
        .rposition(|v| v.abs() > SUPPORT_TOLERANCE * peak)
        .unwrap_or(0)
        .max(3)
        .min(values.len() - 1);
    Cubic::new(
        ctx,
        ctx.copy_slice(&radius[..=last])?,
        ctx.copy_slice(&values[..=last])?,
    )
}

/// The smooth valence densities of the placed atoms summed over their images.
pub fn superposed_valence(
    ctx: &Context,
    grid: &SpatialGrid,
    tables: &[Tables],
    atoms: &[Placed],
    weights: &[f64],
) -> Result<Buffer<f64>> {
    if weights.len() != atoms.len() || atoms.iter().any(|a| a.species >= tables.len()) {
        return Err(invalid("one weight and a known species per placed atom"));
    }
    let mut total = ctx.allocate::<f64>(grid.index.points())?;
    for (species, table) in tables.iter().enumerate() {
        for (atom, &weight) in atoms.iter().zip(weights) {
            if atom.species != species || weight == 0. {
                continue;
            }
            let curve = profile(ctx, table.radius, table.valence_density)?;
            let field = place(ctx, grid, &curve, &[atom.center])?;
            for (to, from) in total.as_mut_slice().iter_mut().zip(field.as_slice()) {
                *to += weight * from;
            }
        }
    }
    Ok(total)
}

/// `(atoms, points)` row-major: each atom's share `n_A / sum_B n_B` of the superposed smooth
/// valence density, the Hirshfeld weights of the free atoms' densities; zero where the sum is not
/// positive. A lone atom's share is one wherever it is positive.
pub fn valence_shares(
    ctx: &Context,
    grid: &SpatialGrid,
    tables: &[Tables],
    atoms: &[Placed],
) -> Result<Buffer<f64>> {
    let n = grid.index.points();
    let mut shares = ctx.allocate::<f64>(crate::product(&[atoms.len(), n])?)?;
    for a in 0..atoms.len() {
        let mut weights = vec![0.; atoms.len()];
        weights[a] = 1.;
        let piece = superposed_valence(ctx, grid, tables, atoms, &weights)?;
        shares.as_mut_slice()[a * n..(a + 1) * n].copy_from_slice(piece.as_slice());
    }
    for i in 0..n {
        let total: f64 = (0..atoms.len()).map(|a| shares.as_slice()[a * n + i]).sum();
        for a in 0..atoms.len() {
            let share = &mut shares.as_mut_slice()[a * n + i];
            *share = if total > 0. { *share / total } else { 0. };
        }
    }
    Ok(shares)
}

/// Superposed smooth valence densities.
pub fn initial_density(
    ctx: &Context,
    grid: &SpatialGrid,
    tables: &[Tables],
    atoms: &[Placed],
) -> Result<Buffer<f64>> {
    superposed_valence(ctx, grid, tables, atoms, &vec![1.; atoms.len()])
}

/// The starting magnetization `n_up - n_down`: each atom's valence density times
/// its initial moment over its valence charge, the ratio clamped to `[-1, 1]` so
/// that neither spin density starts negative.
pub fn initial_magnetization(
    ctx: &Context,
    grid: &SpatialGrid,
    tables: &[Tables],
    atoms: &[Placed],
    charges: &[f64],
    moments: &[f64],
) -> Result<Buffer<f64>> {
    if charges.len() != atoms.len() || moments.len() != atoms.len() {
        return Err(invalid("one charge and one moment per placed atom"));
    }
    let fractions: Vec<f64> = charges
        .iter()
        .zip(moments)
        .map(|(&z, &m)| {
            if m == 0. || z <= 0. {
                0.
            } else {
                (m / z).clamp(-1., 1.)
            }
        })
        .collect();
    superposed_valence(ctx, grid, tables, atoms, &fractions)
}

/// The smooth cores (or, with `kinetic`, their von Weizsaecker kinetic
/// densities) on the grid, with their images; zero when no species has one.
pub fn core(
    ctx: &Context,
    grid: &SpatialGrid,
    tables: &[Tables],
    atoms: &[Placed],
    kinetic: bool,
) -> Result<Buffer<f64>> {
    let mut sources = Vec::new();
    for (species, table) in tables.iter().enumerate() {
        let Some(core) = table.core else { continue };
        let centers: Vec<[f64; 3]> = atoms
            .iter()
            .filter(|a| a.species == species)
            .map(|a| a.center)
            .collect();
        if centers.is_empty() {
            continue;
        }
        sources.push(RadialSource {
            knots: ctx.copy_slice(table.radius)?,
            values: ctx.copy_slice(core)?,
            centers: ctx.copy_slice(&centers)?,
        });
    }
    sample_many(ctx, grid, &sources, kinetic)
}

/// The reference monopoles: `compensation_charge / sqrt(4 pi)` on the monopole
/// of each atom, zero on every other channel.
pub fn initial_charges(sites: &Sites, compensation_charges: &[f64]) -> Result<Vec<Complex64>> {
    sites.check()?;
    if compensation_charges.len() != sites.atoms.len() {
        return Err(invalid("one compensation charge per atom"));
    }
    Ok(sites
        .channels
        .iter()
        .map(|&(atom, l, m)| {
            if l == 0 && m == 0 {
                Complex64::new(compensation_charges[atom] / (4. * PI).sqrt(), 0.)
            } else {
                Complex64::default()
            }
        })
        .collect())
}
