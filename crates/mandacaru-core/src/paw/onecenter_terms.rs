//! The one-body pieces of the one-center terms, per channel: the reference atom's
//! frozen short-range semilocal exchange (screened hybrids) and the frozen
//! exchange-correlation, core exchange and reference density matrix that turn a
//! dataset's Kohn-Sham one-center terms into Hartree-Fock ones.
//!
//! The exchange-correlation potentials here use the atom solver's radial scheme
//! (`models::radial_xc`: derivatives on subset nodes, splined back), the scheme
//! the datasets were unscreened with, not the variational one of
//! `models::spherical`.
use super::{
    checks::interpolate_linear,
    invalid,
    onecenter::{Basis, Channel},
    onecenter_radial::{gradient_first_edge, radial_coulomb, three_j_squared, trapezoid},
};
use crate::{
    models::{radial_xc::xc_potential, xc::exchange_hole_batch},
    Context, Result,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// A channel with the electrons of the reference atom in each partial wave and
/// the radius beyond which its smooth waves are the all-electron ones.
#[derive(Clone, Copy)]
pub struct Occupied<'a> {
    pub channel: Channel<'a>,
    pub occupations: &'a [f64],
    pub projector_radius: f64,
}

/// One block per channel, in the projector basis (the same block for every `m`).
pub struct Block {
    pub l: usize,
    pub matrix: DMatrix<f64>,
}

fn sorted<'a>(channels: &[Occupied<'a>], grid: usize) -> Result<Vec<Occupied<'a>>> {
    let mut out = channels.to_vec();
    out.sort_by_key(|c| c.channel.l);
    for c in &out {
        c.channel.check(grid)?;
    }
    Ok(out)
}

fn symmetrize(m: &DMatrix<f64>) -> DMatrix<f64> {
    (m + m.transpose()) * 0.5
}

/// `searchsorted(r, radius) + 1`, capped at the grid: the nodes of a sphere.
fn nodes_inside(r: &[f64], radius: f64) -> usize {
    (r.partition_point(|x| *x < radius) + 1).min(r.len())
}

/// `(X0 - V.D0, {l: V_l})`: the reference atom's one-center short-range semilocal
/// exchange, linearized, in the hole model with screening `omega`
/// (unpolarized valence densities). `X0 = E[n1] - E[n1~]`, `V_ij` the matrix
/// elements of the exchange potentials of the all-electron and smooth densities.
pub fn frozen_exchange_terms(
    ctx: &Context,
    r: &[f64],
    channels: &[Occupied<'_>],
    omega: f64,
    basis: Basis,
) -> Result<(f64, Vec<Block>)> {
    let grid = r.len();
    let channels = sorted(channels, grid)?;
    let shell = r.iter().map(|r| 4. * PI * r * r).collect::<Vec<_>>();
    let mut energy = [0.; 2];
    let mut potentials = Vec::new();
    for side in 0..2 {
        let mut n = vec![0.; grid];
        for c in &channels {
            for (i, occupation) in c.occupations.iter().enumerate().take(c.channel.waves) {
                let wave = if side == 0 {
                    c.channel.ae_wave(i, grid)
                } else {
                    c.channel.ps_wave(i, grid)
                };
                for (n, w) in n.iter_mut().zip(wave) {
                    *n += occupation * w * w / (4. * PI);
                }
            }
        }
        let slope = gradient_first_edge(r, &n)?;
        let values = n
            .iter()
            .zip(&slope)
            .flat_map(|(n, s)| [*n, s * s])
            .collect::<Vec<_>>();
        let hole = exchange_hole_batch(ctx, &values, omega)?;
        let hole = hole.as_slice();
        let integrand = (0..grid)
            .map(|i| hole[3 * i] * shell[i])
            .collect::<Vec<_>>();
        energy[side] = trapezoid(r, &integrand);
        potentials.push((
            (0..grid).map(|i| hole[3 * i + 1]).collect::<Vec<_>>(),
            (0..grid)
                .map(|i| 2. * hole[3 * i + 2] * slope[i])
                .collect::<Vec<_>>(),
        ));
    }
    let inside = nodes_inside(
        r,
        channels
            .iter()
            .map(|c| c.projector_radius)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let mut blocks = Vec::new();
    let mut linear = 0.;
    for c in &channels {
        let size = c.channel.waves;
        let mut v = DMatrix::<f64>::zeros(size, size);
        for (side, sign) in [(0, 1.), (1, -1.)] {
            let (local, gradient) = &potentials[side];
            for i in 0..size {
                for j in 0..size {
                    let (wi, wj) = if side == 0 {
                        (c.channel.ae_wave(i, grid), c.channel.ae_wave(j, grid))
                    } else {
                        (c.channel.ps_wave(i, grid), c.channel.ps_wave(j, grid))
                    };
                    let product = (0..grid).map(|p| wi[p] * wj[p]).collect::<Vec<_>>();
                    let slope = gradient_first_edge(r, &product)?;
                    let integrand = (0..inside)
                        .map(|p| (local[p] * product[p] + gradient[p] * slope[p]) * r[p] * r[p])
                        .collect::<Vec<_>>();
                    v[(i, j)] += sign * trapezoid(&r[..inside], &integrand);
                }
            }
        }
        let v = symmetrize(&v);
        linear += c
            .occupations
            .iter()
            .enumerate()
            .take(size)
            .map(|(i, o)| o * v[(i, i)])
            .sum::<f64>();
        let matrix = match c.channel.transform(basis)? {
            None => v,
            Some(t) => &t * v * t.transpose(),
        };
        blocks.push(Block {
            l: c.channel.l,
            matrix,
        });
    }
    Ok((energy[0] - energy[1] - linear, blocks))
}

/// A frozen-core subshell: its angular momentum, electrons and radial function `R(r)`.
#[derive(Clone)]
pub struct CoreOrbital {
    pub l: usize,
    pub occupation: f64,
    pub radial: Vec<f64>,
}

/// The core orbitals' radial functions `u/r` on `r` (`u` on `atom_grid`, interpolated
/// unless that grid is `r` or `None`), and the electrons by which their density differs from `stored`.
pub fn core_orbital_tables(
    r: &[f64],
    atom_grid: Option<&[f64]>,
    subshells: &[(usize, f64, Vec<f64>)],
    stored: &[f64],
) -> Result<(Vec<Vec<f64>>, f64)> {
    let mut density = vec![0.; r.len()];
    let mut out = Vec::new();
    for (l, occupation, u) in subshells {
        let u = match atom_grid {
            // The atom's own grid is the dataset's when `np.allclose` says so.
            Some(grid)
                if grid.len() == r.len()
                    && grid
                        .iter()
                        .zip(r)
                        .all(|(a, b)| (a - b).abs() <= 1e-8 + 1e-5 * b.abs()) =>
            {
                u.clone()
            }
            Some(grid) => interpolate_linear(r, grid, u)?,
            None => u.clone(),
        };
        if u.len() != r.len() || stored.len() != r.len() {
            return Err(invalid("core orbital arrays do not match the radial grid"));
        }
        let mut radial = r
            .iter()
            .zip(&u)
            .map(|(r, u)| if *r > 0. { u / r } else { 0. })
            .collect::<Vec<_>>();
        if r[0] <= 0. && r.len() > 1 {
            radial[0] = if *l == 0 { radial[1] } else { 0. };
        }
        for (d, x) in density.iter_mut().zip(&radial) {
            *d += occupation * x * x / (4. * PI);
        }
        out.push(radial);
    }
    let error = if out.is_empty() {
        0.
    } else {
        let integrand = (0..r.len())
            .map(|i| (density[i] - stored[i]).abs() * 4. * PI * r[i] * r[i])
            .collect::<Vec<_>>();
        trapezoid(r, &integrand)
    };
    Ok((out, error))
}

/// Grid points that hold every core orbital to `tail` of its norm.
pub fn core_extent(r: &[f64], orbitals: &[CoreOrbital], tail: f64) -> usize {
    if orbitals.is_empty() {
        return 0;
    }
    let mut last = 0;
    for orbital in orbitals {
        let weight = (0..r.len())
            .map(|i| orbital.radial[i] * orbital.radial[i] * r[i] * r[i])
            .collect::<Vec<_>>();
        let mut cumulative = vec![0.; r.len()];
        for i in 1..r.len() {
            cumulative[i] =
                cumulative[i - 1] + 0.5 * (r[i] - r[i - 1]) * (weight[i - 1] + weight[i]);
        }
        let total = cumulative[r.len() - 1];
        let reached = cumulative
            .iter()
            .position(|c| *c >= (1. - tail) * total)
            .unwrap_or(r.len() - 1);
        last = last.max(reached);
    }
    r.len().min(last + 2)
}

/// `(l, V, K, D0)` of one channel.
pub type ChannelTerms = (usize, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>);

/// What the reference atom's one-center terms are built from.
pub struct ReferenceAtom<'a> {
    pub r: &'a [f64],
    pub channels: Vec<Occupied<'a>>,
    pub core_density: &'a [f64],
    pub valence_density: &'a [f64],
    /// The smooth core of the nonlinear core correction (zeros without one).
    pub smooth_core: &'a [f64],
    pub functional: &'a str,
    pub relativistic: bool,
    pub core: &'a [CoreOrbital],
}

/// Per channel `(V, K, D0)`: the dataset's one-center exchange-correlation linearized
/// at the reference atom, the frozen core's exchange (one spin) and the reference
/// atom's projected density matrix per `m`, in the projector `basis`.
pub fn exact_one_center_terms(
    ctx: &Context,
    atom: &ReferenceAtom<'_>,
    basis: Basis,
) -> Result<Vec<ChannelTerms>> {
    let r = atom.r;
    let grid = r.len();
    let channels = sorted(&atom.channels, grid)?;
    if atom.core_density.len() != grid
        || atom.valence_density.len() != grid
        || atom.smooth_core.len() != grid
        || atom.core.iter().any(|c| c.radial.len() != grid)
    {
        return Err(invalid("density arrays do not match the radial grid"));
    }
    let mut all_electron = atom.core_density.to_vec();
    for c in &channels {
        for (i, occupation) in c.occupations.iter().enumerate().take(c.channel.waves) {
            for (n, w) in all_electron.iter_mut().zip(c.channel.ae_wave(i, grid)) {
                *n += occupation * w * w / (4. * PI);
            }
        }
    }
    let smooth = (0..grid)
        .map(|i| atom.valence_density[i] + atom.smooth_core[i])
        .collect::<Vec<_>>();
    let v_ae = xc_potential(ctx, r, &all_electron, atom.functional, atom.relativistic)?.1;
    let v_ps = xc_potential(ctx, r, &smooth, atom.functional, atom.relativistic)?.1;
    let extent = core_extent(r, atom.core, 1e-14);
    let sphere = channels
        .iter()
        .map(|c| c.projector_radius)
        .fold(f64::NEG_INFINITY, f64::max);
    // The difference is the sphere's: outside it the smooth waves are the all-electron
    // ones, and a scattering reference below zero grows there.
    let inside = nodes_inside(r, sphere);
    let mut out = Vec::new();
    for c in &channels {
        let (size, l) = (c.channel.waves, c.channel.l);
        let mut v = DMatrix::<f64>::zeros(size, size);
        let mut k = DMatrix::<f64>::zeros(size, size);
        for i in 0..size {
            for j in 0..size {
                let (ae_i, ae_j) = (c.channel.ae_wave(i, grid), c.channel.ae_wave(j, grid));
                let (ps_i, ps_j) = (c.channel.ps_wave(i, grid), c.channel.ps_wave(j, grid));
                let integrand = (0..inside)
                    .map(|p| {
                        (v_ae[p] * ae_i[p] * ae_j[p] - v_ps[p] * ps_i[p] * ps_j[p]) * r[p] * r[p]
                    })
                    .collect::<Vec<_>>();
                v[(i, j)] = trapezoid(&r[..inside], &integrand);
                let mut total = 0.;
                for orbital in atom.core {
                    let lc = orbital.l;
                    let per_spin = orbital.occupation / (2. * (2 * lc + 1) as f64);
                    for big_l in l.abs_diff(lc)..=l + lc {
                        let weight = three_j_squared(lc, big_l, l)?;
                        if weight == 0. {
                            continue;
                        }
                        let rc = &orbital.radial[..extent];
                        let left = (0..extent).map(|p| ae_i[p] * rc[p]).collect::<Vec<_>>();
                        let right = (0..extent).map(|p| rc[p] * ae_j[p]).collect::<Vec<_>>();
                        total += per_spin
                            * (2 * lc + 1) as f64
                            * weight
                            * radial_coulomb(&r[..extent], &left, &right, big_l)?;
                    }
                }
                k[(i, j)] = -total;
            }
        }
        let mut d0 = DMatrix::<f64>::zeros(size, size);
        for (i, occupation) in c.occupations.iter().enumerate().take(size) {
            d0[(i, i)] = occupation / (2 * l + 1) as f64;
        }
        let (mut v, mut k) = (symmetrize(&v), symmetrize(&k));
        if let Some(t) = c.channel.transform(basis)? {
            let inverse = t
                .clone()
                .try_inverse()
                .ok_or_else(|| invalid("the projector basis transform is singular"))?;
            v = &t * v * t.transpose();
            k = &t * k * t.transpose();
            d0 = inverse.transpose() * d0 * &inverse;
        }
        out.push((l, v, k, d0));
    }
    Ok(out)
}
