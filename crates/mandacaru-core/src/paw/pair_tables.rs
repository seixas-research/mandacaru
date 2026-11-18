//! The radial pair tables of a PAW species and the one-center Coulomb tensor
//! built from them, independent of any molecule or crystal.
//!
//! A species is described by a [`SpeciesView`]: its radial knots, the all-electron
//! and smooth partial waves of every channel and the reference density matrix.
//! The molecular and the periodic engines each fill a view from their own dataset
//! storage; the tables, the multipole shape functions and `Delta W` are shared.
use super::{
    core_exchange::FrozenCore,
    invalid,
    multipole::{angular_grid, shape_at, AZIMUTHAL_POINTS, POLAR_POINTS},
    onecenter_radial::{radial_coulomb, radial_coulomb_weighted},
};
use crate::{math::spline::simpson_weights, product, Result};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// One angular-momentum channel of a species: its partial waves on the species'
/// knots and the reference atom's density matrix over them (per `m`).
pub struct ChannelView<'a> {
    pub l: usize,
    pub radius: f64,
    pub ae: Vec<&'a [f64]>,
    pub ps: Vec<&'a [f64]>,
    pub reference: &'a DMatrix<f64>,
}

/// The radial tables of a species the one-center screened terms are built from.
pub struct SpeciesView<'a> {
    pub knots: &'a [f64],
    /// The all-electron frozen core density on `knots`, if this species has a frozen core.
    pub core_density: Option<&'a [f64]>,
    /// The smooth core the grid's exchange-correlation sees (the dataset's nonlinear core
    /// correction; zeros when the dataset was not unscreened with one), on `knots`. The
    /// screened hybrid's semilocal short-range exchange is evaluated on the smooth valence
    /// density plus this core on the grid, so the spheres take it off the smooth side.
    pub smooth_core: &'a [f64],
    /// The closed frozen-core subshells for screened core-valence exact exchange, if any.
    pub frozen_core: Option<&'a FrozenCore>,
    pub compensation_radius: f64,
    pub channels: Vec<ChannelView<'a>>,
}

/// One projector of the layout `(channel, m, radial)` of a species.
#[derive(Clone, Copy)]
pub struct Projector {
    pub channel: usize,
    pub radial: usize,
    pub l: usize,
    pub m: i32,
}

/// The projector layout of `species`: channels in order, then `m`, then the radial index.
pub fn layout(species: &SpeciesView<'_>) -> Vec<Projector> {
    let mut out = Vec::new();
    for (channel, ch) in species.channels.iter().enumerate() {
        for m in -(ch.l as i32)..=ch.l as i32 {
            for radial in 0..ch.ae.len() {
                out.push(Projector {
                    channel,
                    radial,
                    l: ch.l,
                    m,
                });
            }
        }
    }
    out
}

pub fn harmonic(l: usize, m: i32, point: [f64; 3]) -> f64 {
    crate::math::angular::cartesian(l, m, point)
}
pub fn angular_rule(polar: usize, azimuthal: usize) -> Result<Vec<([f64; 3], f64)>> {
    let (theta, phi, weights) = angular_grid(polar, azimuthal)?;
    Ok(theta
        .into_iter()
        .zip(phi)
        .zip(weights)
        .map(|((th, ph), w)| {
            let sin = th.sin();
            ([sin * ph.cos(), sin * ph.sin(), th.cos()], w)
        })
        .collect())
}
fn linear(x: &[f64], y: &[f64], at: f64) -> f64 {
    let i = x.partition_point(|v| *v <= at);
    if i == 0 {
        y[0]
    } else if i == x.len() {
        y[x.len() - 1]
    } else {
        y[i - 1] + (y[i] - y[i - 1]) * (at - x[i - 1]) / (x[i] - x[i - 1])
    }
}
pub fn coulomb(r: &[f64], a: &[f64], b: &[f64], l: usize) -> f64 {
    if r.len() < 3 {
        return 0.;
    }
    radial_coulomb(r, a, b, l).expect("valid radial Coulomb inputs")
}

/// Points of the radial quadrature of the one-center pair tables.
pub const POINTS: usize = 600;

/// The radial pair tables of a species, shared by the bare and the screened
/// one-center tensors: the Gaunt blocks of every multipole, the all-electron
/// pair densities and the smooth ones compensated, per level.
pub struct Tables {
    pub p: usize,
    pub lmax: usize,
    pub gaunts: Vec<(usize, i32, DMatrix<f64>)>,
    pub r: Vec<f64>,
    pub ae_pair: Vec<Vec<f64>>,
    pub compensated: Vec<Vec<Vec<f64>>>,
}
impl Tables {
    pub fn new(species: &SpeciesView<'_>, projectors: &[Projector]) -> Result<Self> {
        let p = projectors.len();
        let pp = product(&[p, p])?;
        let lmax = projectors.iter().map(|p| p.l).max().unwrap_or(0) * 2;
        let angular = angular_rule(POLAR_POINTS, AZIMUTHAL_POINTS)?;
        let mut gaunts = Vec::new();
        for l in 0..=lmax {
            for m in -(l as i32)..=l as i32 {
                let mut g = DMatrix::<f64>::zeros(p, p);
                for (direction, weight) in &angular {
                    let top = harmonic(l, m, *direction) * weight;
                    let values: Vec<_> = projectors
                        .iter()
                        .map(|a| harmonic(a.l, a.m, *direction))
                        .collect();
                    for a in 0..p {
                        for b in 0..p {
                            if (l + projectors[a].l + projectors[b].l) % 2 == 0
                                && l >= projectors[a].l.abs_diff(projectors[b].l)
                                && l <= projectors[a].l + projectors[b].l
                            {
                                g[(a, b)] += top * values[a] * values[b];
                            }
                        }
                    }
                }
                for value in g.iter_mut() {
                    if value.abs() < 1e-12 {
                        *value = 0.;
                    }
                }
                gaunts.push((l, m, g));
            }
        }
        let source = species.knots;
        let radius = species
            .channels
            .iter()
            .map(|ch| ch.radius)
            .fold(0., f64::max);
        let r: Vec<_> = (0..POINTS)
            .map(|i| radius * i as f64 / (POINTS - 1) as f64)
            .collect();
        let mut ae_pair = vec![vec![0.; POINTS]; pp];
        let mut ps_pair = vec![vec![0.; POINTS]; pp];
        for (a, pa) in projectors.iter().enumerate() {
            for (b, pb) in projectors.iter().enumerate() {
                let ca = &species.channels[pa.channel];
                let cb = &species.channels[pb.channel];
                for k in 0..POINTS {
                    ae_pair[a * p + b][k] = linear(source, ca.ae[pa.radial], r[k])
                        * linear(source, cb.ae[pb.radial], r[k]);
                    ps_pair[a * p + b][k] = linear(source, ca.ps[pa.radial], r[k])
                        * linear(source, cb.ps[pb.radial], r[k]);
                }
            }
        }
        let simpson_w = simpson_weights(&r)?;
        let mut compensated = Vec::new();
        for l in 0..=lmax {
            let mut pairs = Vec::new();
            let power = (l + 2) as i32;
            for ab in 0..pp {
                let moment: f64 = (0..POINTS)
                    .map(|k| simpson_w[k] * (ae_pair[ab][k] - ps_pair[ab][k]) * r[k].powi(power))
                    .sum();
                pairs.push(
                    (0..POINTS)
                        .map(|k| {
                            ps_pair[ab][k] + moment * shape_at(r[k], species.compensation_radius, l)
                        })
                        .collect::<Vec<_>>(),
                );
            }
            compensated.push(pairs);
        }
        Ok(Self {
            p,
            lmax,
            gaunts,
            r,
            ae_pair,
            compensated,
        })
    }
}

/// `Delta W[(ab), (cd)]`: the all-electron pair-pair Coulomb energy minus the
/// compensated smooth one, summed over multipole levels with the Gaunt weights;
/// under `erfc(omega R)/R` when `long_range` is given (the bare kernel's value
/// minus the `erf` part), under `1/R` otherwise.
pub fn pair_tensor(
    t: &Tables,
    long_range: Option<&super::screened_atom::LongRange>,
) -> Result<DMatrix<f64>> {
    let p = t.p;
    let pp = p * p;
    let r = &t.r;
    let simpson_w = simpson_weights(r)?;
    let mut w = DMatrix::zeros(pp, pp);
    for ab in 0..pp {
        for cd in ab..pp {
            let mut value = 0.;
            for (l, compensated) in t.compensated.iter().enumerate() {
                let angular: f64 = t
                    .gaunts
                    .iter()
                    .filter(|(level, _, _)| *level == l)
                    .map(|(_, _, g)| g[(ab / p, ab % p)] * g[(cd / p, cd % p)])
                    .sum::<f64>()
                    * 4.
                    * PI
                    / (2 * l + 1) as f64;
                if angular.abs() < 1e-14 {
                    continue;
                }
                let bare = 0.5
                    * (radial_coulomb_weighted(r, &t.ae_pair[ab], &t.ae_pair[cd], l, &simpson_w)?
                        + radial_coulomb_weighted(
                            r,
                            &t.ae_pair[cd],
                            &t.ae_pair[ab],
                            l,
                            &simpson_w,
                        )?);
                let smooth = 0.5
                    * (radial_coulomb_weighted(
                        r,
                        &compensated[ab],
                        &compensated[cd],
                        l,
                        &simpson_w,
                    )? + radial_coulomb_weighted(
                        r,
                        &compensated[cd],
                        &compensated[ab],
                        l,
                        &simpson_w,
                    )?);
                let mut difference = bare - smooth;
                if let Some(lr) = long_range {
                    let ae = super::screened_atom::LongRange::form(
                        [&t.ae_pair[ab], &t.ae_pair[cd]],
                        [&lr.ae[l][ab], &lr.ae[l][cd]],
                    );
                    let smooth = super::screened_atom::LongRange::form(
                        [&compensated[ab], &compensated[cd]],
                        [&lr.compensated[l][ab], &lr.compensated[l][cd]],
                    );
                    difference -= ae - smooth;
                }
                value += angular * difference;
            }
            w[(ab, cd)] = value;
            w[(cd, ab)] = value;
        }
    }
    Ok(w)
}

/// A species' channels need distinct angular momenta and as many projectors as waves.
pub fn check(species: &SpeciesView<'_>) -> Result<()> {
    if species.smooth_core.len() != species.knots.len()
        || species
            .core_density
            .is_some_and(|c| c.len() != species.knots.len())
    {
        return Err(invalid(
            "a PAW species' core density and radial knots disagree in size",
        ));
    }
    if species
        .channels
        .iter()
        .any(|c| c.ae.len() != c.ps.len() || c.reference.nrows() != c.ae.len())
    {
        return Err(invalid(
            "a PAW species' waves and reference density matrix disagree in size",
        ));
    }
    Ok(())
}
