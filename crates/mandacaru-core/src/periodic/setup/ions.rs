//! The Gaussian ions of a crystal: the point-ion constant that complements the
//! reciprocal energy, the on-site long-range term the reciprocal sum includes
//! and the dataset convention excludes, and the short-range ion potentials the
//! compensation charges feel (`linear`).
use super::shapes::{visit_lattice, Sites, SHAPE_POINTS};
use crate::{
    math::{angular, radial::integrate, spline::Cubic},
    paw::{
        local_split::{
            angular_rule, linspace, long_range_potential, radial_rule, short_range_radius,
            tail_radius,
        },
        multipole::shape_function,
    },
    Context, Result,
};
use nalgebra::Vector3;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Gaussian widths of the short-range sphere, the tail scan and its tolerance.
const RADIUS_SIGMAS: f64 = 6.5;
const TAIL_TOLERANCE: f64 = 1e-5;
const TAIL_SCAN_MAX: f64 = 8.;
const TAIL_SCAN_POINTS: usize = 7701;
/// Ion pairs are summed out to this many widths.
pub const ION_PAIR_REACH: f64 = 12.;
/// Radial, polar and azimuthal points of the compensation spheres.
pub const SPHERE_RADIAL: usize = 48;
pub const SPHERE_POLAR: usize = 24;
pub const SPHERE_AZIMUTHAL: usize = 48;

/// Point-ion energy minus the Gaussian ions' reciprocal energy:
/// `-sum Z^2 / (2 sqrt(pi) sigma) + 1/2 sum' Z_A Z_B erfc(d / 2 sigma) / d`
/// over pairs and lattice images closer than `12 sigma`.
pub fn ion_constants(sites: &Sites) -> Result<f64> {
    sites.check()?;
    let sigma = sites.sigma;
    let mut energy =
        -sites.atoms.iter().map(|a| a.charge * a.charge).sum::<f64>() / (2. * PI.sqrt() * sigma);
    let reach = ION_PAIR_REACH * sigma;
    for a in sites.atoms {
        for b in sites.atoms {
            let d0 = Vector3::from(b.center) - Vector3::from(a.center);
            visit_lattice(
                &sites.lattice,
                reach + d0.norm(),
                false,
                4096,
                &mut |images| {
                    for image in images {
                        let d = (d0 + Vector3::from(*image)).norm();
                        if d >= 1e-10 && d <= reach {
                            energy += 0.5 * a.charge * b.charge * libm::erfc(d / (2. * sigma)) / d;
                        }
                    }
                    Ok(())
                },
            )?;
        }
    }
    Ok(energy)
}

/// Per channel, `int g_{A,00} v^lr_A` for the monopoles (`sqrt(4 pi)` of the
/// radial integral) and zero for every other channel.
pub fn onsite_ion_compensation(sites: &Sites) -> Result<Vec<f64>> {
    sites.check()?;
    sites
        .channels
        .iter()
        .map(|&(atom, l, _)| {
            if l != 0 {
                return Ok(0.);
            }
            let a = &sites.atoms[atom];
            let r = linspace(0., a.compensation_radius, SHAPE_POINTS);
            let shape = shape_function(&r, a.compensation_radius, 0)?;
            let potential = long_range_potential(a.charge, sites.sigma, &r);
            let integrand: Vec<f64> = (0..r.len())
                .map(|i| shape[i] * potential[i] * r[i] * r[i])
                .collect();
            Ok((4. * PI).sqrt() * integrate(&r, &integrand)?)
        })
        .collect()
}

/// A species' ionic local potential: the dataset's cubic spline inside its
/// table, flat below it, `-Z / r` beyond it.
pub struct LocalPotential {
    spline: Cubic,
    pub charge: f64,
    /// Radius of the dataset's own local cutoff (`r_cut_local`).
    pub cutoff: f64,
}
impl LocalPotential {
    pub fn new(
        ctx: &Context,
        r: &[f64],
        v_local: &[f64],
        charge: f64,
        cutoff: f64,
    ) -> Result<Self> {
        Ok(Self {
            spline: Cubic::new(ctx, ctx.copy_slice(r)?, ctx.copy_slice(v_local)?)?,
            charge,
            cutoff,
        })
    }
    pub fn value(&self, radius: f64) -> f64 {
        let knots = self.spline.knots.as_slice();
        if radius <= knots[knots.len() - 1] {
            self.spline.value_slope(radius.max(knots[0])).0
        } else {
            -self.charge / radius.max(1e-12)
        }
    }
    /// `d value / d radius`: the spline's slope inside the table (zero below its first knot, where
    /// the potential is flat) and `Z / r^2` beyond it.
    pub fn slope(&self, radius: f64) -> f64 {
        let knots = self.spline.knots.as_slice();
        if radius > knots[knots.len() - 1] {
            self.charge / radius.max(1e-12).powi(2)
        } else if radius < knots[0] {
            0.
        } else {
            self.spline.value_slope(radius).1
        }
    }
    /// Radius of the sphere that holds all of `v^sr`: 6.5 widths, never less
    /// than the local cutoff or the measured tail.
    pub fn reach(&self, sigma: f64) -> Result<f64> {
        let scan = linspace(0.3, TAIL_SCAN_MAX, TAIL_SCAN_POINTS);
        let sampled: Vec<f64> = scan.iter().map(|r| self.value(*r)).collect();
        let tail = tail_radius(&scan, &sampled, self.charge, TAIL_TOLERANCE)?;
        Ok(short_range_radius(sigma, RADIUS_SIGMAS, self.cutoff, tail))
    }
}

/// Per channel, `int g_{A,LM} sum' v^sr_B` over every other atom and image: the
/// short-range half of the ions' potential on the compensation charges. One
/// potential per atom, in the atoms' order.
pub fn short_range_ion_compensation(
    ctx: &Context,
    sites: &Sites,
    potentials: &[LocalPotential],
) -> Result<Vec<Complex64>> {
    sites.check()?;
    if potentials.len() != sites.atoms.len() {
        return Err(super::super::invalid("one local potential per atom"));
    }
    let sigma = sites.sigma;
    let reaches = potentials
        .iter()
        .map(|p| p.reach(sigma))
        .collect::<Result<Vec<_>>>()?;
    let mut out = vec![Complex64::default(); sites.channels.len()];
    let (directions, angular_weights) = angular_rule(SPHERE_POLAR, SPHERE_AZIMUTHAL)?;
    let directions: Vec<[f64; 3]> = (0..angular_weights.len())
        .map(|j| [directions[0][j], directions[1][j], directions[2][j]])
        .collect();
    for owner in 0..sites.atoms.len() {
        let rows: Vec<usize> = (0..out.len())
            .filter(|&c| sites.channels[c].0 == owner)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let radius = sites.atoms[owner].compensation_radius;
        let (r, radial_weights) = radial_rule(radius, 0., SPHERE_RADIAL)?;
        let points = r.len() * directions.len();
        let _scratch = ctx.reserve(crate::product(&[points, 8, 4])?)?;
        let center = Vector3::from(sites.atoms[owner].center);
        let at = |p: usize| {
            center + Vector3::from(directions[p % directions.len()]) * r[p / directions.len()]
        };
        let mut potential = vec![0.; points];
        let mut distance = vec![0.; points];
        for (other, species) in potentials.iter().enumerate() {
            let reach = reaches[other] + radius;
            let d0 = center - Vector3::from(sites.atoms[other].center);
            visit_lattice(
                &sites.lattice,
                reach + d0.norm(),
                false,
                4096,
                &mut |images| {
                    for image in images {
                        let shift = Vector3::from(*image);
                        if other == owner && shift.norm() < 1e-10 {
                            continue;
                        }
                        if (d0 - shift).norm() > reach {
                            continue;
                        }
                        let origin = Vector3::from(sites.atoms[other].center) + shift;
                        for (p, d) in distance.iter_mut().enumerate() {
                            *d = (at(p) - origin).norm();
                        }
                        let long = long_range_potential(species.charge, sigma, &distance);
                        for ((total, &d), l) in potential.iter_mut().zip(&distance).zip(long) {
                            *total += species.value(d) - l;
                        }
                    }
                    Ok(())
                },
            )?;
        }
        for row in rows {
            let (_, l, m) = sites.channels[row];
            let shape = shape_function(&r, radius, l)?;
            let mut total = Complex64::default();
            for (p, &v) in potential.iter().enumerate() {
                let (i, j) = (p / directions.len(), p % directions.len());
                let weight = radial_weights[i] * angular_weights[j] * shape[i];
                total += angular::complex(l, m, directions[j]) * (weight * v);
            }
            out[row] = total;
        }
    }
    Ok(out)
}

/// `linear[ch] = short_range[ch] - onsite[ch]`, the ion term the engine adds.
pub fn linear(
    ctx: &Context,
    sites: &Sites,
    potentials: &[LocalPotential],
) -> Result<Vec<Complex64>> {
    let short = short_range_ion_compensation(ctx, sites, potentials)?;
    let onsite = onsite_ion_compensation(sites)?;
    Ok(short.iter().zip(onsite).map(|(s, o)| s - o).collect())
}
