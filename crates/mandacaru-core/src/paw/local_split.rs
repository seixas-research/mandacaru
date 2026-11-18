//! Range separation of the local pseudopotential: the potential of a Gaussian ion
//! (long range, smooth, left on the grid) and the compact remainder integrated on
//! atom-centered spherical quadratures.
//!
//! The dataset's own local potential (a spline, evaluated by its owner) enters as
//! sampled values: the split is exact because the short-range part is the
//! *difference* of that table and the Gaussian ion's potential, never an asymptote.
use super::{invalid, pair_tables::angular_rule as unit_sphere_rule};
use crate::{
    math::{radial::gauss_legendre, reciprocal::lattice_rotations},
    Context, Result,
};
use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

/// How near (in direction and in operation, a norm on unit vectors and orthogonal matrices) an
/// image must come to be the same in [`invariant_angular_rule`]: above the square of any strain
/// the lattice group's slack admits (1e-4), below the 1e-2 spacing of the rules' directions.
pub const ORBIT_MATCH: f64 = 1e-3;
/// Offset (Bohr) added to the last radius that deviates from the ionic tail.
const TAIL_MARGIN: f64 = 0.01;

/// `-Z erf(r / (sqrt(2) sigma)) / r`, the potential of a Gaussian ion; the `r -> 0`
/// limit `-Z sqrt(2/pi) / sigma` is taken explicitly.
pub fn long_range_potential(charge: f64, sigma: f64, radius: &[f64]) -> Vec<f64> {
    let limit = -charge * (2. / PI).sqrt() / sigma;
    radius
        .iter()
        .map(|&r| {
            if r < 1e-12 * sigma {
                return limit;
            }
            let safe = r.max(1e-300);
            -charge * libm::erf(safe / (2f64.sqrt() * sigma)) / safe
        })
        .collect()
}

/// `d/dr` of [`long_range_potential`]; zero at the origin (the potential is even there).
pub fn long_range_potential_derivative(charge: f64, sigma: f64, radius: &[f64]) -> Vec<f64> {
    let a = 1. / (2f64.sqrt() * sigma);
    radius
        .iter()
        .map(|&r| {
            if r < 1e-12 * sigma {
                return 0.;
            }
            let safe = r.max(1e-300);
            -charge
                * (2. / PI.sqrt() * a * (-(a * safe).powi(2)).exp() / safe
                    - libm::erf(a * safe) / (safe * safe))
        })
        .collect()
}

fn check(sampled: &[f64], radius: &[f64]) -> Result<()> {
    if sampled.len() != radius.len() {
        return Err(invalid(
            "the sampled local potential and the radii differ in length",
        ));
    }
    Ok(())
}

/// `v_A - v^lr_A` from the dataset's sampled local potential.
pub fn short_range_potential(
    sampled: &[f64],
    charge: f64,
    sigma: f64,
    radius: &[f64],
) -> Result<Vec<f64>> {
    check(sampled, radius)?;
    let lr = long_range_potential(charge, sigma, radius);
    Ok(sampled.iter().zip(&lr).map(|(v, l)| v - l).collect())
}

/// `d/dr (v_A - v^lr_A)` from the sampled derivative of the dataset's local potential.
pub fn short_range_potential_derivative(
    sampled: &[f64],
    charge: f64,
    sigma: f64,
    radius: &[f64],
) -> Result<Vec<f64>> {
    check(sampled, radius)?;
    let lr = long_range_potential_derivative(charge, sigma, radius);
    Ok(sampled.iter().zip(&lr).map(|(v, l)| v - l).collect())
}

/// `np.linspace(start, stop, num)`.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num < 2 {
        return vec![start; num];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut out = (0..num)
        .map(|i| i as f64 * step + start)
        .collect::<Vec<_>>();
    out[num - 1] = stop;
    out
}

/// Radius (Bohr) beyond which the sampled local potential is the ionic tail `-Z/r`
/// to `tolerance`: just past the last scanned radius that deviates, or zero.
pub fn tail_radius(radius: &[f64], sampled: &[f64], charge: f64, tolerance: f64) -> Result<f64> {
    check(sampled, radius)?;
    Ok(radius
        .iter()
        .zip(sampled)
        .rposition(|(r, v)| (v + charge / r).abs() > tolerance)
        .map_or(0., |i| radius[i] + TAIL_MARGIN))
}

/// Radius of the sphere that holds all of `v^sr`: `sigmas` widths of the Gaussian, but
/// never less than the local cutoff or the measured tail.
pub fn short_range_radius(sigma: f64, sigmas: f64, cutoff: f64, tail: f64) -> f64 {
    (sigmas * sigma).max(cutoff).max(tail)
}

/// The grid's coarsest node spacing: the largest norm of a column of the
/// `3 x 3` lattice-step matrix (row major).
pub fn grid_spacing(step: &[f64; 9]) -> f64 {
    (0..3)
        .map(|i| {
            (0..3)
                .map(|row| step[3 * row + i] * step[3 * row + i])
                .sum::<f64>()
                .sqrt()
        })
        .fold(0., f64::max)
}

/// Bohr below which two panel edges are one: a corner this close to an edge costs the rule about
/// its jump times the distance cubed, far below any energy the engine resolves.
pub const MERGE_EDGES: f64 = 1e-3;

/// Gauss-Legendre nodes and `w r^2` weights on `[0, radius]`, split into two panels
/// at `panel` when that lies strictly inside, so the kink where the pseudized
/// polynomial meets the ionic tail falls on a panel boundary.
pub fn radial_rule(radius: f64, panel: f64, points: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    radial_rule_panels(radius, &[panel], points)
}

/// Gauss-Legendre nodes and `w r^2` weights on `[0, radius]`, split into one panel per gap of
/// `edges` (the points where the integrand has a corner), so that every corner falls on a panel
/// boundary. An edge outside `(0, radius)`, or within [`MERGE_EDGES`] of another edge or of an
/// end, is dropped.
pub fn radial_rule_panels(
    radius: f64,
    edges: &[f64],
    points: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let rule = gauss_legendre(points)?;
    let mut inside: Vec<f64> = edges
        .iter()
        .copied()
        .filter(|edge| 0. < *edge && *edge < radius)
        .collect();
    inside.sort_by(f64::total_cmp);
    let mut bounds = vec![0.];
    for edge in inside {
        if edge - bounds[bounds.len() - 1] > MERGE_EDGES && radius - edge > MERGE_EDGES {
            bounds.push(edge);
        }
    }
    bounds.push(radius);
    let (mut r, mut w) = (Vec::new(), Vec::new());
    for pair in bounds.windows(2) {
        let half = 0.5 * (pair[1] - pair[0]);
        for (x, weight) in &rule {
            r.push(half * (x + 1.) + pair[0]);
            w.push(half * weight);
        }
    }
    let weights = w.iter().zip(&r).map(|(w, r)| w * r * r).collect();
    Ok((r, weights))
}

/// Unit directions (`x`, `y`, `z` lists) and solid-angle weights of the product rule
/// (Gauss-Legendre in `cos(theta)`, azimuths at half steps).
pub fn angular_rule(polar: usize, azimuthal: usize) -> Result<([Vec<f64>; 3], Vec<f64>)> {
    let rule = unit_sphere_rule(polar, azimuthal)?;
    let mut directions = [Vec::new(), Vec::new(), Vec::new()];
    let mut weights = Vec::new();
    for (direction, weight) in rule {
        for (axis, value) in directions.iter_mut().zip(direction) {
            axis.push(value);
        }
        weights.push(weight);
    }
    Ok((directions, weights))
}

/// [`angular_rule`] made invariant under `operations`, a group of orthogonal matrices whose first
/// entry is the identity: the union of the rule rotated by one operation of each left coset of the
/// operations the rule already has (its stabilizer), every direction weighted `1 / cosets`. The
/// product rule keeps the axis of its poles and its azimuths, so for a cubic group it is
/// replicated 3 times, for a hexagonal group about its pole axis not at all (the rule is then
/// returned as it is). An integral of a smooth function over directions then does not depend on
/// the orientation of the crystal in the group's orbit, which a product rule's error does.
///
/// The operations of a strained lattice (polar factors, [`lattice_rotations`]) miss those of
/// its unstrained group by about the square of the strain. Directions and operations therefore
/// match within [`ORBIT_MATCH`], far below the spacing of the rule's directions, so the
/// stabilizer, the cosets and the number of directions do not change under a small strain and
/// the rule moves continuously with it.
pub fn invariant_angular_rule(
    polar: usize,
    azimuthal: usize,
    operations: &[Matrix3<f64>],
) -> Result<([Vec<f64>; 3], Vec<f64>)> {
    if operations.first() != Some(&Matrix3::identity()) {
        return Err(invalid(
            "an invariant rule needs the identity as its first operation",
        ));
    }
    if polar == 0 || azimuthal == 0 {
        return angular_rule(polar, azimuthal);
    }
    if let Some(frame) = super::lebedev::octahedral_frame(operations) {
        // Match the product rule's angular degree; for production short-range cubic spheres
        // (`polar >= 16`, originally 16x36 x 3 = 1728 pts) promote to degree 47 (770 pts) so
        // non-spherical neighbor tails stay invariant under shear-induced frame rotations.
        let min_degree = if polar >= 16 { 47 } else { 1 };
        let degree = (2 * polar - 1).min(azimuthal - 1).max(min_degree);
        if let Some(rule) = super::lebedev::octahedral_rule(degree, &frame) {
            return Ok(rule);
        }
    }
    let (base, weights) = angular_rule(polar, azimuthal)?;
    let points: Vec<Vector3<f64>> = (0..weights.len())
        .map(|j| Vector3::new(base[0][j], base[1][j], base[2][j]))
        .collect();
    let mut sorted: Vec<usize> = (0..points.len()).collect();
    sorted.sort_by(|&a, &b| points[a][0].total_cmp(&points[b][0]));
    let find = |x: &Vector3<f64>| -> Option<usize> {
        let from = sorted.partition_point(|&j| points[j][0] < x[0] - ORBIT_MATCH);
        sorted[from..]
            .iter()
            .take_while(|&&j| points[j][0] <= x[0] + ORBIT_MATCH)
            .copied()
            .find(|&j| (points[j] - x).norm() < ORBIT_MATCH)
    };
    let keeps = |g: &Matrix3<f64>| {
        points.iter().enumerate().all(|(j, p)| {
            find(&(g * p))
                .is_some_and(|image| (weights[image] - weights[j]).abs() <= 1e-12 * weights[j])
        })
    };
    let stabilizer: Vec<&Matrix3<f64>> = operations.iter().filter(|g| keeps(g)).collect();
    let mut covered = vec![false; operations.len()];
    let mut representatives = Vec::new();
    for (i, g) in operations.iter().enumerate() {
        if covered[i] {
            continue;
        }
        representatives.push(g);
        for s in &stabilizer {
            let product = g * *s;
            if let Some(j) = operations
                .iter()
                .position(|h| (h - product).norm() < ORBIT_MATCH)
            {
                covered[j] = true;
            }
        }
        covered[i] = true;
    }
    if representatives.len() == 1 {
        return Ok((base, weights));
    }
    let scale = 1. / representatives.len() as f64;
    let mut directions = [Vec::new(), Vec::new(), Vec::new()];
    let mut out = Vec::new();
    for g in &representatives {
        for (p, w) in points.iter().zip(&weights) {
            let q = *g * p;
            for (axis, value) in directions.iter_mut().zip(q.iter()) {
                axis.push(*value);
            }
            out.push(w * scale);
        }
    }
    Ok((directions, out))
}

/// [`invariant_angular_rule`] under the point group of the Bravais lattice with the columns of
/// `cell` as its vectors: the rule of every angular integral around the atoms of a crystal, which
/// then cannot tell a rotated crystal of the same lattice from the original.
pub fn lattice_angular_rule(
    ctx: &Context,
    cell: &Matrix3<f64>,
    polar: usize,
    azimuthal: usize,
) -> Result<([Vec<f64>; 3], Vec<f64>)> {
    invariant_angular_rule(polar, azimuthal, &lattice_rotations(ctx, cell)?)
}
