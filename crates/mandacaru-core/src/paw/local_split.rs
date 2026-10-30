//! Range separation of the local pseudopotential: the potential of a Gaussian ion
//! (long range, smooth, left on the grid) and the compact remainder integrated on
//! atom-centered spherical quadratures.
//!
//! The dataset's own local potential (a spline, evaluated by its owner) enters as
//! sampled values: the split is exact because the short-range part is the
//! *difference* of that table and the Gaussian ion's potential, never an asymptote.
use super::{invalid, molecular::onecenter::angular_rule as unit_sphere_rule};
use crate::{math::radial::gauss_legendre, Result};
use std::f64::consts::PI;

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

/// Gauss-Legendre nodes and `w r^2` weights on `[0, radius]`, split into two panels
/// at `panel` when that lies strictly inside, so the kink where the pseudized
/// polynomial meets the ionic tail falls on a panel boundary.
pub fn radial_rule(radius: f64, panel: f64, points: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let rule = gauss_legendre(points)?;
    let edges = if 0. < panel && panel < radius {
        vec![0., panel, radius]
    } else {
        vec![0., radius]
    };
    let (mut r, mut w) = (Vec::new(), Vec::new());
    for pair in edges.windows(2) {
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
