//! Spherical electrostatic and exchange-correlation models shared by atoms and PAW.
use super::{invalid, xc};
use crate::{math::radial::validate, Buffer, Context, Result};
use std::f64::consts::PI;

pub fn hartree(ctx: &Context, r: &[f64], density: &[f64]) -> Result<Buffer<f64>> {
    validate(r, density)?;
    let mut out = ctx.allocate::<f64>(r.len())?;
    let outer_total = r
        .windows(2)
        .zip(density.windows(2))
        .map(|(r, d)| 0.5 * (r[1] - r[0]) * (r[0] * d[0] + r[1] * d[1]))
        .sum::<f64>();
    let (mut inner, mut outer) = (0., 0.);
    for i in 0..r.len() {
        if i > 0 {
            let h = r[i] - r[i - 1];
            inner += 0.5 * h * (r[i - 1].powi(2) * density[i - 1] + r[i].powi(2) * density[i]);
            outer += 0.5 * h * (r[i - 1] * density[i - 1] + r[i] * density[i]);
        }
        out.as_mut_slice()[i] =
            4. * PI * (if r[i] == 0. { 0. } else { inner / r[i] } + outer_total - outer);
    }
    Ok(out)
}

pub struct RadialXc {
    pub energy: f64,
    pub per_particle: Buffer<f64>,
    pub potential: Buffer<f64>,
}

/// Smallest half-width of the radial gradient stencil, as a fraction of the radius.
///
/// A gradient-corrected potential is the transpose of the gradient that enters the energy, so
/// it carries a second derivative of the density. A nearest-neighbour difference on the fine
/// uniform grid of a heavy reference atom (1500 Z points to 30 Bohr: 3.4e-4 Bohr for cerium)
/// lets the self-consistent loop lock into a five-point density ripple: PBE cerium "converged"
/// with `v_xc` swinging by +-6 Ha at 4.7 Bohr on a density smooth to 2e-5, and a total energy
/// 111 Ha above the LDA one where it is 6 Ha below on half the points; the lanthanide reference
/// atoms did not converge at all. Physics does not vary on a scale below about one percent of
/// the radius (the legacy atom scheme reached the same value for the same reason), so the
/// gradient is the neighbour difference averaged with a biweight kernel `(1 - x^2)^2` of
/// half-width `min(GRADIENT_HALF_WIDTH r, GRADIENT_SMOOTHING_LENGTH)`: the neighbour difference
/// itself where the grid is that coarse, and blind to shorter wavelengths, which therefore no
/// longer feed back. (A difference across interpolated points `r +- s` was tried first: its
/// transpose deposits on points that beat against the grid and left a one-percent moire in the
/// potential, roughness 7.8e4 on cerium; a triangular kernel still leaked 650 through its
/// sidelobes.)
pub const GRADIENT_HALF_WIDTH: f64 = 0.01;
/// Largest half-width of the smoothing (Bohr). The ripple is a grid-scale mode (cerium is
/// stable at a spacing of 7.5e-4 Bohr and locks in at 3.4e-4), so a width of a few fine-grid
/// steps removes it; a wider kernel only costs accuracy where the density is steep (one percent
/// of the radius missed oxygen's continuum potential by 4.6e-3 Ha in [0.2, 1) Bohr, five times
/// the legacy scheme). Every grid coarser than this (all Z <= 20 generation grids, the stored
/// tables of a dataset) keeps the neighbour difference exactly.
pub const GRADIENT_SMOOTHING_LENGTH: f64 = 1e-3;

/// The neighbour difference of node `j`: `d_j = (rho_b - rho_a) * scale`.
fn neighbours(r: &[f64], j: usize) -> (usize, usize, f64) {
    let a = j.saturating_sub(1);
    let b = (j + 1).min(r.len() - 1);
    (a, b, 1. / (r[b] - r[a]))
}

/// The smoothing window of node `i`: `None` where the grid spacing already exceeds
/// `min(GRADIENT_HALF_WIDTH r_i, GRADIENT_SMOOTHING_LENGTH)` (the gradient is the neighbour
/// difference itself), else the first
/// node and the normalized biweight weights `(1 - ((r_j - r_i) / s)^2)^2` over `|r_j - r_i| < s`.
fn window(r: &[f64], i: usize) -> Option<(usize, Vec<f64>)> {
    let (a, b, _) = neighbours(r, i);
    let width = (GRADIENT_HALF_WIDTH * r[i]).min(GRADIENT_SMOOTHING_LENGTH);
    if 2. * width <= r[b] - r[a] {
        return None;
    }
    let start = r.partition_point(|v| *v <= r[i] - width);
    let end = r.partition_point(|v| *v < r[i] + width);
    let mut weights = (start..end)
        .map(|j| (1. - ((r[j] - r[i]) / width).powi(2)).powi(2))
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    for w in weights.iter_mut() {
        *w /= total;
    }
    Some((start, weights))
}

/// Variational radial LDA/PBE. The transpose derivative differentiates the same
/// discrete energy quadrature, including boundary nodes and unequal spacings; the gradient is
/// taken across at least `GRADIENT_HALF_WIDTH r` on either side.
pub fn exchange_correlation(
    ctx: &Context,
    r: &[f64],
    rho: &[f64],
    functional: &str,
    relativistic: bool,
) -> Result<RadialXc> {
    validate(r, rho)?;
    if !["lda", "pbe"].contains(&functional) || r[0] <= 0. || rho.iter().any(|v| *v < 0.) {
        return Err(invalid(
            "radial XC needs positive radii, nonnegative density and lda or pbe",
        ));
    }
    let n = r.len();
    let mut weights = ctx.allocate::<f64>(n)?;
    let mut flux = ctx.allocate::<f64>(n)?;
    let mut potential = ctx.allocate::<f64>(n)?;
    let mut per_particle = ctx.allocate::<f64>(n)?;
    let gga = functional != "lda";
    let windows = if gga {
        (0..n).map(|i| window(r, i)).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let differences = if gga {
        (0..n)
            .map(|j| {
                let (a, b, scale) = neighbours(r, j);
                (rho[b] - rho[a]) * scale
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut energy = 0.;
    for i in 0..n {
        let width = if i == 0 {
            0.5 * (r[1] - r[0])
        } else if i == n - 1 {
            0.5 * (r[n - 1] - r[n - 2])
        } else {
            0.5 * (r[i + 1] - r[i - 1])
        };
        let weight = 4. * PI * r[i] * r[i] * width;
        weights.as_mut_slice()[i] = weight;
        let gradient = match windows.get(i) {
            None => 0.,
            Some(None) => differences[i],
            Some(Some((start, weights))) => weights
                .iter()
                .enumerate()
                .map(|(k, w)| w * differences[start + k])
                .sum(),
        };
        let sigma = gradient * gradient / 4.;
        let density = rho[i] / 2.;
        let (f, df) = xc::point_partials(
            functional,
            [density, density, sigma, sigma, sigma, 0., 0.],
            relativistic,
        )?;
        energy += weight * f;
        per_particle.as_mut_slice()[i] = if rho[i] > 0. { f / rho[i] } else { 0. };
        potential.as_mut_slice()[i] = 0.5 * (df[0] + df[1]);
        flux.as_mut_slice()[i] = 0.5 * weight * gradient * (df[2] + df[3] + df[4]);
    }
    if gga {
        // The transpose of the smoothing, then of the neighbour difference.
        let mut spread = vec![0.; n];
        for (i, window) in windows.iter().enumerate() {
            let value = flux.as_slice()[i];
            match window {
                None => spread[i] += value,
                Some((start, weights)) => {
                    for (k, w) in weights.iter().enumerate() {
                        spread[start + k] += w * value;
                    }
                }
            }
        }
        for (j, value) in spread.iter().enumerate() {
            let (a, b, scale) = neighbours(r, j);
            let value = value * scale;
            potential.as_mut_slice()[a] -= value / weights.as_slice()[a];
            potential.as_mut_slice()[b] += value / weights.as_slice()[b];
        }
    }
    Ok(RadialXc {
        energy,
        per_particle,
        potential,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn radial_pbe_potential_differentiates_discrete_energy() {
        let ctx = Context::new(Some(1), 1 << 20).unwrap();
        let r = (1..201).map(|i| i as f64 * 0.025).collect::<Vec<_>>();
        let rho = r.iter().map(|r| (-2. * r).exp() / PI).collect::<Vec<_>>();
        let result = exchange_correlation(&ctx, &r, &rho, "pbe", false).unwrap();
        for index in [0, 1, 70, 199] {
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            let step = 1e-5 * rho[index];
            plus[index] += step;
            minus[index] -= step;
            let ep = exchange_correlation(&ctx, &r, &plus, "pbe", false)
                .unwrap()
                .energy;
            let em = exchange_correlation(&ctx, &r, &minus, "pbe", false)
                .unwrap()
                .energy;
            let weight = 4.
                * PI
                * r[index]
                * r[index]
                * 0.025
                * if index == 0 || index == 199 { 0.5 } else { 1. };
            assert!(
                ((ep - em) / (2. * step) - weight * result.potential.as_slice()[index]).abs()
                    < 2e-7
            );
        }
    }
    /// The smoothed branch (a grid finer than `GRADIENT_SMOOTHING_LENGTH`): the potential is
    /// still the exact derivative of the discrete energy, inside the window, at the edge where
    /// the window switches on, and at the clipped last node.
    #[test]
    fn the_smoothed_gradient_potential_differentiates_discrete_energy() {
        let ctx = Context::new(Some(1), 1 << 24).unwrap();
        let h = 2e-4;
        let r = (1..10001).map(|i| i as f64 * h).collect::<Vec<_>>();
        let rho = r
            .iter()
            .map(|r| (-2. * r).exp() / PI + 0.05 * (-0.5 * (r - 1.2f64).powi(2) / 0.04).exp())
            .collect::<Vec<_>>();
        assert!(window(&r, 5000).is_some() && window(&r, 50).is_none());
        let result = exchange_correlation(&ctx, &r, &rho, "pbe", false).unwrap();
        let n = r.len();
        for index in [99, 100, 101, 400, 3000, 6000, n - 2, n - 1] {
            let energy = |step: f64| {
                let mut d = rho.clone();
                d[index] += step;
                exchange_correlation(&ctx, &r, &d, "pbe", false)
                    .unwrap()
                    .energy
            };
            let step = 1e-5 * rho[index];
            let central = |s: f64| (energy(s) - energy(-s)) / (2. * s);
            let fd = (4. * central(step) - central(2. * step)) / 3.;
            let width = if index == n - 1 { 0.5 * h } else { h };
            let weight = 4. * PI * r[index] * r[index] * width;
            let got = result.potential.as_slice()[index];
            // Near the switch-on radius (r = 0.02 Bohr) a node weighs 1e-6 and the difference of
            // the energy cannot resolve better than about 1e-5 (node 99 at steps 1e-4 / 1e-5 /
            // 1e-6: 2.5e-5 / 2.2e-6 / 1.1e-4, round-off taking over; node 101: 9.9e-6). From
            // r = 0.08 on the measured miss is 2.4e-7 to 3.6e-9.
            let gate = if r[index] < 0.05 { 3e-5 } else { 1e-6 };
            assert!(
                (fd / weight - got).abs() < gate * got.abs().max(1.),
                "node {index}: {} against {got}",
                fd / weight
            );
        }
    }
}
