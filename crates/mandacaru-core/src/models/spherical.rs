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

/// Variational radial LDA/PBE. The transpose derivative differentiates the same
/// discrete energy quadrature, including boundary nodes and unequal spacings.
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
    let derivative = |i: usize| -> (usize, usize, f64) {
        let a = i.saturating_sub(1);
        let b = (i + 1).min(n - 1);
        (a, b, 1. / (r[b] - r[a]))
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
        let (a, b, scale) = derivative(i);
        let gradient = if functional == "lda" {
            0.
        } else {
            (rho[b] - rho[a]) * scale
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
    for i in 0..n {
        let (a, b, scale) = derivative(i);
        let value = flux.as_slice()[i] * scale;
        potential.as_mut_slice()[a] -= value / weights.as_slice()[a];
        potential.as_mut_slice()[b] += value / weights.as_slice()[b];
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
}
