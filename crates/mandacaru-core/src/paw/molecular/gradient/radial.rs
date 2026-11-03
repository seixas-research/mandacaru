//! Radial slopes of the functions the isolated PAW integrals evaluate: the
//! compensation potential, the long-range ionic potential, the local ionic
//! potential and the partial core densities. Each slope is the derivative of
//! the exact function the setup evaluates, branch by branch.
use super::super::data::Species;
use crate::{math::spline::Cubic, paw::pair_tables::potential};
use std::f64::consts::PI;

/// `potential(r, radius, l)` and its derivative with respect to `r`. The
/// potential is the sum of an interior part `in` (the charge inside `r`) and an
/// exterior part `out`; the shape function's own terms cancel in the slope,
/// which is `(-(l + 1) in + l out) / r`. Below `1e-12 radius` the slope of the
/// limit at that radius is returned (a p potential has a finite slope at the
/// center, every other one has none).
pub fn potential_slope(r: f64, radius: f64, l: usize) -> (f64, f64) {
    let prefactor = 4. * PI / (2 * l + 1) as f64;
    if r >= radius {
        let value = prefactor / r.powi((l + 1) as i32);
        return (value, -((l + 1) as f64) * value / r);
    }
    let r_eff = r.max(1e-12 * radius);
    let x = r_eff / radius;
    let norm = 1.
        / (radius.powi((2 * l + 3) as i32)
            * (1. / (2 * l + 3) as f64 - 3. / (2 * l + 5) as f64 + 3. / (2 * l + 7) as f64
                - 1. / (2 * l + 9) as f64));
    let (mut enclosed, mut tail) = (0., 0.);
    for (k, c) in [1., -3., 3., -1.].into_iter().enumerate() {
        let power = (2 * l + 3 + 2 * k) as i32;
        enclosed += c * x.powi(power) / power as f64;
        tail += c * (1. - x.powi((2 * k + 2) as i32)) / (2 * k + 2) as f64;
    }
    let inner =
        prefactor * norm * radius.powi((2 * l + 3) as i32) * enclosed / r_eff.powi((l + 1) as i32);
    let outer = prefactor * norm * r_eff.powi(l as i32) * radius * radius * tail;
    (
        potential(r, radius, l),
        (-((l + 1) as f64) * inner + l as f64 * outer) / r_eff,
    )
}

/// Gradient with respect to `delta` of `potential(|delta|) Y_lm(delta)`, the
/// potential of a compensation multipole at the origin.
pub fn multipole_gradient(radius: f64, l: usize, m: i32, delta: [f64; 3]) -> [f64; 3] {
    let r = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
    let (value, slope) = potential_slope(r, radius, l);
    radial_gradient(value, slope, l, m, delta)
}
/// Gradient of `f(|delta|) Y_lm(delta)` for a radial function with the given
/// `value` and `slope` at `|delta|`.
pub fn radial_gradient(value: f64, slope: f64, l: usize, m: i32, delta: [f64; 3]) -> [f64; 3] {
    let r = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
    let harmonic = crate::math::angular::cartesian(l, m, delta);
    let angular = crate::math::angular::cartesian_gradient(l, m, delta);
    let unit = if r > 0. {
        delta.map(|v| v / r)
    } else {
        [0.; 3]
    };
    std::array::from_fn(|i| slope * unit[i] * harmonic + value * angular[i])
}

/// `long_range(charge, sigma, r)` of the setup and its derivative in `r`.
pub fn long_range_slope(charge: f64, sigma: f64, r: f64) -> f64 {
    if r < 1e-12 * sigma {
        return 0.;
    }
    let x = r / (2f64.sqrt() * sigma);
    let scale = 2. / PI.sqrt();
    // x e^{-x^2} - sqrt(pi) / 2 erf(x), the difference of two nearly equal
    // series, summed directly for a small argument.
    let difference = if x < 0.02 {
        let x2 = x * x;
        x * x2 * (-2. / 3. + x2 * (2. / 5. + x2 * (-1. / 7. + x2 / 27.)))
    } else {
        x * (-x * x).exp() - 0.5 * PI.sqrt() * libm::erf(x)
    };
    -charge * scale * difference / (r * r)
}

/// The slope of [`Species::local`] at `r`.
pub fn local_slope(species: &Species, r: f64) -> f64 {
    let knots = species.ionic.knots.as_slice();
    if r > knots[knots.len() - 1] {
        species.charge / (r * r)
    } else if r < knots[0] {
        0.
    } else {
        species.ionic.value_slope(r).1
    }
}

/// The partial-core table as the grid sums it, and its slope: zero past the
/// last node, constant below the first and wherever the value is clipped at 0.
pub fn core_slope(table: &Cubic, r: f64) -> (f64, f64) {
    let knots = table.knots.as_slice();
    if r > knots[knots.len() - 1] {
        return (0., 0.);
    }
    let (value, slope) = table.value_slope(r.max(knots[0]));
    if value <= 0. {
        (0., 0.)
    } else if r < knots[0] {
        (value, 0.)
    } else {
        (value, slope)
    }
}
