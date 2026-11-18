//! Radial slopes of the functions the isolated PAW integrals evaluate: the
//! compensation potential, the long-range ionic potential, the local ionic
//! potential and the partial core densities. Each slope is the derivative of
//! the exact function the setup evaluates, branch by branch.
use super::super::data::Species;
use crate::{math::spline::Cubic, paw::multipole::shape_potential_parts};
use std::f64::consts::PI;

/// `shape_potential_at(r, radius, l)` and its derivative with respect to `r`. The
/// potential is the sum of an interior part `in` (the charge inside `r`) and an
/// exterior part `out`; the shape function's own terms cancel in the slope,
/// which is `(-(l + 1) in + l out) / r`. Below `1e-12 radius` the slope of the
/// limit at that radius is returned (a p potential has a finite slope at the
/// center, every other one has none).
pub fn potential_slope(r: f64, radius: f64, l: usize) -> (f64, f64) {
    let (inner, outer) = shape_potential_parts(r, radius, l);
    if r >= radius {
        return (inner, -((l + 1) as f64) * inner / r);
    }
    let r_eff = r.max(1e-12 * radius);
    let (inner_eff, outer_eff) = if r == r_eff {
        (inner, outer)
    } else {
        shape_potential_parts(r_eff, radius, l)
    };
    (
        inner + outer,
        (-((l + 1) as f64) * inner_eff + l as f64 * outer_eff) / r_eff,
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
    if r == 0. && l == 1 {
        // On the center a p function `f(r) Y_1m` with `f ~ slope r` is the linear polynomial
        // `slope r Y_1m(r_hat)`: its gradient is `slope` times the harmonic of each axis, the same
        // limit from every side (the formula below has neither `r_hat` nor `grad Y` there).
        return std::array::from_fn(|axis| {
            let unit = std::array::from_fn(|i| f64::from(u8::from(i == axis)));
            slope * crate::math::angular::normalized(l, m, unit)
        });
    }
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
    } else {
        species.ionic.value_slope_regular(r, 0).1
    }
}

/// The partial-core table as the grid sums it, and its slope: zero past the
/// last node, regular $a + b r^2$ continuation below the first and zero wherever
/// the value is clipped at 0.
pub fn core_slope(table: &Cubic, r: f64) -> (f64, f64) {
    let knots = table.knots.as_slice();
    if r > knots[knots.len() - 1] {
        return (0., 0.);
    }
    let (value, slope) = table.value_slope_regular(r, 0);
    if value <= 0. {
        (0., 0.)
    } else {
        (value, slope)
    }
}
