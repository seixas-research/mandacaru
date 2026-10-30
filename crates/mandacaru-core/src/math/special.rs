//! Regularized incomplete gamma function and its inverse.
use super::{invalid, roots::brent};
use crate::Result;

const EPS: f64 = 1e-16;
const TINY: f64 = 1e-300;
const ITERATIONS: usize = 10_000;

/// `ln(x^a e^-x / Gamma(a))`, the logarithm of the prefactor of both expansions.
fn log_prefactor(a: f64, x: f64) -> f64 {
    a * x.ln() - x - libm::lgamma(a)
}

/// Regularized upper incomplete gamma function `Q(a, x) = Gamma(a, x) / Gamma(a)`
/// for `a > 0`, `x >= 0`: the series for `P` below `a + 1`, the continued
/// fraction (modified Lentz) above it, so a small tail keeps its relative accuracy.
pub fn gamma_q(a: f64, x: f64) -> Result<f64> {
    if !(a.is_finite() && a > 0. && x.is_finite() && x >= 0.) {
        return Err(invalid(
            "the incomplete gamma function needs a > 0 and x >= 0",
        ));
    }
    if x == 0. {
        return Ok(1.);
    }
    if x < a + 1. {
        let mut term = 1. / a;
        let mut sum = term;
        let mut ap = a;
        for _ in 0..ITERATIONS {
            ap += 1.;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * EPS {
                return Ok(1. - sum * log_prefactor(a, x).exp());
            }
        }
    } else {
        let mut b = x + 1. - a;
        let mut c = 1. / TINY;
        let mut d = 1. / b;
        let mut h = d;
        for i in 1..ITERATIONS {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1. / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.).abs() < EPS {
                return Ok(log_prefactor(a, x).exp() * h);
            }
        }
    }
    Err(invalid("the incomplete gamma expansion did not converge"))
}

/// The `x` with `Q(a, x) = y`, for `a > 0` and `0 < y < 1`: bracketed by doubling,
/// refined by Brent's method on `ln Q - ln y` and polished by two Newton steps.
pub fn gamma_q_inverse(a: f64, y: f64) -> Result<f64> {
    if !(a.is_finite() && a > 0. && y.is_finite() && y > 0. && y < 1.) {
        return Err(invalid(
            "the inverse incomplete gamma needs a > 0 and 0 < y < 1",
        ));
    }
    let target = y.ln();
    let mut high = a + 1.;
    for _ in 0..2048 {
        if gamma_q(a, high)?.ln() < target {
            break;
        }
        high *= 2.;
    }
    let mut x = brent(
        |x| Ok(gamma_q(a, x)?.ln() - target),
        0.,
        high,
        0.,
        4. * f64::EPSILON,
        200,
    )?;
    for _ in 0..2 {
        let density = log_prefactor(a, x).exp() / x;
        if density > 0. && density.is_finite() {
            x += (gamma_q(a, x)? - y) / density;
        }
    }
    Ok(x)
}
