//! Compensation multipoles of PAW-LCAO: the radial shape `g_L` and its Coulomb
//! potential for any `L`, the angular coupling (Gaunt) of real or complex
//! harmonics, and the quadrature rule that evaluates them.
use super::invalid;
use crate::math::{angular, radial::gauss_legendre};
use crate::Result;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Polar (Gauss-Legendre) and azimuthal points of the angular rule. A product
/// rule of these sizes is exact for three harmonics up to l = 3 with margin.
pub const POLAR_POINTS: usize = 24;
pub const AZIMUTHAL_POINTS: usize = 48;

/// `1 / int_0^{r_g} r^{2L+2} (1 - r^2/r_g^2)^3 dr`, the factor that gives the
/// shape a unit `L` multipole moment.
pub fn shape_norm(r_g: f64, l: usize) -> f64 {
    let l = l as f64;
    let bracket = 1. / (2. * l + 3.) - 3. / (2. * l + 5.) + 3. / (2. * l + 7.) - 1. / (2. * l + 9.);
    1. / (r_g.powf(2. * l + 3.) * bracket)
}

fn check(r_g: f64) -> Result<()> {
    if !r_g.is_finite() || r_g <= 0. {
        return Err(invalid(
            "the compensation radius must be positive and finite",
        ));
    }
    Ok(())
}

/// `g_L(r) = N_L r^L (1 - r^2/r_g^2)^3` inside `r_g`, zero beyond.
pub fn shape_function(radius: &[f64], r_g: f64, l: usize) -> Result<Vec<f64>> {
    check(r_g)?;
    let norm = shape_norm(r_g, l);
    Ok(radius
        .iter()
        .map(|&r| {
            let x = r / r_g;
            if x < 1. {
                norm * r.powi(l as i32) * (1. - x * x).powi(3)
            } else {
                0.
            }
        })
        .collect())
}

/// Coulomb potential of `g_L Y_LM` without the harmonic:
/// `4 pi / (2L+1) [r^-(L+1) int_0^r g s^(L+2) ds + r^L int_r^inf g s^(1-L) ds]`,
/// both integrals in closed form.
pub fn shape_potential(radius: &[f64], r_g: f64, l: usize) -> Result<Vec<f64>> {
    check(r_g)?;
    let prefactor = 4. * PI / (2 * l + 1) as f64;
    let norm = shape_norm(r_g, l);
    Ok(radius
        .iter()
        .map(|&r| {
            if r >= r_g {
                return prefactor / r.max(1e-300).powi(l as i32 + 1);
            }
            let x = (r / r_g).max(0.);
            let (mut inner, mut tail) = (0., 0.);
            for (k, c) in [1., -3., 3., -1.].into_iter().enumerate() {
                let power = (2 * l + 3 + 2 * k) as i32;
                inner += c * x.powi(power) / power as f64;
                let exponent = 2 * k as i32 + 2;
                tail += c * (1. - x.powi(exponent)) / exponent as f64;
            }
            inner *= norm * r_g.powi(2 * l as i32 + 3);
            tail *= norm * r_g * r_g;
            let enclosed = if r > 0. {
                inner / r.max(1e-300).powi(l as i32 + 1)
            } else {
                0.
            };
            prefactor * (enclosed + r.powi(l as i32) * tail)
        })
        .collect())
}

/// `(theta, phi, weights)` of the product rule, theta slowest, normalized to
/// `int dOmega = 4 pi`.
pub fn angular_grid(polar: usize, azimuthal: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if azimuthal == 0 {
        return Err(invalid("the angular rule needs azimuthal points"));
    }
    let rule = gauss_legendre(polar)?;
    let weight_phi = 2. * PI / azimuthal as f64;
    let n = polar * azimuthal;
    let (mut theta, mut phi, mut weight) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (mu, w) in rule {
        for j in 0..azimuthal {
            theta.push(mu.acos());
            phi.push(2. * PI * (j as f64 + 0.5) / azimuthal as f64);
            weight.push(w * weight_phi);
        }
    }
    Ok((theta, phi, weight))
}

/// Unit direction of every point of [`angular_grid`].
pub fn directions(theta: &[f64], phi: &[f64]) -> Vec<[f64; 3]> {
    theta
        .iter()
        .zip(phi)
        .map(|(t, p)| [t.sin() * p.cos(), t.sin() * p.sin(), t.cos()])
        .collect()
}

/// Harmonic of the requested kind (real, or complex with the Condon-Shortley phase).
pub fn harmonic(l: usize, m: i32, direction: [f64; 3], real: bool) -> Complex64 {
    if real {
        Complex64::new(angular::cartesian(l, m, direction), 0.)
    } else {
        angular::complex(l, m, direction)
    }
}

/// The `L` a pair of waves with momenta `l1`, `l2` can carry.
pub fn multipole_range(l1: usize, l2: usize) -> Vec<usize> {
    (l1.abs_diff(l2)..=l1 + l2)
        .filter(|l| (l + l1 + l2) % 2 == 0)
        .collect()
}

/// `int Y*_LM Y*_{l1 m1} Y_{l2 m2} dOmega` (complex harmonics) or
/// `int S_LM S_{l1 m1} S_{l2 m2} dOmega` (real), by the product rule. Zero
/// unless the triangle and parity rules hold, and for complex harmonics unless
/// `M = m2 - m1`; entries below 1e-12 are set to zero.
pub fn gaunt(
    level: [usize; 3],
    order: [i32; 3],
    real: bool,
    grid: &(Vec<[f64; 3]>, Vec<f64>),
) -> Complex64 {
    let [l, a, b] = level;
    let [m, m1, m2] = order;
    if (!real && m != m2 - m1) || l < a.abs_diff(b) || l > a + b || (l + a + b) % 2 != 0 {
        return Complex64::new(0., 0.);
    }
    let mut total = Complex64::new(0., 0.);
    for (direction, w) in grid.0.iter().zip(&grid.1) {
        let (y, y1, y2) = (
            harmonic(l, m, *direction, real),
            harmonic(a, m1, *direction, real),
            harmonic(b, m2, *direction, real),
        );
        total += if real {
            y * y1 * y2
        } else {
            y.conj() * y1.conj() * y2
        } * *w;
    }
    if total.norm() < 1e-12 {
        Complex64::new(0., 0.)
    } else {
        total
    }
}

/// The default quadrature grid of [`gaunt`].
pub fn gaunt_grid() -> Result<(Vec<[f64; 3]>, Vec<f64>)> {
    let (theta, phi, weight) = angular_grid(POLAR_POINTS, AZIMUTHAL_POINTS)?;
    Ok((directions(&theta, &phi), weight))
}
