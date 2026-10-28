//! Quadrature, validation and differentiation on radial meshes.
use super::invalid;
use crate::{finite, Result};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

/// Composite trapezoid on a strictly increasing radial mesh.
pub fn integrate(r: &[f64], values: &[f64]) -> Result<f64> {
    validate(r, values)?;
    Ok(r.windows(2)
        .zip(values.windows(2))
        .map(|(r, v)| 0.5 * (r[1] - r[0]) * (v[0] + v[1]))
        .sum())
}
pub fn validate(r: &[f64], values: &[f64]) -> Result<()> {
    if r.len() < 4 || values.len() != r.len() || r[0] < 0. || r.windows(2).any(|w| w[1] <= w[0]) {
        return Err(invalid(
            "radial arrays need at least four increasing nonnegative radii and matching values",
        ));
    }
    finite(r, "radial mesh")?;
    finite(values, "radial values")
}
/// Gauss–Legendre nodes and weights, generated natively by Newton iteration.
pub fn gauss_legendre(n: usize) -> Result<Vec<(f64, f64)>> {
    if !(2..=4096).contains(&n) {
        return Err(invalid("quadrature order must be in 2..4096"));
    }
    let mut nodes = vec![(0., 0.); n];
    for i in 0..n.div_ceil(2) {
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 0.;
        for iteration in 0..100 {
            let (mut prior, mut current) = (1., x);
            for degree in 2..=n {
                let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * prior)
                    / degree as f64;
                prior = current;
                current = next;
            }
            derivative = n as f64 * (x * current - prior) / (x * x - 1.);
            let next = x - current / derivative;
            if (next - x).abs() < 4. * f64::EPSILON {
                x = next;
                break;
            }
            if iteration == 99 {
                return Err(invalid("Gauss–Legendre root did not converge"));
            }
            x = next;
        }
        let weight = 2. / ((1. - x * x) * derivative * derivative);
        nodes[i] = (-x, weight);
        nodes[n - 1 - i] = (x, weight);
    }
    Ok(nodes)
}
/// Local polynomial fit of values and derivatives, with a scaled abscissa.
pub fn derivatives(
    r: &[f64],
    values: &[f64],
    at: f64,
    order: usize,
    window: usize,
) -> Result<Vec<f64>> {
    validate(r, values)?;
    if !at.is_finite() || order > 6 || at < r[0] || at > r[r.len() - 1] {
        return Err(invalid("invalid radial derivative request"));
    }
    let center = r.partition_point(|x| *x < at).min(r.len() - 1);
    let half = window.max(order + 1).min((r.len() - 1) / 2);
    let start = center.saturating_sub(half).min(r.len() - 2 * half - 1);
    let end = start + 2 * half + 1;
    let scale = (r[end - 1] - r[start]).max(f64::MIN_POSITIVE);
    let degree = (order + 4).min(end - start - 1);
    let matrix = DMatrix::from_fn(end - start, degree + 1, |i, j| {
        ((r[start + i] - at) / scale).powi(j as i32)
    });
    let target = DVector::from_column_slice(&values[start..end]);
    let coefficients = matrix
        .svd(true, true)
        .solve(&target, 1e-13)
        .map_err(invalid)?;
    let mut factorial = 1.;
    Ok((0..=order)
        .map(|i| {
            if i > 0 {
                factorial *= i as f64;
            }
            coefficients[i] * factorial / scale.powi(i as i32)
        })
        .collect())
}

/// Four-knot Lagrange interpolation on a mesh validated by `validate`.
/// The caller supplies at least four mesh values and matching samples.
pub fn interpolate(r: &[f64], values: &[f64], x: f64) -> f64 {
    let start = r
        .partition_point(|v| *v < x)
        .saturating_sub(2)
        .min(r.len() - 4);
    (start..start + 4)
        .map(|i| {
            values[i]
                * (start..start + 4)
                    .filter(|&j| j != i)
                    .map(|j| (x - r[j]) / (r[i] - r[j]))
                    .product::<f64>()
        })
        .sum()
}
