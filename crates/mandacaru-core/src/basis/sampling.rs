//! A tabulated radial orbital `R(r) Y_lm` and its analytic gradient at Cartesian points.
//!
//! The radial function is the table of [`super::table`] (its spline, and the continuation below
//! the first node); the angular factor is the harmonic of [`crate::math::harmonics`], real for a
//! molecule and complex for a crystal. The gradient is
//! `R'(r) r_hat Y + R grad Y`, with a point on the center moved out to
//! [`NODE_OFFSET`] along z, where `r_hat` and the harmonic's direction-dependent pieces are 0/0.
use super::{invalid, table};
use crate::math::harmonics::{harmonic, harmonic_gradient, polar, NODE_OFFSET};
use crate::{Context, Result};
use num_complex::Complex64;

/// The radial table, the angular numbers and the center of one orbital.
pub struct Orbital<'a> {
    pub r: &'a [f64],
    pub values: &'a [f64],
    pub l: usize,
    pub m: i32,
    pub center: [f64; 3],
    /// Real (`float64`) or complex (`complex128`) spherical harmonics.
    pub real: bool,
}

/// A sampled field: `float64` for real harmonics, `complex128` otherwise.
#[derive(Clone, Debug, PartialEq)]
pub enum Sampled {
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
}

fn offsets(orbital: &Orbital<'_>, points: [&[f64]; 3]) -> Result<Vec<[f64; 3]>> {
    let n = points[0].len();
    if points[1].len() != n || points[2].len() != n {
        return Err(invalid("x, y and z must have the same length"));
    }
    let c = orbital.center;
    Ok((0..n)
        .map(|i| [points[0][i] - c[0], points[1][i] - c[1], points[2][i] - c[2]])
        .collect())
}

/// `R(r) Y_lm` at the points `x, y, z`.
pub fn values(
    ctx: &Context,
    orbital: &Orbital<'_>,
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> Result<Sampled> {
    let offsets = offsets(orbital, [x, y, z])?;
    let polar: Vec<(f64, f64, f64)> = offsets.iter().map(|p| polar(p[0], p[1], p[2])).collect();
    let radii: Vec<f64> = polar.iter().map(|p| p.0).collect();
    let radial = table::evaluate(ctx, orbital.r, orbital.values, orbital.l, &radii, false)?;
    let (l, m) = (orbital.l, orbital.m);
    Ok(if orbital.real {
        Sampled::Real(
            polar.iter().zip(&radial)
                .map(|(&(_, theta, phi), &r)| r * harmonic(l, m, theta, phi, true).0)
                .collect(),
        )
    } else {
        Sampled::Complex(
            polar.iter().zip(&radial)
                .map(|(&(_, theta, phi), &r)| {
                    let y = harmonic(l, m, theta, phi, false).1;
                    Complex64::new(r * y.re, r * y.im)
                })
                .collect(),
        )
    })
}

/// The gradient of `R(r) Y_lm` at the points, axis-major: `3 * n` entries, `x` then `y` then `z`.
pub fn gradient(
    ctx: &Context,
    orbital: &Orbital<'_>,
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> Result<Sampled> {
    let mut offsets = offsets(orbital, [x, y, z])?;
    let mut radii = Vec::with_capacity(offsets.len());
    for p in &mut offsets {
        let mut r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if r < NODE_OFFSET {
            p[2] = NODE_OFFSET;
            r = NODE_OFFSET;
        }
        radii.push(r);
    }
    let (l, m, n) = (orbital.l, orbital.m, offsets.len());
    let radial = table::evaluate(ctx, orbital.r, orbital.values, l, &radii, false)?;
    let slope = table::evaluate(ctx, orbital.r, orbital.values, l, &radii, true)?;
    let terms = offsets.iter().map(|p| harmonic_gradient(l, m, p[0], p[1], p[2], orbital.real));
    Ok(if orbital.real {
        let mut out = vec![0.; 3 * n];
        for (i, (p, (value, g, _, _))) in offsets.iter().zip(terms).enumerate() {
            for axis in 0..3 {
                out[axis * n + i] =
                    slope[i] * p[axis] / radii[i] * value + radial[i] * g[axis];
            }
        }
        Sampled::Real(out)
    } else {
        let mut out = vec![Complex64::default(); 3 * n];
        for (i, (p, (_, _, value, g))) in offsets.iter().zip(terms).enumerate() {
            for axis in 0..3 {
                let direction = slope[i] * p[axis] / radii[i];
                out[axis * n + i] = Complex64::new(
                    direction * value.re + radial[i] * g[axis].re,
                    direction * value.im + radial[i] * g[axis].im,
                );
            }
        }
        Sampled::Complex(out)
    })
}
