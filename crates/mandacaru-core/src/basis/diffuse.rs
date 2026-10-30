//! Diffuse quasi-Gaussian shells of a PAW-LCAO basis: width and support in
//! closed form, the radial function, the extended grid, the one-center
//! Gram-Schmidt and the geometry of the box diagnostic.
use super::{
    invalid,
    pao::{quasi_gaussian, radial_norm, trapezoid},
};
use crate::{
    math::{special::gamma_q_inverse, spline::Cubic},
    Context, Result,
};

/// Samples of the unit-`alpha` rms-radius integral.
const UNIT_SAMPLES: usize = 20_001;

/// `c_l` with `r_cut = c_l / sqrt(alpha)`: the radius leaving `cutoff_norm` of
/// the squared norm of `r^l exp(-alpha r^2)` outside.
pub fn gaussian_cut(l: usize, cutoff_norm: f64) -> Result<f64> {
    Ok((0.5 * gamma_q_inverse(l as f64 + 1.5, cutoff_norm)?).sqrt())
}

/// rms radius `sqrt(<r^2>)` of the radial function `values` on `r`.
pub fn rms(r: &[f64], values: &[f64]) -> f64 {
    let density: Vec<f64> = r.iter().zip(values).map(|(&r, &v)| v * v * r * r).collect();
    let weighted: Vec<f64> = density.iter().zip(r).map(|(d, &r)| d * r * r).collect();
    (trapezoid(r, &weighted) / trapezoid(r, &density)).sqrt()
}

/// numpy's `linspace(0, stop, count)`.
fn linspace(stop: f64, count: usize) -> Vec<f64> {
    let step = stop / (count - 1) as f64;
    let mut out: Vec<f64> = (0..count).map(|i| i as f64 * step).collect();
    out[count - 1] = stop;
    out
}

/// rms radius of the cut quasi-Gaussian at `alpha = 1` (Bohr).
pub fn unit_rms(l: usize, cutoff_norm: f64) -> Result<f64> {
    let c = gaussian_cut(l, cutoff_norm)?;
    let r = linspace(c, UNIT_SAMPLES);
    let shape = quasi_gaussian(&r, 1., c);
    let values: Vec<f64> = r
        .iter()
        .zip(&shape)
        .map(|(&r, g)| r.powi(l as i32) * g)
        .collect();
    Ok(rms(&r, &values))
}

fn check_recipe(rms: f64, cutoff_norm: f64) -> Result<()> {
    if rms.is_finite()
        && rms > 0.
        && cutoff_norm.is_finite()
        && cutoff_norm > 0.
        && cutoff_norm < 1.
    {
        Ok(())
    } else {
        Err(invalid(
            "a diffuse shell needs a positive rms radius and a cutoff norm in (0, 1)",
        ))
    }
}

/// `(alpha, r_cut)` (Bohr^-2, Bohr) of the cut quasi-Gaussian of angular
/// momentum `l` whose rms radius is `rms` (Bohr).
pub fn parameters(l: usize, rms: f64, cutoff_norm: f64) -> Result<(f64, f64)> {
    check_recipe(rms, cutoff_norm)?;
    let alpha = (unit_rms(l, cutoff_norm)? / rms).powi(2);
    Ok((alpha, gaussian_cut(l, cutoff_norm)? / alpha.sqrt()))
}

/// The unit-norm diffuse shell on `r`: `(R, alpha, r_cut)`.
pub fn radial(r: &[f64], l: usize, rms: f64, cutoff_norm: f64) -> Result<(Vec<f64>, f64, f64)> {
    if r.len() < 4 || r.windows(2).any(|w| w[1] <= w[0]) || r[0] < 0. {
        return Err(invalid("a diffuse shell needs an increasing radial grid"));
    }
    let (alpha, r_cut) = parameters(l, rms, cutoff_norm)?;
    let shape = quasi_gaussian(r, alpha, r_cut);
    let mut values: Vec<f64> = r
        .iter()
        .zip(&shape)
        .map(|(&r, g)| r.powi(l as i32) * g)
        .collect();
    let norm = radial_norm(r, &values).sqrt();
    for v in &mut values {
        *v /= norm;
    }
    Ok((values, alpha, r_cut))
}

/// `r` continued with its last spacing up to `reach` (Bohr).
pub fn extended_grid(r: &[f64], reach: f64) -> Result<Vec<f64>> {
    if r.len() < 2 || r.windows(2).any(|w| w[1] <= w[0]) {
        return Err(invalid("the radial grid must be increasing"));
    }
    let last = r[r.len() - 1];
    let mut out = r.to_vec();
    if reach <= last {
        return Ok(out);
    }
    let h = last - r[r.len() - 2];
    let count = ((reach - last) / h).ceil() as usize;
    if count > 100_000_000 {
        return Err(invalid("the diffuse grid would be too large"));
    }
    out.extend((1..=count).map(|i| last + h * i as f64));
    Ok(out)
}

/// A table on `r_old` resampled on `r_new`, zero past its end.
pub fn on_grid(ctx: &Context, r_new: &[f64], r_old: &[f64], values: &[f64]) -> Result<Vec<f64>> {
    if r_old.len() != values.len() {
        return Err(invalid("the table and its grid differ in length"));
    }
    if r_new.len() >= r_old.len() && r_new[..r_old.len()] == *r_old {
        let mut out = values.to_vec();
        out.resize(r_new.len(), 0.);
        return Ok(out);
    }
    let curve = Cubic::new(ctx, ctx.copy_slice(r_old)?, ctx.copy_slice(values)?)?;
    let (first, last) = (r_old[0], r_old[r_old.len() - 1]);
    Ok(r_new
        .iter()
        .map(|&x| {
            if x > last {
                0.
            } else {
                let out = curve.value_slope(x.clamp(first, last)).0;
                if out.is_nan() {
                    0.
                } else {
                    out
                }
            }
        })
        .collect())
}

/// Squared norm below which a shell is dropped after the projection.
pub const LINEAR_DEPENDENCE_TOL: f64 = 1e-4;

/// One diffuse shell after the one-center Gram-Schmidt.
#[derive(Clone, Debug)]
pub struct Shell {
    /// Squared norm left after the projection.
    pub residual: f64,
    pub accepted: bool,
    /// The orthonormalized shell (empty when rejected).
    pub values: Vec<f64>,
}

fn weight(r: &[f64], a: &[f64], b: &[f64]) -> f64 {
    let y: Vec<f64> = (0..r.len()).map(|i| a[i] * b[i] * r[i] * r[i]).collect();
    trapezoid(r, &y)
}

/// Orthogonalize the shell `values` against the functions `same` of its
/// channel (modified Gram-Schmidt on an orthonormalized copy of them, twice).
/// `renormalize` rescales `values` to unit norm first (after a filter).
pub fn orthogonalize(
    r: &[f64],
    same: &[&[f64]],
    values: &[f64],
    renormalize: bool,
    tolerance: f64,
) -> Result<Shell> {
    let n = r.len();
    if values.len() != n || same.iter().any(|g| g.len() != n) {
        return Err(invalid(
            "the diffuse shell and its channel must share a grid",
        ));
    }
    let mut values = values.to_vec();
    if renormalize {
        let norm = radial_norm(r, &values).sqrt();
        for v in &mut values {
            *v /= norm;
        }
    }
    let mut candidate = values.clone();
    let mut ortho: Vec<Vec<f64>> = Vec::new();
    for g in same {
        let norm = weight(r, g, g);
        if norm <= 0. {
            continue;
        }
        let mut g: Vec<f64> = g.iter().map(|v| v / norm.sqrt()).collect();
        for _pass in 0..2 {
            for o in &ortho {
                let overlap = weight(r, o, &g);
                for (v, o) in g.iter_mut().zip(o) {
                    *v -= overlap * o;
                }
            }
        }
        let norm = weight(r, &g, &g);
        if norm > 1e-12 {
            ortho.push(g.iter().map(|v| v / norm.sqrt()).collect());
        }
    }
    for _pass in 0..2 {
        for o in &ortho {
            let overlap = weight(r, o, &candidate);
            for (v, o) in candidate.iter_mut().zip(o) {
                *v -= overlap * o;
            }
        }
    }
    let residual = weight(r, &candidate, &candidate);
    let accepted = residual >= tolerance;
    if !accepted {
        return Ok(Shell {
            residual,
            accepted,
            values: Vec::new(),
        });
    }
    for v in &mut candidate {
        *v /= residual.sqrt();
    }
    if weight(r, &candidate, &values) < 0. {
        for v in &mut candidate {
            *v = -*v;
        }
    }
    Ok(Shell {
        residual,
        accepted,
        values: candidate,
    })
}

fn inverse3(m: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let cof = |a: usize, b: usize| {
        let (r0, r1) = ((a + 1) % 3, (a + 2) % 3);
        let (c0, c1) = ((b + 1) % 3, (b + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    let det = m[0][0] * cof(0, 0) + m[0][1] * cof(0, 1) + m[0][2] * cof(0, 2);
    if !(det.is_finite() && det.abs() > 0.) {
        return Err(invalid("the grid steps are linearly dependent"));
    }
    let mut out = [[0.; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = cof(j, i) / det;
        }
    }
    Ok(out)
}

/// Distance from `center` to the nearest face of a grid whose node `0` is at
/// `origin`, whose steps are the rows of `step` (Bohr) and which has `shape`
/// nodes per axis.
pub fn face_distance(
    origin: [f64; 3],
    step: [[f64; 3]; 3],
    shape: [usize; 3],
    center: [f64; 3],
) -> Result<f64> {
    if shape.iter().any(|&n| n < 2) {
        return Err(invalid("a grid needs at least two nodes per axis"));
    }
    // span = step @ diag(shape - 1): column j scaled by shape_j - 1.
    let mut span = step;
    for row in &mut span {
        for (j, v) in row.iter_mut().enumerate() {
            *v *= (shape[j] - 1) as f64;
        }
    }
    let inverse = inverse3(span)?;
    let mut best = f64::INFINITY;
    for row in &inverse {
        let width = 1. / row.iter().map(|v| v * v).sum::<f64>().sqrt();
        let fraction: f64 = (0..3).map(|j| row[j] * (center[j] - origin[j])).sum();
        best = best.min(fraction.min(1. - fraction) * width);
    }
    Ok(best)
}
