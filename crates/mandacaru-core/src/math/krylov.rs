//! Conjugate gradients on matrix-free products, for a symmetric system that may be singular.
//!
//! Started from zero, the iteration stays in the Krylov space of the right-hand side, so on a
//! consistent positive-semidefinite system it converges to the minimum-norm solution (the
//! pseudo-inverse applied to `b`): a null direction is never entered. A direction of negative
//! curvature is not a minimum of the quadratic model and is refused; a direction of zero curvature
//! that still carries residual is refused as an inconsistent system.
use super::invalid;
use crate::Result;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Stop when `|b - A x| <= tolerance |b|`.
    pub tolerance: f64,
    /// A curvature `p.A p / p.p` within `+-curvature_floor` is zero.
    pub curvature_floor: f64,
    /// Iteration cap; `None` allows `2 n + 10`.
    pub max_iterations: Option<usize>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            curvature_floor: 1e-8,
            max_iterations: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    pub iterations: usize,
    /// `|b - A x| / |b|` as the iteration saw it.
    pub residual: f64,
    /// The iteration met a zero-curvature direction and left it out (minimum norm).
    pub truncated: bool,
    /// The smallest `p.A p / p.p` along the search directions.
    pub lowest_curvature: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve `A x = b` for the minimum-norm `x`, with `apply` returning `A v`.
pub fn minimum_norm(
    b: &[f64],
    mut apply: impl FnMut(&[f64]) -> Result<Vec<f64>>,
    options: &Options,
) -> Result<Solution> {
    let valid = options.tolerance > 0.
        && options.curvature_floor.is_finite()
        && options.curvature_floor >= 0.
        && b.iter().all(|v| v.is_finite());
    if !valid {
        return Err(invalid("the linear solve needs a finite right-hand side, a positive tolerance and a nonnegative curvature floor"));
    }
    let n = b.len();
    let bnorm = dot(b, b).sqrt();
    let mut x = vec![0.; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs = dot(&r, &r);
    let cap = options.max_iterations.unwrap_or(2 * n + 10);
    let (mut iterations, mut truncated, mut lowest) = (0, false, f64::INFINITY);
    while rs.sqrt() > options.tolerance * bnorm {
        if iterations == cap {
            return Err(invalid(format!(
                "the linear solve did not converge in {cap} products (relative residual {:.3e})",
                rs.sqrt() / bnorm
            )));
        }
        let ap = apply(&p)?;
        if ap.len() != n || ap.iter().any(|v| !v.is_finite()) {
            return Err(invalid("the matrix product returned a wrong-sized or nonfinite vector"));
        }
        let curvature = dot(&p, &ap) / dot(&p, &p);
        lowest = lowest.min(curvature);
        if curvature < -options.curvature_floor {
            return Err(invalid(format!(
                "negative curvature {curvature:.3e} along a search direction: the matrix is indefinite"
            )));
        }
        if curvature <= options.curvature_floor {
            truncated = true;
            break;
        }
        let alpha = rs / (curvature * dot(&p, &p));
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let next = dot(&r, &r);
        let beta = next / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = next;
        iterations += 1;
    }
    let residual = if bnorm > 0. { rs.sqrt() / bnorm } else { 0. };
    if truncated && residual > 1e-4 {
        return Err(invalid(format!(
            "the right-hand side has a component (relative {residual:.3e}) along a null direction of the matrix: the system is inconsistent"
        )));
    }
    Ok(Solution { x, iterations, residual, truncated, lowest_curvature: lowest })
}
