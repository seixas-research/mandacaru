//! Model-free Jacobi orbital localization kernels (`Foster-Boys` and `Pipek-Mezey`).
//!
//! Operates purely on symmetric real matrices in orbital space (`DMatrix<f64>`), leaving all
//! AO/PAW/grid physics in the caller crates (`molecular`, `periodic`).
use super::invalid;
use crate::Result;
use nalgebra::DMatrix;
use std::f64::consts::FRAC_PI_4;

/// Output of an orbital Jacobi localization sweep.
#[derive(Clone, Debug)]
pub struct JacobiResult {
    pub rotation: DMatrix<f64>,
    pub objective: f64,
    pub sweeps: usize,
    pub converged: bool,
}

fn apply_jacobi_pair(
    matrices: &mut [DMatrix<f64>],
    rotation: &mut DMatrix<f64>,
    i: usize,
    j: usize,
    angle: f64,
) {
    let n = rotation.nrows();
    let (s, q) = angle.sin_cos();
    for m in matrices {
        for k in 0..n {
            let (a, b) = (m[(i, k)], m[(j, k)]);
            m[(i, k)] = q * a + s * b;
            m[(j, k)] = -s * a + q * b;
        }
        for k in 0..n {
            let (a, b) = (m[(k, i)], m[(k, j)]);
            m[(k, i)] = q * a + s * b;
            m[(k, j)] = -s * a + q * b;
        }
    }
    for k in 0..n {
        let (a, b) = (rotation[(k, i)], rotation[(k, j)]);
        rotation[(k, i)] = q * a + s * b;
        rotation[(k, j)] = -s * a + q * b;
    }
}

fn validate_square_set(matrices: &[DMatrix<f64>], what: &str) -> Result<usize> {
    let Some(first) = matrices.first() else {
        return Err(invalid(format!(
            "{what} needs at least one operator matrix"
        )));
    };
    let n = first.nrows();
    if matrices
        .iter()
        .any(|m| m.nrows() != n || m.ncols() != n || m.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid(format!(
            "{what} matrices must be square, finite, and matching size"
        )));
    }
    Ok(n)
}

/// Optimal 2x2 angle maximizing `sum_A [(M^A_{ii})^4 + (M^A_{jj})^4]`.
fn quartic_pair_angle(matrices: &[DMatrix<f64>], i: usize, j: usize) -> f64 {
    let eval_d1_d2 = |gamma: f64| -> (f64, f64, f64) {
        let (s2, c2) = (2. * gamma).sin_cos();
        let mut f = 0.;
        let mut df = 0.;
        let mut d2f = 0.;
        for m in matrices {
            let avg = 0.5 * (m[(i, i)] + m[(j, j)]);
            let diff = 0.5 * (m[(i, i)] - m[(j, j)]);
            let off = 0.5 * (m[(i, j)] + m[(j, i)]);
            let d = diff * c2 + off * s2;
            let d1 = -2. * diff * s2 + 2. * off * c2;
            let d2 = -4. * d;
            let qp = avg + d;
            let qm = avg - d;
            f += qp.powi(4) + qm.powi(4);
            df += 4. * (qp.powi(3) - qm.powi(3)) * d1;
            d2f += 12. * (qp * qp + qm * qm) * d1 * d1 + 4. * (qp.powi(3) - qm.powi(3)) * d2;
        }
        (f, df, d2f)
    };
    let (base_f, _, _) = eval_d1_d2(0.);
    // Sample 8 points in (-pi/4, pi/4] to capture any multi-peaked quartic landscape cleanly.
    let mut best_g = 0.;
    let mut best_f = base_f;
    for k in -3..=4 {
        let g = (k as f64 / 4.) * FRAC_PI_4;
        let (f, _, _) = eval_d1_d2(g);
        if f > best_f {
            best_f = f;
            best_g = g;
        }
    }
    let mut g = best_g;
    for _ in 0..16 {
        let (_, df, d2f) = eval_d1_d2(g);
        if df.abs() < 1e-13 {
            break;
        }
        if d2f < -1e-12 {
            let step = (-df / d2f).clamp(-0.2, 0.2);
            let trial = g + step;
            let (f_trial, _, _) = eval_d1_d2(trial);
            let (f_curr, _, _) = eval_d1_d2(g);
            if f_trial >= f_curr {
                g = trial;
                continue;
            }
        }
        let g_step = 1e-3 * df.signum();
        let (f_try, _, _) = eval_d1_d2(g + g_step);
        let (f_curr, _, _) = eval_d1_d2(g);
        if f_try > f_curr {
            g += g_step;
        } else {
            break;
        }
    }
    let (f_final, _, _) = eval_d1_d2(g);
    if f_final <= base_f + 1e-15 {
        0.
    } else {
        g
    }
}

fn quadratic_pair_angle(matrices: &[DMatrix<f64>], i: usize, j: usize) -> f64 {
    let mut aa = 0.;
    let mut bb = 0.;
    for m in matrices {
        let a = 0.5 * (m[(i, j)] + m[(j, i)]);
        let b = m[(i, i)] - m[(j, j)];
        aa += a * a - 0.25 * b * b;
        bb += a * b;
    }
    if aa.abs() < 1e-14 && bb.abs() < 1e-14 {
        0.
    } else {
        0.25 * bb.atan2(-aa)
    }
}

fn jacobi_maximize(
    matrices: &mut [DMatrix<f64>],
    exponent: usize,
    max_iterations: usize,
    tolerance: f64,
    what: &str,
) -> Result<JacobiResult> {
    let n = validate_square_set(matrices, what)?;
    if ![2, 4].contains(&exponent) {
        return Err(invalid(format!(
            "{what} exponent must be 2 or 4; got {exponent}"
        )));
    }
    if max_iterations == 0 || !tolerance.is_finite() || tolerance <= 0. {
        return Err(invalid(format!("invalid {what} options")));
    }
    let mut rotation = DMatrix::identity(n, n);
    let mut converged = false;
    let mut sweeps = 0;
    for sweep in 0..max_iterations {
        sweeps = sweep + 1;
        let mut largest = 0_f64;
        for i in 0..n {
            for j in i + 1..n {
                let angle = if exponent == 2 {
                    quadratic_pair_angle(matrices, i, j)
                } else {
                    quartic_pair_angle(matrices, i, j)
                };
                largest = largest.max(angle.abs());
                if angle.abs() < tolerance {
                    continue;
                }
                apply_jacobi_pair(matrices, &mut rotation, i, j, angle);
            }
        }
        if largest < tolerance {
            converged = true;
            break;
        }
    }
    let p = exponent as i32;
    let objective = (0..n)
        .map(|i| matrices.iter().map(|m| m[(i, i)].powi(p)).sum::<f64>())
        .sum();
    Ok(JacobiResult {
        rotation,
        objective,
        sweeps,
        converged,
    })
}

/// Jacobi Foster-Boys localization maximizing `sum_i sum_a (r^a_{ii})^2` over orthonormal pairs.
/// Mutates `coords` into the rotated frame; returns the orthogonal rotation `U` (`C' = C U`).
pub fn boys(
    coords: &mut [DMatrix<f64>],
    max_iterations: usize,
    tolerance: f64,
) -> Result<JacobiResult> {
    jacobi_maximize(
        coords,
        2,
        max_iterations,
        tolerance,
        "Foster-Boys localization",
    )
}

/// Jacobi Pipek-Mezey localization maximizing `sum_i sum_A (q^A_{ii})^p` (`exponent` 2 or 4)
/// over symmetric atomic population matrices `populations[A]`.
pub fn pipek_mezey(
    populations: &mut [DMatrix<f64>],
    exponent: usize,
    max_iterations: usize,
    tolerance: f64,
) -> Result<JacobiResult> {
    jacobi_maximize(
        populations,
        exponent,
        max_iterations,
        tolerance,
        "Pipek-Mezey localization",
    )
}
