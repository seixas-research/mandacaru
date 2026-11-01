//! Reciprocal metric and strain derivatives for periodic stress tensor evaluation.
//! Differentiates the screened Coulomb kernel v^omega(Q) and reciprocal kinetic / pair terms
//! with respect to cell strain epsilon_{alpha, beta}.

use super::{invalid, Crystal};
use crate::{
    math::reciprocal,
    Result,
};
use std::f64::consts::PI;

/// Strain response of the screened Coulomb kernel or Gygi-Baldereschi divergence term:
///
/// Under strain epsilon_{alpha, beta}, reciprocal wave-vectors transform as
/// Q' = (I - epsilon^T) Q, giving d(Q^2) / d(epsilon_{alpha, beta}) = -2 Q_alpha Q_beta.
///
/// Returns [xx, yy, zz, yz, xz, xy] in Voigt order.
pub fn kernel_strain_derivative(
    q: &[f64; 3],
    omega: f64,
    qc: f64,
    exxdiv: super::screened_exchange::ExxDivTreatment,
) -> [f64; 6] {
    let s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
    if s < 1e-12 {
        return match exxdiv {
            super::screened_exchange::ExxDivTreatment::GygiBaldereschi => {
                let v_bar = super::screened_exchange::gygi_baldereschi_screened_kernel(qc, omega);
                let v_qc = super::screened_exchange::screened_kernel(qc * qc, omega);
                let diag = v_bar - v_qc;
                [diag, diag, diag, 0.0, 0.0, 0.0]
            }
            super::screened_exchange::ExxDivTreatment::None => [0.0; 6],
        };
    }
    let w2 = omega * omega;
    let w4 = w2 * w2;

    // Differentiate v(s) = 4 pi / s * (1 - exp(-s / (4 w^2))) with respect to s:
    // dv / ds
    let dv_ds = if s < 1e-4 {
        // Taylor series expansion for numerical stability as s -> 0:
        // dv/ds = -pi / (8 w^4) + pi s / (48 w^6) - pi s^2 / (512 w^8)
        let w6 = w4 * w2;
        let w8 = w6 * w2;
        -PI / (8.0 * w4) + PI * s / (48.0 * w6) - PI * s * s / (512.0 * w8)
    } else {
        let exp_term = (-s / (4.0 * w2)).exp();
        -4.0 * PI / (s * s) * (1.0 - exp_term) + PI / (w2 * s) * exp_term
    };

    // dv / d(epsilon_{alpha, beta}) = -2 Q_alpha Q_beta * dv/ds
    let pref = -2.0 * dv_ds;
    [
        pref * q[0] * q[0], // xx
        pref * q[1] * q[1], // yy
        pref * q[2] * q[2], // zz
        pref * q[1] * q[2], // yz
        pref * q[0] * q[2], // xz
        pref * q[0] * q[1], // xy
    ]
}

/// Evaluates the screened exchange kernel stress contribution (in Voigt order, Ha / Bohr^3):
/// sigma_{alpha, beta}^{xx, kernel} = - 1 / Omega * d E_xx / d(epsilon_{alpha, beta})
pub fn screened_exchange_kernel_stress(
    c: &Crystal,
    omega: f64,
    pair_transformed_weights: &[f64],
    exxdiv: super::screened_exchange::ExxDivTreatment,
) -> Result<[f64; 6]> {
    let volume = c.grid.index.dv() * c.grid.index.points() as f64;
    if volume <= 0.0 {
        return Err(invalid("crystal grid has non-positive volume"));
    }
    let mut stress = [0.0; 6];
    let n = c.grid.index.points();
    if pair_transformed_weights.len() < n {
        return Ok(stress);
    }

    let nk = c.points.len().max(1);
    let qc = (6.0 * PI * PI / (nk as f64 * volume)).cbrt();

    let b = reciprocal::bloch(&c.context, &c.grid, [0.0, 0.0, 0.0])?;
    for k in 0..n {
        let q = [
            b.vectors.as_slice()[k],
            b.vectors.as_slice()[n + k],
            b.vectors.as_slice()[2 * n + k],
        ];
        let dkernel = kernel_strain_derivative(&q, omega, qc, exxdiv);
        let weight = pair_transformed_weights[k];
        for a in 0..6 {
            stress[a] += weight * dkernel[a];
        }
    }

    for a in 0..6 {
        stress[a] /= volume;
    }
    Ok(stress)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::screened_exchange::{self, ExxDivTreatment};

    #[test]
    fn kernel_strain_derivative_matches_finite_difference() {
        let omega = 0.11;
        let q = [0.25, -0.15, 0.35];
        let ana = kernel_strain_derivative(&q, omega, 0.0, ExxDivTreatment::None);

        let eps = 1e-6;
        let voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

        for (v_idx, &(a, b)) in voigt_pairs.iter().enumerate() {
            let mut q_plus = q;
            let mut q_minus = q;

            // Q' = (I - eps) Q
            q_plus[a] -= eps * q[b];
            if a != b {
                q_plus[b] -= eps * q[a];
            }

            q_minus[a] += eps * q[b];
            if a != b {
                q_minus[b] += eps * q[a];
            }

            let s_plus = q_plus[0] * q_plus[0] + q_plus[1] * q_plus[1] + q_plus[2] * q_plus[2];
            let s_minus = q_minus[0] * q_minus[0] + q_minus[1] * q_minus[1] + q_minus[2] * q_minus[2];

            let v_plus = 4.0 * PI * (1.0 - (-s_plus / (4.0 * omega * omega)).exp()) / s_plus;
            let v_minus = 4.0 * PI * (1.0 - (-s_minus / (4.0 * omega * omega)).exp()) / s_minus;

            let num = (v_plus - v_minus) / (2.0 * eps) * if a == b { 1.0 } else { 0.5 };
            let diff = (ana[v_idx] - num).abs();
            assert!(diff < 1e-7, "Voigt {v_idx} ({a},{b}): analytic {} vs num {}, diff {}", ana[v_idx], num, diff);
        }
    }

    #[test]
    fn kernel_derivative_vanishes_at_origin() {
        let omega = 0.11;
        let q = [0.0, 0.0, 0.0];
        let ana = kernel_strain_derivative(&q, omega, 0.0, ExxDivTreatment::None);
        for a in 0..6 {
            assert_eq!(ana[a], 0.0);
        }
    }

    #[test]
    fn gygi_baldereschi_strain_derivative_matches_finite_difference() {
        let omega = 0.11;
        let qc0 = 0.30;
        let ana = kernel_strain_derivative(
            &[0.0, 0.0, 0.0],
            omega,
            qc0,
            ExxDivTreatment::GygiBaldereschi,
        );
        let eps: f64 = 1e-6;
        let qc_plus = qc0 * (1.0f64 + eps).powf(-1.0 / 3.0);
        let qc_minus = qc0 * (1.0f64 - eps).powf(-1.0 / 3.0);
        let v_plus = screened_exchange::gygi_baldereschi_screened_kernel(qc_plus, omega);
        let v_minus = screened_exchange::gygi_baldereschi_screened_kernel(qc_minus, omega);
        let fd_diag = (v_plus - v_minus) / (2.0 * eps);
        assert!((ana[0] - fd_diag).abs() / fd_diag < 1e-6);
        assert_eq!(ana[0], ana[1]);
        assert_eq!(ana[1], ana[2]);
        assert_eq!(ana[3], 0.0);
        assert_eq!(ana[4], 0.0);
        assert_eq!(ana[5], 0.0);
    }
}
