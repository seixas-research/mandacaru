//! Strain derivatives of the screened Coulomb kernel `v^omega` of the exact exchange.
//!
//! Convention (the stress plan's, shared by every strain derivative of the crystal): the strain
//! `eps` acts as `x -> (1 + eps) x` on lattice vectors, atoms and grid nodes at fixed fractional
//! coordinates, so a wave vector moves as `Q -> (1 - eps^T) Q` and `Q^2` as
//! `d Q^2 / d eps_ab = -2 Q_a Q_b`. Every function here returns the derivative of the energy-side
//! quantity, `d v / d eps_ab` (symmetric in `ab`: Voigt xx, yy, zz, yz, xz, xy), NOT a stress.
//! The stress is `sigma_ab = +(1/Omega) (dE/d eps_ab + dE/d eps_ba) / 2`, the sign ASE uses; a
//! caller that wants `-(1/Omega)` is wrong.
//!
//! The kernel the exchange evaluates is a function of the alias-MEAN square of the wave vector
//! (`math::reciprocal::bloch`), so its strain derivative is `v'(s)` times the alias mean of
//! `-2 Q_a Q_b` over the same vectors (`reciprocal::bloch_outer`), never the outer product of
//! the mean vector.
use crate::{
    math::{reciprocal, SpatialGrid},
    Buffer, Context, Result,
};
use std::f64::consts::PI;

/// The derivative of the screened kernel `v^omega(s)` at the mean square `s`, with `outer` the
/// alias mean of `Q_a Q_b` (xx, yy, zz, yz, xz, xy) belonging to `s`:
/// `d v / d eps_ab = -2 v'(s) outer_ab`, the Taylor series of `v'` below `s = 1e-4`.
///
/// At `s <= 1e-12` (the kernel's own threshold) the frequency is the head: zero for
/// [`super::screened_exchange::ExxDivTreatment::None`] (the limit `pi / omega^2` does not
/// depend on the cell), and for Gygi-Baldereschi `delta_ab (v_bar(q_c) - v(q_c))` with
/// `q_c = (6 pi^2 w_q / Omega)^(1/3)`: `q_c` shrinks as `Omega^(-1/3)`, so
/// `d q_c / d eps_ab = -q_c delta_ab / 3` and `d v_bar / d q_c = 3 (v(q_c) - v_bar) / q_c`.
pub fn kernel_strain_from_moments(
    s: f64,
    outer: [f64; 6],
    omega: f64,
    qc: f64,
    exxdiv: super::screened_exchange::ExxDivTreatment,
) -> [f64; 6] {
    if s <= 1e-12 {
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

    // Differentiate v(s) = 4 pi / s * (1 - exp(-s / (4 w^2))) with respect to s.
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
    outer.map(|o| -2.0 * dv_ds * o)
}

/// [`kernel_strain_from_moments`] of one wave vector `q`, which is its own mean: `s = |q|^2`
/// and `outer = q (x) q`. Returns [xx, yy, zz, yz, xz, xy].
pub fn kernel_strain_derivative(
    q: &[f64; 3],
    omega: f64,
    qc: f64,
    exxdiv: super::screened_exchange::ExxDivTreatment,
) -> [f64; 6] {
    let s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
    let outer = [
        q[0] * q[0],
        q[1] * q[1],
        q[2] * q[2],
        q[1] * q[2],
        q[0] * q[2],
        q[0] * q[1],
    ];
    kernel_strain_from_moments(s, outer, omega, qc, exxdiv)
}

/// `d v / d eps_ab` of the kernel the exchange evaluates for the pair wave vector `k_diff = q - k`
/// (Cartesian, inverse Bohr), `(6, n)` component-major in the order of
/// [`kernel_strain_from_moments`]: the alias-mean square and outer product of
/// `reciprocal::bloch` / `reciprocal::bloch_outer` and, at the head, the Gygi-Baldereschi radius
/// of the PAIR's own weight `pair_weight` (`w_q`; `1 / nk` only on a uniform mesh). The cell
/// volume of the radius is the grid's.
pub fn screened_kernel_strain(
    ctx: &Context,
    grid: &SpatialGrid,
    k_diff: [f64; 3],
    pair_weight: f64,
    omega: f64,
    exxdiv: super::screened_exchange::ExxDivTreatment,
) -> Result<Buffer<f64>> {
    let n = grid.index.points();
    let volume = grid.index.dv() * n as f64;
    let qc = (6.0 * PI * PI * pair_weight / volume).cbrt();
    let symbols = reciprocal::bloch(ctx, grid, k_diff)?;
    let outer = reciprocal::bloch_outer(ctx, grid, k_diff)?;
    let mut out = ctx.allocate::<f64>(crate::product(&[6, n])?)?;
    for at in 0..n {
        let moments = std::array::from_fn(|row| outer.as_slice()[row * n + at]);
        let derivative =
            kernel_strain_from_moments(symbols.squares.as_slice()[at], moments, omega, qc, exxdiv);
        for (row, value) in derivative.iter().enumerate() {
            out.as_mut_slice()[row * n + at] = *value;
        }
    }
    Ok(out)
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
            let s_minus =
                q_minus[0] * q_minus[0] + q_minus[1] * q_minus[1] + q_minus[2] * q_minus[2];

            let v_plus = 4.0 * PI * (1.0 - (-s_plus / (4.0 * omega * omega)).exp()) / s_plus;
            let v_minus = 4.0 * PI * (1.0 - (-s_minus / (4.0 * omega * omega)).exp()) / s_minus;

            let num = (v_plus - v_minus) / (2.0 * eps) * if a == b { 1.0 } else { 0.5 };
            let diff = (ana[v_idx] - num).abs();
            assert!(
                diff < 1e-7,
                "Voigt {v_idx} ({a},{b}): analytic {} vs num {}, diff {}",
                ana[v_idx],
                num,
                diff
            );
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
