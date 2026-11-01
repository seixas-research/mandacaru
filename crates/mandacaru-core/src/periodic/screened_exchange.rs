//! Periodic screened exact exchange operator for hybrid functionals (HSE06).
//! Evaluates the non-singular reciprocal Fock exchange matrix K(k)
//! contracted with the self-consistent density matrices over the k-mesh.
//! Fully supports restricted, unrestricted collinear, and 2-component spinor calculations.

use super::{Crystal, KPoint};
use crate::{
    math::{
        fft::Fft3,
        reciprocal,
        storage::MatrixSource,
    },
    quantum::C64,
    Buffer, Result,
};
use nalgebra::DMatrix;
use std::f64::consts::PI;

type Matrix = DMatrix<C64>;

/// Divergence treatment scheme for periodic exact exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExxDivTreatment {
    /// Standard non-singular point evaluation at Q = 0: lim_{Q->0} v^omega(Q) = pi / omega^2.
    #[default]
    None,
    /// Gygi-Baldereschi spherical Brillouin-zone integration:
    /// analytical spherical average of v^omega(Q) over the equivalent k-point Voronoi sphere.
    GygiBaldereschi,
}

impl ExxDivTreatment {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "false" | "" => Ok(Self::None),
            "gygi-baldereschi" | "gb" | "gygi_baldereschi" | "on" | "true" => {
                Ok(Self::GygiBaldereschi)
            }
            other => Err(crate::Error::Invalid(format!(
                "unknown exxdiv_treatment: '{other}', expected 'none' or 'gygi-baldereschi'"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::GygiBaldereschi => "gygi-baldereschi",
        }
    }
}

/// Reciprocal screened Coulomb kernel v^omega(Q) = 4 pi / Q^2 * (1 - exp(-Q^2 / (4 omega^2))).
/// Strictly finite as Q -> 0: lim_{Q -> 0} v^omega(Q) = pi / omega^2.
#[inline]
pub fn screened_kernel(q2: f64, omega: f64) -> f64 {
    if q2 > 1e-12 {
        4.0 * PI * (1.0 - (-q2 / (4.0 * omega * omega)).exp()) / q2
    } else {
        PI / (omega * omega)
    }
}

/// Gygi-Baldereschi divergence treatment for the screened Coulomb interaction.
/// Evaluates the analytic spherical average of v^omega(Q) over the k-point Voronoi sphere
/// of radius q_c = (6 pi^2 w_q / Omega_cell)^{1/3}:
///
///   v_bar^omega = 12 pi / q_c^2 * [ 1 - omega sqrt(pi) / q_c * erf(q_c / (2 omega)) ]
#[inline]
pub fn gygi_baldereschi_screened_kernel(qc: f64, omega: f64) -> f64 {
    if qc <= 1e-12 {
        return if omega > 1e-12 {
            PI / (omega * omega)
        } else {
            0.0
        };
    }
    if omega <= 1e-12 {
        // Bare Coulomb limit: integral of 4 pi / Q^2 over sphere of radius qc
        return 12.0 * PI / (qc * qc);
    }
    let u = qc / (2.0 * omega);
    if u < 1e-3 {
        // High-order Taylor series for small u = q_c / (2 omega) to avoid cancellation:
        // 1 - sqrt(pi)/(2u) erf(u) = u^2/3 - u^4/10 + u^6/42 - u^8/216 + u^10/1320
        // Multiplying by 12 pi / q_c^2 = 3 pi / (omega^2 u^2):
        // v_bar = pi / omega^2 * (1 - 3/10 u^2 + 9/70 u^4 - 1/24 u^6 + 9/440 u^8)
        let u2 = u * u;
        let u4 = u2 * u2;
        let u6 = u4 * u2;
        let u8 = u4 * u4;
        let w2 = omega * omega;
        (PI / w2) * (1.0 - 0.3 * u2 + (9.0 / 70.0) * u4 - (1.0 / 24.0) * u6 + (9.0 / 440.0) * u8)
    } else {
        let erf_val = libm::erf(u);
        let pref = 12.0 * PI / (qc * qc);
        pref * (1.0 - (omega * PI.sqrt() / qc) * erf_val)
    }
}

/// Occupied Bloch states at a given q-point for one spin channel or spinor.
struct OccupiedStates {
    weight: f64,
    // For each occupied state: [up] for collinear, [up, down] for spinor
    wavefunctions: Vec<Vec<Buffer<C64>>>,
}

/// Extracts occupied Bloch wavefunctions on the real-space grid for a k-point q.
fn extract_occupied(
    c: &Crystal,
    q_idx: usize,
    d_mat: &Matrix,
    spin_scale: f64,
) -> Result<OccupiedStates> {
    let m = c.m;
    let n = c.grid.index.points();
    let point = &c.points[q_idx];
    let spinor = c.spinor;

    // Scale density matrix by spin_scale (0.5 for closed shell, 1.0 for unrestricted or spinor)
    let d_spin = d_mat * C64::new(spin_scale, 0.);
    let (eigenvals, eigenvecs) = crate::math::linalg::eigen(d_spin)?;

    // Read basis samples at point q
    let q_samples = point.psi.read(&c.context)?;

    // Precompute phase array: exp(-i q . r)
    let mut phase = c.context.allocate::<C64>(n)?;
    for (k, val) in phase.as_mut_slice().iter_mut().enumerate() {
        let r = c.grid.point(k);
        let phase_arg = -(0..3).map(|a| point.k[a] * r[a]).sum::<f64>();
        *val = C64::from_polar(1.0, phase_arg);
    }

    let mut wavefunctions = Vec::new();
    let n_spin_components = if spinor { 2 } else { 1 };

    for (j, &val) in eigenvals.iter().enumerate() {
        if val <= 1e-10 {
            continue;
        }
        let occ_sqrt = val.sqrt();
        let mut components = Vec::with_capacity(n_spin_components);

        for comp in 0..n_spin_components {
            let mut psi = c.context.allocate::<C64>(n)?;
            for mu in 0..m {
                let coeff = eigenvecs[(comp * m + mu, j)] * C64::new(occ_sqrt, 0.);
                if coeff.norm_sqr() < 1e-24 {
                    continue;
                }
                let row = &q_samples[mu * n..(mu + 1) * n];
                for k in 0..n {
                    psi.as_mut_slice()[k] += coeff * row[k] * phase.as_slice()[k];
                }
            }
            components.push(psi);
        }
        wavefunctions.push(components);
    }

    Ok(OccupiedStates {
        weight: point.weight,
        wavefunctions,
    })
}

/// Computes the screened exact exchange matrices K_s(k) for all k-points in the crystal mesh.
/// Also returns the exact exchange energy E_xx.
pub fn exchange_operators<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    omega: f64,
    exxdiv: ExxDivTreatment,
) -> Result<(Vec<Matrix>, f64)> {
    let m = c.m;
    let n = c.grid.index.points();
    let nk = c.points.len();
    let dv = c.grid.index.dv();
    let cell_volume = dv * n as f64;
    let spinor = c.spinor;
    let n_spin_channels = if spinor { 1 } else { c.spins };
    let n_basis = if spinor { 2 * m } else { m };
    let spin_scale = if !spinor && c.spins == 1 { 0.5 } else { 1.0 };
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;

    let mut operators = Vec::with_capacity(n_spin_channels * nk);
    let mut total_exchange_energy = 0.0;

    for s in 0..n_spin_channels {
        // 1. Extract occupied wavefunctions for all q-points
        let mut all_occupied = Vec::with_capacity(nk);
        for q in 0..nk {
            let d = matrices[s * nk + q].matrix(&c.context)?;
            let occ = extract_occupied(c, q, &d, spin_scale)?;
            all_occupied.push(occ);
        }

        // 2. Loop over target k-points
        for k in 0..nk {
            let target_point = &c.points[k];
            let k_samples = target_point.psi.read(&c.context)?;

            // Compute periodic basis functions u_{mu, k}(r) = psi_{mu, k}(r) * exp(-i k . r)
            let mut u_k = c.context.allocate::<C64>(m * n)?;
            for r_idx in 0..n {
                let r = c.grid.point(r_idx);
                let phase_arg = -(0..3).map(|a| target_point.k[a] * r[a]).sum::<f64>();
                let phase = C64::from_polar(1.0, phase_arg);
                for mu in 0..m {
                    u_k.as_mut_slice()[mu * n + r_idx] = k_samples[mu * n + r_idx] * phase;
                }
            }

            let mut k_mat = Matrix::zeros(n_basis, n_basis);
            let mut pair_buf = c.context.allocate::<C64>(n)?;

            // Loop over source q-points
            for (q, occ_q) in all_occupied.iter().enumerate() {
                if occ_q.wavefunctions.is_empty() {
                    continue;
                }
                let source_point = &c.points[q];
                let k_diff = [
                    source_point.k[0] - target_point.k[0],
                    source_point.k[1] - target_point.k[1],
                    source_point.k[2] - target_point.k[2],
                ];
                let symbols = reciprocal::bloch(&c.context, &c.grid, k_diff)?;
                let w_q = occ_q.weight;

                // Precompute sqrt_kernel_weights for this (k, q) pair
                let mut sqrt_kernel_weight = c.context.allocate::<f64>(n)?;
                for at in 0..n {
                    let q2 = symbols.squares.as_slice()[at];
                    let v = if q2 > 1e-12 {
                        screened_kernel(q2, omega)
                    } else {
                        match exxdiv {
                            ExxDivTreatment::GygiBaldereschi => {
                                let qc = (6.0 * PI * PI * w_q / cell_volume).cbrt();
                                gygi_baldereschi_screened_kernel(qc, omega)
                            }
                            ExxDivTreatment::None => screened_kernel(0.0, omega),
                        }
                    };
                    let factor = (w_q * dv / n as f64) * v;
                    sqrt_kernel_weight.as_mut_slice()[at] = factor.max(0.0).sqrt();
                }

                // Loop over occupied states j at q
                for psi_j in &occ_q.wavefunctions {
                    let mut transformed_pairs = c.context.allocate::<C64>(n_basis * n)?;

                    for i in 0..n_basis {
                        let comp = i / m;
                        let mu = i % m;
                        let u_mu = &u_k.as_slice()[mu * n..(mu + 1) * n];
                        let psi_comp = psi_j[comp].as_slice();

                        for r_idx in 0..n {
                            pair_buf.as_mut_slice()[r_idx] = u_mu[r_idx].conj() * psi_comp[r_idx];
                        }
                        fft.transform(pair_buf.as_mut_slice(), false)?;
                        for at in 0..n {
                            transformed_pairs.as_mut_slice()[i * n + at] =
                                pair_buf.as_slice()[at] * sqrt_kernel_weight.as_slice()[at];
                        }
                    }

                    // Accumulate into K: K_{i, j} += sum_{at} W_i[at] * W_j[at]^*
                    for i in 0..n_basis {
                        let w_i = &transformed_pairs.as_slice()[i * n..(i + 1) * n];
                        for j in 0..=i {
                            let w_j = &transformed_pairs.as_slice()[j * n..(j + 1) * n];
                            let dot: C64 = (0..n).map(|at| w_i[at] * w_j[at].conj()).sum();
                            k_mat[(i, j)] += dot;
                            if j != i {
                                k_mat[(j, i)] += dot.conj();
                            }
                        }
                    }
                }
            }

            // Symmetrize K to ensure strict hermiticity
            let k_herm = (&k_mat + k_mat.adjoint()) * C64::new(0.5, 0.0);

            // Exact exchange energy contribution
            let d_mat = matrices[s * nk + k].matrix(&c.context)?;
            let d_spin = &*d_mat * C64::new(spin_scale, 0.);
            let tr = (d_spin * &k_herm).trace().re;
            let mult = if !spinor && c.spins == 1 { 2.0 } else { 1.0 };
            total_exchange_energy -= 0.125 * target_point.weight * tr * mult;

            operators.push(k_herm);
        }
    }

    Ok((operators, total_exchange_energy))
}

/// Evaluates the screened exact exchange matrix K(k_path) at an arbitrary off-mesh k-point
/// using converged SCF density matrices.
pub fn exchange_operator_at_k<M: MatrixSource>(
    c: &Crystal,
    matrices: &[M],
    target_point: &KPoint,
    omega: f64,
    spin: usize,
    exxdiv: ExxDivTreatment,
) -> Result<Matrix> {
    let m = c.m;
    let n = c.grid.index.points();
    let nk = c.points.len();
    let dv = c.grid.index.dv();
    let cell_volume = dv * n as f64;
    let spinor = c.spinor;
    let n_basis = if spinor { 2 * m } else { m };
    let spin_scale = if !spinor && c.spins == 1 { 0.5 } else { 1.0 };
    let mut fft = Fft3::new(&c.context, c.grid.index.shape())?;

    // 1. Extract occupied wavefunctions for all q-points
    let mut all_occupied = Vec::with_capacity(nk);
    for q in 0..nk {
        let d = matrices[spin * nk + q].matrix(&c.context)?;
        let occ = extract_occupied(c, q, &d, spin_scale)?;
        all_occupied.push(occ);
    }

    // 2. Periodic basis functions u_{mu, k}(r) at target_point
    let k_samples = target_point.psi.read(&c.context)?;
    let mut u_k = c.context.allocate::<C64>(m * n)?;
    for r_idx in 0..n {
        let r = c.grid.point(r_idx);
        let phase_arg = -(0..3).map(|a| target_point.k[a] * r[a]).sum::<f64>();
        let phase = C64::from_polar(1.0, phase_arg);
        for mu in 0..m {
            u_k.as_mut_slice()[mu * n + r_idx] = k_samples[mu * n + r_idx] * phase;
        }
    }

    let mut k_mat = Matrix::zeros(n_basis, n_basis);
    let mut pair_buf = c.context.allocate::<C64>(n)?;

    for (q, occ_q) in all_occupied.iter().enumerate() {
        if occ_q.wavefunctions.is_empty() {
            continue;
        }
        let source_point = &c.points[q];
        let k_diff = [
            source_point.k[0] - target_point.k[0],
            source_point.k[1] - target_point.k[1],
            source_point.k[2] - target_point.k[2],
        ];
        let symbols = reciprocal::bloch(&c.context, &c.grid, k_diff)?;
        let w_q = occ_q.weight;

        let mut sqrt_kernel_weight = c.context.allocate::<f64>(n)?;
        for at in 0..n {
            let q2 = symbols.squares.as_slice()[at];
            let v = if q2 > 1e-12 {
                screened_kernel(q2, omega)
            } else {
                match exxdiv {
                    ExxDivTreatment::GygiBaldereschi => {
                        let qc = (6.0 * PI * PI * w_q / cell_volume).cbrt();
                        gygi_baldereschi_screened_kernel(qc, omega)
                    }
                    ExxDivTreatment::None => screened_kernel(0.0, omega),
                }
            };
            let factor = (w_q * dv / n as f64) * v;
            sqrt_kernel_weight.as_mut_slice()[at] = factor.max(0.0).sqrt();
        }

        for psi_j in &occ_q.wavefunctions {
            let mut transformed_pairs = c.context.allocate::<C64>(n_basis * n)?;

            for i in 0..n_basis {
                let comp = i / m;
                let mu = i % m;
                let u_mu = &u_k.as_slice()[mu * n..(mu + 1) * n];
                let psi_comp = psi_j[comp].as_slice();

                for r_idx in 0..n {
                    pair_buf.as_mut_slice()[r_idx] = u_mu[r_idx].conj() * psi_comp[r_idx];
                }
                fft.transform(pair_buf.as_mut_slice(), false)?;
                for at in 0..n {
                    transformed_pairs.as_mut_slice()[i * n + at] =
                        pair_buf.as_slice()[at] * sqrt_kernel_weight.as_slice()[at];
                }
            }

            for i in 0..n_basis {
                let w_i = &transformed_pairs.as_slice()[i * n..(i + 1) * n];
                for j in 0..=i {
                    let w_j = &transformed_pairs.as_slice()[j * n..(j + 1) * n];
                    let dot: C64 = (0..n).map(|at| w_i[at] * w_j[at].conj()).sum();
                    k_mat[(i, j)] += dot;
                    if j != i {
                        k_mat[(j, i)] += dot.conj();
                    }
                }
            }
        }
    }

    Ok((&k_mat + k_mat.adjoint()) * C64::new(0.5, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gygi_baldereschi_limits_and_consistency() {
        let omega = 0.11;
        // As qc -> 0, it must converge to pi / omega^2
        let at_origin = PI / (omega * omega);
        let val_small = gygi_baldereschi_screened_kernel(1e-6, omega);
        assert!((val_small - at_origin).abs() < 1e-8);

        // Bare Coulomb limit (omega -> 0): 12 pi / qc^2
        let qc = 0.35;
        let bare = gygi_baldereschi_screened_kernel(qc, 0.0);
        let expected_bare = 12.0 * PI / (qc * qc);
        assert!((bare - expected_bare).abs() < 1e-12);

        // Consistency at Taylor transition boundary u = 1e-3
        let u_trans = 1e-3;
        let qc_trans = 2.0 * omega * u_trans;
        let val_below = gygi_baldereschi_screened_kernel(qc_trans * 0.999, omega);
        let val_above = gygi_baldereschi_screened_kernel(qc_trans * 1.001, omega);
        assert!((val_below - val_above).abs() < 1e-6);

        // Parsing tests
        assert_eq!(ExxDivTreatment::parse("none").unwrap(), ExxDivTreatment::None);
        assert_eq!(ExxDivTreatment::parse("gb").unwrap(), ExxDivTreatment::GygiBaldereschi);
        assert_eq!(ExxDivTreatment::parse("gygi-baldereschi").unwrap(), ExxDivTreatment::GygiBaldereschi);
        assert_eq!(ExxDivTreatment::parse("gygi_baldereschi").unwrap(), ExxDivTreatment::GygiBaldereschi);
        assert!(ExxDivTreatment::parse("unknown_mode").is_err());
    }
}
