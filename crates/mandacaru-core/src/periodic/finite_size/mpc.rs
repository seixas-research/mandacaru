//! Model Periodic Coulomb (MPC) finite-size correction.
//!
//! Fraser et al., Phys. Rev. B 53, 1814 (1996);
//! Williamson et al., Phys. Rev. B 55, R4851 (1997).
//!
//! MPC corrects the exchange-correlation hole image error in supercell electronic structure
//! calculations. In standard Ewald periodic boundary conditions, each electron interacts with
//! its own periodic images and the periodic images of its exchange-correlation hole. MPC replaces
//! the periodic Coulomb interaction $g^E$ with a model interaction $f$ truncated at the boundary
//! of the Wigner-Seitz cell:
//!
//! $$ \Delta E = \frac{1}{2} \sum_{pqrs} (f_{pqrs} - g^E_{pqrs}) (\Gamma_{pqrs} - D_{pr} D_{qs}) $$
//!
//! # Double-Counting Correction
//!
//! When applied to an energy containing the Ewald Madelung term $\frac{1}{2} N_e v_M$, the raw
//! difference $\Delta E_{\text{hole}} = E_{\text{hole}}^{\text{model}} - E_{\text{hole}}^{\text{periodic}}$
//! double-counts exactly half the Madelung term because the periodic hole energy omits the hole's
//! own self-image. The corrected supercell shift is:
//!
//! $$ \Delta E_{\text{MPC}} = \Delta E_{\text{hole}} - \frac{1}{2} N_e v_M $$

use super::Scope;
use crate::periodic::invalid;
use crate::Result;
use num_complex::Complex64 as C64;

/// Result of an MPC finite-size calculation. Energies in Hartree per supercell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mpc {
    /// Corrected supercell shift: `hole_model - hole_periodic - 0.5 * n_electrons * v_madelung`.
    pub correction: f64,
    /// Exchange-correlation hole energy with the Wigner-Seitz model interaction.
    pub hole_model: f64,
    /// Exchange-correlation hole energy with the periodic Ewald interaction.
    pub hole_periodic: f64,
    /// Shift subtracting the Madelung double-counting term: `-0.5 * n_electrons * v_madelung`.
    pub madelung_shift: f64,
    /// Madelung constant $v_M$ of the supercell, Hartree.
    pub v_madelung: f64,
    /// Number of electrons in the supercell.
    pub n_electrons: f64,
    /// The supercell scope.
    pub scope: Scope,
    /// Raw difference `hole_model - hole_periodic` (retains the double-counting for diagnostic comparison).
    pub raw_correction: f64,
}

impl Mpc {
    /// The MPC correction per primitive cell.
    pub fn correction_per_cell(&self) -> f64 {
        self.scope.per_cell(self.correction)
    }

    /// Add the per-cell correction to a variational energy per primitive cell.
    pub fn corrected_energy_per_cell(&self, energy_per_cell: f64) -> f64 {
        energy_per_cell + self.correction_per_cell()
    }
}

/// Compute the MPC correction from computed hole energies and the supercell Madelung constant.
///
/// Validates that all inputs are finite and geometrically consistent, and applies the
/// $\frac{1}{2} N_e v_M$ double-counting correction.
pub fn mpc_from_hole_energies(
    hole_model: f64,
    hole_periodic: f64,
    v_madelung: f64,
    n_electrons: f64,
    n_cells: usize,
    volume: f64,
) -> Result<Mpc> {
    if !hole_model.is_finite() || !hole_periodic.is_finite() {
        return Err(invalid(format!(
            "hole energies must be finite, got model={hole_model}, periodic={hole_periodic}"
        )));
    }
    if !v_madelung.is_finite() {
        return Err(invalid(format!(
            "Madelung constant must be finite, got {v_madelung}"
        )));
    }
    if !n_electrons.is_finite() || n_electrons < 0.0 {
        return Err(invalid(format!(
            "number of electrons must be non-negative and finite, got {n_electrons}"
        )));
    }
    let scope = Scope::new(n_cells, volume)?;
    let raw_correction = hole_model - hole_periodic;
    let madelung_shift = -0.5 * n_electrons * v_madelung;
    let correction = raw_correction + madelung_shift;

    Ok(Mpc {
        correction,
        hole_model,
        hole_periodic,
        madelung_shift,
        v_madelung,
        n_electrons,
        scope,
        raw_correction,
    })
}

/// Verify the hole trace identity $\sum_q (\Gamma_{pqrq} - D_{pr} D_{qq}) = -D_{pr}$ (test T5).
///
/// Returns the maximum absolute residual over orbital indices $p, r$.
pub fn verify_hole_trace(gamma: &[f64], d: &[f64], m: usize) -> Result<f64> {
    if d.len() != m * m {
        return Err(invalid(format!(
            "1-RDM length must be m^2 = {}, got {}",
            m * m,
            d.len()
        )));
    }
    if gamma.len() != m * m * m * m {
        return Err(invalid(format!(
            "2-RDM length must be m^4 = {}, got {}",
            m * m * m * m,
            gamma.len()
        )));
    }

    let mut worst = 0.0_f64;
    for p in 0..m {
        for r in 0..m {
            let mut sum = 0.0_f64;
            for q in 0..m {
                let gamma_pqrq = gamma[((p * m + q) * m + r) * m + q];
                let d_qq = d[q * m + q];
                let d_pr = d[p * m + r];
                sum += gamma_pqrq - d_pr * d_qq;
            }
            let want = -d[p * m + r];
            let res = (sum - want).abs();
            if res > worst {
                worst = res;
            }
        }
    }
    Ok(worst)
}

/// Node-batched reciprocal-space contraction for MPC hole interaction:
///
/// $$ \Delta E_{\text{hole}} = \frac{1}{2} \sum_{\mathbf{k}} w(\mathbf{k}) \sum_{pqrs}
///    (\Gamma_{pqrs} - D_{pr} D_{qs}) \rho_{pq}(\mathbf{k}) \rho_{rs}(-\mathbf{k}) $$
///
/// where $w(\mathbf{k}) = \tilde{f}(\mathbf{k}) - \tilde{g}^E(\mathbf{k})$ is the difference
/// of Fourier-transformed interaction kernels, and $\rho_{pq}(\mathbf{k})$ is the orbital pair
/// transform matrix at reciprocal wave vector $\mathbf{k}$.
///
/// `pair_transforms` has shape `(n_k, m, m)` in row-major order.
pub fn mpc_hole_contraction(
    weights: &[f64],
    pair_transforms: &[C64],
    m: usize,
    gamma: &[f64],
    d: &[f64],
) -> Result<f64> {
    let n_k = weights.len();
    if pair_transforms.len() != n_k * m * m {
        return Err(invalid(format!(
            "pair transforms must have length n_k * m * m = {}, got {}",
            n_k * m * m,
            pair_transforms.len()
        )));
    }
    if d.len() != m * m {
        return Err(invalid(format!(
            "1-RDM length must be m^2 = {}, got {}",
            m * m,
            d.len()
        )));
    }
    if gamma.len() != m * m * m * m {
        return Err(invalid(format!(
            "2-RDM length must be m^4 = {}, got {}",
            m * m * m * m,
            gamma.len()
        )));
    }

    let m2 = m * m;
    let mut total = 0.0_f64;

    // Build the hole tensor H_{pqrs} = Gamma_{pqrs} - D_{pr} D_{qs}
    // and contract per reciprocal node k
    for (k_idx, &wk) in weights.iter().enumerate() {
        if wk == 0.0 {
            continue;
        }
        let rho_k = &pair_transforms[k_idx * m2..(k_idx + 1) * m2];

        // M_{pr} = sum_q rho_pq(k) D_{pr}
        // Scalar contraction: sum_{pqrs} H_{pqrs} rho_{pq}(k) rho_{rs}(-k)
        // With rho_{rs}(-k) = rho_{sr}(k)* for real orbitals
        let mut k_sum = 0.0_f64;
        for p in 0..m {
            for q in 0..m {
                let rho_pq = rho_k[p * m + q];
                if rho_pq.norm_sqr() == 0.0 {
                    continue;
                }
                for r in 0..m {
                    for s in 0..m {
                        let rho_rs_conj = rho_k[r * m + s].conj();
                        let h_pqrs = gamma[((p * m + q) * m + r) * m + s] - d[p * m + r] * d[q * m + s];
                        k_sum += h_pqrs * (rho_pq * rho_rs_conj).re;
                    }
                }
            }
        }
        total += wk * k_sum;
    }

    Ok(0.5 * total)
}
