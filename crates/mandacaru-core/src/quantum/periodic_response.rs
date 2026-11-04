//! Finite-difference and analytical forces and stresses for periodic quantum (VQE / ADAPT-VQE) methods.
use super::C64;
use crate::{Error, Result};

fn invalid(msg: &str) -> Error {
    Error::Invalid(msg.into())
}

/// The Voigt indices (a, b) corresponding to the standard 6 Voigt components:
/// [xx, yy, zz, yz, xz, xy].
pub const VOIGT_INDICES: [(usize, usize); 6] = [
    (0, 0), // 0: xx
    (1, 1), // 1: yy
    (2, 2), // 2: zz
    (1, 2), // 3: yz
    (0, 2), // 4: xz
    (0, 1), // 5: xy
];

/// Results of a periodic quantum stress calculation.
#[derive(Clone, Debug)]
pub struct QuantumStress {
    /// Cartesian 3x3 stress tensor in Hartree/Bohr^3: sigma_ab.
    pub sigma: [[f64; 3]; 3],
    /// 6-component Voigt stress: [xx, yy, zz, yz, xz, xy].
    pub voigt: [f64; 6],
    /// Cell volume in Bohr^3.
    pub volume: f64,
    /// Calculation scheme ("finite-difference" or "analytic").
    pub scheme: &'static str,
}

/// Results of a periodic quantum forces calculation.
#[derive(Clone, Debug)]
pub struct QuantumForces {
    /// Force per atom in Hartree/Bohr: F_ia.
    pub forces: Vec<[f64; 3]>,
    /// Calculation scheme ("finite-difference" or "analytic").
    pub scheme: &'static str,
}

/// Computes periodic quantum forces using the central finite difference algorithm.
///
/// `energies_plus[atom][axis]` is E(R_{atom, axis} + h),
/// `energies_minus[atom][axis]` is E(R_{atom, axis} - h).
///
/// F_{atom, axis} = - (E(+) - E(-)) / (2 * h).
pub fn periodic_quantum_forces_fd(
    energies_plus: &[[f64; 3]],
    energies_minus: &[[f64; 3]],
    h: f64,
) -> Result<QuantumForces> {
    if h <= 0. || !h.is_finite() {
        return Err(invalid("displacement step h must be positive and finite"));
    }
    if energies_plus.len() != energies_minus.len() {
        return Err(invalid("energies_plus and energies_minus must have same atom count"));
    }
    let n_atoms = energies_plus.len();
    let mut forces = Vec::with_capacity(n_atoms);
    for a in 0..n_atoms {
        let mut f_atom = [0.0; 3];
        for axis in 0..3 {
            let ep = energies_plus[a][axis];
            let em = energies_minus[a][axis];
            if !ep.is_finite() || !em.is_finite() {
                return Err(invalid("non-finite energy encountered in finite difference force"));
            }
            f_atom[axis] = -(ep - em) / (2.0 * h);
        }
        forces.push(f_atom);
    }
    Ok(QuantumForces {
        forces,
        scheme: "Central-finite-difference",
    })
}

/// Computes periodic quantum stress using the symmetric strain finite difference algorithm.
///
/// `strained_plus[v]` is E(strain = +eps * e_v),
/// `strained_minus[v]` is E(strain = -eps * e_v),
/// for the 6 Voigt strain components v = 0..5 (xx, yy, zz, yz, xz, xy).
///
/// sigma_v = (E(+) - E(-)) / (2 * eps * volume).
pub fn periodic_quantum_stress_fd(
    strained_plus: &[f64; 6],
    strained_minus: &[f64; 6],
    eps: f64,
    volume: f64,
) -> Result<QuantumStress> {
    if eps <= 0. || !eps.is_finite() {
        return Err(invalid("strain step eps must be positive and finite"));
    }
    if volume <= 0. || !volume.is_finite() {
        return Err(invalid("cell volume must be positive and finite"));
    }
    let mut voigt = [0.0; 6];
    let mut sigma = [[0.0; 3]; 3];

    for v in 0..6 {
        let ep = strained_plus[v];
        let em = strained_minus[v];
        if !ep.is_finite() || !em.is_finite() {
            return Err(invalid("non-finite energy encountered in finite difference stress"));
        }
        let s = (ep - em) / (2.0 * eps * volume);
        voigt[v] = s;
        let (a, b) = VOIGT_INDICES[v];
        sigma[a][b] = s;
        sigma[b][a] = s;
    }

    Ok(QuantumStress {
        sigma,
        voigt,
        volume,
        scheme: "Central-finite-strain",
    })
}

/// Computes analytic periodic quantum forces by contracting state 1- and 2-RDMs
/// with periodic Hamiltonian integral derivatives and overlap Pulay derivatives,
/// plus the analytic periodic Ewald ion-ion force.
///
/// Handles all spin polarization cases:
/// - Unpolarized (rdm1 is m x m spatial, rdm2 is m x m x m x m spatial)
/// - Collinear spin (rdm1 is 2m x 2m spin-orbital)
/// - Spin-orbit coupling (rdm1 is 2m x 2m complex spinor)
#[allow(clippy::too_many_arguments)]
pub fn periodic_quantum_forces_analytic(
    n_atoms: usize,
    m: usize,
    rdm1: &[C64],
    rdm2: Option<&[C64]>,
    dh_atomic: &[Vec<[f64; 3]>],      // [atom][p * m + q] -> [x, y, z]
    dg_atomic: Option<&[Vec<[f64; 3]>]>, // [atom][((p * m + q) * m + r) * m + s] -> [x, y, z]
    ds_atomic: Option<&[Vec<[f64; 3]>]>, // [atom][p * m + q] -> [x, y, z]
    w_matrix: Option<&[C64]>,         // Energy-weighted density matrix [p * m + q]
    ewald_forces: &[[f64; 3]],        // Analytic Ewald force per atom
) -> Result<QuantumForces> {
    if rdm1.len() != m * m && rdm1.len() != 4 * m * m {
        return Err(invalid("1-RDM dimensions do not match orbital space"));
    }
    if dh_atomic.len() != n_atoms || ewald_forces.len() != n_atoms {
        return Err(invalid("atom count mismatch in analytic force contraction"));
    }

    let is_spin_orbital = rdm1.len() == 4 * m * m;
    let mut forces = Vec::with_capacity(n_atoms);

    for a in 0..n_atoms {
        let mut f = [0.0; 3];
        // 1. One-body integral derivative: - tr[D * dh]
        let dh = &dh_atomic[a];
        if dh.len() != m * m {
            return Err(invalid("dh dimensions must match m x m"));
        }

        for p in 0..m {
            for q in 0..m {
                let d_val = if is_spin_orbital {
                    // Sum over alpha and beta blocks: D_{p,q}^{aa} + D_{p,q}^{bb}
                    rdm1[p * (2 * m) + q] + rdm1[(m + p) * (2 * m) + (m + q)]
                } else {
                    rdm1[p * m + q]
                };
                let deriv = dh[p * m + q];
                for axis in 0..3 {
                    f[axis] -= d_val.re * deriv[axis];
                }
            }
        }

        // 2. Two-body integral derivative: - 0.5 * tr[Gamma * dg]
        if let (Some(g2), Some(dg_list)) = (rdm2, dg_atomic) {
            let dg = &dg_list[a];
            if dg.len() == m * m * m * m && g2.len() >= m * m * m * m {
                for p in 0..m {
                    for q in 0..m {
                        for r in 0..m {
                            for s in 0..m {
                                let idx = ((p * m + q) * m + r) * m + s;
                                let gamma_val = g2[idx];
                                let deriv = dg[idx];
                                for axis in 0..3 {
                                    f[axis] -= 0.5 * gamma_val.re * deriv[axis];
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Overlap Pulay term: + tr[W * dS]
        if let (Some(w), Some(ds_list)) = (w_matrix, ds_atomic) {
            let ds = &ds_list[a];
            if ds.len() == m * m && w.len() >= m * m {
                for p in 0..m {
                    for q in 0..m {
                        let w_val = w[p * m + q];
                        let deriv = ds[p * m + q];
                        for axis in 0..3 {
                            f[axis] += w_val.re * deriv[axis];
                        }
                    }
                }
            }
        }

        // 4. Analytic Ewald ion-ion force
        for axis in 0..3 {
            f[axis] += ewald_forces[a][axis];
        }

        forces.push(f);
    }

    Ok(QuantumForces {
        forces,
        scheme: "Analytic-RDM-contraction",
    })
}

/// Computes analytic periodic quantum stress by contracting state 1- and 2-RDMs
/// with periodic strain derivatives of one-body and two-body integrals, overlap
/// Pulay strain, and the analytic Ewald stress tensor.
///
/// Handles all spin polarization cases:
/// - Unpolarized (spatial RDMs)
/// - Collinear spin (spin-orbital RDMs)
/// - Spin-orbit coupling (complex spinor RDMs)
#[allow(clippy::too_many_arguments)]
pub fn periodic_quantum_stress_analytic(
    m: usize,
    volume: f64,
    rdm1: &[C64],
    rdm2: Option<&[C64]>,
    electronic_energy: f64,
    dh_strain: &[[Vec<f64>; 3]; 3],        // [a][b] -> Vec of size m * m
    dg_strain: Option<&[[Vec<f64>; 3]; 3]>,// [a][b] -> Vec of size m^4
    ds_strain: Option<&[[Vec<f64>; 3]; 3]>,// [a][b] -> Vec of size m * m
    w_matrix: Option<&[C64]>,              // Energy-weighted density [p * m + q]
    ewald_stress: &[[f64; 3]; 3],          // Analytic Ewald stress in Hartree/Bohr^3
) -> Result<QuantumStress> {
    if volume <= 0. || !volume.is_finite() {
        return Err(invalid("cell volume must be positive and finite"));
    }
    if rdm1.len() != m * m && rdm1.len() != 4 * m * m {
        return Err(invalid("1-RDM dimensions do not match orbital space"));
    }

    let is_spin_orbital = rdm1.len() == 4 * m * m;
    let mut sigma = [[0.0; 3]; 3];
    let mut voigt = [0.0; 6];

    for a in 0..3 {
        for b in 0..3 {
            let mut val = 0.0;

            // 1. One-body strain derivative: tr[D * dh_strain]
            let dh = &dh_strain[a][b];
            if dh.len() == m * m {
                for p in 0..m {
                    for q in 0..m {
                        let d_val = if is_spin_orbital {
                            rdm1[p * (2 * m) + q] + rdm1[(m + p) * (2 * m) + (m + q)]
                        } else {
                            rdm1[p * m + q]
                        };
                        val += d_val.re * dh[p * m + q];
                    }
                }
            }

            // 2. Two-body strain derivative: 0.5 * tr[Gamma * dg_strain]
            if let (Some(g2), Some(dg_s)) = (rdm2, dg_strain) {
                let dg = &dg_s[a][b];
                if dg.len() == m * m * m * m && g2.len() >= m * m * m * m {
                    for p in 0..m {
                        for q in 0..m {
                            for r in 0..m {
                                for s in 0..m {
                                    let idx = ((p * m + q) * m + r) * m + s;
                                    val += 0.5 * g2[idx].re * dg[idx];
                                }
                            }
                        }
                    }
                }
            }

            // 3. Overlap Pulay strain derivative: - tr[W * dS_strain]
            if let (Some(w), Some(ds_s)) = (w_matrix, ds_strain) {
                let ds = &ds_s[a][b];
                if ds.len() == m * m && w.len() >= m * m {
                    for p in 0..m {
                        for q in 0..m {
                            val -= w[p * m + q].re * ds[p * m + q];
                        }
                    }
                }
            }

            // Electronic derivative term per unit volume
            let elec_stress = val / volume;

            // Kinetic/volume trace term: + delta_ab * E_elec / volume
            let trace_term = if a == b { electronic_energy / volume } else { 0.0 };

            // Total stress = Electronic + Trace + Ewald
            sigma[a][b] = elec_stress + trace_term + ewald_stress[a][b];
        }
    }

    // Symmetrize sigma
    for a in 0..3 {
        for b in (a + 1)..3 {
            let sym = 0.5 * (sigma[a][b] + sigma[b][a]);
            sigma[a][b] = sym;
            sigma[b][a] = sym;
        }
    }

    for (v, &(a, b)) in VOIGT_INDICES.iter().enumerate() {
        voigt[v] = sigma[a][b];
    }

    Ok(QuantumStress {
        sigma,
        voigt,
        volume,
        scheme: "Analytic-RDM-contraction",
    })
}
