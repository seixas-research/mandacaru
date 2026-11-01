//! Finite-size correction from the structure factor (Chiesa-Ceperley-Martin-Holzmann).
//!
//! Chiesa, Ceperley, Martin, and Holzmann, Phys. Rev. Lett. 97, 076404 (2006).
//!
//! In periodic boundary conditions, the electron-electron interaction energy is evaluated as a
//! discrete sum over the reciprocal lattice of the supercell:
//!
//! $$ V_N = \frac{1}{2\Omega} \sum_{\mathbf{k} \neq 0} v(\mathbf{k}) [S(\mathbf{k}) - 1], \qquad
//!    v(\mathbf{k}) = \frac{4\pi}{k^2} $$
//!
//! In the thermodynamic limit this sum becomes an integral. The missing term at $\mathbf{k} = 0$ is:
//!
//! $$ \Delta V = \frac{1}{2\Omega} \lim_{\mathbf{k}\to 0} [v(\mathbf{k}) S(\mathbf{k})]
//!             = \frac{2\pi}{\Omega} \lim_{\mathbf{k}\to 0} \frac{S(\mathbf{k})}{k^2} $$
//!
//! The limit $\lim_{k\to 0} S(k)/k^2$ is finite because long-wavelength density fluctuations in a
//! Coulomb system are suppressed ($S(k) \propto k^2$ at small $k$).
//!
//! # Corrections to the Pre-Port Implementation
//!
//! 1. **Supercell Total vs Per-Electron**: The omitted potential energy per electron is $\Delta V$.
//!    The total supercell correction is $N_e \Delta V$ (the pre-port code reported $\Delta V$ directly
//!    as the supercell total, missing the factor of $N_e$).
//! 2. **Connected Structure Factor**: For neutral or localized systems, the physical density
//!    fluctuations are captured by the connected structure factor $S_c(\mathbf{k}) = S(\mathbf{k})
//!    - \frac{1}{N_e} |\langle \hat{\rho}_\mathbf{k} \rangle|^2$, which strictly vanishes at $k = 0$.
//! 3. **RDM Index Ordering**: Under native 2-RDM $\Gamma_{pqrs} = \langle a_p^\dagger a_q^\dagger a_s a_r \rangle$,
//!    the pair contraction with symmetric orbital transforms $M_{pq} = M_{qp}$ is $\Gamma_{prqs}$.

use super::Scope;
use crate::periodic::invalid;
use crate::Result;
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Maximum allowed fractional spread in $S(k)/k^2$ across fitted shells for the extrapolation to be
/// deemed in the quadratic regime.
pub const QUADRATIC_REGIME_SPREAD: f64 = 0.5;

/// Default number of reciprocal shells to use for the quadratic extrapolation.
pub const DEFAULT_SHELLS: usize = 3;

/// A shell of reciprocal wave vectors with common magnitude $|k|$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CcmhShell {
    /// Magnitude $|k|$ in Bohr$^{-1}$.
    pub k: f64,
    /// Average structure factor $S(k)$ on this shell.
    pub s: f64,
    /// Number of reciprocal-lattice vectors in this shell.
    pub count: usize,
}

/// A computed or synthetic structure factor evaluated on reciprocal vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureFactor {
    /// Reciprocal wave vectors $\mathbf{k}$, Bohr$^{-1}$.
    pub kvectors: Vec<[f64; 3]>,
    /// Magnitudes $|k|$ of the wave vectors, Bohr$^{-1}$.
    pub magnitudes: Vec<f64>,
    /// Values $S(\mathbf{k})$ of the structure factor at each wave vector.
    pub values: Vec<f64>,
    /// Number of electrons in the supercell.
    pub n_electrons: f64,
}

impl StructureFactor {
    /// Create a new `StructureFactor` and validate lengths.
    pub fn new(
        kvectors: Vec<[f64; 3]>,
        values: Vec<f64>,
        n_electrons: f64,
    ) -> Result<Self> {
        if kvectors.len() != values.len() {
            return Err(invalid(format!(
                "kvectors count ({}) must match values count ({})",
                kvectors.len(),
                values.len()
            )));
        }
        if !n_electrons.is_finite() || n_electrons <= 0.0 {
            return Err(invalid(format!(
                "n_electrons must be positive and finite, got {n_electrons}"
            )));
        }
        let magnitudes: Vec<f64> = kvectors
            .iter()
            .map(|k| (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt())
            .collect();
        Ok(Self {
            kvectors,
            magnitudes,
            values,
            n_electrons,
        })
    }

    /// Group reciprocal wave vectors into shells of common magnitude $|k|$.
    ///
    /// Magnitudes within `tolerance` (default $10^{-5}$ Bohr$^{-1}$) belong to the same shell.
    /// Returns up to `max_shells` shells sorted by increasing $|k|$.
    pub fn shells(&self, max_shells: usize) -> Vec<CcmhShell> {
        if self.magnitudes.is_empty() || max_shells == 0 {
            return Vec::new();
        }

        // Pair up (magnitude, value) and sort by magnitude
        let mut pairs: Vec<(f64, f64)> = self
            .magnitudes
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .filter(|(k, _)| *k > 1e-10) // omit k = 0 if present
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut shells = Vec::new();
        let tol = 1e-5;

        let mut current_k_sum = 0.0_f64;
        let mut current_s_sum = 0.0_f64;
        let mut current_count = 0usize;
        let mut current_k_ref = pairs[0].0;

        for (k, s) in pairs {
            if (k - current_k_ref).abs() > tol {
                shells.push(CcmhShell {
                    k: current_k_sum / current_count as f64,
                    s: current_s_sum / current_count as f64,
                    count: current_count,
                });
                if shells.len() >= max_shells {
                    return shells;
                }
                current_k_ref = k;
                current_k_sum = 0.0;
                current_s_sum = 0.0;
                current_count = 0;
            }
            current_k_sum += k;
            current_s_sum += s;
            current_count += 1;
        }

        if current_count > 0 && shells.len() < max_shells {
            shells.push(CcmhShell {
                k: current_k_sum / current_count as f64,
                s: current_s_sum / current_count as f64,
                count: current_count,
            });
        }

        shells
    }
}

/// The Chiesa-Ceperley-Martin-Holzmann (CCMH) finite-size correction. Energies in Hartree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ccmh {
    /// Corrected supercell total: $N_e \Delta V = \frac{2\pi N_e \times \text{limit}}{\Omega}$.
    pub correction: f64,
    /// Omitted potential energy per electron: $\Delta V = \frac{2\pi \times \text{limit}}{\Omega}$.
    pub correction_per_electron: f64,
    /// Extrapolated limit $\lim_{k\to 0} S(k)/k^2$, Bohr$^2$.
    pub limit: f64,
    /// Fractional variation of $S(k)/k^2$ across the fitted shells.
    pub quadratic_spread: f64,
    /// Whether `quadratic_spread <= QUADRATIC_REGIME_SPREAD`.
    pub reliable: bool,
    /// Random-phase approximation (RPA) limit $1 / (2\omega_p)$.
    pub rpa_limit: f64,
    /// Supercell plasma frequency $\omega_p = \sqrt{4\pi N_e / \Omega}$, Hartree.
    pub plasma_frequency: f64,
    /// Supercell electron density $n_e = N_e / \Omega$, Bohr$^{-3}$.
    pub density: f64,
    /// Number of electrons in the supercell.
    pub n_electrons: f64,
    /// The supercell scope.
    pub scope: Scope,
}

impl Ccmh {
    /// The CCMH correction per primitive cell.
    pub fn correction_per_cell(&self) -> f64 {
        self.scope.per_cell(self.correction)
    }

    /// Add the per-cell correction to a variational energy per primitive cell.
    pub fn corrected_energy_per_cell(&self, energy_per_cell: f64) -> f64 {
        energy_per_cell + self.correction_per_cell()
    }
}

/// Extrapolate $\lim_{k\to 0} S(k)/k^2$ from reciprocal shells by linear regression in $k^2$.
///
/// If only 1 shell is supplied, the limit is that shell's ratio $S(k)/k^2$ and the quadratic spread is 0.
/// For $\ge 2$ shells, a linear least-squares fit $S(k)/k^2 = c + m k^2$ gives intercept $c = \text{limit}$.
pub fn ccmh_extrapolate(
    shells: &[CcmhShell],
    volume: f64,
    n_cells: usize,
    n_electrons: f64,
) -> Result<Ccmh> {
    if shells.is_empty() {
        return Err(invalid("no reciprocal shells to extrapolate CCMH from"));
    }
    if !n_electrons.is_finite() || n_electrons <= 0.0 {
        return Err(invalid(format!(
            "number of electrons must be positive and finite, got {n_electrons}"
        )));
    }
    let scope = Scope::new(n_cells, volume)?;

    let mut ks = Vec::with_capacity(shells.len());
    let mut ratios = Vec::with_capacity(shells.len());
    for s in shells {
        if !s.k.is_finite() || s.k <= 0.0 || !s.s.is_finite() {
            return Err(invalid(format!("invalid reciprocal shell k={}, s={}", s.k, s.s)));
        }
        ks.push(s.k);
        ratios.push(s.s / (s.k * s.k));
    }

    let (limit, spread) = if shells.len() == 1 {
        (ratios[0], 0.0)
    } else {
        // Linear fit y = m * x + c with y = ratio, x = k^2
        let n = ratios.len() as f64;
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let (mut min_r, mut max_r) = (f64::INFINITY, f64::NEG_INFINITY);

        for (&k, &r) in ks.iter().zip(ratios.iter()) {
            let x = k * k;
            sum_x += x;
            sum_y += r;
            sum_xx += x * x;
            sum_xy += x * r;
            min_r = min_r.min(r);
            max_r = max_r.max(r);
        }

        let denom = n * sum_xx - sum_x * sum_x;
        let intercept = if denom.abs() < 1e-14 {
            sum_y / n
        } else {
            (sum_y * sum_xx - sum_x * sum_xy) / denom
        };

        let mean_r = sum_y / n;
        let spread = if mean_r.abs() > 0.0 {
            (max_r - min_r) / mean_r.abs()
        } else {
            0.0
        };

        (intercept, spread)
    };

    let density = n_electrons / volume;
    let omega_p = (4.0 * PI * density).sqrt();
    let rpa_limit = 1.0 / (2.0 * omega_p);

    let correction_per_electron = 2.0 * PI * limit / volume;
    let correction = n_electrons * correction_per_electron;
    let reliable = spread <= QUADRATIC_REGIME_SPREAD;

    Ok(Ccmh {
        correction,
        correction_per_electron,
        limit,
        quadratic_spread: spread,
        reliable,
        rpa_limit,
        plasma_frequency: omega_p,
        density,
        n_electrons,
        scope,
    })
}

/// Compute the structure factor $S(\mathbf{k})$ from orbital pair transforms and RDMs.
///
/// For real orbitals where $M_{pq}(\mathbf{k}) = M_{qp}(\mathbf{k})$, the pair term uses
/// the native 2-RDM tensor index $\Gamma_{prqs}$ and exchange term $D_{sp}$:
///
/// $$ N_e S(\mathbf{k}) = \sum_{pqrs} M_{pq}(\mathbf{k}) M^*_{rs}(\mathbf{k}) \Gamma_{prqs}
///                     + \sum_{pqs} M_{pq}(\mathbf{k}) M^*_{qs}(\mathbf{k}) D_{ps} $$
///
/// If `connected` is true, subtracts $|\langle \hat{\rho}_\mathbf{k} \rangle|^2 / N_e = |\text{Tr}(M D)|^2 / N_e$.
pub fn structure_factor_from_rdms(
    k_matrices: &[C64],
    m: usize,
    gamma: &[f64],
    d: &[f64],
    n_electrons: f64,
    connected: bool,
) -> Result<Vec<f64>> {
    let m2 = m * m;
    if k_matrices.len() % m2 != 0 {
        return Err(invalid("k_matrices length must be a multiple of m^2"));
    }
    let n_k = k_matrices.len() / m2;
    if d.len() != m2 {
        return Err(invalid(format!("d length must be m^2 = {m2}, got {}", d.len())));
    }
    if gamma.len() != m2 * m2 {
        return Err(invalid(format!("gamma length must be m^4 = {}, got {}", m2 * m2, gamma.len())));
    }
    if !n_electrons.is_finite() || n_electrons <= 0.0 {
        return Err(invalid("n_electrons must be positive"));
    }

    let mut s_values = Vec::with_capacity(n_k);

    for k_idx in 0..n_k {
        let mk = &k_matrices[k_idx * m2..(k_idx + 1) * m2];

        // pair = sum M_pq conj(M_rs) Gamma_prqs
        let mut pair_sum = 0.0_f64;
        for p in 0..m {
            for q in 0..m {
                let mpq = mk[p * m + q];
                if mpq.norm_sqr() == 0.0 {
                    continue;
                }
                for r in 0..m {
                    for s in 0..m {
                        let mrs_conj = mk[r * m + s].conj();
                        let g_prqs = gamma[((p * m + r) * m + q) * m + s];
                        pair_sum += (mpq * mrs_conj).re * g_prqs;
                    }
                }
            }
        }

        // exchange = sum M_pq conj(M_qs) D_ps
        let mut exch_sum = 0.0_f64;
        for p in 0..m {
            for q in 0..m {
                let mpq = mk[p * m + q];
                if mpq.norm_sqr() == 0.0 {
                    continue;
                }
                for s in 0..m {
                    let mqs_conj = mk[q * m + s].conj();
                    let d_ps = d[p * m + s];
                    exch_sum += (mpq * mqs_conj).re * d_ps;
                }
            }
        }

        let mut n_s = pair_sum + exch_sum;

        if connected {
            // Subtract |<rho_k>|^2 = |Tr(M D)|^2
            let mut tr_md = C64::new(0.0, 0.0);
            for p in 0..m {
                for q in 0..m {
                    tr_md += mk[p * m + q] * d[q * m + p];
                }
            }
            n_s -= tr_md.norm_sqr();
        }

        s_values.push(n_s / n_electrons);
    }

    Ok(s_values)
}
