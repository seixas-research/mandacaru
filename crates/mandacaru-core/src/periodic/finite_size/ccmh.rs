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

/// What [`structure_factor`] returns: every G-vector `k` of the box `|i|,|j|,|k| <= limit` but
/// zero, `S(k)`, `|k|`, `tr D` and `max |M'(0) - 1|` (the orthonormality of the rotated orbitals).
pub struct StructureFactorGrid {
    pub kvectors: Vec<f64>,
    pub values: Vec<f64>,
    pub magnitudes: Vec<f64>,
    pub n_electrons: f64,
    pub overlap_error: f64,
}

/// `S(k)` of the state `(D, Gamma)` in the orbitals `A` of the real grid functions `psi`
/// (`(n_ao, points)` row-major on the `shape` grid of volume element `dv`, lattice rows `cell`):
/// `M_ab(k) = FFT(psi_a psi_b)(k) dv`, `M' = A^dagger M A`, and
/// `N S(k) = sum M'_pq conj(M'_rs) Gamma_prsq + sum M'_pq conj(M'_qs) D_ps`.
#[allow(clippy::too_many_arguments)]
pub fn structure_factor(
    ctx: &crate::Context,
    psi: &[f64],
    shape: [usize; 3],
    dv: f64,
    cell: &[f64],
    limit: i64,
    a_matrix: &[C64],
    a_cols: usize,
    d_slice: &[f64],
    gamma_slice: &[f64],
) -> Result<StructureFactorGrid> {
    use crate::math::fft::Fft3;
    use nalgebra::{Matrix3, Vector3};
    let [nx, ny, nz] = shape;
    let n_grid = crate::product(&shape)?;
    if n_grid == 0 || psi.len() % n_grid != 0 || cell.len() != 9 || !(0..=16).contains(&limit) {
        return Err(invalid("psi must be (n_ao, points of the shape), the cell 3x3, the limit in 0..=16"));
    }
    let n_ao = psi.len() / n_grid;
    if a_cols == 0 || a_matrix.len() != n_ao * a_cols || d_slice.len() != a_cols * a_cols
        || gamma_slice.len() != a_cols.pow(4)
    {
        return Err(invalid("A must be (n_ao, M), D (M, M) and Gamma (M, M, M, M)"));
    }
    let mut fft = Fft3::new(&ctx, [nx, ny, nz])?;

    // Compute reciprocal triples and kvectors
    let c = cell;
    let mat = Matrix3::new(
        c[0], c[1], c[2],
        c[3], c[4], c[5],
        c[6], c[7], c[8],
    );
    let inv_t = mat
        .try_inverse()
        .ok_or_else(|| invalid("singular cell matrix"))?
        .transpose();
    let b = inv_t * (2.0 * std::f64::consts::PI);

    let mut triples = Vec::new();
    let mut kvectors_vec = Vec::new();

    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if i == 0 && j == 0 && k == 0 {
                    continue;
                }
                triples.push((i, j, k));
                let m = Vector3::new(i as f64, j as f64, k as f64);
                let kx = m[0] * b[(0, 0)] + m[1] * b[(1, 0)] + m[2] * b[(2, 0)];
                let ky = m[0] * b[(0, 1)] + m[1] * b[(1, 1)] + m[2] * b[(2, 1)];
                let kz = m[0] * b[(0, 2)] + m[1] * b[(1, 2)] + m[2] * b[(2, 2)];
                kvectors_vec.push(kx);
                kvectors_vec.push(ky);
                kvectors_vec.push(kz);
            }
        }
    }
    let n_triples = triples.len();
    // M and M' over every G-vector plus the zero vector, the pair line and the rotation scratch.
    let _tensors = ctx.reserve(crate::product(&[n_triples + 1, n_ao * n_ao + a_cols * a_cols, 16])?
        .checked_add(crate::product(&[n_grid + n_ao * a_cols, 16])?)
        .ok_or(crate::Error::Overflow)?)?;

    // M: (n_triples + 1, n_ao, n_ao)
    let mut m_tensor = vec![C64::new(0.0, 0.0); (n_triples + 1) * n_ao * n_ao];
    let psi_slice = psi;

    let mut buf = vec![C64::new(0.0, 0.0); n_grid];

    for a in 0..n_ao {
        let psi_a = &psi_slice[a * n_grid..(a + 1) * n_grid];
        for b in a..n_ao {
            let psi_b = &psi_slice[b * n_grid..(b + 1) * n_grid];
            for idx in 0..n_grid {
                buf[idx] = C64::new(psi_a[idx] * psi_b[idx], 0.0);
            }
            fft.transform(&mut buf, false)?;
            for idx in 0..n_grid {
                buf[idx] = buf[idx] * dv;
            }

            let val0 = buf[0];
            m_tensor[0 * n_ao * n_ao + a * n_ao + b] = val0;
            m_tensor[0 * n_ao * n_ao + b * n_ao + a] = val0.conj();

            for (t_idx, &(ti, tj, tk)) in triples.iter().enumerate() {
                let node_x = (ti.rem_euclid(nx as i64)) as usize;
                let node_y = (tj.rem_euclid(ny as i64)) as usize;
                let node_z = (tk.rem_euclid(nz as i64)) as usize;
                let flat_pos = (node_x * ny + node_y) * nz + node_z;

                let node_neg_x = ((-ti).rem_euclid(nx as i64)) as usize;
                let node_neg_y = ((-tj).rem_euclid(ny as i64)) as usize;
                let node_neg_z = ((-tk).rem_euclid(nz as i64)) as usize;
                let flat_neg = (node_neg_x * ny + node_neg_y) * nz + node_neg_z;

                let val = buf[flat_pos];
                let val_neg_conj = buf[flat_neg].conj();

                let k_layer = (t_idx + 1) * n_ao * n_ao;
                m_tensor[k_layer + a * n_ao + b] = val;
                m_tensor[k_layer + b * n_ao + a] = val_neg_conj;
            }
        }
    }

    // Rotate M: M'_k = A^\dagger M_k A
    let a_slice = a_matrix;
    let mut m_prime = vec![C64::new(0.0, 0.0); (n_triples + 1) * a_cols * a_cols];

    // Precompute for each k
    let mut temp = vec![C64::new(0.0, 0.0); n_ao * a_cols];
    for k_idx in 0..=n_triples {
        let mk = &m_tensor[k_idx * n_ao * n_ao..(k_idx + 1) * n_ao * n_ao];
        // temp[a, q] = sum_b M[a, b] * A[b, q]
        for a in 0..n_ao {
            for q in 0..a_cols {
                let mut sum = C64::new(0.0, 0.0);
                for b in 0..n_ao {
                    let a_bq = C64::new(a_slice[b * a_cols + q].re, a_slice[b * a_cols + q].im);
                    sum += mk[a * n_ao + b] * a_bq;
                }
                temp[a * a_cols + q] = sum;
            }
        }
        // m_prime[k, p, q] = sum_a A^*[a, p] * temp[a, q]
        let out_k = &mut m_prime[k_idx * a_cols * a_cols..(k_idx + 1) * a_cols * a_cols];
        for p in 0..a_cols {
            for q in 0..a_cols {
                let mut sum = C64::new(0.0, 0.0);
                for a in 0..n_ao {
                    let a_ap_conj =
                        C64::new(a_slice[a * a_cols + p].re, -a_slice[a * a_cols + p].im);
                    sum += a_ap_conj * temp[a * a_cols + q];
                }
                out_k[p * a_cols + q] = sum;
            }
        }
    }

    // Overlap error: max |M'[0, p, q] - delta_{p,q}|
    let m0 = &m_prime[0..a_cols * a_cols];
    let mut overlap_error = 0.0_f64;
    for p in 0..a_cols {
        for q in 0..a_cols {
            let diff = if p == q {
                m0[p * a_cols + q] - C64::new(1.0, 0.0)
            } else {
                m0[p * a_cols + q]
            };
            overlap_error = overlap_error.max(diff.norm());
        }
    }

    // Electron count: tr(D)
    let mut n_electrons = 0.0_f64;
    for p in 0..a_cols {
        n_electrons += d_slice[p * a_cols + p];
    }

    // Values for each k in 1..=n_triples
    let mut values = Vec::with_capacity(n_triples);
    for k_idx in 1..=n_triples {
        let mk = &m_prime[k_idx * a_cols * a_cols..(k_idx + 1) * a_cols * a_cols];

        let mut pair_sum = 0.0_f64;
        for p in 0..a_cols {
            for q in 0..a_cols {
                let mpq = mk[p * a_cols + q];
                if mpq.norm_sqr() == 0.0 {
                    continue;
                }
                for r in 0..a_cols {
                    for s in 0..a_cols {
                        let mrs_conj = mk[r * a_cols + s].conj();
                        // Python: Gamma[p, r, s, q]
                        let g_prsq = gamma_slice[((p * a_cols + r) * a_cols + s) * a_cols + q];
                        pair_sum += (mpq * mrs_conj).re * g_prsq;
                    }
                }
            }
        }

        let mut exch_sum = 0.0_f64;
        for p in 0..a_cols {
            for q in 0..a_cols {
                let mpq = mk[p * a_cols + q];
                if mpq.norm_sqr() == 0.0 {
                    continue;
                }
                for s in 0..a_cols {
                    let mqs_conj = mk[q * a_cols + s].conj();
                    // Python: D[p, s]
                    let d_ps = d_slice[p * a_cols + s];
                    exch_sum += (mpq * mqs_conj).re * d_ps;
                }
            }
        }

        values.push((pair_sum + exch_sum) / n_electrons);
    }

    let mut magnitudes = Vec::with_capacity(n_triples);
    for k in 0..n_triples {
        let kx = kvectors_vec[k * 3];
        let ky = kvectors_vec[k * 3 + 1];
        let kz = kvectors_vec[k * 3 + 2];
        magnitudes.push((kx * kx + ky * ky + kz * kz).sqrt());
    }

    Ok(StructureFactorGrid { kvectors: kvectors_vec, values, magnitudes, n_electrons, overlap_error })
}
