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
                        let h_pqrs =
                            gamma[((p * m + q) * m + r) * m + s] - d[p * m + r] * d[q * m + s];
                        k_sum += h_pqrs * (rho_pq * rho_rs_conj).re;
                    }
                }
            }
        }
        total += wk * k_sum;
    }

    Ok(0.5 * total)
}

/// The periodic Coulomb kernel `4 pi / |G|^2 / dv` (zero at `G = 0`) on the FFT order of a grid of
/// `shape` points with step `step` (rows, row-major): the cell is `step diag(shape)`.
pub fn periodic_poisson_kernel(shape: [usize; 3], step: &[f64; 9], dv: f64) -> Result<Vec<f64>> {
    let step_mat = nalgebra::Matrix3::from_row_slice(step);
    let diag = nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(
        shape[0] as f64,
        shape[1] as f64,
        shape[2] as f64,
    ));
    let cell = step_mat * diag;
    let inv_t = cell
        .try_inverse()
        .ok_or_else(|| invalid("singular cell"))?
        .transpose();
    let reciprocal = inv_t * (2.0 * std::f64::consts::PI);
    let [nx, ny, nz] = shape;
    let index = |i: usize, n: usize| {
        if i < n.div_ceil(2) {
            i as f64
        } else {
            i as f64 - n as f64
        }
    };
    let mut k = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for l in 0..nz {
                let kvec =
                    reciprocal * nalgebra::Vector3::new(index(i, nx), index(j, ny), index(l, nz));
                let g2 = kvec.dot(&kvec);
                k.push(if g2 > 0.0 {
                    (4.0 * std::f64::consts::PI / g2) / dv
                } else {
                    0.0
                });
            }
        }
    }
    Ok(k)
}

/// The forward transform's real part of a real-space kernel (`shape` points).
pub fn truncated_coulomb_kernel(
    ctx: &crate::Context,
    real_kernel: &[f64],
    shape: [usize; 3],
) -> Result<Vec<f64>> {
    if real_kernel.len() != crate::product(&shape)? {
        return Err(invalid("real_kernel size does not match shape"));
    }
    let mut fft = crate::math::fft::Fft3::new(ctx, shape)?;
    let mut buf: Vec<C64> = real_kernel.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft.transform(&mut buf, false)?;
    Ok(buf.iter().map(|c| c.re).collect())
}

/// `dv * IFFT(kernel * FFT(rho_p))` for every density row `rho_p` of `rho_stack` (rows of `shape`
/// points), row-major.
pub fn poisson_solve_stack(
    ctx: &crate::Context,
    kernel: &[f64],
    shape: [usize; 3],
    rho_stack: &[f64],
    dv: f64,
) -> Result<Vec<f64>> {
    let n_grid = crate::product(&shape)?;
    if kernel.len() != n_grid || rho_stack.len() % n_grid.max(1) != 0 {
        return Err(invalid(
            "the kernel and the density rows must match the grid",
        ));
    }
    let mut fft = crate::math::fft::Fft3::new(ctx, shape)?;
    let mut out = Vec::with_capacity(rho_stack.len());
    let mut buf = vec![C64::new(0.0, 0.0); n_grid];
    for rho_p in rho_stack.chunks(n_grid) {
        for i in 0..n_grid {
            buf[i] = C64::new(rho_p[i], 0.0);
        }
        fft.transform(&mut buf, false)?;
        for i in 0..n_grid {
            buf[i] *= kernel[i];
        }
        fft.transform(&mut buf, true)?;
        out.extend(buf.iter().map(|v| v.re * dv));
    }
    Ok(out)
}

/// `(nonzero volume, largest 1 / v)` of a minimum-image kernel: the volume its nonzero values
/// cover and the reach of its smallest positive value.
pub fn wigner_seitz_metrics(kernel: &[f64], dv: f64) -> (f64, f64) {
    let mut count = 0usize;
    let mut max_inv = 0.0_f64;
    for &v in kernel {
        if v != 0.0 {
            count += 1;
        }
        if v > 0.0 {
            max_inv = max_inv.max(1.0 / v);
        }
    }
    (count as f64 * dv, max_inv)
}

/// `(1/2 sum g^E (Gamma - D D), 1/2 sum f (Gamma - D D), tr D)` of the periodic and the truncated
/// tensors (`m^4`, row-major) against the hole of `gamma` and `d`.
pub fn hole_energies(
    periodic: &[f64],
    truncated: &[f64],
    d: &[f64],
    gamma: &[f64],
    m: usize,
) -> Result<(f64, f64, f64)> {
    let m4 = crate::product(&[m, m, m, m])?;
    if periodic.len() != m4 || truncated.len() != m4 || gamma.len() != m4 || d.len() != m * m {
        return Err(invalid("tensors must be (M, M, M, M) and D must be (M, M)"));
    }
    let (mut p_sum, mut t_sum) = (0.0_f64, 0.0_f64);
    for p in 0..m {
        for q in 0..m {
            for r in 0..m {
                let d_pr = d[p * m + r];
                for s in 0..m {
                    let idx = ((p * m + q) * m + r) * m + s;
                    let hole = gamma[idx] - d_pr * d[q * m + s];
                    p_sum += periodic[idx] * hole;
                    t_sum += truncated[idx] * hole;
                }
            }
        }
    }
    let mut n_electrons = 0.0_f64;
    for i in 0..m {
        n_electrons += d[i * m + i];
    }
    Ok((0.5 * p_sum, 0.5 * t_sum, n_electrons))
}
