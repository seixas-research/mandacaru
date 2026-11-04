//! Linear-response determination of Hubbard U parameters for crystals.
//!
//! Evaluates the screened interaction matrix $U = (\chi^0)^{-1} - \chi^{-1}$
//! from the response of correlated shell occupations to potential shifts $\pm \alpha$,
//! following Cococcioni & de Gironcoli (Phys. Rev. B 71, 035105 (2005)).
//!
//! $\chi^0$ is the non-interacting response (frozen Kohn-Sham potential), which can also
//! be evaluated analytically via sum-over-states from the ground-state eigenpairs.
//! $\chi$ is the self-consistent response obtained by converging the SCF under the
//! shell potential shifts.

use super::{invalid, Crystal, Options as PeriodicOptions, PeriodicResult};
use crate::math::storage::MatrixSource;
use crate::models::hubbard_response::{
    self as model_hr, sum_over_states, Fit, Kernel, Options, Probe, Response, StateBlock,
    DEGENERATE, NO_SLOPE,
};
use crate::periodic::hubbard::Spec;
use crate::quantum::C64;
use crate::Result;
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// Assembles $p \times p$ PAW projector shift operators from a vector of shifts on shells.
fn shift_operators(c: &Crystal, spec: &Spec, shifts: &[f64]) -> Result<Vec<Matrix>> {
    let p = c.p;
    let mut operators = vec![Matrix::zeros(p, p); c.spins];
    let channels = c.spins; // 1 (restricted) or 2 (collinear)
    for (shell_idx, shell) in spec.shells.iter().enumerate() {
        let (p_shell, w) = (shell.size(), shell.weights.len());
        for spin in 0..c.spins {
            let alpha = if channels == 1 {
                shifts[shell_idx]
            } else {
                shifts[shell_idx * channels + spin]
            };
            if alpha == 0.0 {
                continue;
            }
            let op = &mut operators[spin];
            for a in 0..p_shell {
                for i in 0..w {
                    for j in 0..w {
                        let pi = shell.projectors[a * w + i];
                        let pj = shell.projectors[a * w + j];
                        let weight = shell.weights[i] * shell.weights[j];
                        op[(pi, pj)] += C64::new(alpha * weight, 0.);
                    }
                }
            }
        }
    }
    Ok(operators)
}

/// Extracts scalar occupations matching the `Probe` convention:
/// - Restricted ($c.spins == 1$): spin-summed occupation $2 \times \mathrm{Tr}(n)$.
/// - Unrestricted ($c.spins == 2$): per-spin occupation $\mathrm{Tr}(n_{s})$.
fn extract_scalar_occupations(c: &Crystal, spec: &Spec, matrices: &[Vec<C64>]) -> Result<Vec<f64>> {
    let shells = spec.shells.len();
    let channels = c.spins;
    let mut out = Vec::with_capacity(shells * channels);
    for shell_idx in 0..shells {
        let p = spec.shells[shell_idx].size();
        for spin in 0..channels {
            let occ = &matrices[spin * shells + shell_idx];
            let trace = (0..p).map(|a| occ[a * p + a].re).sum::<f64>();
            let value = if channels == 1 {
                c.degeneracy() * trace // 2 * trace for restricted (spin-summed)
            } else {
                trace
            };
            out.push(value);
        }
    }
    Ok(out)
}

/// A linear-response probe evaluating crystal shell occupations under potential shifts.
pub struct PeriodicProbe<'a> {
    pub crystal: &'a Crystal,
    pub ground_state: &'a PeriodicResult,
    pub options: PeriodicOptions,
}

impl<'a> PeriodicProbe<'a> {
    pub fn new(crystal: &'a Crystal, ground_state: &'a PeriodicResult) -> Result<Self> {
        if crystal.hubbard.is_none() {
            return Err(invalid(
                "PeriodicProbe requires a crystal with a hubbard specification",
            ));
        }
        if !ground_state.converged {
            return Err(invalid(
                "PeriodicProbe requires a converged ground-state result",
            ));
        }
        Ok(Self {
            crystal,
            ground_state,
            options: ground_state.options.clone(),
        })
    }
}

impl<'a> Probe for PeriodicProbe<'a> {
    fn shells(&self) -> usize {
        self.crystal
            .hubbard
            .as_ref()
            .map_or(0, |h| h.shells.len())
    }

    fn channels(&self) -> usize {
        self.crystal.spins
    }

    fn frozen(&mut self, shifts: &[f64]) -> Result<Vec<f64>> {
        let hub = self.crystal.hubbard.as_ref().ok_or_else(|| {
            invalid("PeriodicProbe requires a crystal with a hubbard specification")
        })?;
        let s_ops = shift_operators(self.crystal, hub, shifts)?;
        let occ_matrices = self
            .crystal
            .frozen_occupations(self.ground_state, Some(&s_ops))?;
        extract_scalar_occupations(self.crystal, hub, &occ_matrices)
    }

    fn scf(&mut self, shifts: &[f64]) -> Result<Vec<f64>> {
        let hub = self.crystal.hubbard.as_ref().ok_or_else(|| {
            invalid("PeriodicProbe requires a crystal with a hubbard specification")
        })?;
        let s_ops = shift_operators(self.crystal, hub, shifts)?;
        let res = self
            .crystal
            .solve_with(&self.options, Some(self.ground_state), Some(&s_ops))?;
        if !res.converged {
            return Err(invalid(format!(
                "SCF under Hubbard shift failed to converge after {} iterations",
                self.options.max_iterations
            )));
        }
        let occ_matrices = match &res.hubbard {
            Some(rep) => rep.occupations.clone(),
            None => hub.occupations(self.crystal, &res.matrices)?,
        };
        extract_scalar_occupations(self.crystal, hub, &occ_matrices)
    }
}

/// The result of linear-response Hubbard U measurement on a crystal.
#[derive(Clone, Debug)]
pub struct ResponseResult {
    /// Full interaction matrix $U = (\chi^0)^{-1} - \chi^{-1}$.
    pub u_matrix: DMatrix<f64>,
    /// Charge-screened interaction matrix.
    pub u_charge: DMatrix<f64>,
    /// Symmetrized non-interacting response matrix $\chi^0$.
    pub chi0: DMatrix<f64>,
    /// Symmetrized interacting response matrix $\chi$.
    pub chi: DMatrix<f64>,
    /// Shifts $\alpha$ (Hartree) used for finite difference.
    pub alphas: Vec<f64>,
    /// Per-shell exchange component $J = F_{ud} - F_{uu}$ (collinear spin only).
    pub j_response: Option<Vec<f64>>,
    /// The full underlying `Response` object.
    pub response: Response,
}

/// Evaluates the non-interacting response matrix $\chi^0$ analytically via sum-over-states.
pub fn sum_over_states_chi0(crystal: &Crystal, result: &PeriodicResult) -> Result<DMatrix<f64>> {
    let hub = crystal.hubbard.as_ref().ok_or_else(|| {
        invalid("sum_over_states_chi0 requires a crystal with a hubbard specification")
    })?;
    let shells = hub.shells.len();
    let channels = crystal.spins; // 1 or 2
    let nk = crystal.points.len();
    let m = crystal.bands();
    let degeneracy = crystal.degeneracy();
    let width = result.options.width;
    let smearing = result.options.smearing;

    if channels == 1 {
        let mut blocks = Vec::with_capacity(nk);
        let mut all_elements: Vec<Vec<Vec<C64>>> = Vec::with_capacity(nk);
        for k in 0..nk {
            let coeff = result.coefficients[k].matrix(&crystal.context)?;
            let mut shell_matrices = Vec::with_capacity(shells);
            for shell in &hub.shells {
                let w = &shell.weights;
                let proj = &shell.projectors;
                let cols = Matrix::from_fn(m, proj.len(), |a, b| {
                    crystal.points[k].projections[(a, proj[b])]
                });
                let a_mat = coeff.adjoint() * cols;
                let p_shell = shell.size();
                let mut m_shell = vec![C64::new(0., 0.); m * m];
                for n_idx in 0..m {
                    for m_idx in 0..m {
                        let mut sum = C64::new(0., 0.);
                        for alpha in 0..p_shell {
                            let mut v_n = C64::new(0., 0.);
                            let mut v_m = C64::new(0., 0.);
                            for i in 0..w.len() {
                                v_n += a_mat[(n_idx, alpha * w.len() + i)] * w[i];
                                v_m += a_mat[(m_idx, alpha * w.len() + i)] * w[i];
                            }
                            sum += v_n.conj() * v_m;
                        }
                        m_shell[n_idx * m + m_idx] = sum;
                    }
                }
                shell_matrices.push(m_shell);
            }
            all_elements.push(shell_matrices);
        }

        let mut occ_storage = Vec::with_capacity(nk);
        let mut slope_storage = Vec::with_capacity(nk);
        for k in 0..nk {
            let occ: Vec<f64> = result.occupations[k]
                .iter()
                .map(|f| f / degeneracy)
                .collect();
            let slopes: Vec<f64> = result.eigenvalues[k]
                .iter()
                .map(|e| smearing.slope((e - result.fermi) / width) / width)
                .collect();
            occ_storage.push(occ);
            slope_storage.push(slopes);
        }

        let mut block_slices = Vec::with_capacity(nk);
        for k in 0..nk {
            let slices: Vec<&[C64]> = all_elements[k].iter().map(|v| v.as_slice()).collect();
            block_slices.push(slices);
        }

        for k in 0..nk {
            blocks.push(StateBlock {
                weight: crystal.points[k].weight,
                levels: &result.eigenvalues[k],
                occupations: &occ_storage[k],
                slopes: &slope_storage[k],
                elements: &block_slices[k],
            });
        }

        let sum = sum_over_states(&blocks, degeneracy)?;
        Ok(sum.total())
    } else {
        let n_dim = shells * channels;
        let mut pairs = DMatrix::zeros(n_dim, n_dim);
        let mut s_vec = vec![0.0; n_dim];
        let mut total_slope = 0.0;

        for spin in 0..2 {
            for k in 0..nk {
                let weight = crystal.points[k].weight;
                let coeff = result.coefficients[spin * nk + k].matrix(&crystal.context)?;
                let levels = &result.eigenvalues[spin * nk + k];
                let occ: Vec<f64> = result.occupations[spin * nk + k].clone();
                let slopes: Vec<f64> = levels
                    .iter()
                    .map(|e| smearing.slope((e - result.fermi) / width) / width)
                    .collect();

                let mut shell_matrices = Vec::with_capacity(shells);
                for shell in &hub.shells {
                    let w = &shell.weights;
                    let proj = &shell.projectors;
                    let cols = Matrix::from_fn(m, proj.len(), |a, b| {
                        crystal.points[k].projections[(a, proj[b])]
                    });
                    let a_mat = coeff.adjoint() * cols;
                    let p_shell = shell.size();
                    let mut m_shell = vec![C64::new(0., 0.); m * m];
                    for n_idx in 0..m {
                        for m_idx in 0..m {
                            let mut sum = C64::new(0., 0.);
                            for alpha in 0..p_shell {
                                let mut v_n = C64::new(0., 0.);
                                let mut v_m = C64::new(0., 0.);
                                for i in 0..w.len() {
                                    v_n += a_mat[(n_idx, alpha * w.len() + i)] * w[i];
                                    v_m += a_mat[(m_idx, alpha * w.len() + i)] * w[i];
                                }
                                sum += v_n.conj() * v_m;
                            }
                            m_shell[n_idx * m + m_idx] = sum;
                        }
                    }
                    shell_matrices.push(m_shell);
                }

                for n_idx in 0..m {
                    total_slope += weight * slopes[n_idx];
                    for i in 0..shells {
                        let idx_i = i * 2 + spin;
                        s_vec[idx_i] +=
                            weight * slopes[n_idx] * shell_matrices[i][n_idx * m + n_idx].re;
                    }
                    for m_idx in 0..m {
                        let gap = levels[n_idx] - levels[m_idx];
                        let c = if gap.abs() < DEGENERATE {
                            0.5 * (slopes[n_idx] + slopes[m_idx])
                        } else {
                            (occ[n_idx] - occ[m_idx]) / gap
                        };
                        for i in 0..shells {
                            let idx_i = i * 2 + spin;
                            let a = shell_matrices[i][n_idx * m + m_idx];
                            for j in 0..shells {
                                let idx_j = j * 2 + spin;
                                let b = shell_matrices[j][m_idx * m + n_idx];
                                pairs[(idx_i, idx_j)] += weight * c * (a * b).re;
                            }
                        }
                    }
                }
            }
        }

        let fermi = if total_slope.abs() < NO_SLOPE {
            DMatrix::zeros(n_dim, n_dim)
        } else {
            DMatrix::from_fn(n_dim, n_dim, |i, j| -s_vec[i] * s_vec[j] / total_slope)
        };

        Ok(pairs + fermi)
    }
}

/// Evaluates the linear-response Hubbard interaction matrix using the converged ground state of `crystal`.
pub fn hubbard_response(
    crystal: &Crystal,
    result: &PeriodicResult,
    alphas: &[f64],
) -> Result<ResponseResult> {
    if alphas.is_empty() {
        return Err(invalid("alphas cannot be empty"));
    }
    let fit = if alphas.len() >= 2 {
        Fit::Richardson
    } else {
        Fit::Central
    };
    let options = Options::new(alphas, fit)?;
    hubbard_response_with_options(crystal, result, &options)
}

/// Evaluates linear-response Hubbard parameters with custom response options.
pub fn hubbard_response_with_options(
    crystal: &Crystal,
    result: &PeriodicResult,
    options: &Options,
) -> Result<ResponseResult> {
    let mut probe = PeriodicProbe::new(crystal, result)?;
    let response = model_hr::measure(&crystal.context, &mut probe, options)?;
    let u_charge = if response.channels == 1 {
        response.kernel(Kernel::SpinSummed)?
    } else {
        response.kernel(Kernel::Charge)?
    };
    let j_response = if response.channels == 2 {
        Some(response.j_response()?)
    } else {
        None
    };
    Ok(ResponseResult {
        u_matrix: response.u.clone(),
        u_charge,
        chi0: response.chi0.clone(),
        chi: response.chi.clone(),
        alphas: options.alphas.clone(),
        j_response,
        response,
    })
}
