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
use crate::models::hubbard::Form;
use crate::models::hubbard_response::{
    self as model_hr, sum_over_states, ColumnSource, Fit, Group, Kernel, Options, Probe,
    Response, StateBlock, DEGENERATE, NO_SLOPE,
};
use crate::periodic::hubbard::Spec;
use crate::quantum::C64;
use crate::Result;
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// Number of response channels per Hubbard shell: `2` for collinear spin-polarized or magnetic
/// spinor crystals (`!crystal.time_reversal`), `1` for restricted or time-reversal spinor crystals.
pub(in crate::periodic) fn response_channels(c: &Crystal) -> usize {
    if c.spinor && !c.time_reversal {
        2
    } else {
        c.spins
    }
}

/// Unit magnetization direction $\hat{\mathbf{m}}_I$ for each spinor shell (`[0, 0, 1]` fallback).
fn shell_spin_axes(c: &Crystal, spec: &Spec, result: &PeriodicResult) -> Vec<[f64; 3]> {
    let shells = spec.shells.len();
    if !c.spinor || c.time_reversal {
        return vec![[0., 0., 1.]; shells];
    }
    let Some(report) = &result.hubbard else {
        return vec![[0., 0., 1.]; shells];
    };
    (0..shells)
        .map(|i| {
            let p = spec.shells[i].size();
            let d = 2 * p;
            let occ = report.occupation(0, i);
            if occ.len() != d * d {
                return [0., 0., 1.];
            }
            let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
            for a in 0..p {
                let ud = occ[a * d + (p + a)];
                let du = occ[(p + a) * d + a];
                let uu = occ[a * d + a].re;
                let dd = occ[(p + a) * d + (p + a)].re;
                mx += (ud + du).re;
                my += (C64::new(0., 1.) * (ud - du)).re;
                mz += uu - dd;
            }
            let norm = (mx * mx + my * my + mz * mz).sqrt();
            if norm > 1e-4 {
                [mx / norm, my / norm, mz / norm]
            } else {
                [0., 0., 1.]
            }
        })
        .collect()
}

/// Spinor `2x2` matrix for `alpha_+ P_+ + alpha_- P_-` along unit axis `axis`.
fn spin_shift_matrix(channels: usize, shifts: &[f64], shell_idx: usize, axis: [f64; 3]) -> [C64; 4] {
    if channels == 1 {
        let a = C64::new(shifts[shell_idx], 0.);
        return [a, C64::default(), C64::default(), a];
    }
    let (ap, am) = (shifts[2 * shell_idx], shifts[2 * shell_idx + 1]);
    let avg = 0.5 * (ap + am);
    let diff = 0.5 * (ap - am);
    let [mx, my, mz] = axis;
    [
        C64::new(avg + diff * mz, 0.),
        C64::new(diff * mx, -diff * my),
        C64::new(diff * mx, diff * my),
        C64::new(avg - diff * mz, 0.),
    ]
}

/// Assembles `c.spins` PAW projector shift operators (`p x p` scalar, `2p x 2p` spinor).
fn shift_operators(
    c: &Crystal,
    spec: &Spec,
    axes: &[[f64; 3]],
    shifts: &[f64],
) -> Result<Vec<Matrix>> {
    let p = c.p;
    let channels = response_channels(c);
    if c.spinor {
        let mut op = Matrix::zeros(2 * p, 2 * p);
        for (shell_idx, shell) in spec.shells.iter().enumerate() {
            let v = spin_shift_matrix(channels, shifts, shell_idx, axes[shell_idx]);
            if v.iter().all(|z| z.norm() == 0.) {
                continue;
            }
            let (p_shell, w) = (shell.size(), shell.weights.len());
            for a in 0..p_shell {
                for i in 0..w {
                    for j in 0..w {
                        let pi = shell.projectors[a * w + i];
                        let pj = shell.projectors[a * w + j];
                        let weight = shell.weights[i] * shell.weights[j];
                        for s in 0..2 {
                            for t in 0..2 {
                                op[(s * p + pi, t * p + pj)] += v[s * 2 + t] * weight;
                            }
                        }
                    }
                }
            }
        }
        return Ok(vec![op]);
    }
    let mut operators = vec![Matrix::zeros(p, p); c.spins];
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
/// - Restricted ($c.spins == 1$, non-spinor): $2 \times \mathrm{Tr}(n)$.
/// - Time-reversal spinor: $\mathrm{Tr}(n)$ (`channels = 1`).
/// - Magnetic spinor: $(N_+, N_-)$ projected along $\hat{\mathbf{m}}_I$ (`channels = 2`).
/// - Collinear spin-polarized ($c.spins == 2$): per-spin $\mathrm{Tr}(n_s)$ (`channels = 2`).
fn extract_scalar_occupations(
    c: &Crystal,
    spec: &Spec,
    axes: &[[f64; 3]],
    matrices: &[Vec<C64>],
) -> Result<Vec<f64>> {
    let shells = spec.shells.len();
    let channels = response_channels(c);
    let mut out = Vec::with_capacity(shells * channels);
    if c.spinor && channels == 2 {
        for (shell_idx, shell) in spec.shells.iter().enumerate() {
            let p = shell.size();
            let d = 2 * p;
            let occ = &matrices[shell_idx];
            let (mut n_tot, mut mx, mut my, mut mz) = (0.0, 0.0, 0.0, 0.0);
            for a in 0..p {
                let uu = occ[a * d + a].re;
                let dd = occ[(p + a) * d + (p + a)].re;
                let ud = occ[a * d + (p + a)];
                let du = occ[(p + a) * d + a];
                n_tot += uu + dd;
                mx += (ud + du).re;
                my += (C64::new(0., 1.) * (ud - du)).re;
                mz += uu - dd;
            }
            let [hx, hy, hz] = axes[shell_idx];
            let m_proj = hx * mx + hy * my + hz * mz;
            out.push(0.5 * (n_tot + m_proj));
            out.push(0.5 * (n_tot - m_proj));
        }
        return Ok(out);
    }
    for shell_idx in 0..shells {
        let p = spec.shells[shell_idx].size();
        let d = if c.spinor { 2 * p } else { p };
        for spin in 0..channels {
            let occ = &matrices[spin * shells + shell_idx];
            let trace = (0..d).map(|a| occ[a * d + a].re).sum::<f64>();
            out.push(c.degeneracy() * trace);
        }
    }
    Ok(out)
}

/// A linear-response probe evaluating crystal shell occupations under potential shifts.
pub struct PeriodicProbe<'a> {
    pub crystal: &'a Crystal,
    pub ground_state: &'a PeriodicResult,
    pub options: PeriodicOptions,
    hubbard_override: Option<Spec>,
    axes: Vec<[f64; 3]>,
}

impl<'a> PeriodicProbe<'a> {
    pub fn new(crystal: &'a Crystal, ground_state: &'a PeriodicResult) -> Result<Self> {
        Self::with_hubbard(crystal, ground_state, None)
    }

    pub fn with_hubbard(
        crystal: &'a Crystal,
        ground_state: &'a PeriodicResult,
        hubbard_override: Option<Spec>,
    ) -> Result<Self> {
        let hub = hubbard_override
            .as_ref()
            .or(crystal.hubbard.as_ref())
            .ok_or_else(|| {
                invalid("PeriodicProbe requires a crystal with a hubbard specification")
            })?;
        if !ground_state.converged {
            return Err(invalid(
                "PeriodicProbe requires a converged ground-state result",
            ));
        }
        super::hubbard_symmetry::require_fixed_atoms(hub, crystal)?;
        let axes = shell_spin_axes(crystal, hub, ground_state);
        Ok(Self {
            crystal,
            ground_state,
            options: ground_state.options.clone(),
            hubbard_override,
            axes,
        })
    }

    fn hubbard_spec(&self) -> Result<&Spec> {
        self.hubbard_override
            .as_ref()
            .or(self.crystal.hubbard.as_ref())
            .ok_or_else(|| invalid("PeriodicProbe requires a crystal with a hubbard specification"))
    }
}

impl<'a> Probe for PeriodicProbe<'a> {
    fn shells(&self) -> usize {
        self.hubbard_override
            .as_ref()
            .or(self.crystal.hubbard.as_ref())
            .map_or(0, |h| h.shells.len())
    }

    fn channels(&self) -> usize {
        response_channels(self.crystal)
    }

    fn frozen(&mut self, shifts: &[f64]) -> Result<Vec<f64>> {
        let hub = self.hubbard_spec()?;
        let s_ops = shift_operators(self.crystal, hub, &self.axes, shifts)?;
        let occ_matrices = self.crystal.frozen_occupations_with(
            self.ground_state,
            Some(&s_ops),
            self.hubbard_override.as_ref(),
        )?;
        extract_scalar_occupations(self.crystal, hub, &self.axes, &occ_matrices)
    }

    fn scf(&mut self, shifts: &[f64]) -> Result<Vec<f64>> {
        let hub = self.hubbard_spec()?;
        let s_ops = shift_operators(self.crystal, hub, &self.axes, shifts)?;
        let res = self.crystal.solve_with_hubbard(
            &self.options,
            Some(self.ground_state),
            Some(&s_ops),
            self.hubbard_override.as_ref(),
        )?;
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
        extract_scalar_occupations(self.crystal, hub, &self.axes, &occ_matrices)
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
    /// Per-shell exchange component $J = F_{ud} - F_{uu}$ (collinear or magnetic spinor).
    pub j_response: Option<Vec<f64>>,
    /// Element symbol per shell.
    pub symbols: Vec<String>,
    /// Whether this response came from a self-consistent outer U loop (`LR-12`).
    pub self_consistent: bool,
    /// The full underlying `Response` object.
    pub response: Response,
}

impl ResponseResult {
    /// Off-diagonal opposite-spin block `0.5 * (F_ud + F_du)` across shells when `channels == 2`.
    pub fn u_cross(&self) -> Option<DMatrix<f64>> {
        if self.response.channels != 2 {
            return None;
        }
        let s = self.response.shells;
        Some(DMatrix::from_fn(s, s, |i, j| {
            0.5 * (self.u_matrix[(2 * i, 2 * j + 1)] + self.u_matrix[(2 * i + 1, 2 * j)])
        }))
    }

    /// Reduces the response matrix onto per-element `(Group, effective_u_or_u, j)` checking shell spread.
    pub fn hubbard_u(
        &self,
        form: Form,
        kernel: Kernel,
        j_override: Option<f64>,
        tolerance: f64,
    ) -> Result<Vec<(Group, f64, f64)>> {
        if let Some(jv) = j_override {
            if !jv.is_finite() || jv < 0. {
                return Err(invalid("Hubbard J must be finite and non-negative"));
            }
        }
        let keys: Vec<&str> = self.symbols.iter().map(String::as_str).collect();
        let groups = self.response.hubbard_u(kernel, &keys, tolerance)?;
        let j_vec = self.j_response.as_deref();
        groups
            .into_iter()
            .map(|g| {
                let j_val = match (form, j_override) {
                    (Form::Dudarev, _) => j_override.unwrap_or(0.0),
                    (Form::Liechtenstein, Some(jv)) => jv,
                    (Form::Liechtenstein, None) => match j_vec {
                        Some(jv) => {
                            let vals: Vec<f64> = g.shells.iter().map(|&s| jv[s].max(0.0)).collect();
                            vals.iter().sum::<f64>() / vals.len() as f64
                        }
                        None => 0.0,
                    },
                };
                if j_val > g.u.max(0.0) {
                    return Err(invalid(format!(
                        "Hubbard J ({j_val:.4} Ha) exceeds extracted U ({:.4} Ha) for {:?}",
                        g.u, g.key
                    )));
                }
                Ok((g.clone(), g.u, j_val))
            })
            .collect()
    }
}

/// Evaluates the non-interacting response matrix $\chi^0$ analytically via sum-over-states.
pub fn sum_over_states_chi0(crystal: &Crystal, result: &PeriodicResult) -> Result<DMatrix<f64>> {
    let hub = crystal.hubbard.as_ref().ok_or_else(|| {
        invalid("sum_over_states_chi0 requires a crystal with a hubbard specification")
    })?;
    let shells = hub.shells.len();
    let channels = response_channels(crystal);
    let nk = crystal.points.len();
    let m = crystal.bands();
    let degeneracy = crystal.degeneracy();
    let width = result.options.width;
    let smearing = result.options.smearing;
    let axes = shell_spin_axes(crystal, hub, result);

    if crystal.spins == 1 {
        let mut blocks = Vec::with_capacity(nk);
        let mut all_elements: Vec<Vec<Vec<C64>>> = Vec::with_capacity(nk);
        for k in 0..nk {
            let coeff = result.coefficients[k].matrix(&crystal.context)?;
            let mut shell_matrices = Vec::with_capacity(shells * channels);
            for (shell_idx, shell) in hub.shells.iter().enumerate() {
                let w = &shell.weights;
                let proj = &shell.projectors;
                let cols = Matrix::from_fn(crystal.m, proj.len(), |a, b| {
                    crystal.points[k].projections[(a, proj[b])]
                });
                let p_shell = shell.size();
                if crystal.spinor && channels == 2 {
                    let u_mat = coeff.rows(0, crystal.m).adjoint() * &cols;
                    let d_mat = coeff.rows(crystal.m, crystal.m).adjoint() * &cols;
                    let [mx, my, mz] = axes[shell_idx];
                    for ch in 0..2 {
                        let sign = if ch == 0 { 1.0 } else { -1.0 };
                        let mut m_shell = vec![C64::new(0., 0.); m * m];
                        let p00 = C64::new(0.5 * (1.0 + sign * mz), 0.0);
                        let p11 = C64::new(0.5 * (1.0 - sign * mz), 0.0);
                        let p01 = C64::new(0.5 * sign * mx, -0.5 * sign * my);
                        let p10 = C64::new(0.5 * sign * mx, 0.5 * sign * my);
                        for n_idx in 0..m {
                            for m_idx in 0..m {
                                let mut sum = C64::new(0., 0.);
                                for alpha in 0..p_shell {
                                    let (mut u_n, mut u_m, mut d_n, mut d_m) =
                                        (C64::default(), C64::default(), C64::default(), C64::default());
                                    for i in 0..w.len() {
                                        let idx = alpha * w.len() + i;
                                        u_n += u_mat[(n_idx, idx)] * w[i];
                                        u_m += u_mat[(m_idx, idx)] * w[i];
                                        d_n += d_mat[(n_idx, idx)] * w[i];
                                        d_m += d_mat[(m_idx, idx)] * w[i];
                                    }
                                    sum += u_n * p00 * u_m.conj()
                                        + u_n * p01 * d_m.conj()
                                        + d_n * p10 * u_m.conj()
                                        + d_n * p11 * d_m.conj();
                                }
                                m_shell[n_idx * m + m_idx] = sum;
                            }
                        }
                        shell_matrices.push(m_shell);
                    }
                } else {
                    let mut m_shell = vec![C64::new(0., 0.); m * m];
                    let blocks_a = if crystal.spinor {
                        vec![
                            coeff.rows(0, crystal.m).adjoint() * &cols,
                            coeff.rows(crystal.m, crystal.m).adjoint() * &cols,
                        ]
                    } else {
                        vec![coeff.adjoint() * cols]
                    };
                    for n_idx in 0..m {
                        for m_idx in 0..m {
                            let mut sum = C64::new(0., 0.);
                            for alpha in 0..p_shell {
                                for a_mat in &blocks_a {
                                    let mut v_n = C64::new(0., 0.);
                                    let mut v_m = C64::new(0., 0.);
                                    for i in 0..w.len() {
                                        v_n += a_mat[(n_idx, alpha * w.len() + i)] * w[i];
                                        v_m += a_mat[(m_idx, alpha * w.len() + i)] * w[i];
                                    }
                                    sum += v_n.conj() * v_m;
                                }
                            }
                            m_shell[n_idx * m + m_idx] = sum;
                        }
                    }
                    shell_matrices.push(m_shell);
                }
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

/// Identifies symmetry-equivalent full-mesh shell columns (`LR-6`) when crystal space-group
/// operations map Hubbard shells bijectively with matching species, `l`, `U`, `J`, total charge,
/// and magnetic moment magnitude (`flip` if moments reverse sign).
pub fn detect_column_sources(
    crystal: &Crystal,
    result: &PeriodicResult,
) -> Vec<ColumnSource> {
    let Some(spec) = &crystal.hubbard else {
        return Vec::new();
    };
    let n = spec.shells.len();
    let default_out = || vec![ColumnSource::Direct; n];
    if n <= 1 {
        return default_out();
    }
    let Some(desc) = &crystal.description else {
        return default_out();
    };
    if desc.point_group.is_some() {
        return default_out();
    }
    let Some(ops) = &desc.operations else {
        return default_out();
    };
    let Some(report) = &result.hubbard else {
        return default_out();
    };
    let charges: Vec<f64> = (0..n)
        .map(|i| {
            let p = spec.shells[i].size();
            (0..crystal.spins)
                .map(|s| crystal.degeneracy() * report.occupation_trace(s, i, p))
                .sum()
        })
        .collect();
    let moments: Vec<f64> = if crystal.spins == 2 {
        (0..n)
            .map(|i| {
                let p = spec.shells[i].size();
                report.occupation_trace(0, i, p) - report.occupation_trace(1, i, p)
            })
            .collect()
    } else if crystal.spinor && !crystal.time_reversal {
        let axes = shell_spin_axes(crystal, spec, result);
        (0..n)
            .map(|i| {
                let p = spec.shells[i].size();
                let d = 2 * p;
                let occ = report.occupation(0, i);
                let (mut mx, mut my, mut mz) = (0., 0., 0.);
                for a in 0..p {
                    let ud = occ[a * d + (p + a)];
                    let du = occ[(p + a) * d + a];
                    mx += (ud + du).re;
                    my += (C64::new(0., 1.) * (ud - du)).re;
                    mz += occ[a * d + a].re - occ[(p + a) * d + (p + a)].re;
                }
                mx * axes[i][0] + my * axes[i][1] + mz * axes[i][2]
            })
            .collect()
    } else {
        vec![0.0; n]
    };

    let mut atom_to_shell = std::collections::HashMap::new();
    for (s, shell) in spec.shells.iter().enumerate() {
        atom_to_shell.entry(shell.atom).or_insert_with(Vec::new).push(s);
    }
    let mut out = vec![ColumnSource::Direct; n];
    'shell: for j in 1..n {
        for (op_idx, atom_map) in ops.atom_maps.iter().enumerate() {
            let anti = ops
                .anti_unitary
                .as_ref()
                .and_then(|v| v.get(op_idx).copied())
                .unwrap_or(false);
            let mut rows = vec![(0usize, false); n];
            let mut valid = true;
            for dst in 0..n {
                // Find src atom mapped to spec.shells[dst].atom under ops[op_idx]
                let target_atom = spec.shells[dst].atom;
                let Some((src_atom, _)) = atom_map
                    .iter()
                    .enumerate()
                    .find(|&(_, &mapped)| mapped == target_atom)
                else {
                    valid = false;
                    break;
                };
                let Some(src_list) = atom_to_shell.get(&src_atom) else {
                    valid = false;
                    break;
                };
                let Some(&src) = src_list.iter().find(|&&s| {
                    let a = &spec.shells[s];
                    let b = &spec.shells[dst];
                    a.symbol == b.symbol
                        && a.l == b.l
                        && (a.u - b.u).abs() < 1e-10
                        && (a.j - b.j).abs() < 1e-10
                }) else {
                    valid = false;
                    break;
                };
                if (charges[src] - charges[dst]).abs() > 1e-4 {
                    valid = false;
                    break;
                }
                let flip = if (moments[src] - moments[dst]).abs() <= 1e-4 {
                    false
                } else if (moments[src] + moments[dst]).abs() <= 1e-4 {
                    true
                } else {
                    valid = false;
                    break;
                };
                let _ = anti;
                rows[dst] = (src, flip);
            }
            if !valid {
                continue;
            }
            // A global space-group operation either preserves every shell's spin orientation
            // (`flip == false`) or reverses all of them uniformly (`flip == true`, anti-unitary).
            let global_flip = rows[0].1;
            if rows.iter().any(|&(_, f)| f != global_flip) {
                continue;
            }
            let (rep, _) = rows[j];
            if rep < j && matches!(out[rep], ColumnSource::Direct) {
                out[j] = ColumnSource::Mapped { rep, rows };
                continue 'shell;
            }
        }
    }
    out
}

/// Evaluates linear-response Hubbard parameters with custom response options.
pub fn hubbard_response_with_options(
    crystal: &Crystal,
    result: &PeriodicResult,
    options: &Options,
) -> Result<ResponseResult> {
    let column_sources = detect_column_sources(crystal, result);
    hubbard_response_with_symmetry(crystal, result, options, &column_sources)
}

/// Evaluates linear-response Hubbard parameters with an explicit `Spec` and `ColumnSource` map.
pub fn hubbard_response_with_spec_and_symmetry(
    crystal: &Crystal,
    result: &PeriodicResult,
    hubbard_override: Option<&Spec>,
    options: &Options,
    column_sources: &[ColumnSource],
) -> Result<ResponseResult> {
    let mut probe = PeriodicProbe::with_hubbard(crystal, result, hubbard_override.cloned())?;
    let response =
        model_hr::measure_with_symmetry(&crystal.context, &mut probe, options, column_sources)?;
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
    let symbols = hubbard_override
        .or(crystal.hubbard.as_ref())
        .map(|h| h.shells.iter().map(|s| s.symbol.clone()).collect())
        .unwrap_or_default();
    Ok(ResponseResult {
        u_matrix: response.u.clone(),
        u_charge,
        chi0: response.chi0.clone(),
        chi: response.chi.clone(),
        alphas: options.alphas.clone(),
        j_response,
        symbols,
        self_consistent: false,
        response,
    })
}

/// Evaluates linear-response Hubbard parameters with explicit `ColumnSource` symmetry map.
pub fn hubbard_response_with_symmetry(
    crystal: &Crystal,
    result: &PeriodicResult,
    options: &Options,
    column_sources: &[ColumnSource],
) -> Result<ResponseResult> {
    hubbard_response_with_spec_and_symmetry(crystal, result, None, options, column_sources)
}
