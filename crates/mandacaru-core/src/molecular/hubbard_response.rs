//! Linear-response determination of Hubbard `U` parameters for molecules (`MP-7`).
//!
//! Uses [`crate::models::hubbard_response`] (`Probe`, `measure`, `sum_over_states`) on top of
//! [`MolecularIntegrals::kohn_sham_from`] (collinear) and
//! [`MolecularIntegrals::spinor_kohn_sham_shifted`] (spinor), holding particle count fixed via
//! Fermi-Dirac chemical-potential re-solving in both frozen and restarted SCF passes.

use super::{
    hubbard::MolecularHubbard,
    integrals::MolecularIntegrals,
    invalid, occupation,
    scf::{MeanField, ScfOptions},
};
use crate::math::smearing::Smearing;
use crate::models::{
    hubbard::Form,
    hubbard_response::{
        self as model_hr, sum_over_states, Fit, Group, Kernel, Options, Probe, Response,
        StateBlock, DEGENERATE, NO_SLOPE,
    },
};
use crate::quantum::C64;
use crate::Result;
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// Returns `(is_spinor, channels_per_shell, degeneracy, shell_spin_axes)`.
fn response_layout(
    hub: &MolecularHubbard,
    gs: &MeanField,
    m: usize,
) -> (bool, usize, f64, Vec<[f64; 3]>) {
    let spinor = gs.method == "ghf";
    if !spinor {
        let channels = if gs.method == "uhf" { 2 } else { 1 };
        let degen = if channels == 1 { 2.0 } else { 1.0 };
        return (false, channels, degen, vec![[0., 0., 1.]; hub.shells.len()]);
    }
    let d = &gs.densities[0];
    let mut axes = Vec::with_capacity(hub.shells.len());
    let mut magnetic = false;
    for shell in &hub.shells {
        let q = shell.q.map(|v| C64::new(v, 0.));
        let uu = (&q.transpose() * d.view((0, 0), (m, m)) * &q).trace().re;
        let dd = (&q.transpose() * d.view((m, m), (m, m)) * &q).trace().re;
        let ud = (&q.transpose() * d.view((0, m), (m, m)) * &q).trace();
        let du = (&q.transpose() * d.view((m, 0), (m, m)) * &q).trace();
        let mx = (ud + du).re;
        let my = (C64::new(0., 1.) * (ud - du)).re;
        let mz = uu - dd;
        let norm = (mx * mx + my * my + mz * mz).sqrt();
        if norm > 1e-4 {
            magnetic = true;
            axes.push([mx / norm, my / norm, mz / norm]);
        } else {
            axes.push([0., 0., 1.]);
        }
    }
    let channels = if magnetic { 2 } else { 1 };
    (true, channels, 1.0, axes)
}

fn build_shift_matrices(
    hub: &MolecularHubbard,
    m: usize,
    spinor: bool,
    channels: usize,
    axes: &[[f64; 3]],
    shifts: &[f64],
) -> Vec<Matrix> {
    if spinor {
        let mut op = Matrix::zeros(2 * m, 2 * m);
        for (i, shell) in hub.shells.iter().enumerate() {
            let q = shell.q.map(|v| C64::new(v, 0.));
            let proj = &q * q.transpose();
            let add_block = |op: &mut Matrix, r0: usize, c0: usize, s: C64| {
                let mut block = op.view_mut((r0, c0), (m, m));
                block += &proj * s;
            };
            if channels == 1 {
                let a = C64::new(shifts[i], 0.);
                add_block(&mut op, 0, 0, a);
                add_block(&mut op, m, m, a);
            } else {
                let (ap, am) = (shifts[2 * i], shifts[2 * i + 1]);
                let avg = 0.5 * (ap + am);
                let diff = 0.5 * (ap - am);
                let [mx, my, mz] = axes[i];
                add_block(&mut op, 0, 0, C64::new(avg + diff * mz, 0.));
                add_block(&mut op, m, m, C64::new(avg - diff * mz, 0.));
                add_block(&mut op, 0, m, C64::new(diff * mx, -diff * my));
                add_block(&mut op, m, 0, C64::new(diff * mx, diff * my));
            }
        }
        return vec![op];
    }
    let mut ops = vec![Matrix::zeros(m, m); channels];
    for (i, shell) in hub.shells.iter().enumerate() {
        let q = shell.q.map(|v| C64::new(v, 0.));
        let proj = &q * q.transpose();
        for s in 0..channels {
            let alpha = shifts[i * channels + s];
            if alpha != 0. {
                ops[s] += &proj * C64::new(alpha, 0.);
            }
        }
    }
    ops
}

fn extract_shell_occupations(
    hub: &MolecularHubbard,
    d: &[Matrix],
    m: usize,
    spinor: bool,
    channels: usize,
    degen: f64,
    axes: &[[f64; 3]],
) -> Vec<f64> {
    let mut out = Vec::with_capacity(hub.shells.len() * channels);
    if spinor {
        let dm = &d[0];
        for (i, shell) in hub.shells.iter().enumerate() {
            let q = shell.q.map(|v| C64::new(v, 0.));
            let uu = (&q.transpose() * dm.view((0, 0), (m, m)) * &q).trace().re;
            let dd = (&q.transpose() * dm.view((m, m), (m, m)) * &q).trace().re;
            if channels == 1 {
                out.push(uu + dd);
            } else {
                let ud = (&q.transpose() * dm.view((0, m), (m, m)) * &q).trace();
                let du = (&q.transpose() * dm.view((m, 0), (m, m)) * &q).trace();
                let mx = (ud + du).re;
                let my = (C64::new(0., 1.) * (ud - du)).re;
                let mz = uu - dd;
                let n_tot = uu + dd;
                let [hx, hy, hz] = axes[i];
                let m_proj = hx * mx + hy * my + hz * mz;
                out.push(0.5 * (n_tot + m_proj));
                out.push(0.5 * (n_tot - m_proj));
            }
        }
        return out;
    }
    for shell in &hub.shells {
        let q = shell.q.map(|v| C64::new(v, 0.));
        for ds in d.iter().take(channels) {
            out.push(degen * (&q.transpose() * ds * &q).trace().re);
        }
    }
    out
}

/// A linear-response probe measuring molecular Hubbard shell occupations under shifts.
pub struct MolecularProbe<'a> {
    pub integrals: &'a MolecularIntegrals,
    pub ground_state: &'a MeanField,
    pub functional: &'a str,
    pub width: f64,
    pub options: ScfOptions,
    spinor: bool,
    channels: usize,
    degen: f64,
    axes: Vec<[f64; 3]>,
    ground_fock: Vec<Matrix>,
    ground_integer: bool,
}

impl<'a> MolecularProbe<'a> {
    pub fn new(
        integrals: &'a MolecularIntegrals,
        ground_state: &'a MeanField,
        functional: &'a str,
    ) -> Result<Self> {
        let hub = integrals
            .corrections
            .as_ref()
            .and_then(|c| c.hubbard.as_ref())
            .ok_or_else(|| {
                invalid("molecular hubbard_response requires a molecule with hubbard_u shells")
            })?;
        if !ground_state.converged {
            return Err(invalid(
                "molecular hubbard_response requires a converged SCF reference",
            ));
        }
        let precision = match integrals.precision() {
            crate::math::storage::Precision::Float64 => "float64",
            crate::math::storage::Precision::Float32 => "float32",
        };
        crate::models::hubbard_request::check_scope(functional, precision)?;

        let m = integrals.m();
        let (spinor, channels, degen, axes) = response_layout(hub, ground_state, m);
        let ground_fock = if spinor {
            vec![integrals.spinor_kohn_sham_fock(functional, &ground_state.densities[0])?.0]
        } else {
            integrals.kohn_sham_fock(functional, &ground_state.densities)?.0
        };
        let width = ground_state
            .smearing_width
            .unwrap_or(occupation::DEFAULT_WIDTH);
        let ground_integer = ground_state.occupations.iter().all(|o| occupation::is_integer(o));
        Ok(Self {
            integrals,
            ground_state,
            functional,
            width,
            options: ScfOptions {
                tolerance: 1e-8,
                guesses: 1,
                ..ScfOptions::default()
            },
            spinor,
            channels,
            degen,
            axes,
            ground_fock,
            ground_integer,
        })
    }
}

impl<'a> Probe for MolecularProbe<'a> {
    fn shells(&self) -> usize {
        self.integrals
            .corrections
            .as_ref()
            .and_then(|c| c.hubbard.as_ref())
            .map_or(0, |h| h.shells.len())
    }

    fn channels(&self) -> usize {
        self.channels
    }

    fn frozen(&mut self, shifts: &[f64]) -> Result<Vec<f64>> {
        let hub = self
            .integrals
            .corrections
            .as_ref()
            .and_then(|c| c.hubbard.as_ref())
            .expect("checked in new");
        let m = self.integrals.m();
        let s_ops = build_shift_matrices(hub, m, self.spinor, self.channels, &self.axes, shifts);
        let occ_counts: Vec<usize> = if self.spinor {
            vec![self.ground_state.particles[0] + self.ground_state.particles[1]]
        } else if self.channels == 1 {
            vec![self.ground_state.particles[0]]
        } else {
            self.ground_state.particles.to_vec()
        };
        let mut densities = Vec::with_capacity(self.ground_fock.len());
        for ((f0, sh), &n) in self.ground_fock.iter().zip(&s_ops).zip(&occ_counts) {
            let (levels, coeff) = crate::math::linalg::eigen(f0 + sh)?;
            let occ = occupation::fermi_dirac(&levels, n, self.width)?;
            densities.push(occupation::density(&coeff, &occ));
        }
        Ok(extract_shell_occupations(
            hub,
            &densities,
            m,
            self.spinor,
            self.channels,
            self.degen,
            &self.axes,
        ))
    }

    fn scf(&mut self, shifts: &[f64]) -> Result<Vec<f64>> {
        let hub = self
            .integrals
            .corrections
            .as_ref()
            .and_then(|c| c.hubbard.as_ref())
            .expect("checked in new");
        let m = self.integrals.m();
        let s_ops = build_shift_matrices(hub, m, self.spinor, self.channels, &self.axes, shifts);
        let res = if self.spinor {
            self.integrals.spinor_kohn_sham_shifted(
                self.functional,
                self.ground_state,
                Some(&s_ops[0]),
                Some(self.width),
                &self.options,
            )?
        } else {
            self.integrals.kohn_sham_from(
                self.functional,
                self.ground_state,
                Some(&s_ops),
                Some(self.width),
                &self.options,
            )?
        };
        if !res.converged {
            return Err(invalid(format!(
                "molecular SCF under Hubbard shift failed to converge after {} iterations",
                self.options.max_iterations
            )));
        }
        if self.ground_integer && res.aufbau_inversion.is_some() {
            return Err(invalid(
                "Hubbard shift alpha crossed a frontier gap on an integer-occupied molecule; use smaller alphas or a smeared reference",
            ));
        }
        Ok(extract_shell_occupations(
            hub,
            &res.densities,
            m,
            self.spinor,
            self.channels,
            self.degen,
            &self.axes,
        ))
    }
}

/// Result of molecular linear-response Hubbard U measurement (`MP-7`).
#[derive(Clone, Debug)]
pub struct ResponseResult {
    pub u_matrix: DMatrix<f64>,
    pub u_charge: DMatrix<f64>,
    pub chi0: DMatrix<f64>,
    pub chi: DMatrix<f64>,
    pub alphas: Vec<f64>,
    pub j_response: Option<Vec<f64>>,
    pub symbols: Vec<String>,
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

    /// Groups shells by element symbol checking spread tolerance via `models::hubbard_response`.
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

/// Analytic non-interacting response $\chi^0$ via sum-over-states (`StateBlock`) on a molecule.
pub fn sum_over_states_chi0(
    integrals: &MolecularIntegrals,
    ground_state: &MeanField,
    width: f64,
) -> Result<DMatrix<f64>> {
    let hub = integrals
        .corrections
        .as_ref()
        .and_then(|c| c.hubbard.as_ref())
        .ok_or_else(|| {
            invalid("sum_over_states_chi0 requires a molecule with hubbard_u shells")
        })?;
    let m = integrals.m();
    let (spinor, channels, degen, axes) = response_layout(hub, ground_state, m);
    let shells = hub.shells.len();
    let mu_fermi = |levels: &[f64], occ: &[f64]| -> f64 {
        let mut pairs: Vec<(f64, f64)> = levels.iter().copied().zip(occ.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        if let Some(pos) = pairs.iter().position(|(_, f)| *f < 0.5) {
            if pos > 0 {
                0.5 * (pairs[pos - 1].0 + pairs[pos].0)
            } else {
                pairs[0].0 - 10.0 * width
            }
        } else {
            pairs.last().map_or(0.0, |(e, _)| e + 10.0 * width)
        }
    };

    if ground_state.densities.len() == 1 {
        let c = &ground_state.coefficients[0];
        let levels = &ground_state.energies[0];
        let n_orb = levels.len();
        let occ = &ground_state.occupations[0];
        let mu = mu_fermi(levels, occ);
        let slopes: Vec<f64> = levels
            .iter()
            .map(|e| Smearing::FermiDirac.slope((e - mu) / width) / width)
            .collect();
        let mut matrices = Vec::with_capacity(shells * channels);
        for (i, shell) in hub.shells.iter().enumerate() {
            let q = shell.q.map(|v| C64::new(v, 0.));
            if !spinor {
                let a = c.adjoint() * &q;
                let mat = &a * a.adjoint();
                matrices.push(
                    (0..n_orb)
                        .flat_map(|r| (0..n_orb).map(move |s| (r, s)))
                        .map(|(r, s)| mat[(r, s)])
                        .collect(),
                );
            } else {
                let u_a = c.rows(0, m).adjoint() * &q;
                let d_a = c.rows(m, m).adjoint() * &q;
                let uu = &u_a * u_a.adjoint();
                let dd = &d_a * d_a.adjoint();
                if channels == 1 {
                    let mat = &uu + &dd;
                    matrices.push(
                        (0..n_orb)
                            .flat_map(|r| (0..n_orb).map(move |s| (r, s)))
                            .map(|(r, s)| mat[(r, s)])
                            .collect(),
                    );
                } else {
                    let ud = &u_a * d_a.adjoint();
                    let du = &d_a * u_a.adjoint();
                    let [mx, my, mz] = axes[i];
                    for ch in 0..2 {
                        let sgn = if ch == 0 { 1.0 } else { -1.0 };
                        let mat = (&uu + &dd) * C64::new(0.5, 0.0)
                            + (&uu - &dd) * C64::new(0.5 * sgn * mz, 0.0)
                            + &ud * C64::new(0.5 * sgn * mx, -0.5 * sgn * my)
                            + &du * C64::new(0.5 * sgn * mx, 0.5 * sgn * my);
                        matrices.push(
                            (0..n_orb)
                                .flat_map(|r| (0..n_orb).map(move |s| (r, s)))
                                .map(|(r, s)| mat[(r, s)])
                                .collect(),
                        );
                    }
                }
            }
        }
        let slices: Vec<&[C64]> = matrices.iter().map(Vec::as_slice).collect();
        let blocks = [StateBlock {
            weight: 1.0,
            levels,
            occupations: occ,
            slopes: &slopes,
            elements: &slices,
        }];
        return Ok(sum_over_states(&blocks, degen)?.total());
    }

    // Collinear unrestricted (two spin channels, one coupled chemical potential is spin-independent
    // because particle numbers per spin are held fixed independently -> two separate StateBlock runs
    // placed on the block-diagonal channels in `2*shells x 2*shells`? Wait: in Probe each spin N_s is
    // held fixed separately, so each spin solves its own chemical potential -> chi0_spin is diagonal in
    // spin, identically matching two independent single-spin sum-over-states calls!).
    let mut chi0 = DMatrix::zeros(shells * 2, shells * 2);
    for spin in 0..2 {
        let c = &ground_state.coefficients[spin];
        let levels = &ground_state.energies[spin];
        let n_orb = levels.len();
        let occ = &ground_state.occupations[spin];
        let mu = mu_fermi(levels, occ);
        let slopes: Vec<f64> = levels
            .iter()
            .map(|e| Smearing::FermiDirac.slope((e - mu) / width) / width)
            .collect();
        let mut matrices = Vec::with_capacity(shells);
        for shell in &hub.shells {
            let q = shell.q.map(|v| C64::new(v, 0.));
            let a = c.adjoint() * &q;
            let mat = &a * a.adjoint();
            matrices.push(
                (0..n_orb)
                    .flat_map(|r| (0..n_orb).map(move |s| (r, s)))
                    .map(|(r, s)| mat[(r, s)])
                    .collect::<Vec<_>>(),
            );
        }
        let slices: Vec<&[C64]> = matrices.iter().map(Vec::as_slice).collect();
        let blocks = [StateBlock {
            weight: 1.0,
            levels,
            occupations: occ,
            slopes: &slopes,
            elements: &slices,
        }];
        let chi_s = sum_over_states(&blocks, 1.0)?.total();
        for i in 0..shells {
            for j in 0..shells {
                chi0[(i * 2 + spin, j * 2 + spin)] = chi_s[(i, j)];
            }
        }
    }
    let _ = (DEGENERATE, NO_SLOPE);
    Ok(chi0)
}

/// Evaluates molecular linear-response Hubbard `U` parameters with default fit options.
pub fn hubbard_response(
    integrals: &MolecularIntegrals,
    ground_state: &MeanField,
    functional: &str,
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
    hubbard_response_with_options(integrals, ground_state, functional, &options)
}

/// Evaluates molecular linear-response Hubbard `U` parameters with custom response options.
pub fn hubbard_response_with_options(
    integrals: &MolecularIntegrals,
    ground_state: &MeanField,
    functional: &str,
    options: &Options,
) -> Result<ResponseResult> {
    let mut probe = MolecularProbe::new(integrals, ground_state, functional)?;
    let response = model_hr::measure(&integrals.context, &mut probe, options)?;
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
    let symbols = integrals
        .corrections
        .as_ref()
        .and_then(|c| c.hubbard.as_ref())
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
        response,
    })
}
