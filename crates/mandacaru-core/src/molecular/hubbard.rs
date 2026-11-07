//! DFT+U on-site correction for molecular PAW-LCAO Kohn-Sham states.
use crate::{
    models::hubbard::{self as model, Form, Hubbard, Spin},
    Context, Result,
};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

type Matrix = DMatrix<C64>;

/// One correlated atomic shell of a molecule.
#[derive(Clone)]
pub struct MolecularHubbardShell {
    pub atom: usize,
    pub symbol: String,
    pub l: usize,
    /// Hartree.
    pub u: f64,
    /// Hartree.
    pub j: f64,
    pub weights: Vec<f64>,
    /// Subspace projector in the AO basis, shape `(m, 2l+1)`.
    pub q_ao: DMatrix<f64>,
    /// Subspace projector in the Loewdin orthonormal basis, shape `(m, 2l+1)`.
    pub q: DMatrix<f64>,
}

impl MolecularHubbardShell {
    pub fn size(&self) -> usize {
        2 * self.l + 1
    }
    pub fn u_eff(&self) -> f64 {
        self.u - self.j
    }
}

/// The molecular Hubbard correction container.
#[derive(Clone)]
pub struct MolecularHubbard {
    pub form: Form,
    pub shells: Vec<MolecularHubbardShell>,
}

/// The converged report data of one shell for snapshot export.
#[derive(Clone)]
pub struct MolecularHubbardReport {
    pub atom: usize,
    pub symbol: String,
    pub l: usize,
    pub u_ha: f64,
    pub j_ha: f64,
    pub u_eff_ha: f64,
    pub energy_ha: f64,
    pub trace: Vec<f64>,
    /// Occupation matrix in the complex spherical harmonic basis `m = -l..l`.
    /// Shape `(spins, 2l+1, 2l+1)`.
    pub n: Vec<Matrix>,
}

/// Unitary matrix `U` converting real (Cartesian) spherical harmonics `m = -l..l`
/// into Condon-Shortley complex spherical harmonics `m = -l..l`, such that
/// `Y_lm = sum_{m'} U_{m, m'} S_lm'`.
pub fn unitary_real_to_complex(l: usize) -> DMatrix<C64> {
    let p = 2 * l + 1;
    let mut u = DMatrix::<C64>::zeros(p, p);
    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
    for m in -(l as i32)..=(l as i32) {
        let row = (m + l as i32) as usize;
        if m == 0 {
            u[(row, l)] = C64::new(1.0, 0.0);
        } else if m > 0 {
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
            let col_pos = (m + l as i32) as usize;
            let col_neg = (-m + l as i32) as usize;
            u[(row, col_pos)] = C64::new(sign * inv_sqrt2, 0.0);
            u[(row, col_neg)] = C64::new(0.0, sign * inv_sqrt2);
        } else {
            let k = m.abs();
            let col_pos = (k + l as i32) as usize;
            let col_neg = (-k + l as i32) as usize;
            u[(row, col_pos)] = C64::new(inv_sqrt2, 0.0);
            u[(row, col_neg)] = C64::new(0.0, -inv_sqrt2);
        }
    }
    u
}

impl MolecularHubbard {
    /// Evaluates the Hubbard energy `E_U`, per-spin Fock additions `Delta F_s = Q V^s Q^T`,
    /// and per-shell report data at the orthonormal Loewdin densities `d`.
    pub fn evaluate(
        &self,
        ctx: &Context,
        d: &[Matrix],
    ) -> Result<(f64, Vec<DMatrix<f64>>, Vec<MolecularHubbardReport>)> {
        let spins = d.len();
        let m = d[0].nrows();
        let mut total_energy = 0.0;
        let mut delta_focks = vec![DMatrix::<f64>::zeros(m, m); spins];
        let mut reports = Vec::with_capacity(self.shells.len());

        let spin = if spins == 1 {
            Spin::Restricted
        } else {
            Spin::Unrestricted
        };

        for shell in &self.shells {
            let p = shell.size();
            let mut shell_n_real = Vec::with_capacity(spins);
            for s in 0..spins {
                let ds_real = d[s].map(|v| v.re);
                let n_real = shell.q.transpose() * &ds_real * &shell.q;
                shell_n_real.push(n_real);
            }

            let h_model = Hubbard {
                form: self.form,
                ..Hubbard::new(shell.u, shell.j)?
            };

            let n_channels: Vec<Vec<C64>> = shell_n_real
                .iter()
                .map(|nr| {
                    (0..p * p)
                        .map(|idx| C64::new(nr[(idx / p, idx % p)], 0.0))
                        .collect()
                })
                .collect();
            let channel_refs: Vec<&[C64]> = n_channels.iter().map(|v| v.as_slice()).collect();

            let eval = model::evaluate(ctx, &h_model, shell.l, spin, &channel_refs)?;
            total_energy += eval.energy;

            for s in 0..spins {
                let mut v_real = DMatrix::<f64>::zeros(p, p);
                let pot = eval.potentials[s].as_slice();
                for a in 0..p {
                    for b in 0..p {
                        v_real[(a, b)] = pot[a * p + b].re;
                    }
                }
                delta_focks[s] += &shell.q * v_real * shell.q.transpose();
            }

            // Report in complex spherical harmonics
            let u_mat = unitary_real_to_complex(shell.l);
            let mut complex_n = Vec::with_capacity(spins);
            let mut traces = Vec::with_capacity(spins);
            for s in 0..spins {
                let n_cplx_real = shell_n_real[s].map(|v| C64::new(v, 0.0));
                let nc = &u_mat * n_cplx_real * u_mat.adjoint();
                traces.push((0..p).map(|i| nc[(i, i)].re).sum());
                complex_n.push(nc);
            }

            reports.push(MolecularHubbardReport {
                atom: shell.atom,
                symbol: shell.symbol.clone(),
                l: shell.l,
                u_ha: shell.u,
                j_ha: shell.j,
                u_eff_ha: shell.u_eff(),
                energy_ha: eval.energy,
                trace: traces,
                n: complex_n,
            });
        }

        Ok((total_energy, delta_focks, reports))
    }
}
