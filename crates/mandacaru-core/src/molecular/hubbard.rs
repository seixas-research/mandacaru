//! DFT+U on-site correction for molecular PAW-LCAO Kohn-Sham states.
use super::invalid;
use crate::{
    models::hubbard::{self as model, Form, Hubbard, Spin},
    Context, Result,
};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

type Matrix = DMatrix<C64>;

/// Largest imaginary part of a +U potential of a real occupation matrix, relative to the
/// largest entry (and 1): rounding of the basis change is 1e-15.
const POTENTIAL_IMAGINARY: f64 = 1e-10;

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
    /// The model's [`Hubbard`] of `shell`, in this container's form.
    fn parameters(&self, shell: &MolecularHubbardShell) -> Result<Hubbard> {
        Ok(Hubbard {
            form: self.form,
            ..Hubbard::new(shell.u, shell.j)?
        })
    }

    /// The model evaluated at occupation matrices given in the **real** (Cartesian) harmonics
    /// of the shell's projector, the one place that bridges the two bases. `E_U` and `V` come
    /// back in the same real harmonics.
    ///
    /// The Liechtenstein tensor is written in the complex harmonics (`models::hubbard`), so its
    /// `n` is `U n U^dagger` and its `V` returns as `U^dagger V U`, `U` being
    /// [`unitary_real_to_complex`]; without the rotation the energy of a molecule depends on its
    /// orientation (`J != 0`). The Dudarev form is invariant under a unitary change of basis and
    /// is evaluated on the matrices as they are, so its numbers are untouched.
    fn evaluate_real_harmonics(
        &self,
        ctx: &Context,
        shell: &MolecularHubbardShell,
        spin: Spin,
        channels: &[DMatrix<C64>],
    ) -> Result<(f64, Vec<DMatrix<C64>>)> {
        let p = shell.size();
        let rotation =
            (self.form == Form::LiechtensteinFll).then(|| unitary_real_to_complex(shell.l));
        let flat: Vec<Vec<C64>> = channels
            .iter()
            .map(|n| {
                let n = match &rotation {
                    Some(u) => u * n * u.adjoint(),
                    None => n.clone(),
                };
                (0..p * p).map(|i| n[(i / p, i % p)]).collect()
            })
            .collect();
        let refs: Vec<&[C64]> = flat.iter().map(|v| v.as_slice()).collect();
        let eval = model::evaluate(ctx, &self.parameters(shell)?, shell.l, spin, &refs)?;
        let potentials = eval
            .potentials
            .iter()
            .map(|v| {
                let v = DMatrix::from_fn(p, p, |a, b| v.as_slice()[a * p + b]);
                match &rotation {
                    Some(u) => u.adjoint() * v * u,
                    None => v,
                }
            })
            .collect();
        Ok((eval.energy, potentials))
    }

    /// The energy of one shell and its per-spin potential `V^s = dE / dn^s` (real, `(2l+1)^2`)
    /// at the real occupation matrices `n` of its spins (one for a restricted state, two for an
    /// unrestricted one): the single evaluation the Fock matrix and the nuclear gradient share.
    pub fn shell_response(
        &self,
        ctx: &Context,
        shell: &MolecularHubbardShell,
        n: &[DMatrix<f64>],
    ) -> Result<(f64, Vec<DMatrix<f64>>)> {
        let spin = if n.len() == 1 {
            Spin::Restricted
        } else {
            Spin::Unrestricted
        };
        let channels: Vec<DMatrix<C64>> = n.iter().map(|m| m.map(|x| C64::new(x, 0.))).collect();
        let (energy, potentials) = self.evaluate_real_harmonics(ctx, shell, spin, &channels)?;
        // A real symmetric `n` is invariant under complex conjugation of the real orbitals, and so
        // is `E_U`: `V` is real. An imaginary part is a defect, never something to drop.
        let scale = potentials.iter().fold(1f64, |t, v| t.max(v.camax()));
        let potentials = potentials
            .iter()
            .map(|v| {
                if v.iter().any(|z| z.im.abs() > POTENTIAL_IMAGINARY * scale) {
                    return Err(invalid(
                        "the +U potential of a real occupation matrix has an imaginary part",
                    ));
                }
                Ok(v.map(|z| z.re))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((energy, potentials))
    }

    /// The change of one shell's per-spin potentials along the occupation changes `changes`
    /// (complex Hermitian; one matrix for a restricted state, two for an unrestricted one):
    /// `V(changes) - V(0)`. Exact, because the energy is quadratic in the occupations (Dudarev and
    /// Liechtenstein) and the potential affine. A complex change is the antisymmetric response
    /// channel: the same expression in a Hermitian occupation matrix.
    pub(crate) fn shell_linear(
        &self,
        ctx: &Context,
        shell: &MolecularHubbardShell,
        changes: &[DMatrix<C64>],
    ) -> Result<Vec<DMatrix<C64>>> {
        let spin = if changes.len() == 1 {
            Spin::Restricted
        } else {
            Spin::Unrestricted
        };
        let zeros: Vec<DMatrix<C64>> = changes
            .iter()
            .map(|c| DMatrix::zeros(c.nrows(), c.ncols()))
            .collect();
        let (_, moved) = self.evaluate_real_harmonics(ctx, shell, spin, changes)?;
        let (_, origin) = self.evaluate_real_harmonics(ctx, shell, spin, &zeros)?;
        Ok(moved.iter().zip(&origin).map(|(a, b)| a - b).collect())
    }
    /// The Hubbard energy `E_U` and the per-spin Fock additions `Delta F_s = Q V^s Q^T` at
    /// orthonormal densities `d` that may be complex (real-time propagation). The occupations
    /// `n^s = Q^T D_s Q` keep their imaginary (antisymmetric) part, which a current through the
    /// shell carries, and the potentials `V^s = dE / dn^s` come back complex Hermitian, through
    /// the same real-harmonics bridge as the SCF ([`Self::evaluate_real_harmonics`]). The energy
    /// is real for Hermitian `n`; an imaginary part is a defect and raises.
    pub fn evaluate_complex(&self, ctx: &Context, d: &[Matrix]) -> Result<(f64, Vec<Matrix>)> {
        let spin = if d.len() == 1 {
            Spin::Restricted
        } else {
            Spin::Unrestricted
        };
        let m = d[0].nrows();
        let mut total = 0.;
        let mut focks = vec![Matrix::zeros(m, m); d.len()];
        for shell in &self.shells {
            let q = shell.q.map(|v| C64::new(v, 0.));
            let n: Vec<Matrix> = d.iter().map(|ds| q.transpose() * ds * &q).collect();
            let (energy, potentials) = self.evaluate_real_harmonics(ctx, shell, spin, &n)?;
            if !energy.is_finite() {
                return Err(invalid("the +U energy of a complex occupation is not finite"));
            }
            total += energy;
            let scale = potentials.iter().fold(1f64, |t, v| t.max(v.camax()));
            for (fock, v) in focks.iter_mut().zip(&potentials) {
                if (v - v.adjoint()).camax() > POTENTIAL_IMAGINARY * scale {
                    return Err(invalid(
                        "the +U potential of a Hermitian occupation matrix is not Hermitian",
                    ));
                }
                *fock += &q * v * q.transpose();
            }
        }
        Ok((total, focks))
    }
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

        for shell in &self.shells {
            let p = shell.size();
            let shell_n_real: Vec<DMatrix<f64>> = d
                .iter()
                .map(|ds| shell.q.transpose() * ds.map(|v| v.re) * &shell.q)
                .collect();
            let (energy, potentials) = self.shell_response(ctx, shell, &shell_n_real)?;
            total_energy += energy;
            for (delta, v) in delta_focks.iter_mut().zip(&potentials) {
                *delta += &shell.q * v * shell.q.transpose();
            }

            // Report in complex spherical harmonics
            let u_mat = unitary_real_to_complex(shell.l);
            let mut complex_n = Vec::with_capacity(spins);
            let mut traces = Vec::with_capacity(spins);
            for n_real in &shell_n_real {
                let nc = &u_mat * n_real.map(|v| C64::new(v, 0.0)) * u_mat.adjoint();
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
                energy_ha: energy,
                trace: traces,
                n: complex_n,
            });
        }

        Ok((total_energy, delta_focks, reports))
    }
}
