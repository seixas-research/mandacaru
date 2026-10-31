//! DFT+U in the periodic SCF: shell tables, occupations, energy and per-spin operators.
//!
//! The correction acts on the occupation matrix `n^sigma = W^T R^sigma W` of a few shells
//! (`R^sigma = sum_k w_k C^dagger D^sigma_k C` over the shell's projectors, `W` the PAW
//! partial-wave weights of `paw::shell`). `models::hubbard` supplies `E_U(n)` and
//! `V = dE_U/dn`; the operator `O = W V W^T` is scattered into the `p x p` projector space and
//! enters the Fock matrix through the same `C O C^dagger` seam as the PAW moment operator.
//!
//! Energies are Hartree, like the rest of the snapshot. A restricted run holds the spin-summed
//! density, so each spin channel gets `n = (W^T R W) / 2` and `E_U` counts both.
//!
//! Persistence: the snapshot JSON carries a self-contained `hubbard` object
//! (`form`, `shells[{atom, symbol, l, u, j, projectors, weights, projector_basis, seed}]`; `form` is
//! a [`Form`] name). The table is built from the user's option by `hubbard_table`. Projectors are
//! global indices ordered `m = -l..l` then radial index; `seed` is `{n_electrons, moment}` or
//! explicit per-spin occupations (`{occupations: [spin][p*p][re, im]}`).
use super::{
    input::{count, number},
    invalid, Crystal,
};
use crate::{
    math::storage::{MatrixSource, Precision, StoredMatrix},
    models::hubbard::{self as model, Form, Hubbard, Spin},
    paw::shell,
    product,
    quantum::C64,
    Result,
};
use nalgebra::DMatrix;
use serde_json::{json, Value};
use std::mem::size_of;

type Matrix = DMatrix<C64>;
pub const MAX_SHELLS: usize = 4096;
const MAX_WEIGHTS: usize = 8;
/// Heap bytes one parsed shell holds at most (logical sizes): eight weights, the `7 x 8`
/// projector indices of an f shell and a two-spin explicit seed. `Spec::parse` takes no
/// `Context`: this bound times [`MAX_SHELLS`] (about 9 MB, 2x with `Vec` slack) is all the table
/// adds outside the budget.
pub const MAX_SHELL_BYTES: usize = size_of::<Shell>()
    + MAX_WEIGHTS * size_of::<f64>()
    + 7 * MAX_WEIGHTS * size_of::<usize>()
    + 2 * 49 * size_of::<C64>();
/// Bound on the table of a snapshot ([`MAX_SHELL_BYTES`] per shell).
pub const MAX_TABLE_BYTES: usize = MAX_SHELLS * MAX_SHELL_BYTES;

/// Where the occupation matrices start: a diagonal seed cannot break orbital degeneracy.
#[derive(Debug, Clone, PartialEq)]
pub enum Seed {
    /// `n_sigma = clamp((N +- moment) / (2 (2l+1)), 0, 1)` times the identity; the first spin
    /// channel takes `+ moment`, the shell's own signed moment (an antiferromagnetic seed gives
    /// its sites opposite signs). A restricted run has none: `N / (2 (2l+1))`; a nonzero moment
    /// there is refused at parse time.
    Moment { n_electrons: f64, moment: f64 },
    /// One Hermitian `p x p` row-major matrix per spin channel.
    Occupations(Vec<Vec<C64>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shell {
    pub atom: usize,
    pub symbol: String,
    pub l: usize,
    /// Hartree.
    pub u: f64,
    pub j: f64,
    /// Global projector indices, `m = -l..l` then radial index.
    pub projectors: Vec<usize>,
    /// One partial-wave weight per radial index (`paw::shell::weights`).
    pub weights: Vec<f64>,
    /// Informational: the projector basis the weights were taken in.
    pub basis: Option<String>,
    pub seed: Seed,
}

impl Shell {
    pub fn size(&self) -> usize {
        2 * self.l + 1
    }
    /// `U - J` in Hartree.
    pub fn u_eff(&self) -> f64 {
        self.u - self.j
    }
    fn parse(v: &Value, projectors: usize, spins: usize) -> Result<Self> {
        let l = count(&v["l"], "Hubbard shell l")?;
        if !(2..=3).contains(&l) {
            return Err(invalid(format!(
                "Hubbard shells are d or f (l = 2 or 3), not l = {l}"
            )));
        }
        let (u, j) = (number(&v["u"], "Hubbard U")?, number(&v["j"], "Hubbard J")?);
        Hubbard::new(u, j)?;
        let weights = v["weights"]
            .as_array()
            .filter(|w| (1..=MAX_WEIGHTS).contains(&w.len()))
            .ok_or_else(|| invalid("Hubbard weights need one to eight values"))?
            .iter()
            .map(|w| number(w, "Hubbard weight"))
            .collect::<Result<Vec<_>>>()?;
        let listed = v["projectors"]
            .as_array()
            .filter(|p| p.len() == (2 * l + 1) * weights.len())
            .ok_or_else(|| {
                invalid(format!(
                    "a Hubbard shell with l = {l} and {} radial projectors needs {} projector indices",
                    weights.len(),
                    (2 * l + 1) * weights.len()
                ))
            })?
            .iter()
            .map(|i| count(i, "Hubbard projector index"))
            .collect::<Result<Vec<_>>>()?;
        if listed.iter().any(|i| *i >= projectors) {
            return Err(invalid(format!(
                "a Hubbard projector index is outside the basis ({projectors} projectors)"
            )));
        }
        let seed = &v["seed"];
        let seed = if seed["occupations"].is_null() {
            let n_electrons = number(&seed["n_electrons"], "Hubbard seed electrons")?;
            if !(0. ..=(4 * l + 2) as f64).contains(&n_electrons) {
                return Err(invalid("Hubbard seed electrons must lie in [0, 2(2l+1)]"));
            }
            let moment = number(&seed["moment"], "Hubbard seed moment")?;
            if spins == 1 && moment != 0. {
                return Err(invalid(format!(
                    "a restricted run has no spin polarization: the Hubbard seed moment {moment} \
                     of a shell must be 0 (use an unrestricted crystal)"
                )));
            }
            Seed::Moment {
                n_electrons,
                moment,
            }
        } else {
            let channels = seed["occupations"]
                .as_array()
                .filter(|a| a.len() == spins)
                .ok_or_else(|| invalid("Hubbard seed needs one occupation matrix per spin"))?;
            let p = 2 * l + 1;
            let matrices = channels
                .iter()
                .map(|channel| {
                    let pairs = channel
                        .as_array()
                        .filter(|a| a.len() == p * p)
                        .ok_or_else(|| invalid("Hubbard seed matrix has the wrong size"))?;
                    let n = pairs
                        .iter()
                        .map(|z| {
                            let z = z
                                .as_array()
                                .filter(|z| z.len() == 2)
                                .ok_or_else(|| invalid("Hubbard seed needs [re, im] pairs"))?;
                            Ok(C64::new(
                                number(&z[0], "Hubbard seed")?,
                                number(&z[1], "Hubbard seed")?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    model::validate(l, &n)?;
                    Ok(n)
                })
                .collect::<Result<Vec<_>>>()?;
            Seed::Occupations(matrices)
        };
        Ok(Self {
            atom: count(&v["atom"], "Hubbard atom")?,
            symbol: v["symbol"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| invalid("a Hubbard shell needs an element symbol"))?
                .to_string(),
            l,
            u,
            j,
            projectors: listed,
            weights,
            basis: v["projector_basis"].as_str().map(str::to_string),
            seed,
        })
    }
    fn to_json(&self) -> Value {
        let seed = match &self.seed {
            Seed::Moment {
                n_electrons,
                moment,
            } => json!({"n_electrons": n_electrons, "moment": moment}),
            Seed::Occupations(m) => json!({"occupations": m.iter().map(|n| n.iter()
                .map(|z| [z.re, z.im]).collect::<Vec<_>>()).collect::<Vec<_>>()}),
        };
        json!({"atom": self.atom, "symbol": self.symbol, "l": self.l, "u": self.u, "j": self.j,
               "projectors": self.projectors, "weights": self.weights, "projector_basis": self.basis,
               "seed": seed})
    }
}

/// The `hubbard` object of a snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    pub form: Form,
    pub shells: Vec<Shell>,
}

/// Occupations and the converged per-spin operators, as the SCF leaves them.
pub struct Report {
    pub shells: usize,
    /// `[spin * shells + shell]`, each `(2l+1)^2` row-major and Hermitian.
    pub occupations: Vec<Vec<C64>>,
    /// The `p x p` operator of each spin channel that the last diagonalization used.
    pub operators: Vec<Matrix>,
}

impl Report {
    pub fn occupation(&self, spin: usize, shell: usize) -> &[C64] {
        &self.occupations[spin * self.shells + shell]
    }
}

impl Spec {
    /// Validates the `hubbard` value of a snapshot with `projectors` projectors and `spins` spin
    /// channels; `None` when the key is absent. Combinations the engine does not implement raise.
    pub fn parse(
        value: &Value,
        projectors: usize,
        spins: usize,
        spinor: bool,
        precision: Precision,
        permutations: usize,
    ) -> Result<Option<Self>> {
        if value.is_null() {
            return Ok(None);
        }
        if spinor {
            return Err(invalid(
                "DFT+U is not implemented for spin-orbit (spinor) crystals",
            ));
        }
        if precision != Precision::Float64 {
            return Err(invalid("DFT+U needs float64 precision"));
        }
        if permutations != 0 {
            return Err(invalid(
                "DFT+U needs the shell occupations symmetrized with the density, and the symmetry table of this \
                 crystal has no such map yet: use a full k mesh without symmetry (symmetry=False)",
            ));
        }
        let form = Form::parse(
            value["form"]
                .as_str()
                .ok_or_else(|| invalid("a Hubbard object needs a form name"))?,
        )?;
        let listed = value["shells"]
            .as_array()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("a Hubbard object needs at least one shell"))?;
        if listed.len() > MAX_SHELLS {
            return Err(invalid(format!(
                "a Hubbard object holds at most {MAX_SHELLS} shells, not {}",
                listed.len()
            )));
        }
        let shells = listed
            .iter()
            .map(|s| Shell::parse(s, projectors, spins))
            .collect::<Result<Vec<_>>>()?;
        let mut seen = vec![false; projectors];
        for (i, shell) in shells.iter().enumerate() {
            if shells[..i]
                .iter()
                .any(|o| o.atom == shell.atom && o.l == shell.l)
            {
                return Err(invalid("one Hubbard shell per atom and l"));
            }
            for &index in &shell.projectors {
                if std::mem::replace(&mut seen[index], true) {
                    return Err(invalid("Hubbard shells share a projector"));
                }
            }
        }
        Ok(Some(Self { form, shells }))
    }
    /// Logical heap bytes of the table: at most [`MAX_TABLE_BYTES`] for a table that parsed.
    pub fn table_bytes(&self) -> usize {
        let seed = |s: &Shell| match &s.seed {
            Seed::Moment { .. } => 0,
            Seed::Occupations(m) => m.iter().map(Vec::len).sum::<usize>() * size_of::<C64>(),
        };
        (self.shells.iter())
            .map(|s| {
                size_of::<Shell>()
                    + s.projectors.len() * size_of::<usize>()
                    + s.weights.len() * size_of::<f64>()
                    + seed(s)
            })
            .sum()
    }
    pub fn to_json(&self) -> Value {
        json!({"form": self.form.name(), "shells": self.shells.iter().map(Shell::to_json).collect::<Vec<_>>()})
    }
    /// Sets `U` and `J` (Hartree) on every shell.
    pub fn set_interaction(&mut self, u: f64, j: f64) -> Result<()> {
        Hubbard::new(u, j)?;
        for shell in &mut self.shells {
            shell.u = u;
            shell.j = j;
        }
        Ok(())
    }
    /// False when `U_eff = 0` on every shell: the SCF then runs exactly as plain DFT.
    pub fn is_active(&self) -> bool {
        self.shells.iter().any(|s| s.u - s.j != 0.)
    }
    /// Bytes of the occupations and operators held during the SCF.
    pub fn bytes(&self, c: &Crystal) -> Result<usize> {
        let numbers: usize = self.shells.iter().map(|s| s.size() * s.size()).sum();
        product(&[c.spins, numbers, 16, 4])?
            .checked_add(product(&[c.spins, c.p, c.p, 16, 3])?)
            .ok_or(crate::Error::Overflow)
    }
    /// The starting occupations, `[spin * shells + shell]`.
    pub fn seed(&self, spins: usize) -> Result<Vec<Vec<C64>>> {
        let k = self.shells.len();
        let mut out = vec![vec![]; spins * k];
        for (index, shell) in self.shells.iter().enumerate() {
            let p = shell.size();
            for spin in 0..spins {
                out[spin * k + index] = match &shell.seed {
                    Seed::Moment {
                        n_electrons,
                        moment,
                    } => {
                        let sign = if spin == 0 { 1. } else { -1. };
                        let total = if spins == 1 {
                            *n_electrons
                        } else {
                            n_electrons + sign * moment
                        };
                        let value = (total / (2 * p) as f64).clamp(0., 1.);
                        (0..p * p)
                            .map(|i| C64::new(if i / p == i % p { value } else { 0. }, 0.))
                            .collect()
                    }
                    Seed::Occupations(matrices) => {
                        if matrices.len() != spins {
                            return Err(invalid("Hubbard seed needs one matrix per spin"));
                        }
                        model::validate(shell.l, &matrices[spin])?;
                        matrices[spin].clone()
                    }
                };
            }
        }
        Ok(out)
    }
    /// `n^sigma = W^T R^sigma W` of every shell from the band density matrices, `[spin * shells + shell]`.
    pub fn occupations(&self, c: &Crystal, matrices: &[StoredMatrix]) -> Result<Vec<Vec<C64>>> {
        let (ctx, nk, k) = (&c.context, c.points.len(), self.shells.len());
        let widest = self
            .shells
            .iter()
            .map(|s| s.projectors.len())
            .max()
            .unwrap_or(0);
        let _work = ctx.reserve(product(&[c.m, widest, 16, 6])?)?;
        let mut out = Vec::with_capacity(c.spins * k);
        for spin in 0..c.spins {
            let mut projected: Vec<Matrix> = self
                .shells
                .iter()
                .map(|s| Matrix::zeros(s.projectors.len(), s.projectors.len()))
                .collect();
            for (index, data) in c.points.iter().enumerate() {
                let d = matrices[spin * nk + index].matrix(ctx)?;
                for (shell, sum) in self.shells.iter().zip(&mut projected) {
                    let columns = Matrix::from_fn(c.m, shell.projectors.len(), |a, b| {
                        data.projections[(a, shell.projectors[b])]
                    });
                    // The restricted density holds both spins: each channel gets half.
                    let weight = C64::new(data.weight / c.degeneracy(), 0.);
                    *sum += columns.adjoint() * (&*d * &columns) * weight;
                }
            }
            for (shell, sum) in self.shells.iter().zip(&projected) {
                let n = shell::occupation(shell.l, &shell.weights, sum)?;
                let p = shell.size();
                out.push(
                    (0..p * p)
                        .map(|i| 0.5 * (n[(i / p, i % p)] + n[(i % p, i / p)].conj()))
                        .collect(),
                );
            }
        }
        Ok(out)
    }
    fn evaluate(&self, c: &Crystal, n: &[Vec<C64>], shell: usize) -> Result<model::Evaluation> {
        let k = self.shells.len();
        let s = &self.shells[shell];
        let (spin, channels): (_, Vec<&[C64]>) = if c.spins == 1 {
            (Spin::Restricted, vec![&n[shell]])
        } else {
            (Spin::Unrestricted, vec![&n[shell], &n[k + shell]])
        };
        let hubbard = Hubbard {
            form: self.form,
            ..Hubbard::new(s.u, s.j)?
        };
        model::evaluate(&c.context, &hubbard, s.l, spin, &channels)
    }
    /// `E_U` (Hartree) of the occupations `n`.
    pub fn energy(&self, c: &Crystal, n: &[Vec<C64>]) -> Result<f64> {
        Ok(self.shell_energies(c, n)?.iter().sum())
    }
    /// `E_U` (Hartree) of each shell, in table order; their sum is [`Spec::energy`].
    pub fn shell_energies(&self, c: &Crystal, n: &[Vec<C64>]) -> Result<Vec<f64>> {
        (0..self.shells.len())
            .map(|shell| Ok(self.evaluate(c, n, shell)?.energy))
            .collect()
    }
    /// The `p x p` operator `O^sigma = sum_shells W V^sigma W^T` of each spin channel.
    pub fn operators(&self, c: &Crystal, n: &[Vec<C64>]) -> Result<Vec<Matrix>> {
        let mut out: Vec<Matrix> = (0..c.spins).map(|_| Matrix::zeros(c.p, c.p)).collect();
        for (index, shell) in self.shells.iter().enumerate() {
            let evaluation = self.evaluate(c, n, index)?;
            let (p, w) = (shell.size(), shell.weights.len());
            for (spin, operator) in out.iter_mut().enumerate() {
                let v = evaluation.potentials[spin].as_slice();
                for (a, b) in (0..p).flat_map(|a| (0..p).map(move |b| (a, b))) {
                    for (i, j) in (0..w).flat_map(|i| (0..w).map(move |j| (i, j))) {
                        operator[(shell.projectors[a * w + i], shell.projectors[b * w + j])] +=
                            v[a * p + b] * (shell.weights[i] * shell.weights[j]);
                    }
                }
            }
        }
        Ok(out)
    }
    pub fn report(&self, occupations: Vec<Vec<C64>>, operators: Vec<Matrix>) -> Report {
        Report {
            shells: self.shells.len(),
            occupations,
            operators,
        }
    }
}

/// The occupations as one vector, for the mixer.
pub fn flatten(n: &[Vec<C64>]) -> Vec<C64> {
    n.iter().flatten().copied().collect()
}

/// Inverse of [`flatten`]; the matrices are made exactly Hermitian.
pub fn restore(n: &mut [Vec<C64>], flat: &[C64]) {
    let mut at = 0;
    for matrix in n.iter_mut() {
        let p = (matrix.len() as f64).sqrt().round() as usize;
        let raw = &flat[at..at + p * p];
        for (i, to) in matrix.iter_mut().enumerate() {
            *to = 0.5 * (raw[i] + raw[(i % p) * p + i / p].conj());
        }
        at += p * p;
    }
}
