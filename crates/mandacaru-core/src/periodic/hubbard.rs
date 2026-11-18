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
//! On a k mesh reduced by the crystal's operations or time reversal the occupations are completed
//! over them (`hubbard_symmetry`), so the wedge gives the occupations of the full mesh.
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
    models::{
        hubbard::{self as model, Form, Hubbard, Spin},
        hubbard_liechtenstein as liechtenstein_model,
        hubbard_slater::{Basis, Ratios},
        hubbard_spinor::{self as spinor_model, Layout as SpinorLayout, Occupation as SpinorOcc},
    },
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
/// Most integer starts [`Spec::integer_starts`] offers a stalled run.
pub const MAX_INTEGER_STARTS: usize = 64;
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
    /// there is refused at parse time. A run that this seed leaves unconverged is repeated from
    /// [`Spec::integer_starts`] (`Crystal::solve_with`).
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
    pub ratios: Ratios,
    /// Global projector indices, `m = -l..l` then radial index.
    pub projectors: Vec<usize>,
    /// One partial-wave weight per radial index (`paw::shell::weights`).
    pub weights: Vec<f64>,
    /// Informational: the projector basis the weights were taken in.
    pub basis: Option<String>,
    /// Origin of `(U, J)` (e.g. `"user"`, `"linear-response"`, `"lr-scf"`).
    pub source: Option<String>,
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
    fn parse(v: &Value, projectors: usize, spins: usize, spinor: bool) -> Result<Self> {
        let l = count(&v["l"], "Hubbard shell l")?;
        if !(2..=3).contains(&l) {
            return Err(invalid(format!(
                "Hubbard shells are d or f (l = 2 or 3), not l = {l}"
            )));
        }
        let (u, j) = (number(&v["u"], "Hubbard U")?, number(&v["j"], "Hubbard J")?);
        Hubbard::new(u, j)?;
        let f4_f2 = (!v["f4_f2"].is_null())
            .then(|| number(&v["f4_f2"], "Hubbard F^4/F^2 ratio"))
            .transpose()?;
        let f6_f2 = (!v["f6_f2"].is_null())
            .then(|| number(&v["f6_f2"], "Hubbard F^6/F^2 ratio"))
            .transpose()?;
        let ratios = Ratios::new(l, f4_f2, f6_f2)?;
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
            if spins == 1 && !spinor && moment != 0. {
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
                .filter(|a| a.len() == if spinor { 1 } else { spins })
                .ok_or_else(|| invalid("Hubbard seed needs one occupation matrix per spin"))?;
            let p = 2 * l + 1;
            let side = if spinor { 2 * p } else { p };
            let matrices = channels
                .iter()
                .map(|channel| {
                    let pairs = channel
                        .as_array()
                        .filter(|a| a.len() == side * side)
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
                    validate_occupation(l, spinor, &n)?;
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
            ratios,
            projectors: listed,
            weights,
            basis: v["projector_basis"].as_str().map(str::to_string),
            source: v["U_source"]
                .as_str()
                .or_else(|| v["u_source"].as_str())
                .map(str::to_string),
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
        let mut out = json!({"atom": self.atom, "symbol": self.symbol, "l": self.l, "u": self.u, "j": self.j,
               "projectors": self.projectors, "weights": self.weights, "projector_basis": self.basis,
               "seed": seed});
        if let Some(source) = &self.source {
            out.as_object_mut()
                .expect("shell JSON is an object")
                .insert("U_source".into(), json!(source));
        }
        if Ratios::default_for(self.l).is_ok_and(|d| self.ratios != d) {
            let map = out.as_object_mut().expect("shell JSON is an object");
            map.insert("f4_f2".into(), json!(self.ratios.f4_f2));
            if let Some(f6) = self.ratios.f6_f2 {
                map.insert("f6_f2".into(), json!(f6));
            }
        }
        out
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

fn validate_occupation(l: usize, spinor: bool, n: &[C64]) -> Result<()> {
    if !spinor {
        return model::validate(l, n).map(|_| ());
    }
    let layout = SpinorLayout::new(l, false)?;
    let d = layout.size();
    if n.len() != d * d {
        return Err(invalid(format!(
            "spinor occupation matrix has {} entries, a shell with l = {l} needs {}",
            n.len(),
            d * d
        )));
    }
    if n.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
        return Err(invalid("spinor occupation matrix has a non-finite entry"));
    }
    let scale = n.iter().fold(1f64, |m, x| m.max(x.norm()));
    for a in 0..d {
        for b in a..d {
            if (n[a * d + b] - n[b * d + a].conj()).norm() > model::HERMITIAN_TOLERANCE * scale {
                return Err(invalid(format!(
                    "spinor occupation matrix is not Hermitian at ({a}, {b})"
                )));
            }
        }
    }
    Ok(())
}

impl Report {
    pub fn occupation(&self, spin: usize, shell: usize) -> &[C64] {
        &self.occupations[spin * self.shells + shell]
    }
    /// The trace of the occupation matrix of `shell` (`p x p`, or `2p x 2p` for a spinor,
    /// row-major) in `spin`.
    pub fn occupation_trace(&self, spin: usize, shell: usize, p: usize) -> f64 {
        let n = self.occupation(spin, shell);
        let d = (n.len() as f64).sqrt().round() as usize;
        debug_assert!(d == p || d == 2 * p);
        (0..d).map(|a| n[a * d + a].re).sum::<f64>()
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
    ) -> Result<Option<Self>> {
        if value.is_null() {
            return Ok(None);
        }
        if precision != Precision::Float64 {
            return Err(invalid("DFT+U needs float64 precision"));
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
            .map(|s| Shell::parse(s, projectors, spins, spinor))
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
    /// Sets `(U, J)` per shell (Hartree) and records a `U_source` provenance label.
    pub fn set_interactions_with_source(
        &mut self,
        updates: &[(f64, f64)],
        source: &str,
    ) -> Result<()> {
        if updates.len() != self.shells.len() {
            return Err(invalid("one (U, J) update per Hubbard shell is needed"));
        }
        for (shell, &(u, j)) in self.shells.iter_mut().zip(updates) {
            Hubbard::new(u, j)?;
            shell.u = u;
            shell.j = j;
            shell.source = Some(source.to_string());
        }
        Ok(())
    }
    /// False when the shell interactions are identically zero: the SCF then runs as plain DFT.
    pub fn is_active(&self) -> bool {
        match self.form {
            Form::Dudarev => self.shells.iter().any(|s| s.u_eff() != 0.),
            Form::Liechtenstein => self.shells.iter().any(|s| s.u != 0. || s.j != 0.),
        }
    }
    /// Bytes of the occupations and operators held during the SCF.
    pub fn bytes(&self, c: &Crystal) -> Result<usize> {
        let factor = if c.spinor { 4 } else { 1 };
        let numbers: usize = self
            .shells
            .iter()
            .map(|s| factor * s.size() * s.size())
            .sum();
        product(&[c.spins, numbers, 16, 4])?
            .checked_add(product(&[c.spins, factor, c.p, c.p, 16, 3])?)
            .ok_or(crate::Error::Overflow)
    }
    /// The starting occupations, `[spin * shells + shell]` (`p x p` collinear, `2p x 2p` spinor).
    pub fn seed(&self, spins: usize) -> Result<Vec<Vec<C64>>> {
        self.seed_for(spins, false)
    }
    /// [`Spec::seed`] for either collinear (`spinor = false`) or spin-major `2p x 2p` spinor shells.
    pub fn seed_for(&self, spins: usize, spinor: bool) -> Result<Vec<Vec<C64>>> {
        let k = self.shells.len();
        if spinor {
            let mut out = Vec::with_capacity(k);
            for shell in &self.shells {
                let p = shell.size();
                let d = 2 * p;
                match &shell.seed {
                    Seed::Moment {
                        n_electrons,
                        moment,
                    } => {
                        let up = ((n_electrons + moment) / d as f64).clamp(0., 1.);
                        let dn = ((n_electrons - moment) / d as f64).clamp(0., 1.);
                        let mut m = vec![C64::default(); d * d];
                        for a in 0..p {
                            m[a * d + a] = C64::new(up, 0.);
                            m[(p + a) * d + (p + a)] = C64::new(dn, 0.);
                        }
                        out.push(m);
                    }
                    Seed::Occupations(matrices) => {
                        if matrices.len() != 1 {
                            return Err(invalid("spinor Hubbard seed needs one 2p x 2p matrix"));
                        }
                        validate_occupation(shell.l, true, &matrices[0])?;
                        out.push(matrices[0].clone());
                    }
                }
            }
            return Ok(out);
        }
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
    /// Integer diagonal starts for a run that the moment seed leaves unconverged, in
    /// `[spin * shells + shell]` layout like [`Spec::seed`] (`2p x 2p` spin-diagonal for spinors).
    pub fn integer_starts(&self, c: &Crystal) -> Result<Vec<Vec<Vec<C64>>>> {
        let (effective_spins, k) = (if c.spinor { 2 } else { c.spins }, self.shells.len());
        let mut counts = vec![0usize; effective_spins * k];
        // (element, l, electrons, the orbital choices of that class)
        let mut classes: Vec<(&str, usize, usize, Vec<usize>)> = vec![];
        for (index, shell) in self.shells.iter().enumerate() {
            let Seed::Moment {
                n_electrons,
                moment,
            } = &shell.seed
            else {
                return Ok(vec![]);
            };
            let p = shell.size();
            for spin in 0..effective_spins {
                let electrons = if c.spinor && c.time_reversal {
                    n_electrons / 2.
                } else if effective_spins == 1 {
                    n_electrons / 2.
                } else {
                    (n_electrons + if spin == 0 { *moment } else { -moment }) / 2.
                };
                let whole = electrons.round();
                if (electrons - whole).abs() > 1e-9 || whole < 0. || whole > p as f64 {
                    return Ok(vec![]);
                }
                let whole = whole as usize;
                counts[spin * k + index] = whole;
                if 0 < whole
                    && whole < p
                    && !classes
                        .iter()
                        .any(|(s, l, e, _)| *s == shell.symbol && *l == shell.l && *e == whole)
                {
                    let choices: Vec<usize> = (0..1usize << p)
                        .filter(|mask| mask.count_ones() as usize == whole)
                        .collect();
                    classes.push((&shell.symbol, shell.l, whole, choices));
                }
            }
        }
        let mut total = 1usize;
        for (_, _, _, choices) in &classes {
            total = total.saturating_mul(choices.len());
        }
        if total > MAX_INTEGER_STARTS {
            return Ok(vec![]);
        }
        let mut starts = Vec::with_capacity(total);
        for combination in 0..total {
            let mut rest = combination;
            let mut picked = vec![0usize; classes.len()];
            for (slot, (_, _, _, choices)) in picked.iter_mut().zip(&classes).rev() {
                *slot = choices[rest % choices.len()];
                rest /= choices.len();
            }
            let start = if c.spinor {
                let mut per_shell = Vec::with_capacity(k);
                for (index, shell) in self.shells.iter().enumerate() {
                    let p = shell.size();
                    let d = 2 * p;
                    let mut m = vec![C64::default(); d * d];
                    for spin in 0..2 {
                        let whole = counts[spin * k + index];
                        let mask = match classes.iter().position(|(s, l, e, _)| {
                            *s == shell.symbol && *l == shell.l && *e == whole
                        }) {
                            Some(class) => picked[class],
                            None if whole == 0 => 0,
                            None => (1usize << p) - 1,
                        };
                        for a in 0..p {
                            if (mask >> a) & 1 == 1 {
                                let row = spin * p + a;
                                m[row * d + row] = C64::new(1., 0.);
                            }
                        }
                    }
                    per_shell.push(m);
                }
                per_shell
            } else {
                let mut per_chan = vec![vec![]; c.spins * k];
                for (index, shell) in self.shells.iter().enumerate() {
                    let p = shell.size();
                    for spin in 0..c.spins {
                        let whole = counts[spin * k + index];
                        let mask = match classes.iter().position(|(s, l, e, _)| {
                            *s == shell.symbol && *l == shell.l && *e == whole
                        }) {
                            Some(class) => picked[class],
                            None if whole == 0 => 0,
                            None => (1usize << p) - 1,
                        };
                        per_chan[spin * k + index] = (0..p * p)
                            .map(|i| {
                                let filled = i / p == i % p && mask >> (i / p) & 1 == 1;
                                C64::new(if filled { 1. } else { 0. }, 0.)
                            })
                            .collect();
                    }
                }
                per_chan
            };
            if super::hubbard_symmetry::check_seed(self, c, &start).is_ok() {
                starts.push(start);
            }
        }
        Ok(starts)
    }
    /// [`Spec::seed`] of the crystal's spin channels, refused when it breaks the symmetry the
    /// crystal's mesh was reduced with ([`super::hubbard_symmetry::check_seed`]).
    pub fn start(&self, c: &Crystal) -> Result<Vec<Vec<C64>>> {
        let seed = self.seed_for(c.spins, c.spinor)?;
        super::hubbard_symmetry::check_seed(self, c, &seed)?;
        Ok(seed)
    }
    /// `n^sigma = W^T R^sigma W` of every shell from the band density matrices (`[spin * shells + shell]`
    /// for collinear, or `[shell]` with `2p x 2p` spin-major matrices for spinor).
    /// On a mesh reduced by symmetry or time reversal the sum over the wedge is completed first
    /// ([`super::hubbard_symmetry`]).
    pub fn occupations(&self, c: &Crystal, matrices: &[StoredMatrix]) -> Result<Vec<Vec<C64>>> {
        let mut out = self.raw_occupations(c, matrices)?;
        // The wedge sum visits one point of every star: complete it over the kept operations and
        // time reversal, so `n` is symmetric exactly when the density is.
        if let Some(averager) = super::hubbard_symmetry::Averager::for_crystal(self, c)? {
            averager.apply(self, &mut out);
        }
        Ok(out)
    }
    /// Unsymmetrized shell occupations `n^sigma = W^T R^sigma W` summed directly over `c.points`
    /// with no point-group or time-reversal averaging; for full-BZ propagations where an external
    /// field breaks time reversal (such as the moved-basis `k + A` mesh).
    pub fn raw_occupations(&self, c: &Crystal, matrices: &[StoredMatrix]) -> Result<Vec<Vec<C64>>> {
        let (ctx, nk, k) = (&c.context, c.points.len(), self.shells.len());
        let widest = self
            .shells
            .iter()
            .map(|s| s.projectors.len())
            .max()
            .unwrap_or(0);
        let _work = ctx.reserve(product(&[
            c.bands(),
            widest,
            16,
            if c.spinor { 12 } else { 6 },
        ])?)?;
        let mut out = if c.spinor {
            let mut projected: Vec<[Matrix; 4]> = self
                .shells
                .iter()
                .map(|s| {
                    std::array::from_fn(|_| Matrix::zeros(s.projectors.len(), s.projectors.len()))
                })
                .collect();
            for (index, data) in c.points.iter().enumerate() {
                let d = matrices[index].matrix(ctx)?;
                for (shell, blocks) in self.shells.iter().zip(&mut projected) {
                    let columns = Matrix::from_fn(c.m, shell.projectors.len(), |a, b| {
                        data.projections[(a, shell.projectors[b])]
                    });
                    let weight = C64::new(data.weight, 0.);
                    for st in 0..4 {
                        let (s, t) = (st / 2, st % 2);
                        let sub = d.view((s * c.m, t * c.m), (c.m, c.m));
                        blocks[st] += columns.adjoint() * (&sub * &columns) * weight;
                    }
                }
            }
            let mut spinor_out = Vec::with_capacity(k);
            for (shell, blocks) in self.shells.iter().zip(&projected) {
                let p = shell.size();
                let dim = 2 * p;
                let mut flat = vec![C64::default(); dim * dim];
                for st in 0..4 {
                    let (s, t) = (st / 2, st % 2);
                    let n_st = shell::occupation(shell.l, &shell.weights, &blocks[st])?;
                    for a in 0..p {
                        for b in 0..p {
                            flat[(s * p + a) * dim + (t * p + b)] = n_st[(a, b)];
                        }
                    }
                }
                spinor_out.push(flat);
            }
            spinor_out
        } else {
            let mut col_out = Vec::with_capacity(c.spins * k);
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
                    col_out.push((0..p * p).map(|i| n[(i / p, i % p)]).collect());
                }
            }
            col_out
        };
        for matrix in &mut out {
            let p = (matrix.len() as f64).sqrt().round() as usize;
            let raw = matrix.clone();
            for (i, to) in matrix.iter_mut().enumerate() {
                *to = 0.5 * (raw[i] + raw[(i % p) * p + i / p].conj());
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
        match self.form {
            Form::Dudarev => {
                let hubbard = Hubbard {
                    form: self.form,
                    ..Hubbard::new(s.u, s.j)?
                };
                model::evaluate(&c.context, &hubbard, s.l, spin, &channels)
            }
            Form::Liechtenstein => liechtenstein_model::evaluate_with_ratios(
                &c.context,
                s.u,
                s.j,
                s.l,
                &s.ratios,
                Basis::Complex,
                spin,
                &channels,
            ),
        }
    }
    fn evaluate_spinor(
        &self,
        c: &Crystal,
        n: &[Vec<C64>],
        shell: usize,
    ) -> Result<spinor_model::Evaluation> {
        let s = &self.shells[shell];
        let layout = SpinorLayout::new(s.l, false)?;
        let occ = SpinorOcc::new(&c.context, layout, &n[shell])?;
        let hubbard = Hubbard {
            form: self.form,
            ..Hubbard::new(s.u, s.j)?
        };
        match self.form {
            Form::Dudarev => spinor_model::evaluate(&c.context, &hubbard, &occ),
            Form::Liechtenstein => spinor_model::evaluate_liechtenstein_with_ratios(
                &c.context, &hubbard, &s.ratios, &occ,
            ),
        }
    }
    /// `E_U` (Hartree) of the occupations `n`.
    pub fn energy(&self, c: &Crystal, n: &[Vec<C64>]) -> Result<f64> {
        Ok(self.shell_energies(c, n)?.iter().sum())
    }
    /// `E_U` (Hartree) of each shell, in table order; their sum is [`Spec::energy`].
    pub fn shell_energies(&self, c: &Crystal, n: &[Vec<C64>]) -> Result<Vec<f64>> {
        if c.spinor {
            return (0..self.shells.len())
                .map(|shell| Ok(self.evaluate_spinor(c, n, shell)?.energy))
                .collect();
        }
        (0..self.shells.len())
            .map(|shell| Ok(self.evaluate(c, n, shell)?.energy))
            .collect()
    }
    /// The `p x p` operator `O^sigma = sum_shells W V^sigma W^T` per spin channel, or a single
    /// spin-major `2p x 2p` operator for a spinor crystal.
    pub fn operators(&self, c: &Crystal, n: &[Vec<C64>]) -> Result<Vec<Matrix>> {
        if c.spinor {
            let mut op = Matrix::zeros(2 * c.p, 2 * c.p);
            for (index, shell) in self.shells.iter().enumerate() {
                let evaluation = self.evaluate_spinor(c, n, index)?;
                let (p, w) = (shell.size(), shell.weights.len());
                let d = 2 * p;
                let v = evaluation.potential.as_slice();
                for s in 0..2 {
                    for t in 0..2 {
                        for (a, b) in (0..p).flat_map(|a| (0..p).map(move |b| (a, b))) {
                            let val = v[(s * p + a) * d + (t * p + b)];
                            for (i, j) in (0..w).flat_map(|i| (0..w).map(move |j| (i, j))) {
                                op[(
                                    s * c.p + shell.projectors[a * w + i],
                                    t * c.p + shell.projectors[b * w + j],
                                )] += val * (shell.weights[i] * shell.weights[j]);
                            }
                        }
                    }
                }
            }
            return Ok(vec![op]);
        }
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

    /// Quadratic response kernel `dV_dc / dn_ba` for periodic optics (`dielectric` and `bse`):
    /// builds the shell projector contraction metadata and the `size x size` Hermitian response
    /// matrix in Hartree (singlet/unrestricted/spinor, plus restricted triplet).
    pub fn optical_kernel(&self, c: &Crystal) -> Result<OpticalKernel> {
        let mut shells_meta = Vec::with_capacity(self.shells.len());
        let mut offset = 0usize;
        let spin_channels = if c.spinor { 1 } else { c.spins };
        let mut diag_offsets = Vec::with_capacity(self.shells.len());
        for s in &self.shells {
            let p = s.size();
            let width = if c.spinor { 2 * p } else { p };
            let block = width * width;
            shells_meta.push(OpticalShell {
                offset,
                p,
                width,
                projectors: s.projectors.clone(),
                weights: s.weights.clone(),
            });
            diag_offsets.push((offset, block, width, p));
            offset += spin_channels * block;
        }
        let size = offset;
        let mut singlet = Matrix::zeros(size, size);
        let mut triplet = (!c.spinor && c.spins == 1).then(|| Matrix::zeros(size, size));
        let eps = 1e-3;
        let half = C64::new(0.5, 0.);
        let hermitian_pair = |dim: usize, row: usize, col: usize| -> (Vec<C64>, Vec<C64>) {
            let mut h = vec![C64::default(); dim * dim];
            let mut a = vec![C64::default(); dim * dim];
            h[row * dim + col] += C64::new(0.5, 0.);
            h[col * dim + row] += C64::new(0.5, 0.);
            a[row * dim + col] += C64::new(0., 0.5);
            a[col * dim + row] += C64::new(0., -0.5);
            (h, a)
        };
        let add_scaled = |base: &[C64], delta: &[C64], factor: f64| -> Vec<C64> {
            base.iter()
                .zip(delta)
                .map(|(&b, &d)| b + d * factor)
                .collect()
        };
        for (shell_idx, &(base_off, block, width, p)) in diag_offsets.iter().enumerate() {
            let s = &self.shells[shell_idx];
            let mut n_ref = vec![C64::default(); width * width];
            for i in 0..width {
                n_ref[i * width + i] = half;
            }
            if c.spinor {
                let eval_v = |occ_slice: &[C64]| -> Result<Vec<C64>> {
                    let layout = SpinorLayout::new(s.l, false)?;
                    let occ = SpinorOcc::new(&c.context, layout, occ_slice)?;
                    let hubbard = Hubbard {
                        form: self.form,
                        ..Hubbard::new(s.u, s.j)?
                    };
                    let eval = match self.form {
                        Form::Dudarev => spinor_model::evaluate(&c.context, &hubbard, &occ)?,
                        Form::Liechtenstein => spinor_model::evaluate_liechtenstein_with_ratios(
                            &c.context, &hubbard, &s.ratios, &occ,
                        )?,
                    };
                    Ok(eval.potential.as_slice().to_vec())
                };
                for a in 0..width {
                    for b in 0..width {
                        let col_idx = base_off + a * width + b;
                        // Perturbation delta n_{ba} = 1 corresponds to E_{ba} = H_{ba} - i A_{ba}.
                        let (h_mat, a_mat) = hermitian_pair(width, b, a);
                        let vh = {
                            let vp = eval_v(&add_scaled(&n_ref, &h_mat, eps))?;
                            let vm = eval_v(&add_scaled(&n_ref, &h_mat, -eps))?;
                            vp.into_iter()
                                .zip(vm)
                                .map(|(p, m)| (p - m) / (2. * eps))
                                .collect::<Vec<_>>()
                        };
                        let va = {
                            let vp = eval_v(&add_scaled(&n_ref, &a_mat, eps))?;
                            let vm = eval_v(&add_scaled(&n_ref, &a_mat, -eps))?;
                            vp.into_iter()
                                .zip(vm)
                                .map(|(p, m)| (p - m) / (2. * eps))
                                .collect::<Vec<_>>()
                        };
                        for c_idx in 0..width {
                            for d_idx in 0..width {
                                let row_idx = base_off + c_idx * width + d_idx;
                                let val =
                                    vh[d_idx * width + c_idx] - C64::i() * va[d_idx * width + c_idx];
                                singlet[(row_idx, col_idx)] = val;
                            }
                        }
                    }
                }
            } else {
                let eval_v_unres = |up: &[C64], dn: &[C64]| -> Result<[Vec<C64>; 2]> {
                    let chans = [up, dn];
                    let eval = match self.form {
                        Form::Dudarev => {
                            let hubbard = Hubbard {
                                form: self.form,
                                ..Hubbard::new(s.u, s.j)?
                            };
                            model::evaluate(&c.context, &hubbard, s.l, Spin::Unrestricted, &chans)?
                        }
                        Form::Liechtenstein => liechtenstein_model::evaluate_with_ratios(
                            &c.context,
                            s.u,
                            s.j,
                            s.l,
                            &s.ratios,
                            Basis::Complex,
                            Spin::Unrestricted,
                            &chans,
                        )?,
                    };
                    Ok([
                        eval.potentials[0].as_slice().to_vec(),
                        eval.potentials[1].as_slice().to_vec(),
                    ])
                };
                for s_in in 0..c.spins {
                    for a in 0..p {
                        for b in 0..p {
                            let col_idx = base_off + s_in * block + a * p + b;
                            let (h_mat, a_mat) = hermitian_pair(p, b, a);
                            let pert = |sign_dn: f64, mat: &[C64], step: f64| -> Result<[Vec<C64>; 2]> {
                                let p_in = add_scaled(&n_ref, mat, step);
                                if c.spins == 1 {
                                    let p_dn = add_scaled(&n_ref, mat, sign_dn * step);
                                    eval_v_unres(&p_in, &p_dn)
                                } else if s_in == 0 {
                                    eval_v_unres(&p_in, &n_ref)
                                } else {
                                    eval_v_unres(&n_ref, &p_in)
                                }
                            };
                            let fill_mode = |sign_dn: f64, target: &mut Matrix| -> Result<()> {
                                let vp_h = pert(sign_dn, &h_mat, eps)?;
                                let vm_h = pert(sign_dn, &h_mat, -eps)?;
                                let vp_a = pert(sign_dn, &a_mat, eps)?;
                                let vm_a = pert(sign_dn, &a_mat, -eps)?;
                                for s_out in 0..c.spins {
                                    for c_idx in 0..p {
                                        for d_idx in 0..p {
                                            let row_idx = base_off + s_out * block + c_idx * p + d_idx;
                                            let dv_h = (vp_h[s_out][d_idx * p + c_idx]
                                                - vm_h[s_out][d_idx * p + c_idx])
                                                / (2. * eps);
                                            let dv_a = (vp_a[s_out][d_idx * p + c_idx]
                                                - vm_a[s_out][d_idx * p + c_idx])
                                                / (2. * eps);
                                            target[(row_idx, col_idx)] = dv_h - C64::i() * dv_a;
                                        }
                                    }
                                }
                                Ok(())
                            };
                            fill_mode(1.0, &mut singlet)?;
                            if let Some(tri) = triplet.as_mut() {
                                fill_mode(-1.0, tri)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(OpticalKernel {
            spinor: c.spinor,
            spins: c.spins,
            size,
            singlet: (&singlet + singlet.adjoint()) * C64::new(0.5, 0.),
            triplet: triplet.map(|t| (&t + t.adjoint()) * C64::new(0.5, 0.)),
            shells: shells_meta,
        })
    }
}

/// Projector contraction metadata for one Hubbard shell in optical pair space.
#[derive(Clone, Debug)]
pub struct OpticalShell {
    offset: usize,
    p: usize,
    width: usize,
    projectors: Vec<usize>,
    weights: Vec<f64>,
}

/// Finite-rank adiabatic DFT+U response kernel `dV / dn` across all Hubbard shells.
#[derive(Clone, Debug)]
pub struct OpticalKernel {
    pub spinor: bool,
    pub spins: usize,
    pub size: usize,
    pub singlet: Matrix,
    pub triplet: Option<Matrix>,
    pub shells: Vec<OpticalShell>,
}

impl OpticalKernel {
    /// Projects one Bloch band (`band` into `frame.projections`, which has `(component * bands, p)`)
    /// onto every shell's Hubbard orbital basis (`p` or `2p` entries per shell).
    pub fn band_amplitudes(&self, projections: &Matrix, bands: usize, band: usize) -> Vec<Vec<C64>> {
        self.shells
            .iter()
            .map(|shell| {
                let w = shell.weights.len();
                let components = if self.spinor { 2 } else { 1 };
                let mut out = vec![C64::default(); components * shell.p];
                for s in 0..components {
                    let row = s * bands + band;
                    for a in 0..shell.p {
                        let mut acc = C64::default();
                        for i in 0..w {
                            acc += projections[(row, shell.projectors[a * w + i])]
                                * shell.weights[i];
                        }
                        out[s * shell.p + a] = acc;
                    }
                }
                out
            })
            .collect()
    }

    /// Fills `forward` (`<v | w_a><w_b | c>`) and `backward` (`<c | w_a><w_b | v>`) transition rows
    /// of length `self.size` for transition `(v, c)` in spin channel `channel`.
    pub fn transition_rows(
        &self,
        channel: usize,
        v_amp: &[Vec<C64>],
        c_amp: &[Vec<C64>],
        forward: &mut [C64],
        backward: &mut [C64],
    ) {
        forward.fill(C64::default());
        backward.fill(C64::default());
        let spin_slot = if self.spinor { 0 } else { channel };
        for (i, shell) in self.shells.iter().enumerate() {
            let d = shell.width;
            let start = shell.offset + spin_slot * (d * d);
            let qv = &v_amp[i];
            let qc = &c_amp[i];
            for a in 0..d {
                for b in 0..d {
                    let idx = start + a * d + b;
                    forward[idx] = qv[a] * qc[b].conj();
                    backward[idx] = qc[a] * qv[b].conj();
                }
            }
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
