//! Hubbard correction on an angular shell: the simplified rotationally invariant
//! form with the fully-localized-limit double counting.
//!
//! The model sees only the occupation matrix `n^sigma` of one shell (`p x p`,
//! `p = 2l + 1`, row-major, Hermitian, one matrix per spin channel) and the
//! effective interaction `U_eff = U - J`. It does not know how `n` is built from
//! a density (that is the subspace map of the solver), so it depends on no
//! other model.
//!
//! ```text
//! E_U = (U_eff / 2) sum_sigma Tr[ n^sigma (1 - n^sigma) ]
//! V^sigma = dE_U / dn^sigma = U_eff (1/2 - n^sigma)
//! ```
//!
//! `E_U` splits as `E_Hub - E_dc` (see [`interaction`] and [`double_counting`]).
//! `dE_U / dU = g = E_U / U_eff` is the Hellmann-Feynman building block, and
//! `V / U_eff = 1/2 - n` is the building block of the eigenvalue shifts.
//!
//! Units are the caller's: `U`, `J`, `E_U` and `V` share one energy unit.
use super::invalid;
use crate::{Buffer, Context, Result};
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64 as C64;
use std::mem::size_of;

/// Largest supported angular momentum (`f` shells, `p = 7`).
pub const MAX_L: usize = 3;
/// Anti-Hermitian part tolerated in `n`, relative to `max(1, max |n_ij|)`.
pub const HERMITIAN_TOLERANCE: f64 = 1e-10;

/// Variant of the functional. Only the simplified fully-localized-limit form
/// exists; others (around-mean-field, full Slater integrals) get a variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    DudarevFll,
}

impl Form {
    /// Every form the engine implements, in the order messages list them.
    pub const ALL: [Form; 1] = [Form::DudarevFll];
    /// The name the option and the crystal snapshot write.
    pub fn name(self) -> &'static str {
        match self {
            Self::DudarevFll => "dudarev-fll",
        }
    }
    /// The form named `name`; raises, listing the implemented ones, for any other.
    pub fn parse(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|form| form.name() == name)
            .ok_or_else(|| {
                let known: Vec<String> = Self::ALL
                    .iter()
                    .map(|f| format!("'{}'", f.name()))
                    .collect();
                invalid(format!(
                    "unknown Hubbard form '{name}'; the implemented forms are {}",
                    known.join(", ")
                ))
            })
    }
}

/// The letter of a correlated shell (`d` or `f`) and its angular momentum.
pub const SHELLS: [(char, usize); 2] = [('d', 2), ('f', 3)];

/// The letter of the shell with angular momentum `l`; `None` for an `l` that takes no `U`.
pub fn shell_letter(l: usize) -> Option<char> {
    SHELLS.iter().find(|s| s.1 == l).map(|s| s.0)
}

/// The angular momentum of the shell `letter`; `None` for a letter that takes no `U`.
pub fn shell_l(letter: &str) -> Option<usize> {
    SHELLS
        .iter()
        .find(|s| letter.chars().eq([s.0]))
        .map(|s| s.1)
}

/// Spin layout of the channels handed to [`evaluate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spin {
    /// One matrix, standing for both (identical) channels: `E_U` counts it twice
    /// and the returned potential serves both spins. The caller passes the
    /// per-spin occupation `n^sigma` (for a spin-summed projection `W^T R W` that
    /// is already multiplied by 1/2); the model never halves it.
    Restricted,
    /// Two matrices, spin up then spin down.
    Unrestricted,
}

impl Spin {
    pub fn channels(self) -> usize {
        match self {
            Self::Restricted => 1,
            Self::Unrestricted => 2,
        }
    }
    fn multiplicity(self) -> f64 {
        match self {
            Self::Restricted => 2.,
            Self::Unrestricted => 1.,
        }
    }
}

/// Validated interaction parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hubbard {
    pub u: f64,
    pub j: f64,
    pub form: Form,
}

impl Hubbard {
    /// `U >= 0`, `J >= 0`, `J <= U` (so `U_eff >= 0`), all finite.
    pub fn new(u: f64, j: f64) -> Result<Self> {
        if !u.is_finite() || !j.is_finite() {
            return Err(invalid("Hubbard U and J must be finite"));
        }
        if u < 0. || j < 0. {
            return Err(invalid(format!(
                "Hubbard U and J must not be negative (U = {u}, J = {j})"
            )));
        }
        if j > u {
            return Err(invalid(format!(
                "Hubbard J = {j} exceeds U = {u}: U_eff = U - J would be negative"
            )));
        }
        Ok(Self {
            u,
            j,
            form: Form::DudarevFll,
        })
    }
    /// Dudarev's single parameter.
    pub fn u_eff(&self) -> f64 {
        self.u - self.j
    }
    /// True when the correction is exactly the identity (no energy, no operator).
    pub fn is_inert(&self) -> bool {
        self.u_eff() == 0.
    }
}

/// `E_U`, its split and the operators for one shell.
pub struct Evaluation {
    /// `E_U` over all spins (restricted counts both).
    pub energy: f64,
    /// `g = E_U / U_eff = (1/2) sum_sigma Tr[n(1-n)]`, i.e. `dE/dU` at fixed `n`
    /// and `J` (and `-dE/dJ`). Defined for `U_eff = 0` too.
    pub per_u_eff: f64,
    /// `E_Hub` of the simplified form (see [`interaction`]).
    pub interaction: f64,
    /// `E_dc` of the fully localized limit (see [`double_counting`]).
    pub double_counting: f64,
    /// `V^sigma` row-major `p x p`, one buffer per channel handed in.
    pub potentials: Vec<Buffer<C64>>,
}

/// Width `p = 2l + 1` of a shell, raising for `l > MAX_L`.
pub fn shell_size(l: usize) -> Result<usize> {
    if l > MAX_L {
        return Err(invalid(format!(
            "Hubbard shell l = {l} is not supported (l <= {MAX_L})"
        )));
    }
    Ok(2 * l + 1)
}

/// Checks size, finiteness and Hermiticity of `n`; returns `p`.
pub fn validate(l: usize, n: &[C64]) -> Result<usize> {
    let p = shell_size(l)?;
    if n.len() != p * p {
        return Err(invalid(format!(
            "occupation matrix has {} entries, a shell with l = {l} needs {}",
            n.len(),
            p * p
        )));
    }
    if n.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
        return Err(invalid("occupation matrix has a non-finite entry"));
    }
    let scale = n.iter().fold(1f64, |m, x| m.max(x.norm()));
    for a in 0..p {
        for b in a..p {
            if (n[a * p + b] - n[b * p + a].conj()).norm() > HERMITIAN_TOLERANCE * scale {
                return Err(invalid(format!(
                    "occupation matrix is not Hermitian at ({a}, {b})"
                )));
            }
        }
    }
    Ok(p)
}

/// `(Re Tr n, Re Tr n^2)` of the Hermitian part of `n`.
fn traces(p: usize, n: &[C64]) -> (f64, f64) {
    let mut trace = 0.;
    let mut square = 0.;
    for a in 0..p {
        trace += n[a * p + a].re;
        for b in 0..p {
            // Tr n^2 = sum_ab n_ab n_ba, taken with the Hermitian part so that
            // E and V are consistent for the n that was handed in.
            let h = 0.5 * (n[a * p + b] + n[b * p + a].conj());
            square += h.norm_sqr();
        }
    }
    (trace, square)
}

/// Total occupation `Tr n` of a channel.
pub fn trace(l: usize, n: &[C64]) -> Result<f64> {
    let p = validate(l, n)?;
    Ok(traces(p, n).0)
}

/// `E_dc` of the fully localized limit for the total occupations of the two spin
/// channels: `U/2 N(N-1) - J/2 sum_sigma N_sigma (N_sigma - 1)`, `N = N_up + N_dn`.
pub fn double_counting(h: &Hubbard, n_up: f64, n_down: f64) -> f64 {
    let n = n_up + n_down;
    0.5 * h.u * n * (n - 1.) - 0.5 * h.j * (n_up * (n_up - 1.) + n_down * (n_down - 1.))
}

/// `E_Hub` of the simplified form, `U/2 N^2 - J/2 sum_sigma N_sigma^2
/// - U_eff/2 sum_sigma Tr n_sigma^2`, so that `E_Hub - E_dc = E_U` exactly.
pub fn interaction(h: &Hubbard, n_up: f64, n_down: f64, square_up: f64, square_down: f64) -> f64 {
    let n = n_up + n_down;
    0.5 * h.u * n * n
        - 0.5 * h.j * (n_up * n_up + n_down * n_down)
        - 0.5 * h.u_eff() * (square_up + square_down)
}

/// Evaluates the correction for one shell. `channels` holds one Hermitian `p x p`
/// matrix per channel of `spin`. The output buffers are reserved from `ctx` before
/// anything is written; on error nothing is kept.
pub fn evaluate(
    ctx: &Context,
    h: &Hubbard,
    l: usize,
    spin: Spin,
    channels: &[&[C64]],
) -> Result<Evaluation> {
    if channels.len() != spin.channels() {
        return Err(invalid(format!(
            "{spin:?} Hubbard evaluation needs {} occupation matrices, got {}",
            spin.channels(),
            channels.len()
        )));
    }
    let mut p = 0;
    for n in channels {
        p = validate(l, n)?;
    }
    // Reserve every output before computing any of it.
    let mut potentials = Vec::with_capacity(channels.len());
    for _ in channels {
        potentials.push(ctx.allocate::<C64>(p * p)?);
    }
    let mut g = 0.;
    let mut totals = [0.; 2];
    let mut squares = [0.; 2];
    let half = C64::new(0.5, 0.);
    for (c, n) in channels.iter().enumerate() {
        let (trace, square) = traces(p, n);
        g += 0.5 * (trace - square);
        totals[c] = trace;
        squares[c] = square;
        let u_eff = h.u_eff();
        let out = potentials[c].as_mut_slice();
        if !h.is_inert() {
            for a in 0..p {
                for b in 0..p {
                    let hermitian = 0.5 * (n[a * p + b] + n[b * p + a].conj());
                    let identity = if a == b { half } else { C64::default() };
                    out[a * p + b] = (identity - hermitian) * u_eff;
                }
            }
        }
    }
    let m = spin.multiplicity();
    g *= m;
    if spin == Spin::Restricted {
        totals[1] = totals[0];
        squares[1] = squares[0];
    }
    Ok(Evaluation {
        energy: h.u_eff() * g,
        per_u_eff: g,
        interaction: interaction(h, totals[0], totals[1], squares[0], squares[1]),
        double_counting: double_counting(h, totals[0], totals[1]),
        potentials,
    })
}

/// Smallest and largest eigenvalue of the Hermitian `n`. Eigenvalues of an
/// occupation matrix lie in `[0, 1]` for orthonormal projectors; the PAW partial-wave
/// projection only approximates that, so callers gate on a measured range.
pub fn eigenvalue_range(ctx: &Context, l: usize, n: &[C64]) -> Result<(f64, f64)> {
    let p = validate(l, n)?;
    let bytes = 2 * p * 2 * p * size_of::<f64>() * 4;
    let _work = ctx.reserve(bytes)?;
    // [[Re, -Im], [Im, Re]] is real symmetric with the spectrum of n, doubled.
    let embedded = DMatrix::from_fn(2 * p, 2 * p, |i, j| {
        let (a, b) = (i % p, j % p);
        let h = 0.5 * (n[a * p + b] + n[b * p + a].conj());
        match (i / p, j / p) {
            (0, 0) | (1, 1) => h.re,
            (0, 1) => -h.im,
            _ => h.im,
        }
    });
    let eig = SymmetricEigen::try_new(embedded, 1e-15, 10_000)
        .ok_or_else(|| invalid("occupation matrix eigenproblem did not converge"))?;
    let lo = eig
        .eigenvalues
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let hi = eig
        .eigenvalues
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok((lo, hi))
}
