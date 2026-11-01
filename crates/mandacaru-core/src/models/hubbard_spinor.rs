//! Noncollinear Hubbard correction: the rotationally invariant Dudarev functional on
//! the full `2p x 2p` Hermitian occupation matrix of one shell (`p = 2l + 1`), the
//! form a spin-orbit (spinor) calculation needs. The collinear model
//! [`super::hubbard`] is untouched; a block-diagonal `n` reproduces it.
//!
//! ```text
//! E_U = (U_eff / 2) Tr[ n - n n ]            (trace over all 2p spin-orbital indices)
//! V   = dE_U / dn = (U_eff / 2) (1 - 2 n)    (an operator in the basis of n)
//! ```
//!
//! # Orientation (the contract; energy tests cannot check it)
//!
//! `E_U` of a real Slater tensor does not distinguish `n` from `n^T`, and for this
//! functional `V(n^T) = V(n)^T` is the only difference between a right and a
//! transposed hand-off, so a transposition is invisible to every energy test. The
//! orientation is therefore part of the types: [`Occupation`] and [`Potential`] are
//! distinct, both carry their [`Layout`], and only [`Layout::index`] maps
//! `(spin, m)` to a row.
//!
//! * Index: spin-major, alpha (up) block first, `index = s * p + a`, `a = m + l`
//!   (`m = -l..l`, the order of [`super::spin_orbit::angular`]). The shell is in
//!   complex harmonics (a crystal) or real ones (a molecule), as in `angular`.
//! * Density matrix: with `rho` the one-body density operator (`sum_k f_k |k><k|`)
//!   `n_{(a s),(b s')} = <a s| rho |b s'> = sum_k f_k c_{a s}^k conj(c_{b s'}^k)`. The ROW is the
//!   bra, the COLUMN the ket. [`Occupation::entry`]`(i, j)` is `n_ij`; block `(s, s')` of a
//!   `from_blocks` call is `n^{ss'}_{ab}`.
//! * Shell weights: `n^{ss'}_{ab} = sum_ij w_i R^{ss'}_{(a i),(b j)} w_j`, the existing
//!   scalar-channel contraction `paw::shell::occupation` applied to each of the four
//!   spin blocks of the projected matrix `R`; the weights `w` are spin independent
//!   (the dataset stores no j-resolved partial waves).
//! * Derivative: `dE = Re Tr[V dn] = Re sum_ab V_ab dn_ba`, i.e. `V_ab = dE/dn_ba`.
//!   `Tr[rho h] = sum_ab h_ab n_ba`, so the one-electron operator whose matrix is
//!   `h_ab = <a|h|b>` and that lowers `E_U` consistently is `h = V` itself, never
//!   `V^T`. [`Potential::first_order`] is that contraction.
//!
//! # Double counting
//!
//! The fully localized limit for a noncollinear shell is the collinear one in the
//! frame where the shell moment `m = Tr[sigma n]` points along z. There the two
//! channel totals are `N_+- = (N +- |m|) / 2`, so
//! `sum_s N_s (N_s - 1) = (N^2 + |m|^2)/2 - N` and
//! `E_dc = U/2 N (N - 1) - J/2 [(N^2 + |m|^2)/2 - N]`. The simplified interaction is
//! `E_Hub = U/2 N^2 - J/4 (N^2 + |m|^2) - U_eff/2 Tr[n n]`, and
//! `E_Hub - E_dc = U_eff/2 Tr[n - n n] = E_U` exactly: the noncollinear form has no
//! ambiguity, depends on `U - J` only, and is invariant under any `U(2p)`.
//! `|m|^2` is invariant under an SU(2) spin rotation; `m_z^2` is not.
//!
//! # Time reversal
//!
//! `Theta = U K` is antiunitary, `Theta^2 = -1`, `Theta (L.S) Theta^-1 = L.S`.
//! Complex harmonics: `Theta|m up> = (-1)^m |-m dn>` and
//! `Theta|m dn> = -(-1)^m |-m up>`. Real harmonics are real functions, so the orbital
//! part is trivial: `Theta|a up> = |a dn>`, `Theta|a dn> = -|a up>`. A density matrix
//! transforms as `Theta n Theta^-1 = U conj(n) U^dagger`. [`complete`] returns
//! `(n + Theta n Theta^-1)/2`: idempotent, trace preserving, with a zero spin vector;
//! the potential of a completed `n` is itself time-reversal invariant. Nothing
//! symmetrizes silently: a caller that wants it calls [`complete`], one that holds an
//! explicit occupation checks it with [`require_time_reversal_invariant`].
//!
//! Units are the caller's: `U`, `J`, `E_U` and `V` share one energy unit.
use super::hubbard::{self, Form, Hubbard, HERMITIAN_TOLERANCE};
use super::invalid;
use crate::{product, Buffer, Context, Error, Result};
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64 as C64;
use std::mem::size_of;

/// The index map of a shell: `2p` spin-orbitals, spin-major, alpha first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    l: usize,
    real: bool,
}

impl Layout {
    /// A shell of angular momentum `l <= MAX_L` in real (`real`) or complex harmonics.
    pub fn new(l: usize, real: bool) -> Result<Self> {
        hubbard::shell_size(l)?;
        Ok(Self { l, real })
    }
    pub fn l(&self) -> usize {
        self.l
    }
    pub fn real(&self) -> bool {
        self.real
    }
    /// `p = 2l + 1`.
    pub fn width(&self) -> usize {
        2 * self.l + 1
    }
    /// `2p`, the side of `n` and `V`.
    pub fn size(&self) -> usize {
        2 * self.width()
    }
    /// Row of spin `spin` (0 = up, 1 = down) and orbital `a = m + l`.
    pub fn index(&self, spin: usize, a: usize) -> Result<usize> {
        if spin >= 2 || a >= self.width() {
            return Err(invalid(format!(
                "spin-orbital (spin {spin}, orbital {a}) is outside a shell with l = {}",
                self.l
            )));
        }
        Ok(spin * self.width() + a)
    }
    /// Where `Theta` sends basis state `j`: `Theta|j> = sign |image>`.
    fn image(&self, j: usize) -> (usize, f64) {
        let p = self.width();
        let (s, a) = (j / p, j % p);
        let spin = if s == 0 { 1. } else { -1. };
        if self.real {
            ((1 - s) * p + a, spin)
        } else {
            let odd = (a as isize - self.l as isize).rem_euclid(2) == 1;
            ((1 - s) * p + (p - 1 - a), if odd { -spin } else { spin })
        }
    }
}

/// Checks size, finiteness and Hermiticity of a `2p x 2p` row-major matrix.
fn check(layout: Layout, n: &[C64]) -> Result<()> {
    let d = layout.size();
    if n.len() != d * d {
        return Err(invalid(format!(
            "spinor occupation matrix has {} entries, a shell with l = {} needs {}",
            n.len(),
            layout.l,
            d * d
        )));
    }
    if n.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
        return Err(invalid("spinor occupation matrix has a non-finite entry"));
    }
    let scale = n.iter().fold(1f64, |m, x| m.max(x.norm()));
    for a in 0..d {
        for b in a..d {
            if (n[a * d + b] - n[b * d + a].conj()).norm() > HERMITIAN_TOLERANCE * scale {
                return Err(invalid(format!(
                    "spinor occupation matrix is not Hermitian at ({a}, {b})"
                )));
            }
        }
    }
    Ok(())
}

/// The Hermitian part of entry `(a, b)`.
fn hermitian(d: usize, n: &[C64], a: usize, b: usize) -> C64 {
    0.5 * (n[a * d + b] + n[b * d + a].conj())
}

/// `Tr[sigma n]` of the Hermitian part: the shell's spin vector.
fn pauli(layout: Layout, n: &[C64]) -> [f64; 3] {
    let (p, d) = (layout.width(), layout.size());
    let mut m = [0.; 3];
    for a in 0..p {
        let (ud, du) = (hermitian(d, n, a, p + a), hermitian(d, n, p + a, a));
        m[0] += (ud + du).re;
        m[1] += (C64::new(0., 1.) * (ud - du)).re;
        m[2] += n[a * d + a].re - n[(p + a) * d + p + a].re;
    }
    m
}

/// A validated Hermitian `2p x 2p` occupation matrix in the orientation of the header.
#[derive(Debug)]
pub struct Occupation {
    layout: Layout,
    data: Buffer<C64>,
}

impl Occupation {
    /// Copies the row-major `n` (`n[i * 2p + j] = <i|rho|j>`) after validating it.
    pub fn new(ctx: &Context, layout: Layout, n: &[C64]) -> Result<Self> {
        check(layout, n)?;
        Ok(Self {
            layout,
            data: ctx.copy_slice(n)?,
        })
    }
    /// Assembles `n` from the four `p x p` blocks `n^{ss'}` in the order
    /// `[up-up, up-down, down-up, down-down]` (block `(s, s')` is `<a s|rho|b s'>`).
    pub fn from_blocks(ctx: &Context, layout: Layout, blocks: [&[C64]; 4]) -> Result<Self> {
        let (p, d) = (layout.width(), layout.size());
        if blocks.iter().any(|b| b.len() != p * p) {
            return Err(invalid(format!(
                "each spin block needs {} entries for l = {}",
                p * p,
                layout.l
            )));
        }
        let mut data = ctx.allocate::<C64>(d * d)?;
        for (k, block) in blocks.iter().enumerate() {
            let (s, t) = (k / 2, k % 2);
            for a in 0..p {
                for b in 0..p {
                    data.as_mut_slice()[(s * p + a) * d + t * p + b] = block[a * p + b];
                }
            }
        }
        check(layout, data.as_slice())?;
        Ok(Self { layout, data })
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// Row-major `n`.
    pub fn as_slice(&self) -> &[C64] {
        self.data.as_slice()
    }
    /// `n_ij = <i|rho|j>`.
    pub fn entry(&self, i: usize, j: usize) -> C64 {
        self.data.as_slice()[i * self.layout.size() + j]
    }
    /// `Tr n`, the shell's electrons.
    pub fn trace(&self) -> f64 {
        let d = self.layout.size();
        (0..d).map(|a| self.data.as_slice()[a * d + a].re).sum()
    }
    /// The spin vector `Tr[sigma n]` (Pauli matrices, so `|m| = N_up - N_down` in its frame).
    pub fn spin_vector(&self) -> [f64; 3] {
        pauli(self.layout, self.data.as_slice())
    }
}

/// `V = dE_U/dn` as an operator in the basis and orientation of the matching [`Occupation`].
#[derive(Debug)]
pub struct Potential {
    layout: Layout,
    data: Buffer<C64>,
}

impl Potential {
    pub fn layout(&self) -> Layout {
        self.layout
    }
    /// Row-major `V`; the one-electron operator is `h_ab = V_ab`, never `V^T`.
    pub fn as_slice(&self) -> &[C64] {
        self.data.as_slice()
    }
    /// `Re Tr[V dn] = Re sum_ab V_ab dn_ba`, the first-order change of `E_U` along `dn`.
    pub fn first_order(&self, dn: &[C64]) -> Result<f64> {
        let d = self.layout.size();
        if dn.len() != d * d || dn.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
            return Err(invalid("the direction needs 2p x 2p finite entries"));
        }
        let v = self.data.as_slice();
        Ok((0..d)
            .flat_map(|a| (0..d).map(move |b| (a, b)))
            .map(|(a, b)| (v[a * d + b] * dn[b * d + a]).re)
            .sum())
    }
}

/// `E_U`, its split and the operator of one shell.
#[derive(Debug)]
pub struct Evaluation {
    pub energy: f64,
    /// `E_U / U_eff = (1/2) Tr[n - n n]`, the `dE/dU` Hellmann-Feynman factor (defined at `U_eff = 0`).
    pub per_u_eff: f64,
    /// `E_Hub` of the simplified form (see the header).
    pub interaction: f64,
    /// `E_dc` with the vector moment (see the header).
    pub double_counting: f64,
    /// `Tr[sigma n]`.
    pub spin_vector: [f64; 3],
    pub potential: Potential,
}

/// Evaluates the correction for one shell. The potential is reserved from `ctx`
/// before anything is computed; on error nothing is kept. Only the simplified
/// Dudarev form exists for a spinor shell; any other form raises.
pub fn evaluate(ctx: &Context, h: &Hubbard, n: &Occupation) -> Result<Evaluation> {
    if h.form != Form::DudarevFll {
        return Err(Error::Unsupported(format!(
            "the noncollinear Hubbard model implements only the '{}' form, not '{}'",
            Form::DudarevFll.name(),
            h.form.name()
        )));
    }
    let (layout, d, p) = (n.layout, n.layout.size(), n.layout.width());
    let mut potential = ctx.allocate::<C64>(d * d)?;
    let data = n.data.as_slice();
    let (mut trace, mut square) = ([0.; 2], [[0.; 2]; 2]);
    for a in 0..d {
        for b in 0..d {
            square[a / p][b / p] += hermitian(d, data, a, b).norm_sqr();
        }
        trace[a / p] += data[a * d + a].re;
    }
    // Block by block in the collinear order, then the two spin-flip blocks: for a
    // block-diagonal n the cross terms are exact zeros and E equals the collinear value.
    let mut g = 0.;
    g += 0.5 * (trace[0] - square[0][0]);
    g += 0.5 * (trace[1] - square[1][1]);
    g -= 0.5 * (square[0][1] + square[1][0]);
    let u_eff = h.u_eff();
    if !h.is_inert() {
        let half = C64::new(0.5, 0.);
        let out = potential.as_mut_slice();
        for a in 0..d {
            for b in 0..d {
                let identity = if a == b { half } else { C64::default() };
                out[a * d + b] = (identity - hermitian(d, data, a, b)) * u_eff;
            }
        }
    }
    let spin_vector = pauli(layout, data);
    let (total, moment) = (
        trace[0] + trace[1],
        spin_vector.iter().map(|x| x * x).sum::<f64>().sqrt(),
    );
    let (plus, minus) = (0.5 * (total + moment), 0.5 * (total - moment));
    let squares = square[0][0] + square[1][1] + square[0][1] + square[1][0];
    Ok(Evaluation {
        energy: u_eff * g,
        per_u_eff: g,
        interaction: hubbard::interaction(h, plus, minus, squares, 0.),
        double_counting: hubbard::double_counting(h, plus, minus),
        spin_vector,
        potential: Potential {
            layout,
            data: potential,
        },
    })
}

/// Smallest and largest eigenvalue of the Hermitian `n` (the report SO-4 gates on).
pub fn eigenvalue_range(ctx: &Context, n: &Occupation) -> Result<(f64, f64)> {
    let d = n.layout.size();
    let _work = ctx.reserve(product(&[2 * d, 2 * d, size_of::<f64>(), 4])?)?;
    let data = n.data.as_slice();
    // [[Re, -Im], [Im, Re]] is real symmetric with the spectrum of n, doubled.
    let embedded = DMatrix::from_fn(2 * d, 2 * d, |i, j| {
        let h = hermitian(d, data, i % d, j % d);
        match (i / d, j / d) {
            (0, 0) | (1, 1) => h.re,
            (0, 1) => -h.im,
            _ => h.im,
        }
    });
    let eig = SymmetricEigen::try_new(embedded, 1e-15, 10_000)
        .ok_or_else(|| invalid("spinor occupation eigenproblem did not converge"))?;
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

/// `U` of `Theta = U K` as a dense row-major `2p x 2p` matrix (a signed permutation).
pub fn time_reversal(ctx: &Context, layout: Layout) -> Result<Buffer<C64>> {
    let d = layout.size();
    let mut u = ctx.allocate::<C64>(d * d)?;
    for j in 0..d {
        let (i, sign) = layout.image(j);
        u.as_mut_slice()[i * d + j] = C64::new(sign, 0.);
    }
    Ok(u)
}

/// `Theta n Theta^-1 = U conj(n) U^dagger`.
pub fn time_reversed(ctx: &Context, n: &Occupation) -> Result<Occupation> {
    let (layout, d) = (n.layout, n.layout.size());
    let mut data = ctx.allocate::<C64>(d * d)?;
    for j in 0..d {
        let (ij, sj) = layout.image(j);
        for k in 0..d {
            let (ik, sk) = layout.image(k);
            data.as_mut_slice()[ij * d + ik] = n.data.as_slice()[j * d + k].conj() * (sj * sk);
        }
    }
    Ok(Occupation { layout, data })
}

/// The completion `(n + Theta n Theta^-1) / 2`: the input on a half mesh, where time
/// reversal supplies the other half. Never applied implicitly.
pub fn complete(ctx: &Context, n: &Occupation) -> Result<Occupation> {
    let (layout, d) = (n.layout, n.layout.size());
    let mut data = ctx.allocate::<C64>(d * d)?;
    for (out, x) in data.as_mut_slice().iter_mut().zip(n.data.as_slice()) {
        *out = *x * 0.5;
    }
    for j in 0..d {
        let (ij, sj) = layout.image(j);
        for k in 0..d {
            let (ik, sk) = layout.image(k);
            data.as_mut_slice()[ij * d + ik] +=
                n.data.as_slice()[j * d + k].conj() * (0.5 * sj * sk);
        }
    }
    Ok(Occupation { layout, data })
}

/// `max |n - Theta n Theta^-1|`: zero for a time-reversal invariant occupation.
pub fn time_reversal_defect(n: &Occupation) -> f64 {
    let (layout, d) = (n.layout, n.layout.size());
    let data = n.data.as_slice();
    let mut worst = 0f64;
    for j in 0..d {
        let (ij, sj) = layout.image(j);
        for k in 0..d {
            let (ik, sk) = layout.image(k);
            let image = data[j * d + k].conj() * (sj * sk);
            worst = worst.max((data[ij * d + ik] - image).norm());
        }
    }
    worst
}

/// Raises unless `n` is time-reversal invariant to `tolerance` (an explicit occupation is
/// validated, never symmetrized).
pub fn require_time_reversal_invariant(n: &Occupation, tolerance: f64) -> Result<()> {
    let defect = time_reversal_defect(n);
    if defect > tolerance {
        return Err(invalid(format!(
            "the occupation matrix is not time-reversal invariant: max |n - Theta n Theta^-1| = {defect:e} exceeds {tolerance:e}"
        )));
    }
    Ok(())
}
