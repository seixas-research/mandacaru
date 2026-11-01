//! Slater coupling tensor of a correlated `d` or `f` shell: the angular part of the on-site
//! Coulomb interaction that the rotationally invariant (Liechtenstein) form of DFT+U needs.
//!
//! With `p = 2l + 1` orbitals `phi_a` of one shell, the interaction matrix elements are
//!
//! ```text
//! <ab|v|cd> = sum_k F^k a_k(a,b,c,d),    k = 0, 2, .., 2l
//! a_k(a,b,c,d) = c^k(a,c) c^k(d,b)
//! c^k(m,m')    = sqrt(4 pi / (2k+1)) int Y_lm* Y_{k,m-m'} Y_lm' dOmega
//! ```
//!
//! for `m_a + m_b = m_c + m_d` and zero otherwise (electron 1 carries `a, c`; electron 2
//! carries `b, d`). Each `c^k` is one Gaunt coefficient of
//! [`crate::math::angular::gaunt`], so the tensor holds no radial information: the Slater
//! integrals `F^k` are inputs (see [`Coupling::integrals`]). Both bases are real-valued tensors:
//! the complex harmonics with the Condon-Shortley phase (`m = -l..l`) and the real harmonics
//! of [`crate::math::angular::real`] (`m < 0` sine-like, `m > 0` cosine-like, no Condon-Shortley
//! phase). The two real conventions differ by `(-1)^|m|` per orbital; every nonzero tensor entry
//! contains an even number of orbitals of odd `|m|` (the interaction is invariant under
//! `z -> -z`), so the real-basis tensor is the same in either.
//!
//! Flat layout: entry `(a, b, c, d)` of `a_k` is at `((a p + b) p + c) p + d`, with orbital index
//! `m + l` (or the real order `m_r + l`).
//!
//! **The exchange parameter.** `J` is the fully-localized-limit consistent exchange of the
//! tensor, fixed by `E_Hub(n = 1) = E_dc` for the filled shell with `U = F^0`:
//! `J = sum_{k>=2} F^k S_k / (p (p-1))`, `S_k = sum_ab a_k(a,b,b,a)`. It is NOT the average of
//! `v_abab - v_abba` over `a != b` (other coefficients, see P2-7 of the plan).
//!
//! The ratios `F^4/F^2` (and `F^6/F^2` for `f`) are fractions of the radial integrals. The
//! defaults are plan decision HL-D2: `d` 5/8, the usual empirical value (to verify), `f` 451/675
//! and 1001/2025 (the hydrogenic 4f ratios). Any ratio is overridable; the range
//! `0 < ratio <= 1` is an engine bound, not a literature value.
use super::hubbard::shell_letter;
use super::invalid;
use crate::math::angular::gaunt;
use crate::{Buffer, Context, Result};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Default `F^4/F^2` of a `d` shell (HL-D2: the usual empirical value, to verify; the
/// hydrogenic 3d value is 15/23).
pub const D_F4_F2: f64 = 5. / 8.;
/// Default `F^4/F^2` of an `f` shell (hydrogenic 4f).
pub const F_F4_F2: f64 = 451. / 675.;
/// Default `F^6/F^2` of an `f` shell (hydrogenic 4f).
pub const F_F6_F2: f64 = 1001. / 2025.;
/// Largest accepted `F^4/F^2` or `F^6/F^2` (an engine bound).
pub const RATIO_MAX: f64 = 1.;

/// The orbital basis of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// `Y_lm` with the Condon-Shortley phase, `m = -l..l`.
    Complex,
    /// The real harmonics of `math::angular::real`, `m_r = -l..l`.
    Real,
}

/// Ratios of the Slater integrals, resolved: `f6_f2` is `Some` exactly for `f`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratios {
    pub f4_f2: f64,
    pub f6_f2: Option<f64>,
}

impl Ratios {
    /// The defaults of the shell with angular momentum `l` (HL-D2).
    pub fn default_for(l: usize) -> Result<Self> {
        match check_shell(l)? {
            2 => Ok(Self {
                f4_f2: D_F4_F2,
                f6_f2: None,
            }),
            _ => Ok(Self {
                f4_f2: F_F4_F2,
                f6_f2: Some(F_F6_F2),
            }),
        }
    }
    /// The ratios of shell `l`, an override replacing its default; raises for a ratio out of
    /// range or an `F^6/F^2` given to a `d` shell (it has no `F^6`).
    pub fn new(l: usize, f4_f2: Option<f64>, f6_f2: Option<f64>) -> Result<Self> {
        let default = Self::default_for(l)?;
        let out = Self {
            f4_f2: f4_f2.unwrap_or(default.f4_f2),
            f6_f2: if l == 2 {
                f6_f2
            } else {
                Some(f6_f2.unwrap_or(F_F6_F2))
            },
        };
        out.validate(l)?;
        Ok(out)
    }
    /// Raises unless the ratios fit shell `l` and lie in `(0, RATIO_MAX]`.
    pub fn validate(&self, l: usize) -> Result<()> {
        check_shell(l)?;
        match (l, self.f6_f2) {
            (2, Some(_)) => {
                return Err(invalid(
                    "f6_f2 is not defined for a d shell: it has Slater integrals F0, F2 and F4",
                ))
            }
            (3, None) => return Err(invalid("an f shell needs f6_f2 (F6/F2)")),
            _ => {}
        }
        for (name, value) in [("f4_f2", Some(self.f4_f2)), ("f6_f2", self.f6_f2)] {
            if let Some(v) = value {
                if !v.is_finite() || v <= 0. || v > RATIO_MAX {
                    return Err(invalid(format!(
                        "{name} must be finite and in (0, {RATIO_MAX}], got {v}"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn check_shell(l: usize) -> Result<usize> {
    shell_letter(l).map(|_| l).ok_or_else(|| {
        invalid(format!(
            "Slater coupling needs a d or f shell (l = 2 or 3), got l = {l}"
        ))
    })
}

/// The reduced coefficient `c^k(m, m')` of shell `l` (the `d`/`f` convention above).
pub fn reduced(l: usize, k: usize, m: i32, mp: i32) -> Result<f64> {
    let q = m - mp;
    if m.unsigned_abs() as usize > l || mp.unsigned_abs() as usize > l {
        return Err(invalid(format!(
            "|m| exceeds l = {l} in a Slater coefficient"
        )));
    }
    if q.unsigned_abs() as usize > k {
        return Ok(0.);
    }
    let sign = if q % 2 == 0 { 1. } else { -1. };
    Ok(sign * (4. * PI / (2 * k + 1) as f64).sqrt() * gaunt([l, k, l], [m, -q, mp])?)
}

/// `T[r][m]`, row-major `p x p`: the real harmonic `r` of `math::angular::real` (orders
/// `-l..l`, no Condon-Shortley phase) as `sum_m T[r][m] Y_m`, `Y_m` the complex harmonics.
/// `R_{+mu} = ((-1)^mu Y_mu + Y_-mu)/sqrt2`, `R_{-mu} = (-i (-1)^mu Y_mu + i Y_-mu)/sqrt2`.
pub fn real_coefficients(l: usize) -> Result<Vec<C64>> {
    let l = check_shell(l)?;
    let p = 2 * l + 1;
    let mut t = vec![C64::default(); p * p];
    let s = std::f64::consts::FRAC_1_SQRT_2;
    t[l * p + l] = C64::new(1., 0.);
    for mu in 1..=l {
        let sign = if mu % 2 == 0 { 1. } else { -1. };
        // R_{+mu} = ((-1)^mu Y_mu + Y_-mu)/sqrt2, R_{-mu} = (-i (-1)^mu Y_mu + i Y_-mu)/sqrt2
        t[(l + mu) * p + l + mu] = C64::new(sign * s, 0.);
        t[(l + mu) * p + l - mu] = C64::new(s, 0.);
        t[(l - mu) * p + l + mu] = C64::new(0., -sign * s);
        t[(l - mu) * p + l - mu] = C64::new(0., s);
    }
    Ok(t)
}

/// Rotate one index of the flat `p^4` tensor `w` (`axis` 0 is the slowest) by `coef`.
fn rotate(w: &mut [C64], p: usize, axis: usize, coef: &[C64]) {
    let stride = p.pow(3 - axis as u32);
    let mut line = vec![C64::default(); p];
    for base in 0..w.len() {
        if (base / stride) % p != 0 {
            continue;
        }
        for (r, out) in line.iter_mut().enumerate() {
            *out = (0..p).map(|m| coef[r * p + m] * w[base + m * stride]).sum();
        }
        for (r, value) in line.iter().enumerate() {
            w[base + r * stride] = *value;
        }
    }
}

/// The tensors `a_k` of one shell and basis, reserved from a [`Context`].
#[derive(Debug)]
pub struct Coupling {
    l: usize,
    basis: Basis,
    orders: Vec<usize>,
    data: Buffer<f64>,
}

impl Coupling {
    /// Builds `a_k` for `k = 0, 2, .., 2l`. The stored tensors (`f64`, 15 KB for `d`, 77 KB for
    /// `f`) and, for the real basis, one complex workspace of `p^4` entries (38 KB for `f`) are
    /// reserved before allocation; a refusal keeps nothing.
    pub fn new(ctx: &Context, l: usize, basis: Basis) -> Result<Self> {
        let l = check_shell(l)?;
        let p = 2 * l + 1;
        let orders: Vec<usize> = (0..=2 * l).step_by(2).collect();
        let size = p.pow(4);
        let mut data = ctx.allocate::<f64>(orders.len() * size)?;
        let mut work = match basis {
            Basis::Real => Some(ctx.allocate::<C64>(size)?),
            Basis::Complex => None,
        };
        let t = real_coefficients(l)?;
        let conj_t: Vec<C64> = t.iter().map(|v| v.conj()).collect();
        for (index, &k) in orders.iter().enumerate() {
            let mut c = vec![0.; p * p];
            for a in 0..p {
                for b in 0..p {
                    c[a * p + b] = reduced(l, k, a as i32 - l as i32, b as i32 - l as i32)?;
                }
            }
            let out = &mut data.as_mut_slice()[index * size..(index + 1) * size];
            for a in 0..p {
                for b in 0..p {
                    for cc in 0..p {
                        for d in 0..p {
                            // the sum over q of the addition theorem leaves one term, and only
                            // when m_a + m_b = m_c + m_d
                            out[((a * p + b) * p + cc) * p + d] = if a + b == cc + d {
                                c[a * p + cc] * c[d * p + b]
                            } else {
                                0.
                            };
                        }
                    }
                }
            }
            if let Some(work) = work.as_mut() {
                let w = work.as_mut_slice();
                for (dst, src) in w.iter_mut().zip(out.iter()) {
                    *dst = C64::new(*src, 0.);
                }
                for (axis, coef) in [&conj_t, &conj_t, &t, &t].into_iter().enumerate() {
                    rotate(w, p, axis, coef);
                }
                let imag = w.iter().map(|v| v.im.abs()).fold(0., f64::max);
                if imag > 1e-12 {
                    return Err(invalid(format!(
                        "real-basis Slater tensor has imaginary part {imag:e}"
                    )));
                }
                for (dst, src) in out.iter_mut().zip(w.iter()) {
                    *dst = src.re;
                }
            }
        }
        Ok(Self {
            l,
            basis,
            orders,
            data,
        })
    }
    pub fn l(&self) -> usize {
        self.l
    }
    /// The orbital count `2l + 1`.
    pub fn p(&self) -> usize {
        2 * self.l + 1
    }
    pub fn basis(&self) -> Basis {
        self.basis
    }
    /// The multipole orders `k = 0, 2, .., 2l`, the order of every per-`k` vector.
    pub fn orders(&self) -> &[usize] {
        &self.orders
    }
    /// Bytes reserved by the stored tensors.
    pub fn bytes(&self) -> usize {
        std::mem::size_of_val(self.data.as_slice())
    }
    /// The flat `p^4` tensor `a_k`; raises for a `k` that is not one of [`Self::orders`].
    pub fn tensor(&self, k: usize) -> Result<&[f64]> {
        let size = self.p().pow(4);
        let index = self
            .orders
            .iter()
            .position(|&o| o == k)
            .ok_or_else(|| invalid(format!("no Slater order k = {k} for l = {}", self.l)))?;
        Ok(&self.data.as_slice()[index * size..(index + 1) * size])
    }
    /// `a_k(a, b, c, d)` for orbital indices in `0..p`.
    pub fn value(&self, k: usize, [a, b, c, d]: [usize; 4]) -> Result<f64> {
        let p = self.p();
        if [a, b, c, d].iter().any(|&i| i >= p) {
            return Err(invalid(format!("orbital index out of range 0..{p}")));
        }
        Ok(self.tensor(k)?[((a * p + b) * p + c) * p + d])
    }
    fn trace(&self, k: usize, pair: impl Fn(usize, usize) -> [usize; 4]) -> Result<f64> {
        let (p, t) = (self.p(), self.tensor(k)?);
        let at = |[a, b, c, d]: [usize; 4]| t[((a * p + b) * p + c) * p + d];
        Ok((0..p)
            .flat_map(|a| (0..p).map(move |b| (a, b)))
            .map(|(a, b)| at(pair(a, b)))
            .sum())
    }
    /// The direct sum `sum_ab a_k(a,b,a,b)`: `p^2` for `k = 0`, zero otherwise.
    pub fn direct_sum(&self, k: usize) -> Result<f64> {
        self.trace(k, |a, b| [a, b, a, b])
    }
    /// The averaged-exchange sum `S_k = sum_ab a_k(a,b,b,a) = p^2 (l k l; 0 0 0)^2`.
    pub fn exchange_sum(&self, k: usize) -> Result<f64> {
        self.trace(k, |a, b| [a, b, b, a])
    }
    /// The weights `w_k` of `J = sum_k w_k F^k`, per [`Self::orders`], derived from the tensor:
    /// `w_k = (S_k - p [k = 0]) / (p (p - 1))` (the filled-shell identity; `w_0 = 0`).
    pub fn j_weights(&self) -> Result<Vec<f64>> {
        let p = self.p() as f64;
        self.orders
            .iter()
            .map(|&k| Ok((self.exchange_sum(k)? - if k == 0 { p } else { 0. }) / (p * (p - 1.))))
            .collect()
    }
    /// `J` of the Slater integrals `f` (per [`Self::orders`], eV or any one unit).
    pub fn exchange_j(&self, f: &[f64]) -> Result<f64> {
        if f.len() != self.orders.len() || f.iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(invalid(format!(
                "Slater integrals must be {} finite non-negative values (F0, F2, ..), got {f:?}",
                self.orders.len()
            )));
        }
        Ok(self.j_weights()?.iter().zip(f).map(|(w, f)| w * f).sum())
    }
    /// The Slater integrals `F^k` (per [`Self::orders`]) of `U = F^0`, `J` and the ratios:
    /// `F^2 = J / sum_{k>=2} w_k r_k` with `r_2 = 1`; raises for a negative or non-finite `U`
    /// or `J`, and for ratios that do not fit the shell.
    pub fn integrals(&self, u: f64, j: f64, ratios: &Ratios) -> Result<Vec<f64>> {
        ratios.validate(self.l)?;
        if !u.is_finite() || u < 0. || !j.is_finite() || j < 0. {
            return Err(invalid(format!(
                "U and J must be finite and non-negative for a Slater shell, got U = {u}, J = {j}"
            )));
        }
        let weights = self.j_weights()?;
        let scale = [1., ratios.f4_f2, ratios.f6_f2.unwrap_or(0.)];
        let denominator: f64 = (1..self.orders.len())
            .map(|i| weights[i] * scale[i - 1])
            .sum();
        let f2 = j / denominator;
        let mut out = vec![u];
        out.extend((1..self.orders.len()).map(|i| f2 * scale[i - 1]));
        Ok(out)
    }
}
