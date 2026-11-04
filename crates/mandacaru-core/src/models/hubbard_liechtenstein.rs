//! Rotationally invariant Liechtenstein DFT+U functional with the fully-localized-limit
//! double counting (Liechtenstein, Anisimov, Zaanen, PRB 52, R5467 (1995)).
//!
//! With `p = 2l + 1` orbitals `phi_a` of one shell and Slater integrals `F^k`, the
//! Coulomb interaction tensor is
//!
//! ```text
//! v_abcd = <ab|v|cd> = sum_k F^k a_k(a,b,c,d)
//! ```
//!
//! where `a_k(a,b,c,d)` is the angular coupling tensor from [`super::hubbard_slater`].
//! The Liechtenstein interaction energy and double-counting energy are
//!
//! ```text
//! E_Hub = 0.5 * sum_abcd v_abcd [ N_ca N_db - sum_s n^s_da n^s_cb ]
//! E_dc  = 0.5 * U * N (N - 1) - 0.5 * J * sum_s N_s (N_s - 1)
//! E_U   = E_Hub - E_dc
//! ```
//!
//! where `N_ca = sum_s n^s_ca`, `N_s = Tr(n^s)`, and `N = sum_s N_s`.
//! The one-electron potential `V^s_ab = dE_U / dn^s_ba` is
//!
//! ```text
//! V^s_ab = sum_cd [ v_acbd N_dc - v_acdb n^s_dc ] - delta_ab [ U (N - 1/2) - J (N_s - 1/2) ]
//! ```
//!
//! Units are the caller's: `U`, `J`, `E_U` and `V` share one energy unit.
use super::hubbard::{self, Evaluation, Form, Hubbard, Spin};
use super::hubbard_slater::{Basis, Coupling, Ratios};
use super::invalid;
use crate::{Context, Result};
use num_complex::Complex64 as C64;

/// Validates that angular momentum `l` corresponds to a `d` or `f` shell (`l = 2` or `3`).
pub fn check_shell(l: usize) -> Result<usize> {
    if l == 2 || l == 3 {
        Ok(l)
    } else {
        Err(invalid(format!(
            "Liechtenstein functional needs a d or f shell (l = 2 or 3), got l = {l}"
        )))
    }
}

/// Interaction parameters for a Liechtenstein functional evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct Liechtenstein {
    pub u: f64,
    pub j: f64,
    pub ratios: Ratios,
}

impl Liechtenstein {
    /// Builds interaction parameters for shell `l` with `U >= 0`, `J >= 0`, `J <= U` and
    /// valid ratios (or defaults for `l`).
    pub fn new(l: usize, u: f64, j: f64, ratios: Option<Ratios>) -> Result<Self> {
        check_shell(l)?;
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
                "Hubbard J = {j} exceeds U = {u}: U - J would be negative"
            )));
        }
        let ratios = match ratios {
            Some(r) => {
                r.validate(l)?;
                r
            }
            None => Ratios::default_for(l)?,
        };
        Ok(Self { u, j, ratios })
    }

    /// True when the correction is exactly the identity (no energy, no operator).
    /// For Liechtenstein, this requires both `U = 0` and `J = 0`.
    pub fn is_inert(&self) -> bool {
        self.u == 0. && self.j == 0.
    }

    /// Evaluates the functional using a pre-allocated [`Coupling`].
    pub fn evaluate(
        &self,
        ctx: &Context,
        coupling: &Coupling,
        spin: Spin,
        channels: &[&[C64]],
    ) -> Result<Evaluation> {
        evaluate_with_coupling(ctx, self.u, self.j, coupling, &self.ratios, spin, channels)
    }

    /// Evaluates the functional building a [`Coupling`] in `basis`.
    pub fn evaluate_with_basis(
        &self,
        ctx: &Context,
        l: usize,
        basis: Basis,
        spin: Spin,
        channels: &[&[C64]],
    ) -> Result<Evaluation> {
        let coupling = Coupling::new(ctx, l, basis)?;
        self.evaluate(ctx, &coupling, spin, channels)
    }
}

/// Evaluates the Liechtenstein functional with a given [`Coupling`] and [`Ratios`].
pub fn evaluate_with_coupling(
    ctx: &Context,
    u: f64,
    j: f64,
    coupling: &Coupling,
    ratios: &Ratios,
    spin: Spin,
    channels: &[&[C64]],
) -> Result<Evaluation> {
    let l = check_shell(coupling.l())?;
    ratios.validate(l)?;
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
            "Hubbard J = {j} exceeds U = {u}: U - J would be negative"
        )));
    }
    if channels.len() != spin.channels() {
        return Err(invalid(format!(
            "{spin:?} Liechtenstein evaluation needs {} occupation matrices, got {}",
            spin.channels(),
            channels.len()
        )));
    }
    let p = 2 * l + 1;
    for n in channels {
        let p_n = hubbard::validate(l, n)?;
        if p_n != p {
            return Err(invalid(format!(
                "occupation matrix size mismatch: got {p_n}, expected {p}"
            )));
        }
    }

    let mut potentials = Vec::with_capacity(channels.len());
    for _ in channels {
        potentials.push(ctx.allocate::<C64>(p * p)?);
    }

    let is_inert = u == 0. && j == 0.;

    // Slater integrals F^k and derivative dF^k/dJ
    let integrals = coupling.integrals(u, j, ratios)?;
    let weights = coupling.j_weights()?;
    let scale = [1., ratios.f4_f2, ratios.f6_f2.unwrap_or(0.)];
    let denominator: f64 = (1..coupling.orders().len())
        .map(|i| weights[i] * scale[i - 1])
        .sum();

    let p4 = p.pow(4);
    let mut v = vec![0.0f64; p4];
    let mut v_j = vec![0.0f64; p4];
    for (i, &k) in coupling.orders().iter().enumerate() {
        let fk = integrals[i];
        let dfk = if i == 0 {
            0.
        } else {
            scale[i - 1] / denominator
        };
        let tensor = coupling.tensor(k)?;
        for idx in 0..p4 {
            v[idx] += fk * tensor[idx];
            v_j[idx] += dfk * tensor[idx];
        }
    }

    let n0 = channels[0];
    let n1 = if spin == Spin::Restricted {
        channels[0]
    } else {
        channels[1]
    };

    let mut n_total = vec![C64::default(); p * p];
    for idx in 0..p * p {
        n_total[idx] = n0[idx] + n1[idx];
    }

    let mut n_s = [0.0f64; 2];
    for a in 0..p {
        n_s[0] += n0[a * p + a].re;
        n_s[1] += n1[a * p + a].re;
    }
    let n_tot = n_s[0] + n_s[1];

    let mut e_dir = 0.0f64;
    let mut de_dir_dj = 0.0f64;
    let mut e_exch = 0.0f64;
    let mut de_exch_dj = 0.0f64;

    for a in 0..p {
        for b in 0..p {
            for c in 0..p {
                for d in 0..p {
                    let idx = ((a * p + b) * p + c) * p + d;
                    let v_val = v[idx];
                    let v_j_val = v_j[idx];
                    if v_val == 0.0 && v_j_val == 0.0 {
                        continue;
                    }

                    // Direct: N_ca * N_db
                    let term_dir = (n_total[c * p + a] * n_total[d * p + b]).re;
                    e_dir += v_val * term_dir;
                    de_dir_dj += v_j_val * term_dir;

                    // Exchange: sum_s n^s_da * n^s_cb
                    let term_exch = (n0[d * p + a] * n0[c * p + b]).re
                        + (n1[d * p + a] * n1[c * p + b]).re;
                    e_exch += v_val * term_exch;
                    de_exch_dj += v_j_val * term_exch;
                }
            }
        }
    }

    let interaction = 0.5 * (e_dir - e_exch);
    let d_interaction_dj = 0.5 * (de_dir_dj - de_exch_dj);

    let double_counting = 0.5 * u * n_tot * (n_tot - 1.0)
        - 0.5 * j * (n_s[0] * (n_s[0] - 1.0) + n_s[1] * (n_s[1] - 1.0));
    let d_double_counting_dj = -0.5 * (n_s[0] * (n_s[0] - 1.0) + n_s[1] * (n_s[1] - 1.0));

    let energy = if is_inert {
        0.0
    } else {
        interaction - double_counting
    };
    let per_j = d_interaction_dj - d_double_counting_dj;

    // dE/dU = (1/2) * sum_s Tr[n_s(1 - n_s)]
    let mut per_u_eff = 0.0f64;
    for &ch in &[n0, n1] {
        let mut tr = 0.0f64;
        let mut sq = 0.0f64;
        for a in 0..p {
            tr += ch[a * p + a].re;
            for b in 0..p {
                let h = 0.5 * (ch[a * p + b] + ch[b * p + a].conj());
                sq += h.norm_sqr();
            }
        }
        per_u_eff += 0.5 * (tr - sq);
    }

    // Potential V^s_ab = sum_cd [ v_acbd N_dc - v_acdb n^s_dc ] - delta_ab [ U (N - 1/2) - J (N_s - 1/2) ]
    if !is_inert {
        for s in 0..channels.len() {
            let ns = channels[s];
            let ns_scalar = n_s[s];
            let out = potentials[s].as_mut_slice();

            for a in 0..p {
                for b in 0..p {
                    let mut dir = C64::default();
                    let mut exch = C64::default();
                    for c in 0..p {
                        for d in 0..p {
                            let v_dir = v[((a * p + c) * p + b) * p + d];
                            let v_exc = v[((a * p + c) * p + d) * p + b];
                            dir += v_dir * n_total[d * p + c];
                            exch += v_exc * ns[d * p + c];
                        }
                    }
                    let dc = if a == b {
                        C64::new(u * (n_tot - 0.5) - j * (ns_scalar - 0.5), 0.)
                    } else {
                        C64::default()
                    };
                    out[a * p + b] = dir - exch - dc;
                }
            }

            // Ensure exact Hermiticity (eliminating floating-point asymmetry < 1e-16)
            for a in 0..p {
                out[a * p + a] = C64::new(out[a * p + a].re, 0.0);
                for b in (a + 1)..p {
                    let h = 0.5 * (out[a * p + b] + out[b * p + a].conj());
                    out[a * p + b] = h;
                    out[b * p + a] = h.conj();
                }
            }
        }
    }

    Ok(Evaluation {
        energy,
        per_u_eff,
        per_j,
        interaction: if is_inert { 0.0 } else { interaction },
        double_counting: if is_inert { 0.0 } else { double_counting },
        potentials,
    })
}

/// Evaluates the Liechtenstein functional with specified ratios and basis.
pub fn evaluate_with_ratios(
    ctx: &Context,
    u: f64,
    j: f64,
    l: usize,
    ratios: &Ratios,
    basis: Basis,
    spin: Spin,
    channels: &[&[C64]],
) -> Result<Evaluation> {
    let model = Liechtenstein::new(l, u, j, Some(*ratios))?;
    model.evaluate_with_basis(ctx, l, basis, spin, channels)
}

/// Evaluates the Liechtenstein functional with default ratios and specified basis.
pub fn evaluate_with_basis(
    ctx: &Context,
    u: f64,
    j: f64,
    l: usize,
    basis: Basis,
    spin: Spin,
    channels: &[&[C64]],
) -> Result<Evaluation> {
    let ratios = Ratios::default_for(l)?;
    evaluate_with_ratios(ctx, u, j, l, &ratios, basis, spin, channels)
}

/// Evaluates the Liechtenstein functional for shell `l` with default ratios and `Basis::Complex`.
pub fn evaluate(
    ctx: &Context,
    h: &Hubbard,
    l: usize,
    spin: Spin,
    channels: &[&[C64]],
) -> Result<Evaluation> {
    check_shell(l)?;
    if h.form != Form::LiechtensteinFll {
        return Err(invalid(format!(
            "expected Liechtenstein functional, got {:?}",
            h.form
        )));
    }
    let ratios = Ratios::default_for(l)?;
    let model = Liechtenstein::new(l, h.u, h.j, Some(ratios))?;
    model.evaluate_with_basis(ctx, l, Basis::Complex, spin, channels)
}
