//! Noncollinear spin in the local frame, shared by the crystal and molecular spinor engines.
//!
//! A spinor state has four density channels in the Pauli basis: the charge `n` and the
//! magnetization `m = (m_x, m_y, m_z)`, one block of `points` each, and for a meta-GGA the
//! kinetic density `(tau_0, tau_vec)` likewise. A collinear functional is evaluated point by
//! point in the frame of `m_hat = m / |m|`:
//! - `n_up/dn = (n +- |m|) / 2`;
//! - `tau_up/dn = (tau_0 +- m_hat . tau_vec) / 2`, with an even split where `|m|` vanishes.
//! The potentials `v_up/dn` (and `v_tau,up/dn`) return in the Pauli channels as
//! `v_0 = (v_up + v_dn) / 2` and `v_a = (v_up - v_dn) / 2 m_hat_a`. Because `tau_up/dn` depend on
//! `m_hat`, the magnetization potential of a meta-GGA also carries the derivative of the frame,
//! `(v_tau,up - v_tau,dn) / 2 (tau_a - (m_hat . tau_vec) m_hat_a) / |m|`, so the potential stays
//! the derivative of the energy (`tests/periodic_spinor.rs`).
//!
//! [`component`] and [`assemble`] are the matrix side: the Pauli components `D_a` of a `2m x 2m`
//! spinor density matrix, whose densities are the four channels, and the `2m x 2m` operator of
//! four channel matrices, its adjoint.
use crate::{Buffer, Context, Result};
use nalgebra::DMatrix;
use num_complex::Complex64 as C64;

/// Below this `|m|` the frame is undefined: the spin split is even and there is no transverse part.
pub const NORM_FLOOR: f64 = 1e-14;

fn invalid(message: &str) -> crate::Error {
    crate::Error::Invalid(message.into())
}

/// A spinor state rotated into its local frame.
pub struct Rotated {
    /// `(n_up, n_dn)`, two blocks of `points`.
    pub densities: Buffer<f64>,
    /// `|m|` per point.
    pub norms: Buffer<f64>,
    /// `(tau_up, tau_dn)` for a meta-GGA.
    pub tau: Option<Buffer<f64>>,
}

/// Rotates `fields` (`4 x points`: `n, m_x, m_y, m_z`) and `tau` (likewise) into the local frame.
pub fn rotate(ctx: &Context, points: usize, fields: &[f64], tau: Option<&[f64]>) -> Result<Rotated> {
    let n = points;
    if fields.len() != 4 * n || fields.iter().any(|v| !v.is_finite()) {
        return Err(invalid("a spinor state needs charge and three finite magnetization fields"));
    }
    if tau.is_some_and(|t| t.len() != 4 * n) {
        return Err(invalid("a spinor meta-GGA needs four kinetic-density channels"));
    }
    let mut densities = ctx.allocate::<f64>(2 * n)?;
    let mut norms = ctx.allocate::<f64>(n)?;
    for k in 0..n {
        let norm = (1..4).map(|a| fields[a * n + k].powi(2)).sum::<f64>().sqrt();
        norms.as_mut_slice()[k] = norm;
        densities.as_mut_slice()[k] = 0.5 * (fields[k] + norm);
        densities.as_mut_slice()[n + k] = 0.5 * (fields[k] - norm);
    }
    let tau = tau
        .map(|t| -> Result<Buffer<f64>> {
            let mut out = ctx.allocate::<f64>(2 * n)?;
            for k in 0..n {
                let along = along(fields, t, norms.as_slice()[k], n, k);
                out.as_mut_slice()[k] = 0.5 * (t[k] + along);
                out.as_mut_slice()[n + k] = 0.5 * (t[k] - along);
            }
            Ok(out)
        })
        .transpose()?;
    Ok(Rotated { densities, norms, tau })
}

/// `m_hat . tau_vec` at point `k` (zero where the frame is undefined).
fn along(fields: &[f64], tau: &[f64], norm: f64, n: usize, k: usize) -> f64 {
    if norm > NORM_FLOOR {
        (1..4).map(|a| fields[a * n + k] * tau[a * n + k]).sum::<f64>() / norm
    } else {
        0.
    }
}

/// The Pauli-channel potentials (`4 x points`) of the frame potentials `potential`
/// (`v_up, v_dn`) and, for a meta-GGA, `kinetic` (`v_tau,up, v_tau,dn`) with the state's `tau`:
/// `(v, v_tau)`, the magnetization part of `v` including the frame term.
pub fn unrotate(
    ctx: &Context,
    points: usize,
    fields: &[f64],
    rotated: &Rotated,
    potential: &[f64],
    kinetic: Option<&[f64]>,
    tau: Option<&[f64]>,
) -> Result<(Buffer<f64>, Option<Buffer<f64>>)> {
    let n = points;
    if potential.len() != 2 * n || kinetic.is_some_and(|v| v.len() != 2 * n) {
        return Err(invalid("frame potentials are two blocks of the grid"));
    }
    let mut out = ctx.allocate::<f64>(4 * n)?;
    let mut kinetic_out = match (kinetic, tau) {
        (Some(_), Some(_)) => Some(ctx.allocate::<f64>(4 * n)?),
        _ => None,
    };
    for k in 0..n {
        let (up, down) = (potential[k], potential[n + k]);
        out.as_mut_slice()[k] = 0.5 * (up + down);
        let norm = rotated.norms.as_slice()[k];
        if norm > NORM_FLOOR {
            let radial = 0.5 * (up - down) / norm;
            for a in 1..4 {
                out.as_mut_slice()[a * n + k] = radial * fields[a * n + k];
            }
        }
        if let (Some(to), Some(v), Some(t)) = (&mut kinetic_out, kinetic, tau) {
            let (vu, vd) = (v[k], v[n + k]);
            to.as_mut_slice()[k] = 0.5 * (vu + vd);
            if norm > NORM_FLOOR {
                let half = 0.5 * (vu - vd);
                let along = along(fields, t, norm, n, k);
                for a in 1..4 {
                    let direction = fields[a * n + k] / norm;
                    to.as_mut_slice()[a * n + k] = half * direction;
                    // d tau_up/dn / d m_a = +-(tau_a - along m_hat_a) / (2 |m|).
                    out.as_mut_slice()[a * n + k] += half * (t[a * n + k] - along * direction) / norm;
                }
            }
        }
    }
    Ok((out, kinetic_out))
}

/// The frame density gradients of a semilocal functional given the Pauli-channel gradients
/// `gradients` (`[channel][axis][point]`, `4 x 3 x points`): `grad n_up/dn = (grad n +- m_hat .
/// grad m) / 2` per axis (`[spin][axis][point]`, `2 x 3 x points`), where `m_hat . grad m =
/// grad |m|` and an even split where `|m|` vanishes. A caller that supplies its own density
/// gradient (the molecular product rule) needs these instead of differentiating the rotated
/// densities on the grid.
pub fn rotate_gradients(ctx: &Context, points: usize, fields: &[f64], rotated: &Rotated, gradients: &[f64]) -> Result<Buffer<f64>> {
    let n = points;
    if fields.len() != 4 * n || gradients.len() != 12 * n {
        return Err(invalid("frame gradients need four fields and their three derivatives each"));
    }
    let mut out = ctx.allocate::<f64>(6 * n)?;
    for k in 0..n {
        let norm = rotated.norms.as_slice()[k];
        for axis in 0..3 {
            let charge = gradients[axis * n + k];
            let along = if norm > NORM_FLOOR {
                (1..4).map(|a| fields[a * n + k] * gradients[(a * 3 + axis) * n + k]).sum::<f64>() / norm
            } else {
                0.
            };
            out.as_mut_slice()[axis * n + k] = 0.5 * (charge + along);
            out.as_mut_slice()[(3 + axis) * n + k] = 0.5 * (charge - along);
        }
    }
    Ok(out)
}

/// The adjoint of [`rotate_gradients`]: the Pauli-channel flux potentials (`dE / d grad n_a`,
/// `4 x 3 x points`) of the frame fluxes `flux` (`dE / d grad n_up/dn`, `2 x 3 x points`), and the
/// frame term added to the magnetization potentials `potential` (`4 x points`, channels 1..3):
/// `d (m_hat . grad m) / d m_a = (grad m_a - (m_hat . grad m) m_hat_a) / |m|`, the gradient
/// counterpart of the kinetic frame term of [`unrotate`].
pub fn unrotate_gradients(
    ctx: &Context,
    points: usize,
    fields: &[f64],
    rotated: &Rotated,
    gradients: &[f64],
    flux: &[f64],
    potential: &mut [f64],
) -> Result<Buffer<f64>> {
    let n = points;
    if fields.len() != 4 * n || gradients.len() != 12 * n || flux.len() != 6 * n || potential.len() != 4 * n {
        return Err(invalid("frame fluxes are two spins of three axes of the grid"));
    }
    let mut out = ctx.allocate::<f64>(12 * n)?;
    for k in 0..n {
        let norm = rotated.norms.as_slice()[k];
        for axis in 0..3 {
            let (up, down) = (flux[axis * n + k], flux[(3 + axis) * n + k]);
            out.as_mut_slice()[axis * n + k] = 0.5 * (up + down);
            if norm > NORM_FLOOR {
                let half = 0.5 * (up - down);
                let along = (1..4).map(|a| fields[a * n + k] * gradients[(a * 3 + axis) * n + k]).sum::<f64>() / norm;
                for a in 1..4 {
                    let direction = fields[a * n + k] / norm;
                    out.as_mut_slice()[(a * 3 + axis) * n + k] = half * direction;
                    potential[a * n + k] += half * (gradients[(a * 3 + axis) * n + k] - along * direction) / norm;
                }
            }
        }
    }
    Ok(out)
}

/// The Pauli component `D_a` (`m x m`) of a `2m x 2m` spinor matrix (alpha block first):
/// `D_0 = D_aa + D_bb`, `D_x = D_ab + D_ba`, `D_y = i (D_ab - D_ba)`, `D_z = D_aa - D_bb`.
pub fn component(d: &DMatrix<C64>, m: usize, a: usize) -> DMatrix<C64> {
    DMatrix::from_fn(m, m, |i, j| match a {
        0 => d[(i, j)] + d[(m + i, m + j)],
        1 => d[(i, m + j)] + d[(m + i, j)],
        2 => C64::i() * (d[(i, m + j)] - d[(m + i, j)]),
        _ => d[(i, j)] - d[(m + i, m + j)],
    })
}

/// Adds `sum_a sigma_a x V_a` to the `2m x 2m` operator `h` for the channel matrices `channels`
/// (`m x m`; one channel is the charge alone, four are `0, x, y, z`), the adjoint of
/// [`component`]: `tr(D h) = sum_a tr(D_a V_a)`.
pub fn assemble_into(h: &mut DMatrix<C64>, channels: &[DMatrix<C64>], m: usize) {
    for (a, v) in channels.iter().enumerate() {
        for i in 0..m {
            for j in 0..m {
                let value = v[(i, j)];
                match a {
                    0 => {
                        h[(i, j)] += value;
                        h[(m + i, m + j)] += value;
                    }
                    1 => {
                        h[(i, m + j)] += value;
                        h[(m + i, j)] += value;
                    }
                    2 => {
                        h[(i, m + j)] -= C64::i() * value;
                        h[(m + i, j)] += C64::i() * value;
                    }
                    _ => {
                        h[(i, j)] += value;
                        h[(m + i, m + j)] -= value;
                    }
                }
            }
        }
    }
}
