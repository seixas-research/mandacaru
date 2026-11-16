//! Noncollinear spin in the local frame, shared by the crystal and molecular spinor engines.
//!
//! A spinor state has four density channels in the Pauli basis: the charge `n` and the
//! magnetization `m = (m_x, m_y, m_z)`, one block of `points` each, and for a meta-GGA the
//! kinetic density `(tau_0, tau_vec)` likewise. A collinear functional is evaluated point by
//! point in the frame of `m_hat = m / |m|`, `n_up/dn = (n +- |m|) / 2` (a semilocal functional).
//! The potentials `v_up/dn` return in the Pauli channels as `v_0 = (v_up + v_dn) / 2` and
//! `v_a = (v_up - v_dn) / 2 m_hat_a`.
//!
//! A meta-GGA (MG-F8) takes its frame `u` from the magnetization AND the spin kinetic density:
//! `u` is the leading eigenvector of `M = zeta zeta^T + eta eta^T`, with the relative
//! polarizations `zeta = m / n` and `eta = tau_vec / tau_0`, and every spin-odd variable is
//! projected on it: `n_up/dn = (n +- u . m) / 2`, `tau_up/dn = (tau_0 +- u . tau_vec) / 2` and
//! `grad n_up/dn = (grad n +- u . grad m) / 2` (the molecule's product rule and the crystal's
//! spectral gradients of the Pauli channels alike). A collinear state
//! has `M` of rank one and `u` its axis, so the spinor energy is the UKS energy exactly. The
//! `m_hat` frame of the semilocal functionals is the case without `eta`; for a meta-GGA it jumps
//! where `|m|` passes near zero while `tau_vec` does not, the frame derivative growing as
//! `|tau_vec| / |m|` (the Au-Be r2SCAN SCF that does not converge, `PAW_MIGRATION_LOG.md`
//! 2026-11-09 Q8/Q9). With `eta` in `M`, `u` follows `tau_vec` there and stays smooth; it is
//! undefined only where the two leading eigenvalues of `M` meet (`zeta` perpendicular to `eta`
//! with `|zeta| = |eta|`, curves in space), and where `M` vanishes the projected variables do.
//! The frame derivative `du = R dM u`, `R = (1 - u u^T) / l_1 + u_2 u_2^T l_2 / (l_1 (l_1 - l_2))`
//! (`l_1 >= l_2` the nonzero eigenvalues, `u_2` the second eigenvector), returns through
//! [`unrotate`] and [`unrotate_gradients`], so the potential stays the derivative of the energy
//! (`tests/periodic_spinor.rs`, `tests/molecular_spin_orbit.rs`).
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

/// Below this a density or kinetic density does not scale its polarization (`zeta = m / n`).
const SCALE_FLOOR: f64 = 1e-14;
/// Below this leading eigenvalue of `M` the frame of a meta-GGA is undefined (no polarization).
const FRAME_FLOOR: f64 = 1e-24;
/// Entries per point of [`Rotated::frames`]: `u`, `u_2`, `l_1`, `l_2`.
const FRAME: usize = 8;

/// A spinor state rotated into its local frame.
pub struct Rotated {
    /// `(n_up, n_dn)`, two blocks of `points`.
    pub densities: Buffer<f64>,
    /// `|m|` per point.
    pub norms: Buffer<f64>,
    /// `(tau_up, tau_dn)` for a meta-GGA.
    pub tau: Option<Buffer<f64>>,
    /// A meta-GGA's frame per point, `[point][u, u_2, l_1, l_2]` (module docs).
    frames: Option<Buffer<f64>>,
}

/// The polarizations `(zeta, eta)` of point `k` and the scales they divide by.
fn polarizations(fields: &[f64], tau: &[f64], n: usize, k: usize) -> ([f64; 3], [f64; 3], f64, f64) {
    let (charge, kinetic) = (fields[k].max(SCALE_FLOOR), tau[k].max(SCALE_FLOOR));
    (
        std::array::from_fn(|a| fields[(a + 1) * n + k] / charge),
        std::array::from_fn(|a| tau[(a + 1) * n + k] / kinetic),
        charge,
        kinetic,
    )
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The leading eigenvector `u` of `zeta zeta^T + eta eta^T`, oriented so that `u . zeta >= 0`
/// (`u . eta >= 0` without `zeta`), the second one `u_2` and the two nonzero eigenvalues, from
/// the 2 x 2 Gram matrix of `(zeta, eta)`.
fn frame(zeta: &[f64; 3], eta: &[f64; 3]) -> [f64; FRAME] {
    let (a, b, c) = (dot(zeta, zeta), dot(eta, eta), dot(zeta, eta));
    let root = (0.25 * (a - b) * (a - b) + c * c).sqrt();
    let (first, second) = (0.5 * (a + b) + root, (0.5 * (a + b) - root).max(0.));
    if first <= FRAME_FLOOR {
        return [0., 0., 1., 1., 0., 0., 0., 0.];
    }
    // The Gram eigenvector of `first`, from the better conditioned of its two forms.
    let w = if a >= b { [first - b, c] } else { [c, first - a] };
    let combine = |w: [f64; 2]| -> [f64; 3] { std::array::from_fn(|i| w[0] * zeta[i] + w[1] * eta[i]) };
    let mut u = combine(w);
    let length = dot(&u, &u).sqrt();
    u.iter_mut().for_each(|x| *x /= length);
    let reference = if dot(zeta, zeta) > 0. { dot(&u, zeta) } else { dot(&u, eta) };
    if reference < 0. {
        u.iter_mut().for_each(|x| *x = -*x);
    }
    let mut v = if second > FRAME_FLOOR { combine([-w[1], w[0]]) } else { [0.; 3] };
    // Orthogonal to `u` by construction up to roundoff; made exact.
    let along = dot(&v, &u);
    v.iter_mut().zip(&u).for_each(|(x, y)| *x -= along * y);
    let length = dot(&v, &v).sqrt();
    if length > 0. {
        v.iter_mut().for_each(|x| *x /= length);
    }
    [u[0], u[1], u[2], v[0], v[1], v[2], first, second]
}

/// `p = R q` of a frame (module docs): the response of the projections to the frame's turn.
fn turn(f: &[f64], q: &[f64; 3]) -> [f64; 3] {
    let (u, v, first, second) = ([f[0], f[1], f[2]], [f[3], f[4], f[5]], f[6], f[7]);
    if first <= FRAME_FLOOR {
        return [0.; 3];
    }
    let (along, across) = (dot(&u, q), dot(&v, q));
    // A meeting of the two leading eigenvalues leaves the frame undefined; the gap is floored.
    let gap = (first - second).max(1e-12 * first);
    let pair = if second > 0. { second / (first * gap) } else { 0. };
    std::array::from_fn(|i| (q[i] - along * u[i]) / first + pair * across * v[i])
}

/// Adds the frame term of `q = dE/du` at point `k` to the Pauli potentials: `dE = p . dM u` with
/// `p = R q`, through `zeta = m / n` (the magnetization and charge channels of `potential`) and
/// `eta = tau_vec / tau_0` (the kinetic channels of `kinetic`).
fn add_turn(
    f: &[f64],
    q: &[f64; 3],
    (fields, tau): (&[f64], &[f64]),
    n: usize,
    k: usize,
    potential: &mut [f64],
    kinetic: &mut [f64],
) {
    let p = turn(f, q);
    let u = [f[0], f[1], f[2]];
    let (zeta, eta, charge, scale) = polarizations(fields, tau, n, k);
    let response = |x: &[f64; 3]| -> [f64; 3] {
        let (xu, px) = (dot(x, &u), dot(&p, x));
        std::array::from_fn(|i| p[i] * xu + u[i] * px)
    };
    let (dzeta, deta) = (response(&zeta), response(&eta));
    for a in 0..3 {
        potential[(a + 1) * n + k] += dzeta[a] / charge;
        kinetic[(a + 1) * n + k] += deta[a] / scale;
    }
    if fields[k] > SCALE_FLOOR {
        potential[k] -= dot(&dzeta, &zeta) / charge;
    }
    if tau[k] > SCALE_FLOOR {
        kinetic[k] -= dot(&deta, &eta) / scale;
    }
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
    let Some(t) = tau else {
        return Ok(Rotated { densities, norms, tau: None, frames: None });
    };
    if t.iter().any(|v| !v.is_finite()) {
        return Err(invalid("a spinor meta-GGA needs finite kinetic densities"));
    }
    let mut frames = ctx.allocate::<f64>(FRAME * n)?;
    let mut out = ctx.allocate::<f64>(2 * n)?;
    for k in 0..n {
        let (zeta, eta, _, _) = polarizations(fields, t, n, k);
        let f = frame(&zeta, &eta);
        frames.as_mut_slice()[FRAME * k..FRAME * (k + 1)].copy_from_slice(&f);
        let u = [f[0], f[1], f[2]];
        let s = (0..3).map(|a| u[a] * fields[(a + 1) * n + k]).sum::<f64>();
        let along = (0..3).map(|a| u[a] * t[(a + 1) * n + k]).sum::<f64>();
        densities.as_mut_slice()[k] = 0.5 * (fields[k] + s);
        densities.as_mut_slice()[n + k] = 0.5 * (fields[k] - s);
        out.as_mut_slice()[k] = 0.5 * (t[k] + along);
        out.as_mut_slice()[n + k] = 0.5 * (t[k] - along);
    }
    Ok(Rotated { densities, norms, tau: Some(out), frames: Some(frames) })
}

/// The Pauli-channel potentials (`4 x points`) of the frame potentials `potential`
/// (`v_up, v_dn`) and, for a meta-GGA, `kinetic` (`v_tau,up, v_tau,dn`) with the state's `tau`:
/// `(v, v_tau)`, a meta-GGA's with the turn of its frame (module docs) through both.
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
    let meta = match (&mut kinetic_out, kinetic, tau, rotated.frames.as_ref()) {
        (Some(to), Some(v), Some(t), Some(frames)) => Some((to, v, t, frames)),
        (None, None, None, None) | (None, None, Some(_), None) => None,
        _ => return Err(invalid("a meta-GGA frame needs its kinetic densities and potentials")),
    };
    let Some((to, v, t, frames)) = meta else {
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
        }
        return Ok((out, kinetic_out));
    };
    for k in 0..n {
        let f = &frames.as_slice()[FRAME * k..FRAME * (k + 1)];
        let u = [f[0], f[1], f[2]];
        let (spin, kinetic_spin) = (0.5 * (potential[k] - potential[n + k]), 0.5 * (v[k] - v[n + k]));
        out.as_mut_slice()[k] = 0.5 * (potential[k] + potential[n + k]);
        to.as_mut_slice()[k] = 0.5 * (v[k] + v[n + k]);
        for a in 0..3 {
            out.as_mut_slice()[(a + 1) * n + k] = spin * u[a];
            to.as_mut_slice()[(a + 1) * n + k] = kinetic_spin * u[a];
        }
        // dE/du of the two projections, `u . m` and `u . tau_vec`.
        let q: [f64; 3] = std::array::from_fn(|a| spin * fields[(a + 1) * n + k] + kinetic_spin * t[(a + 1) * n + k]);
        add_turn(f, &q, (fields, t), n, k, out.as_mut_slice(), to.as_mut_slice());
    }
    Ok((out, kinetic_out))
}

/// The frame density gradients of a semilocal functional given the Pauli-channel gradients
/// `gradients` (`[channel][axis][point]`, `4 x 3 x points`): `grad n_up/dn = (grad n +- m_hat .
/// grad m) / 2` per axis (`[spin][axis][point]`, `2 x 3 x points`), where `m_hat . grad m =
/// grad |m|` and an even split where `|m|` vanishes; a meta-GGA's `(grad n +- u . grad m) / 2`
/// in its own frame (module docs). A caller that supplies its own density
/// gradient (the molecular product rule, the crystal's spectral gradients of a meta-GGA) needs
/// these instead of differentiating the rotated densities on the grid.
pub fn rotate_gradients(ctx: &Context, points: usize, fields: &[f64], rotated: &Rotated, gradients: &[f64]) -> Result<Buffer<f64>> {
    let n = points;
    if fields.len() != 4 * n || gradients.len() != 12 * n {
        return Err(invalid("frame gradients need four fields and their three derivatives each"));
    }
    let mut out = ctx.allocate::<f64>(6 * n)?;
    for k in 0..n {
        let norm = rotated.norms.as_slice()[k];
        let frame = rotated.frames.as_ref().map(|f| &f.as_slice()[FRAME * k..FRAME * k + 3]);
        for axis in 0..3 {
            let charge = gradients[axis * n + k];
            let along = if let Some(u) = frame {
                (0..3).map(|a| u[a] * gradients[((a + 1) * 3 + axis) * n + k]).sum::<f64>()
            } else if norm > NORM_FLOOR {
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
/// `d (m_hat . grad m) / d m_a = (grad m_a - (m_hat . grad m) m_hat_a) / |m|`. A meta-GGA's frame
/// turns with `tau_vec` too: `kinetic` (its kinetic densities and Pauli kinetic potentials) then
/// takes the gradient projections' share of the turn (module docs).
#[allow(clippy::too_many_arguments)]
pub fn unrotate_gradients(
    ctx: &Context,
    points: usize,
    fields: &[f64],
    rotated: &Rotated,
    gradients: &[f64],
    flux: &[f64],
    potential: &mut [f64],
    kinetic: Option<(&[f64], &mut [f64])>,
) -> Result<Buffer<f64>> {
    let n = points;
    if fields.len() != 4 * n || gradients.len() != 12 * n || flux.len() != 6 * n || potential.len() != 4 * n {
        return Err(invalid("frame fluxes are two spins of three axes of the grid"));
    }
    let mut out = ctx.allocate::<f64>(12 * n)?;
    if let Some(frames) = rotated.frames.as_ref() {
        let Some((tau, kinetic)) = kinetic else {
            return Err(invalid("a meta-GGA frame needs its kinetic densities and potentials"));
        };
        if tau.len() != 4 * n || kinetic.len() != 4 * n {
            return Err(invalid("a spinor meta-GGA needs four kinetic-density channels"));
        }
        for k in 0..n {
            let f = &frames.as_slice()[FRAME * k..FRAME * (k + 1)];
            let mut q = [0.; 3];
            for axis in 0..3 {
                let (up, down) = (flux[axis * n + k], flux[(3 + axis) * n + k]);
                out.as_mut_slice()[axis * n + k] = 0.5 * (up + down);
                let half = 0.5 * (up - down);
                for a in 0..3 {
                    out.as_mut_slice()[((a + 1) * 3 + axis) * n + k] = half * f[a];
                    q[a] += half * gradients[((a + 1) * 3 + axis) * n + k];
                }
            }
            add_turn(f, &q, (fields, tau), n, k, potential, kinetic);
        }
        return Ok(out);
    }
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
