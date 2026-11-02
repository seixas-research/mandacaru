//! Scattering of the smooth channels against the all-electron atom.
//!
//! The smooth logarithmic derivative `R'/R` at a channel's matching radius is the outward
//! Numerov solution of the smooth radial equation with the projectors in the energy-dependent
//! coupling `D^scr - E q^norm` ([`log_derivative`]). A channel scatters like the atom when
//! the phase `arctan L` agrees with the all-electron one, modulo `pi` (a pole of `L` is not an
//! error), over a window of energies around its references ([`scattering_errors`]); a channel
//! without projectors scatters off the local potential alone ([`unprojected_errors`]).
use super::{
    reference::{AeLogs, AllElectron},
    view::{Channel, Key, Smooth},
    Error, Outcome,
};
use crate::{
    kernels,
    models::numerov::{radial_f, with_origin},
    paw::{
        checks::{log_derivative_of_u, phase_summary, window_offsets},
        invalid,
    },
    Context, Result,
};
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

/// A repaired channel must scatter within this many radians near its references...
pub const PHASE_TOLERANCE: f64 = 0.05;
/// ... `|offset| <= PHASE_WINDOW` Hartree, on a grid of `PHASE_STEP`.
pub const PHASE_WINDOW: f64 = 0.5;
pub const PHASE_STEP: f64 = 0.05;
/// A resonance just outside that window is caught by a looser bound over a wider one.
pub const RESONANCE_WINDOW: f64 = 1.0;
pub const RESONANCE_TOLERANCE: f64 = 0.3;
/// A channel without projectors is held to this many radians over the highest valence
/// reference `+- PHASE_WINDOW`.
pub const UNPROJECTED_TOLERANCE: f64 = 0.2;

/// `integral y dx` by the trapezoid rule on a uniform step (`np.trapezoid(y, dx=dx)`).
fn trapezoid(y: &[f64], dx: f64) -> f64 {
    y.windows(2).map(|w| dx * (w[0] + w[1]) / 2.).sum()
}

/// The smooth logarithmic derivative `R'/R` at `r_cut` (default: the projector radius,
/// `max(r_cut, r_cut_local)`) of the generalized problem at `energy`; `channel = None` is a
/// channel without projectors, which sees the screened local potential alone and needs `r_cut`.
pub fn log_derivative(
    ctx: &Context,
    smooth: &Smooth,
    l: usize,
    channel: Option<&Channel>,
    energy: f64,
    r_cut: Option<f64>,
) -> Result<f64> {
    let r_cut = match (r_cut, channel) {
        (Some(r), _) => r,
        (None, Some(c)) => c.projector_radius(smooth.r_cut_local),
        (None, None) => {
            return Err(invalid(format!(
                "diagnose: channel l={l} has no projectors: pass r_cut"
            )))
        }
    };
    let r = smooth.r.as_slice();
    let v = smooth.v_local_screened.as_slice();
    if r.len() < 5 || v.len() != r.len() {
        return Err(invalid("diagnose: the dataset's radial table is too short"));
    }
    let r0 = with_origin(r);
    let mut v0 = Vec::with_capacity(v.len() + 1);
    v0.push(v[0]);
    v0.extend_from_slice(v);
    let f = radial_f(&r0, &v0, l, energy);
    let h2 = (r0[1] - r0[0]).powi(2);
    let n0 = r0.len();
    let power = (l + 1) as f64;
    let mut start = vec![0.; n0];
    start[1] = r0[1].powf(power);
    start[2] = r0[2].powf(power);
    let zero = vec![0.; n0];
    let u0 = kernels::numerov(ctx, &f, &zero, &start, h2, 1, false)?;
    let Some(channel) = channel else {
        return log_derivative_of_u(&r0, u0.as_slice(), r_cut);
    };
    let rank = channel.rank;
    let n = r.len();
    let p_u = (0..rank)
        .map(|i| {
            let mut p = vec![0.];
            p.extend(
                channel.projectors.as_slice()[i * n..(i + 1) * n]
                    .iter()
                    .zip(r)
                    .map(|(p, r)| p * r),
            );
            p
        })
        .collect::<Vec<_>>();
    let dr = r0[1] - r0[0];
    let uj = p_u
        .iter()
        .map(|p| {
            let source = p.iter().map(|v| 2. * v).collect::<Vec<_>>();
            kernels::numerov(ctx, &f, &source, &vec![0.; n0], h2, 1, false)
        })
        .collect::<Result<Vec<_>>>()?;
    let product = |a: &[f64], b: &[f64]| {
        trapezoid(&a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<_>>(), dr)
    };
    let m0 = DVector::from_iterator(rank, p_u.iter().map(|p| product(p, u0.as_slice())));
    let m = DMatrix::from_fn(rank, rank, |i, j| product(&p_u[i], uj[j].as_slice()));
    let d = DMatrix::from_fn(rank, rank, |i, j| {
        channel.coupling[i * rank + j] - energy * channel.norm[i * rank + j]
    });
    let system = DMatrix::<f64>::identity(rank, rank) - &d * &m;
    let a = system
        .lu()
        .solve(&(&d * m0))
        .ok_or_else(|| invalid("diagnose: the smooth scattering problem is singular"))?;
    let mut u = u0.as_slice().to_vec();
    for (j, uj) in uj.iter().enumerate() {
        for (x, y) in u.iter_mut().zip(uj.as_slice()) {
            *x += a[j] * y;
        }
    }
    log_derivative_of_u(&r0, &u, r_cut)
}

fn unevaluated(smooth: &Smooth, what: &str, energy: f64) -> Error {
    Error::Unevaluated(format!(
        "{}: {what} at {energy:+.3} Ha is not finite; a check that cannot be evaluated is a failure, not a pass",
        smooth.symbol
    ))
}

/// `(near, far)` phase errors over `energies` (`offsets` from the window center).
fn summary(
    ctx: &Context,
    smooth: &Smooth,
    ae: &AllElectron<'_>,
    cache: &mut AeLogs,
    key: Key,
    channel: Option<&Channel>,
    what: &str,
    energies: &[f64],
    offsets: &[f64],
    r_match: f64,
) -> Outcome<(f64, f64)> {
    let all_electron = cache.at(ctx, ae, key, energies, r_match)?;
    let smooth_side = energies
        .iter()
        .map(|&e| log_derivative(ctx, smooth, key.l, channel, e, Some(r_match)))
        .collect::<Result<Vec<_>>>()?;
    let (near, far, bad) = phase_summary(&smooth_side, &all_electron, offsets, PHASE_WINDOW)?;
    match bad {
        Some(i) => Err(unevaluated(
            smooth,
            &format!("the l={} {what}", key.l),
            energies[i],
        )),
        None => Ok((near, far)),
    }
}

/// `{channel: (near, far)}`: the largest phase error `|arctan L_ps - arctan L_ae|` at each
/// channel's matching radius, within `eps_ref +- PHASE_WINDOW` (`near`) and
/// `+- RESONANCE_WINDOW` (`far`), around every occupied reference. A channel whose first
/// reference is a scattering energy (the empty f above a frozen 4f) is evaluated at the
/// positive energies of the same windows.
pub fn scattering_errors(
    ctx: &Context,
    smooth: &Smooth,
    ae: &AllElectron<'_>,
    cache: &mut AeLogs,
) -> Outcome<BTreeMap<Key, [f64; 2]>> {
    let offsets = window_offsets(RESONANCE_WINDOW, PHASE_STEP);
    let mut out = BTreeMap::new();
    for (&key, channel) in &smooth.channels {
        let reference = channel.energies[0];
        let r_match = smooth.match_radius(channel);
        let (mut near, mut far) = (0f64, 0f64);
        for center in channel.occupied_energies() {
            let kept = offsets
                .iter()
                .filter(|&&o| !(reference >= 0. && center + o <= 0.))
                .copied()
                .collect::<Vec<_>>();
            if kept.is_empty() {
                continue;
            }
            let energies = kept.iter().map(|o| center + o).collect::<Vec<_>>();
            let (n, f) = summary(
                ctx,
                smooth,
                ae,
                cache,
                key,
                Some(channel),
                "phase error",
                &energies,
                &kept,
                r_match,
            )?;
            near = near.max(n);
            far = far.max(f);
        }
        out.insert(key, [near, far]);
    }
    Ok(out)
}

/// `{l: error}`: the largest phase error of every `j`-averaged angular momentum *without*
/// projectors (up to one above the highest constructed channel), within `PHASE_WINDOW` of the
/// highest bound valence reference, compared past the local radius and every cutoff.
pub fn unprojected_errors(
    ctx: &Context,
    smooth: &Smooth,
    ae: &AllElectron<'_>,
    cache: &mut AeLogs,
) -> Outcome<BTreeMap<usize, f64>> {
    let constructed = smooth.angular_momenta();
    let bound = smooth
        .channels
        .values()
        .filter(|c| c.kappa.is_none())
        .flat_map(|c| c.occupied_energies())
        .filter(|&e| e < 0.)
        .collect::<Vec<_>>();
    let Some(reference) = bound.iter().copied().reduce(f64::max) else {
        return Err(Error::Unevaluated(format!(
            "{}: the unprojected-channel scattering check not evaluated: no bound occupied reference to center its energy window on",
            smooth.symbol
        )));
    };
    let widest = smooth
        .channels
        .values()
        .map(|c| c.r_cut)
        .fold(smooth.r_cut_local, f64::max);
    let r_match = super::snap_to(smooth.r.as_slice(), widest);
    let offsets = window_offsets(PHASE_WINDOW, PHASE_STEP);
    let energies = offsets.iter().map(|o| reference + o).collect::<Vec<_>>();
    let highest = constructed.iter().copied().max().unwrap_or(0);
    let mut out = BTreeMap::new();
    for l in 0..highest + 2 {
        if constructed.contains(&l) {
            continue;
        }
        let key = Key { l, kappa: None };
        let (_near, far) = summary(
            ctx,
            smooth,
            ae,
            cache,
            key,
            None,
            "(no projectors) phase error",
            &energies,
            &offsets,
            r_match,
        )?;
        out.insert(l, far);
    }
    Ok(out)
}
