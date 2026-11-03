//! Ghost states and missing states of the smooth Hamiltonians.
//!
//! A *ghost* is a bound level the all-electron atom does not have: a deep local well that the
//! nonlocal projectors do not lift (below the reference, [`ghost_errors`]), a spurious level
//! above a compact reference (the excited ghost), or a bound state the local potential alone
//! binds in a channel without projectors ([`local_potential_ghosts`]). A *missing state* is the
//! opposite failure: a physical bound reference the smooth Hamiltonian does not bind at its
//! energy ([`missing_states`]).
use super::{
    reference::AllElectron,
    spectrum::{ghost_levels, spectrum},
    view::{Channel, Key, Smooth},
    Error, Outcome,
};
use crate::{
    paw::checks::{bound_levels, interpolate_linear, node_count},
    Context,
};
use std::collections::BTreeMap;

/// How far (Hartree) a channel's lowest eigenvalue must lie below its bound reference, with
/// the reference itself displaced to second place, to count as a ghost: the displacement tells
/// a ghost from an inaccurate level (a d channel whose one level sits 6e-4 Ha low has no extra
/// state, a ghosted s channel has an extra state 0.1 to 100 Ha down and its true level above).
pub const GHOST_TOLERANCE: f64 = 1e-4;
/// Bound levels deeper than this (Hartree) are counted against the atom (excited ghosts);
/// shallower ones depend on the box.
pub const EXCITED_GHOST_MARGIN: f64 = 0.1;

fn finite(smooth: &Smooth, values: &[f64], what: &str) -> Outcome<()> {
    match values.iter().find(|v| !v.is_finite()) {
        Some(v) => Err(Error::Unevaluated(format!(
            "{}: {what} is {v}; a check that cannot be evaluated is a failure, not a pass",
            smooth.symbol
        ))),
        None => Ok(()),
    }
}

/// Valence states of the all-electron atom below `energy` in channel `l`, its core and frozen
/// shells not counted: the nodes of the regular solution at `energy` (oscillation theorem).
pub fn all_electron_count(
    ctx: &Context,
    smooth: &Smooth,
    ae: &AllElectron<'_>,
    l: usize,
    energy: f64,
) -> Outcome<i64> {
    let treatment = ae.scalar_treatment();
    let (nodes, total) = node_count(
        ctx,
        ae.atom.r.as_slice(),
        ae.atom.effective.as_slice(),
        l,
        energy,
        ae.z,
        treatment,
    )?;
    if !total.is_finite() {
        return Err(Error::Unevaluated(format!(
            "{}: the l={l} all-electron wave at {energy:+.3} Ha is {total}",
            smooth.symbol
        )));
    }
    Ok(nodes as i64 - ae.core_count(l) as i64)
}

/// `{l: eps_ps - eps_ae}` for every angular momentum *without* projectors (up to one above the
/// highest channel) in which the screened local potential alone binds more states than the
/// all-electron atom has valence states of that `l`. Both spectra are counted below zero in the
/// same box, on the reference atom's uniform grid.
pub fn local_potential_ghosts(
    smooth: &Smooth,
    ae: &AllElectron<'_>,
) -> Outcome<BTreeMap<usize, f64>> {
    let r = ae.atom.r.as_slice();
    let local = interpolate_linear(r, smooth.r.as_slice(), smooth.v_local_screened.as_slice())?;
    let constructed = smooth.angular_momenta();
    let highest = constructed.iter().copied().max().unwrap_or(0);
    let bound = |v: &[f64], what: &str, l: usize| -> Outcome<Vec<f64>> {
        bound_levels(r, v, l, None, 0.).map_err(|e| {
            Error::Unevaluated(format!(
                "{}: {what} (l={l}) is not finite: {e}",
                smooth.symbol
            ))
        })
    };
    let mut out = BTreeMap::new();
    for l in 0..highest + 2 {
        if constructed.contains(&l) {
            continue;
        }
        let ps = bound(&local, "the screened local potential", l)?;
        let atom = bound(
            ae.atom.effective.as_slice(),
            "the all-electron potential",
            l,
        )?;
        let allowed = atom.get(ae.core_count(l)..).unwrap_or(&[]);
        if ps.len() > allowed.len() {
            let reference = allowed.first().copied().unwrap_or(0.);
            out.insert(l, ps[0] - reference);
        }
    }
    Ok(out)
}

/// The depth (Hartree, negative) of the extra state a bound channel holds below its references,
/// from its lowest `levels`: the lowest level lies more than [`GHOST_TOLERANCE`] below the
/// reference and the second one is closer to the reference than the first; a semicore channel's
/// second bound reference must be its second level.
fn extra_state(channel: &Channel, levels: &[f64]) -> Option<f64> {
    let reference = channel.energies[0];
    let (first, second) = (levels[0], levels[1]);
    let mut found = (first - reference < -GHOST_TOLERANCE
        && (second - reference).abs() < (first - reference).abs())
    .then_some(first - reference);
    let occupied = channel.occupied_energies();
    if occupied.len() > 1 && levels.len() > 2 {
        let (upper, third) = (levels[1], levels[2]);
        if upper - occupied[1] < -GHOST_TOLERANCE
            && (third - occupied[1]).abs() < (upper - occupied[1]).abs()
        {
            found = Some(found.unwrap_or(0.).min(upper - occupied[1]));
        }
    }
    found
}

/// `{l: eps_0 - eps_ref}` for every channel holding a ghost state. A ghost is an *extra*
/// state: the lowest level lies more than [`GHOST_TOLERANCE`] below the reference and the
/// second one is closer to the reference than the first; a semicore channel's second
/// reference must be its second level; a frozen-core scattering channel binds nothing. With the
/// all-electron atom, also a channel holding more levels below `-EXCITED_GHOST_MARGIN` than the
/// atom has valence states there, and the channels without projectors. A level that is not
/// finite is [`Error::Unevaluated`].
pub fn ghost_errors(
    ctx: &Context,
    smooth: &Smooth,
    ae: Option<&AllElectron<'_>>,
) -> Outcome<BTreeMap<usize, f64>> {
    let mut out = BTreeMap::new();
    for channel in smooth.channels.values().filter(|c| c.kappa.is_none()) {
        let l = channel.l;
        let reference = channel.energies[0];
        if reference >= 0. {
            // A scattering-only channel binds nothing unless a frozen subshell sits above it;
            // without one its spectrum is not even looked at.
            if !smooth.frozen.is_empty() {
                let levels = ghost_levels(ctx, smooth, channel)?;
                finite(smooth, &levels, &format!("an l={l} level"))?;
                if levels[0] < -GHOST_TOLERANCE {
                    out.insert(l, levels[0]);
                }
            }
            continue;
        }
        let levels = ghost_levels(ctx, smooth, channel)?;
        finite(smooth, &levels, &format!("an l={l} level"))?;
        if let Some(depth) = extra_state(channel, &levels) {
            out.insert(l, depth);
        }
        if let Some(ae) = ae {
            let mut deep = levels
                .iter()
                .copied()
                .filter(|&e| e < -EXCITED_GHOST_MARGIN)
                .collect::<Vec<_>>();
            deep.sort_by(f64::total_cmp);
            let allowed =
                all_electron_count(ctx, smooth, ae, l, -EXCITED_GHOST_MARGIN)?.max(0) as usize;
            if deep.len() > allowed {
                let deepest = out.get(&l).copied().unwrap_or(0.).min(deep[allowed]);
                out.insert(l, deepest);
            }
        }
    }
    if let Some(ae) = ae {
        out.extend(local_potential_ghosts(smooth, ae)?);
    }
    Ok(out)
}

/// `{(l, kappa): eps_0 - eps_ref}` for every `j` branch of a fully relativistic dataset
/// ([`super::kappa::branches`]) that holds a ghost: the same test as [`ghost_errors`] on each
/// branch as a channel of its own, against its own Dirac reference energy.
///
/// Only a branch with an occupied *bound* reference is a candidate. A branch whose reference is a
/// scattering energy (an empty f channel above the frozen 4f shell) has no bound pseudo state to
/// compare with: its lowest level less the reference, `kappa::level_errors`, is the box
/// continuum following the wall, not a ghost (Au and Pb f, -0.20 Ha), and is not reported here.
/// There is no all-electron count for a branch, so the excited-ghost test of the scalar channels
/// is not applied.
pub fn branch_ghost_errors(ctx: &Context, branches: &Smooth) -> Outcome<BTreeMap<Key, f64>> {
    let mut out = BTreeMap::new();
    for (key, channel) in &branches.channels {
        if key.kappa.is_none() || channel.energies[0] >= 0. {
            continue;
        }
        let levels = ghost_levels(ctx, branches, channel)?;
        finite(
            branches,
            &levels,
            &format!("an l={} kappa={} level", key.l, key.kappa.unwrap_or(0)),
        )?;
        if let Some(depth) = extra_state(channel, &levels) {
            out.insert(*key, depth);
        }
    }
    Ok(out)
}

/// How far (Hartree) the nearest smooth level may sit from an occupied bound reference before
/// the reference counts as not bound by the smooth Hamiltonian. Set from [`reference_offsets`]
/// over the shipped libraries (A5 log): the files that store no defect stay within 7.5e-3 (Ta's
/// frozen 4f) and otherwise within 4.5e-3; only four flagged heavy-lanthanide 4f channels
/// (Yb, Tm) reach 1.1e-2 to 1.7e-2. A removed oxygen 2p projector is 36 times this, a removed
/// sodium 2s projector 139 times.
pub const MISSING_TOLERANCE: f64 = 1e-2;

/// `{l: nearest level - reference}` for every channel with an occupied bound state, the
/// reference whose nearest smooth level is farthest away (signed, Hartree). A well-built channel
/// has a level on each of its occupied references; a ghost adds a level *below* them and does not
/// move them.
pub fn reference_offsets(ctx: &Context, smooth: &Smooth) -> Outcome<BTreeMap<usize, f64>> {
    let mut out = BTreeMap::new();
    for channel in smooth.channels.values().filter(|c| c.kappa.is_none()) {
        let bound = channel.bound_occupied_energies();
        if bound.is_empty() {
            continue;
        }
        // One level more than the occupied references and at least three, so a ghost below a
        // reference does not hide it.
        let count = (channel.occupied_energies().len() + 2).max(3);
        let levels = spectrum(ctx, smooth, channel, count)?;
        finite(smooth, &levels, &format!("an l={} level", channel.l))?;
        let mut worst = 0f64;
        for &reference in &bound {
            let nearest = levels
                .iter()
                .map(|&e| e - reference)
                .min_by(|a, b| a.abs().total_cmp(&b.abs()))
                .unwrap_or(f64::NAN);
            if nearest.abs() > worst.abs() {
                worst = nearest;
            }
        }
        out.insert(channel.l, worst);
    }
    Ok(out)
}

/// The channels of [`reference_offsets`] beyond [`MISSING_TOLERANCE`]: a physical bound state
/// the smooth Hamiltonian does not have where the atom does. The opposite of a ghost, and not
/// seen by [`ghost_errors`] (a level too high is no extra state).
pub fn missing_states(ctx: &Context, smooth: &Smooth) -> Outcome<BTreeMap<usize, f64>> {
    Ok(reference_offsets(ctx, smooth)?
        .into_iter()
        .filter(|(_, d)| d.abs() > MISSING_TOLERANCE)
        .collect())
}
