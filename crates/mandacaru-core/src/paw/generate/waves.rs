//! The smooth partial waves of every channel and the local potential they are assembled in.
//! Atomic units.
//!
//! Each channel's references are matched in value and first three derivatives at its cutoff and
//! the residual kinetic energy beyond `q_cut` is minimized in the freedom that is left
//! ([`pseudize::optimize`]). **No norm is conserved**: with a norm deficit `s` the inner-norm
//! matrix is set to `(1 - s)` times the all-electron one, so the overlap correction
//! `q = s <phi_i|phi_j>` is positive definite and the PAW-LCAO overlap operator is bounded below
//! by one. A dataset built with free waves has no such guarantee. `s = 0` is the unitary
//! construction (UPAW-LCAO): `q` vanishes identically. A channel without a bound state (an
//! `extra_l` channel) carries no charge, so asking it to also give up norm would over-constrain
//! the expansion; its norm is conserved exactly and its `q` is identically zero.
use super::{
    recipe::Recipe,
    reference::{anchors, equation, reference_waves, ChannelReference},
    shells::Shells,
    tables_elements::GGA_LOCAL_FIT_FRACTION,
};
use crate::{
    models::atom::Atom,
    paw::{
        checks::spurious_nodes,
        pseudize::{optimize, NormCondition, PseudoInput, PseudoWaves},
        radial::local_potential,
    },
    Buffer, Context, Result,
};
use std::collections::{BTreeMap, BTreeSet};

/// The references of every channel with their smooth partial waves.
pub struct Pseudized {
    pub channels: BTreeMap<usize, ChannelReference>,
    pub smooth: BTreeMap<usize, PseudoWaves>,
    /// The first reference energy the empty channels ended on (`None`: the highest occupied
    /// level, or no empty channel).
    pub anchor: Option<f64>,
    pub notes: Vec<String>,
}

/// The Bessel expansions of the smooth waves of one channel.
pub fn smooth_waves(
    ctx: &Context,
    recipe: &Recipe,
    atom: &Atom,
    channel: &ChannelReference,
    bound: bool,
) -> Result<PseudoWaves> {
    let norm = NormCondition::Factor(if bound { 1. - recipe.norm_deficit } else { 1. });
    optimize(
        ctx,
        &PseudoInput {
            r: atom.r.as_slice(),
            v_ae: atom.effective.as_slice(),
            l: channel.l,
            waves: &channel.waves,
            energies: &channel.energies,
            r_cut: channel.r_cut,
            q_cut: recipe.q_cut,
            n_bessel: recipe.n_bessel,
            norm,
            equation: equation(recipe),
        },
    )
}

/// Whether a smooth wave has more nodes inside its cutoff than its all-electron wave:
/// pseudization removes nodes, it never has to add one. Lithium's p channel anchored at the 2s
/// energy came out with a node inside 2.6 Bohr that its all-electron wave does not have,
/// couplings of -1.8 to -2.7 Ha and bcc Li 3 percent too short.
fn gained_node(atom: &Atom, channel: &ChannelReference, smooth: &PseudoWaves) -> Result<bool> {
    let count = channel.energies.len();
    let terms = smooth.wavevectors.len();
    let mut coefficients = Vec::with_capacity(count * terms);
    let mut wavevectors = Vec::with_capacity(count * terms);
    for i in 0..count {
        for k in 0..terms {
            coefficients.push(smooth.coefficients[(i, k)]);
            wavevectors.push(smooth.wavevectors[k]);
        }
    }
    Ok(!spurious_nodes(
        channel.l,
        smooth.r_cut,
        atom.r.as_slice(),
        &channel.waves,
        &wavevectors,
        &coefficients,
        count,
    )?
    .is_empty())
}

/// The references and smooth waves of every channel. The empty channels' first reference tries
/// the anchors of [`anchors`] in order and keeps the first whose smooth waves gain no node (the
/// last one when none does).
pub fn pseudize_channels(
    ctx: &Context,
    recipe: &Recipe,
    atom: &Atom,
    shells: &Shells,
    table: &BTreeMap<usize, f64>,
    extra_l: usize,
) -> Result<Pseudized> {
    let bound = shells
        .valence
        .iter()
        .map(|s| s.0 .1)
        .collect::<BTreeSet<_>>();
    let candidates = anchors(ctx, atom, shells, extra_l)?;
    let mut last = None;
    for anchor in candidates {
        let (channels, notes) = reference_waves(ctx, recipe, atom, shells, table, extra_l, anchor)?;
        let mut smooth = BTreeMap::new();
        for (&l, channel) in &channels {
            smooth.insert(
                l,
                smooth_waves(ctx, recipe, atom, channel, bound.contains(&l))?,
            );
        }
        let mut gained = false;
        for (l, channel) in &channels {
            if !bound.contains(l) && gained_node(atom, channel, &smooth[l])? {
                gained = true;
                break;
            }
        }
        last = Some(Pseudized {
            channels,
            smooth,
            anchor,
            notes,
        });
        if !gained {
            break;
        }
    }
    Ok(last.expect("the anchor list is never empty"))
}

/// The screened local potential: the even-polynomial continuation of the all-electron potential
/// inside `r_local`, with the origin raised by the recipe's shift. A gradient-corrected
/// potential is read over a window of [`GGA_LOCAL_FIT_FRACTION`] of the radius.
pub fn screened_local_potential(
    ctx: &Context,
    recipe: &Recipe,
    atom: &Atom,
    r_local: f64,
) -> Result<Buffer<f64>> {
    let fit_window = (recipe.functional != "lda").then(|| GGA_LOCAL_FIT_FRACTION * r_local);
    local_potential(
        ctx,
        atom.r.as_slice(),
        atom.effective.as_slice(),
        r_local,
        recipe.local_shift,
        fit_window,
    )
}
