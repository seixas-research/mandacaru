//! The reference atom of a PAW-LCAO dataset and the all-electron inputs of its channels: the
//! atom solve, the split of its subshells into valence and frozen core (with the frozen and
//! semicore moves), the bound references and the scattering references each channel
//! pseudizes. Atomic units.
//!
//! A channel pseudizes two partial waves: the occupied bound states (at most two: a semicore
//! channel holds sodium's 2s and 3s) and, when there is only one, a scattering wave
//! `energy_offset` above it (or at an absolute energy), normalized inside the sphere. An empty
//! channel above the valence (`extra_l`) has no bound state: both references are scattering
//! waves at the widest cutoff, anchored at the highest occupied level or at a positive energy.
use super::{
    recipe::{Cutoffs, Recipe},
    shells::Shells,
    tables_elements::{
        CUTOFF_GRID_FRACTION, FROZEN_SCATTERING_ENERGY, REFERENCE_MAX_ITERATIONS, REFERENCE_MIXING,
        REFERENCE_TOLERANCE,
    },
};
use crate::{
    models::{
        atom::{self, Atom, AtomModel, AtomOptions},
        relativity::solve_radial,
        relativity::Treatment,
        selection::relaxed_configuration_with,
        xc_scheme::{Variational, XcSource},
    },
    paw::{
        checks::peak_radius,
        invalid,
        pseudize::{bound_reference, scattering_reference, snap, Equation},
    },
    Context, Result,
};
use std::collections::BTreeMap;

/// Half-width (Hartree) of the window the Numerov shoot searches for a bound state in.
pub const BOUND_WINDOW: f64 = 5e-3;
/// Fallback cutoffs beyond this radius (Bohr) are reported: a smooth partial-wave channel is
/// normally trustworthy inside it.
pub const CUTOFF_WARN_RADIUS: f64 = 3.3;

/// A configuration as the atom solver takes it.
type Occupations = Vec<(usize, usize, f64)>;

/// The equation the all-electron waves of `recipe` obey. A Dirac dataset is the
/// scalar-relativistic construction plus the j-resolved spin-orbit term, so its channel waves
/// obey the scalar (j-averaged) equation; the two `j` branches have their own, [`branch_equation`].
pub fn equation(recipe: &Recipe) -> Equation {
    Equation {
        treatment: match recipe.relativity {
            Treatment::Dirac => Treatment::Scalar,
            other => other,
        },
        kappa: None,
        z: recipe.atomic_number as f64,
    }
}

/// The Dirac equation of the branch `kappa` of a channel.
pub fn branch_equation(recipe: &Recipe, kappa: i32) -> Equation {
    Equation {
        treatment: Treatment::Dirac,
        kappa: Some(kappa),
        z: recipe.atomic_number as f64,
    }
}

/// Solve the reference atom of `recipe`: the relaxed configuration with the recipe's occupancy
/// overrides laid over it, at the generator's tolerance and mixing, with the variational radial
/// exchange-correlation scheme (the scheme the dataset is unscreened with, D-2b).
pub fn solve_reference(ctx: &Context, recipe: &Recipe) -> Result<Atom> {
    solve_reference_with(ctx, recipe, &Variational)
}

#[doc(hidden)]
/// [`solve_reference`] with the radial exchange-correlation scheme as a parameter: the legacy
/// `AtomSolver` reproduces the atom the shipped PBE datasets were generated on (the parity seam;
/// the scheme also decides the relaxed configuration of an f element).
pub fn solve_reference_with(ctx: &Context, recipe: &Recipe, source: &dyn XcSource) -> Result<Atom> {
    recipe.validate()?;
    let z = recipe.atomic_number;
    let configuration = if recipe.reference_configuration.is_empty() {
        None
    } else {
        let mut occupations: Occupations = relaxed_configuration_with(
            ctx,
            z,
            &recipe.functional,
            recipe.relativity,
            recipe.r_max,
            Some(recipe.points),
            source,
        )?;
        for &((n, l), q) in &recipe.reference_configuration {
            match occupations.iter_mut().find(|s| (s.0, s.1) == (n, l)) {
                Some(state) => state.2 = q,
                None => occupations.push((n, l, q)),
            }
        }
        // The order is part of the answer: an f-element's SCF path depends on the order the
        // subshells are summed in (a replaced occupancy keeps its place, a new subshell goes last).
        occupations.retain(|s| s.2 > 0.);
        Some(occupations)
    };
    atom::solve_model_with(
        ctx,
        z,
        &AtomOptions {
            points: recipe.points,
            r_max: recipe.r_max,
            max_iterations: REFERENCE_MAX_ITERATIONS,
            tolerance: REFERENCE_TOLERANCE,
            mixing: REFERENCE_MIXING,
        },
        &AtomModel {
            functional: recipe.functional.clone(),
            relativity: recipe.relativity,
            polish: atom::POLISH_ROUNDS,
        },
        configuration.as_deref(),
        None,
        source,
    )
}

/// Refuse an atom that did not converge, or that was solved differently from the dataset being
/// asked for: an unconverged atom is not a reference, and the generator records the functional
/// and the treatment from the recipe (a dataset unscreened with another functional has the wrong
/// ionic potential).
pub fn check_atom(atom: &Atom, recipe: &Recipe) -> Result<()> {
    if !atom.converged {
        return Err(invalid(format!(
            "the all-electron reference atom Z={} (xc={:?}, relativity={:?}) did not converge in {} iterations; a dataset built on it would pseudize the wrong orbitals",
            recipe.atomic_number,
            atom.functional,
            atom.relativity.name(),
            atom.iterations
        )));
    }
    if atom.functional != recipe.functional || atom.relativity != recipe.relativity {
        return Err(invalid(format!(
            "the atom passed in was solved with xc={:?}, relativity={:?}, but the dataset asks for xc={:?}, relativity={:?}; pass matching options or let the generator solve the atom",
            atom.functional,
            atom.relativity.name(),
            recipe.functional,
            recipe.relativity.name()
        )));
    }
    Ok(())
}

/// One occupied bound state of a channel.
#[derive(Debug, Clone, Copy)]
pub struct BoundState {
    pub n: usize,
    pub energy: f64,
    pub occupancy: f64,
}

/// The all-electron input of one channel.
#[derive(Debug)]
pub struct ChannelReference {
    pub l: usize,
    /// The occupied bound states; an empty channel carries one state of zero occupancy.
    pub states: Vec<BoundState>,
    /// The augmentation radius, a node of the grid.
    pub r_cut: f64,
    /// The energies of the (at most two) references.
    pub energies: Vec<f64>,
    /// `R(r)` of the references, `energies.len() x len(r)` flat.
    pub waves: Vec<f64>,
}

impl ChannelReference {
    /// Principal quantum number of the first reference.
    pub fn n(&self) -> usize {
        self.states[0].n
    }

    /// Electrons of each bound state.
    pub fn occupations(&self) -> Vec<f64> {
        self.states.iter().map(|s| s.occupancy).collect()
    }
}

/// The augmentation radius of channel `l`: explicit, tabulated, else `rc_factor` times the
/// outermost peak of its first reference. The fallback is bounded and the bound raises rather
/// than clamps: the cutoff sets where the smooth wave stops matching the all-electron one, so a
/// value silently moved to fit the grid produces a dataset that looks fine and is not.
fn cutoff_for(
    recipe: &Recipe,
    table: &BTreeMap<usize, f64>,
    r: &[f64],
    l: usize,
    radial: &[f64],
    energy: f64,
    notes: &mut Vec<String>,
) -> Result<f64> {
    match &recipe.cutoffs {
        Cutoffs::Uniform(radius) => return Ok(*radius),
        Cutoffs::PerChannel(radii) => {
            return radii
                .get(&l)
                .copied()
                .ok_or_else(|| invalid(format!("r_cut names no radius for the channel l={l}")))
        }
        Cutoffs::Rule => {}
    }
    if let Some(radius) = table.get(&l) {
        return Ok(*radius);
    }
    let peak = peak_radius(r, radial)?;
    let candidate = recipe.rc_factor * peak;
    let end = r[r.len() - 1];
    let detail = format!(
        "its reference state peaks at {peak:.3} Bohr with eps = {energy:.5} Ha on a grid reaching {end:.3} Bohr"
    );
    if candidate > CUTOFF_GRID_FRACTION * end {
        return Err(invalid(format!(
            "{} l={l}: the fallback cutoff {candidate:.3} Bohr exceeds {CUTOFF_GRID_FRACTION} of the radial grid, because {detail}. A reference state peaking that far out is bound by the box rather than by the atom, and no cutoff fixes that: change the reference configuration, or enlarge r_max. Pass an explicit r_cut to override.",
            recipe.symbol
        )));
    }
    if candidate > CUTOFF_WARN_RADIUS {
        notes.push(format!(
            "{} l={l}: the fallback cutoff is {candidate:.3} Bohr, beyond the {CUTOFF_WARN_RADIUS} Bohr where a smooth partial-wave channel is normally trustworthy ({detail})",
            recipe.symbol
        ));
    }
    Ok(candidate)
}

/// The all-electron references of every channel: the occupied valence states, a scattering wave
/// when a channel has one, and `extra_l` empty channels above the highest valence one.
/// `extra_energy` fixes the first reference of the empty channels (default: the highest occupied
/// level). The second return value collects the notes a caller may surface.
pub fn reference_waves(
    ctx: &Context,
    recipe: &Recipe,
    atom: &Atom,
    shells: &Shells,
    table: &BTreeMap<usize, f64>,
    extra_l: usize,
    extra_energy: Option<f64>,
) -> Result<(BTreeMap<usize, ChannelReference>, Vec<String>)> {
    let eq = equation(recipe);
    reference_waves_with(
        ctx,
        recipe,
        atom,
        shells,
        table,
        extra_l,
        extra_energy,
        &|_| eq,
    )
}

/// The level the Numerov shoot of the bound state `(n, l)` starts from: the atom's own
/// eigenvalue, or for a Dirac branch the level of the `(n, l, kappa)` state.
fn bound_guess(atom: &Atom, at: usize, n: usize, l: usize, eq: &Equation) -> f64 {
    if eq.treatment == Treatment::Dirac {
        if let Some(j) = atom
            .j_states
            .iter()
            .find(|j| (j.n, j.l) == (n, l) && Some(j.kappa) == eq.kappa)
        {
            return j.eigenvalue;
        }
    }
    atom.eigenvalues[at]
}

/// [`reference_waves`] with the radial equation of each channel chosen by `equations(l)`: the
/// scalar one for the channels of the dataset, a Dirac branch for the spin-orbit construction.
#[allow(clippy::too_many_arguments)]
pub fn reference_waves_with(
    ctx: &Context,
    recipe: &Recipe,
    atom: &Atom,
    shells: &Shells,
    table: &BTreeMap<usize, f64>,
    extra_l: usize,
    extra_energy: Option<f64>,
    equations: &dyn Fn(usize) -> Equation,
) -> Result<(BTreeMap<usize, ChannelReference>, Vec<String>)> {
    let r = atom.r.as_slice();
    let v_ae = atom.effective.as_slice();
    let n_points = r.len();
    let mut notes = Vec::new();
    let mut per_l: BTreeMap<usize, Vec<(BoundState, Vec<f64>)>> = BTreeMap::new();
    let mut ordered = shells.valence.clone();
    ordered.sort_by_key(|s| s.0);
    for &((n, l), occupancy) in &ordered {
        let at = atom
            .occupations
            .iter()
            .position(|s| (s.0, s.1) == (n, l))
            .ok_or_else(|| invalid("a valence subshell is missing from the reference atom"))?;
        let eq = equations(l);
        let (wave, energy) = bound_reference(
            ctx,
            r,
            v_ae,
            l,
            bound_guess(atom, at, n, l, &eq),
            &eq,
            Some(BOUND_WINDOW),
        )?;
        // A channel's first reference is a statement about a bound state, so a non-negative
        // energy is not a hard case to handle: it is the wrong input (Ac's 5f under strict
        // aufbau comes out at +0.0065 Ha, a box state with an atomic label).
        if !(energy < 0.) {
            return Err(invalid(format!(
                "{}: the {n}{} reference state is not bound (eps = {energy:+.5} Ha). A channel cannot be built on an unbound reference state; the reference configuration is what needs changing, not the cutoff or the grid.",
                recipe.symbol,
                "spdfghik".chars().nth(l).unwrap_or('?')
            )));
        }
        per_l.entry(l).or_default().push((
            BoundState {
                n,
                energy,
                occupancy,
            },
            wave,
        ));
    }
    let mut channels = BTreeMap::new();
    for (&l, states) in &per_l {
        if states.len() > 2 {
            return Err(invalid(format!(
                "the channel l={l} holds {} occupied references; a PAW channel holds at most two bound ones",
                states.len()
            )));
        }
        let first = &states[0];
        let radius = snap(
            r,
            cutoff_for(recipe, table, r, l, &first.1, first.0.energy, &mut notes)?,
        );
        let mut energies = states.iter().map(|s| s.0.energy).collect::<Vec<_>>();
        let mut waves = Vec::with_capacity(2 * n_points);
        for state in states {
            waves.extend_from_slice(&state.1);
        }
        if states.len() == 1 {
            let second = match recipe.second_reference.get(&l) {
                Some(&energy) if !(energy > energies[0]) => {
                    return Err(invalid(format!(
                        "{} l={l}: the second reference energy {energy:+.4} Ha is not above the bound state ({:+.4} Ha)",
                        recipe.symbol, energies[0]
                    )))
                }
                Some(&energy) => energy,
                None => energies[0] + recipe.energy_offset,
            };
            waves.extend(scattering_reference(
                ctx,
                r,
                v_ae,
                l,
                second,
                &equations(l),
                radius,
            )?);
            energies.push(second);
        }
        channels.insert(
            l,
            ChannelReference {
                l,
                states: states.iter().map(|s| s.0).collect(),
                r_cut: radius,
                energies,
                waves,
            },
        );
    }
    if extra_l > 0 {
        let highest = *channels
            .keys()
            .next_back()
            .expect("a valence channel exists");
        let anchor = extra_energy.unwrap_or_else(|| {
            channels
                .values()
                .flat_map(|c| c.states.iter().map(|s| s.energy))
                .fold(f64::NEG_INFINITY, f64::max)
        });
        let widest = channels
            .values()
            .map(|c| c.r_cut)
            .fold(f64::NEG_INFINITY, f64::max);
        for l in highest + 1..=highest + extra_l {
            let energies = vec![anchor, anchor + recipe.energy_offset];
            let mut waves = Vec::with_capacity(2 * n_points);
            for &energy in &energies {
                waves.extend(scattering_reference(
                    ctx,
                    r,
                    v_ae,
                    l,
                    energy,
                    &equations(l),
                    widest,
                )?);
            }
            channels.insert(
                l,
                ChannelReference {
                    l,
                    states: vec![BoundState {
                        n: l + 1,
                        energy: energies[0],
                        occupancy: 0.,
                    }],
                    r_cut: widest,
                    energies,
                    waves,
                },
            );
        }
    }
    Ok((channels, notes))
}

/// The all-electron atom's lowest level of the empty angular momentum `l` above its `n_core`
/// core shells of that `l` (lithium's 2p at -0.042 Ha), or `None` when `l` binds nothing.
pub fn unoccupied_level(
    ctx: &Context,
    atom: &Atom,
    l: usize,
    n_core: usize,
) -> Result<Option<f64>> {
    let (_u, level) = solve_radial(
        ctx,
        atom.r.as_slice(),
        atom.effective.as_slice(),
        l,
        n_core,
        Treatment::None,
        None,
        0.,
        None,
        None,
    )?;
    Ok((level < 0.).then_some(level))
}

/// The first reference energies the empty channels try, in order: a frozen shell's scattering
/// energy as given; otherwise the highest occupied level (`None`), the channel's own bound
/// level, then the fallback scattering energies.
pub fn anchors(
    ctx: &Context,
    atom: &Atom,
    shells: &Shells,
    extra_l: usize,
) -> Result<Vec<Option<f64>>> {
    if extra_l == 0 {
        return Ok(vec![None]);
    }
    if !shells.frozen.is_empty() {
        return Ok(vec![Some(FROZEN_SCATTERING_ENERGY)]);
    }
    let highest = shells.valence.iter().map(|s| s.0 .1).max().unwrap_or(0);
    let n_core = shells.core.iter().filter(|s| s.0 .1 == highest + 1).count();
    let mut out = vec![None];
    if let Some(level) = unoccupied_level(ctx, atom, highest + 1, n_core)? {
        out.push(Some(level));
    }
    out.extend(
        super::tables_elements::EXTRA_ANCHOR_ENERGIES
            .iter()
            .map(|&e| Some(e)),
    );
    Ok(out)
}
