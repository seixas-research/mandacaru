//! Which configuration a reference atom is solved in, and whether its valence is bound.
//!
//! **The aufbau filling is used unless it produces an unbound valence state.** That is the
//! whole rule, deliberately narrower than "whichever configuration has the lowest energy":
//! the margins by which LDA prefers chromium's `3d^5 4s^1` shrink toward zero with the grid,
//! so acting on them would be acting on noise, and this module declines to reproduce the Cr
//! and Cu anomalies. What it repairs is sharp: under strict aufbau the `f` shell of La, Ac,
//! Th and Pa can be a box state, not a valence orbital, and a pseudopotential channel built
//! from one is meaningless. Moving one electron to the `d` shell binds it.
//!
//! Only an occupied `f` shell in the valence can be the unbound one, so every other atom
//! returns immediately with no solve at all.
use super::{
    atom::{self, Atom, AtomModel, AtomOptions},
    configuration::{self, valence_subshells, Configuration},
    relativity::Treatment,
};
use crate::{Context, Error, Result};

/// A valence eigenvalue at or above this (Hartree) is not a bound state.
pub const BOUND_STATE_CEILING: f64 = -1e-3;
/// Fraction of the radial box beyond which a valence orbital's maximum means the box, not the
/// atom, holds the state.
pub const BOX_STATE_FRACTION: f64 = 0.5;

/// Uniform radial points a configuration is judged on. Not a free parameter: the floor scales
/// with `Z` because a bound-state test on a coarse grid reads discretization error (cerium's
/// aufbau `4f` and `6s` come out unbound at 6000 points and bound at 12000).
pub fn selection_points(z: u8) -> usize {
    12000usize.max(300 * z as usize)
}

fn peak_radius(r: &[f64], u: &[f64]) -> f64 {
    let mut best = 0;
    for (i, v) in u.iter().enumerate() {
        if v.abs() > u[best].abs() {
            best = i;
        }
    }
    r[best]
}

/// Is every valence orbital of `atom` bound by the atom? Two ways to fail: an eigenvalue at or
/// above the ceiling, or a tiny negative one whose maximum sits against the box wall. A Dirac
/// atom is tested level by level, since the `(2j+1)` average can be bound while a branch is not.
pub fn bound_valence(atom: &Atom) -> bool {
    let valence = valence_subshells(&atom.occupations);
    let r = atom.r.as_slice();
    let edge = BOX_STATE_FRACTION * r[r.len() - 1];
    for state in &valence {
        let Some(index) = atom.occupations.iter().position(|s| (s.0, s.1) == *state) else {
            return false;
        };
        if atom.eigenvalues[index] > BOUND_STATE_CEILING
            || peak_radius(r, atom.orbitals[index].as_slice()) > edge
        {
            return false;
        }
    }
    for level in &atom.j_states {
        if !valence.contains(&(level.n, level.l)) {
            continue;
        }
        if level.eigenvalue > BOUND_STATE_CEILING || peak_radius(r, level.orbital.as_slice()) > edge
        {
            return false;
        }
    }
    true
}

/// The reference configuration to solve this atom in: aufbau unless it leaves the valence
/// unbound, then the lowest-energy bound one-electron rearrangement. `points` defaults to
/// `selection_points`. When nothing is bound the aufbau filling is kept, so a generator
/// refuses it by name rather than by a silently substituted configuration.
pub fn relaxed_configuration(
    ctx: &Context,
    z: u8,
    functional: &str,
    relativity: Treatment,
    r_max: f64,
    points: Option<usize>,
) -> Result<Configuration> {
    let candidates = configuration::rearrangements(z, 3)?;
    let aufbau = candidates[0].clone();
    if candidates.len() == 1
        || !valence_subshells(&aufbau)
            .iter()
            .any(|&(n, l)| l == 3 && aufbau.iter().any(|s| (s.0, s.1) == (n, l) && s.2 > 0.))
    {
        return Ok(aufbau);
    }
    let options = AtomOptions {
        points: points.unwrap_or_else(|| selection_points(z)),
        r_max,
        max_iterations: 200,
        tolerance: 1e-6,
        mixing: 0.3,
    };
    let model = AtomModel {
        functional: functional.to_string(),
        relativity,
        polish: atom::POLISH_ROUNDS,
    };
    // A candidate the solver cannot follow is skipped; a refused option or an exhausted
    // budget is not a candidate failing, and propagates.
    let solved = |candidate: &Configuration| -> Result<Option<Atom>> {
        match atom::solve_model(ctx, z, &options, &model, Some(candidate), None) {
            Ok(atom) => Ok(Some(atom)),
            Err(Error::Kernel(_)) => Ok(None),
            Err(error) => Err(error),
        }
    };
    if let Some(atom) = solved(&aufbau)? {
        if atom.converged && bound_valence(&atom) {
            return Ok(aufbau);
        }
    }
    let mut best: Option<(&Configuration, f64)> = None;
    for candidate in &candidates[1..] {
        if let Some(trial) = solved(candidate)? {
            if trial.converged
                && bound_valence(&trial)
                && best.is_none_or(|(_, energy)| trial.energy < energy)
            {
                best = Some((candidate, trial.energy));
            }
        }
    }
    Ok(best.map_or(aufbau, |(candidate, _)| candidate.clone()))
}
