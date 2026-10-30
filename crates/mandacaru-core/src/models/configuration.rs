//! Ground-state and trial electron configurations of a spherical reference atom.
//!
//! A configuration is `{(n, l): occupation}` kept as a sorted list of `(n, l, q)`. The
//! aufbau (Madelung) order is a rule of thumb that is wrong about exactly one electron where
//! `(n-1)d` or `(n-2)f` crosses `ns`; `rearrangements` lists the aufbau filling followed by
//! every one-electron move across those crossings, and the atom solver decides which one the
//! functional prefers. No table of experimental configurations enters this module.
use super::{elements::AUFBAU, invalid, relativity};
use crate::Result;

pub type Subshell = (usize, usize);
pub type Configuration = Vec<(usize, usize, f64)>;

/// `l` values the aufbau order interleaves and so can order wrongly (`p` never is).
const COMPETING_L: [usize; 3] = [0, 2, 3];

fn capacity(l: usize) -> f64 {
    (2 * (2 * l + 1)) as f64
}

/// Neutral-atom aufbau filling, sorted by `(n, l)`.
pub fn ground_state(z: u8) -> Result<Configuration> {
    if z == 0 {
        return Err(invalid("atomic_number must be >= 1"));
    }
    let mut left = z as f64;
    let mut out = Vec::new();
    for (n, l) in AUFBAU {
        if left <= 0. {
            break;
        }
        let occupied = left.min(capacity(l));
        left -= occupied;
        out.push((n, l, occupied));
    }
    if left > 0. {
        return Err(invalid(format!(
            "atomic number {z} is beyond the supported filling table"
        )));
    }
    out.sort_by_key(|s| (s.0, s.1));
    Ok(out)
}

fn occupation(configuration: &[(usize, usize, f64)], state: Subshell) -> f64 {
    configuration
        .iter()
        .find(|s| (s.0, s.1) == state)
        .map_or(0., |s| s.2)
}

/// Outermost occupied `(n, l)` subshells plus the semicore `(n-1)d` and `(n-2)f` when
/// occupied: the minimal valence set (Fe -> 3d, 4s; Ce -> 4f, 5d, 6s).
pub fn valence_subshells(configuration: &[(usize, usize, f64)]) -> Vec<Subshell> {
    let occupied = configuration
        .iter()
        .filter(|s| s.2 > 0.)
        .map(|s| (s.0, s.1))
        .collect::<Vec<_>>();
    let Some(n_max) = occupied.iter().map(|s| s.0).max() else {
        return Vec::new();
    };
    let mut valence = occupied
        .iter()
        .copied()
        .filter(|s| s.0 == n_max)
        .collect::<Vec<_>>();
    for (l, offset) in [(2usize, 1usize), (3, 2)] {
        if n_max >= offset {
            let state = (n_max - offset, l);
            if occupied.contains(&state) && !valence.contains(&state) {
                valence.push(state);
            }
        }
    }
    valence.sort_unstable();
    valence
}

/// The `count` lowest unoccupied subshells in aufbau order; a partly filled one is occupied.
fn unoccupied(z: u8, count: usize) -> Result<Vec<Subshell>> {
    let base = ground_state(z)?;
    let empty = AUFBAU
        .iter()
        .copied()
        .filter(|s| occupation(&base, *s) <= 0.)
        .collect::<Vec<_>>();
    if empty.len() < count {
        return Err(invalid(format!(
            "element Z={z} has only {} unoccupied subshell(s) below the end of the filling table",
            empty.len()
        )));
    }
    Ok(empty[..count].to_vec())
}

/// The aufbau configuration first, then each one-electron rearrangement of its valence.
///
/// A move takes one electron from an occupied valence subshell to another valence subshell
/// or to one of the `virtuals` lowest unoccupied ones, never emptying a destination above
/// its capacity and never entering a principal shell above the outermost occupied one. Both
/// subshells must have different `l` drawn from `{s, d, f}`.
pub fn rearrangements(z: u8, virtuals: usize) -> Result<Vec<Configuration>> {
    let base = ground_state(z)?;
    let sources = valence_subshells(&base)
        .into_iter()
        .filter(|s| occupation(&base, *s) > 0.)
        .collect::<Vec<_>>();
    let mut empty = Vec::new();
    for count in (1..=virtuals).rev() {
        if let Ok(found) = unoccupied(z, count) {
            empty = found;
            break;
        }
    }
    let n_outer = base.iter().map(|s| s.0).max().unwrap_or(0);
    let mut targets: Vec<Subshell> = Vec::new();
    for state in sources.iter().chain(&empty) {
        if state.0 <= n_outer && !targets.contains(state) {
            targets.push(*state);
        }
    }
    let mut out = vec![base.clone()];
    for src in &sources {
        if !COMPETING_L.contains(&src.1) {
            continue;
        }
        for dst in &targets {
            if !COMPETING_L.contains(&dst.1)
                || dst.1 == src.1
                || occupation(&base, *dst) >= capacity(dst.1)
            {
                continue;
            }
            let mut moved = base.clone();
            for s in moved.iter_mut() {
                if (s.0, s.1) == *src {
                    s.2 -= 1.;
                }
            }
            match moved.iter_mut().find(|s| (s.0, s.1) == *dst) {
                Some(s) => s.2 += 1.,
                None => moved.push((dst.0, dst.1, 1.)),
            }
            moved.retain(|s| s.2 > 0.);
            moved.sort_by_key(|s| (s.0, s.1));
            out.push(moved);
        }
    }
    Ok(out)
}

/// Resolve `{(n, l): q}` into `{(n, l, kappa): q}` for a Dirac atom: a subshell is shared
/// between its two `j` in proportion to `2j+1`, the lower `j = l - 1/2` filling first.
pub fn split_configuration(configuration: &[(usize, usize, f64)]) -> Vec<(usize, usize, i32, f64)> {
    let mut out = Vec::new();
    for &(n, l, q) in configuration {
        let mut left = q;
        for kappa in relativity::kappa_values(l) {
            let room = relativity::degeneracy(kappa) as f64;
            out.push((n, l, kappa, left.min(room)));
            left = (left - room).max(0.);
        }
    }
    out
}
