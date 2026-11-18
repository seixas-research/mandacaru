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

/// How an open subshell of a Dirac atom divides its electrons between its two `j`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JShares {
    /// `j-j` filling: the lower `j = l - 1/2` fills first (the ground state of the atom).
    LowerFirst,
    /// In proportion to `2j + 1`: every state of the subshell holds the same fraction of an
    /// electron, so the density is the spherical one the scalar partial waves of a PAW dataset
    /// are built for. A closed subshell is the same in both.
    Proportional,
}

/// Resolve `{(n, l): q}` into `{(n, l, kappa): q}` for a Dirac atom, the subshell shared between
/// its two `j` as `shares` says.
pub fn split_configuration(
    configuration: &[(usize, usize, f64)],
    shares: JShares,
) -> Vec<(usize, usize, i32, f64)> {
    let mut out = Vec::new();
    for &(n, l, q) in configuration {
        let mut left = q;
        for kappa in relativity::kappa_values(l) {
            let room = relativity::degeneracy(kappa) as f64;
            match shares {
                JShares::LowerFirst => {
                    out.push((n, l, kappa, left.min(room)));
                    left = (left - room).max(0.);
                }
                JShares::Proportional => {
                    out.push((n, l, kappa, q * room / (2. * (2 * l + 1) as f64)))
                }
            }
        }
    }
    out
}

/// Build a neutral spherically symmetric proatomic radial density $\rho^0_A(r)$ for Hirshfeld
/// partitioning from an atom's unique radial basis shells `(l, r, R(r))`, normalized so
/// $\int_0^\infty 4\pi r^2 \rho^0_A(r)\,dr = \text{valence\_charge}$.
pub fn proatomic_radial_from_shells(
    atomic_number: u8,
    valence_charge: f64,
    shells: &[(usize, &[f64], &[f64])],
) -> crate::math::partial_charges::ProatomicRadial {
    use std::f64::consts::PI;
    let fallback_scale = match atomic_number {
        1..=2 => 0.45,
        3..=10 => 0.65,
        11..=18 => 0.90,
        _ => 1.05,
    };
    let empty = crate::math::partial_charges::ProatomicRadial {
        r: Vec::new(),
        density: Vec::new(),
        valence_charge,
        fallback_scale,
    };
    if shells.is_empty() || valence_charge <= 0. {
        return empty;
    }
    let Ok(config) = ground_state(atomic_number) else {
        return empty;
    };
    let core_target = (atomic_number as f64 - valence_charge).max(0.);
    // Keep valence (and semicore if present in valence_charge) occupations matching `valence_charge`.
    let mut valence_shells: Vec<(usize, f64)> = Vec::new();
    let mut dropped = 0.;
    for &(_n, l, occ) in &config {
        if dropped + occ <= core_target + 1e-6 {
            dropped += occ;
        } else {
            let take = (occ - (core_target - dropped).max(0.)).max(0.);
            dropped = core_target;
            if take > 0. {
                valence_shells.push((l, take));
            }
        }
    }
    if valence_shells.is_empty() {
        valence_shells.push((0, valence_charge));
    }
    // Pair each valence shell `(l, occ)` with the first unused radial table of that `l` (or
    // closest radial shell available) and sum spherical densities on the first table's radial grid.
    let grid_r = shells[0].1;
    if grid_r.len() < 2 {
        return empty;
    }
    let integrate_r2 = |r: &[f64], f: &[f64]| -> f64 {
        (0..r.len() - 1)
            .map(|i| {
                0.5 * (r[i + 1] - r[i]) * (f[i] * r[i] * r[i] + f[i + 1] * r[i + 1] * r[i + 1])
            })
            .sum::<f64>()
    };
    let sample_linear = |src_r: &[f64], src_v: &[f64], x: f64| -> f64 {
        if src_r.is_empty() || x > src_r[src_r.len() - 1] {
            return 0.;
        }
        if x <= src_r[0] {
            return src_v[0];
        }
        let idx = src_r.partition_point(|&rk| rk <= x);
        let i0 = idx.saturating_sub(1).min(src_r.len() - 2);
        let span = src_r[i0 + 1] - src_r[i0];
        let t = if span > 0. {
            (x - src_r[i0]) / span
        } else {
            0.
        };
        (1. - t) * src_v[i0] + t * src_v[i0 + 1]
    };
    let mut used = vec![false; shells.len()];
    let mut rho = vec![0.; grid_r.len()];
    let mut matched = 0.;
    for (l, occ) in valence_shells {
        if let Some(pos) = shells
            .iter()
            .enumerate()
            .position(|(i, s)| !used[i] && s.0 == l && s.1.len() >= 2 && s.1.len() == s.2.len())
        {
            used[pos] = true;
            let (_, sr, sv) = shells[pos];
            let sq: Vec<f64> = sv.iter().map(|&v| v * v).collect();
            let norm = integrate_r2(sr, &sq);
            if norm > 1e-12 {
                let factor = occ / (4. * PI * norm);
                for (k, &rk) in grid_r.iter().enumerate() {
                    let v = sample_linear(sr, sv, rk);
                    rho[k] += factor * v * v;
                }
                matched += occ;
            }
        }
    }
    if matched <= 0. {
        return empty;
    }
    let integral = 4. * PI * integrate_r2(grid_r, &rho);
    if integral > 1e-12 {
        let scale = valence_charge / integral;
        for v in &mut rho {
            *v *= scale;
        }
    }
    crate::math::partial_charges::ProatomicRadial {
        r: grid_r.to_vec(),
        density: rho,
        valence_charge,
        fallback_scale,
    }
}
