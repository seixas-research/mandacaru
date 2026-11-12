//! The occupation-weighted lattice field of a metal (model-free): on every plaquette the bands up
//! to a cap are cut into groups, and each group contributes its occupation times its
//! Fukui-Hatsugai-Suzuki flux. `dev_plan/BERRY_PLAN.md` section 9 holds the derivation.
//!
//! - A cut between bands `n - 1` and `n` is allowed where they are more than
//!   [`SEPARATION`](super::SEPARATION) apart at all four corners; touching or degenerate bands
//!   share a group.
//! - Consecutive groups full at every corner merge into one sea group (so an insulator gives
//!   exactly [`links::lattice_field`]); groups empty at every corner are skipped.
//! - A group whose sub-block link has a singular value below [`MERGE`] (two bands swapped order
//!   inside the plaquette), or whose flux exceeds [`links::ALIASING`] (a point node near the
//!   plaquette center, which keeps every link large and carries pi at any mesh), merges with its
//!   neighbor; only a group spanning every linked band can still alias, and that raises.
//! - A group's weight is the mean occupation of its members over the four corners.
use super::{links, SEPARATION};
use crate::{periodic::invalid, quantum::C64, Result};
use nalgebra::DMatrix;

type Matrix = DMatrix<C64>;

/// A band holding less than this fraction of a full band counts as empty, more than one minus it
/// as full.
pub const EMPTY: f64 = 1e-6;
/// A group's sub-block link with a smaller singular value holds a crossing: merge.
pub const MERGE: f64 = 0.5;

/// The weighted flux of a periodic plane.
pub struct WeightedField {
    /// The plaquettes of the plane.
    pub plaquettes: usize,
    /// `flux[i * n_b + j]`: `sum_g w_g phi_g` of the plaquette at `(i, j)` (electrons times
    /// radians).
    pub flux: Vec<f64>,
    /// `(1/2 pi) sum flux`: the filling-weighted plane number (`f C` for an insulator).
    pub number: f64,
    /// The largest `|phi_g|` of any group.
    pub max_flux: f64,
    /// Plaquettes where a crossing or an aliased flux merged two groups.
    pub merged: usize,
}

/// The dimensionless smearing argument beyond which `occupation` stays below [`EMPTY`].
pub fn occupation_window(occupation: impl Fn(f64) -> f64) -> f64 {
    let mut last = 0.;
    let mut x = 0.;
    while x < 200. {
        if occupation(x).abs() > EMPTY {
            last = x;
        }
        x += 0.05;
    }
    last + 0.05
}

#[derive(Clone, Copy)]
struct Group {
    lo: usize,
    hi: usize,
}

/// The plaquette fluxes of the plane `size = (n_a, n_b)`: `levels[p]` and `occupations[p]` (at
/// least `cap` values, occupations in electrons with `full` per band) and the `cap x cap` links
/// `along_a[p]`, `along_b[p]` from point `p = i * n_b + j` to its next point along `a` and `b`.
pub fn weighted_field(
    levels: &[Vec<f64>],
    occupations: &[Vec<f64>],
    along_a: &[Matrix],
    along_b: &[Matrix],
    size: [usize; 2],
    full: f64,
) -> Result<WeightedField> {
    let [na, nb] = size;
    let count = na
        .checked_mul(nb)
        .ok_or_else(|| invalid("the k plane is too large"))?;
    let cap = along_a.first().map_or(0, |m| m.nrows());
    if na < 2
        || nb < 2
        || cap == 0
        || !(full > 0.)
        || [
            levels.len(),
            occupations.len(),
            along_a.len(),
            along_b.len(),
        ]
        .iter()
        .any(|n| *n != count)
        || along_a
            .iter()
            .chain(along_b)
            .any(|m| m.nrows() != cap || m.ncols() != cap)
        || levels
            .iter()
            .chain(occupations)
            .any(|v| v.len() < cap || v.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid(
            "a weighted lattice field needs levels, occupations and cap x cap links at every \
             point of at least a 2 x 2 plane",
        ));
    }
    let at = |i: usize, j: usize| (i % na) * nb + j % nb;
    let mut out = WeightedField {
        plaquettes: count,
        flux: Vec::with_capacity(count),
        number: 0.,
        max_flux: 0.,
        merged: 0,
    };
    for i in 0..na {
        for j in 0..nb {
            let corners = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
            // The four edges: a from corner 0, b from corner 1, a from corner 3, b from corner 0.
            let edges = [
                &along_a[corners[0]],
                &along_b[corners[1]],
                &along_a[corners[3]],
                &along_b[corners[0]],
            ];
            let full_band = |n: usize| {
                corners
                    .iter()
                    .all(|c| occupations[*c][n] >= (1. - EMPTY) * full)
            };
            let empty_band = |n: usize| corners.iter().all(|c| occupations[*c][n] <= EMPTY * full);
            let cut = |n: usize| {
                corners
                    .iter()
                    // Past the last band of the basis nothing can touch.
                    .all(|c| {
                        levels[*c]
                            .get(n)
                            .is_none_or(|e| e - levels[*c][n - 1] > SEPARATION)
                    })
            };
            // Groups between allowed cuts; full neighbors merge into the sea.
            let mut groups: Vec<Group> = vec![];
            let mut lo = 0;
            for n in 1..=cap {
                if n == cap || cut(n) {
                    let group = Group { lo, hi: n };
                    let sea = |g: &Group| (g.lo..g.hi).all(full_band);
                    match groups.last_mut() {
                        Some(last) if sea(last) && sea(&group) => last.hi = n,
                        _ => groups.push(group),
                    }
                    lo = n;
                }
            }
            let empty = |g: &Group| (g.lo..g.hi).all(empty_band);
            if let Some(top) = groups.last() {
                if !empty(top) && top.hi == cap && !cut(cap) {
                    return Err(invalid(
                        "an occupied band group reaches the band cap of the weighted field without \
                         a gap above it: more bands are needed",
                    ));
                }
            }
            // Merge a group with its neighbor where its sub-block links reveal a crossing inside
            // the plaquette, or where its flux aliases (a point node near the plaquette center:
            // the links stay large and the flux is pi, with no mesh at which it resolves).
            let mut merged_here = false;
            let fluxes = loop {
                let mut fluxes: Vec<Option<f64>> = Vec::with_capacity(groups.len());
                let mut merge = None;
                for (g, group) in groups.iter().enumerate() {
                    if empty(group) {
                        fluxes.push(None);
                        continue;
                    }
                    let m = group.hi - group.lo;
                    let mut d = [C64::default(); 4];
                    let mut smallest = f64::INFINITY;
                    for (e, edge) in edges.iter().enumerate() {
                        let block = edge.view((group.lo, group.lo), (m, m)).into_owned();
                        let (unitary, s) = links::unitarize(&block)?;
                        let det = unitary.lu().determinant();
                        d[e] = det / det.norm();
                        smallest = smallest.min(s);
                    }
                    let phi = links::plaquette_flux(d[0], d[1], d[2], d[3]);
                    if (smallest < MERGE || phi.abs() > links::ALIASING) && groups.len() > 1 {
                        merge = Some(g);
                        break;
                    }
                    if !smallest.is_finite() || smallest < super::LINK_TOLERANCE {
                        return Err(invalid(
                            "a singular band-group link in the weighted lattice field: refine the \
                             k mesh",
                        ));
                    }
                    if phi.abs() > links::ALIASING {
                        return Err(invalid(format!(
                            "the band group of a Berry plaquette that spans every linked band \
                             carries a flux of {:.3} rad, near pi, where the lattice field \
                             aliases: refine the k mesh",
                            phi.abs()
                        )));
                    }
                    fluxes.push(Some(phi));
                }
                let Some(g) = merge else { break fluxes };
                merged_here = true;
                let other = if g + 1 < groups.len() { g + 1 } else { g - 1 };
                let (first, second) = (g.min(other), g.max(other));
                groups[first].hi = groups[second].hi;
                groups.remove(second);
            };
            if merged_here {
                out.merged += 1;
            }
            let mut flux = 0.;
            for (group, phi) in groups.iter().zip(&fluxes) {
                let Some(phi) = *phi else { continue };
                out.max_flux = out.max_flux.max(phi.abs());
                let members = (group.hi - group.lo) as f64;
                let weight = corners
                    .iter()
                    .map(|c| occupations[*c][group.lo..group.hi].iter().sum::<f64>())
                    .sum::<f64>()
                    / (4. * members);
                flux += weight * phi;
            }
            out.number += flux;
            out.flux.push(flux);
        }
    }
    out.number /= 2. * std::f64::consts::PI;
    Ok(out)
}
