//! Topological invariants of hybrid Wannier-center flows, model-free: nothing here knows whether a
//! set of centers came from a crystal's PAW links or from a lattice model.
//!
//! A flow is the sorted Wannier charge centers `x_n(k_perp)` in [0, 1) of the strings of one plane
//! ([`super::links::wannier_centers`]) as `k_perp` runs along the transverse direction.
//!
//! - [`winding`]: the Chern number is minus the total displacement of the centers over the period
//!   `k_perp in [0, 1]`, in units of the cell (an integer; the centers are `-theta / 2 pi`).
//! - [`z2`]: in a time-reversal-invariant plane, with `k_perp in [0, 1/2]` between the two
//!   time-reversal-invariant lines, the Z2 index is the parity of the number of times the centers
//!   cross a reference line (Soluyanov and Vanderbilt, PRB 83, 235401 (2011); Yu, Qi, Bernevig,
//!   Fang and Dai, PRB 84, 075119 (2011)). The line sits in the middle of the largest gap between
//!   consecutive centers (circularly, the wrap-around gap included) of each sample, so that no
//!   center is near it; when the largest gap changes from one sample to the next the line moves
//!   with the centers lying between its old and new place counted as crossings, which keeps the
//!   count that of one continuous line. A time-reversal-invariant set holds an even number of
//!   centers that come in Kramers pairs on the two end lines, so the parity does not depend on the
//!   line and on the arc along which it is moved.
//! - [`adaptive`]: the transverse sampling. A fixed mesh can step over a double crossing near a
//!   small gap and miscount silently, so intervals where a center moves by more than a quarter of
//!   the largest gap are bisected until they do not, and beyond a depth cap the flow raises.
use crate::{periodic::invalid, Result};
use std::collections::BTreeMap;

/// Two Kramers partners closer than this (a fraction of the cell) are one pair. Measured on bulk
/// Bi2Se3 (SZ, 78 centers, the plane `k_0 = 1/2`): a 6-point string leaves pair deviations of up to
/// 1.0e-6 at `k_perp = 0` (4.9e-7 on another SCF mesh) and a 10-point string 8e-12 on the same states,
/// with link singular values of 0.45 and 0.67: the discretization of the loop, not noise of the states
/// nor a badly conditioned link. 1e-4 leaves two orders of margin over 6-point strings; the splitting a
/// time-reversal-breaking term gives was not measured, so a very weak one passes the check.
pub const KRAMERS_TOLERANCE: f64 = 1e-4;
/// An interval is bisected where a center moves by more than this fraction of the largest gap.
pub const MOVE_FRACTION: f64 = 0.25;

/// The centers of every sample of a flow.
#[derive(Clone, Debug, PartialEq)]
pub struct Flow {
    /// The transverse coordinate of each sample, ascending (fractional).
    pub k: Vec<f64>,
    /// `[sample][n]`: sorted centers in [0, 1).
    pub centers: Vec<Vec<f64>>,
}

/// The index of the Z2 count of a flow and what it was counted from.
#[derive(Clone, Debug, PartialEq)]
pub struct Z2 {
    /// The invariant, 0 or 1.
    pub parity: u8,
    /// Centers that crossed the reference line, over every interval and every move of the line.
    pub crossings: usize,
    /// The reference line (middle of the largest gap) of each sample.
    pub lines: Vec<f64>,
    /// The largest gap of each sample.
    pub gaps: Vec<f64>,
}

/// `d` on (-1/2, 1/2].
fn circular(d: f64) -> f64 {
    d - d.round()
}

fn flow_check(flow: &Flow) -> Result<usize> {
    let n = flow.centers.first().map_or(0, Vec::len);
    if flow.k.len() != flow.centers.len()
        || flow.k.len() < 2
        || n == 0
        || flow.centers.iter().any(|c| c.len() != n)
        || flow
            .centers
            .iter()
            .flatten()
            .any(|x| !x.is_finite() || !(0. ..1.).contains(x))
        || flow.k.windows(2).any(|w| !(w[1] > w[0]))
    {
        return Err(invalid(
            "a center flow needs at least two ascending samples of the same nonzero number of \
             centers in [0, 1)",
        ));
    }
    Ok(n)
}

/// `(midpoint, size)` of the largest gap between consecutive sorted `centers` on the circle: the
/// first of equal gaps (so a tie is deterministic). One center has the whole circle as its gap.
pub fn largest_gap(centers: &[f64]) -> Result<(f64, f64)> {
    if centers.is_empty() {
        return Err(invalid("a set of centers cannot be empty"));
    }
    let n = centers.len();
    let mut best = (0, f64::NEG_INFINITY);
    for i in 0..n {
        let next = if i + 1 < n {
            centers[i + 1]
        } else {
            centers[0] + 1.
        };
        let gap = next - centers[i];
        if gap > best.1 + 1e-12 {
            best = (i, gap);
        }
    }
    let (i, gap) = best;
    Ok(((centers[i] + 0.5 * gap).rem_euclid(1.), gap))
}

/// The displacement of each center of `from` to its partner in `to` (on (-1/2, 1/2]): the pairing
/// of the two sorted sets that leaves the cyclic order alone and has the smallest total move.
pub fn displacements(from: &[f64], to: &[f64]) -> Result<Vec<f64>> {
    let n = from.len();
    if n == 0 || to.len() != n {
        return Err(invalid(
            "two center sets of one flow hold the same number of centers",
        ));
    }
    let cost = |shift: usize| -> f64 {
        (0..n)
            .map(|i| circular(to[(i + shift) % n] - from[i]).abs())
            .sum()
    };
    let shift = (0..n)
        .map(|s| (s, cost(s)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(s, _)| s);
    Ok((0..n)
        .map(|i| circular(to[(i + shift) % n] - from[i]))
        .collect())
}

/// Every center of the set has a partner within [`KRAMERS_TOLERANCE`] (a time-reversal-invariant
/// line): consecutive sorted centers pair up, starting at either of the two ways around the circle.
pub fn check_kramers(centers: &[f64], at: f64) -> Result<()> {
    let n = centers.len();
    if n == 0 || n % 2 == 1 {
        return Err(invalid(format!(
            "not time-reversal invariant: {n} Wannier centers at k_perp = {at} cannot form Kramers \
             pairs"
        )));
    }
    let paired = |offset: usize| {
        (0..n / 2).all(|j| {
            let (a, b) = ((offset + 2 * j) % n, (offset + 2 * j + 1) % n);
            circular(centers[b] - centers[a]).abs() <= KRAMERS_TOLERANCE
        })
    };
    if paired(0) || paired(1) {
        return Ok(());
    }
    Err(invalid(format!(
        "not time-reversal invariant: the Wannier centers {centers:?} at k_perp = {at} do not form \
         degenerate Kramers pairs (tolerance {KRAMERS_TOLERANCE})"
    )))
}

/// The Chern number: minus the total displacement of the centers over the period. The flow runs from
/// `k_perp = 0` to `1` with the centers of the last sample those of the first.
pub fn winding(flow: &Flow) -> Result<i64> {
    flow_check(flow)?;
    let (first, last) = (flow.k[0], flow.k[flow.k.len() - 1]);
    if first.abs() > 1e-9 || (last - 1.).abs() > 1e-9 {
        return Err(invalid(
            "a winding is taken over the whole period: k_perp from 0 to 1",
        ));
    }
    let n = flow.centers[0].len();
    let closes = flow.centers[0]
        .iter()
        .zip(&flow.centers[flow.centers.len() - 1])
        .all(|(a, b)| circular(a - b).abs() < 1e-9);
    if !closes {
        return Err(invalid(
            "the centers at k_perp = 1 are not those at k_perp = 0: the flow is not periodic",
        ));
    }
    let mut total = 0.;
    for w in flow.centers.windows(2) {
        let d = displacements(&w[0], &w[1])?;
        if d.iter().any(|v| v.abs() > 0.45) {
            return Err(invalid(
                "a center moves by almost half a cell between two samples: its direction is \
                 ambiguous, refine the transverse samples",
            ));
        }
        total += d.iter().sum::<f64>();
    }
    if (total - total.round()).abs() > 1e-6 * n as f64 {
        return Err(invalid(format!(
            "the centers' displacement over the period is {total}, not an integer: refine the \
             transverse samples"
        )));
    }
    // The centers are `-theta / 2 pi`, so the lattice Chern number of the plaquette fluxes is minus
    // their displacement (pinned against `links::lattice_field` on Qi-Wu-Zhang in the tests).
    Ok(-(total.round() as i64))
}

/// The Z2 index of a flow from `k_perp = 0` to `1/2`, both time-reversal-invariant lines
/// (Kramers pairs, checked there).
pub fn z2(flow: &Flow) -> Result<Z2> {
    flow_check(flow)?;
    let samples = flow.k.len();
    if flow.k[0].abs() > 1e-9 || (flow.k[samples - 1] - 0.5).abs() > 1e-9 {
        return Err(invalid(
            "a Z2 flow runs from k_perp = 0 to 1/2, the two time-reversal-invariant lines",
        ));
    }
    check_kramers(&flow.centers[0], flow.k[0])?;
    check_kramers(&flow.centers[samples - 1], flow.k[samples - 1])?;
    let (mut lines, mut gaps) = (Vec::with_capacity(samples), Vec::with_capacity(samples));
    for centers in &flow.centers {
        let (line, gap) = largest_gap(centers)?;
        lines.push(line);
        gaps.push(gap);
    }
    let mut crossings = 0;
    for i in 0..samples - 1 {
        let (from, to) = (&flow.centers[i], &flow.centers[i + 1]);
        let d = displacements(from, to)?;
        if d.iter().any(|v| v.abs() > 0.45) {
            return Err(invalid(
                "a center moves by almost half a cell between two samples: refine the transverse \
                 samples",
            ));
        }
        let line = lines[i];
        // The line is held while the centers move to the next sample, ...
        crossings += from
            .iter()
            .zip(&d)
            .filter(|(x, d)| {
                let end = (**x - line).rem_euclid(1.) + **d;
                !(0. ..1.).contains(&end)
            })
            .count();
        // ... then moved to the largest gap of that sample past the centers between the two places.
        // (The shorter way round: the other differs by the number of centers, which is even.)
        let reach = (lines[i + 1] - line).rem_euclid(1.);
        crossings += to
            .iter()
            .filter(|x| {
                let at = (**x - line).rem_euclid(1.);
                if reach <= 0.5 {
                    at < reach
                } else {
                    at >= reach
                }
            })
            .count();
    }
    Ok(Z2 {
        parity: (crossings % 2) as u8,
        crossings,
        lines,
        gaps,
    })
}

/// How the transverse samples of a flow are chosen.
#[derive(Clone, Copy, Debug)]
pub struct Adaptive {
    /// Equal intervals of the first pass.
    pub intervals: usize,
    /// An interval is halved at most this many times.
    pub depth: u32,
}

/// The flow from `k_perp = 0` to `end` (1/2 or 1) of `evaluate`, which returns the sorted centers
/// of the string at a `k_perp`. The first pass takes `settings.intervals` equal intervals; then
/// every interval is halved while a center moves by more than [`MOVE_FRACTION`] of the smaller of
/// the two largest gaps at its ends, and, up to two halvings only, while the largest gap changes
/// (the reference line of [`z2`] moves exactly across such a jump, so the jump needs no more
/// samples than its neighbors). An interval still moving too much after `settings.depth` halvings
/// raises, naming `label`. A flow to 1 is periodic: `evaluate` is not called there, and the centers
/// of the last sample are the first's.
pub fn adaptive(
    label: &str,
    end: f64,
    settings: Adaptive,
    mut evaluate: impl FnMut(f64) -> Result<Vec<f64>>,
) -> Result<Flow> {
    if (end - 0.5).abs() > 1e-12 && (end - 1.).abs() > 1e-12 {
        return Err(invalid("a flow runs to k_perp = 1/2 or 1"));
    }
    if settings.intervals == 0 || settings.depth > 20 {
        return Err(invalid(
            "a flow needs at least one interval and a depth of at most 20",
        ));
    }
    let unit: u64 = 1 << settings.depth;
    let total = settings.intervals as u64 * unit;
    let at = |position: u64| end * position as f64 / total as f64;
    let periodic = end > 0.75;
    let mut samples: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for i in 0..=settings.intervals as u64 {
        let position = i * unit;
        let centers = if periodic && i == settings.intervals as u64 {
            samples[&0].clone()
        } else {
            evaluate(at(position))?
        };
        samples.insert(position, centers);
    }
    let soft = unit >> settings.depth.min(2);
    loop {
        let mut refined = false;
        let positions: Vec<u64> = samples.keys().copied().collect();
        for pair in positions.windows(2) {
            let (lower, mut upper) = (pair[0], pair[1]);
            while upper - lower > 1 {
                let (a, b) = (&samples[&lower], &samples[&upper]);
                let ((line_a, gap_a), (line_b, gap_b)) = (largest_gap(a)?, largest_gap(b)?);
                let gap = gap_a.min(gap_b);
                let moved = displacements(a, b)?
                    .iter()
                    .fold(0f64, |m, d| m.max(d.abs()));
                let jumped = circular(line_b - line_a).abs() > 0.5 * gap;
                if moved <= MOVE_FRACTION * gap && !(jumped && upper - lower > soft) {
                    break;
                }
                let middle = (lower + upper) / 2;
                let centers = evaluate(at(middle))?;
                samples.insert(middle, centers);
                refined = true;
                upper = middle;
            }
            if upper - lower <= 1 {
                let (a, b) = (&samples[&lower], &samples[&upper]);
                let gap = largest_gap(a)?.1.min(largest_gap(b)?.1);
                let moved = displacements(a, b)?
                    .iter()
                    .fold(0f64, |m, d| m.max(d.abs()));
                if moved > MOVE_FRACTION * gap {
                    return Err(invalid(format!(
                        "the Wannier-center flow of {label} still moves by {moved:.4} cell (a \
                         quarter of its largest gap is {:.4}) between k_perp = {:.6} and {:.6} \
                         after {} halvings of its transverse intervals: the gap of the occupied \
                         manifold closes there, or the strings are too coarse (raise the string \
                         length)",
                        MOVE_FRACTION * gap,
                        at(lower),
                        at(upper),
                        settings.depth,
                    )));
                }
            }
        }
        if !refined {
            break;
        }
    }
    Ok(Flow {
        k: samples.keys().map(|p| at(*p)).collect(),
        centers: samples.into_values().collect(),
    })
}
