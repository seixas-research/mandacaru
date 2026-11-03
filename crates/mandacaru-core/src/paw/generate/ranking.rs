//! Ranking the constructions that failed: which defective attempt is the least bad.
//!
//! The ranking decides only what a `ghosts = "flag"` search returns when no attempt is clean
//! (the repair search itself stops at the first attempt that is). Its criteria are the ones of
//! the generator this replaces, in the same order, and compared the same way (a tuple, the
//! earlier attempt winning a tie):
//!
//! 1. **A ghost is worst.** Among ghosted attempts the shallowest deepest ghost wins, then the
//!    one with fewer affected channels.
//! 2. **A failed s completeness ranks next** (tier 1): a missed intruding 1s can collapse a bond
//!    where a wrongly scattering channel costs percent-level accuracy.
//! 3. **Otherwise how far the worst defect exceeds its own tolerance:** a phase against
//!    [`PHASE_TOLERANCE`] near the reference (`RESONANCE_TOLERANCE` farther out), an unprojected
//!    channel against [`UNPROJECTED_TOLERANCE`], an s miss against 1.
//!
//! A missing state (a bound reference the smooth Hamiltonian does not bind) has no counterpart
//! in that generator and no tolerance on its scale, so it only breaks a tie: the attempt whose
//! worst reference is closer to its level wins.
use crate::paw::diagnose::{
    intruding::failed_acceptance,
    scattering::{PHASE_TOLERANCE, RESONANCE_TOLERANCE, UNPROJECTED_TOLERANCE},
    Diagnosis,
};
use std::cmp::Ordering;

/// How bad a failed attempt is; smaller is better.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Badness {
    /// 2: a ghost; 1: a failed s completeness; 0: scattering only.
    pub tier: u8,
    /// Depth of the deepest ghost (Ha), or the largest defect over its tolerance.
    pub severity: f64,
    /// Number of ghosted channels (0 without a ghost).
    pub count: usize,
    /// Distance (Ha) of the worst missing state; 0 when none.
    pub missing: f64,
}

impl Badness {
    /// The badness of a diagnosis, by the rules above.
    pub fn of(diagnosis: &Diagnosis) -> Self {
        let missing = diagnosis
            .missing
            .values()
            .map(|d| d.abs())
            .fold(0., f64::max);
        if !diagnosis.ghosts.is_empty() {
            let deepest = diagnosis
                .ghosts
                .values()
                .copied()
                .fold(f64::INFINITY, f64::min);
            return Self {
                tier: 2,
                severity: -deepest,
                count: diagnosis.ghosts.len(),
                missing,
            };
        }
        let phase = diagnosis
            .wrong_phases()
            .values()
            .map(|[near, far]| (near / PHASE_TOLERANCE).max(far / RESONANCE_TOLERANCE))
            .fold(0., f64::max);
        let unprojected = diagnosis
            .unprojected
            .values()
            .copied()
            .filter(|&e| e > UNPROJECTED_TOLERANCE)
            .fold(0., f64::max);
        let s_miss = failed_acceptance(diagnosis.s_miss);
        Self {
            tier: u8::from(s_miss.is_some()),
            severity: phase
                .max(s_miss.unwrap_or(0.))
                .max(unprojected / UNPROJECTED_TOLERANCE),
            count: 0,
            missing,
        }
    }

    /// The order of the ranking: tier, severity, count, then the missing distance.
    pub fn compare(&self, other: &Self) -> Ordering {
        self.tier
            .cmp(&other.tier)
            .then(self.severity.total_cmp(&other.severity))
            .then(self.count.cmp(&other.count))
            .then(self.missing.total_cmp(&other.missing))
    }
}

/// Keeps the least defective item of a stream, and nothing else: a heavy dataset holds many
/// full-grid arrays, and retaining every failed attempt until the search ends exhausts memory.
pub struct Keeper<T> {
    best: Option<(Badness, T)>,
}

impl<T> Default for Keeper<T> {
    fn default() -> Self {
        Self { best: None }
    }
}

impl<T> Keeper<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take `item` if `badness` is strictly less bad than the one held (the earlier attempt wins
    /// a tie); the loser is dropped.
    pub fn offer(&mut self, badness: Badness, item: T) {
        let better = self
            .best
            .as_ref()
            .map_or(true, |(held, _)| badness.compare(held) == Ordering::Less);
        if better {
            self.best = Some((badness, item));
        }
    }

    pub fn into_best(self) -> Option<T> {
        self.best.map(|(_, item)| item)
    }
}
