//! Negative controls of the search (A6): each wrong variant is the search itself with one piece
//! replaced through its own review hook (`search_reviewed`: the loop of `search.rs`, not a copy
//! of it in a test), run on the same ghost-ridden recipe and compared with what the faithful
//! search returns. A variant that does not change the selection, or changes it by less than 100
//! ghost tolerances, would mean the search does not depend on the piece that was broken.
//!
//! The recipe is oxygen with its local potential lowered by 3 Ha (an s ghost at -0.868 Ha); the
//! search removes it with the first remedy. The ranking control uses silicon with a short local
//! radius, whose first eight attempts all bind a ghost between -0.4 and -19.5 Ha.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_search_support;

use mandacaru_core::models::{atom::Atom, relativity::Treatment, xc_scheme::AtomSolver};
use mandacaru_core::paw::diagnose::{ghosts::GHOST_TOLERANCE, reference::AeLogs, Diagnosis};
use mandacaru_core::paw::generate::{
    ranking::Badness,
    recipe::Recipe,
    reference::solve_reference_with,
    search::{
        examine, search_reviewed, Attempt, Error, Faithful, Mode, Options, Pinned, Review, Searched,
    },
};
use mandacaru_core::Context;
use paw_search_support::context;
use std::cmp::Ordering;

fn ghosted_oxygen() -> Recipe {
    Recipe {
        local_shift: -3.0,
        ..Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap()
    }
}

/// Silicon with a local radius a fifth of the cutoff: an s ghost at -8.08 Ha.
fn deep_silicon() -> Recipe {
    Recipe {
        local_factor: 0.2,
        local_shift: 0.0,
        ..Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap()
    }
}

/// Nothing pinned: the odd settings of these recipes are the generator's to repair.
fn free(mode: Mode) -> Options {
    Options {
        mode,
        pinned: Pinned::default(),
    }
}

/// The deepest ghost (Ha, negative) of a diagnosis, 0 when none; a NaN depth is returned as NaN.
fn deepest(d: &Diagnosis) -> f64 {
    d.ghosts.values().copied().fold(0., |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    })
}

fn search(
    ctx: &Context,
    recipe: &Recipe,
    atom: &Atom,
    mode: Mode,
    review: &dyn Review,
) -> Result<Searched, Error> {
    search_reviewed(ctx, recipe, &free(mode), atom, &AtomSolver, review)
}

/// The ghost of the construction a search returned, measured by the search's own `examine`.
fn ghost_of(ctx: &Context, atom: &Atom, searched: &Searched) -> f64 {
    deepest(&examine(ctx, &searched.generated, atom, &mut AeLogs::new()).unwrap())
}

/// A search without the ghost criterion: every other check runs on the full diagnosis (the
/// faithful search stops at a ghost), the ghost list is cleared and the first construction that
/// passes is kept.
struct GhostBlind;

impl Review for GhostBlind {
    fn stops_at_ghost(&self) -> bool {
        false
    }
    fn judge(&self, mut d: Diagnosis) -> Diagnosis {
        d.ghosts.clear();
        d.branch_ghosts.clear();
        d
    }
}

/// A search without its first remedy.
struct SkipFirstRemedy;

impl Review for SkipFirstRemedy {
    fn remedies(&self, table: Vec<Attempt>) -> Vec<Attempt> {
        table.into_iter().skip(1).collect()
    }
}

/// A search that picks the worst attempt where the ranking picks the best, over the first eight
/// attempts (all ghosted, so the table has no clean one and `Flag` has to rank).
struct ReversedRanking;

impl Review for ReversedRanking {
    fn remedies(&self, table: Vec<Attempt>) -> Vec<Attempt> {
        table.into_iter().take(7).collect()
    }
    fn rank(&self, a: &Badness, b: &Badness) -> Ordering {
        b.compare(a)
    }
}

/// The oracle of the ranking control: the faithful ranking over the same eight attempts.
struct FirstEight;

impl Review for FirstEight {
    fn remedies(&self, table: Vec<Attempt>) -> Vec<Attempt> {
        table.into_iter().take(7).collect()
    }
}

#[test]
fn without_the_ghost_criterion_a_ghosted_construction_is_accepted() {
    // Where the scattering check does not see the ghost either, the search without the ghost
    // criterion keeps the first construction; where it does (a ghost near the window), the label
    // is unchanged and the ghost is reported as caught twice. At least one recipe must show the
    // difference, and in every case the construction that is kept binds a ghost or is the oracle's.
    let ctx = context();
    let mut changed = 0;
    for (name, recipe) in [
        ("oxygen -3 Ha", ghosted_oxygen()),
        ("silicon 0.2", deep_silicon()),
    ] {
        let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
        let right = search(&ctx, &recipe, &atom, Mode::Repair, &Faithful).unwrap();
        let blind = search(&ctx, &recipe, &atom, Mode::Repair, &GhostBlind).unwrap();
        let ghost = ghost_of(&ctx, &atom, &blind);
        println!(
            "{name}: the search selects {} ({}); without the ghost criterion {} ({}), ghost {ghost:+.3} Ha ({:.0} tolerances)",
            right.label,
            right.index,
            blind.label,
            blind.index,
            -ghost / GHOST_TOLERANCE
        );
        assert_eq!(
            ghost_of(&ctx, &atom, &right),
            0.0,
            "the oracle is ghost-free"
        );
        if blind.label != right.label {
            changed += 1;
            assert!(
                -ghost / GHOST_TOLERANCE > 100.0,
                "{name}: the construction the blind search keeps binds a ghost {} tolerances deep",
                -ghost / GHOST_TOLERANCE
            );
        }
    }
    assert!(changed >= 1, "the ghost criterion changed no selection");
}

#[test]
fn skipping_the_first_remedy_selects_another_attempt_and_a_different_construction() {
    let ctx = context();
    let recipe = ghosted_oxygen();
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let oracle = search(&ctx, &recipe, &atom, Mode::Repair, &Faithful).unwrap();
    assert_eq!(oracle.label, "norm_deficit=0");
    let skipped = search(&ctx, &recipe, &atom, Mode::Repair, &SkipFirstRemedy).unwrap();
    assert_ne!(skipped.label, oracle.label);
    // And a different dataset: the oracle's construction keeps the lowered local potential at its
    // radius, the one this control keeps has the 0.75 times shorter radius and no lowering.
    let (a, b) = (
        oracle.generated.dataset.r_cut_local,
        skipped.generated.dataset.r_cut_local,
    );
    println!(
        "skipped remedy: {}; r_cut_local {a} against {b}; local_shift {} against {}",
        skipped.label, oracle.generated.dataset.local_shift, skipped.generated.dataset.local_shift
    );
    assert!(
        (a - b).abs() / a > 100.0 * 1e-12,
        "the local radius moved by {:.2e}",
        (a - b).abs() / a
    );
    assert_ne!(
        oracle.generated.dataset.local_shift,
        skipped.generated.dataset.local_shift
    );
}

#[test]
fn keeping_the_first_attempt_always_returns_the_ghost_the_repair_removes() {
    let ctx = context();
    let recipe = ghosted_oxygen();
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let repaired = search(&ctx, &recipe, &atom, Mode::Repair, &Faithful).unwrap();
    let kept = search(&ctx, &recipe, &atom, Mode::Keep, &Faithful).unwrap();
    // The oracle is the Repair result: clean, and a different construction from the first.
    assert!(repaired.diagnosis.as_ref().unwrap().is_clean());
    assert_eq!((repaired.index, kept.index), (1, 0));
    assert_ne!(
        repaired.generated.dataset.one_center_energy.to_bits(),
        kept.generated.dataset.one_center_energy.to_bits(),
        "the repair changed the construction"
    );
    let (clean, ghost) = (
        ghost_of(&ctx, &atom, &repaired),
        ghost_of(&ctx, &atom, &kept),
    );
    println!(
        "keep: ghost {ghost:+.4} Ha ({:.0} tolerances); repair: {clean}",
        -ghost / GHOST_TOLERANCE
    );
    assert_eq!(clean, 0.0);
    assert!(-ghost / GHOST_TOLERANCE > 100.0);
}

#[test]
fn the_reversed_ranking_keeps_the_deepest_ghost_among_real_attempts() {
    let ctx = context();
    let recipe = deep_silicon();
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let right = search(&ctx, &recipe, &atom, Mode::Flag, &FirstEight).unwrap();
    let reversed = search(&ctx, &recipe, &atom, Mode::Flag, &ReversedRanking).unwrap();
    assert!(right.flagged && reversed.flagged);
    let (a, b) = (
        ghost_of(&ctx, &atom, &right),
        ghost_of(&ctx, &atom, &reversed),
    );
    println!(
        "ranking keeps {} ({a:+.3} Ha), reversed keeps {} ({b:+.3} Ha)",
        right.label, reversed.label
    );
    assert_eq!(right.label, "own cutoffs, shift 20");
    assert_eq!(reversed.label, "local radius x 0.45, no shift");
    assert!((b - a).abs() / GHOST_TOLERANCE > 100.0);
    // The record carries the kept attempt's ghost, not the other's.
    let recorded = |s: &Searched| s.generated.dataset.defects.ghosts[&0];
    assert!((recorded(&right) - a).abs() < 1e-9 && (recorded(&reversed) - b).abs() < 1e-9);
}
