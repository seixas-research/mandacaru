//! Negative controls of the search (A6): each wrong variant is built from the same public pieces
//! as the search (the attempt table, `examine`, the ranking), run on the same ghost-ridden recipe,
//! and compared with what the search selects. A variant that does not change the selection, or
//! changes it by less than 100 ghost tolerances, would mean the search does not depend on the
//! piece that was broken.
//!
//! The recipe is oxygen with its local potential lowered by 3 Ha (an s ghost at -0.868 Ha); the
//! search removes it with the first remedy. The ranking control uses silicon with a short local
//! radius, whose first eight attempts all bind a ghost between -0.4 and -19.5 Ha.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_search_support;

use mandacaru_core::models::{atom::Atom, relativity::Treatment};
use mandacaru_core::paw::diagnose::{
    ghosts::GHOST_TOLERANCE,
    reference::{AeLogs, AllElectron},
    view::Smooth,
    Diagnosis,
};
use mandacaru_core::paw::generate::{
    energies::AtomSolver,
    generate_with_atom,
    ranking::Badness,
    recipe::Recipe,
    reference::solve_reference,
    search::{examine, plan, radii_of, search_with_atom, Mode, Options, Why},
};
use mandacaru_core::Context;
use paw_search_support::context;

fn ghosted_oxygen() -> Recipe {
    Recipe {
        local_shift: -3.0,
        ..Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap()
    }
}

/// The deepest ghost (Ha, negative) of a diagnosis, 0 when none.
fn deepest(d: &Diagnosis) -> f64 {
    d.ghosts.values().copied().fold(0., f64::min)
}

/// A search that does not look at ghosts: every other check runs on the full diagnosis (the
/// search itself stops at a ghost), the ghost list is cleared, and the first construction that
/// passes is kept. Returns its position, label and the deepest ghost it binds.
fn blind(ctx: &Context, recipe: &Recipe, atom: &Atom) -> (usize, String, f64) {
    let first = generate_with_atom(ctx, recipe, atom, &AtomSolver).unwrap();
    let table = plan(recipe, &radii_of(&first.dataset));
    let mut cache = AeLogs::new();
    let mut candidates = std::iter::once(("first construction".to_string(), Some(first)))
        .chain(table.into_iter().map(|a| {
            let built = generate_with_atom(ctx, &a.recipe, atom, &AtomSolver).ok();
            (a.label, built)
        }))
        .enumerate();
    // Lazily: the remedies are only built as far as the blind search needs them.
    while let Some((index, (label, generated))) = candidates.next() {
        let Some(generated) = generated else { continue };
        let smooth = Smooth::scalar(ctx, &generated.dataset).unwrap();
        let ae = AllElectron::new(atom, smooth.atomic_number, &smooth.frozen, &smooth.semicore);
        let mut d = Diagnosis::run(ctx, &smooth, &ae, &mut cache).unwrap();
        let ghost = deepest(&d);
        d.ghosts.clear();
        if Why::of(&d).is_none() {
            return (index, label, ghost);
        }
    }
    panic!("no attempt is clean");
}

/// Silicon with a local radius a fifth of the cutoff: an s ghost at -8.08 Ha.
fn deep_silicon() -> Recipe {
    Recipe {
        local_factor: 0.2,
        local_shift: 0.0,
        ..Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap()
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
        let atom = solve_reference(&ctx, &recipe).unwrap();
        let right = search_with_atom(
            &ctx,
            &recipe,
            &Options::new(&recipe, Mode::Repair),
            &atom,
            &AtomSolver,
        )
        .unwrap();
        let (index, label, ghost) = blind(&ctx, &recipe, &atom);
        println!(
            "{name}: the search selects {} ({}); without the ghost criterion {label} ({index}), ghost {ghost:+.3} Ha ({:.0} tolerances)",
            right.label,
            right.index,
            -ghost / GHOST_TOLERANCE
        );
        if label != right.label {
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
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let first = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let table = plan(&recipe, &radii_of(&first.dataset));
    assert_eq!(table[0].label, "norm_deficit=0");
    let mut cache = AeLogs::new();
    let mut chosen = None;
    for attempt in table.iter().skip(1) {
        let Ok(built) = generate_with_atom(&ctx, &attempt.recipe, &atom, &AtomSolver) else {
            continue;
        };
        if let Ok(d) = examine(&ctx, &built, &atom, &mut cache) {
            if Why::of(&d).is_none() {
                chosen = Some((attempt.label.clone(), built));
                break;
            }
        }
    }
    let (label, built) = chosen.expect("a later attempt is clean");
    assert_ne!(label, "norm_deficit=0");
    // And a different dataset: the oracle's construction keeps the lowered local potential at its
    // radius, the one this control keeps has the 0.75 times shorter radius and no lowering.
    let oracle = generate_with_atom(&ctx, &table[0].recipe, &atom, &AtomSolver).unwrap();
    let (a, b) = (oracle.dataset.r_cut_local, built.dataset.r_cut_local);
    println!(
        "skipped remedy: {label}; r_cut_local {a} against {b}; local_shift {} against {}",
        oracle.dataset.local_shift, built.dataset.local_shift
    );
    assert!(
        (a - b).abs() / a > 100.0 * 1e-12,
        "the local radius moved by {:.2e}",
        (a - b).abs() / a
    );
    assert_ne!(oracle.dataset.local_shift, built.dataset.local_shift);
}

#[test]
fn keeping_the_first_attempt_always_returns_the_ghost() {
    let ctx = context();
    let recipe = ghosted_oxygen();
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let kept = search_with_atom(
        &ctx,
        &recipe,
        &Options::new(&recipe, Mode::Keep),
        &atom,
        &AtomSolver,
    )
    .unwrap();
    let d = examine(&ctx, &kept.generated, &atom, &mut AeLogs::new()).unwrap();
    assert!(-deepest(&d) / GHOST_TOLERANCE > 100.0);
}

#[test]
fn the_reversed_ranking_keeps_the_deepest_ghost_among_real_attempts() {
    let ctx = context();
    let base = Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap();
    let recipe = Recipe {
        local_factor: 0.2,
        local_shift: 0.0,
        ..base
    };
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let first = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let table = plan(&recipe, &radii_of(&first.dataset));
    let mut cache = AeLogs::new();
    let mut ranked = vec![(
        "first construction".to_string(),
        examine(&ctx, &first, &atom, &mut cache).unwrap(),
    )];
    drop(first);
    for attempt in table.iter().take(7) {
        let built = generate_with_atom(&ctx, &attempt.recipe, &atom, &AtomSolver).unwrap();
        ranked.push((
            attempt.label.clone(),
            examine(&ctx, &built, &atom, &mut cache).unwrap(),
        ));
    }
    assert!(ranked.iter().all(|(_, d)| !d.ghosts.is_empty()));
    let key = |d: &Diagnosis| Badness::of(d);
    let right = (0..ranked.len())
        .min_by(|&a, &b| key(&ranked[a].1).compare(&key(&ranked[b].1)))
        .unwrap();
    let reversed = (0..ranked.len())
        .max_by(|&a, &b| key(&ranked[a].1).compare(&key(&ranked[b].1)))
        .unwrap();
    println!(
        "ranking keeps {} ({:+.3} Ha), reversed keeps {} ({:+.3} Ha)",
        ranked[right].0,
        deepest(&ranked[right].1),
        ranked[reversed].0,
        deepest(&ranked[reversed].1)
    );
    assert_eq!(ranked[right].0, "own cutoffs, shift 20");
    assert_eq!(ranked[reversed].0, "local radius x 0.45, no shift");
    assert!(
        (deepest(&ranked[reversed].1) - deepest(&ranked[right].1)).abs() / GHOST_TOLERANCE > 100.0
    );
}
