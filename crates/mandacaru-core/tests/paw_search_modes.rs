//! The four modes of the search (`Keep`, `Flag`, `Refuse`, `Repair`) and the choices a caller
//! can pin, on recipes that are ghost-ridden or incomplete on purpose: oxygen with its local
//! potential lowered by 3 Ha binds an extra s state at -0.868 Ha (8,677 ghost tolerances), and
//! the unitary aluminum without its 5 Ha raise misses an intruding hydrogen 1s by 1.19. Every
//! error text is checked for its cause. The attempt table is in `paw_search_table.rs`.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_search_support;

use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::paw::diagnose::{ghosts::GHOST_TOLERANCE, reference::AeLogs};
use mandacaru_core::paw::generate::{
    energies::AtomSolver,
    generate_with_atom,
    recipe::Recipe,
    reference::solve_reference,
    search::{
        examine, general, search, search_with_atom, Error, Mode, Options, Pinned, Searched, Why,
    },
};
use paw_search_support::context;

/// Oxygen with the local potential lowered 3 Ha below its table value: the first construction
/// binds an s ghost at -0.8677 Ha, and zero norm deficit (the first remedy) removes it.
fn ghosted_oxygen() -> Recipe {
    let base = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    Recipe {
        local_shift: -3.0,
        ..base
    }
}

fn run(recipe: &Recipe, options: &Options) -> Result<Searched, Error> {
    let ctx = context();
    let atom = solve_reference(&ctx, recipe).unwrap();
    search_with_atom(&ctx, recipe, options, &atom, &AtomSolver)
}

fn texts(searched: &Searched) -> Vec<String> {
    searched.tried.iter().map(|t| t.to_string()).collect()
}

#[test]
fn keep_returns_the_ghosted_first_construction_unexamined() {
    let recipe = ghosted_oxygen();
    let ctx = context();
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let kept = search_with_atom(
        &ctx,
        &recipe,
        &Options::new(&recipe, Mode::Keep),
        &atom,
        &AtomSolver,
    )
    .unwrap();
    assert_eq!((kept.index, kept.label.as_str()), (0, "first construction"));
    assert!(kept.diagnosis.is_none() && kept.tried.is_empty() && !kept.flagged);
    let d = examine(&ctx, &kept.generated, &atom, &mut AeLogs::new()).unwrap();
    let depth = d.ghosts[&0];
    assert!(
        (depth + 0.8677).abs() < 5e-4,
        "the first construction binds an s ghost at {depth} Ha"
    );
    assert!(-depth / GHOST_TOLERANCE > 8000.0);
}

#[test]
fn repair_finds_the_attempt_that_removes_the_ghost() {
    let recipe = ghosted_oxygen();
    let searched = run(&recipe, &Options::new(&recipe, Mode::Repair)).unwrap();
    assert_eq!(
        (searched.index, searched.label.as_str()),
        (1, "norm_deficit=0")
    );
    assert!(searched.tried.is_empty() && !searched.flagged);
    let d = searched.diagnosis.as_ref().unwrap();
    assert!(d.is_clean() && d.ghosts.is_empty() && d.missing.is_empty());
    assert_eq!(searched.generated.dataset.defects, Default::default());
    // The same input gives the same attempt and the same construction.
    let again = run(&recipe, &Options::new(&recipe, Mode::Repair)).unwrap();
    assert_eq!(
        (again.index, &again.label),
        (searched.index, &searched.label)
    );
    assert_eq!(
        again.generated.dataset.one_center_energy.to_bits(),
        searched.generated.dataset.one_center_energy.to_bits()
    );
}

#[test]
fn the_production_path_repairs_the_same_way() {
    // `search` solves its own atom and uses the variational radial scheme.
    let ctx = context();
    let recipe = ghosted_oxygen();
    let searched = search(&ctx, &recipe, &Options::new(&recipe, Mode::Flag)).unwrap();
    assert_eq!(searched.label, "norm_deficit=0");
    assert!(!searched.flagged);
}

#[test]
fn refuse_raises_naming_the_ghost() {
    let recipe = ghosted_oxygen();
    let error = run(&recipe, &Options::new(&recipe, Mode::Refuse)).unwrap_err();
    let text = error.to_string();
    assert!(
        text.starts_with(
            "O: ghost state below the reference (l=0 -0.868 Ha) and ghosts='refuse'.  Leave the cutoffs"
        ),
        "{text}"
    );
    assert!(matches!(error, Error::Refused { .. }));
}

#[test]
fn a_choice_pinned_by_the_caller_is_refused_not_overridden() {
    let recipe = ghosted_oxygen();
    let pinned = Pinned {
        local_shift: true,
        ..Pinned::default()
    };
    for mode in [Mode::Repair, Mode::Flag] {
        let error = run(&recipe, &Options { mode, pinned }).unwrap_err();
        let text = error.to_string();
        assert!(
            text.starts_with(
                "O: ghost state below the reference (l=0 -0.868 Ha) with local_shift fixed by the caller."
            ),
            "{mode:?}: {text}"
        );
    }
    let two = Pinned {
        r_cut: true,
        local_shift: true,
        r_cut_local: false,
    };
    assert_eq!(two.names(), ["r_cut", "local_shift"]);
}

#[test]
fn a_pinned_incomplete_construction_is_returned_flagged() {
    // Unitary aluminum without its 5 Ha raise: the s projectors miss an intruding 1s by 1.19.
    // The raise is the table's, so a repair would change what the caller pinned.
    let base = Recipe::upaw("Al", "lda", Treatment::Scalar).unwrap();
    let recipe = Recipe {
        local_shift: 0.0,
        ..base
    };
    for mode in [Mode::Repair, Mode::Flag] {
        let searched = run(&recipe, &Options::new(&recipe, mode)).unwrap();
        assert!(searched.flagged && searched.index == 0, "{mode:?}");
        let miss = searched.generated.dataset.defects.s_miss.unwrap();
        assert!((miss - 1.19).abs() < 0.01, "the recorded miss is {miss}");
        assert!(searched.generated.dataset.defects.ghosts.is_empty());
    }
    let error = run(&recipe, &Options::new(&recipe, Mode::Refuse)).unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("Al: an incomplete construction (intruding-1s miss 1.19)"),
        "{error}"
    );
    let kept = run(&recipe, &Options::new(&recipe, Mode::Keep)).unwrap();
    assert!(!kept.flagged && kept.generated.dataset.defects == Default::default());
}

#[test]
fn a_deep_ghost_walks_the_remedies_in_the_recorded_order() {
    // Silicon with a local radius of 0.2 times the cutoff and no raise: the s ghost sits at
    // -8.08 Ha, and only the 40 Ha raise at the own cutoffs removes it (the shorter local radii
    // and the smaller raises leave a ghost of -4.2 to -19.5 Ha).
    let base = Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap();
    let recipe = Recipe {
        local_factor: 0.2,
        local_shift: 0.0,
        ..base
    };
    let searched = run(&recipe, &Options::new(&recipe, Mode::Repair)).unwrap();
    assert_eq!(
        (searched.index, searched.label.as_str()),
        (8, "own cutoffs, shift 40")
    );
    let labels = [
        "norm_deficit=0",
        "local radius x 0.75, no shift",
        "local radius x 0.6, no shift",
        "local radius x 0.45, no shift",
        "own cutoffs, shift 5",
        "own cutoffs, shift 10",
        "own cutoffs, shift 20",
    ];
    assert_eq!(searched.tried.len(), labels.len());
    for (tried, label) in searched.tried.iter().zip(labels) {
        assert_eq!(tried.label, label);
        assert!(
            tried
                .to_string()
                .starts_with(&format!("{label}: ghost l=0 -")),
            "{tried}"
        );
    }
}

/// Silicon with a local radius a tenth of the cutoff: every one of the 33 constructions binds an
/// s ghost between -7.9 and -35 Ha, so no remedy removes it.
fn hopeless_silicon() -> Recipe {
    let base = Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap();
    Recipe {
        local_factor: 0.1,
        local_shift: 0.0,
        ..base
    }
}

#[test]
fn when_no_remedy_works_repair_raises_naming_the_ghost_and_every_attempt() {
    let recipe = hopeless_silicon();
    let ctx = context();
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let first = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let depth = examine(&ctx, &first, &atom, &mut AeLogs::new())
        .unwrap()
        .ghosts[&0];
    drop(first);
    let error = search_with_atom(
        &ctx,
        &recipe,
        &Options::new(&recipe, Mode::Repair),
        &atom,
        &AtomSolver,
    )
    .err()
    .expect("no remedy removes the ghost");
    let text = error.to_string();
    let cause = format!(
        "Si: ghost state below the reference (l=0 {} Ha) and no remedy removed it while keeping the scattering (norm_deficit=0: ghost l=0 -",
        general(depth, 3, true)
    );
    assert!(text.starts_with(&cause), "{text}");
    assert!(text.ends_with(")."), "{text}");
    match error {
        Error::Exhausted { tried, .. } => {
            assert_eq!(tried.len(), 32, "every remedy is listed");
            assert!(tried.iter().all(|t| matches!(t.why, Why::Ghost(_))));
        }
        other => panic!("{other}"),
    }
}

#[test]
fn flag_returns_the_attempt_with_the_shallowest_deepest_ghost_and_records_it() {
    let recipe = hopeless_silicon();
    let ctx = context();
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let first = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let first_depth = examine(&ctx, &first, &atom, &mut AeLogs::new())
        .unwrap()
        .ghosts[&0];
    drop(first);
    let searched = search_with_atom(
        &ctx,
        &recipe,
        &Options::new(&recipe, Mode::Flag),
        &atom,
        &AtomSolver,
    )
    .unwrap();
    assert!(searched.flagged);
    let recorded = searched.generated.dataset.defects.clone();
    assert!(
        recorded.phases.is_empty() && recorded.unprojected.is_empty() && recorded.s_miss.is_none()
    );
    // The ranking, recomputed from the list of failed attempts: the first construction and every
    // remedy that was built, by the depth of the deepest ghost, then by the channels it affects.
    let mut best = (first_depth.abs(), 1usize, 0usize);
    for (i, t) in searched.tried.iter().enumerate() {
        if let Why::Ghost(levels) = &t.why {
            let deepest = levels.values().copied().fold(0., f64::min).abs();
            if (deepest, levels.len()) < (best.0, best.1) {
                best = (deepest, levels.len(), i + 1);
            }
        }
    }
    let selected = searched
        .tried
        .iter()
        .position(|t| t.label == searched.label)
        .map(|i| i + 1);
    assert_eq!(selected.unwrap_or(0), best.2, "the least defective attempt");
    assert_eq!(searched.index, best.2);
    let deepest = recorded.ghosts.values().copied().fold(0., f64::min).abs();
    assert_eq!(deepest, best.0);
    // The record carries exactly what the attempt's own verdict says.
    assert!(
        searched.index > 0,
        "the first construction is not the least bad here"
    );
    let verdict = &searched.tried[searched.index - 1].why;
    match verdict {
        Why::Ghost(levels) => assert_eq!(
            levels
                .iter()
                .map(|(&l, &e)| (l as u32, e))
                .collect::<Vec<_>>(),
            recorded
                .ghosts
                .iter()
                .map(|(&l, &e)| (l, e))
                .collect::<Vec<_>>()
        ),
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_attempt_equal_to_an_earlier_one_is_not_built_again() {
    // At zero norm deficit the first remedy is the first construction: it is reported with the
    // verdict of the first, ghost depth included, and the search moves on.
    let ctx = context();
    let base = Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap();
    let recipe = Recipe {
        norm_deficit: 0.0,
        local_factor: 0.2,
        local_shift: 0.0,
        ..base
    };
    let atom = solve_reference(&ctx, &recipe).unwrap();
    let first = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let d = examine(&ctx, &first, &atom, &mut AeLogs::new()).unwrap();
    let depth = d.ghosts[&0];
    drop(first);
    let searched = search_with_atom(
        &ctx,
        &recipe,
        &Options::new(&recipe, Mode::Repair),
        &atom,
        &AtomSolver,
    )
    .unwrap();
    let tried = texts(&searched);
    assert_eq!(
        tried[0],
        format!("norm_deficit=0: ghost l=0 {} Ha", general(depth, 3, true))
    );
    assert!(searched.index > 1);
}
