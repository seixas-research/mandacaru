//! The four modes of the search (`Keep`, `Flag`, `Refuse`, `Repair`) and the choices a caller
//! can pin, on recipes that are ghost-ridden or incomplete on purpose: oxygen with its local
//! potential lowered by 3 Ha binds an extra s state at -0.868 Ha (8,677 ghost tolerances), and
//! the unitary aluminum without its 5 Ha raise misses an intruding hydrogen 1s by 1.19. Every
//! error text is checked for its cause. The attempt table is in `paw_search_table.rs`.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_search_support;

use mandacaru_core::models::relativity::Treatment;
use mandacaru_core::models::xc_scheme::AtomSolver;
use mandacaru_core::paw::diagnose::Diagnosis;
use mandacaru_core::paw::diagnose::{ghosts::GHOST_TOLERANCE, reference::AeLogs};
use mandacaru_core::paw::generate::{
    generate_with_atom,
    ranking::Badness,
    recipe::Recipe,
    reference::solve_reference_with,
    search::{
        examine, general, search, search_reviewed, search_with_atom, Attempt, Error, Faithful,
        Mode, Options, Pinned, Review, Searched, Why,
    },
};
use paw_search_support::{context, nan_min};

/// Oxygen with the local potential lowered 3 Ha below its table value: the first construction
/// binds an s ghost at -0.8677 Ha, and zero norm deficit (the first remedy) removes it.
fn ghosted_oxygen() -> Recipe {
    let base = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    Recipe {
        local_shift: -3.0,
        ..base
    }
}

/// The options of a search over a recipe whose odd settings are the generator's own to repair
/// (these recipes lower the local potential or shorten its radius on purpose): nothing is
/// pinned. A recipe whose `local_shift` a caller assigned is pinned by `Options::new`.
fn free(mode: Mode) -> Options {
    Options {
        mode,
        pinned: Pinned::default(),
    }
}

fn run(recipe: &Recipe, options: &Options) -> Result<Searched, Error> {
    let ctx = context();
    let atom = solve_reference_with(&ctx, recipe, &AtomSolver).unwrap();
    search_with_atom(&ctx, recipe, options, &atom, &AtomSolver)
}

fn texts(searched: &Searched) -> Vec<String> {
    searched.tried.iter().map(|t| t.to_string()).collect()
}

#[test]
fn keep_returns_the_ghosted_first_construction_unexamined() {
    let recipe = ghosted_oxygen();
    let ctx = context();
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let kept = search_with_atom(&ctx, &recipe, &free(Mode::Keep), &atom, &AtomSolver).unwrap();
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
    let searched = run(&recipe, &free(Mode::Repair)).unwrap();
    assert_eq!(
        (searched.index, searched.label.as_str()),
        (1, "norm_deficit=0")
    );
    assert!(searched.tried.is_empty() && !searched.flagged);
    let d = searched.diagnosis.as_ref().unwrap();
    assert!(d.is_clean() && d.ghosts.is_empty() && d.missing.is_empty());
    assert_eq!(searched.generated.dataset.defects, Default::default());
    // The same input gives the same attempt and the same construction.
    let again = run(&recipe, &free(Mode::Repair)).unwrap();
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
    let searched = search(&ctx, &recipe, &free(Mode::Flag)).unwrap();
    assert_eq!(searched.label, "norm_deficit=0");
    assert!(!searched.flagged);
}

#[test]
fn refuse_raises_naming_the_ghost() {
    let recipe = ghosted_oxygen();
    let error = run(&recipe, &free(Mode::Refuse)).unwrap_err();
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
    let kept = run(&recipe, &free(Mode::Keep)).unwrap();
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
    let searched = run(&recipe, &free(Mode::Repair)).unwrap();
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
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let first = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let depth = examine(&ctx, &first, &atom, &mut AeLogs::new())
        .unwrap()
        .ghosts[&0];
    drop(first);
    let error = search_with_atom(&ctx, &recipe, &free(Mode::Repair), &atom, &AtomSolver)
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
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let first = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let first_depth = examine(&ctx, &first, &atom, &mut AeLogs::new())
        .unwrap()
        .ghosts[&0];
    drop(first);
    let searched = search_with_atom(&ctx, &recipe, &free(Mode::Flag), &atom, &AtomSolver).unwrap();
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
            let deepest = levels.values().copied().fold(0., nan_min).abs();
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
    let deepest = recorded.ghosts.values().copied().fold(0., nan_min).abs();
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

/// Repair of silicon with the zero-deficit construction already first: its first remedy is the
/// first construction, so the search must not build it twice.
fn duplicated_silicon() -> Recipe {
    Recipe {
        norm_deficit: 0.0,
        local_factor: 0.2,
        local_shift: 0.0,
        ..Recipe::for_dataset("Si", "lda", Treatment::Scalar).unwrap()
    }
}

/// The counter of builds against a search that does build the duplicate: `Searched::built`
/// counts constructions actually generated, so an attempt equal to an earlier one is proved
/// not rebuilt by `built == index` (the first construction and the attempts 2..=index), and the
/// control that turns the reuse off builds one more.
#[test]
fn an_attempt_equal_to_an_earlier_one_is_not_built_again() {
    struct Rebuilds;
    impl Review for Rebuilds {
        fn reuses_duplicates(&self) -> bool {
            false
        }
    }
    let ctx = context();
    let recipe = duplicated_silicon();
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let first = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let d = examine(&ctx, &first, &atom, &mut AeLogs::new()).unwrap();
    let depth = d.ghosts[&0];
    drop(first);
    let run = |review: &dyn Review| {
        search_reviewed(
            &ctx,
            &recipe,
            &free(Mode::Repair),
            &atom,
            &AtomSolver,
            review,
        )
        .unwrap()
    };
    let searched = run(&Faithful);
    let tried = texts(&searched);
    assert_eq!(
        tried[0],
        format!("norm_deficit=0: ghost l=0 {} Ha", general(depth, 3, true))
    );
    assert!(searched.index > 1);
    // The first construction, plus every attempt after the reused first one up to the selected.
    assert_eq!(searched.built, searched.index);
    let wasteful = run(&Rebuilds);
    assert_eq!(
        (wasteful.index, &wasteful.label),
        (searched.index, &searched.label),
        "the reuse changes the cost, not the answer"
    );
    assert_eq!(
        wasteful.built,
        searched.built + 1,
        "the control builds the duplicate"
    );
}

/// Aligned with the generator this replaces: a `local_shift` the caller assigned is a choice
/// the repair would change, so a ghost it binds is refused with the cause, never repaired and
/// never returned flagged. Live Python, oxygen with `local_shift=-3.0` and `ghosts="repair"`
/// (and `"flag"`): `GhostStateError: O: ghost state below the reference (l=0 -0.868 Ha) with
/// local_shift fixed by the caller.  Leave the cutoffs and the local potential to the
/// generator, or pass ghosts='keep' to study it.`
#[test]
fn a_local_shift_assigned_by_the_caller_is_refused_with_the_python_cause() {
    let recipe = ghosted_oxygen();
    assert!(
        Pinned::of(&recipe).local_shift,
        "the shift differs from the table's"
    );
    let expected = "O: ghost state below the reference (l=0 -0.868 Ha) with local_shift fixed by \
                    the caller.  Leave the cutoffs and the local potential to the generator, or \
                    pass ghosts='keep' to study it.";
    for mode in [Mode::Repair, Mode::Flag] {
        let error = run(&recipe, &Options::new(&recipe, mode)).unwrap_err();
        assert!(matches!(error, Error::Refused { .. }), "{mode:?}");
        assert_eq!(error.to_string(), expected, "{mode:?}");
    }
    // The control: the same recipe with nothing pinned is repaired, so the refusal is the pin's.
    let repaired = run(&recipe, &free(Mode::Repair)).unwrap();
    assert_eq!(repaired.label, "norm_deficit=0");
    // And the table's own shift is not a choice of the caller's.
    let table = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    assert_eq!(Pinned::of(&table), Pinned::default());
}

/// A defect the record has a key for only in the new format (a missing state) is judged on every
/// construction, so every attempt fails and `Flag` must return one with its record filled in: the
/// flag is never silent. The judgment is injected through the search's own review hook and the
/// table is cut to two remedies so that the loop stays cheap.
#[test]
fn a_flagged_dataset_always_carries_what_is_wrong_with_it() {
    struct MissingState;
    impl Review for MissingState {
        fn remedies(&self, table: Vec<Attempt>) -> Vec<Attempt> {
            table.into_iter().take(2).collect()
        }
        fn judge(&self, mut d: Diagnosis) -> Diagnosis {
            d.missing.insert(1, 0.0345);
            d
        }
    }
    let ctx = context();
    let recipe = Recipe::for_dataset("O", "lda", Treatment::Scalar).unwrap();
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let go = |mode: Mode| {
        search_reviewed(
            &ctx,
            &recipe,
            &free(mode),
            &atom,
            &AtomSolver,
            &MissingState,
        )
    };
    let searched = go(Mode::Flag).unwrap();
    assert!(searched.flagged);
    let defects = &searched.generated.dataset.defects;
    assert_eq!(defects.missing.get(&1), Some(&0.0345), "{defects:?}");
    assert!(
        defects.ghosts.is_empty() && defects.phases.is_empty() && defects.s_miss.is_none(),
        "nothing else is recorded: {defects:?}"
    );
    assert_eq!(searched.diagnosis.as_ref().unwrap().missing[&1], 0.0345);
    // Repair and Refuse name the cause instead.
    for mode in [Mode::Repair, Mode::Refuse] {
        let error = go(mode).unwrap_err().to_string();
        assert!(
            error.starts_with(
                "O: a bound reference the smooth Hamiltonian does not bind (l=1 +0.0345 Ha)"
            ),
            "{mode:?}: {error}"
        );
    }
    // The control: the faithful search finds no missing state in the same construction, so the
    // answer above is the injected state's, and the flag is not an artifact of the loop.
    let clean = search_with_atom(&ctx, &recipe, &free(Mode::Flag), &atom, &AtomSolver).unwrap();
    assert!(!clean.flagged && clean.generated.dataset.defects.missing.is_empty());
    // The ranking prefers the nearer missing level among otherwise equal attempts.
    let (near, far) = (
        Badness {
            tier: 0,
            severity: 0.0,
            count: 0,
            missing: 0.02,
        },
        Badness {
            tier: 0,
            severity: 0.0,
            count: 0,
            missing: 0.03,
        },
    );
    assert_eq!(near.compare(&far), std::cmp::Ordering::Less);
}
