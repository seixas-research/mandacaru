//! The search over constructions (`paw::generate::search`, A6) against the A0 freeze of the
//! Python `ghost_free`: for the repaired datasets the same attempt (index and label), the same
//! failed attempts text for text, and the selected construction's scalars and record at the
//! gates of the generator tests; for every other dataset the first attempt, clean, with nothing
//! tried. The freeze is Python and uses the atom-solver radial scheme, so every search runs
//! through that seam. Modes and negative controls are in `paw_search_modes.rs` and
//! `paw_search_controls.rs`.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_search_support;

use mandacaru_core::paw::diagnose::kappa::level_errors;
use mandacaru_core::paw::generate::search::{Mode, Searched};
use mandacaru_core::paw::record::Defects;
use paw_generate_support::{HEAVY, LIGHT};
use paw_search_support::{assert_same_selection, context, open};

/// The datasets whose first construction the search repairs, with the attempt it selects.
const REPAIRED: [(&str, usize); 3] = [("lda-sr_Si", 2), ("lda-sr_K", 1), ("pbe-fr_Si", 1)];

fn clean_first(name: &str, searched: &Searched) {
    assert_eq!(searched.index, 0, "{name}: the first attempt is selected");
    assert_eq!(searched.label, "first construction", "{name}: label");
    assert!(searched.tried.is_empty(), "{name}: nothing was tried");
    assert!(!searched.flagged, "{name}: not flagged");
    assert_eq!(
        searched.generated.dataset.defects,
        Defects::default(),
        "{name}: no defect recorded"
    );
    let diagnosis = searched.diagnosis.as_ref().expect("a repair examines");
    assert!(
        diagnosis.is_clean(),
        "{name}: the first construction is clean"
    );
    assert!(
        diagnosis.missing.is_empty(),
        "{name}: no missing state on the selected construction: {:?}",
        diagnosis.missing
    );
}

/// One frozen dataset searched in `Repair` mode and compared with the freeze; returns the
/// attempt the Python search selected.
fn check(name: &str) -> usize {
    let ctx = context();
    let case = open(&ctx, name);
    let started = std::time::Instant::now();
    let searched = case
        .search(&ctx, Mode::Repair)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    println!(
        "{name}: selected {} ({}), {} tried, {:.1} s, context peak {} B",
        searched.label,
        searched.index,
        searched.tried.len(),
        started.elapsed().as_secs_f64(),
        ctx.memory_peak()
    );
    assert_same_selection(&case, &searched, &|_| CE_EXACT);
    let frozen = paw_search_support::outcome(&case.frozen).index;
    if frozen == 0 {
        clean_first(name, &searched);
    }
    if name.contains("-fr_") {
        // The j branches are examined for ghosts (a clean Dirac dataset has none); their level
        // errors are reported, and the spin-orbit splitting is the stored one to the generator's gate.
        let errors = level_errors(&ctx, &searched.generated.dataset).unwrap();
        for (l, [up, down]) in &errors {
            println!("{name}: j-branch level errors l={l}: {up:+.2e} {down:+.2e} Ha");
        }
    }
    drop(searched);
    drop(case);
    assert_eq!(
        ctx.memory_used(),
        0,
        "{name}: the search gave its memory back"
    );
    frozen
}

#[test]
#[ignore = "slow: 29 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_search -- --ignored the_repaired_light_datasets_select_the_frozen_attempt"]
fn the_repaired_light_datasets_select_the_frozen_attempt() {
    for (name, index) in REPAIRED.iter().filter(|(n, _)| !n.contains("-fr_")) {
        assert!(LIGHT.contains(name));
        assert_eq!(check(name), *index, "{name}: the table of the freeze");
    }
}

#[test]
#[ignore = "slow: 107 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_search -- --ignored every_other_light_dataset_selects_the_first_attempt"]
fn every_other_light_dataset_selects_the_first_attempt() {
    for name in LIGHT {
        if REPAIRED.iter().all(|(n, _)| *n != name) {
            assert_eq!(check(name), 0, "{name}");
        }
    }
}

#[test]
#[ignore = "slow: 12 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_search -- --ignored the_repaired_dirac_dataset_selects_the_frozen_attempt"]
fn the_repaired_dirac_dataset_selects_the_frozen_attempt() {
    // Silicon in PBE with the Dirac atom: the scalar channels are searched exactly as in the
    // scalar-relativistic case, the spin-orbit block follows the selected construction.
    assert_eq!(check("pbe-fr_Si"), 1);
}

#[test]
#[ignore = "slow: 40 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test paw_search -- --ignored the_light_dirac_datasets_select_the_first_attempt"]
fn the_light_dirac_datasets_select_the_first_attempt() {
    for name in ["lda-fr_Na", "lda-fr_Fe"] {
        assert_eq!(check(name), 0, "{name}");
    }
}

macro_rules! heavy {
    ($($test:ident => $name:expr),*) => {$(
        #[test]
        #[ignore = "minutes: cargo test --release -p mandacaru-core --test paw_search -- --ignored"]
        fn $test() {
            check($name);
        }
    )*};
}

heavy!(
    heavy_lda_sr_au => "lda-sr_Au",
    heavy_lda_sr_pb => "lda-sr_Pb",
    heavy_lda_sr_u => "lda-sr_U",
    heavy_lda_fr_au => "lda-fr_Au"
);

#[test]
fn the_heavy_names_are_the_ones_of_the_generator_tests() {
    for name in ["lda-sr_Au", "lda-sr_Pb", "lda-sr_U"] {
        assert!(HEAVY.contains(&name));
    }
}

/// The A0 freeze of the cerium refusal: the error text and the failed attempts.
fn frozen_refusal() -> serde_json::Value {
    let path = paw_assemble_support::directory().join("refuse_lda-sr_Ce.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("frozen refuse_lda-sr_Ce.json"))
        .unwrap()
}

/// The phase of cerium's constructed f channel sits on a sharp resonance: the Python and Rust
/// datasets agree to 1e-12 (`paw_generate`), yet the phase of the selected attempt differs by
/// 3.3e-5 rad (0.27954 against 0.27957), the large log-derivative floor A5 measured on the Ce
/// files. The gate is 1e-3 rad, 30 times that, and the attempt does not depend on it (the
/// unprojected channels decide its rank, at 7.12 tolerances against 5.6 for the phase).
const CE_PHASE_GATE: f64 = 1e-3;
/// The failed attempts of cerium that are wrong by a radian or more (a 5.1 Bohr sphere with an
/// 80 Ha raise, the `balanced at` attempts) are ill-conditioned: the same dataset to 1e-12 gives
/// phases that differ in the third decimal, 9.0e-3 rad measured on pbe-sr attempt 32 (l = 0:
/// 1.515 against 1.506); lda-sr and lda-fr print the same text for every attempt. So the gates
/// are three and each is applied where it is needed: the lda-sr and lda-fr failed attempts are
/// compared **exactly** (`CE_EXACT`), the pbe-sr ones that are not ill-conditioned to one printed
/// quantum (`CE_QUANTUM`: two phases 3.3e-5 rad apart print one unit of the third decimal
/// apart), and only the pbe-sr `balanced at` attempts to `CE_FAILED_GATE`, 1.7 times the
/// measured worst. They never decide a rank (a construction wrong by a radian is far from the
/// least defective).
const CE_EXACT: f64 = 0.;
const CE_QUANTUM: f64 = 1e-4;
const CE_FAILED_GATE: f64 = 1.5e-2;

/// The gate on the printed numbers of a failed attempt of cerium: exact except for the
/// ill-conditioned `balanced at` attempts of the PBE set (`CE_FAILED_GATE`), and one printed
/// quantum for the other PBE attempts (`CE_QUANTUM` plus the rounding).
fn cerium_gate(name: &str) -> impl Fn(&str) -> f64 + '_ {
    move |label: &str| {
        if !name.starts_with("pbe-sr") {
            CE_EXACT
        } else if label.starts_with("balanced at") {
            CE_FAILED_GATE
        } else {
            CE_QUANTUM
        }
    }
}

fn flagged_cerium(name: &str) {
    let ctx = context();
    let case = open(&ctx, name);
    let searched = case.search(&ctx, Mode::Flag).unwrap();
    println!("{name}: flagged {} ({})", searched.label, searched.index);
    assert!(searched.flagged);
    assert_same_selection(&case, &searched, &cerium_gate(name));
    paw_search_support::assert_same_defects(&case.frozen, &searched, 1e-9, CE_PHASE_GATE);
    let diagnosis = searched.diagnosis.as_ref().unwrap();
    assert!(diagnosis.ghosts.is_empty() && diagnosis.missing.is_empty());
}

#[test]
#[ignore = "minutes: cargo test --release -p mandacaru-core --test paw_search -- --ignored"]
fn heavy_cerium_is_flagged_at_the_frozen_attempt() {
    flagged_cerium("lda-sr_Ce");
}

#[test]
#[ignore = "minutes: cargo test --release -p mandacaru-core --test paw_search -- --ignored"]
fn heavy_pbe_cerium_is_flagged_at_the_frozen_attempt() {
    flagged_cerium("pbe-sr_Ce");
}

#[test]
#[ignore = "minutes: cargo test --release -p mandacaru-core --test paw_search -- --ignored"]
fn heavy_dirac_cerium_is_flagged_at_the_frozen_attempt() {
    flagged_cerium("lda-fr_Ce");
}

#[test]
#[ignore = "minutes: cargo test --release -p mandacaru-core --test paw_search -- --ignored"]
fn heavy_cerium_is_refused_with_the_frozen_cause_after_every_remedy() {
    let ctx = context();
    let case = open(&ctx, "lda-sr_Ce");
    let error = case
        .search(&ctx, Mode::Repair)
        .err()
        .expect("cerium has no clean attempt");
    let frozen = frozen_refusal();
    let cause = frozen["ghost_free"]["error"]
        .as_str()
        .unwrap()
        .strip_prefix("GhostStateError: ")
        .unwrap();
    let text = error.to_string();
    assert!(text.starts_with(cause), "{text}\nagainst\n{cause}");
    match error {
        mandacaru_core::paw::generate::search::Error::Exhausted { tried, .. } => {
            let want = frozen["ghost_free"]["tried"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_str().unwrap().to_string())
                .collect::<Vec<_>>();
            paw_search_support::assert_same_tried(
                "lda-sr_Ce",
                &tried,
                &want,
                &cerium_gate("lda-sr_Ce"),
            );
        }
        other => panic!("{other}"),
    }
}

// ---------------------------------------------------------------------------------------------
// The ghost of a j branch (DEC-03 default (b)): lead's j = 3/2 p branch.
// ---------------------------------------------------------------------------------------------

/// `level_errors` of the p branches of lda-fr Pb regenerated by the live Python generator
/// (`generate_paw("Pb", xc="lda", relativity="dirac", ghosts="keep")`, 2026-11-03):
/// `[j = 1/2, j = 3/2]`. The j = 3/2 branch of the FIRST construction binds a state 3.7363 Ha
/// below its reference; the shipped lda-fr file (an earlier generator) does not have it.
const PB_PYTHON_P_BRANCHES: [f64; 2] = [-0.0023127105576690177, -3.7362539618892057];

/// Only the first two remedies: lead's branch ghost survives them, so `Flag` has to return a
/// flagged dataset and `Repair` has to refuse (the third remedy removes it, below).
struct FirstTwoRemedies;

impl mandacaru_core::paw::generate::search::Review for FirstTwoRemedies {
    fn remedies(
        &self,
        table: Vec<mandacaru_core::paw::generate::search::Attempt>,
    ) -> Vec<mandacaru_core::paw::generate::search::Attempt> {
        table.into_iter().take(2).collect()
    }
}

/// Lead, lda-fr, the Dirac atom solved once (573 s in all on a loaded machine, 295 MB). What
/// the scalar generator could not see:
///
/// 1. the first construction's `j = 3/2` p branch holds a ghost, 3.7 Ha below its reference, the
///    live Python generator's number (`PB_PYTHON_P_BRANCHES`) to a percent (the Python level
///    error is one grid without extrapolation, the criterion's is Richardson), while every
///    scalar channel of the same construction is clean (the old search accepted it);
/// 2. `Repair` finds a remedy, `local radius x 0.6, no shift` (the third attempt: the ghost is
///    in the local potential's well, not in the spin-orbit term), and the repaired dataset
///    holds no branch ghost, no scalar one, and a p3/2 level error of the order of the others;
/// 3. when the remedies fail (here: cut to two), `Flag` returns the construction flagged, with
///    the ghost in the record under `"1:-2"`, and `Repair` refuses with the cause, as it does
///    for a ghost of a scalar channel.
#[test]
#[ignore = "ten minutes: cargo test --release -p mandacaru-core --test paw_search heavy_dirac_lead -- --ignored --nocapture"]
fn heavy_dirac_lead_p_branch_ghost_is_found_repaired_flagged_and_refused() {
    use mandacaru_core::paw::diagnose::reference::AeLogs;
    use mandacaru_core::paw::generate::{
        generate_with_atom,
        search::{examine, search_reviewed, Error, Options},
    };
    let ctx = context();
    let case = open(&ctx, "lda-fr_Pb");
    let atom = &case.atom;
    let started = std::time::Instant::now();
    // 1. The first construction.
    let first = generate_with_atom(
        &ctx,
        &case.recipe,
        atom,
        &mandacaru_core::models::xc_scheme::AtomSolver,
    )
    .unwrap();
    let d = examine(&ctx, &first, atom, &mut AeLogs::new()).unwrap();
    assert!(
        d.ghosts.is_empty(),
        "no scalar channel holds a ghost: {:?}",
        d.ghosts
    );
    let key = mandacaru_core::paw::diagnose::view::Key {
        l: 1,
        kappa: Some(-2),
    };
    let depth = d.branch_ghosts[&key];
    println!(
        "Pb first construction: branch ghost {depth:+.5} Ha, live Python {:+.5} ({:.0} s)",
        PB_PYTHON_P_BRANCHES[1],
        started.elapsed().as_secs_f64()
    );
    assert!((depth - PB_PYTHON_P_BRANCHES[1]).abs() < 0.01 * PB_PYTHON_P_BRANCHES[1].abs());
    assert_eq!(d.branch_ghosts.len(), 1, "{:?}", d.branch_ghosts);
    drop(first);
    // 2. Repair, the faithful search.
    let free = Options::new(&case.recipe, Mode::Repair);
    let repaired = case.search(&ctx, Mode::Repair).unwrap();
    println!(
        "Pb repaired by {} ({}), {} constructions, tried {:?}",
        repaired.label,
        repaired.index,
        repaired.built,
        repaired
            .tried
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
    );
    assert!(!repaired.flagged && repaired.index > 0);
    assert_eq!(repaired.label, "local radius x 0.6, no shift");
    let diagnosis = repaired.diagnosis.as_ref().unwrap();
    assert!(diagnosis.branch_ghosts.is_empty() && diagnosis.is_clean());
    let errors = level_errors(&ctx, &repaired.generated.dataset).unwrap();
    println!("Pb repaired: p level errors {:?}", errors[&1]);
    assert!(
        errors[&1][1] > -0.1,
        "the j = 3/2 level is back: {:?}",
        errors[&1]
    );
    // 3. The remedies that fail: Flag flags, Repair refuses.
    let flag = search_reviewed(
        &ctx,
        &case.recipe,
        &Options {
            mode: Mode::Flag,
            ..free
        },
        atom,
        &mandacaru_core::models::xc_scheme::AtomSolver,
        &FirstTwoRemedies,
    )
    .unwrap();
    assert!(flag.flagged, "a branch ghost is reported, never dropped");
    let recorded = &flag.generated.dataset.defects;
    println!("Pb flagged: {recorded:?}");
    let depth = recorded.branch_ghosts[&Defects::branch_key(1, -2)];
    assert!((depth - PB_PYTHON_P_BRANCHES[1]).abs() < 0.05 * PB_PYTHON_P_BRANCHES[1].abs());
    assert!(recorded.ghosts.is_empty() && recorded.branch_ghosts.len() == 1);
    let refused = search_reviewed(
        &ctx,
        &case.recipe,
        &free,
        atom,
        &mandacaru_core::models::xc_scheme::AtomSolver,
        &FirstTwoRemedies,
    )
    .err()
    .expect("the remedies fail");
    let text = refused.to_string();
    assert!(
        text.starts_with(
            "Pb: ghost state below the reference (l=1 kappa=-2 -3.74 Ha) and no remedy removed it"
        ),
        "{text}"
    );
    assert!(matches!(refused, Error::Exhausted { .. }));
    println!("Pb: all four in {:.0} s", started.elapsed().as_secs_f64());
}
