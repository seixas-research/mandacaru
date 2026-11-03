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
    assert_same_selection(&case, &searched, 0.);
    let frozen = paw_search_support::outcome(&case.frozen).index;
    if frozen == 0 {
        clean_first(name, &searched);
    }
    if name.contains("-fr_") {
        // The j branches are not searched over; their level errors are reported, and the spin-orbit
        // splitting of a clean Dirac dataset is the stored one to the generator's gate.
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
fn the_repaired_light_datasets_select_the_frozen_attempt() {
    for (name, index) in REPAIRED.iter().filter(|(n, _)| !n.contains("-fr_")) {
        assert!(LIGHT.contains(name));
        assert_eq!(check(name), *index, "{name}: the table of the freeze");
    }
}

#[test]
fn every_other_light_dataset_selects_the_first_attempt() {
    for name in LIGHT {
        if REPAIRED.iter().all(|(n, _)| *n != name) {
            assert_eq!(check(name), 0, "{name}");
        }
    }
}

#[test]
fn the_repaired_dirac_dataset_selects_the_frozen_attempt() {
    // Silicon in PBE with the Dirac atom: the scalar channels are searched exactly as in the
    // scalar-relativistic case, the spin-orbit block follows the selected construction.
    assert_eq!(check("pbe-fr_Si"), 1);
}

#[test]
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
    heavy_lda_fr_au => "lda-fr_Au",
    heavy_lda_fr_pb => "lda-fr_Pb"
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
/// 80 Ha raise) are ill-conditioned: the same dataset to 1e-12 gives phases that differ in the
/// third decimal, 9.0e-3 rad measured on pbe-sr (lda-sr and lda-fr print the same text) (the printed value of attempt 32, l = 0: 1.515
/// against 1.506). They are compared structurally and by this gate; they never decide a rank
/// (a construction wrong by a radian is far from the least defective).
const CE_FAILED_GATE: f64 = 3e-2;

fn flagged_cerium(name: &str) {
    let ctx = context();
    let case = open(&ctx, name);
    let searched = case.search(&ctx, Mode::Flag).unwrap();
    println!("{name}: flagged {} ({})", searched.label, searched.index);
    assert!(searched.flagged);
    assert_same_selection(&case, &searched, CE_FAILED_GATE);
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
            paw_search_support::assert_same_tried("lda-sr_Ce", &tried, &want, CE_FAILED_GATE);
        }
        other => panic!("{other}"),
    }
}
