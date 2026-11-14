//! `paw::generate::core_states`: the core states of a format-2 library file, solved again on the
//! atom it was generated on and checked against its stored core density, written as format 3 with
//! every other column bit for bit, naming the radial scheme (`AtomSolver` or `Variational`) that
//! reproduces the file's stored core density. Reads the library (`MANDACARU_PAW_PATH`), writes only to a
//! temporary folder.
use mandacaru_core::paw::digest::dataset_content_sha256;
use mandacaru_core::paw::generate::core_states::{
    upgrade_file, DENSITY_TOLERANCE, TIE_ENERGY_TOLERANCE, TIE_ORBITAL_TOLERANCE,
};
use mandacaru_core::paw::record::{PawDataset, FORMAT_VERSION};
use mandacaru_core::Context;
use std::path::PathBuf;

fn library(set: &str, symbol: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("MANDACARU_PAW_PATH").expect("MANDACARU_PAW_PATH"))
        .join(set)
        .join(format!("{symbol}.parquet"))
}

fn folder(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "mandacaru_core_states_{label}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn a_shipped_file_gains_the_core_states_of_its_own_atom() {
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let dir = folder("li_si");
    for (symbol, states) in [("Li", vec![(1, 0)]), ("Si", vec![(1, 0), (2, 0), (2, 1)])] {
        let output = dir.join(format!("{symbol}.parquet"));
        let found = upgrade_file(&ctx, &library("pbe-sr", symbol), &output).unwrap();
        println!("{found:?}");
        assert_eq!(
            found.states.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>(),
            states
        );
        assert!(found.density_deviation <= DENSITY_TOLERANCE);
        // The shipped PBE files were made with the legacy radial scheme, which the production one
        // misses (an LDA file is solved alike by both schemes).
        assert_eq!(found.scheme, Some("atom-solver"));
        assert!(found.tie.is_none());
        let back = PawDataset::read(&ctx, &output).unwrap();
        assert_eq!(back.format_version(), FORMAT_VERSION);
        assert_eq!(back.core_scheme.as_deref(), Some("atom-solver"));
        assert_eq!(
            dataset_content_sha256(&ctx, &output).unwrap(),
            dataset_content_sha256(&ctx, &library("pbe-sr", symbol)).unwrap()
        );
        let total: f64 = back
            .core_states
            .as_ref()
            .unwrap()
            .iter()
            .map(|s| s.occupation)
            .sum();
        // A neutral reference atom: the core holds Z - valence_charge electrons.
        let core = f64::from(back.atomic_number) - back.valence_charge;
        assert!((total - core).abs() < 1e-9, "{total} {found:?}");
        assert!((found.core_charge - core).abs() < 1e-3, "{found:?}");
        // Never over a file, and never an upgrade of an upgrade.
        assert!(upgrade_file(&ctx, &library("pbe-sr", symbol), &output).is_err());
        assert!(upgrade_file(&ctx, &output, &dir.join("again.parquet")).is_err());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_lda_file_and_a_core_free_file_tie_and_are_upgraded_without_a_scheme() {
    // LDA has no density gradient for the two radial schemes to differ on, and a core-free file
    // has no core: both schemes reproduce the file, so it records no scheme (owner Q23).
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let dir = folder("tie");
    for (set, symbol, cored) in [("lda-sr", "Li", true), ("pbe-sr", "H", false)] {
        let output = dir.join(format!("{set}-{symbol}.parquet"));
        let found = upgrade_file(&ctx, &library(set, symbol), &output).unwrap();
        println!("{found:?}");
        assert_eq!(found.scheme, None, "{found:?}");
        let (orbital, energy) = found.tie.expect("a tie");
        assert!(orbital <= TIE_ORBITAL_TOLERANCE && energy <= TIE_ENERGY_TOLERANCE);
        assert_eq!(found.states.is_empty(), !cored);
        let back = PawDataset::read(&ctx, &output).unwrap();
        assert_eq!(back.format_version(), FORMAT_VERSION);
        assert!(back.core_scheme.is_none());
        assert_eq!(back.core_states.as_ref().unwrap().is_empty(), !cored);
        drop(back);
        assert_eq!(
            dataset_content_sha256(&ctx, &output).unwrap(),
            dataset_content_sha256(&ctx, &library(set, symbol)).unwrap()
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "slow: 34 s (the arsenic reference atom, solved with both schemes; measured 2026-11-12); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test paw_generate_core_states -- --ignored"]
fn a_file_written_in_generation_order_is_upgraded_and_proved_by_column_name() {
    // `lda-sr/As` lists its l = 2 channel first (the Python writer's generation order); the
    // record writes channels ascending, so the proof matches columns by name.
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let dir = folder("as");
    let output = dir.join("As.parquet");
    let found = upgrade_file(&ctx, &library("lda-sr", "As"), &output).unwrap();
    assert!(found.density_deviation <= DENSITY_TOLERANCE, "{found:?}");
    assert_eq!(found.scheme, None, "an LDA file ties: {found:?}");
    assert_eq!(
        PawDataset::read(&ctx, &output).unwrap().format_version(),
        FORMAT_VERSION
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "slow: 323 s (the francium reference atom, solved with both schemes; measured 2026-11-12); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test paw_generate_core_states -- --ignored"]
fn a_file_with_reader_defaults_is_upgraded_with_them_written_explicitly() {
    // `lda-sr/Fr` predates `semicore_subshells` and a channel's `occupations`; the reader fills
    // both with its documented defaults, which the upgraded file then writes out.
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let source = PawDataset::read(&ctx, &library("lda-sr", "Fr")).unwrap();
    assert!(!source.defaults_applied.is_empty());
    drop(source);
    let dir = folder("fr");
    let output = dir.join("Fr.parquet");
    let found = upgrade_file(&ctx, &library("lda-sr", "Fr"), &output).unwrap();
    assert_eq!(found.scheme, None, "an LDA file ties: {found:?}");
    assert!(PawDataset::read(&ctx, &output)
        .unwrap()
        .defaults_applied
        .is_empty());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "slow: 45 s (the bromine reference atom, solved with both schemes; measured 2026-11-12); run: MANDACARU_PAW_PATH=... cargo test --profile release-test -p mandacaru-core --test paw_generate_core_states -- --ignored"]
fn a_file_regenerated_with_the_production_scheme_is_identified_by_its_core_density() {
    // `pbe-sr/Br` was regenerated with `Variational`; the legacy `AtomSolver` atom misses its
    // stored core density, so the upgrade records the production scheme.
    let ctx = Context::new(Some(1), 2 << 30).unwrap();
    let dir = folder("br");
    let output = dir.join("Br.parquet");
    let found = upgrade_file(&ctx, &library("pbe-sr", "Br"), &output).unwrap();
    assert_eq!(found.scheme, Some("variational"), "{found:?}");
    assert!(found.density_deviation <= DENSITY_TOLERANCE, "{found:?}");
    let back = PawDataset::read(&ctx, &output).unwrap();
    assert_eq!(back.core_scheme.as_deref(), Some("variational"));
    drop(back);
    assert_eq!(
        dataset_content_sha256(&ctx, &output).unwrap(),
        dataset_content_sha256(&ctx, &library("pbe-sr", "Br")).unwrap()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
