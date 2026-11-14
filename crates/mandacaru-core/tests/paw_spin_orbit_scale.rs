//! The measured spin-orbit calibration of the Dirac datasets (`paw::generate::tables_spin_orbit`,
//! gate G13 of the 2026-11 regeneration): the table is well formed, and the generator applies it
//! to the stored `coupling` of the channel it names and to nothing else.
use mandacaru_core::paw::generate::{recipe::Recipe, tables_spin_orbit};
use std::collections::BTreeSet;

#[test]
fn the_table_is_well_formed() {
    let mut seen = BTreeSet::new();
    for (symbol, functional, l, scale) in tables_spin_orbit::entries() {
        assert!(
            ["lda", "pbe"].contains(&functional),
            "{symbol}: functional {functional}"
        );
        assert!((1..=3).contains(&l), "{symbol}: l = {l}");
        // A calibration moves the splitting by tens of percent at most; anything else is a
        // misidentified shell in the measurement.
        assert!(
            (0.9..=1.35).contains(&scale),
            "{symbol} {functional} l={l}: scale {scale}"
        );
        assert!(
            seen.insert((symbol, functional, l)),
            "{symbol} {functional} l={l} is listed twice"
        );
        assert!(
            Recipe::for_dataset(
                symbol,
                functional,
                mandacaru_core::models::relativity::Treatment::Dirac
            )
            .is_ok(),
            "{symbol} {functional} is not a recipe"
        );
    }
}

#[test]
fn an_absent_shell_keeps_its_coupling() {
    assert_eq!(tables_spin_orbit::scale("H", "pbe", 1), 1.);
    assert_eq!(tables_spin_orbit::scale("Bi", "lda", 3), 1.);
}
