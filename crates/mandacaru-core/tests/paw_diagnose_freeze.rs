//! The ghost, scattering and completeness checks against the values the A0 freeze recorded for
//! the selected candidate of every non-Dirac generated dataset
//! (`gen_*.json`, `ghost_free.attempts[selected].checks` and `.intruding_s_miss`): the dataset is
//! generated again by the Rust generator through the atom-solver seam (the scheme of the
//! frozen numbers), at full resolution, and diagnosed against the same reference atom.
//!
//! The frozen Dirac (`fr`) datasets are not here: their generation is the spin-orbit chain
//! (A4) and the freeze stores no table of them at full resolution. The `kappa`-resolved checks
//! are exercised on the shipped fr files in `paw_diagnose_kappa`.
//!
//! Gates: the ghost list equal; the phases (rad) and the intruding miss at `PHASE_GATE` /
//! `MISS_GATE`, set from the measurement of the A5 log.

mod paw_assemble_support;
mod paw_diagnose_support;
mod paw_generate_support;

use mandacaru_core::models::xc_scheme::AtomSolver;
use mandacaru_core::paw::diagnose::{
    reference::{AeLogs, AllElectron},
    view::Smooth,
    Diagnosis,
};
use mandacaru_core::paw::generate::{generate_with_atom, reference::solve_reference_with};
use paw_assemble_support::Frozen;
use paw_diagnose_support::*;
use paw_generate_support::{recipe_of, HEAVY, LIGHT};

/// Measured on the 14 light datasets: 6.4e-12 rad (pbe-sr Cu) and 5.2e-12 relative (lda-sr Mg);
/// the gates are 150 and 20 times those. The heavy datasets are not run here.
const PHASE_GATE: f64 = 1e-9;
const MISS_GATE: f64 = 1e-10;

/// The values the freeze recorded for the selected attempt.
fn recorded(frozen: &Frozen) -> (serde_json::Value, f64) {
    let search = &frozen.doc["ghost_free"];
    let selected = search["selected_index"].as_u64().unwrap();
    let attempt = search["attempts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["index"].as_u64() == Some(selected))
        .unwrap();
    (
        attempt["checks"].clone(),
        attempt["intruding_s_miss"].as_f64().unwrap(),
    )
}

/// `(worst phase deviation, miss deviation)` of a regenerated dataset against the freeze.
fn parity(name: &str) -> (f64, f64) {
    let frozen = Frozen::open(name);
    let (checks, miss) = recorded(&frozen);
    let ctx = context();
    let recipe = recipe_of(&frozen);
    let atom = solve_reference_with(&ctx, &recipe, &AtomSolver).unwrap();
    let generated = generate_with_atom(&ctx, &recipe, &atom, &AtomSolver).unwrap();
    let smooth = Smooth::scalar(&ctx, &generated.dataset).unwrap();
    let ae = AllElectron::new(
        &atom,
        recipe.atomic_number,
        &smooth.frozen,
        &smooth.semicore,
    );
    let got = Diagnosis::run(&ctx, &smooth, &ae, &mut AeLogs::new()).unwrap();

    let ghosts = checks["ghost_errors"].as_object().unwrap();
    assert_eq!(
        got.ghosts.keys().map(|l| l.to_string()).collect::<Vec<_>>(),
        ghosts.keys().cloned().collect::<Vec<_>>(),
        "{name}: ghost channels"
    );
    let mut phase = 0f64;
    let want = checks["scattering_errors"].as_object().unwrap();
    assert_eq!(want.len(), got.phases.len(), "{name}: scattering channels");
    for (key, [near, far]) in &got.phases {
        let frozen = numbers(&want[&key.l.to_string()]);
        phase = nan_max(phase, (near - frozen[0]).abs());
        phase = nan_max(phase, (far - frozen[1]).abs());
    }
    let want = checks["unprojected_scattering_errors"].as_object().unwrap();
    assert_eq!(
        want.len(),
        got.unprojected.len(),
        "{name}: unprojected channels"
    );
    for (l, value) in &got.unprojected {
        phase = nan_max(
            phase,
            (value - want[&l.to_string()].as_f64().unwrap()).abs(),
        );
    }
    let dev = (got.s_miss.expect("every frozen dataset has an s channel") - miss).abs() / miss;
    eprintln!("{name}: phases {phase:.2e} rad, intruding miss {dev:.2e}");
    (phase, dev)
}

fn gate(name: &str) {
    let (phase, miss) = parity(name);
    assert!(
        phase <= PHASE_GATE,
        "{name}: phase deviation {phase:.3e} rad"
    );
    assert!(miss <= MISS_GATE, "{name}: miss deviation {miss:.3e}");
}

macro_rules! sample {
    ($($test:ident: $name:literal),+ $(,)?) => {$(
        #[test]
        fn $test() {
            assert!(LIGHT.contains(&$name));
            gate($name);
        }
    )+};
}

sample! {
    h_lda: "lda-sr_H", c_lda: "lda-sr_C", o_lda: "lda-sr_O", na_lda: "lda-sr_Na",
    mg_lda: "lda-sr_Mg", si_lda_after_a_repair: "lda-sr_Si", k_lda_after_a_repair: "lda-sr_K",
    fe_lda: "lda-sr_Fe", o_pbe: "pbe-sr_O", cu_pbe: "pbe-sr_Cu",
    h_upaw: "upaw_H", o_upaw: "upaw_O", al_upaw: "upaw_Al", cr_upaw: "upaw_Cr",
}

#[test]
#[ignore = "five heavy datasets (Ce, Au, Pb, U): atom solves of minutes each; run with --ignored --test-threads=1"]
fn the_heavy_datasets_equal_the_freeze() {
    for name in HEAVY {
        gate(name);
    }
}
