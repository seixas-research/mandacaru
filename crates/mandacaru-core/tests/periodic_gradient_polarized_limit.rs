//! F1, the restricted limit: the force of a `spins = 2` run without a moment is the `spins = 1`
//! force of the same crystal. The LiH snapshot of CF-0 (restricted) is re-labeled with two spin
//! channels and solved again: the two spin densities stay equal, the energy is the restricted
//! one, so the frozen restricted oracle applies unchanged, and the engine's polarized route
//! (per-spin contraction with degeneracy 1, spin-averaged partial core) must reproduce the
//! restricted route (one spin channel with degeneracy 2) to the round-off of two SCF runs.
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::gradient::gradient;
use periodic_gradient_crystals_support::{gates::*, *};
use serde_json::Value;

/// The LiH snapshot of `case` with two spin channels and no moment.
fn two_channels(case: &'static str) -> Data {
    let path = root().join(format!("{case}.json"));
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut snapshot = snapshot_of(case, &manifest);
    assert_eq!(snapshot["spins"], 1);
    assert!(snapshot["initial_m"]
        .as_array()
        .unwrap()
        .iter()
        .all(|x| x.as_f64() == Some(0.)));
    snapshot["spins"] = Value::from(2);
    Data::build(case, manifest, snapshot)
}

#[test]
fn zero_moment_with_two_spin_channels_is_the_restricted_force() {
    let restricted = Data::load("lih");
    let polarized = two_channels("lih");
    assert_eq!((restricted.crystal.spins, polarized.crystal.spins), (1, 2));
    let (a, b) = (restricted.solve(), polarized.solve());
    eprintln!(
        "lih: free energy restricted {:.12}, polarized {:.12}, moment {:e}",
        a.free_energy, b.free_energy, b.magnetization
    );
    assert!((a.free_energy - b.free_energy).abs() < 1e-9);
    assert!(b.magnetization.abs() < 1e-8);
    let cores = restricted.cores();
    let fa = gradient(&provider(&restricted, &cores), &restricted.state(&a), None).unwrap();
    let fb = gradient(&provider(&polarized, &cores), &polarized.state(&b), None).unwrap();
    // Both against the one frozen oracle of the restricted energy.
    let worst_a = compare("lih (spins = 1)", &restricted, None, &fa.forces());
    let worst_b = compare(
        "lih (spins = 2, no moment)",
        &restricted,
        None,
        &fb.forces(),
    );
    eprintln!("lih: worst miss {worst_a:.3} (spins = 1) and {worst_b:.3} (spins = 2) gates");
    // And against each other, term by term, to the round-off of the two SCF runs.
    let mut worst = 0f64;
    for ((name, ta), (_, tb)) in fa.terms().iter().zip(fb.terms().iter()) {
        let gap = ta
            .iter()
            .zip(tb.iter())
            .flat_map(|(x, y)| (0..3).map(move |i| (x[i] - y[i]).abs()))
            .fold(0f64, f64::max);
        eprintln!("lih: {name}: |spins=1 - spins=2| = {gap:.2e}");
        worst = worst.max(gap);
    }
    assert!(worst < 1e-9, "the restricted limit is off by {worst:e}");
}
