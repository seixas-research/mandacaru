//! AS-6: the analytic crystal stress on a REAL converged SCF crystal (scalar LDA, whole k mesh),
//! against the reconverged free-energy finite difference of the SAME discretized energy. The
//! oracle (`freeze_periodic_strain.py`) strains the cell one entry `eps_ab` at a time
//! (unsymmetrized, all nine, both orders of every shear), solves the Rust SCF on each strained
//! cell with the pins of `ANALYTIC_STRESS_PLAN.md` section 2 held (node counts, ion width, filter
//! cutoff, the dense list of the unstrained cell, the alias operation set, the k fractions), and
//! extrapolates in the step (Richardson). The gate of an entry is 100x its measured floor, never
//! looser than the plan's. This file: the skew LiH cell (one k-point), with the equation of
//! state and the controls that remove one term. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use periodic_stress_crystals_support::{
    gates::{bites, entries},
    load,
    run::{check, equation_of_state, solve, weak_terms, SCF_DENSITY_TOLERANCE, SCF_TOLERANCE},
};

#[test]
fn the_stress_of_a_converged_lih_state() {
    let (data, stress, want) = check("lih", true);
    equation_of_state(&data.manifest, &stress);
    let weak = weak_terms(&stress, &want);
    assert_eq!(
        weak,
        ["xc_gradient"],
        "terms the oracle cannot discriminate (only the empty gradient term of LDA may be one)"
    );
}

/// The stress of a loosely converged state is wrong at first order in the SCF residual: the
/// state of the oracle's own tolerances (1e-11 Ha, 1e-9 in the density) still meets the gate (it
/// is the state the oracle was differenced at), and the miss grows as the tolerance is loosened.
#[test]
#[ignore = "slow: 14 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_crystals -- --ignored the_scf_residual_enters_the_stress"]
fn the_scf_residual_enters_the_stress() {
    let mut data = load("lih");
    let volume = data.manifest["volume_bohr3"].as_f64().unwrap();
    let want = entries(&data.manifest, volume);
    let scf = data.manifest["scf"].clone();
    let loose = (
        scf["tolerance"].as_f64().unwrap(),
        scf["density_tolerance"].as_f64().unwrap(),
    );
    let (_, _, reference) = solve(&mut data, SCF_TOLERANCE, SCF_DENSITY_TOLERANCE);
    let (_, _, oracle_tolerance) = solve(&mut data, loose.0, loose.1);
    let gap = bites(&oracle_tolerance.gradient, &want);
    let shift: f64 = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| (oracle_tolerance.gradient[a][b] - reference.gradient[a][b]).abs())
        .fold(0., f64::max);
    println!(
        "  oracle tolerances {loose:?}: worst entry {gap:.2} gates, moved {shift:.2e} Ha from the tight state"
    );
    assert!(
        gap < 1.,
        "the stress at the oracle's own tolerances misses its gate"
    );
    let mut previous = f64::INFINITY;
    let mut first = 0.;
    for (tolerance, density) in [(1e-3, 1e-1), (1e-5, 1e-3), (1e-7, 1e-5)] {
        let (_, _, rough) = solve(&mut data, tolerance, density);
        let bite = bites(&rough.gradient, &want);
        println!(
            "  a state converged to {tolerance:.0e} / {density:.0e}: worst entry {bite:.2e} gates"
        );
        assert!(bite < previous, "the miss does not fall with the tolerance");
        if previous.is_infinite() {
            first = bite;
        }
        previous = bite;
    }
    assert!(first > 1., "the stress does not feel the SCF residual");
}
