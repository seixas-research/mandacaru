//! HX-W negative controls: the force of a hybrid crystal with ONE thing wrong must miss the frozen
//! reconverged finite-difference oracle by more than 100x its gate. Each variant is compared
//! with the oracle (never merely asserted nonzero): the phase of the compensation shapes, the
//! basis functions of the pair density, the projection response of the exchange (rows, columns,
//! both) left out; the exchange terms charged to the wrong atom; `W` built without the exchange
//! (the whole Hamiltonian, or only the spectrum); the rows or the projectors owned by the wrong
//! atom. Needs `MANDACARU_PAW_PATH`.
//!
//! The terms dropped from the `Breakdown` are free (the force is minus the gradient); the
//! provider-level variants are a whole gradient each.
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::gradient::{gradient, Breakdown};
use periodic_gradient_crystals_support::{altered::*, gates::*, *};

/// The force with `term` removed from the gradient.
fn without(total: &Breakdown, term: &[[f64; 3]]) -> Vec<[f64; 3]> {
    total
        .forces()
        .iter()
        .zip(term)
        .map(|(f, t)| [f[0] + t[0], f[1] + t[1], f[2] + t[2]])
        .collect()
}

/// The force with `term` charged to the other atom of a pair (the owners exchanged).
fn swapped(total: &Breakdown, term: &[[f64; 3]]) -> Vec<[f64; 3]> {
    assert_eq!(term.len(), 2, "the swap control is written for a pair");
    let f = total.forces();
    (0..2)
        .map(|a| {
            let b = 1 - a;
            std::array::from_fn(|x| f[a][x] + term[a][x] - term[b][x])
        })
        .collect()
}

fn sum(a: &[[f64; 3]], b: &[[f64; 3]]) -> Vec<[f64; 3]> {
    a.iter()
        .zip(b)
        .map(|(a, b)| std::array::from_fn(|x| a[x] + b[x]))
        .collect()
}

/// The force with the sign of `term` flipped (the gradient gains twice the term).
fn flipped(total: &Breakdown, term: &[[f64; 3]]) -> Vec<[f64; 3]> {
    total
        .forces()
        .iter()
        .zip(term)
        .map(|(f, t)| std::array::from_fn(|x| f[x] + 2. * t[x]))
        .collect()
}

/// One case: the right force is within a gate of the oracle; every wrong variant is far from it.
/// `provider_level` also runs the variants that need a gradient of their own.
///
/// `reported` lists the controls that cannot reach 100 gates on this crystal because the term
/// itself is small there (or, for an owner swap, because the two atoms' terms are alike): they
/// are printed and must still miss by more than ten gates, and the 100-gate bite of the same
/// control is asserted on the crystal where the term is large.
fn controls(case: &'static str, reported: &[&str], provider_level: bool) {
    let data = Data::load(case);
    let result = data.solve();
    let state = data.state(&result);
    let cores = data.cores();
    let right = gradient(&provider(&data, &cores), &state, None).unwrap();
    let honest = bite(&data, &right.forces());
    assert!(
        honest < 1.,
        "{case}: the right force misses by {honest} gates"
    );
    let projection = sum(&right.exchange_rows, &right.exchange_columns);
    let mut report = vec![
        (
            "no phase of the compensation shapes",
            bite(&data, &without(&right, &right.exchange_phase)),
        ),
        (
            "no basis functions of the pair density",
            bite(&data, &without(&right, &right.exchange_functions)),
        ),
        (
            "no projection response Z (rows and columns)",
            bite(&data, &without(&right, &projection)),
        ),
        (
            "no basis-moving rows of the response",
            bite(&data, &without(&right, &right.exchange_rows)),
        ),
        (
            "no projector-moving columns of the response",
            bite(&data, &without(&right, &right.exchange_columns)),
        ),
        (
            "basis functions of the pair owned by the wrong atom",
            bite(&data, &swapped(&right, &right.exchange_functions)),
        ),
        (
            "phase owned by the wrong atom",
            bite(&data, &swapped(&right, &right.exchange_phase)),
        ),
        (
            "phase sign flipped",
            bite(&data, &flipped(&right, &right.exchange_phase)),
        ),
    ];
    if provider_level {
        for (name, wrong) in [
            (
                "K dropped from the bands W is built from",
                Wrong::NoExchangeInBands,
            ),
            ("semilocal eigenvalues in W", Wrong::SemilocalEigenvalues),
            ("rows owned by the wrong atom", Wrong::Ownership),
            ("projectors owned by the wrong atom", Wrong::ProjectorOwner),
        ] {
            let altered = Altered::hybrid(
                provider(&data, &cores),
                wrong,
                &data.crystal,
                data.options.clone(),
            );
            let forces = gradient(&altered, &state, None).unwrap().forces();
            report.push((name, bite(&data, &forces)));
        }
    }
    eprintln!(
        "{case}: the right force is {honest:.2} gates from the oracle; variants, in gates: {report:#?}"
    );
    for name in reported {
        // The conditional list took effect: the control was run, and the term is not invisible.
        let gates = report
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{case}: no control named '{name}'"))
            .1;
        assert!(gates > 10., "{case}: '{name}' is {gates} gates");
    }
    for (name, gates) in report.iter().filter(|(n, _)| !reported.contains(n)) {
        assert!(
            *gates > 100.,
            "{case}: '{name}' bites by only {gates:.1} gates"
        );
    }
}

#[test]
#[ignore = "slow: 23 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_hybrid_controls -- --ignored lih_every_wrong_variant_misses_the_oracle_by_more_than_100_gates"]
fn lih_every_wrong_variant_misses_the_oracle_by_more_than_100_gates() {
    // The phase terms of the two atoms are alike in this pair (7.1e-5 and 5.8e-5 Ha/Bohr along x),
    // so exchanging their owners moves the force by only 66 gates: reported; dropping the phase
    // (362 gates) and flipping its sign are asserted above 100.
    controls("lih_hse06", &["phase owned by the wrong atom"], true);
}

#[test]
#[ignore = "slow: 149 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_hybrid_controls -- --ignored si_dropped_and_misowned_exchange_terms_miss_the_oracle_by_more_than_100_gates"]
fn si_dropped_and_misowned_exchange_terms_miss_the_oracle_by_more_than_100_gates() {
    // The phase force of this Si cell is about 2.3e-6 Ha/Bohr, 12 plan gates: dropping it cannot
    // bite 100 gates here. The same control bites 362 gates on the LiH pair.
    controls(
        "si_hse06",
        &[
            "no phase of the compensation shapes",
            "phase owned by the wrong atom",
            "phase sign flipped",
        ],
        false,
    );
}

#[test]
#[ignore = "five gradients of the hybrid Si crystal: run explicitly (-- --ignored); the time is in the log"]
fn si_every_wrong_variant_misses_the_oracle_by_more_than_100_gates() {
    controls(
        "si_hse06",
        &[
            "no phase of the compensation shapes",
            "phase owned by the wrong atom",
            "phase sign flipped",
        ],
        true,
    );
}
