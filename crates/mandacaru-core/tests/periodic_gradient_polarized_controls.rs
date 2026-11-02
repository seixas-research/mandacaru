//! F1 negative controls: the spin-polarized force with ONE thing wrong must miss the frozen
//! reconverged finite-difference oracle by more than 100x the gate of its coordinate. Each variant is
//! compared with the oracle, never merely asserted nonzero: the partial core against `v_up`
//! alone (instead of the spin average), the spin-down channel contracted with the spin-up local
//! potential, both channels counted with the restricted degeneracy, the spin-down bands taken
//! from the spin-up ones, and a term dropped. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::gradient::{gradient, Breakdown};
use periodic_gradient_crystals_support::{altered::*, gates::*, *};

fn subtract(total: &Breakdown, term: &[[f64; 3]]) -> Vec<[f64; 3]> {
    total
        .forces()
        .iter()
        .zip(term)
        .map(|(f, t)| [f[0] + t[0], f[1] + t[1], f[2] + t[2]])
        .collect()
}

fn controls(case: &'static str) {
    let data = Data::load(case);
    let result = data.solve();
    let state = data.state(&result);
    let cores = data.cores();
    let right = gradient(&provider(&data, &cores), &state, None).unwrap();
    let other = second_family(case);
    let honest = bite_families(&data, Some(&other), &right.forces());
    assert!(
        honest < 1.,
        "{case}: the right force misses by {honest} gates"
    );
    let mut report = vec![];
    for (name, term) in [
        ("no core", &right.core),
        ("no hats", &right.hats),
        ("no local-potential term", &right.local),
    ] {
        report.push((
            name,
            bite_families(&data, Some(&other), &subtract(&right, term)),
        ));
    }
    for (name, wrong) in [
        ("partial core against v_up alone", Wrong::CoreSpinUp),
        (
            "spin-down contracted with the v_up local potential",
            Wrong::LocalSpinUp,
        ),
        (
            "restricted degeneracy 2 on both channels",
            Wrong::DegeneracyTwo,
        ),
        (
            "spin-down bands replaced by the spin-up ones",
            Wrong::BandsSpinUp,
        ),
    ] {
        let altered = Altered::new(provider(&data, &cores), wrong);
        let forces = gradient(&altered, &state, None).unwrap().forces();
        report.push((name, bite_families(&data, Some(&other), &forces)));
    }
    eprintln!("{case}: the right force is {honest:.2} gates from the oracle; variants, in gates: {report:#?}");
    for (name, gates) in &report {
        assert!(
            *gates > 100.,
            "{case}: '{name}' bites by only {gates:.1} gates"
        );
    }
}

#[test]
fn beh_lda_wrong_spin_handling_misses_the_oracle_by_more_than_100_gates() {
    controls("beh_lda");
}

#[test]
fn beh_pbe_wrong_spin_handling_misses_the_oracle_by_more_than_100_gates() {
    controls("beh_pbe");
}
