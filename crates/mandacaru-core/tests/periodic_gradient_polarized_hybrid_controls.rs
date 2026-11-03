//! F1 negative controls of the spin-polarized HSE06 force: a force with ONE thing wrong must miss
//! the frozen reconverged finite-difference oracle (both step families) by more than 100x the
//! gate of its coordinate. Each variant is compared with the oracle: the exchange derivative of
//! the spin-down channel taken from the spin-up density, the spin densities
//! summed before the exchange contraction, the spin-down bands replaced by the spin-up ones, the
//! Hamiltonian diagonalized without its exchange, and the exchange terms dropped one by one. The
//! controls of the projection response (the compensation charges and the spheres, the
//! spin-scaling remainder among them) are in `periodic_gradient_polarized_hybrid.rs`: they need the
//! model, not the provider. Needs `MANDACARU_PAW_PATH`.
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

#[test]
fn h2_hse06_wrong_exchange_handling_misses_the_oracle_by_more_than_100_gates() {
    let case = "h2_hse06";
    let data = Data::load(case);
    let result = data.solve();
    let state = data.state(&result);
    let cores = data.cores();
    let other = second_family(case);
    let right = gradient(&provider(&data, &cores), &state, None).unwrap();
    let honest = bite_families(&data, Some(&other), &right.forces());
    assert!(honest < 1., "the right force misses by {honest} gates");
    let mut report = vec![];
    for (name, term) in [
        ("no exchange phase", &right.exchange_phase),
        ("no exchange basis functions", &right.exchange_functions),
    ] {
        report.push((
            name,
            bite_families(&data, Some(&other), &subtract(&right, term)),
        ));
    }
    for (name, wrong) in [
        (
            "spin-down exchange from the spin-up density",
            Wrong::ExchangeSpinUp,
        ),
        (
            "spin densities summed before the exchange contraction",
            Wrong::ExchangeSummedDensity,
        ),
        (
            "spin-down bands replaced by the spin-up ones",
            Wrong::BandsSpinUp,
        ),
    ] {
        let altered = Altered::new(provider(&data, &cores), wrong);
        let forces = gradient(&altered, &state, None).unwrap().forces();
        let bite = bite_families(&data, Some(&other), &forces);
        report.push((name, bite));
    }
    let altered = Altered::hybrid(
        provider(&data, &cores),
        Wrong::NoExchangeInBands,
        &data.crystal,
        data.options.clone(),
    );
    let b = gradient(&altered, &state, None).unwrap();
    report.push((
        "bands of the semilocal Hamiltonian",
        bite_families(&data, Some(&other), &b.forces()),
    ));
    eprintln!("{case}: the right force is {honest:.2} gates from the oracle; variants, in gates: {report:#?}");
    for (name, gates) in &report {
        assert!(
            *gates > 100.,
            "{case}: '{name}' bites by only {gates:.1} gates"
        );
    }
}
