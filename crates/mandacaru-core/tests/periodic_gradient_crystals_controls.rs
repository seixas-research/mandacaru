//! CF-4 negative controls: a force with ONE thing wrong must miss the frozen reconverged
//! finite-difference oracle by more than 100x its gate. Each variant is compared with the oracle
//! (never merely asserted nonzero): a Pulay or projector term dropped, the sign of the core flipped,
//! the kinetic matrix replaced by the overlap, `W` built with the shifted eigenvalues, the rows
//! or the projectors owned by the wrong atom. Needs `MANDACARU_PAW_PATH`.
//!
//! Cost: a term dropped from the `Breakdown`, and the flipped core (the core force doubled the
//! other way: exact, no second gradient), are free; each provider-level variant is a whole
//! gradient (76 s Si, 131 s Al, 1 s LiH on the loaded 2026-11-02 machine, one thread). The
//! default tests therefore run the provider-level variants on LiH and the cheap ones on all three
//! cases; the full sweep on Si and Al is `#[ignore]`d and run explicitly
//! (`-- --ignored`, 5 gradients each: 378 s Si, 646 s Al2 measured on 2026-11-02, over the 180 s
//! budget; every variant bites by more than 100 gates there).
//!
//! The transposed-lattice control is a refusal now (`periodic_gradient_crystals_refusals.rs`):
//! `with_scf` raises for a lattice that is not the crystal's cell.
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

/// One case: the right force is within a gate of the oracle; every wrong variant is far from it.
/// `provider_level` also runs the variants that need a gradient of their own.
///
/// `reported` lists the controls that cannot reach 100 gates on this crystal because the term
/// itself is small there: they are printed and must still miss by more than ten gates, and the
/// 100-gate bite of the same control is asserted on the crystals where the term is large.
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
    let mut report = vec![];
    // Breakdown-level variants: one term removed (the force is minus the gradient).
    for (name, term) in [
        ("no overlap (-W S) term", &right.overlap),
        ("no projector-moving columns", &right.projection_columns),
        (
            "no basis-moving rows of the projections",
            &right.projection_rows,
        ),
        ("no short-range spheres", &right.short_range),
        ("no hats", &right.hats),
        ("no ion overlap", &right.ion),
        ("no core", &right.core),
    ] {
        report.push((name, bite(&data, &subtract(&right, term))));
    }
    // The flipped core sign: the gradient loses twice the core term (the force gains it).
    let flipped: Vec<[f64; 3]> = right
        .forces()
        .iter()
        .zip(&right.core)
        .map(|(f, c)| [f[0] + 2. * c[0], f[1] + 2. * c[1], f[2] + 2. * c[2]])
        .collect();
    report.push(("core sign flipped", bite(&data, &flipped)));
    // Provider-level variants: each is a gradient of its own.
    if provider_level {
        for (name, wrong) in [
            ("kinetic replaced by overlap", Wrong::Kinetic),
            ("W with the reference kept", Wrong::ReferenceKept),
            ("rows owned by the wrong atom", Wrong::Ownership),
            ("projectors owned by the wrong atom", Wrong::ProjectorOwner),
        ] {
            let altered = Altered::new(provider(&data, &cores), wrong);
            let forces = gradient(&altered, &state, None).unwrap().forces();
            report.push((name, bite(&data, &forces)));
        }
    }
    eprintln!("{case}: the right force is {honest:.2} gates from the oracle; variants, in gates: {report:#?}");
    for name in reported {
        // The conditional list took effect: the control was run, and the term is not invisible.
        let gates = report
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{case}: no control named '{name}'"))
            .1;
        // Floor: measured 33 gates on Si; a collapse of the control to a few gates would show.
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
fn si_dropped_terms_and_the_core_sign_miss_the_oracle_by_more_than_100_gates() {
    // The Si hat force is about 6e-6 Ha/Bohr, 31 plan gates: dropping it cannot bite 100 gates
    // there. The same control bites 1.4e3 (Al2) and 2.6e3 (LiH) gates below.
    controls("si", &["no hats"], false);
}

#[test]
fn al2_dropped_terms_and_the_core_sign_miss_the_oracle_by_more_than_100_gates() {
    controls("al2", &[], false);
}

#[test]
fn lih_every_wrong_variant_misses_the_oracle_by_more_than_100_gates() {
    controls("lih", &[], true);
}

#[test]
#[ignore = "five gradients of 76 s each on one thread: 378 s measured in full (2026-11-02), over the 180 s budget"]
fn si_every_wrong_variant_misses_the_oracle_by_more_than_100_gates() {
    controls("si", &["no hats"], true);
}

#[test]
#[ignore = "five gradients of 131 s each on one thread: 646 s measured in full (2026-11-02), over the 180 s budget"]
fn al2_every_wrong_variant_misses_the_oracle_by_more_than_100_gates() {
    controls("al2", &[], true);
}
