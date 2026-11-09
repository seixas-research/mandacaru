//! AS-9: the analytic stress of a scalar collinear HSE06 crystal on a REAL converged SCF, against
//! the reconverged free-energy finite difference of the same discretized energy (the frozen
//! Richardson oracle of `freeze_periodic_strain.py`): the triclinic LiH box at Gamma with both
//! divergence treatments of the exact exchange. The stress is the semilocal one with `W` from the
//! bands of the hybrid Hamiltonian plus the explicit exchange terms (H-1..H-6). Needs
//! `MANDACARU_PAW_PATH`; the Si cell on the whole 2x2x2 mesh is
//! `periodic_stress_crystals_hybrid_silicon`.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_hybrid_support;
mod periodic_stress_crystals_support;

use mandacaru_core::{
    periodic::{
        gradient::{Layout, State},
        stress_terms::{check_hybrid_request, check_request, crystal::crystal_stress},
    },
    Error,
};
use periodic_gradient_crystals_support::{inputs, Data};
use periodic_stress_crystals_hybrid_support::run;
use periodic_stress_crystals_support::load;

#[test]
#[ignore = "slow: 15 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_crystals_hybrid -- --ignored the_stress_of_a_converged_hybrid_state_without_a_divergence_treatment"]
fn the_stress_of_a_converged_hybrid_state_without_a_divergence_treatment() {
    // The compensation response `Z` of the pair moments is 3e-5 .. 5e-5 Ha of a stress whose
    // gates are 2e-7 .. 4e-6 Ha: it bites 76 .. 92 gates, named, not hidden (the fixed-state
    // difference gates it at 1e-8).
    run("lih_hse06", true, &["exchange_compensation"]);
}

#[test]
#[ignore = "slow: 16 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_crystals_hybrid -- --ignored the_stress_of_a_converged_hybrid_state_with_gygi_baldereschi"]
fn the_stress_of_a_converged_hybrid_state_with_gygi_baldereschi() {
    run("lih_hse06_gb", false, &["exchange_compensation"]);
}

/// The refusal (`Unsupported`, with its cause) of a stress asked of `data` with no bands: every
/// refusal precedes the freeze.
fn refused(data: &Data, edit: impl FnOnce(&mut Layout)) -> String {
    let cores = data.cores();
    let mut layout = data.layout();
    edit(&mut layout);
    match crystal_stress(
        &data.crystal.context,
        inputs(data, &cores),
        layout,
        data.operators(),
        &data.crystal,
        &data.options,
        &State { bands: Vec::new() },
        None,
    )
    .map(|s| s.volume)
    .expect_err("this crystal must be refused")
    {
        Error::Unsupported(message) => message,
        other => panic!("expected an Unsupported refusal, got {other:?}"),
    }
}

#[test]
fn what_a_hybrid_stress_does_not_differentiate_raises_with_the_cause() {
    let mut data = load("lih_hse06");
    // A spin-polarized hybrid stress is supported; a layout whose spin channels disagree with
    // the crystal's is an invalid request, not a refusal (`stress_terms/crystal.rs`).
    let cases: [(&str, fn(&mut Layout)); 2] = [
        ("spin-orbit projectors", |l| l.request.spinor = true),
        ("float32", |l| l.request.single_precision = true),
    ];
    for (word, edit) in cases {
        let message = refused(&data, edit);
        assert!(message.contains(word), "{word}: {message}");
    }
    // A hybrid crystal without the PAW spheres of its sites block is refused, not differentiated
    // as a semilocal one.
    let spheres = data.crystal.hybrid.take();
    let message = refused(&data, |_| {});
    assert!(message.contains("PAW spheres"), "{message}");
    data.crystal.hybrid = spheres;
    // The two entries refuse each other's functionals.
    let layout = data.layout();
    let message = match check_request(&layout).unwrap_err() {
        Error::Unsupported(message) => message,
        other => panic!("{other:?}"),
    };
    assert!(message.contains("stress_hybrid"), "{message}");
    check_hybrid_request(&layout).unwrap();
    let mut lda = data.layout();
    lda.request.functional = "lda".into();
    assert!(check_hybrid_request(&lda).is_err());
    check_request(&lda).unwrap();
}
