//! AS-6 on two-atom Si in the circulant fcc primitive cell, a 2x2x2 Gamma-centered whole mesh,
//! LDA: eight k-points, complex Bloch phases on every translation, basis functions that reach
//! the neighbor cells. `si` has the second atom off its ideal site by a generic vector, so
//! nothing is symmetric (its stress is not proportional to the identity and the isotropy of the
//! stress does not apply to it); `si_ideal` is the undisplaced diamond cell, cubic, whose stress
//! must be proportional to the identity. The analytic stress takes about 26 s per k-point
//! (the projection and short-range spheres), so each case runs about 4 minutes: both are
//! `#[ignore]`d from the default run. Needs `MANDACARU_PAW_PATH`.
//!
//! `cargo test --release -p mandacaru-core --test periodic_stress_crystals_silicon -- --ignored --test-threads=1`
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_support;

use periodic_stress_crystals_support::run::{check, equation_of_state, isotropy, weak_terms};

/// Terms of a Si stress the 100x rule cannot discriminate: the compensation hats and the
/// short-range ion-compensation sum are about 1e-5 Ha (26 to 130 gates), and the gradient term
/// is empty for LDA. The first two are gated at 1e-9 of their own size by the term oracles of
/// `periodic_stress_electrostatic`.
const SMALL: [&str; 3] = ["hats", "linear", "xc_gradient"];

#[test]
#[ignore = "about 4 minutes: 8 k-points at 26 s each; run with --ignored"]
fn the_stress_of_a_converged_displaced_silicon_state() {
    let (data, stress, want) = check("si", true);
    equation_of_state(&data.manifest, &stress);
    let weak = weak_terms(&stress, &want);
    assert!(
        weak.iter().all(|w| SMALL.contains(w)),
        "terms the oracle cannot discriminate: {weak:?}"
    );
}

#[test]
#[ignore = "about 4 minutes: 8 k-points at 26 s each; run with --ignored"]
fn the_stress_of_undisplaced_silicon_is_isotropic() {
    let (data, stress, want) = check("si_ideal", false);
    equation_of_state(&data.manifest, &stress);
    isotropy(&data.manifest, &stress);
    let weak = weak_terms(&stress, &want);
    assert!(
        weak.iter().all(|w| SMALL.contains(w)),
        "terms the oracle cannot discriminate: {weak:?}"
    );
}
