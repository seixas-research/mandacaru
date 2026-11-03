//! AS-9 on a k MESH, real SCF: the displaced Si cell on the whole 2x2x2 Gamma-centered mesh with
//! HSE06, both divergence treatments, against the reconverged free-energy difference. The
//! stress of the 8-point mesh takes minutes (the strain tangents of eight k-points and 64
//! exchange pairs), so the tests are ignored.
//!
//! Run: `MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test
//! periodic_stress_crystals_hybrid_silicon -- --ignored --test-threads=1`.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_hybrid_support;
mod periodic_stress_crystals_support;

use periodic_stress_crystals_hybrid_support::run;

#[test]
#[ignore = "minutes: 8 k-points of strain tangents and 64 exchange pairs"]
fn the_stress_of_converged_hybrid_silicon_without_a_divergence_treatment() {
    run("si_hse06", true, &["exchange_compensation"]);
}

#[test]
#[ignore = "minutes: 8 k-points of strain tangents and 64 exchange pairs"]
fn the_stress_of_converged_hybrid_silicon_with_gygi_baldereschi() {
    run("si_hse06_gb", false, &["exchange_compensation"]);
}
