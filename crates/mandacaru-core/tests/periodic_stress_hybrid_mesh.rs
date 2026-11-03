//! AS-9 on a k MESH: the exchange strain of the displaced Si cell on the whole 2x2x2
//! Gamma-centered mesh (every pair has its own `k_diff`, the Nyquist faces of the alias averages
//! are live) against the central differences of the production exchange energy at fixed density
//! matrices, with both divergence treatments. 64 (k, q) pairs, four differences of nine entries
//! each: over the 180 s budget, so the tests are ignored.
//!
//! Run: `MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test
//! periodic_stress_hybrid_mesh -- --ignored --test-threads=1`.
mod periodic_gradient_crystals_support;
mod periodic_stress_hybrid_support;

use mandacaru_core::periodic::screened_exchange::ExxDivTreatment;
use periodic_stress_hybrid_support::{check, controls_kernel};

const STEP: f64 = 2.5e-6;

#[test]
#[ignore = "over 180 s: 4 differences x 9 entries x 6 energies of 64 pairs"]
fn the_exchange_strain_of_si_matches_the_difference_with_no_divergence_treatment() {
    check("si_hse06", ExxDivTreatment::None, STEP);
}

#[test]
#[ignore = "over 180 s: 4 differences x 9 entries x 6 energies of 64 pairs"]
fn the_exchange_strain_of_si_matches_the_difference_with_gygi_baldereschi() {
    check("si_hse06", ExxDivTreatment::GygiBaldereschi, STEP);
}

#[test]
#[ignore = "over 180 s: one difference of nine entries over 64 pairs, then the controls"]
fn the_kernel_controls_bite_on_the_mesh() {
    controls_kernel("si_hse06", ExxDivTreatment::GygiBaldereschi, STEP);
}
