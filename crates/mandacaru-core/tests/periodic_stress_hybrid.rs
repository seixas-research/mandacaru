//! AS-9: the exchange strain of a hybrid crystal (H-1..H-6) against central differences of the
//! production exchange energy at FIXED density matrices (no SCF), on the triclinic LiH box at Gamma
//! with both divergence treatments. The crystal is strained by `x -> (1 + eps) x` in three groups
//! (the cell, the Bloch samples, the projections) whose derivatives add, so every term is gated
//! against a difference that moves only its own members, and the sum against the difference of all
//! three together. All nine entries `(a, b)`, unsymmetrized. Needs `MANDACARU_PAW_PATH`; the Si
//! cell on the whole 2x2x2 mesh is `periodic_stress_hybrid_mesh`.
mod periodic_gradient_crystals_support;
mod periodic_stress_hybrid_support;

use mandacaru_core::periodic::screened_exchange::ExxDivTreatment;
use periodic_stress_hybrid_support::check;

/// The central-difference step (a strain).
const STEP: f64 = 2.5e-6;

#[test]
fn the_exchange_strain_of_lih_matches_the_difference_with_no_divergence_treatment() {
    check("lih_hse06", ExxDivTreatment::None, STEP);
}

#[test]
fn the_exchange_strain_of_lih_matches_the_difference_with_gygi_baldereschi() {
    check("lih_hse06", ExxDivTreatment::GygiBaldereschi, STEP);
}
