//! AS-10: the exchange strain of a SPINOR hybrid crystal (H-1..H-6 on `2m` rows) against central
//! differences of the production exchange energy at FIXED density matrices (no SCF), on the
//! triclinic LiH box at Gamma rewritten as a spinor with a zero spin-orbit matrix and flagged
//! magnetic, with hand-built `2m x 2m` density matrices whose cross-spin blocks are populated.
//! The groups, the step and the gates are those of `periodic_stress_hybrid` (all nine entries,
//! unsymmetrized); the sphere term is gated with the full spinor operator of the one-center model
//! (the cross blocks and the non-collinear remainder of SH-2 included).
//! Needs `MANDACARU_PAW_PATH`. SU(2) covariance and the controls are `periodic_stress_spinor_hybrid_controls`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_hybrid_support;
mod periodic_stress_spinor_hybrid_support;

use mandacaru_core::periodic::screened_exchange::ExxDivTreatment;
use periodic_stress_hybrid_support::check_rig;
use periodic_stress_spinor_hybrid_support::rig;

const STEP: f64 = 2.5e-6;

#[test]
fn the_spinor_exchange_strain_of_lih_matches_the_difference_with_no_divergence_treatment() {
    let r = rig("lih_hse06", ExxDivTreatment::None, false);
    check_rig(r, "lih_hse06 spinor", STEP);
}

#[test]
fn the_spinor_exchange_strain_of_lih_matches_the_difference_with_gygi_baldereschi() {
    let r = rig("lih_hse06", ExxDivTreatment::GygiBaldereschi, false);
    check_rig(r, "lih_hse06 spinor", STEP);
}
