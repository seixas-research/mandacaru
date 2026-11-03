//! AS-10 on a k MESH with complex wave vectors: the exchange strain of the displaced Si cell with
//! HSE06 and spin-orbit coupling on the whole 3x1x1 Gamma-centered mesh (the points 0 and +-1/3,
//! so the pair wave vectors `k_diff` are complex, not time-reversal invariant) against central
//! differences of the production exchange energy at FIXED hand-built `2m x 2m` densities (the
//! spinor flagged magnetic, so the hand-built magnetization is admitted), no SCF. The groups, the
//! step and the gates are those of `periodic_stress_hybrid`.
//!
//! Slow (all four groups, 9 entries, 5 energies each): `#[ignore]`d; run
//! `cargo test --release -p mandacaru-core --test periodic_stress_spinor_hybrid_mesh -- --ignored
//! --test-threads=1`. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_crystals_support;
mod periodic_stress_hybrid_support;
mod periodic_stress_spinor_hybrid_mesh_support;
mod periodic_stress_spinor_hybrid_support;

use mandacaru_core::periodic::screened_exchange::ExxDivTreatment;
use periodic_stress_hybrid_support::check_rig;
use periodic_stress_spinor_hybrid_mesh_support::mesh_rig;

const STEP: f64 = 2.5e-6;

#[test]
#[ignore = "minutes: 3 k-points of 2m rows, four groups of 9 entries; see the module doc"]
fn the_spinor_exchange_strain_on_the_complex_k_mesh_matches_the_difference() {
    let rig = mesh_rig("si_hsefr_tr3", ExxDivTreatment::None, false);
    check_rig(rig, "si_hsefr_tr3 spinor mesh", STEP);
}
