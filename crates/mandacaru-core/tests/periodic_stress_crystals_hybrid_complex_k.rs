//! AS-10 companion: the SCALAR hybrid stress at complex pair wave vectors. The displaced Si cell
//! with HSE06 (`pbe-sr` datasets, no spin-orbit coupling) on the whole 3x1x1 Gamma-centered mesh
//! (the points 0 and +-1/3, so `k_diff = +-1/3` is not time-reversal invariant) against the
//! Richardson-reconverged free energy, the scalar twin of `si_hsefr_tr3`. AS-9 gated the hybrid on
//! Gamma and on a mesh of time-reversal-invariant points only; this is the case that tells a defect
//! of the hybrid at complex k from one of the spinor frame.
//!
//! Run: `MANDACARU_PAW_PATH=... cargo test --release -p mandacaru-core --test
//! periodic_stress_crystals_hybrid_complex_k -- --test-threads=1`.
mod periodic_gradient_crystals_support;
mod periodic_stress_crystals_hybrid_support;
mod periodic_stress_crystals_support;

use periodic_stress_crystals_hybrid_support::run_enveloped;
use periodic_stress_crystals_support::run::stress_at;

/// Measured 2.86 and 3.8 times the plan's gate on the shears (see the spinor twin).
const ENVELOPE: f64 = 4.5;

#[test]
fn the_scalar_hybrid_stress_at_complex_k_is_within_the_measured_envelope() {
    run_enveloped(
        "si_hse06_tr3",
        &|d, s, o| stress_at(d, s, o),
        ENVELOPE,
        &[],
        &["_b"],
    );
}
