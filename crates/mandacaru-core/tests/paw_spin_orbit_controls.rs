//! Negative control of the spin-orbit construction: the recovery of each `j` from the stored
//! average and coupling holds to round-off for the right weights and fails by order one for the
//! swapped ones.
mod paw_spin_orbit_support;

use mandacaru_core::paw::generate::spin_orbit::{branches, combine, weights};
use paw_spin_orbit_support::items::Owned;
use paw_spin_orbit_support::{context, open};

const NAME: &str = "lda-fr_Na";

/// The recovery identity (`V_j = V_avg + V_SO (L.S)_j`) holds to round-off for the weights
/// `(2j+1)` and fails by order one for the swapped ones: the control bites by far more than 100
/// times the 1e-14 gate.
#[test]
fn wrong_weights_break_the_recovery_of_each_j() {
    use mandacaru_core::paw::generate::spin_orbit::recover;
    use nalgebra::DMatrix;
    let ctx = context();
    let case = open(&ctx, NAME);
    let generated = case.generate(&ctx);
    let owned = Owned::of(&case, &generated);
    let inputs = owned.inputs(&case, &generated);
    let l = 1;
    let [down, up] = branches(&ctx, &inputs, l).unwrap();
    let count = down.projectors.len();
    let (down_ionic, up_ionic) = (down.ionic.clone(), up.ionic.clone());
    let matrix = |a: &mandacaru_core::paw::record::Array| {
        DMatrix::from_row_slice(a.shape()[0], a.shape()[1], a.data())
    };
    let miss = |weights: [f64; 2], down: &DMatrix<f64>, up: &DMatrix<f64>| {
        let so = combine(&ctx, l, branches(&ctx, &inputs, l).unwrap(), weights).unwrap();
        let (average, coupling) = (matrix(&so.average), matrix(&so.coupling));
        let scale = down.amax().max(up.amax());
        let low = (recover(l, &average, &coupling, false, count) - down).amax() / scale;
        let high = (recover(l, &average, &coupling, true, count) - up).amax() / scale;
        low.max(high)
    };
    let right = miss(weights(l), &down_ionic, &up_ionic);
    let [lower, upper] = weights(l);
    let wrong = miss([upper, lower], &down_ionic, &up_ionic);
    println!("recovery miss: right weights {right:.2e}, swapped {wrong:.2e}");
    assert!(right < 1e-14);
    assert!(
        wrong > 100. * 1e-14,
        "swapped weights missed by only {wrong:e}"
    );
}
