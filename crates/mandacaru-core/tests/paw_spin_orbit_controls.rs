//! Negative controls of the spin-orbit construction. Each rebuilds the block wrongly from the
//! public pieces of `paw::generate::spin_orbit` and compares it with the Dirac freeze through
//! the same items as the parity test; every wrong variant must miss by more than 100 times the
//! gate of its worst item (never "nonzero"). The right construction sits below its gate, and a
//! field the construction does not read moves nothing.
mod paw_assemble_support;
mod paw_generate_support;
mod paw_spin_orbit_support;

use mandacaru_core::paw::generate::reference::{branch_equation, equation};
use mandacaru_core::paw::generate::spin_orbit::{
    branch, branches, channels, combine, kappas, weights, Inputs,
};
use mandacaru_core::paw::record::SpinOrbit;
use paw_generate_support::Decimated;
use paw_spin_orbit_support::items::{spin_orbit_items, Owned};
use paw_spin_orbit_support::{context, open, Case};
use std::collections::BTreeMap;

const NAME: &str = "lda-fr_Na";

/// The block `build` makes of each channel `l >= 1` from its two branches.
fn block(
    inputs: &Inputs<'_>,
    build: &dyn Fn(usize) -> mandacaru_core::Result<SpinOrbit>,
) -> BTreeMap<u32, SpinOrbit> {
    inputs
        .radii
        .keys()
        .filter(|&&l| l >= 1)
        .map(|&l| (l as u32, build(l).unwrap()))
        .collect()
}

/// The ratio of the worst item of `spin_orbit` to its gate.
fn worst(case: &Case, data: &Decimated, spin_orbit: &BTreeMap<u32, SpinOrbit>, title: &str) -> f64 {
    let comparison = spin_orbit_items(case, data, spin_orbit);
    let item = comparison.worst();
    println!(
        "{title}: worst {} {:.3e} (gate {:.1e}, ratio {:.3e})",
        item.name,
        item.deviation,
        item.gate,
        item.ratio()
    );
    item.ratio()
}

#[test]
fn a_wrong_spin_orbit_block_misses_the_freeze_by_far_more_than_its_gate() {
    let ctx = context();
    let case = open(&ctx, NAME);
    let data = Decimated::open(NAME);
    let generated = case.generate(&ctx);
    let owned = Owned::of(&case, &generated);
    let inputs = owned.inputs(&case, &generated);

    let right = block(&inputs, &|l| {
        combine(&ctx, l, branches(&ctx, &inputs, l)?, weights(l))
    });
    let ratio = worst(&case, &data, &right, "the construction");
    assert!(ratio < 1., "the right block must pass: {ratio}");
    // The production entry point is the same block.
    let production = channels(&ctx, &inputs).unwrap();
    assert!(worst(&case, &data, &production, "channels") < 1.);

    let none = BTreeMap::new();
    let swapped = block(&inputs, &|l| {
        // kappa swapped: the j = l - 1/2 radial equation labelled as the upper branch.
        let [down, up] = kappas(l);
        let one = |kappa: i32| {
            branch(&ctx, &inputs, l, &|ll| {
                branch_equation(
                    inputs.recipe,
                    if ll == l { kappa } else { -(ll as i32 + 1) },
                )
            })
        };
        combine(&ctx, l, [one(up)?, one(down)?], weights(l))
    });
    let scalar = block(&inputs, &|l| {
        // The scalar (j-averaged) waves standing in for both j branches.
        let one = || branch(&ctx, &inputs, l, &|_| equation(inputs.recipe));
        combine(&ctx, l, [one()?, one()?], weights(l))
    });
    let weighted = block(&inputs, &|l| {
        let [lower, upper] = weights(l);
        combine(&ctx, l, branches(&ctx, &inputs, l)?, [upper, lower])
    });
    for (title, variant) in [
        ("no spin-orbit channel", &none),
        ("swapped kappa", &swapped),
        ("scalar stand-in for the j average", &scalar),
        ("wrong (2j+1) weights", &weighted),
    ] {
        let ratio = worst(&case, &data, variant, title);
        assert!(ratio > 100., "{title}: only {ratio:.3e} times its gate");
    }
}

/// A recipe field the construction does not read (the radii are pinned by the scalar
/// construction, so the rule factor is not consulted) moves nothing.
#[test]
fn a_field_the_construction_does_not_read_moves_nothing() {
    let ctx = context();
    let case = open(&ctx, NAME);
    let data = Decimated::open(NAME);
    let generated = case.generate(&ctx);
    let owned = Owned::of(&case, &generated);
    let mut other = case.recipe.clone();
    other.rc_factor += 1.;
    let mut inputs = owned.inputs(&case, &generated);
    inputs.recipe = &other;
    let moved = channels(&ctx, &inputs).unwrap();
    let ratio = worst(&case, &data, &moved, "rc_factor + 1");
    assert!(ratio < 1., "an unread field moved the block: {ratio}");
}

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
