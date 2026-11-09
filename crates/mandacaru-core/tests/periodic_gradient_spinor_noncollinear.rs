//! F2b: the force of a MAGNETIC spinor crystal (not time-reversal symmetric, four live Pauli
//! channels, the eigenframe functional) whose moment points along a GENERIC direction, so the
//! transverse spin blocks carry most of the magnetization: the BeH radical (LDA, PBE) as a spinor
//! with the spin-orbit coupling off.
//!
//! With the coupling off the energy is invariant under a global spin rotation, so the converged
//! state is the scalar unrestricted one with its moment turned to the start direction and its force
//! is the scalar oracle's (the frozen reconverged Richardson differences of `beh_lda` / `beh_pbe`
//! with both step families, every coordinate of both atoms). The oracle never saw a spinor: a
//! wrong Pauli structure of the engine cannot hide in a shared convention.
//!
//! * the SCF reaches the density tolerance 1e-9 of the oracle on a magnetic spinor (the stall of the
//!   magnetization direction in the eigenframe is what this package removed);
//! * the converged moment is asserted FIRST: its norm, its direction (the start's, to rounding) and
//!   that every Cartesian component is above a floor, so the comparison cannot be vacuous;
//! * the force equals the oracle for the moment along `z`, along `x` and along `(1, 2, 2) / 3`;
//! * negative controls on the generic state, each against the oracle: the transverse blocks
//!   dropped, `m_x` and `m_y` exchanged, the three magnetization fields permuted cyclically, `m_z`
//!   flipped, the magnetization dropped, one spin block only.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::periodic::{gradient::gradient, PeriodicResult};
use periodic_gradient_crystals_support::{gates::*, provider, Data};
use periodic_gradient_spinor_support::{
    altered::{SpinorAltered, Wrong},
    assert_direction, largest_gap, moment, Spinor,
};

const GENERIC: [f64; 3] = [1., 2., 2.];

struct Run {
    spinor: Spinor,
    result: PeriodicResult,
    forces: Vec<[f64; 3]>,
    moment: [f64; 3],
}

fn run(case: &'static str, direction: [f64; 3]) -> Run {
    let start = std::time::Instant::now();
    let spinor = Spinor::magnetic_twin(case, direction);
    let result = spinor.data.solve();
    let moment = moment(&spinor.data, &result);
    let cores = spinor.data.cores();
    let forces = gradient(&spinor.provider(&cores), &spinor.data.state(&result), None)
        .unwrap()
        .forces();
    eprintln!(
        "{case} {direction:?}: {} iterations, moment {moment:.6?}, free energy {:.10}, {:.0} s",
        result.history.len(),
        result.free_energy,
        start.elapsed().as_secs_f64()
    );
    Run {
        spinor,
        result,
        forces,
        moment,
    }
}

fn assert_moment(run: &Run, direction: [f64; 3], floor: f64) {
    assert_direction(run.moment, direction, 0.5, floor, 1e-6);
}

fn variant(run: &Run, wrong: Wrong) -> Vec<[f64; 3]> {
    let cores = run.spinor.data.cores();
    let altered = SpinorAltered::new(run.spinor.provider(&cores), wrong);
    gradient(&altered, &run.spinor.data.state(&run.result), None)
        .unwrap()
        .forces()
}

fn against_the_oracle(case: &'static str, run: &Run) -> f64 {
    let family = second_family(case);
    compare(case, &run.spinor.data, Some(&family), &run.forces)
}

#[test]
#[ignore = "slow: 13 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_spinor_noncollinear -- --ignored beh_lda_generic_moment_force_is_the_collinear_oracle"]
fn beh_lda_generic_moment_force_is_the_collinear_oracle() {
    let case = "beh_lda";
    let generic = run(case, GENERIC);
    // FIRST: a state with all three components of the moment live (1/3, 2/3, 2/3 of the norm).
    assert_moment(&generic, GENERIC, 0.3);
    let family = second_family(case);
    let worst = against_the_oracle(case, &generic);
    eprintln!("{case} generic: worst miss {worst:.3} gates");
    // The same problem started along z and along x: the same energy, the same force.
    for direction in [[0., 0., 1.], [1., 0., 0.]] {
        let other = run(case, direction);
        assert_moment(&other, direction, 0.9);
        let gap = largest_gap(&other.forces, &generic.forces);
        let energy = (other.result.free_energy - generic.result.free_energy).abs();
        eprintln!(
            "{case} {direction:?} against generic: free energy {energy:.2e}, force {gap:.2e}"
        );
        assert!(energy < 1e-9, "{energy:e}");
        assert!(gap < PLAN_GATE, "{gap:e}");
        against_the_oracle(case, &other);
    }
    // Each wrong Pauli structure misses the oracle by more than 100 gates of its coordinate.
    for (wrong, name) in [
        (Wrong::NoTransverse, "transverse blocks dropped"),
        (Wrong::XySwapped, "m_x and m_y exchanged"),
        (
            Wrong::PauliCycled,
            "magnetization fields permuted cyclically",
        ),
        (Wrong::FlippedMz, "m_z of the other sign"),
        (Wrong::ChargeChannelOnly, "magnetization fields dropped"),
        (Wrong::OneSpinBlock, "alpha spin block only"),
    ] {
        let gates = bite_families(
            &generic.spinor.data,
            Some(&family),
            &variant(&generic, wrong),
        );
        eprintln!("{case} generic, {name}: bites {gates:.2e} gates");
        assert!(gates > 100., "{name}: {gates:e}");
    }
}

#[test]
fn beh_pbe_generic_moment_force_is_the_collinear_oracle() {
    let case = "beh_pbe";
    let generic = run(case, GENERIC);
    assert_moment(&generic, GENERIC, 0.3);
    let worst = against_the_oracle(case, &generic);
    eprintln!("{case} generic: worst miss {worst:.3} gates");
    let family = second_family(case);
    // The GGA gradient is taken on the eigenframe spin fields and rotated back: a wrong Pauli
    // structure of the gradient correction would show here.
    for (wrong, name) in [
        (Wrong::NoTransverse, "transverse blocks dropped"),
        (
            Wrong::PauliCycled,
            "magnetization fields permuted cyclically",
        ),
    ] {
        let gates = bite_families(
            &generic.spinor.data,
            Some(&family),
            &variant(&generic, wrong),
        );
        eprintln!("{case} generic, {name}: bites {gates:.2e} gates");
        assert!(gates > 100., "{name}: {gates:e}");
    }
}

/// The one-signed `m_z` limit: the spinor along `z` is the collinear spin-polarized crystal of F1.
/// Its force is the scalar unrestricted engine's on the same snapshot, term by term of the
/// result: the free energy and the force differ by rounding.
#[test]
fn the_collinear_limit_is_the_spin_polarized_force() {
    for case in ["beh_lda", "beh_pbe"] {
        let scalar = Data::load(case);
        let result = scalar.solve();
        let cores = scalar.cores();
        let reference = gradient(&provider(&scalar, &cores), &scalar.state(&result), None)
            .unwrap()
            .forces();
        let spinor = run(case, [0., 0., 1.]);
        let energy = (spinor.result.free_energy - result.free_energy).abs();
        let gap = largest_gap(&spinor.forces, &reference);
        eprintln!("{case}: spinor along z against the spin-polarized crystal: free energy {energy:.2e}, force {gap:.2e}");
        assert!(energy < 1e-9, "{case}: {energy:e}");
        assert!(gap < 1e-9, "{case}: {gap:e}");
        // The polarized force is itself far from the non-magnetic one: the comparison is not vacuous.
        assert!(spinor.moment[2] > 0.9, "{case}: {:?}", spinor.moment);
    }
}
