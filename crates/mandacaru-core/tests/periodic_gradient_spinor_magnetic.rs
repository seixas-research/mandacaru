//! CS-3 / F2b: the four-channel local term on a MAGNETIC spinor (not time-reversal symmetric), where
//! the magnetization fields `m_x, m_y, m_z` are live: a high-spin N atom (3 Bohr magnetons) with a Ne
//! atom next to it, in a box (`nne_ldafr`), with the spin-orbit coupling on.
//!
//! * the force of the state started along `z` equals the Richardson reconverged central difference of
//!   the same discretized spinor free energy, both step families, every coordinate of both atoms. The
//!   oracle energies are solved by the crate's SCF (`periodic_gradient_spinor_freeze.rs`), because
//!   the spin-orbit anisotropy (about 7e-7 Ha) tilts the converged moment 0.15 rad from its start:
//!   the transverse spin blocks carry 0.45 Bohr magnetons, where the earlier oracle (a state that had
//!   not finished rotating, stopped at a density residual of 1e-6) had 1e-4. Each wrong Pauli
//!   structure moves the force by more than 100 gates on this state;
//! * with the spin-orbit coupling off the problem is invariant under a global spin rotation: the state
//!   started along `x` converges to the same energy and the same force as the one started along `z`,
//!   and its converged moment is asserted to be transverse FIRST, so the comparison cannot be vacuous.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::periodic::gradient::gradient;
use periodic_gradient_crystals_support::gates::*;
use periodic_gradient_spinor_support::{
    altered::{SpinorAltered, Wrong},
    moment, Axis, Spinor,
};

const CASE: &str = "nne_ldafr";
/// The moment a converged state must keep, in Bohr magnetons, for the state to count as magnetic.
const MOMENT_FLOOR: f64 = 0.5;

struct Run {
    spinor: Spinor,
    result: mandacaru_core::periodic::PeriodicResult,
    forces: Vec<[f64; 3]>,
    moment: [f64; 3],
}

fn run(spinor: Spinor) -> Run {
    let start = std::time::Instant::now();
    let result = spinor.data.solve();
    let moment = moment(&spinor.data, &result);
    let cores = spinor.data.cores();
    let provider = spinor.provider(&cores);
    let forces = gradient(&provider, &spinor.data.state(&result), None)
        .unwrap()
        .forces();
    eprintln!(
        "{CASE}: {} iterations, moment {moment:?}, {:.0} s",
        result.history.len(),
        start.elapsed().as_secs_f64()
    );
    Run {
        result,
        forces,
        moment,
        spinor,
    }
}

/// The forces of `wrong` on the state of `run`, as a provider-level variant.
fn variant(run: &Run, wrong: Wrong) -> Vec<[f64; 3]> {
    let cores = run.spinor.data.cores();
    let altered = SpinorAltered::new(run.spinor.provider(&cores), wrong);
    gradient(&altered, &run.spinor.data.state(&run.result), None)
        .unwrap()
        .forces()
}

fn largest_gap(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| (0..3).map(move |i| (a[i] - b[i]).abs()))
        .fold(0f64, f64::max)
}

#[test]
fn the_magnetic_spinor_force_is_the_reconverged_finite_difference() {
    let z = run(Spinor::load(CASE));
    // FIRST: the state is magnetic and canted by the anisotropy: the transverse components of the
    // moment are above a floor (0.45 of 3 Bohr magnetons), not the 1e-4 of a state that stalled.
    let size = (z.moment[0].powi(2) + z.moment[1].powi(2) + z.moment[2].powi(2)).sqrt();
    assert!(
        size > MOMENT_FLOOR && z.moment[2] > 0.9 * size,
        "{:?}",
        z.moment
    );
    assert!(
        z.moment[0].abs() > 0.1 && z.moment[1].abs() > 0.1,
        "the transverse blocks must be populated: {:?}",
        z.moment
    );
    let family = second_family(CASE);
    let worst = compare(CASE, &z.spinor.data, Some(&family), &z.forces);
    eprintln!("{CASE}: worst miss {worst:.3} gates");
    // Controls: a wrong m_z, the magnetization fields dropped, the transverse blocks dropped, a
    // wrong Pauli channel or one spin block each move the force by far more than a gate.
    let data = &z.spinor.data;
    for (wrong, name) in [
        (Wrong::FlippedMz, "m_z of the other sign"),
        (Wrong::ChargeChannelOnly, "magnetization fields dropped"),
        (Wrong::OneSpinBlock, "alpha spin block only"),
        (Wrong::NoTransverse, "transverse blocks dropped"),
        (Wrong::XySwapped, "m_x and m_y exchanged"),
        (
            Wrong::PauliCycled,
            "magnetization fields permuted cyclically",
        ),
    ] {
        let gates = bite_families(data, Some(&family), &variant(&z, wrong));
        eprintln!("z, {name}: bites {gates:.2e} gates");
        assert!(gates > 100., "{name}: {gates:e}");
    }
}

#[test]
fn a_rotated_moment_has_the_same_force_and_the_transverse_blocks_matter() {
    let z = run(Spinor::rotated(CASE, Axis::Z, false));
    let x = run(Spinor::rotated(CASE, Axis::X, false));
    // FIRST: the rotated state is transverse, otherwise the comparison is vacuous.
    assert!(
        z.moment[2] > MOMENT_FLOOR && z.moment[0].abs() < 1e-6,
        "{:?}",
        z.moment
    );
    assert!(x.moment[0] > MOMENT_FLOOR, "{:?}", x.moment);
    assert!(
        x.moment[1].abs() < 1e-6 && x.moment[2].abs() < 1e-6,
        "{:?}",
        x.moment
    );
    assert!(
        (x.moment[0] - z.moment[2]).abs() < 1e-5,
        "the same total moment"
    );
    let energy_gap = (x.result.free_energy - z.result.free_energy).abs();
    let force_gap = largest_gap(&x.forces, &z.forces);
    eprintln!("x against z, no spin-orbit: free energy {energy_gap:.2e} Ha, force {force_gap:.2e} Ha/Bohr");
    assert!(energy_gap < 1e-9, "{energy_gap:e}");
    assert!(force_gap < PLAN_GATE, "{force_gap:e}");
    // Each wrong Pauli structure moves the x state's force away from the covariant one by more than
    // 100 gates; flipping m_z does nothing there (m_z = 0): the guard is asserted, not assumed.
    for (wrong, name) in [
        (Wrong::NoTransverse, "transverse blocks dropped"),
        (
            Wrong::XySwapped,
            "m_x and m_y fields exchanged (wrong Pauli channel)",
        ),
        (Wrong::ChargeChannelOnly, "magnetization fields dropped"),
        (Wrong::OneSpinBlock, "alpha spin block only"),
    ] {
        let gates = largest_gap(&variant(&x, wrong), &z.forces) / PLAN_GATE;
        eprintln!("x, {name}: bites {gates:.2e} gates");
        assert!(gates > 100., "{name}: {gates:e}");
    }
    let gates = largest_gap(&variant(&x, Wrong::FlippedMz), &z.forces) / PLAN_GATE;
    assert!(
        gates < 1.0,
        "m_z of the other sign must not move an x state: {gates:e}"
    );
}
