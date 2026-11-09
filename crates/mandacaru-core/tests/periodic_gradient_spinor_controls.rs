//! CS-3 on a HEAVY molecule: AuH in a box (`auh_ldafr`, closed shell, time-reversal symmetric), where
//! the spin-orbit force is large enough to be resolved (on Si it is a fraction of a plan gate).
//!
//! * the converged force equals the Richardson reconverged central difference of the same
//!   discretized spinor free energy on the frozen (transverse) coordinates;
//! * the spin-orbit term is not vacuous: damaging it in the breakdown (dropped, sign flipped, rows
//!   only, projector columns only) moves the force by more than 100 gates;
//! * each provider-level variant (a whole gradient with one thing wrong: kinetic replaced by
//!   overlap, core sign, `W` with the reference kept, the owners of the rows or of the projectors,
//!   one spin block only, `W` weighted by the spectrum without spin-orbit) misses by more than
//!   100 gates;
//! * the spin-orbit force is linear in the coupling: the zero-coupling crystal has none.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::periodic::gradient::{gradient, Breakdown};
use periodic_gradient_crystals_support::{
    altered::{Altered, Wrong as Scalar},
    gates::*,
};
use periodic_gradient_spinor_support::{
    altered::{SpinorAltered, Wrong},
    oracle::against_the_oracle,
    Axis, Spinor,
};

const CASE: &str = "auh_ldafr";

fn with_spin_orbit_scaled(b: &Breakdown, rows: f64, columns: f64) -> Vec<[f64; 3]> {
    // force = -gradient, and the spin-orbit share of the gradient is rows + columns.
    let mut forces = b.forces();
    for (a, f) in forces.iter_mut().enumerate() {
        for x in 0..3 {
            f[x] += rows * b.spin_orbit_rows[a][x] + columns * b.spin_orbit_columns[a][x];
        }
    }
    forces
}

#[test]
#[ignore = "slow: 42 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_gradient_spinor_controls -- --ignored the_heavy_spinor_force_and_its_spin_orbit_term_are_resolved"]
fn the_heavy_spinor_force_and_its_spin_orbit_term_are_resolved() {
    let spinor = Spinor::load(CASE);
    let result = spinor.data.solve();
    let state = spinor.data.state(&result);
    let cores = spinor.data.cores();
    let breakdown = gradient(&spinor.provider(&cores), &state, None).unwrap();
    against_the_oracle(&spinor.data, result.free_energy, &breakdown.forces(), CASE);
    let so: Vec<[f64; 3]> = breakdown
        .spin_orbit_rows
        .iter()
        .zip(&breakdown.spin_orbit_columns)
        .map(|(r, c)| std::array::from_fn(|a| r[a] + c[a]))
        .collect();
    let peak = so.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    eprintln!(
        "spin-orbit share of the gradient: peak {peak:.3e} Ha/Bohr = {:.0} plan gates; rows {:?}, columns {:?}",
        peak / PLAN_GATE,
        breakdown.spin_orbit_rows,
        breakdown.spin_orbit_columns
    );
    // Free controls: the breakdown with the spin-orbit term damaged, against the oracle. The
    // whole term is 219 plan gates, so dropping or flipping it bites 100 gates; either of its two
    // shares alone is smaller (the projector columns are 15 gates on these coordinates) and is a
    // named report that must still exceed 10 gates, the same shape as the hat and phase controls
    // of the scalar and hybrid forces. The gate was not tightened to manufacture a bite.
    for (name, rows, columns, floor) in [
        ("spin-orbit term dropped", 1.0, 1.0, 100.),
        ("spin-orbit sign flipped", 2.0, 2.0, 100.),
        (
            "spin-orbit projector columns dropped (reported)",
            0.0,
            1.0,
            10.,
        ),
        ("spin-orbit basis rows dropped (reported)", 1.0, 0.0, 10.),
    ] {
        let gates = bite(
            &spinor.data,
            &with_spin_orbit_scaled(&breakdown, rows, columns),
        );
        eprintln!("{name}: bites {gates:.2e} gates");
        assert!(gates > floor, "{name}: {gates:e}");
    }
    // Provider-level variants: a whole engine run with ONE thing wrong.
    let mut rows: Vec<(&str, f64)> = Vec::new();
    for (wrong, name) in [
        (Scalar::Kinetic, "kinetic replaced by overlap"),
        (Scalar::CoreSign, "core sign flipped"),
        (Scalar::ReferenceKept, "W with the reference kept"),
        (Scalar::Ownership, "rows owned by the wrong atom"),
        (Scalar::ProjectorOwner, "projectors owned by the wrong atom"),
    ] {
        let altered = Altered::new(spinor.provider(&cores), wrong);
        let forces = gradient(&altered, &state, None).unwrap().forces();
        rows.push((name, bite(&spinor.data, &forces)));
    }
    let one = SpinorAltered::new(spinor.provider(&cores), Wrong::OneSpinBlock);
    rows.push((
        "alpha spin block only",
        bite(
            &spinor.data,
            &gradient(&one, &state, None).unwrap().forces(),
        ),
    ));
    let without = Spinor::rotated(CASE, Axis::Z, false);
    let scalar = SpinorAltered::scalar_eigenvalues(
        spinor.provider(&cores),
        &spinor.data.crystal,
        &without.data.crystal,
        spinor.data.options.clone(),
    );
    rows.push((
        "W from the eigenvalues without spin-orbit",
        bite(
            &spinor.data,
            &gradient(&scalar, &state, None).unwrap().forces(),
        ),
    ));
    for (name, gates) in &rows {
        eprintln!("{name}: bites {gates:.3e} gates");
    }
    for (name, gates) in rows {
        assert!(gates > 100., "{name}: {gates:e}");
    }
}

/// The spin-orbit force is linear in the coupling: the crystal without the coupling carries none,
/// and its engine force is the force of the same state with the term removed.
#[test]
fn without_the_coupling_there_is_no_spin_orbit_force() {
    let without = Spinor::rotated(CASE, Axis::Z, false);
    let result = without.data.solve();
    let cores = without.data.cores();
    let breakdown = gradient(
        &without.provider(&cores),
        &without.data.state(&result),
        None,
    )
    .unwrap();
    let peak = breakdown
        .spin_orbit_rows
        .iter()
        .chain(&breakdown.spin_orbit_columns)
        .flatten()
        .fold(0f64, |m, v| m.max(v.abs()));
    eprintln!("zero coupling: spin-orbit term {peak:.2e}");
    assert!(peak < 1e-14, "{peak:e}");
}
