//! CS-3: the crystal force of a SPINOR (spin-orbit) LDA crystal on a real converged SCF state.
//!
//! * the zero-coupling limit equals the scalar engine on the same case;
//! * the force of the `lda-fr` Si cell equals the Richardson reconverged central difference of
//!   the same discretized spinor free energy (the frozen oracle of `freeze_periodic_gradients.py`).
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::periodic::gradient::gradient;
use periodic_gradient_crystals_support::{gates::*, provider, Data};
use periodic_gradient_spinor_support::{oracle::against_the_oracle, Spinor};

fn forces_of(spinor: &Spinor) -> (Vec<[f64; 3]>, f64) {
    let result = spinor.data.solve();
    let cores = spinor.data.cores();
    let provider = spinor.provider(&cores);
    let breakdown = gradient(&provider, &spinor.data.state(&result), None).unwrap();
    (breakdown.forces(), result.free_energy)
}

#[test]
fn the_zero_coupling_limit_is_the_scalar_force() {
    let scalar = Data::load("lih");
    let result = scalar.solve();
    let cores = scalar.cores();
    let reference = gradient(&provider(&scalar, &cores), &scalar.state(&result), None)
        .unwrap()
        .forces();
    let twin = Spinor::zero_soc_twin("lih");
    let (forces, free) = forces_of(&twin);
    let energy_gap = (free - result.free_energy).abs();
    let gap = forces
        .iter()
        .zip(&reference)
        .flat_map(|(a, b)| (0..3).map(move |i| (a[i] - b[i]).abs()))
        .fold(0f64, f64::max);
    eprintln!("LiH zero-coupling twin: energy gap {energy_gap:e}, force gap {gap:e} Ha/Bohr");
    assert!(gap < 1e-9, "{gap:e}");
}

/// The spinor force of a frozen case against its oracle, every coordinate. Returns the breakdown.
fn check(case: &'static str) -> mandacaru_core::periodic::gradient::Breakdown {
    let start = std::time::Instant::now();
    let spinor = Spinor::load(case);
    let result = spinor.data.solve();
    let scf_seconds = start.elapsed().as_secs_f64();
    let cores = spinor.data.cores();
    let provider = spinor.provider(&cores);
    let breakdown = gradient(&provider, &spinor.data.state(&result), None).unwrap();
    eprintln!(
        "{case}: SCF {scf_seconds:.1} s ({} iterations), engine {:.1} s",
        result.history.len(),
        start.elapsed().as_secs_f64() - scf_seconds
    );
    against_the_oracle(&spinor.data, result.free_energy, &breakdown.forces(), case);
    breakdown
}

/// `forces + scale * (spin-orbit share)`: the force with the spin-orbit term replaced by
/// `(1 - scale)` times itself (`scale = 1` drops it, `-1` doubles it with the opposite sign...).
fn with_spin_orbit_scaled(
    breakdown: &mandacaru_core::periodic::gradient::Breakdown,
    rows: f64,
    columns: f64,
) -> Vec<[f64; 3]> {
    // force = -gradient; the spin-orbit share of the gradient is rows + columns.
    let mut forces = breakdown.forces();
    for (a, f) in forces.iter_mut().enumerate() {
        for x in 0..3 {
            f[x] += rows * breakdown.spin_orbit_rows[a][x]
                + columns * breakdown.spin_orbit_columns[a][x];
        }
    }
    forces
}

fn sum_over_atoms(term: &[[f64; 3]]) -> f64 {
    (0..3)
        .map(|a| term.iter().map(|g| g[a]).sum::<f64>().abs())
        .fold(0., f64::max)
}

#[test]
fn si_lda_fr_force_is_the_reconverged_finite_difference() {
    let breakdown = check("si_ldafr");
    let data = Data::load("si_ldafr");
    // The spin-orbit term is not vacuous: it is the difference to the scalar-relativistic force.
    let so_total: Vec<[f64; 3]> = breakdown
        .spin_orbit_rows
        .iter()
        .zip(&breakdown.spin_orbit_columns)
        .map(|(r, c)| [r[0] + c[0], r[1] + c[1], r[2] + c[2]])
        .collect();
    let peak = so_total.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    eprintln!(
        "spin-orbit share of the gradient: peak {peak:.3e} Ha/Bohr = {:.1} plan gates; rows {:?}, columns {:?}",
        peak / PLAN_GATE,
        breakdown.spin_orbit_rows,
        breakdown.spin_orbit_columns
    );
    // The spin-orbit force on Si is a fraction of a plan gate (the rows and the columns cancel to
    // 3e-8 Ha/Bohr): the Si oracle cannot resolve the term, so NO control on it is asserted here.
    // The heavy case that resolves it (AuH, controls biting by thousands of gates) is
    // `periodic_gradient_spinor_controls.rs`.
    for (name, rows, columns) in [
        ("spin-orbit term dropped", 1.0, 1.0),
        ("spin-orbit projector columns dropped", 0.0, 1.0),
        ("spin-orbit basis rows dropped", 1.0, 0.0),
    ] {
        let b = bite(&data, &with_spin_orbit_scaled(&breakdown, rows, columns));
        eprintln!("{name}: moves the Si force by {b:.2} gates (reported)");
    }
    assert!(
        peak < PLAN_GATE,
        "the Si spin-orbit force is below one plan gate"
    );
    // Mesh-free sum rules (the dense lattice sums and the sphere rules). The spin-orbit rows and
    // columns cancel by construction of the ownership scatter: structural, not physical.
    for (name, term) in [
        ("dense", &breakdown.dense),
        ("constants", &breakdown.constants),
        ("short_range", &breakdown.short_range),
    ] {
        let s = sum_over_atoms(term);
        eprintln!("sum rule {name}: {s:.2e}");
        assert!(s < 1e-12, "{name}: {s:e}");
    }
    let projections: Vec<[f64; 3]> = breakdown
        .projection_rows
        .iter()
        .zip(&breakdown.projection_columns)
        .map(|(r, c)| [r[0] + c[0], r[1] + c[1], r[2] + c[2]])
        .collect();
    assert!(sum_over_atoms(&projections) < 1e-12);
    assert!(sum_over_atoms(&so_total) < 1e-12, "structural");
    // Mesh-bound terms carry the egg-box ripple: REPORTED.
    eprintln!(
        "egg-box (sum over atoms): overlap {:.2e}, kinetic {:.2e}, local {:.2e}, hats {:.2e}, ion {:.2e}, core {:.2e}; net force {:.2e} Ha/Bohr",
        sum_over_atoms(&breakdown.overlap),
        sum_over_atoms(&breakdown.kinetic),
        sum_over_atoms(&breakdown.local),
        sum_over_atoms(&breakdown.hats),
        sum_over_atoms(&breakdown.ion),
        sum_over_atoms(&breakdown.core),
        sum_over_atoms(&breakdown.gradient()),
    );
}

/// The spinor GGA: PBE rotated to the local spin eigenframe (`pbe-fr` datasets). Its engine
/// differentiates the same fields; the partial core sees `0.5 (v_up + v_dn)` of the GGA potential.
#[test]
fn si_pbe_fr_force_is_the_reconverged_finite_difference() {
    check("si_pbefr");
}
