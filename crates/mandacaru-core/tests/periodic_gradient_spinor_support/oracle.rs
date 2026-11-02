//! The comparison of an engine force with the frozen reconverged finite-difference oracle of a case.
#![allow(dead_code)]

use crate::periodic_gradient_crystals_support::{gates::*, Data};

/// Asserts that the SCF reached the oracle's free energy and that `forces` match the oracle on
/// every frozen coordinate within its gate (14.4: about 100x the floor, never above the plan's
/// number), with the oracle itself far above the gate. Returns the worst miss in Ha/Bohr.
pub fn against_the_oracle(data: &Data, free_energy: f64, forces: &[[f64; 3]], label: &str) -> f64 {
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (free_energy - reference).abs() < 1e-9,
        "{label}: the SCF is not the oracle's: {:e}",
        free_energy - reference
    );
    let mut worst = 0f64;
    for entry in data.manifest["coordinates"].as_array().unwrap() {
        let (atom, axis) = (
            entry["atom"].as_u64().unwrap() as usize,
            entry["axis"].as_u64().unwrap() as usize,
        );
        let oracle = -entry["rr"].as_f64().unwrap();
        let floor = entry["floor_richardson"].as_f64().unwrap();
        let extrapolated = entry["floor_extrapolated"].as_f64().unwrap();
        let miss = (forces[atom][axis] - oracle).abs();
        eprintln!(
            "{label} atom {atom} axis {axis}: force {:+.9e}, oracle {oracle:+.9e}, miss {miss:.2e} Ha/Bohr = {:.2e} eV/A, floors {:.1e} / {:.1e}, gate {:.2e}",
            forces[atom][axis],
            miss * TO_EV_PER_A,
            floor.abs(),
            extrapolated.abs(),
            gate(floor)
        );
        assert!(
            miss < gate(floor),
            "{label} atom {atom} axis {axis}: {miss:e}"
        );
        assert!(
            oracle.abs() > 100. * gate(floor),
            "{label} atom {atom} axis {axis}: the oracle {oracle:e} is not above its gate"
        );
        worst = worst.max(miss);
    }
    worst
}
