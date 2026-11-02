//! The semilocal half of the hybrid on its own: the force of a PBE crystal (the `pbe-sr`
//! datasets, Si on the whole 2x2x2 mesh) against the reconverged free-energy finite difference
//! of the same discretized energy. The gradient correction enters the force through the xc
//! potential the local term and the partial core read; the LDA gates (`periodic_gradient_crystals.rs`)
//! do not exercise it. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;

use mandacaru_core::periodic::gradient::gradient;
use periodic_gradient_crystals_support::{gates::*, *};

#[test]
fn si_pbe_force_is_the_reconverged_finite_difference() {
    let data = Data::load("si_pbe");
    let result = data.solve();
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!((result.free_energy - reference).abs() < 1e-9);
    let cores = data.cores();
    let forces = gradient(&provider(&data, &cores), &data.state(&result), None)
        .unwrap()
        .forces();
    for entry in data.manifest["coordinates"].as_array().unwrap() {
        let (atom, axis) = (
            entry["atom"].as_u64().unwrap() as usize,
            entry["axis"].as_u64().unwrap() as usize,
        );
        let oracle = -entry["rr"].as_f64().unwrap();
        let floor = entry["floor_richardson"].as_f64().unwrap();
        let miss = (forces[atom][axis] - oracle).abs();
        eprintln!(
            "si_pbe atom {atom} axis {axis}: force {:+.9e}, oracle {oracle:+.9e}, miss {miss:.2e} Ha/Bohr, floor {:.1e}, gate {:.2e}",
            forces[atom][axis],
            floor.abs(),
            gate(floor)
        );
        assert!(miss < gate(floor), "atom {atom} axis {axis}: {miss:e}");
        assert!(oracle.abs() > 100. * gate(floor));
    }
}
