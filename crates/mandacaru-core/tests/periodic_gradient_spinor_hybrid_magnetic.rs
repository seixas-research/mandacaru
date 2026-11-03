//! F2b, HSE06: the force of a MAGNETIC spinor hybrid crystal (not time-reversal symmetric: four live
//! Pauli channels, the eigenframe functional on the grid, the non-collinear remainder in the
//! spheres' one-center operator) whose moment points along a generic direction: the H2 triplet in a
//! box as a spinor without spin-orbit coupling.
//!
//! With the coupling off the energy is invariant under a global spin rotation, so the converged
//! state is the scalar unrestricted hybrid with its moment turned, and its force is the scalar
//! oracle's (`h2_hse06` and its second step family: the frozen reconverged central differences of
//! the unrestricted energy, which carries the collinear spin-scaling remainder). The spinor reaches
//! the same numbers through the vector remainder and the full spin-structured `Z^oc`.
//!
//! * the SCF reaches 1e-9 (density, energy and exchange operator) on a magnetic spinor hybrid;
//! * the free energy and the hybrid energy tie equal the oracle's;
//! * the converged moment is asserted FIRST (norm, direction, every component above a floor);
//! * the force along `(1, 2, 2) / 3`, along `z` and along `x` equals the oracle on every
//!   coordinate of both atoms (two step families), and the three forces agree;
//! * negative controls on the generic state, each against the oracle: the transverse blocks of the
//!   local term dropped, `m_x` and `m_y` exchanged, the magnetization fields permuted cyclically,
//!   `m_z` flipped, the magnetization dropped, one spin block only.
//!
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;

use mandacaru_core::periodic::{
    gradient::{contract, freeze, gradient, scf},
    PeriodicResult,
};
use periodic_gradient_crystals_support::{gates::*, provider, Data};
use periodic_gradient_spinor_support::{
    altered::{SpinorAltered, Wrong},
    assert_direction, largest_gap, moment, norm, Spinor,
};

const CASE: &str = "h2_hse06";
const GENERIC: [f64; 3] = [1., 2., 2.];
const ENERGY_GATE: f64 = 1e-9;
const BAND_GATE: f64 = 1e-9;

struct Run {
    spinor: Spinor,
    result: PeriodicResult,
    forces: Vec<[f64; 3]>,
    moment: [f64; 3],
}

fn run(direction: [f64; 3]) -> Run {
    let start = std::time::Instant::now();
    let spinor = Spinor::magnetic_twin(CASE, direction);
    let data = &spinor.data;
    let c = &data.crystal;
    assert!(c.spinor && !c.time_reversal && c.spins == 1);
    let result = data.solve();
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (result.free_energy - reference).abs() < ENERGY_GATE,
        "{direction:?}: the SCF is not the oracle's: {:e}",
        result.free_energy - reference
    );
    let tie = scf::hybrid_state_energy(c, &result.matrices, data.options.exxdiv_treatment).unwrap()
        - result.energy;
    assert!(tie.abs() < 1e-12, "{direction:?}: energy tie {tie:e}");
    let moment = moment(data, &result);
    let cores = data.cores();
    let provider = spinor.provider(&cores);
    let frozen = freeze(&provider, &data.state(&result), None).unwrap();
    // The re-diagonalized OCCUPIED bands (the triplet has two occupied spinor bands) are the SCF's;
    // the empty ones are the eigenvalues of an operator on a vanishing minority density.
    let mut band_gap = 0f64;
    for (k, eps) in result.eigenvalues.iter().enumerate() {
        for (a, b) in eps
            .iter()
            .zip(&frozen.state.bands[0][k].eigenvalues)
            .take(2)
        {
            band_gap = band_gap.max((a - b).abs());
        }
    }
    assert!(band_gap < BAND_GATE, "{direction:?}: bands {band_gap:e}");
    let forces = contract(&provider, &frozen).unwrap().forces();
    eprintln!(
        "{CASE} {direction:?}: {} iterations, moment {moment:.6?}, free energy {:.10}, tie {tie:.1e}, bands {band_gap:.1e}, {:.0} s",
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

fn variant(run: &Run, wrong: Wrong) -> Vec<[f64; 3]> {
    let cores = run.spinor.data.cores();
    let altered = SpinorAltered::new(run.spinor.provider(&cores), wrong);
    let frozen = freeze(&altered, &run.spinor.data.state(&run.result), None).unwrap();
    contract(&altered, &frozen).unwrap().forces()
}

#[test]
fn h2_hse06_generic_moment_force_is_the_collinear_oracle() {
    let generic = run(GENERIC);
    // FIRST: a state with all three components of the moment live, norm 2 mu_B.
    assert_direction(generic.moment, GENERIC, 1.5, 0.3, 1e-6);
    let data = &generic.spinor.data;
    let family = second_family(CASE);
    let worst = compare(CASE, data, Some(&family), &generic.forces);
    eprintln!("{CASE} generic: worst miss {worst:.3} gates");
    // The same problem started along z and along x: the same energy and force.
    for direction in [[0., 0., 1.], [1., 0., 0.]] {
        let other = run(direction);
        assert_direction(other.moment, direction, 1.5, 0.9, 1e-6);
        let gap = largest_gap(&other.forces, &generic.forces);
        let energy = (other.result.free_energy - generic.result.free_energy).abs();
        eprintln!(
            "{CASE} {direction:?} against generic: free energy {energy:.2e}, force {gap:.2e}"
        );
        assert!(energy < ENERGY_GATE, "{energy:e}");
        assert!(gap < PLAN_GATE, "{gap:e}");
        compare(CASE, &other.spinor.data, Some(&family), &other.forces);
    }
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
        let gates = bite_families(data, Some(&family), &variant(&generic, wrong));
        eprintln!("{CASE} generic, {name}: bites {gates:.2e} gates");
        assert!(gates > 100., "{name}: {gates:e}");
    }
}

/// The spin-orbit hybrid on the four-field branch: the Si cell of F2a (`pbe-fr`, HSE06, the whole
/// 3x1x1 mesh with complex k), flagged magnetic with a zero moment. The moment stays zero, the
/// non-collinear remainder vanishes bitwise, and the free energy and the force are the
/// time-reversal crystal's: the oracle of `si_hsefr_tr3` (both step families, every coordinate).
/// This is the spin-orbit hybrid gate of the magnetic code path that does not need a magnetic
/// snapshot (the extension refuses to build one).
#[test]
fn a_magnetic_flagged_silicon_hybrid_with_spin_orbit_is_the_time_reversal_force() {
    let case = "si_hsefr_tr3";
    let start = std::time::Instant::now();
    let spinor = Spinor::magnetic_flagged(case);
    let data = &spinor.data;
    let c = &data.crystal;
    assert!(c.spinor && !c.time_reversal && c.points.len() == 3);
    let result = data.solve();
    let reference = data.manifest["free_energy_ha"].as_f64().unwrap();
    assert!(
        (result.free_energy - reference).abs() < ENERGY_GATE,
        "the SCF is not the oracle's: {:e}",
        result.free_energy - reference
    );
    let m = moment(data, &result);
    assert!(norm(m) < 1e-6, "the moment must stay zero: {m:?}");
    let tie = scf::hybrid_state_energy(c, &result.matrices, data.options.exxdiv_treatment).unwrap()
        - result.energy;
    assert!(tie.abs() < 1e-12, "energy tie {tie:e}");
    let cores = data.cores();
    let provider = spinor.provider(&cores);
    let frozen = freeze(&provider, &data.state(&result), None).unwrap();
    let forces = contract(&provider, &frozen).unwrap().forces();
    eprintln!(
        "{case} (magnetic flag): {} iterations, moment {m:?}, tie {tie:.1e}, {:.0} s",
        result.history.len(),
        start.elapsed().as_secs_f64()
    );
    let other = second_family(case);
    let worst = compare(case, data, Some(&other), &forces);
    eprintln!("{case} (magnetic flag): worst miss {worst:.3} gates");
}

/// The one-signed `m_z` limit of the hybrid: the spinor along `z` is the collinear unrestricted
/// HSE06 crystal of F1 (the H2 triplet, with the collinear spin-scaling remainder): the same
/// free energy and force to rounding.
#[test]
fn the_collinear_limit_is_the_spin_polarized_hybrid_force() {
    let scalar = Data::load(CASE);
    let result = scalar.solve();
    let cores = scalar.cores();
    let reference = gradient(&provider(&scalar, &cores), &scalar.state(&result), None)
        .unwrap()
        .forces();
    let spinor = run([0., 0., 1.]);
    let energy = (spinor.result.free_energy - result.free_energy).abs();
    let gap = largest_gap(&spinor.forces, &reference);
    eprintln!("{CASE}: spinor along z against the unrestricted hybrid: free energy {energy:.2e}, force {gap:.2e}");
    assert!(energy < ENERGY_GATE, "{energy:e}");
    assert!(gap < 1e-9, "{gap:e}");
}
