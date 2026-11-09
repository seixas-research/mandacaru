//! AS-10, the MAGNETIC spinor hybrid: the H2 triplet in a box with HSE06, as a non-collinear
//! spinor (`time_reversal = false`, four live field channels, the sphere remainder of SH-2 in the
//! one-center operator) with a zero spin-orbit matrix, against the Richardson-reconverged free
//! energy of the COLLINEAR spin-polarized crystal of the same discretized functional
//! (`freeze_periodic_strain.py case h2_hse06`). With a zero coupling the energy is invariant under
//! a global spin rotation, so the unrestricted derivative is the oracle of every moment direction
//! (all nine entries, unsymmetrized). The converged moment is asserted FIRST: it survives, it
//! points along the start, and its transverse components are populated.
//!
//! The same state started along z and along (1, 2, 2)/3 must give the same stress (SU(2)). A
//! magnetic spinor hybrid with a LIVE spin-orbit coupling has no real-crystal gate (the extension
//! refuses to build its snapshot): see the log.
//! Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_crystals_hybrid_support;
mod periodic_stress_crystals_support;
mod periodic_stress_spinor_hybrid_magnetic_support;
mod periodic_stress_spinor_support;

use periodic_gradient_spinor_support::{assert_direction, moment, Spinor};
use periodic_stress_crystals_hybrid_support::exchange_controls;
use periodic_stress_crystals_support::{
    assert_pins,
    gates::entries,
    run::{report, SCF_DENSITY_TOLERANCE, SCF_TOLERANCE},
};
use periodic_stress_spinor_hybrid_magnetic_support::magnetic_twin;
use periodic_stress_spinor_support::{miss, peak, stress_at};

struct Run {
    free_energy: f64,
    stress: mandacaru_core::periodic::stress_terms::Stress,
    moment: [f64; 3],
}

fn run(direction: [f64; 3]) -> (Spinor, Run) {
    let mut sp = magnetic_twin("h2_hse06", direction);
    assert_pins(&sp.data);
    assert!(sp.data.crystal.spinor && !sp.data.crystal.time_reversal);
    sp.data.options.tolerance = SCF_TOLERANCE;
    sp.data.options.density_tolerance = SCF_DENSITY_TOLERANCE;
    let result = sp.data.crystal.solve(&sp.data.options).unwrap();
    assert!(result.converged, "the SCF must converge");
    let m = moment(&sp.data, &result);
    println!(
        "  start {direction:?}: moment {m:?}, {} iterations, free energy {:.12}",
        result.history.len(),
        result.free_energy
    );
    let state = sp.data.state(&result);
    let stress = stress_at(&sp, &state, &sp.data.options);
    (
        sp,
        Run {
            free_energy: result.free_energy,
            stress,
            moment: m,
        },
    )
}

#[test]
fn the_magnetic_spinor_hybrid_stress_matches_the_collinear_reconverged_free_energy() {
    let direction = [1., 2., 2.];
    let (sp, r) = run(direction);
    // The state is magnetic and non-collinear: the converged moment points along the start and
    // every component is populated (the cross-spin blocks of D are large).
    assert_direction(r.moment, direction, 0.5, 0.2, 1e-3);
    let frozen = sp.data.manifest["free_energy_ha"].as_f64().unwrap();
    println!(
        "  free energy minus the unrestricted one: {:.2e}",
        r.free_energy - frozen
    );
    assert!((r.free_energy - frozen).abs() < 1e-9);
    let volume = sp.data.manifest["volume_bohr3"].as_f64().unwrap();
    assert!((r.stress.volume - volume).abs() < 1e-6 * volume);
    let want = entries(&sp.data.manifest, volume);
    let worst = report(
        "h2_hse06 (magnetic spinor)",
        &r.stress.gradient,
        &want,
        volume,
    );
    assert!(
        worst < 1.,
        "the worst entry misses its gate by {worst:.2} x"
    );
    for (name, term) in r.stress.breakdown.terms() {
        println!("  term {name}: peak {:.3e} Ha", peak(term));
    }
    // Each exchange term is needed; the sphere term carries the non-collinear operator.
    let weak = exchange_controls(&r.stress, &want);
    println!("  exchange terms below 100 gates when removed: {weak:?}");
    for (name, bite) in &weak {
        assert!(*bite >= 10., "{name} bites only {bite:.1} gates");
    }
}

#[test]
#[ignore = "slow: 12 s (measured 2026-11-08); run: cargo test --profile release-test -p mandacaru-core --test periodic_stress_spinor_hybrid_magnetic -- --ignored the_magnetic_spinor_hybrid_stress_does_not_depend_on_the_direction_of_the_moment"]
fn the_magnetic_spinor_hybrid_stress_does_not_depend_on_the_direction_of_the_moment() {
    let (_, along_z) = run([0., 0., 1.]);
    let (_, generic) = run([1., 2., 2.]);
    assert!(
        (along_z.free_energy - generic.free_energy).abs() < 1e-9,
        "the free energy is invariant under a global spin rotation"
    );
    assert!(
        along_z.moment[2].abs() > 0.5
            && generic.moment[0].abs() > 0.1
            && generic.moment[1].abs() > 0.1,
        "the two states point along different axes"
    );
    let gap =
        miss(&along_z.stress.gradient, &generic.stress.gradient) / peak(&generic.stress.gradient);
    println!("  stress, z start against (1, 2, 2) start: {gap:.2e} of the peak");
    assert!(gap < 1e-8, "{gap:e}");
}
