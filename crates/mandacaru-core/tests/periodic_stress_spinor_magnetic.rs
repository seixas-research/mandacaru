//! AS-8: the four-channel local term, the transverse spin blocks and the eigenframe xc on a MAGNETIC
//! spinor (not time-reversal symmetric), where the magnetization fields `m_x, m_y, m_z` are live: a
//! high-spin N atom (3 Bohr magnetons) next to a Ne atom in a box (`nne_ldafr`, `nne_fr`).
//!
//! * with the spin-orbit coupling off the problem is invariant under a global spin rotation: the
//!   state started along `x` converges to the same free energy and the same stress as the one
//!   started along `z`, and its converged moment is asserted to be transverse FIRST, so the
//!   comparison cannot be vacuous (with the coupling on the two differ by the magnetic anisotropy);
//! * each wrong Pauli structure moves the stress of the `x` state away from the covariant `z` one
//!   by more than 100 gates: the transverse blocks dropped, the `m_x` and `m_y` fields exchanged,
//!   the magnetization fields dropped, the kinetic matrix on the alpha block only; flipping `m_z`
//!   does nothing to an `x` state (asserted below one gate: the guard that the controls are
//!   not vacuous) and a great deal to the `z` one;
//! * with the coupling on, the `z` state's stress is the Richardson-reconverged free energy of
//!   `nne_fr` (LDA) and `nne_pbefr` (PBE on the eigenframe fields; the magnetic SCF stalls above a
//!   density residual of 5e-7, so its density tolerance is 1e-6 and the stress, first order in the
//!   residual, carries that). The spin-orbit term of N and Ne is 2e-8 Ha, below the gates: the
//!   magnetic spin-orbit term is NOT resolved by this case (the closed-shell heavy cases resolve it).
//!
//! The antisymmetric part of the derivative is a physical torque of a magnetic spin-orbit state,
//! reported and never gated here. Needs `MANDACARU_PAW_PATH`.
mod periodic_gradient_crystals_support;
mod periodic_gradient_spinor_support;
mod periodic_stress_crystals_support;
mod periodic_stress_spinor_support;

use mandacaru_core::periodic::{
    gradient::{State, Term},
    setup::strain_provider::StrainFields,
    stress_terms::{contract, Stress, Tensor},
    PeriodicResult,
};
use periodic_gradient_spinor_support::{moment, Axis, Spinor};
use periodic_stress_crystals_support::{
    assert_pins,
    gates::{bites, entries},
    load, manifest as load_manifest,
    run::report,
    summary,
};
use periodic_stress_spinor_support::{
    altered::{Altered, Wrong},
    miss, peak, spinor_of, strain_of, stress_at,
};

const CASE: &str = "nne_ldafr";
const MOMENT_FLOOR: f64 = 0.5;
/// The covariance gate, relative to the peak entry: the x and z states measure 6e-12 apart, so 1e-9
/// is about 100x (the magnetic SCF stops at a density residual of 1e-6, but both starts stop alike).
const COVARIANCE_GATE: f64 = 1e-9;

struct Run {
    sp: Spinor,
    state: State,
    result: PeriodicResult,
    moment: [f64; 3],
    stress: Stress,
}

fn run(sp: Spinor) -> Run {
    let start = std::time::Instant::now();
    let result = sp.data.solve();
    let moment = moment(&sp.data, &result);
    let state = sp.data.state(&result);
    let stress = stress_at(&sp, &state, &sp.data.options);
    println!(
        "{}: {} iterations, moment {moment:?}, {:.0} s",
        sp.data.case,
        result.history.len(),
        start.elapsed().as_secs_f64()
    );
    Run {
        sp,
        state,
        result,
        moment,
        stress,
    }
}

/// The gradient of the stress with `damage` applied to the frozen state of `run`.
fn damaged(run: &Run, damage: impl FnOnce(&mut StrainFields)) -> Tensor<f64> {
    let cores = run.sp.data.cores();
    let mut c = strain_of(&run.sp, &cores, &run.state, &run.sp.data.options);
    damage(&mut c.frozen.fields);
    contract(&c.strain, &c.frozen).unwrap().gradient()
}

fn relative(a: &Tensor<f64>, b: &Tensor<f64>) -> f64 {
    miss(a, b) / peak(b)
}

#[test]
fn a_rotated_moment_has_the_same_stress_and_the_transverse_blocks_matter() {
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
    let gap = relative(&x.stress.gradient, &z.stress.gradient);
    println!(
        "x against z, no spin-orbit: free energy {energy_gap:.2e} Ha, stress {:.2e} Ha = {gap:.2e} of the peak {:.3e}",
        miss(&x.stress.gradient, &z.stress.gradient),
        peak(&z.stress.gradient)
    );
    assert!(energy_gap < 1e-9, "{energy_gap:e}");
    assert!(gap < COVARIANCE_GATE, "{gap:e}");
    // The spin-orbit term of a zero coupling is exactly zero.
    assert_eq!(peak(&x.stress.breakdown.spin_orbit), 0.);

    let gates = |wrong: &Tensor<f64>| {
        miss(wrong, &z.stress.gradient) / (COVARIANCE_GATE * peak(&z.stress.gradient))
    };
    let channels = |fields: &mut StrainFields, f: &dyn Fn(&mut Vec<Vec<f64>>)| f(&mut fields.local);
    let controls: [(&str, Tensor<f64>); 3] = [
        (
            "transverse blocks dropped",
            damaged(&x, |f| {
                channels(f, &|l| {
                    l[1].iter_mut().for_each(|v| *v = 0.);
                    l[2].iter_mut().for_each(|v| *v = 0.);
                })
            }),
        ),
        (
            "m_x and m_y fields exchanged (wrong Pauli channel)",
            damaged(&x, |f| channels(f, &|l| l.swap(1, 2))),
        ),
        (
            "magnetization fields dropped",
            damaged(&x, |f| {
                channels(f, &|l| {
                    l[1..]
                        .iter_mut()
                        .for_each(|c| c.iter_mut().for_each(|v| *v = 0.));
                })
            }),
        ),
    ];
    for (name, wrong) in controls {
        let bite = gates(&wrong);
        println!("x, {name}: bites {bite:.2e} gates");
        assert!(bite > 100., "{name}: {bite:.2e}");
    }
    // The kinetic matrix on the alpha block only (T not doubled).
    let cores = x.sp.data.cores();
    let c = strain_of(&x.sp, &cores, &x.state, &x.sp.data.options);
    let wrong = Altered {
        inner: &c.strain,
        wrong: Wrong::AlphaOnly(Term::Kinetic),
    };
    let bite = gates(&contract(&wrong, &c.frozen).unwrap().gradient());
    println!("x, kinetic on the alpha block only: bites {bite:.2e} gates");
    assert!(bite > 100.);
    // The guard: m_z of the other sign (the diagonal spin blocks exchanged) does nothing to an x
    // state and a great deal to the z one.
    let flip = |f: &mut StrainFields| {
        for v in f.local[3].iter_mut() {
            *v = -*v;
        }
    };
    let on_x = gates(&damaged(&x, flip));
    let on_z =
        miss(&damaged(&z, flip), &z.stress.gradient) / (COVARIANCE_GATE * peak(&z.stress.gradient));
    println!("m_z flipped: {on_x:.3} gates on the x state, {on_z:.2e} on the z state");
    assert!(
        on_x < 1.,
        "m_z of the other sign must not move an x state: {on_x:e}"
    );
    assert!(on_z > 100., "...and must move the z one: {on_z:e}");
}

/// The magnetic spin-orbit state of `case` (started along z): its stress against the reconverged
/// free energy, entry by entry, and (`second`) against the second step family `<case>_b`.
fn against_the_oracle(case: &'static str, second: bool) -> Stress {
    let data = load(case);
    assert_pins(&data);
    let volume = data.manifest["volume_bohr3"].as_f64().unwrap();
    let result = {
        let result = data.crystal.solve(&data.options).unwrap();
        assert!(result.converged, "the oracle SCF must converge");
        result
    };
    let sp = spinor_of(data);
    let m = moment(&sp.data, &result);
    println!("{case}: moment {m:?}");
    // The state is magnetic. With the coupling on its orientation is a soft mode (the anisotropy of N and Ne
    // is 1e-8 Ha): the SCF stops anywhere along it (here a tilt of 7 degrees), and the stress does not see it
    // at the gates (the spin-orbit term is 2e-8 Ha).
    let norm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        norm > MOMENT_FLOOR && (norm - 3.).abs() < 0.05,
        "the moment of a high-spin N atom: {m:?}"
    );
    // The oracle's free energy was frozen under the SCF of the Python extension; the orientation of the
    // state is a soft mode and the current SCF stops elsewhere on it: the free energies agree to the
    // anisotropy energy of the tilt (1.2e-9 and 1.4e-9 Ha measured, the gate is 1e-8).
    let frozen = sp.data.manifest["free_energy_ha"].as_f64().unwrap();
    println!(
        "  free energy minus the oracle's: {:.2e} Ha",
        result.free_energy - frozen
    );
    assert!(
        (result.free_energy - frozen).abs() < 1e-8,
        "not the oracle's state: {:e}",
        result.free_energy - frozen
    );
    let stress = stress_at(&sp, &sp.data.state(&result), &sp.data.options);
    summary(case, &stress);
    let want = entries(&sp.data.manifest, volume);
    let worst = report(case, &stress.gradient, &want, volume);
    assert!(
        worst < 1.,
        "the magnetic stress misses its gate by {worst:.2} x"
    );
    if second {
        let other = load_manifest(&format!("{case}_b"));
        let worst_b = report(
            &format!("{case} (second family)"),
            &stress.gradient,
            &entries(&other, volume),
            volume,
        );
        assert!(
            worst_b < 1.,
            "the second family: the stress misses its gate by {worst_b:.2} x"
        );
    }
    // The spin-orbit term of the light atoms is below the oracle's gates (reported, not controlled).
    let so = &stress.breakdown.spin_orbit;
    let without =
        std::array::from_fn(|a| std::array::from_fn(|b| stress.gradient[a][b] - so[a][b]));
    let bite = bites(&without, &want);
    println!(
        "  spin-orbit term peak {:.3e} Ha; without it: bites {bite:.2e} gates (N and Ne: not resolved)",
        peak(so)
    );
    stress
}

#[test]
fn the_magnetic_spin_orbit_stress_is_the_reconverged_free_energy() {
    against_the_oracle("nne_fr", true);
}

#[test]
fn the_magnetic_gradient_correction_is_the_reconverged_free_energy() {
    // PBE on the eigenframe fields of a state with live transverse components: the gradients and the
    // flux of the rotated spin channels reach the spectral-gradient term through `spinor.rs`.
    let stress = against_the_oracle("nne_pbefr", false);
    let gradient = &stress.breakdown.xc_gradient;
    assert!(peak(gradient) > 1e-4, "the GGA term: {:e}", peak(gradient));
}
